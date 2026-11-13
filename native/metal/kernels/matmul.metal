// ---- Projections Y[b, row] = W[row, :] . X[b, :] (Rust [out, in], no transpose) ----
// n0 rows, n1 cols, n2 = 1 when rows and X are 4-aligned (vector loads), n3
// batch tokens. W is f32 or BF16 (E = float / ushort, widened exactly).

// Level 1 (also the decode GEMV at every level): one 32-lane SIMD group per
// row; lanes stride the row (coalesced), then simd_sum. A batch is handled 8
// tokens per pass over the row, and a token's summation order does not depend
// on the batch, so batched prefill equals per-token GEMV bit for bit. That
// order differs from the CPU loop; ADR 0004 tolerances cover it.
constant int kTokens = 8;
template <typename E, typename E4>
inline void gemm_rows(device float* y, device const E* m, device const float* x, int rows, int cols,
                      bool vec, int batch, int row, uint lane, uint width) {
    batch = max(batch, 1);
    if (row >= rows) return; // uniform per SIMD group, so simd_sum stays valid
    device const E* r = m + ulong(row) * ulong(cols);
    for (int b0 = 0; b0 < batch; b0 += kTokens) {
        int nb = min(kTokens, batch - b0);
        float sum[kTokens] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        if (vec) { // aligned rows (float4: 16B, ushort4: 8B) and X rows: vector loads
            device const E4* r4 = (device const E4*)r;
            for (int j = int(lane); j < cols / 4; j += int(width)) {
                float4 w = widen4(r4[j]);
                for (int b = 0; b < kTokens; ++b) {
                    if (b < nb) {
                        device const float4* x4 = (device const float4*)(x + ulong(b0 + b) * ulong(cols));
                        float4 p = w * x4[j];
                        sum[b] += (p.x + p.y) + (p.z + p.w);
                    }
                }
            }
        } else {
            for (int j = int(lane); j < cols; j += int(width)) {
                float w = widen(r[j]);
                for (int b = 0; b < kTokens; ++b)
                    if (b < nb) sum[b] += w * x[ulong(b0 + b) * ulong(cols) + ulong(j)];
            }
        }
        for (int b = 0; b < kTokens; ++b) {
            if (b < nb) {
                float s = simd_sum(sum[b]);
                if (lane == 0) y[ulong(b0 + b) * ulong(rows) + ulong(row)] = s;
            }
        }
    }
}
#define GEMM_ARGS constant Scalars& a [[buffer(8)]], uint group [[threadgroup_position_in_grid]], \
    uint sg [[simdgroup_index_in_threadgroup]], uint groups [[simdgroups_per_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]], uint width [[threads_per_simdgroup]]
kernel void gemv(device float* y [[buffer(0)]], device const float* m [[buffer(1)]],
                 device const float* x [[buffer(2)]], GEMM_ARGS) {
    gemm_rows<float, float4>(y, m, x, a.n[0], a.n[1], a.n[2] != 0, a.n[3], int(group * groups + sg), lane, width);
}
kernel void gemv_bf16(device float* y [[buffer(0)]], device const ushort* m [[buffer(1)]],
                      device const float* x [[buffer(2)]], GEMM_ARGS) {
    gemm_rows<ushort, ushort4>(y, m, x, a.n[0], a.n[1], a.n[2] != 0, a.n[3], int(group * groups + sg), lane, width);
}

// Several BF16 matrices that read the same X (DeltaNet qkv/z/a/b, attention
// q/k/v, FFN gate/up) in one dispatch, for fewer than 8 tokens (ADR 0019's
// grouped projections, for Metal). Threadgroups are laid out matrix after
// matrix (`first[i]` = first threadgroup of matrix i); each row runs exactly
// the gemv_bf16 code, so every output is bitwise equal to separate calls.
// Buffers: X at 0, Y_i at 1 + i, W_i at 5 + i (i < 4); group description at 9.
constant constexpr int kGroupMax = 4;
struct GemvGroup { int count, cols, batch, x_vec; int rows[kGroupMax], vec[kGroupMax], first[kGroupMax + 1]; };
kernel void gemv_bf16_group(device const float* x [[buffer(0)]],
                            device float* y0 [[buffer(1)]], device float* y1 [[buffer(2)]],
                            device float* y2 [[buffer(3)]], device float* y3 [[buffer(4)]],
                            device const ushort* m0 [[buffer(5)]], device const ushort* m1 [[buffer(6)]],
                            device const ushort* m2 [[buffer(7)]], device const ushort* m3 [[buffer(8)]],
                            constant GemvGroup& g [[buffer(9)]], uint group [[threadgroup_position_in_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint groups [[simdgroups_per_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]], uint width [[threads_per_simdgroup]]) {
    int i = 0;
    while (i + 1 < g.count && int(group) >= g.first[i + 1]) ++i;
    device float* y = i == 0 ? y0 : i == 1 ? y1 : i == 2 ? y2 : y3;
    device const ushort* m = i == 0 ? m0 : i == 1 ? m1 : i == 2 ? m2 : m3;
    int row = (int(group) - g.first[i]) * int(groups) + int(sg);
    gemm_rows<ushort, ushort4>(y, m, x, g.rows[i], g.cols, g.vec[i] != 0, g.batch, row, lane, width);
}

// Level 2, a few tokens (a verify pass; ADR 0024): the gemm_rows row code with
// the token count NB fixed at compile time. gemm_rows' batch loop runs to a
// runtime count, so its sum[8] array and the X loads are not unrolled; that
// cost about 1.1 ms per extra token on M5 (2-7 tokens), with NB fixed it
// costs 0.1-0.3 ms. Same lanes, same j order and the same pairwise float4
// sums per token as gemm_rows: bitwise equal outputs. Needs vector-aligned
// rows (n2 = 1). One token stays on gemm_rows (a fixed NB = 1 was not faster).
template <int NB>
inline void gemm_rows_fixed(device float* y, device const ushort* m, device const float* x, int rows,
                            int cols, int row, uint lane, uint width) {
    if (row >= rows) return; // uniform per SIMD group
    device const ushort4* r4 = (device const ushort4*)(m + ulong(row) * ulong(cols));
    float sum[NB];
    #pragma clang loop unroll(full)
    for (int b = 0; b < NB; ++b) sum[b] = 0.f;
    for (int j = int(lane); j < cols / 4; j += int(width)) {
        float4 w = widen4(r4[j]);
        #pragma clang loop unroll(full)
        for (int b = 0; b < NB; ++b) {
            float4 p = w * ((device const float4*)(x + ulong(b) * ulong(cols)))[j];
            sum[b] += (p.x + p.y) + (p.z + p.w);
        }
    }
    #pragma clang loop unroll(full)
    for (int b = 0; b < NB; ++b) {
        float s = simd_sum(sum[b]);
        if (lane == 0) y[ulong(b) * ulong(rows) + ulong(row)] = s;
    }
}
#define GEMV_FIXED(NB)                                                                              \
    kernel void gemv_bf16_t##NB(device float* y [[buffer(0)]], device const ushort* m [[buffer(1)]], \
                                device const float* x [[buffer(2)]], GEMM_ARGS) {                    \
        gemm_rows_fixed<NB>(y, m, x, a.n[0], a.n[1], int(group * groups + sg), lane, width);        \
    }                                                                                               \
    kernel void gemv_bf16_group_t##NB(device const float* x [[buffer(0)]],                          \
        device float* y0 [[buffer(1)]], device float* y1 [[buffer(2)]],                             \
        device float* y2 [[buffer(3)]], device float* y3 [[buffer(4)]],                             \
        device const ushort* m0 [[buffer(5)]], device const ushort* m1 [[buffer(6)]],               \
        device const ushort* m2 [[buffer(7)]], device const ushort* m3 [[buffer(8)]],               \
        constant GemvGroup& g [[buffer(9)]], uint group [[threadgroup_position_in_grid]],           \
        uint sg [[simdgroup_index_in_threadgroup]], uint groups [[simdgroups_per_threadgroup]],     \
        uint lane [[thread_index_in_simdgroup]], uint width [[threads_per_simdgroup]]) {            \
        int i = 0;                                                                                  \
        while (i + 1 < g.count && int(group) >= g.first[i + 1]) ++i;                                \
        device float* y = i == 0 ? y0 : i == 1 ? y1 : i == 2 ? y2 : y3;                             \
        device const ushort* m = i == 0 ? m0 : i == 1 ? m1 : i == 2 ? m2 : m3;                      \
        gemm_rows_fixed<NB>(y, m, x, g.rows[i], g.cols, (int(group) - g.first[i]) * int(groups) + int(sg), lane, width); \
    }
GEMV_FIXED(2) GEMV_FIXED(3) GEMV_FIXED(4) GEMV_FIXED(5) GEMV_FIXED(6) GEMV_FIXED(7)

// Level 2 (ADR 0006, 0015): tiled GEMM on 8x8 simdgroup matrices for a prefill
// chunk (>= 8 tokens). A threadgroup of 4 SIMD groups computes BM rows x BN
// tokens of Y; each SIMD group owns I x J 8x8 tiles (I over tokens, J over
// rows). Per K block of kK columns, X and W are staged in threadgroup memory
// (vector loads when n2 allows, zero past the edges) and consumed 8 columns at
// a time in order. The K order does not depend on the tile shape, so every
// tile shape gives bit-identical results. Shapes:
//   64 x 64, I = J = 4: the projections (16 multiply-adds per 8 tile loads);
//   16 x 64, I = J = 2: matrices of at most 16 rows (DeltaNet a/b).
// Full unrolling of the I x J loops is required: otherwise the accumulator
// arrays spill (4x slower).
constant int kK = 32;
// (Metal allows no static members at program scope: constexpr functions.)
constexpr int tile_sgr(int bm, int j) { return bm / (8 * j); }                     // SIMD groups along rows
constexpr int tile_bn(int bm, int i, int j) { return (4 / tile_sgr(bm, j)) * 8 * i; } // tokens
constexpr int tile_buf(int bm, int i, int j) {
    return (tile_bn(bm, i, j) + bm) * kK > tile_bn(bm, i, j) * bm ? (tile_bn(bm, i, j) + bm) * kK
                                                                  : tile_bn(bm, i, j) * bm;
}
template <typename E, typename E4, int BM, int I, int J>
inline void gemm_tiled(device float* y, device const E* m, device const float* x, constant Scalars& a,
                       threadgroup float* buf, uint2 tg, uint tid, uint sg) {
    constexpr int SGR = tile_sgr(BM, J), BN = tile_bn(BM, I, J);
    int rows = a.n[0], cols = a.n[1], batch = a.n[3];
    bool vec = a.n[2] != 0;
    int r0 = int(tg.x) * BM, b0 = int(tg.y) * BN;
    int sr = int(sg % SGR) * 8 * J, st = int(sg / SGR) * 8 * I; // this SIMD group's block
    threadgroup float* xs = buf;           // [BN][kK] tokens x k
    threadgroup float* ws = buf + BN * kK; // [BM][kK] rows x k
    simdgroup_float8x8 acc[I][J];
    #pragma clang loop unroll(full)
    for (int i = 0; i < I; ++i)
        #pragma clang loop unroll(full)
        for (int j = 0; j < J; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    for (int k0 = 0; k0 < cols; k0 += kK) {
        if (vec) {
            for (int i = int(tid); i < BN * kK / 4; i += 128) {
                int rr = i / (kK / 4), k = k0 + (i % (kK / 4)) * 4, at = rr * kK + (i % (kK / 4)) * 4;
                bool kin = k < cols; // cols % 4 == 0, so k..k+3 are all in or all out
                float4 xv = (kin && b0 + rr < batch)
                    ? *((device const float4*)(x + ulong(b0 + rr) * ulong(cols) + ulong(k))) : float4(0.f);
                xs[at] = xv.x; xs[at + 1] = xv.y; xs[at + 2] = xv.z; xs[at + 3] = xv.w;
                if (rr < BM) {
                    float4 wv = (kin && r0 + rr < rows)
                        ? widen4(*((device const E4*)(m + ulong(r0 + rr) * ulong(cols) + ulong(k)))) : float4(0.f);
                    ws[at] = wv.x; ws[at + 1] = wv.y; ws[at + 2] = wv.z; ws[at + 3] = wv.w;
                }
            }
        } else {
            for (int i = int(tid); i < BN * kK; i += 128) {
                int rr = i / kK, k = k0 + i % kK;
                bool kin = k < cols;
                xs[i] = (kin && b0 + rr < batch) ? x[ulong(b0 + rr) * ulong(cols) + ulong(k)] : 0.f;
                if (rr < BM) ws[i] = (kin && r0 + rr < rows) ? widen(m[ulong(r0 + rr) * ulong(cols) + ulong(k)]) : 0.f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
        for (int kk = 0; kk < kK; kk += 8) {
            simdgroup_float8x8 xa[I], wb[J];
            #pragma clang loop unroll(full)
            for (int i = 0; i < I; ++i) simdgroup_load(xa[i], xs + (st + i * 8) * kK + kk, kK);
            // ws is [row][k]; transposed load gives the [k][row] operand of X . W^T.
            #pragma clang loop unroll(full)
            for (int j = 0; j < J; ++j)
                simdgroup_load(wb[j], ws + (sr + j * 8) * kK + kk, kK, ulong2(0, 0), true);
            #pragma clang loop unroll(full)
            for (int i = 0; i < I; ++i)
                #pragma clang loop unroll(full)
                for (int j = 0; j < J; ++j) simdgroup_multiply_accumulate(acc[i][j], xa[i], wb[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Through threadgroup memory (BN x BM) so edge tiles store only valid elements.
    #pragma clang loop unroll(full)
    for (int i = 0; i < I; ++i)
        #pragma clang loop unroll(full)
        for (int j = 0; j < J; ++j) simdgroup_store(acc[i][j], buf + (st + i * 8) * BM + sr + j * 8, BM);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = int(tid); i < BN * BM; i += 128) {
        int rr = i / BM, cc = i % BM;
        if (b0 + rr < batch && r0 + cc < rows) y[ulong(b0 + rr) * ulong(rows) + ulong(r0 + cc)] = buf[i];
    }
}
#define TILED_ARGS constant Scalars& a [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]], \
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]
#define GEMM_TILED(NAME, E, E4, BM, I, J)                                                     \
    kernel void NAME(device float* y [[buffer(0)]], device const E* m [[buffer(1)]],               \
                     device const float* x [[buffer(2)]], TILED_ARGS) {                             \
        threadgroup float buf[tile_buf(BM, I, J)]; /* staging, then the output tile */           \
        gemm_tiled<E, E4, BM, I, J>(y, m, x, a, buf, tg, tid, sg);                                 \
    }
GEMM_TILED(gemm_tiled_f32, float, float4, 64, 4, 4)
GEMM_TILED(gemm_tiled_bf16, ushort, ushort4, 64, 4, 4)
GEMM_TILED(gemm_tiled16_f32, float, float4, 16, 2, 2)
GEMM_TILED(gemm_tiled16_bf16, ushort, ushort4, 16, 2, 2)
