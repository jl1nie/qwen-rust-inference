// Apache-2.0. HIP kernels of the ROCm bridge (ADR 0013). gfx1100 runs wave32.
//
// The reference bodies (reference.h) give one GPU thread a whole head, group
// or column, which left the RX 7900 XTX nearly idle (8 attention threads for
// the whole GPU). Here every heavy op is split so that the work a thread does
// in sequence is short:
//   projection  BF16 GEMV: one wave per weight row (16-byte loads, <= 8
//               tokens share each load); prefill: LDS-tiled GEMM, so a chunk
//               reads each weight T/32 times instead of T times
//   RMSNorm     one block per group, block reduction
//   DeltaRule   a value column is owned by 4 lanes (32 key rows each) whose
//               state stays in registers for the whole chunk
//   attention   flash attention: a block per (KV head, query rows), K/V
//               staged in LDS and shared by the G query heads of that KV
//               head; decode also splits the positions (flash-decoding) and
//               merges the partial softmaxes in attention_combine
// Sums are reordered (wave/block reductions, tiles), so results match the
// CPU reference within the ADR 0004 tolerances, not bitwise.
#pragma once
#include <hip/hip_runtime.h>
#include "reference.h"

constexpr int kWave = 32;

__device__ inline float wave_sum(float v) {
    for (int o = kWave / 2; o > 0; o >>= 1) v += __shfl_xor(v, o, kWave);
    return v;
}
// Sum over a block of `blockDim.x` threads (a multiple of kWave, <= 1024).
__device__ inline float block_sum(float v) {
    __shared__ float partial[1024 / kWave];
    int lane = threadIdx.x % kWave, wave = threadIdx.x / kWave, waves = blockDim.x / kWave;
    v = wave_sum(v);
    if (lane == 0) partial[wave] = v;
    __syncthreads();
    v = lane < waves ? partial[lane] : 0.f;
    v = wave_sum(v);
    __syncthreads(); // partial[] may be reused by the caller's next block_sum
    return v;
}
// Eight BF16 values packed in 16 bytes, element 0 in the low half of .x.
__device__ inline void widen8(uint4 p, float* w) {
    const uint32_t q[4] = {p.x, p.y, p.z, p.w};
    for (int i = 0; i < 4; ++i) {
        w[2 * i] = __uint_as_float(q[i] << 16);
        w[2 * i + 1] = __uint_as_float(q[i] & 0xffff0000u);
    }
}

// ---- generic body (cheap elementwise ops, fallbacks) ----
__global__ void reference_kernel(int op, int count, Params a) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < count) body(op, t, a);
}

// ---- projections: Y[T, rows] = X[T, cols] . W^T, W row-major BF16 [rows, cols] ----

// Fallback for shapes without 16-byte rows (cols % 8 != 0): one block per (row, token).
__global__ void gemv_bf16_block(float* y, const uint16_t* a, const float* xs, int rows, int cols) {
    __shared__ float part[256];
    const uint16_t* r = a + size_t(blockIdx.x) * size_t(cols);
    const float* x = xs + size_t(blockIdx.y) * size_t(cols);
    float sum = 0.f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) sum += bf16(r[j]) * x[j];
    part[threadIdx.x] = sum; __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (int(threadIdx.x) < k) part[threadIdx.x] += part[threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[size_t(blockIdx.y) * size_t(rows) + blockIdx.x] = part[0];
}

// Decode and short chunks. One wave per weight row: lane l reads the 16-byte
// pieces l, l+32, ... so a wave reads 512 contiguous bytes per step; the
// weights are read once for up to kGemvTokens token rows (blockIdx.y).
constexpr int kGemvTokens = 8;
constexpr int kGemvRowsPerBlock = 8; // 256 threads
__global__ __launch_bounds__(256) void gemv_bf16_wave(float* y, const uint16_t* a, const float* x,
                                                      int rows, int cols, int tokens) {
    int row = blockIdx.x * kGemvRowsPerBlock + threadIdx.x / kWave, lane = threadIdx.x % kWave;
    if (row >= rows) return; // the whole wave leaves together
    int t0 = blockIdx.y * kGemvTokens, n = min(kGemvTokens, tokens - t0);
    const uint4* w = reinterpret_cast<const uint4*>(a + size_t(row) * cols);
    float sum[kGemvTokens] = {};
    for (int j = lane; j < cols / 8; j += kWave) {
        float wv[8];
        widen8(w[j], wv);
#pragma unroll
        for (int b = 0; b < kGemvTokens; ++b) {
            if (b >= n) break;
            const float4* xv = reinterpret_cast<const float4*>(x + size_t(t0 + b) * cols) + 2 * j;
            float4 lo = xv[0], hi = xv[1];
            float s = sum[b];
            s = fmaf(wv[0], lo.x, s); s = fmaf(wv[1], lo.y, s); s = fmaf(wv[2], lo.z, s); s = fmaf(wv[3], lo.w, s);
            s = fmaf(wv[4], hi.x, s); s = fmaf(wv[5], hi.y, s); s = fmaf(wv[6], hi.z, s); s = fmaf(wv[7], hi.w, s);
            sum[b] = s;
        }
    }
#pragma unroll
    for (int b = 0; b < kGemvTokens; ++b) {
        if (b >= n) break;
        float s = wave_sum(sum[b]);
        if (lane == 0) y[size_t(t0 + b) * rows + row] = s;
    }
}

// Prefill. A block computes a 64-row x 32-token tile of Y; each step stages a
// 32-column slice of W (widened to f32) and of X in LDS, and each of the 256
// threads accumulates 4 rows x 2 tokens. Needs cols % 8 == 0.
constexpr int kTileRows = 64, kTileTokens = 32, kTileK = 32;
constexpr int kGemmMinTokens = 16; // fewer tokens: gemv_bf16_wave
__global__ __launch_bounds__(256) void gemm_bf16_tiled(float* y, const uint16_t* a, const float* x,
                                                       int rows, int cols, int tokens) {
    __shared__ float ws[kTileK][kTileRows + 4];   // ws[k][row]
    __shared__ float xs[kTileK][kTileTokens + 4]; // xs[k][token]
    int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
    int r0 = blockIdx.x * kTileRows, t0 = blockIdx.y * kTileTokens;
    int wr = tid / 4, wk = (tid % 4) * 8; // this thread loads W[r0+wr][k0+wk .. +8]
    int xt = tid / 8, xk = (tid % 8) * 4; // and X[t0+xt][k0+xk .. +4]
    float acc[4][2] = {};
    for (int k0 = 0; k0 < cols; k0 += kTileK) {
        float wv[8] = {};
        if (r0 + wr < rows && k0 + wk < cols)
            widen8(*reinterpret_cast<const uint4*>(a + size_t(r0 + wr) * cols + k0 + wk), wv);
        for (int i = 0; i < 8; ++i) ws[wk + i][wr] = wv[i];
        float4 xv = make_float4(0.f, 0.f, 0.f, 0.f);
        if (t0 + xt < tokens && k0 + xk < cols)
            xv = *reinterpret_cast<const float4*>(x + size_t(t0 + xt) * cols + k0 + xk);
        xs[xk][xt] = xv.x; xs[xk + 1][xt] = xv.y; xs[xk + 2][xt] = xv.z; xs[xk + 3][xt] = xv.w;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < kTileK; ++k) {
            float4 w = *reinterpret_cast<const float4*>(&ws[k][tx * 4]);
            float2 v = *reinterpret_cast<const float2*>(&xs[k][ty * 2]);
            const float wr4[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                acc[i][0] = fmaf(wr4[i], v.x, acc[i][0]);
                acc[i][1] = fmaf(wr4[i], v.y, acc[i][1]);
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 2; ++j) {
            int r = r0 + tx * 4 + i, t = t0 + ty * 2 + j;
            if (r < rows && t < tokens) y[size_t(t) * rows + r] = acc[i][j];
        }
}

// ---- RMSNorm (op 1): one block per group ----
__global__ __launch_bounds__(256) void rms_norm_block(Params a) {
    const float *x = a.p[1], *w = a.p[2];
    float* o = a.p[0];
    int d = a.n[0];
    size_t off = size_t(blockIdx.x) * d;
    float sum = 0.f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) sum = fmaf(x[off + i], x[off + i], sum);
    float r = 1.f / sqrtf(block_sum(sum) / float(d) + a.f[0]);
    for (int i = threadIdx.x; i < d; i += blockDim.x) o[off + i] = (x[off + i] * r) * (w[i] + a.f[1]);
}

// ---- DeltaRule (op 7): state in registers, one column split over P lanes ----
// As in the reference body a value column j of head h is owned for the whole
// chunk, but by P adjacent lanes, each keeping K = key_dim / P state rows in
// registers (one VRAM read and write per chunk, not per token). The two dot
// products over key_dim are summed within the P lanes by shuffles, so the
// summation order differs from the reference (ADR 0004 tolerances).
template <int K, int P>
__global__ __launch_bounds__(128) void delta_rule_registers(Params a, int count) {
    int t = blockIdx.x * blockDim.x + threadIdx.x, column = t / P, part = t % P;
    if (column >= count) return; // whole groups of P lanes leave together
    const float *x = a.p[1], *w = a.p[2];
    float* o = a.p[0];
    int kd = a.n[0], vd = a.n[1], kh = a.n[2], vh = a.n[3], h = column / vd, j = column % vd;
    int tokens = a.n[4] > 0 ? a.n[4] : 1, stride = a.n[5];
    float* state = a.p[4] + size_t(h) * kd * vd;
    int first = part * K; // this lane's rows: first .. first + K (guarded by kd)
    float s[K];
#pragma unroll
    for (int i = 0; i < K; ++i) s[i] = first + i < kd ? state[(first + i) * vd + j] : 0.f;
    for (int r = 0; r < tokens; ++r) {
        const float *q = x + r * vh * kd + h * kd + first, *k = w + r * vh * kd + h * kd + first;
        const float* v = a.p[3] + r * stride;
        float decay = expf(a.p[5][r * vh + h]), memory = 0.f;
#pragma unroll
        for (int i = 0; i < K; ++i)
            if (first + i < kd) { s[i] *= decay; memory += s[i] * k[i]; }
        for (int m = 1; m < P; m <<= 1) memory += __shfl_xor(memory, m, kWave);
        float delta = (v[2 * kh * kd + h * vd + j] - memory) * a.p[6][r * vh + h], out = 0.f;
#pragma unroll
        for (int i = 0; i < K; ++i)
            if (first + i < kd) { s[i] += k[i] * delta; out += s[i] * q[i]; }
        for (int m = 1; m < P; m <<= 1) out += __shfl_xor(out, m, kWave);
        if (part == 0) o[r * vh * vd + column] = out;
    }
#pragma unroll
    for (int i = 0; i < K; ++i)
        if (first + i < kd) state[(first + i) * vd + j] = s[i];
}

// ---- AttentionRead (op 10): flash attention with K/V blocks in LDS ----
// Same slots as the reference body: p0 out, p1 query [T,heads,d], p2 keys and
// p3 values [capacity,kv,d], p4 gate, p5 per-head scratch of n6 floats; n3 =
// keys seen by row 0, n5 = T rows.
//
// A block serves one KV head and `rows` consecutive query rows: one wave per
// (row, query head of that KV head), G = heads/kv waves per row (the ADR 0007
// grouping of Metal's attention_mma). The block walks its positions in blocks
// of kAttnBlock: all threads stage K and V of those positions in LDS once,
// shared by every wave, then each wave
//   1. accumulates per-lane partial dot products for the kAttnBlock positions
//      (lane l owns dims l, l+32, ...: L per lane),
//   2. reduces them with a butterfly so lane l ends up holding the full score
//      of position l / (32 / kAttnBlock), 15+1 shuffles instead of 16 x 5,
//   3. applies the causal mask and an online softmax update for the block
//      (running max m, denominator l), and adds P V from LDS.
// RDNA3 matrix instructions take FP16/BF16 inputs only, so unlike Metal's
// simdgroup_float8x8 tiles this stays FP32 FMA.
//
// splits == 1 writes the gated output. Decode (one row) splits the positions
// (flash-decoding, ADR 0008): split s covers [s*span, (s+1)*span) and stores
// (max, denominator, unnormalized sums[d]) at p5 + head*n6 + s*(d+2), merged
// by attention_combine.
constexpr int kAttnBlock = 16;                  // positions staged per step
constexpr int kAttnMaxDim = 256;                // LDS: 2 x 16 x 256 floats = 32 KiB
constexpr int kAttnWaves = 8;                   // waves per block for prefill
__device__ inline float wave_max(float v) {
    for (int o = kWave / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, kWave));
    return v;
}
// v[i] holds this lane's partial sum for position i (kAttnBlock of them).
// Afterwards v[0] holds the full sum over the wave for position
// lane / (kWave / kAttnBlock): each step halves the positions a lane keeps
// and trades the other half with the partner lane.
__device__ inline float butterfly_sum(float* v) {
    int lane = threadIdx.x % kWave, count = kAttnBlock;
#pragma unroll
    for (int offset = kWave / 2; offset >= kWave / kAttnBlock; offset >>= 1) {
        bool upper = (lane & offset) != 0;
        count /= 2;
#pragma unroll
        for (int i = 0; i < kAttnBlock / 2; ++i) {
            if (i >= count) break;
            float send = upper ? v[i] : v[i + count], keep = upper ? v[i + count] : v[i];
            v[i] = keep + __shfl_xor(send, offset, kWave);
        }
    }
    float sum = v[0];
    for (int offset = kWave / kAttnBlock / 2; offset > 0; offset >>= 1) sum += __shfl_xor(sum, offset, kWave);
    return sum;
}
template <int L>
__global__ __launch_bounds__(256) void attention_tiled(Params a, int rows, int span, int splits) {
    __shared__ float ks[kAttnBlock * kAttnMaxDim], vs[kAttnBlock * kAttnMaxDim];
    constexpr int kSpread = kWave / kAttnBlock; // lanes holding the same position's score
    int d = a.n[0], heads = a.n[1], kv = a.n[2], tokens = a.n[5] > 0 ? a.n[5] : 1, group = heads / kv;
    int lane = threadIdx.x % kWave, wave = threadIdx.x / kWave;
    int h = blockIdx.y, s = blockIdx.x % splits, r0 = blockIdx.x / splits * rows;
    int r = r0 + wave / group, head = h * group + wave % group;
    bool active = r < tokens; // idle waves still help stage K/V and meet the barriers
    int last = min(r0 + rows, tokens) - 1, length = a.n[3] + r;
    int begin = s * span, end = min(a.n[3] + last, begin + span);
    size_t row = (size_t(r) * heads + head) * d;
    float q[L], acc[L];
#pragma unroll
    for (int i = 0; i < L; ++i) {
        int c = lane + i * kWave;
        q[i] = active && c < d ? a.p[1][row + c] : 0.f;
        acc[i] = 0.f;
    }
    const float scale = sqrtf(float(d));
    float m = -INFINITY, l = 0.f;
    for (int p0 = begin; p0 < end; p0 += kAttnBlock) {
        int n = min(kAttnBlock, end - p0);
        for (int e = threadIdx.x; e < n * d; e += blockDim.x) {
            int j = e / d, c = e % d;
            size_t at = (size_t(p0 + j) * kv + h) * d + c;
            ks[j * d + c] = a.p[2][at];
            vs[j * d + c] = a.p[3][at];
        }
        __syncthreads();
        if (active && p0 < length) {
            float part[kAttnBlock];
#pragma unroll
            for (int j = 0; j < kAttnBlock; ++j) {
                float sum = 0.f;
#pragma unroll
                for (int i = 0; i < L; ++i) {
                    int c = lane + i * kWave;
                    if (j < n && c < d) sum = fmaf(q[i], ks[j * d + c], sum);
                }
                part[j] = sum;
            }
            int j = lane / kSpread; // the position whose score this lane holds
            float score = butterfly_sum(part) / scale;
            if (j >= n || p0 + j >= length) score = -INFINITY; // causal mask / block tail
            float next = fmaxf(m, wave_max(score));
            float e = expf(score - next), correction = expf(m - next);
            l = l * correction + wave_sum(e) / float(kSpread);
#pragma unroll
            for (int i = 0; i < L; ++i) acc[i] *= correction;
#pragma unroll
            for (int jj = 0; jj < kAttnBlock; ++jj) {
                float weight = __shfl(e, jj * kSpread, kWave);
#pragma unroll
                for (int i = 0; i < L; ++i) {
                    int c = lane + i * kWave;
                    if (jj < n && c < d) acc[i] = fmaf(weight, vs[jj * d + c], acc[i]);
                }
            }
            m = next;
        }
        __syncthreads();
    }
    if (!active) return;
    if (splits == 1) {
#pragma unroll
        for (int i = 0; i < L; ++i) {
            int c = lane + i * kWave;
            if (c < d) a.p[0][row + c] = (acc[i] / l) * sigmoid(a.p[4][row + c]);
        }
        return;
    }
    float* out = a.p[5] + size_t(head) * a.n[6] + size_t(s) * (d + 2);
    if (lane == 0) { out[0] = m; out[1] = l; }
#pragma unroll
    for (int i = 0; i < L; ++i) {
        int c = lane + i * kWave;
        if (c < d) out[2 + c] = acc[i];
    }
}
// Decode only (one row): merge the splits of attention_waves. Thread per (dim, head).
__global__ void attention_combine(Params a, int splits) {
    int d = a.n[0], heads = a.n[1], t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= d * heads) return;
    int head = t / d, c = t % d;
    const float* part = a.p[5] + size_t(head) * a.n[6];
    float m = -INFINITY;
    for (int s = 0; s < splits; ++s) m = fmaxf(m, part[s * (d + 2)]);
    float den = 0.f, sum = 0.f;
    for (int s = 0; s < splits; ++s) {
        const float* p = part + s * (d + 2);
        float weight = expf(p[0] - m);
        den = fmaf(p[1], weight, den);
        sum = fmaf(p[2 + c], weight, sum);
    }
    a.p[0][t] = (sum / den) * sigmoid(a.p[4][t]);
}
