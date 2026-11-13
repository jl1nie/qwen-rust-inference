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
constexpr int kGemvLoads = 4;        // 16-byte weight loads issued together per lane
// Rows [block * kGemvRowsPerBlock, +kGemvRowsPerBlock) of one matrix for the
// tokens of blockIdx.y; shared by the single and grouped launches below so a
// row's sum is the same code (and value) either way.
__device__ inline void gemv_bf16_rows(float* y, const uint16_t* a, const float* x, int rows, int cols, int tokens,
                                      int block) {
    int row = block * kGemvRowsPerBlock + threadIdx.x / kWave, lane = threadIdx.x % kWave;
    if (row >= rows) return; // the whole wave leaves together
    int t0 = blockIdx.y * kGemvTokens, n = min(kGemvTokens, tokens - t0);
    const uint4* w = reinterpret_cast<const uint4*>(a + size_t(row) * cols);
    float sum[kGemvTokens] = {};
    auto step = [&](uint4 packed, int j) {
        float wv[8];
        widen8(packed, wv);
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
    };
    // kGemvLoads weight loads in flight per lane before the FMAs use them
    // (#19): a cold 3584x1024 GEMV went from ~320 to ~600 GB/s. The pieces
    // are still consumed in the order j = lane, lane + 32, ..., so the sums
    // are bitwise the same as one load per step.
    int j = lane, pieces = cols / 8;
    for (; j + (kGemvLoads - 1) * kWave < pieces; j += kGemvLoads * kWave) {
        uint4 p[kGemvLoads];
#pragma unroll
        for (int u = 0; u < kGemvLoads; ++u) p[u] = w[j + u * kWave];
#pragma unroll
        for (int u = 0; u < kGemvLoads; ++u) step(p[u], j + u * kWave);
    }
    for (; j < pieces; j += kWave) step(w[j], j);
#pragma unroll
    for (int b = 0; b < kGemvTokens; ++b) {
        if (b >= n) break;
        float s = wave_sum(sum[b]);
        if (lane == 0) y[size_t(t0 + b) * rows + row] = s;
    }
}
__global__ __launch_bounds__(256) void gemv_bf16_wave(float* y, const uint16_t* a, const float* x,
                                                      int rows, int cols, int tokens) {
    gemv_bf16_rows(y, a, x, rows, cols, tokens, blockIdx.x);
}
// One-token GEMV followed by Add + RmsNorm of its output (ADR 0019): the
// block that finishes last (atomic ticket) computes sum = u + v and
// norm = RmsNorm(sum) with add_rms_norm_block's expressions and 256-thread
// reduction, so no block waits and the AddRmsNorm dispatch disappears. One of
// u, v is y. Other blocks' rows are read with device-scope loads after their
// fence; the last block resets the ticket for the next launch.
struct AddNormTail {
    float* sum;
    const float* u;
    const float* v;
    const float* weight;
    float* norm;
    float eps, offset;
    unsigned* ticket;
};
__device__ inline float load_agent(const float* p) {
    return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__global__ __launch_bounds__(256) void gemv_bf16_add_norm(float* y, const uint16_t* a, const float* x, int rows,
                                                          int cols, AddNormTail n) {
    __shared__ bool last;
    gemv_bf16_rows(y, a, x, rows, cols, 1, blockIdx.x);
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(n.ticket, 1u) == gridDim.x - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    float sum = 0.f;
    for (int i = threadIdx.x; i < rows; i += blockDim.x) {
        float v = load_agent(n.u + i) + load_agent(n.v + i);
        n.sum[i] = v;
        sum = fmaf(v, v, sum);
    }
    float r = 1.f / sqrtf(block_sum(sum) / float(rows) + n.eps);
    for (int i = threadIdx.x; i < rows; i += blockDim.x) n.norm[i] = (n.sum[i] * r) * (n.weight[i] + n.offset);
    if (threadIdx.x == 0) *n.ticket = 0u;
}
// Several matrices that project the same X (q/k/v, DeltaNet qkv/z/a/b, FFN
// gate/up) in one launch (#19): fewer dispatches, the same per-row sums.
// Matrix m owns blocks [first[m], first[m + 1]) of blockIdx.x.
constexpr int kGroupMax = 4;
struct GemvGroup {
    float* y[kGroupMax];
    const uint16_t* a[kGroupMax];
    int rows[kGroupMax];
    int first[kGroupMax + 1];
    int count;
};
__global__ __launch_bounds__(256) void gemv_bf16_group(GemvGroup g, const float* x, int cols, int tokens) {
    int b = blockIdx.x, m = 0;
    while (m + 1 < g.count && b >= g.first[m + 1]) ++m;
    gemv_bf16_rows(g.y[m], g.a[m], x, g.rows[m], cols, tokens, b - g.first[m]);
}
// FFN gate and up rows of the same index in one wave, then SiluProduct:
// p[t, row] = silu(gate . x_t) * (up . x_t). Each dot product runs the same
// loop as gemv_bf16_rows (same per-lane order and wave_sum), so the product
// equals gemv_bf16_wave twice followed by the SiluProduct body.
__global__ __launch_bounds__(256) void gemv_bf16_glu(float* p, const uint16_t* gate, const uint16_t* up,
                                                     const float* x, int rows, int cols, int tokens) {
    int row = blockIdx.x * kGemvRowsPerBlock + threadIdx.x / kWave, lane = threadIdx.x % kWave;
    if (row >= rows) return;
    int t0 = blockIdx.y * kGemvTokens, n = min(kGemvTokens, tokens - t0);
    const uint4* wg = reinterpret_cast<const uint4*>(gate + size_t(row) * cols);
    const uint4* wu = reinterpret_cast<const uint4*>(up + size_t(row) * cols);
    float sg[kGemvTokens] = {}, su[kGemvTokens] = {};
    auto step = [&](uint4 pg, uint4 pu, int j) {
        float g8[8], u8[8];
        widen8(pg, g8);
        widen8(pu, u8);
#pragma unroll
        for (int b = 0; b < kGemvTokens; ++b) {
            if (b >= n) break;
            const float4* xv = reinterpret_cast<const float4*>(x + size_t(t0 + b) * cols) + 2 * j;
            float4 lo = xv[0], hi = xv[1];
            float s = sg[b];
            s = fmaf(g8[0], lo.x, s); s = fmaf(g8[1], lo.y, s); s = fmaf(g8[2], lo.z, s); s = fmaf(g8[3], lo.w, s);
            s = fmaf(g8[4], hi.x, s); s = fmaf(g8[5], hi.y, s); s = fmaf(g8[6], hi.z, s); s = fmaf(g8[7], hi.w, s);
            sg[b] = s;
            s = su[b];
            s = fmaf(u8[0], lo.x, s); s = fmaf(u8[1], lo.y, s); s = fmaf(u8[2], lo.z, s); s = fmaf(u8[3], lo.w, s);
            s = fmaf(u8[4], hi.x, s); s = fmaf(u8[5], hi.y, s); s = fmaf(u8[6], hi.z, s); s = fmaf(u8[7], hi.w, s);
            su[b] = s;
        }
    };
    // As in gemv_bf16_rows: loads first, consumed in the same order.
    int j = lane, pieces = cols / 8;
    for (; j + (kGemvLoads - 1) * kWave < pieces; j += kGemvLoads * kWave) {
        uint4 pg[kGemvLoads], pu[kGemvLoads];
#pragma unroll
        for (int u = 0; u < kGemvLoads; ++u) { pg[u] = wg[j + u * kWave]; pu[u] = wu[j + u * kWave]; }
#pragma unroll
        for (int u = 0; u < kGemvLoads; ++u) step(pg[u], pu[u], j + u * kWave);
    }
    for (; j < pieces; j += kWave) step(wg[j], wu[j], j);
#pragma unroll
    for (int b = 0; b < kGemvTokens; ++b) {
        if (b >= n) break;
        float g = wave_sum(sg[b]), u = wave_sum(su[b]);
        if (lane == 0) p[size_t(t0 + b) * rows + row] = silu(g) * u;
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

// ---- fused RMSNorm ops (12, 13): the same block reduction as rms_norm_block ----
// AddRmsNorm writes the sum and normalizes it; a thread reads back only the
// elements it wrote, so no barrier is needed and every value (including the
// reduction order) equals Add followed by rms_norm_block.
__global__ __launch_bounds__(256) void add_rms_norm_block(Params a) {
    float *s = a.p[0], *o = a.p[3];
    const float *u = a.p[1], *v = a.p[2], *w = a.p[4];
    int d = a.n[0];
    size_t off = size_t(blockIdx.x) * d;
    float sum = 0.f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float x = u[off + i] + v[off + i];
        s[off + i] = x;
        sum = fmaf(x, x, sum);
    }
    float r = 1.f / sqrtf(block_sum(sum) / float(d) + a.f[0]);
    for (int i = threadIdx.x; i < d; i += blockDim.x) o[off + i] = (s[off + i] * r) * (w[i] + a.f[1]);
}
// RmsNormGate: rms_norm_block followed by SiluProduct (o = silu(z) * norm).
__global__ __launch_bounds__(256) void rms_norm_gate_block(Params a) {
    const float *x = a.p[1], *w = a.p[2], *z = a.p[3];
    float* o = a.p[0];
    int d = a.n[0];
    size_t off = size_t(blockIdx.x) * d;
    float sum = 0.f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) sum = fmaf(x[off + i], x[off + i], sum);
    float r = 1.f / sqrtf(block_sum(sum) / float(d) + a.f[0]);
    for (int i = threadIdx.x; i < d; i += blockDim.x) o[off + i] = silu(z[off + i]) * ((x[off + i] * r) * (w[i] + a.f[1]));
}

// ---- DeltaQk, AttentionQuery, AttentionKv (ops 5, 8, 9): one wave per head ----
// The reference bodies give one thread a whole head (16, 8 and 2 threads in a
// Qwen3.5-0.8B decode step), which made these three ops about a third of the
// decode time (#19). Here a wave owns the head: the sums of squares are wave
// reductions (summation order differs from the reference), and every output
// element is the same expression as in reference.h, computed by one lane.
constexpr int kHeadWaves = 8; // 256 threads
__device__ inline int head_index() { return blockIdx.x * kHeadWaves + threadIdx.x / kWave; }

// One item t = token * vh + value head, computed by the calling wave.
__device__ inline void delta_qk_item(const Params& a, int t) {
    int lane = threadIdx.x % kWave;
    float *o = a.p[0], *x = a.p[1], *w = a.p[2];
    int kh = a.n[0], vh = a.n[1], d = a.n[2], h = (t % vh) / (vh / kh), kd = kh * d;
    x += (t / vh) * a.n[3];
    float sq = 0.f, sk = 0.f;
    for (int i = lane; i < d; i += kWave) {
        float q = x[h * d + i], k = x[kd + h * d + i];
        sq = fmaf(q, q, sq); sk = fmaf(k, k, sk);
    }
    float rq = 1.f / sqrtf(wave_sum(sq) + 1e-6f), rk = 1.f / sqrtf(wave_sum(sk) + 1e-6f);
    for (int i = lane; i < d; i += kWave) {
        o[t * d + i] = (x[h * d + i] * rq) / sqrtf(float(d));
        w[t * d + i] = x[kd + h * d + i] * rk;
    }
}
__global__ __launch_bounds__(256) void delta_qk_wave(Params a, int count) {
    int t = head_index();
    if (t < count) delta_qk_item(a, t);
}
// RMSNorm with Qwen's (1 + w) scale of src[0..d), written to dst, then the
// rope() of reference.h on the first `rot` elements. Lane-parallel over the
// rotated pairs: each lane normalizes both elements of its pair itself, so
// every value is the reference's expression.
__device__ inline void norm_rope_wave(float* dst, const float* src, const float* w, int d, int rot, int pos,
                                      float eps, float theta) {
    int lane = threadIdx.x % kWave, half = rot / 2;
    float sum = 0.f;
    for (int i = lane; i < d; i += kWave) sum = fmaf(src[i], src[i], sum);
    float r = 1.f / sqrtf(wave_sum(sum) / float(d) + eps);
    for (int i = lane; i < half; i += kWave) {
        float x0 = (src[i] * r) * (1.f + w[i]), x1 = (src[i + half] * r) * (1.f + w[i + half]);
        float angle = float(pos) / powf(theta, float(2 * i) / float(rot));
        float s = sinf(angle), c = cosf(angle);
        dst[i] = x0 * c - x1 * s; dst[i + half] = x1 * c + x0 * s;
    }
    for (int i = rot + lane; i < d; i += kWave) dst[i] = (src[i] * r) * (1.f + w[i]);
}
__device__ inline void attention_query_head(const Params& a, int t) {
    int lane = threadIdx.x % kWave;
    int d = a.n[0], pos = a.n[2] + (a.n[3] > 0 ? t / a.n[3] : 0);
    const float* x = a.p[1] + size_t(t) * 2 * d;
    norm_rope_wave(a.p[0] + size_t(t) * d, x, a.p[2], d, a.n[1], pos, a.f[0], a.f[1]);
    for (int i = lane; i < d; i += kWave) a.p[3][size_t(t) * d + i] = x[d + i];
}
__device__ inline void attention_kv_head(const Params& a, int t) {
    int lane = threadIdx.x % kWave;
    int d = a.n[0], kv = a.n[3], pos = a.n[2] + t / kv, hh = t % kv;
    size_t dest = (size_t(pos) * kv + hh) * d;
    norm_rope_wave(a.p[0] + dest, a.p[1] + size_t(t) * d, a.p[2], d, a.n[1], pos, a.f[0], a.f[1]);
    for (int i = lane; i < d; i += kWave) a.p[4][dest + i] = a.p[3][size_t(t) * d + i];
}
__global__ __launch_bounds__(256) void attention_query_wave(Params a, int count) {
    int t = head_index();
    if (t < count) attention_query_head(a, t);
}
__global__ __launch_bounds__(256) void attention_kv_wave(Params a, int count) {
    int t = head_index();
    if (t < count) attention_kv_head(a, t);
}
// Op 15: both in one dispatch (query heads first, then KV heads).
__global__ __launch_bounds__(256) void attention_query_kv_wave(Params a, int count) {
    int t = head_index();
    if (t >= count) return;
    Params part;
    int item;
    if (query_kv_part(a, t, part, item)) attention_query_head(part, item);
    else attention_kv_head(part, item);
}

// ---- DeltaRule (op 7): state in registers, one column split over P lanes ----
// As in the reference body a value column j of head h is owned for the whole
// chunk, but by P adjacent lanes, each keeping K = key_dim / P state rows in
// registers (one VRAM read and write per chunk, not per token). The two dot
// products over key_dim are summed within the P lanes by shuffles, so the
// summation order differs from the reference (ADR 0004 tolerances).
// GATED: op DeltaRuleGated, the gates come from delta_gate() (reference.h)
// instead of DeltaGates' outputs, saving that dispatch.
template <int K, int P, bool FULL, bool GATED>
__device__ inline void delta_rule_column(const Params& a, int count, int t) {
    int column = t / P, part = t % P;
    if (column >= count) return; // whole groups of P lanes leave together
    const float *x = a.p[1], *w = a.p[2];
    float* o = a.p[0];
    int kd = a.n[0], vd = a.n[1], kh = a.n[2], vh = a.n[3], h = column / vd, j = column % vd;
    int tokens = a.n[4] > 0 ? a.n[4] : 1, stride = a.n[5];
    float* state = a.p[4] + size_t(h) * kd * vd;
    int first = part * K; // this lane's rows: first .. first + K (guarded by kd)
    // FULL (kd == K * P): every row exists, so no per-row guard, and q/k are
    // read as float4 (first and kd are multiples of 4). Same arithmetic in the
    // same order, bitwise equal; the guards cost ~130 branches (ADR 0019).
    auto has = [&](int i) { return FULL || first + i < kd; };
    float s[K];
#pragma unroll
    for (int i = 0; i < K; ++i) s[i] = has(i) ? state[(first + i) * vd + j] : 0.f;
    for (int r = 0; r < tokens; ++r) {
        const float *q = x + r * vh * kd + h * kd + first, *k = w + r * vh * kd + h * kd + first;
        const float* v = a.p[3] + r * stride;
        float qv[K], kv[K];
#pragma unroll
        for (int i = 0; i < K; ++i) {
            if (FULL && K % 4 == 0) {
                if (i % 4 == 0) {
                    float4 q4 = *reinterpret_cast<const float4*>(q + i), k4 = *reinterpret_cast<const float4*>(k + i);
                    qv[i] = q4.x; qv[i + 1] = q4.y; qv[i + 2] = q4.z; qv[i + 3] = q4.w;
                    kv[i] = k4.x; kv[i + 1] = k4.y; kv[i + 2] = k4.z; kv[i + 3] = k4.w;
                }
            } else if (has(i)) {
                qv[i] = q[i]; kv[i] = k[i];
            }
        }
        float log_decay, beta;
        delta_gate(a, GATED, r * vh + h, h, vh, log_decay, beta);
        float decay = expf(log_decay), memory = 0.f;
#pragma unroll
        for (int i = 0; i < K; ++i)
            if (has(i)) { s[i] *= decay; memory += s[i] * kv[i]; }
        for (int m = 1; m < P; m <<= 1) memory += __shfl_xor(memory, m, kWave);
        float delta = (v[2 * kh * kd + h * vd + j] - memory) * beta, out = 0.f;
#pragma unroll
        for (int i = 0; i < K; ++i)
            if (has(i)) { s[i] += kv[i] * delta; out += s[i] * qv[i]; }
        for (int m = 1; m < P; m <<= 1) out += __shfl_xor(out, m, kWave);
        if (part == 0) o[r * vh * vd + column] = out;
    }
#pragma unroll
    for (int i = 0; i < K; ++i)
        if (has(i)) state[(first + i) * vd + j] = s[i];
}
template <int K, int P, bool FULL, bool GATED>
__global__ __launch_bounds__(128) void delta_rule_registers(Params a, int count) {
    delta_rule_column<K, P, FULL, GATED>(a, count, blockIdx.x * blockDim.x + threadIdx.x);
}

// ---- DeltaNet mixer of one value head in one block (decode; ADR 0019) ----
// ConvSilu -> DeltaQk -> DeltaRuleGated -> RmsNormGate for the few tokens of
// a decode step, one block of value_dim * P threads per value head, stages
// separated by barriers. Each stage runs the same per-item code as its own op
// (conv_silu_channel, delta_qk_item, delta_rule_column, and the 128-thread
// reduction of rms_norm_gate_block), so the outputs are bitwise those of the
// four dispatches. Needs key_heads == value_heads (a head owns its channels),
// value_dim == 128 and key_dim == K * P.
struct DeltaMixer {
    float* convolved;
    const float* projected;
    const float* conv_w;
    float* conv_state;
    float* query;
    float* key;
    float* core;
    float* recurrent;
    const float* a;
    const float* b;
    const float* gates;
    float* gated;
    const float* norm_w;
    const float* z;
    int key_heads, value_heads, key_dim, value_dim, kernel, tokens, channels;
    float eps;
};
constexpr int kNormGroup = 128; // threads of rms_norm_gate_block for value_dim 128
constexpr int kDeltaFusedTokens = 4; // more tokens: the four ops (they parallelize better)
// block_sum of rms_norm_gate_block's 128-thread block, for each group of 128
// threads of a larger block (all threads must call it).
__device__ inline float group_sum(float v) {
    __shared__ float partial[1024 / kWave];
    constexpr int waves = kNormGroup / kWave;
    int lane = threadIdx.x % kWave, wave = threadIdx.x / kWave, base = wave / waves * waves;
    v = wave_sum(v);
    if (lane == 0) partial[wave] = v;
    __syncthreads();
    v = lane < waves ? partial[base + lane] : 0.f;
    v = wave_sum(v);
    __syncthreads();
    return v;
}
template <int K, int P>
__global__ __launch_bounds__(512) void delta_mixer_block(DeltaMixer d) {
    int h = blockIdx.x, tid = threadIdx.x, wave = tid / kWave, waves = blockDim.x / kWave;
    int kd = d.key_dim, vd = d.value_dim, kh = d.key_heads, vh = d.value_heads, T = d.tokens;
    Params conv{};
    conv.p[0] = d.convolved; conv.p[1] = const_cast<float*>(d.projected); conv.p[2] = const_cast<float*>(d.conv_w);
    conv.p[3] = d.conv_state; conv.n[0] = d.kernel; conv.n[1] = T; conv.n[2] = d.channels;
    for (int i = tid; i < 2 * kd + vd; i += blockDim.x) {
        int ch = i < kd ? h * kd + i : i < 2 * kd ? kh * kd + h * kd + (i - kd) : 2 * kh * kd + h * vd + (i - 2 * kd);
        conv_silu_channel(ch, conv);
    }
    __syncthreads();
    Params qk{};
    qk.p[0] = d.query; qk.p[1] = d.convolved; qk.p[2] = d.key;
    qk.n[0] = kh; qk.n[1] = vh; qk.n[2] = kd; qk.n[3] = d.channels;
    for (int r = wave; r < T; r += waves) delta_qk_item(qk, r * vh + h);
    __syncthreads();
    Params rule{};
    rule.p[0] = d.core; rule.p[1] = d.query; rule.p[2] = d.key; rule.p[3] = d.convolved; rule.p[4] = d.recurrent;
    rule.p[5] = const_cast<float*>(d.a); rule.p[6] = const_cast<float*>(d.b); rule.p[7] = const_cast<float*>(d.gates);
    rule.n[0] = kd; rule.n[1] = vd; rule.n[2] = kh; rule.n[3] = vh; rule.n[4] = T; rule.n[5] = d.channels;
    delta_rule_column<K, P, true, true>(rule, vh * vd, h * vd * P + tid);
    __syncthreads();
    int groups = blockDim.x / kNormGroup, g = tid / kNormGroup, i = tid % kNormGroup;
    for (int r0 = 0; r0 < T; r0 += groups) {
        int r = r0 + g;
        size_t off = (size_t(r) * vh + h) * vd;
        float x = r < T ? d.core[off + i] : 0.f;
        float sum = fmaf(x, x, 0.f);
        float rr = 1.f / sqrtf(group_sum(sum) / float(vd) + d.eps);
        if (r < T) d.gated[off + i] = silu(d.z[off + i]) * ((d.core[off + i] * rr) * (d.norm_w[i] + 0.f));
    }
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
            // Read both, then select: `upper ? v[i] : v[i + count]` became a
            // runtime index into v (a compare/select chain over all 16
            // registers per element, ADR 0019). Same values either way.
            float lo = v[i], hi = v[i + count];
            float send = upper ? lo : hi, keep = upper ? hi : lo;
            v[i] = keep + __shfl_xor(send, offset, kWave);
        }
    }
    float sum = v[0];
    for (int offset = kWave / kAttnBlock / 2; offset > 0; offset >>= 1) sum += __shfl_xor(sum, offset, kWave);
    return sum;
}
// FULL (d == L * 32, e.g. head_dim 256 with L = 8): every lane owns L whole
// dims and the K/V block past the last position is zero-filled in LDS, so the
// inner loops need no per-element bounds checks. A zero K row adds
// fmaf(q, 0, 0) = 0 to a score that is masked anyway, and a zero V row adds
// fmaf(0, 0, acc) = acc, so the results are bitwise those of the checked
// loops; the kernel is much smaller (each dispatch fetches its instructions
// anew, ADR 0019).
template <int L, bool FULL>
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
        if (FULL || d % 4 == 0) { // float4 copies, no per-element division by d
            int q4 = d / 4, staged = FULL ? kAttnBlock : n;
            for (int e = threadIdx.x; e < staged * q4; e += blockDim.x) {
                int j = e / q4, c = (e - j * q4) * 4;
                float4 kz = make_float4(0.f, 0.f, 0.f, 0.f), vz = kz;
                if (j < n) {
                    size_t at = (size_t(p0 + j) * kv + h) * d + c;
                    kz = *reinterpret_cast<const float4*>(&a.p[2][at]);
                    vz = *reinterpret_cast<const float4*>(&a.p[3][at]);
                }
                *reinterpret_cast<float4*>(&ks[j * d + c]) = kz;
                *reinterpret_cast<float4*>(&vs[j * d + c]) = vz;
            }
        } else {
            for (int e = threadIdx.x; e < n * d; e += blockDim.x) {
                int j = e / d, c = e % d;
                size_t at = (size_t(p0 + j) * kv + h) * d + c;
                ks[j * d + c] = a.p[2][at];
                vs[j * d + c] = a.p[3][at];
            }
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
                    if (FULL || (j < n && c < d)) sum = fmaf(q[i], ks[j * d + c], sum);
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
                    if (FULL || (jj < n && c < d)) acc[i] = fmaf(weight, vs[jj * d + c], acc[i]);
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
// Merge split partial softmaxes. Query vector v (= row * heads + head) has
// `splits` records of (max, denominator, sums[d]) starting at
// parts + v * stride; thread per (vector, dim). Decode: parts = p5, stride =
// n6 (one row); 16-bit prefill: the bridge's own scratch.
__global__ void attention_combine(Params a, const float* parts, long stride, int vectors, int splits) {
    int d = a.n[0], t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= d * vectors) return;
    int v = t / d, c = t % d;
    const float* part = parts + size_t(v) * stride;
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

// ---- Opt-in 16-bit prefill attention (qr_set_option "attention_f16") ----
// The same flash attention, but S = Q K^T and O += P V run on RDNA3's
// matrix instruction v_wmma_f32_16x16x16_f16: FP16 inputs, FP32 accumulate.
// Q (pre-scaled by 1/sqrt(d)), K, V and the softmax probabilities P are
// rounded to FP16 when they enter a WMMA; the KV cache, the softmax
// statistics and O stay FP32. This changes values (ADR 0014), so it is never
// the default and decode (T = 1) keeps the FP32 kernels.
//
// A block = 16 query vectors: 16/G rows x the G query heads of KV head
// blockIdx.y; blockIdx.z is a split of the positions (span each), merged by
// attention_combine. head_dim is split over kWmmaWaves(DT) waves of at most
// 4 tiles each (DT = d/16 = 16 in one wave needs 511 spilled VGPRs): each
// wave forms S over its dims, the partial S are summed in LDS in a fixed
// order, every wave runs the same softmax, and keeps O only for its dims.
// Fragment layouts for wave32 (checked on gfx1100): A[m][k]: lane l holds row
// l%16, k = 0..15; B[k][n]: lane l holds column l%16, k = 0..15; C[m][n]:
// lane l element i is row 2i + l/16, column l%16 (lanes 16..31 repeat the
// A/B operands of lanes 0..15).
typedef _Float16 half16 __attribute__((ext_vector_type(16)));
typedef float float8 __attribute__((ext_vector_type(8)));
constexpr int kWmma = 16;
constexpr int kWmmaSpan = 256; // positions per split
constexpr int kWmmaWaves(int dt) { return dt > 4 ? dt / 4 : 1; }
template <int DT>
__global__ __launch_bounds__(kWave * kWmmaWaves(DT)) void attention_wmma(Params a, int span, int splits, float* parts) {
    constexpr int d = DT * kWmma, W = kWmmaWaves(DT), TW = DT / W; // TW tiles per wave
    __shared__ _Float16 qs[kWmma][d];         // A operand of S = Q K^T
    __shared__ _Float16 ps[kWmma][kWmma + 2]; // P, C layout -> A layout
    __shared__ float partial[W][8][kWave];    // per-wave S, C layout
    int heads = a.n[1], kv = a.n[2], tokens = a.n[5], group = heads / kv, rows = kWmma / group;
    int lane = threadIdx.x % kWave, wave = threadIdx.x / kWave, n = lane % kWmma, half_ = lane / kWmma;
    int h = blockIdx.y, s = blockIdx.z, r0 = blockIdx.x * rows, t0 = wave * TW;
    const float scale = 1.f / sqrtf(float(d));
    for (int e = threadIdx.x; e < kWmma * d; e += blockDim.x) {
        int m = e / d, c = e % d, r = r0 + m / group, head = h * group + m % group;
        qs[m][c] = _Float16(r < tokens ? a.p[1][(size_t(r) * heads + head) * d + c] * scale : 0.f);
    }
    // This lane's C rows: m = 2i + half_.
    float m_row[8], l_row[8];
    int length[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int r = r0 + (2 * i + half_) / group;
        m_row[i] = -INFINITY; l_row[i] = 0.f;
        length[i] = r < tokens ? a.n[3] + r : 0; // padding rows see nothing
    }
    float8 o[TW];
#pragma unroll
    for (int t = 0; t < TW; ++t) o[t] = float8{};
    int last = min(r0 + rows, tokens) - 1;
    int begin = s * span, end = min(a.n[3] + last, begin + span);
    __syncthreads();
    for (int p0 = begin; p0 < end; p0 += kWmma) {
        // S[m][n] for positions p0 + n, over this wave's dims.
        float8 score = {};
        const float* key = a.p[2] + (size_t(p0 + n) * kv + h) * d;
        bool inside = p0 + n < end;
#pragma unroll
        for (int t = t0; t < t0 + TW; ++t) {
            half16 qa, kb;
#pragma unroll
            for (int k = 0; k < kWmma; ++k) {
                qa[k] = qs[n][t * kWmma + k];
                kb[k] = _Float16(inside ? key[t * kWmma + k] : 0.f);
            }
            score = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(qa, kb, score);
        }
        if (W > 1) {
#pragma unroll
            for (int i = 0; i < 8; ++i) partial[wave][i][lane] = score[i];
            __syncthreads();
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                float sum = partial[0][i][lane];
                for (int w = 1; w < W; ++w) sum += partial[w][i][lane];
                score[i] = sum;
            }
        }
        // Online softmax per row (identical in every wave); a row's 16
        // columns live in the 16 lanes of this half.
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            float v = p0 + n < length[i] ? score[i] : -INFINITY, top = v;
            for (int off = 1; off < kWmma; off <<= 1) top = fmaxf(top, __shfl_xor(top, off, kWave));
            float next = fmaxf(m_row[i], top);
            float e = next == -INFINITY ? 0.f : expf(v - next);
            float correction = next == -INFINITY ? 1.f : expf(m_row[i] - next);
            float sum = e;
            for (int off = 1; off < kWmma; off <<= 1) sum += __shfl_xor(sum, off, kWave);
            l_row[i] = l_row[i] * correction + sum;
            m_row[i] = next;
#pragma unroll
            for (int t = 0; t < TW; ++t) o[t][i] *= correction;
            if (wave == 0) ps[2 * i + half_][n] = _Float16(e);
        }
        __syncthreads();
        // O[m][c] += sum_j P[m][j] V[p0 + j][c] for this wave's dims.
        half16 pa;
#pragma unroll
        for (int k = 0; k < kWmma; ++k) pa[k] = ps[n][k];
#pragma unroll
        for (int t = 0; t < TW; ++t) {
            half16 vb;
            int c = (t0 + t) * kWmma + n;
#pragma unroll
            for (int k = 0; k < kWmma; ++k)
                vb[k] = _Float16(p0 + k < end ? a.p[3][(size_t(p0 + k) * kv + h) * d + c] : 0.f);
            o[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pa, vb, o[t]);
        }
        __syncthreads(); // ps and partial are rewritten by the next block
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int m = 2 * i + half_, r = r0 + m / group, head = h * group + m % group;
        if (r >= tokens) continue;
        size_t v = size_t(r) * heads + head;
        if (splits == 1) {
#pragma unroll
            for (int t = 0; t < TW; ++t) {
                size_t at = v * d + (t0 + t) * kWmma + n;
                a.p[0][at] = (o[t][i] / l_row[i]) * sigmoid(a.p[4][at]);
            }
            continue;
        }
        float* out = parts + (v * splits + s) * (d + 2);
        if (wave == 0 && n == 0) { out[0] = m_row[i]; out[1] = l_row[i]; }
#pragma unroll
        for (int t = 0; t < TW; ++t) out[2 + (t0 + t) * kWmma + n] = o[t][i];
    }
}
