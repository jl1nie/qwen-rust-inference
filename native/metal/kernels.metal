// Apache-2.0. Metal Shading Language twin of native/rocm/bridge.cpp body().
// Equations mirror src/{ops,attention,deltanet}.rs; buffer slot i == Params.p[i].
// Compiled at runtime by bridge.mm with safe math (no fast-math approximations).
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Scalars travel separately from pointers: Metal binds each buffer slot itself.
struct Scalars { int n[8]; float f[4]; int count; };

inline float sigmoid_(float x) {
    if (x >= 0.f) return 1.f / (1.f + exp(-x));
    float e = exp(x); return e / (1.f + e);
}
inline float silu_(float x) { return x * sigmoid_(x); }
// MSL has no log1p; for x <= 20 the absolute error of log(1+e^x) is tiny.
inline float softplus_(float x) { return x > 20.f ? x : log(1.f + exp(x)); }
inline void rope(device float* x, int dim, int pos, float theta) {
    for (int i = 0; i < dim / 2; ++i) {
        float angle = float(pos) / pow(theta, float(2 * i) / float(dim));
        float s = sin(angle), c = cos(angle), a = x[i], b = x[i + dim / 2];
        x[i] = a * c - b * s; x[i + dim / 2] = b * c + a * s;
    }
}
#define T uint t [[thread_position_in_grid]], constant Scalars& a [[buffer(8)]]
#define GUARD if (int(t) >= a.count) return;

// grouped RMSNorm; direct scale (f1=0) or Qwen's (1+w) (f1=1). One thread per group.
kernel void rms_norm(device float* o [[buffer(0)]], device const float* x [[buffer(1)]],
                     device const float* w [[buffer(2)]], T) {
    GUARD
    int d = a.n[0], off = int(t) * d; float sum = 0.f;
    for (int i = 0; i < d; ++i) sum += x[off + i] * x[off + i];
    float r = 1.f / sqrt(sum / float(d) + a.f[0]);
    for (int i = 0; i < d; ++i) o[off + i] = (x[off + i] * r) * (w[i] + a.f[1]);
}
kernel void add(device float* o [[buffer(0)]], device const float* x [[buffer(1)]],
                device const float* w [[buffer(2)]], T) {
    GUARD o[t] = x[t] + w[t];
}
kernel void silu_product(device float* o [[buffer(0)]], device const float* x [[buffer(1)]],
                         device const float* w [[buffer(2)]], T) {
    GUARD o[t] = silu_(x[t]) * w[t];
}
// conv channels are independent; history is oldest -> newest
kernel void conv_silu(device float* o [[buffer(0)]], device const float* x [[buffer(1)]],
                      device const float* w [[buffer(2)]], device float* history [[buffer(3)]], T) {
    GUARD
    // n1 tokens (rows of n2 channels) in time order through this channel's history.
    int k = a.n[0], tokens = max(a.n[1], 1), stride = a.n[2], ti = int(t);
    device float* hist = history + ti * (k - 1);
    for (int r = 0; r < tokens; ++r) {
        float xv = x[r * stride + ti], sum = 0.f;
        for (int i = 0; i < k - 1; ++i) sum += hist[i] * w[ti * k + i];
        sum += xv * w[ti * k + k - 1];
        for (int i = 0; i < k - 2; ++i) hist[i] = hist[i + 1];
        if (k > 1) hist[k - 2] = xv;
        o[r * stride + ti] = silu_(sum);
    }
}
// Delta Q/K L2, repeat grouped key heads, Q / sqrt(key_dim).
// One thread per (token, value head); n3 = row stride of the convolved input.
kernel void delta_qk(device float* o [[buffer(0)]], device const float* xs [[buffer(1)]],
                     device float* w [[buffer(2)]], T) {
    GUARD
    int kh = a.n[0], vh = a.n[1], d = a.n[2], h = (int(t) % vh) / (vh / kh), kd = kh * d;
    device const float* x = xs + (int(t) / vh) * a.n[3];
    float sq = 0.f, sk = 0.f;
    for (int i = 0; i < d; ++i) { float q = x[h * d + i], k = x[kd + h * d + i]; sq += q * q; sk += k * k; }
    float rq = 1.f / sqrt(sq + 1e-6f), rk = 1.f / sqrt(sk + 1e-6f);
    for (int i = 0; i < d; ++i) {
        o[int(t) * d + i] = (x[h * d + i] * rq) / sqrt(float(d));
        w[int(t) * d + i] = x[kd + h * d + i] * rk;
    }
}
// p0=g (log decay), p1=a, p2=b, p3=dt, p4=A_log, p5=beta
kernel void delta_gates(device float* o [[buffer(0)]], device const float* x [[buffer(1)]],
                        device const float* w [[buffer(2)]], device const float* dt [[buffer(3)]],
                        device const float* alog [[buffer(4)]], device float* beta [[buffer(5)]], T) {
    GUARD
    int head = a.n[0] > 0 ? int(t) % a.n[0] : int(t); // t = token*heads + head
    o[t] = -exp(alog[head]) * softplus_(x[t] + dt[head]); beta[t] = sigmoid_(w[t]);
}
// S[head,key,value]; one thread owns the entire value column
kernel void delta_rule(device float* o [[buffer(0)]], device const float* x [[buffer(1)]],
                       device const float* w [[buffer(2)]], device const float* mixed [[buffer(3)]],
                       device float* states [[buffer(4)]], device const float* g [[buffer(5)]],
                       device const float* beta [[buffer(6)]], T) {
    GUARD
    // n4 tokens in time order; n5 = row stride of the convolved (value) input.
    int kd = a.n[0], vd = a.n[1], kh = a.n[2], vh = a.n[3], h = int(t) / vd, j = int(t) % vd;
    int tokens = max(a.n[4], 1), stride = a.n[5];
    device float* state = states + h * kd * vd;
    for (int r = 0; r < tokens; ++r) {
        device const float* q = x + r * vh * kd;
        device const float* k = w + r * vh * kd;
        device const float* v = mixed + r * stride;
        float decay = exp(g[r * vh + h]), memory = 0.f;
        for (int i = 0; i < kd; ++i) { state[i * vd + j] *= decay; memory += state[i * vd + j] * k[h * kd + i]; }
        float delta = (v[2 * kh * kd + h * vd + j] - memory) * beta[r * vh + h], out = 0.f;
        for (int i = 0; i < kd; ++i) { state[i * vd + j] += k[h * kd + i] * delta; out += state[i * vd + j] * q[h * kd + i]; }
        o[r * vh * vd + int(t)] = out;
    }
}
// split head-interleaved Q/gate, RMSNorm, partial RoPE. One thread per query head.
kernel void attention_query(device float* o [[buffer(0)]], device const float* x [[buffer(1)]],
                            device const float* w [[buffer(2)]], device float* gate [[buffer(3)]], T) {
    GUARD
    // t = token*n3 + head (n3 = heads; 0 means a single token at n2)
    int ti = int(t), d = a.n[0], rot = a.n[1], pos = a.n[2] + (a.n[3] > 0 ? ti / a.n[3] : 0);
    float sum = 0.f;
    for (int i = 0; i < d; ++i) { float v = x[ti * 2 * d + i]; sum += v * v; }
    float r = 1.f / sqrt(sum / float(d) + a.f[0]);
    for (int i = 0; i < d; ++i) { o[ti * d + i] = (x[ti * 2 * d + i] * r) * (1.f + w[i]); gate[ti * d + i] = x[ti * 2 * d + d + i]; }
    rope(o + ti * d, rot, pos, a.f[1]);
}
// normalize/rotate K then append K and original V to time-major KV
kernel void attention_kv(device float* keys [[buffer(0)]], device const float* x [[buffer(1)]],
                         device const float* w [[buffer(2)]], device const float* value [[buffer(3)]],
                         device float* values [[buffer(4)]], T) {
    GUARD
    // t = token*kv + kv head; token r is appended at position n2 + r.
    int ti = int(t), d = a.n[0], rot = a.n[1], kv = a.n[3], pos = a.n[2] + ti / kv, hh = ti % kv;
    float sum = 0.f;
    for (int i = 0; i < d; ++i) sum += x[ti * d + i] * x[ti * d + i];
    float r = 1.f / sqrt(sum / float(d) + a.f[0]);
    device float* dest = keys + (pos * kv + hh) * d;
    for (int i = 0; i < d; ++i) { dest[i] = (x[ti * d + i] * r) * (1.f + w[i]); values[(pos * kv + hh) * d + i] = value[ti * d + i]; }
    rope(dest, rot, pos, a.f[1]);
}
// Attention read for one decode query, split into three parallel passes so
// the work per thread no longer grows with context length * head_dim:
//   1. scores[h,p] = q_h . k_p / sqrt(d)   one thread per (position, head)
//   2. softmax over p per head              one threadgroup per head
//   3. o[h,i] = sum_p prob * v_p[i] * gate  one thread per (dim, head)
// bridge.mm dispatches all three for op AttentionRead=10, so the C ABI and
// the HIP/CPU single-thread-per-head body are unchanged. Pass 1 keeps the
// serial dot product of the CPU loop; pass 2/3 reorder sums (ADR 0004 tolerances).
kernel void attention_scores(device const float* x [[buffer(1)]], device const float* w [[buffer(2)]],
                             device float* score_rows [[buffer(5)]], constant Scalars& a [[buffer(8)]],
                             uint2 g [[thread_position_in_grid]]) {
    int d = a.n[0], heads = a.n[1], kv = a.n[2], length = a.n[3], capacity = a.n[4];
    int p = int(g.x), ti = int(g.y);
    if (p >= length || ti >= heads) return;
    int h = ti / (heads / kv); float sum = 0.f;
    device const float* q = x + ti * d;
    device const float* k = w + (ulong(p) * ulong(kv) + ulong(h)) * ulong(d);
    for (int i = 0; i < d; ++i) sum += q[i] * k[i];
    score_rows[ulong(ti) * ulong(capacity) + ulong(p)] = sum / sqrt(float(d));
}
// Threadgroup-wide reduction: SIMD-group partials through threadgroup memory.
template <typename F>
inline float group_reduce(float v, threadgroup float* scratch, uint lane, uint sg, uint sgs, F op) {
    v = op(v, 0); // per-SIMD reduce
    if (lane == 0) scratch[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = scratch[0];
    for (uint i = 1; i < sgs; ++i) r = op(r, scratch[i], 1);
    threadgroup_barrier(mem_flags::mem_threadgroup); // scratch reusable afterwards
    return r;
}
struct MaxOp {
    float operator()(float v, int) const { return simd_max(v); }
    float operator()(float a, float b, int) const { return fmax(a, b); }
};
struct SumOp {
    float operator()(float v, int) const { return simd_sum(v); }
    float operator()(float a, float b, int) const { return a + b; }
};
kernel void attention_softmax(device float* score_rows [[buffer(5)]], constant Scalars& a [[buffer(8)]],
                              uint head [[threadgroup_position_in_grid]],
                              uint tid [[thread_position_in_threadgroup]],
                              uint threads [[threads_per_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]],
                              uint sgs [[simdgroups_per_threadgroup]]) {
    threadgroup float scratch[32];
    int length = a.n[3], capacity = a.n[4];
    device float* scores = score_rows + ulong(head) * ulong(capacity);
    float mx = -INFINITY;
    for (int p = int(tid); p < length; p += int(threads)) mx = fmax(mx, scores[p]);
    mx = group_reduce(mx, scratch, lane, sg, sgs, MaxOp());
    float den = 0.f;
    for (int p = int(tid); p < length; p += int(threads)) { float e = exp(scores[p] - mx); scores[p] = e; den += e; }
    den = group_reduce(den, scratch, lane, sg, sgs, SumOp());
    for (int p = int(tid); p < length; p += int(threads)) scores[p] /= den;
}
kernel void attention_values(device float* o [[buffer(0)]], device const float* values [[buffer(3)]],
                             device const float* gate [[buffer(4)]], device const float* score_rows [[buffer(5)]],
                             constant Scalars& a [[buffer(8)]], uint2 g [[thread_position_in_grid]]) {
    int d = a.n[0], heads = a.n[1], kv = a.n[2], length = a.n[3], capacity = a.n[4];
    int i = int(g.x), ti = int(g.y);
    if (i >= d || ti >= heads) return;
    int h = ti / (heads / kv); float sum = 0.f;
    device const float* scores = score_rows + ulong(ti) * ulong(capacity);
    // Adjacent threads read adjacent dims of the same V row: coalesced.
    for (int p = 0; p < length; ++p) sum += scores[p] * values[(ulong(p) * ulong(kv) + ulong(h)) * ulong(d) + ulong(i)];
    o[ti * d + i] = sum * sigmoid_(gate[ti * d + i]);
}
// device-to-device copy (embedding row lookup, layer output -> next input)
kernel void copy_floats(device float* o [[buffer(0)]], device const float* x [[buffer(1)]], T) {
    GUARD o[t] = x[t];
}
// bf16 is the upper half of an f32: widening is exact (no rounding).
inline float bf16_(ushort v) { return as_type<float>(uint(v) << 16); }
inline float widen(float v) { return v; }
inline float widen(ushort v) { return bf16_(v); }
inline float4 widen4(float4 v) { return v; }
inline float4 widen4(ushort4 u) { return float4(bf16_(u.x), bf16_(u.y), bf16_(u.z), bf16_(u.w)); }
// Widen embedding row n0 (row width n1) of a BF16 matrix. One thread per element.
kernel void embed_bf16(device float* o [[buffer(0)]], device const ushort* m [[buffer(1)]], T) {
    GUARD o[t] = bf16_(m[ulong(a.n[0]) * ulong(a.n[1]) + ulong(t)]);
}
// Y[b,row] = A[row,:] . X[b,:] for row-major A [rows, cols] (Rust [out,in], no
// transpose) and n3 = batch token rows of X/Y. One 32-lane SIMD group per row:
// lanes stride the row (coalesced), then simd_sum. Each pass reads the weight
// row once for up to 8 tokens; a token's summation order does not depend on
// the batch, so batched prefill equals per-token GEMV bit for bit. That order
// differs from the CPU loop; ADR 0004 tolerances cover it. E = float or BF16.
constant int kTokens = 8;
template <typename E, typename E4>
inline void gemm_rows(device float* y, device const E* m, device const float* x,
                      constant Scalars& a, uint group, uint sg, uint groups, uint lane, uint width) {
    int rows = a.n[0], cols = a.n[1], batch = max(a.n[3], 1), row = int(group * groups + sg);
    if (row >= rows) return; // uniform per SIMD group, so simd_sum stays valid
    device const E* r = m + ulong(row) * ulong(cols);
    for (int b0 = 0; b0 < batch; b0 += kTokens) {
        int nb = min(kTokens, batch - b0);
        float sum[kTokens] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        if (a.n[2]) { // aligned rows (float4: 16B, ushort4: 8B) and X rows: vector loads
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
    gemm_rows<float, float4>(y, m, x, a, group, sg, groups, lane, width);
}
kernel void gemv_bf16(device float* y [[buffer(0)]], device const ushort* m [[buffer(1)]],
                      device const float* x [[buffer(2)]], GEMM_ARGS) {
    gemm_rows<ushort, ushort4>(y, m, x, a, group, sg, groups, lane, width);
}
// Batched prefill (n3 >= 8 tokens): tiled GEMM on 8x8 simdgroup matrices.
// A threadgroup of 4 SIMD groups owns Y[32 tokens, 32 rows]; each K step
// stages X[32,32] and W[32,32] (BF16 widened exactly) in threadgroup memory,
// zero-padded past the edges. Summation order is blockwise (not the GEMV
// order), so results match per-token decode within tolerance, not bitwise.
constant int kTile = 32;
template <typename E>
inline void gemm_tiled(device float* y, device const E* m, device const float* x, constant Scalars& a,
                       threadgroup float* xs, threadgroup float* ws, uint2 tg, uint tid, uint sg) {
    int rows = a.n[0], cols = a.n[1], batch = a.n[3];
    int r0 = int(tg.x) * kTile, b0 = int(tg.y) * kTile;
    int sr = int(sg & 1) * 16, st = int(sg >> 1) * 16; // this SIMD group's 16x16 quadrant
    simdgroup_float8x8 acc[2][2];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    for (int k0 = 0; k0 < cols; k0 += kTile) {
        for (int i = int(tid); i < kTile * kTile; i += 128) {
            int rr = i / kTile, k = k0 + i % kTile;
            bool kin = k < cols;
            xs[i] = (kin && b0 + rr < batch) ? x[ulong(b0 + rr) * ulong(cols) + ulong(k)] : 0.f;
            ws[i] = (kin && r0 + rr < rows) ? widen(m[ulong(r0 + rr) * ulong(cols) + ulong(k)]) : 0.f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < kTile; kk += 8) {
            simdgroup_float8x8 xa[2], wb[2];
            for (int i = 0; i < 2; ++i) simdgroup_load(xa[i], xs + (st + i * 8) * kTile + kk, kTile);
            // ws is [row][k]; transposed load gives the [k][row] operand of X . W^T.
            for (int j = 0; j < 2; ++j)
                simdgroup_load(wb[j], ws + (sr + j * 8) * kTile + kk, kTile, ulong2(0, 0), true);
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j) simdgroup_multiply_accumulate(acc[i][j], xa[i], wb[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) simdgroup_store(acc[i][j], xs + (st + i * 8) * kTile + sr + j * 8, kTile);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = int(tid); i < kTile * kTile; i += 128) {
        int rr = i / kTile, cc = i % kTile;
        if (b0 + rr < batch && r0 + cc < rows) y[ulong(b0 + rr) * ulong(rows) + ulong(r0 + cc)] = xs[i];
    }
}
#define TILED_ARGS constant Scalars& a [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]], \
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]
kernel void gemm_tiled_f32(device float* y [[buffer(0)]], device const float* m [[buffer(1)]],
                           device const float* x [[buffer(2)]], TILED_ARGS) {
    threadgroup float xs[kTile * kTile], ws[kTile * kTile];
    gemm_tiled<float>(y, m, x, a, xs, ws, tg, tid, sg);
}
kernel void gemm_tiled_bf16(device float* y [[buffer(0)]], device const ushort* m [[buffer(1)]],
                            device const float* x [[buffer(2)]], TILED_ARGS) {
    threadgroup float xs[kTile * kTile], ws[kTile * kTile];
    gemm_tiled<ushort>(y, m, x, a, xs, ws, tg, tid, sg);
}
