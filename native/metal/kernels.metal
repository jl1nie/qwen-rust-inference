// Apache-2.0. Metal Shading Language twin of native/rocm/bridge.cpp body().
// Equations mirror src/{ops,attention,deltanet}.rs; buffer slot i == Params.p[i].
// Compiled at runtime by bridge.mm with safe math (no fast-math approximations).
#include <metal_stdlib>
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
    int k = a.n[0]; device float* hist = history + int(t) * (k - 1); float sum = 0.f;
    for (int i = 0; i < k - 1; ++i) sum += hist[i] * w[int(t) * k + i];
    sum += x[t] * w[int(t) * k + k - 1];
    for (int i = 0; i < k - 2; ++i) hist[i] = hist[i + 1];
    if (k > 1) hist[k - 2] = x[t];
    o[t] = silu_(sum);
}
// Delta Q/K L2, repeat grouped key heads, Q / sqrt(key_dim). One thread per value head.
kernel void delta_qk(device float* o [[buffer(0)]], device const float* x [[buffer(1)]],
                     device float* w [[buffer(2)]], T) {
    GUARD
    int kh = a.n[0], vh = a.n[1], d = a.n[2], h = int(t) / (vh / kh), kd = kh * d;
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
    o[t] = -exp(alog[t]) * softplus_(x[t] + dt[t]); beta[t] = sigmoid_(w[t]);
}
// S[head,key,value]; one thread owns the entire value column
kernel void delta_rule(device float* o [[buffer(0)]], device const float* x [[buffer(1)]],
                       device const float* w [[buffer(2)]], device const float* mixed [[buffer(3)]],
                       device float* states [[buffer(4)]], device const float* g [[buffer(5)]],
                       device const float* beta [[buffer(6)]], T) {
    GUARD
    int kd = a.n[0], vd = a.n[1], kh = a.n[2], h = int(t) / vd, j = int(t) % vd;
    device float* state = states + h * kd * vd;
    float decay = exp(g[h]), memory = 0.f;
    for (int i = 0; i < kd; ++i) { state[i * vd + j] *= decay; memory += state[i * vd + j] * w[h * kd + i]; }
    float delta = (mixed[2 * kh * kd + h * vd + j] - memory) * beta[h], out = 0.f;
    for (int i = 0; i < kd; ++i) { state[i * vd + j] += w[h * kd + i] * delta; out += state[i * vd + j] * x[h * kd + i]; }
    o[t] = out;
}
// split head-interleaved Q/gate, RMSNorm, partial RoPE. One thread per query head.
kernel void attention_query(device float* o [[buffer(0)]], device const float* x [[buffer(1)]],
                            device const float* w [[buffer(2)]], device float* gate [[buffer(3)]], T) {
    GUARD
    int d = a.n[0], rot = a.n[1], pos = a.n[2], ti = int(t); float sum = 0.f;
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
    int d = a.n[0], rot = a.n[1], pos = a.n[2], kv = a.n[3], ti = int(t); float sum = 0.f;
    for (int i = 0; i < d; ++i) sum += x[ti * d + i] * x[ti * d + i];
    float r = 1.f / sqrt(sum / float(d) + a.f[0]);
    device float* dest = keys + (pos * kv + ti) * d;
    for (int i = 0; i < d; ++i) { dest[i] = (x[ti * d + i] * r) * (1.f + w[i]); values[(pos * kv + ti) * d + i] = value[ti * d + i]; }
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
// y[row] = A[row,:] . x for row-major A [rows, cols] (Rust [out,in], no transpose).
// One 32-lane SIMD group per row: lanes stride the row (coalesced), then simd_sum.
// Summation order differs from the CPU loop; tolerances in ADR 0004 cover it.
kernel void gemv(device float* y [[buffer(0)]], device const float* m [[buffer(1)]],
                 device const float* x [[buffer(2)]], constant Scalars& a [[buffer(8)]],
                 uint group [[threadgroup_position_in_grid]],
                 uint sg [[simdgroup_index_in_threadgroup]],
                 uint groups [[simdgroups_per_threadgroup]],
                 uint lane [[thread_index_in_simdgroup]],
                 uint width [[threads_per_simdgroup]]) {
    int rows = a.n[0], cols = a.n[1], row = int(group * groups + sg);
    if (row >= rows) return; // uniform per SIMD group, so simd_sum stays valid
    device const float* r = m + ulong(row) * ulong(cols);
    float sum = 0.f;
    if (a.n[2]) { // 16-byte aligned rows and x: float4 loads
        device const float4* r4 = (device const float4*)r;
        device const float4* x4 = (device const float4*)x;
        for (int j = int(lane); j < cols / 4; j += int(width)) {
            float4 p = r4[j] * x4[j];
            sum += (p.x + p.y) + (p.z + p.w);
        }
    } else {
        for (int j = int(lane); j < cols; j += int(width)) sum += r[j] * x[j];
    }
    sum = simd_sum(sum);
    if (lane == 0) y[row] = sum;
}
