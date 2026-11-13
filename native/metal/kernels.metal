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
// Same rule with each state column split over L lanes of one SIMD group
// (KD/L rows each, in registers). The two dot products per token become L
// partial sums combined with simd shuffles, so the per-token dependency chain
// is KD/L long instead of KD. Only the summation order differs from
// delta_rule (ADR 0016). Thread t: column (t / L), part (t % L).
template <int KD, int L>
kernel void delta_rule_split(device float* o [[buffer(0)]], device const float* x [[buffer(1)]],
                             device const float* w [[buffer(2)]], device const float* mixed [[buffer(3)]],
                             device float* states [[buffer(4)]], device const float* g [[buffer(5)]],
                             device const float* beta [[buffer(6)]], T) {
    constexpr int R = KD / L;
    int vd = a.n[1], kh = a.n[2], vh = a.n[3], tokens = max(a.n[4], 1), stride = a.n[5];
    int col = int(t) / L, part = int(t) % L;
    bool live = col < a.count; // count = value_heads * value_dim columns
    int h = live ? col / vd : 0, j = live ? col % vd : 0, i0 = part * R;
    device float* state = states + h * KD * vd;
    float s[R];
    for (int i = 0; i < R; ++i) s[i] = live ? state[(i0 + i) * vd + j] : 0.f;
    for (int r = 0; r < tokens; ++r) {
        device const float* q = x + r * vh * KD + h * KD + i0;
        device const float* k = w + r * vh * KD + h * KD + i0;
        float decay = exp(g[r * vh + h]), memory = 0.f;
        for (int i = 0; i < R; ++i) { s[i] *= decay; memory += s[i] * k[i]; }
        for (int d = 1; d < L; d *= 2) memory += simd_shuffle_xor(memory, ushort(d));
        float delta = (mixed[r * stride + 2 * kh * KD + h * vd + j] - memory) * beta[r * vh + h], out = 0.f;
        for (int i = 0; i < R; ++i) { s[i] += k[i] * delta; out += s[i] * q[i]; }
        for (int d = 1; d < L; d *= 2) out += simd_shuffle_xor(out, ushort(d));
        if (live && part == 0) o[r * vh * vd + col] = out;
    }
    if (live)
        for (int i = 0; i < R; ++i) state[(i0 + i) * vd + j] = s[i];
}
#define DELTA_RULE_SPLIT(KD, L) \
    template [[host_name("delta_rule_split_" #KD "_" #L)]] kernel void delta_rule_split<KD, L>( \
        device float*, device const float*, device const float*, device const float*, \
        device float*, device const float*, device const float*, uint, constant Scalars&);
DELTA_RULE_SPLIT(128, 4)
DELTA_RULE_SPLIT(64, 4)
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
// Grouped RMSNorm (op 1) with one threadgroup per group: the sum of squares
// is a threadgroup reduction instead of one thread looping d elements (the
// hidden-size norm has a single group). Reordered sum: ADR 0004 tolerances.
kernel void rms_norm_group(device float* o [[buffer(0)]], device const float* x [[buffer(1)]],
                           device const float* w [[buffer(2)]], constant Scalars& a [[buffer(8)]],
                           uint group [[threadgroup_position_in_grid]],
                           uint tid [[thread_position_in_threadgroup]],
                           uint threads [[threads_per_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]],
                           uint sgs [[simdgroups_per_threadgroup]]) {
    threadgroup float scratch[32];
    int d = a.n[0];
    ulong off = ulong(group) * ulong(d);
    float sum = 0.f;
    for (int i = int(tid); i < d; i += int(threads)) sum += x[off + i] * x[off + i];
    sum = group_reduce(sum, scratch, lane, sg, sgs, SumOp());
    float r = 1.f / sqrt(sum / float(d) + a.f[0]);
    for (int i = int(tid); i < d; i += int(threads)) o[off + i] = (x[off + i] * r) * (w[i] + a.f[1]);
}
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
constant constexpr int FLASH_LANES = 16; // dims per lane: head_dim <= 512
// Decode attention read split over positions (op 10, T = 1, n6 = per-head
// scratch floats). Threadgroup (split, kv head): G SIMD groups, one per query
// head sharing the KV head, stream positions [split*B, min(L, split*B+B))
// with the online softmax of attention_flash (B = n7, chosen by bridge.mm).
// Each writes its partial (max m, denominator l, unnormalized acc[d]) to
// scratch[head][split*(d+2)...]; attention_combine merges the splits.
// Every split holds >= 1 position, so m is finite.
kernel void attention_split(device const float* x [[buffer(1)]], device const float* keys [[buffer(2)]],
                            device const float* values [[buffer(3)]], device float* scratch [[buffer(5)]],
                            constant Scalars& a [[buffer(8)]],
                            uint2 block [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]]) {
    int d = a.n[0], heads = a.n[1], kv = a.n[2], length = a.n[3], row = a.n[6], span = a.n[7];
    int group = heads / kv, kvh = int(block.y), head = kvh * group + int(sg), split = int(block.x);
    int begin = split * span, end = min(length, begin + span);
    device const float* qh = x + ulong(head) * ulong(d);
    float q[FLASH_LANES], acc[FLASH_LANES];
#pragma unroll
    for (int e = 0; e < FLASH_LANES; ++e) { q[e] = e * 32 + int(lane) < d ? qh[e * 32 + lane] : 0.f; acc[e] = 0.f; }
    float scale = sqrt(float(d)), m = -INFINITY, l = 0.f;
    ulong step = ulong(kv) * ulong(d);
    device const float* k = keys + ulong(begin) * step + ulong(kvh) * ulong(d) + lane;
    device const float* v = values + ulong(begin) * step + ulong(kvh) * ulong(d) + lane;
    for (int p = begin; p < end; ++p, k += step, v += step) {
        float s = 0.f;
#pragma unroll
        for (int e = 0; e < FLASH_LANES; ++e) if (e * 32 + int(lane) < d) s += q[e] * k[e * 32];
        s = simd_sum(s) / scale;
        float next = fmax(m, s), correction = exp(m - next), weight = exp(s - next);
        l = l * correction + weight;
        m = next;
#pragma unroll
        for (int e = 0; e < FLASH_LANES; ++e) if (e * 32 + int(lane) < d) acc[e] = acc[e] * correction + weight * v[e * 32];
    }
    device float* out = scratch + ulong(head) * ulong(row) + ulong(split) * ulong(d + 2);
    if (lane == 0) { out[0] = m; out[1] = l; }
#pragma unroll
    for (int e = 0; e < FLASH_LANES; ++e) if (e * 32 + int(lane) < d) out[2 + e * 32 + lane] = acc[e];
}
// Decode attention split, position-parallel: 32 positions at a time, one
// per lane. Each lane computes its position's full dot product (float4 loads),
// the softmax max/sum are taken once per 32 positions (simd_max/simd_sum)
// instead of once per position, and the weighted V sum has lanes own
// dimensions with the weights broadcast by simd_shuffle. Same partial result
// layout as attention_split; only the summation order differs.
kernel void attention_split_par(device const float* x [[buffer(1)]], device const float* keys [[buffer(2)]],
                                device const float* values [[buffer(3)]], device float* scratch [[buffer(5)]],
                                constant Scalars& a [[buffer(8)]],
                                uint2 block [[threadgroup_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]]) {
    int d = a.n[0], heads = a.n[1], kv = a.n[2], length = a.n[3], row = a.n[6], span = a.n[7];
    int group = heads / kv, kvh = int(block.y), head = kvh * group + int(sg), split = int(block.x);
    int begin = split * span, end = min(length, begin + span);
    device const float* qh = x + ulong(head) * ulong(d);
    float acc[FLASH_LANES];
#pragma unroll
    for (int e = 0; e < FLASH_LANES; ++e) acc[e] = 0.f;
    float scale = sqrt(float(d)), m = -INFINITY, l = 0.f;
    ulong step = ulong(kv) * ulong(d);
    device const float* kb = keys + ulong(kvh) * ulong(d);
    device const float* vb = values + ulong(kvh) * ulong(d) + lane;
    for (int c0 = begin; c0 < end; c0 += 32) {
        int p = c0 + int(lane);
        float s = -INFINITY;
        if (p < end) {
            device const float4* k4 = (device const float4*)(kb + ulong(p) * step);
            device const float4* q4 = (device const float4*)qh;
            float4 t0 = 0.f, t1 = 0.f;
            for (int i = 0; i < d / 4; i += 2) {
                t0 += q4[i] * k4[i];
                if (i + 1 < d / 4) t1 += q4[i + 1] * k4[i + 1];
            }
            float4 t = t0 + t1;
            s = (t.x + t.y + t.z + t.w) / scale;
        }
        float next = fmax(m, simd_max(s));
        float correction = exp(m - next), w = p < end ? exp(s - next) : 0.f;
        l = l * correction + simd_sum(w);
        m = next;
#pragma unroll
        for (int e = 0; e < FLASH_LANES; ++e) acc[e] *= correction;
        int n = min(32, end - c0);
        for (int j = 0; j < n; ++j) {
            float wj = simd_shuffle(w, ushort(j));
            device const float* vr = vb + ulong(c0 + j) * step;
#pragma unroll
            for (int e = 0; e < FLASH_LANES; ++e) if (e * 32 + int(lane) < d) acc[e] += wj * vr[e * 32];
        }
    }
    device float* out = scratch + ulong(head) * ulong(row) + ulong(split) * ulong(d + 2);
    if (lane == 0) { out[0] = m; out[1] = l; }
#pragma unroll
    for (int e = 0; e < FLASH_LANES; ++e) if (e * 32 + int(lane) < d) out[2 + e * 32 + lane] = acc[e];
}
// As attention_split_par, but one SIMD group serves all G query heads of a KV
// head: every K/V element loaded is used for G heads (D = head_dim, a
// multiple of 32; lanes own dims lane + 32e). Same partial result layout.
template <int D, int G>
kernel void attention_split_grp(device const float* x [[buffer(1)]], device const float* keys [[buffer(2)]],
                                device const float* values [[buffer(3)]], device float* scratch [[buffer(5)]],
                                constant Scalars& a [[buffer(8)]],
                                uint2 block [[threadgroup_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]]) {
    constexpr int E = D / 32;
    int heads = a.n[1], kv = a.n[2], length = a.n[3], row = a.n[6], span = a.n[7];
    int kvh = int(block.y), split = int(block.x), h0 = kvh * G;
    int begin = split * span, end = min(length, begin + span);
    float acc[G][E], m[G], l[G];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        m[g] = -INFINITY; l[g] = 0.f;
#pragma unroll
        for (int e = 0; e < E; ++e) acc[g][e] = 0.f;
    }
    float scale = sqrt(float(D));
    ulong step = ulong(kv) * ulong(D);
    device const float* kb = keys + ulong(kvh) * ulong(D);
    device const float* vb = values + ulong(kvh) * ulong(D) + lane;
    device const float4* q4 = (device const float4*)(x + ulong(h0) * ulong(D));
    for (int c0 = begin; c0 < end; c0 += 32) {
        int p = c0 + int(lane);
        float s[G];
#pragma unroll
        for (int g = 0; g < G; ++g) s[g] = -INFINITY;
        if (p < end) {
            device const float4* k4 = (device const float4*)(kb + ulong(p) * step);
            float4 t[G];
#pragma unroll
            for (int g = 0; g < G; ++g) t[g] = 0.f;
            for (int i = 0; i < D / 4; ++i) {
                float4 kk = k4[i];
#pragma unroll
                for (int g = 0; g < G; ++g) t[g] += q4[g * (D / 4) + i] * kk;
            }
#pragma unroll
            for (int g = 0; g < G; ++g) s[g] = (t[g].x + t[g].y + t[g].z + t[g].w) / scale;
        }
        float w[G];
#pragma unroll
        for (int g = 0; g < G; ++g) {
            float next = fmax(m[g], simd_max(s[g]));
            float correction = exp(m[g] - next);
            w[g] = p < end ? exp(s[g] - next) : 0.f;
            l[g] = l[g] * correction + simd_sum(w[g]);
            m[g] = next;
#pragma unroll
            for (int e = 0; e < E; ++e) acc[g][e] *= correction;
        }
        int n = min(32, end - c0);
        for (int j = 0; j < n; ++j) {
            device const float* vr = vb + ulong(c0 + j) * step;
            float vv[E];
#pragma unroll
            for (int e = 0; e < E; ++e) vv[e] = vr[e * 32];
#pragma unroll
            for (int g = 0; g < G; ++g) {
                float wj = simd_shuffle(w[g], ushort(j));
#pragma unroll
                for (int e = 0; e < E; ++e) acc[g][e] += wj * vv[e];
            }
        }
    }
#pragma unroll
    for (int g = 0; g < G; ++g) {
        device float* out = scratch + ulong(h0 + g) * ulong(row) + ulong(split) * ulong(D + 2);
        if (lane == 0) { out[0] = m[g]; out[1] = l[g]; }
#pragma unroll
        for (int e = 0; e < E; ++e) out[2 + e * 32 + lane] = acc[g][e];
    }
    (void)heads;
}
#define ATTENTION_SPLIT_GRP(D, G) \
    template [[host_name("attention_split_grp_" #D "_" #G)]] kernel void attention_split_grp<D, G>( \
        device const float*, device const float*, device const float*, device float*, constant Scalars&, uint2, uint);
ATTENTION_SPLIT_GRP(256, 4)
ATTENTION_SPLIT_GRP(128, 4)
ATTENTION_SPLIT_GRP(64, 4)
// o[h,i] = sum_s exp(m_s - M) acc_s[i] / sum_s exp(m_s - M) l_s, then the
// sigmoid output gate. One thread per (dim, head); n5 = number of splits.
kernel void attention_combine(device float* o [[buffer(0)]], device const float* gate [[buffer(4)]],
                              device const float* scratch [[buffer(5)]], constant Scalars& a [[buffer(8)]],
                              uint2 g [[thread_position_in_grid]]) {
    int d = a.n[0], heads = a.n[1], row = a.n[6], splits = a.n[5];
    int i = int(g.x), head = int(g.y);
    if (i >= d || head >= heads) return;
    device const float* part = scratch + ulong(head) * ulong(row);
    float mx = -INFINITY;
    for (int s = 0; s < splits; ++s) mx = fmax(mx, part[s * (d + 2)]);
    float den = 0.f, sum = 0.f;
    for (int s = 0; s < splits; ++s) {
        device const float* ps = part + s * (d + 2);
        float w = exp(ps[0] - mx);
        den += w * ps[1];
        sum += w * ps[2 + i];
    }
    o[head * d + i] = sum / den * sigmoid_(gate[head * d + i]);
}
// Batched causal attention read for a prefill chunk (op 10 with n5 = T > 1).
// No score rows: each SIMD group streams K/V once with an online softmax
// (running max m and denominator l, rescaling the accumulator when m grows).
// Threadgroup = `rows` query rows x G query heads sharing one KV head
// (G = heads/kv), all reading the same K/V positions at about the same time,
// so a position is fetched from memory once per threadgroup, not per head.
// SIMD group sg: row r = block*rows + sg/G, head = kvh*G + sg%G. Lane owns dims
// lane, lane+32, ... below head_dim (coalesced). Row r sees n3 + r positions (causal).
// Summation order differs from the 3-pass decode read (ADR 0004 tolerances).
kernel void attention_flash(device float* o [[buffer(0)]], device const float* x [[buffer(1)]],
                            device const float* keys [[buffer(2)]], device const float* values [[buffer(3)]],
                            device const float* gate [[buffer(4)]], constant Scalars& a [[buffer(8)]],
                            uint2 block [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]],
                            uint sgs [[simdgroups_per_threadgroup]]) {
    int d = a.n[0], heads = a.n[1], kv = a.n[2], tokens = a.n[5];
    int group = heads / kv, rows = int(sgs) / group;
    int r = int(block.x) * rows + int(sg) / group, kvh = int(block.y), head = kvh * group + int(sg) % group;
    if (r >= tokens || int(sg) >= rows * group) return; // no barriers below
    int length = a.n[3] + r; // lane owns dim e*32+lane while it is < d
    ulong row = (ulong(r) * ulong(heads) + ulong(head)) * ulong(d);
    float q[FLASH_LANES], acc[FLASH_LANES];
#pragma unroll
    for (int e = 0; e < FLASH_LANES; ++e) { q[e] = e * 32 + int(lane) < d ? x[row + ulong(e * 32) + lane] : 0.f; acc[e] = 0.f; }
    float scale = sqrt(float(d)), m = -INFINITY, l = 0.f;
    ulong step = ulong(kv) * ulong(d);
    device const float* k = keys + ulong(kvh) * ulong(d) + lane;
    device const float* v = values + ulong(kvh) * ulong(d) + lane;
    for (int p = 0; p < length; ++p, k += step, v += step) {
        float s = 0.f;
#pragma unroll
        for (int e = 0; e < FLASH_LANES; ++e) if (e * 32 + int(lane) < d) s += q[e] * k[e * 32];
        s = simd_sum(s) / scale;
        float next = fmax(m, s), correction = exp(m - next), weight = exp(s - next);
        l = l * correction + weight;
        m = next;
#pragma unroll
        for (int e = 0; e < FLASH_LANES; ++e) if (e * 32 + int(lane) < d) acc[e] = acc[e] * correction + weight * v[e * 32];
    }
#pragma unroll
    for (int e = 0; e < FLASH_LANES; ++e)
        if (e * 32 + int(lane) < d) { ulong i = row + ulong(e * 32) + lane; o[i] = acc[e] / l * sigmoid_(gate[i]); }
}
// Matrix-unit variant of attention_flash for head_dim D in {64,128,256}
// (op 10, n5 = T > 1). Threadgroup = 8 consecutive query rows x G heads that
// share KV head block.y; SIMD group sg owns head kvh*G+sg for those 8 rows.
// Per block of 8 positions: K/V rows are staged once in threadgroup memory
// (zero past the written KV), S[8x8] = Q K^T with simdgroup_float8x8, lanes
// 0..7 apply the causal mask and the online softmax of their row, then
// O = diag(correction) O + P V. The query buffer is padded to a multiple of 8
// rows (Rust), rows >= T are computed with P = 0 and never written.
constant constexpr int MMA_MAX_GROUP = 8;
// Prefill attention, flash style with 8x8 simdgroup matrices. One
// threadgroup = 8 query rows x G heads sharing one KV head; each simdgroup
// (one head) runs independently: no threadgroup staging or barriers.
// Q (8 rows x D) stays in registers; positions are taken 32 at a time (4
// score tiles), so the softmax correction of the accumulator happens once per
// 32 positions. K/V 8x8 tiles are read straight from device memory (the G
// simdgroups of a threadgroup read the same tiles, served by the cache). The
// KV buffers are padded to a multiple of 32 rows (src/device.rs); rows past
// a query row's causal limit are masked to probability 0.
constant constexpr int MMA_BLOCK = 32;
template <int D>
kernel void attention_mma(device float* o [[buffer(0)]], device const float* x [[buffer(1)]],
                          device const float* keys [[buffer(2)]], device const float* values [[buffer(3)]],
                          device const float* gate [[buffer(4)]], constant Scalars& a [[buffer(8)]],
                          uint2 block [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint2 threads [[threads_per_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr int TILES = D / 8, SUB = MMA_BLOCK / 8;
    threadgroup float sp[MMA_MAX_GROUP][8 * MMA_BLOCK], diag[MMA_MAX_GROUP][64];
    int heads = a.n[1], kv = a.n[2], base = a.n[3], tokens = a.n[5];
    int group = heads / kv, kvh = int(block.y), head = kvh * group + int(sg), r0 = int(block.x) * 8;
    int span = base + min(r0 + 7, tokens - 1); // positions any row of this block sees
    ulong qstride = ulong(heads) * ulong(D), kvstride = ulong(kv) * ulong(D);
    device const float* q = x + ulong(r0) * qstride + ulong(head) * ulong(D);
    device const float* kbase = keys + ulong(kvh) * ulong(D);
    device const float* vbase = values + ulong(kvh) * ulong(D);
    threadgroup float* S = sp[sg];
    threadgroup float* C = diag[sg];
    for (int i = int(lane); i < 64; i += 32) C[i] = 0.f;
    // Softmax lanes: 4 per row, 8 positions each.
    int row = int(lane) / 4, col = (int(lane) % 4) * 8;
    bool live = r0 + row < tokens;
    int limit = base + r0 + row; // this row sees positions < limit
    float m = -INFINITY, l = 0.f, scale = sqrt(float(D));
    simdgroup_float8x8 acc[TILES];
    for (int t = 0; t < TILES; ++t) acc[t] = simdgroup_float8x8(0.f);
    for (int p0 = 0; p0 < span; p0 += MMA_BLOCK) {
        simdgroup_float8x8 s[SUB], kb, qa;
        for (int b = 0; b < SUB; ++b) s[b] = simdgroup_float8x8(0.f);
        for (int t = 0; t < TILES; ++t) {
            simdgroup_load(qa, q + t * 8, qstride);
            for (int b = 0; b < SUB; ++b) {
                simdgroup_load(kb, kbase + ulong(p0 + 8 * b) * kvstride + t * 8, kvstride, ulong2(0), true);
                simdgroup_multiply_accumulate(s[b], qa, kb, s[b]);
            }
        }
        for (int b = 0; b < SUB; ++b) simdgroup_store(s[b], S + 8 * b, MMA_BLOCK);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        float e[8], local = -INFINITY;
        for (int j = 0; j < 8; ++j) {
            int p = p0 + col + j;
            e[j] = live && p < limit ? S[row * MMA_BLOCK + col + j] / scale : -INFINITY;
            local = fmax(local, e[j]);
        }
        local = fmax(local, simd_shuffle_xor(local, 1));
        local = fmax(local, simd_shuffle_xor(local, 2));
        float next = fmax(m, local);
        // Live rows see position 0 in the first block, so next is finite from then on.
        float correction = live ? exp(m - next) : 1.f, sum = 0.f;
        for (int j = 0; j < 8; ++j) {
            float w = live ? exp(e[j] - next) : 0.f;
            S[row * MMA_BLOCK + col + j] = w;
            sum += w;
        }
        sum += simd_shuffle_xor(sum, 1);
        sum += simd_shuffle_xor(sum, 2);
        l = l * correction + sum;
        m = next;
        if (col == 0) C[row * 9] = correction;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 cm, pm[SUB], vb;
        simdgroup_load(cm, C, 8);
        for (int b = 0; b < SUB; ++b) simdgroup_load(pm[b], S + 8 * b, MMA_BLOCK);
        for (int t = 0; t < TILES; ++t) {
            simdgroup_multiply(acc[t], cm, acc[t]);
            for (int b = 0; b < SUB; ++b) {
                simdgroup_load(vb, vbase + ulong(p0 + 8 * b) * kvstride + t * 8, kvstride);
                simdgroup_multiply_accumulate(acc[t], pm[b], vb, acc[t]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup); // S, C reused next block
    }
    if (col == 0) C[row * 9] = live ? 1.f / l : 0.f;
    for (int t = 0; t < TILES; ++t) {
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_store(acc[t], S, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = int(lane); i < 64; i += 32) {
            int rr = i / 8;
            if (r0 + rr >= tokens) continue;
            ulong at = ulong(r0 + rr) * qstride + ulong(head) * ulong(D) + ulong(t * 8 + i % 8);
            o[at] = S[i] * C[rr * 9] * sigmoid_(gate[at]);
        }
    }
    (void)tid; (void)threads;
}
#define ATTENTION_MMA(D)                                                                              \
    template [[host_name("attention_mma_" #D)]] kernel void attention_mma<D>(                         \
        device float*, device const float*, device const float*, device const float*,                \
        device const float*, constant Scalars&, uint2, uint, uint2, uint, uint);
ATTENTION_MMA(64)
ATTENTION_MMA(128)
ATTENTION_MMA(256)
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
// Prefill GEMM Y[batch, rows] = X[batch, cols] . W[rows, cols]^T.
// One threadgroup = 64 rows x 64 tokens of Y, 4 SIMD groups of 32x32
// (4x4 accumulator tiles: 16 multiply-adds per 8 tile loads). Each K block of
// kK columns is staged in threadgroup memory (vector loads when n2 says rows
// are 4-aligned) and consumed 8 columns at a time in order, so every output
// element is accumulated in exactly the same order as before (bit-identical).
// Template: BM rows per threadgroup, each SIMD group I x J 8x8 tiles (I over
// tokens, J over rows), 4 SIMD groups. 64x64 (I=J=4) for the projections;
// 16x64 (I=J=2) for matrices of at most 16 rows (DeltaNet a/b), which would
// otherwise leave 48 of 64 tile rows empty. The K order is the same in both.
constant int kK = 32;
// (Metal allows no static members at program scope: constexpr functions.)
constexpr int tile_sgr(int bm, int j) { return bm / (8 * j); }             // SIMD groups along rows
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
            // 64 x 32 floats = 512 float4 per operand; 128 threads x 4.
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
