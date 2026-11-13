// ---- RMSNorm over groups of n0 elements: direct scale (f1 = 0) or Qwen's (1 + w) (f1 = 1) ----

// Level 1: one thread per group loops its n0 elements, like src/ops.rs.
kernel void rms_norm(device float* o [[buffer(0)]], device const float* x [[buffer(1)]],
                     device const float* w [[buffer(2)]], ITEM_ARGS) {
    GUARD
    int d = a.n[0], off = int(t) * d; float sum = 0.f;
    for (int i = 0; i < d; ++i) sum += x[off + i] * x[off + i];
    float r = 1.f / sqrt(sum / float(d) + a.f[0]);
    for (int i = 0; i < d; ++i) o[off + i] = (x[off + i] * r) * (w[i] + a.f[1]);
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

// Level 2 (ADR 0009): one threadgroup per group, the sum of squares is a
// threadgroup reduction (the hidden-size norm is a single group, which level 1
// computes on one thread). Reordered sum: ADR 0004 tolerances.
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

// ---- Fused norms (ops 12 and 13, ADR 0021), each equal to the pair it replaces ----
// AddRmsNorm: p0 = p1 + p2, then p3 = RMSNorm(p0) * (p4 + f1)   (Add, then RmsNorm)
// RmsNormGate: p0 = silu(p3) * RMSNorm(p1) * (p2 + f1)          (RmsNorm, then SiluProduct)

// Level 1: one thread per group, like rms_norm.
kernel void add_rms_norm(device float* sum [[buffer(0)]], device const float* u [[buffer(1)]],
                         device const float* v [[buffer(2)]], device float* out [[buffer(3)]],
                         device const float* w [[buffer(4)]], ITEM_ARGS) {
    GUARD
    int d = a.n[0], off = int(t) * d; float ss = 0.f;
    for (int i = 0; i < d; ++i) { sum[off + i] = u[off + i] + v[off + i]; ss += sum[off + i] * sum[off + i]; }
    float r = 1.f / sqrt(ss / float(d) + a.f[0]);
    for (int i = 0; i < d; ++i) out[off + i] = (sum[off + i] * r) * (w[i] + a.f[1]);
}
kernel void rms_norm_gate(device float* o [[buffer(0)]], device const float* x [[buffer(1)]],
                          device const float* w [[buffer(2)]], device const float* z [[buffer(3)]], ITEM_ARGS) {
    GUARD
    int d = a.n[0], off = int(t) * d; float ss = 0.f;
    for (int i = 0; i < d; ++i) ss += x[off + i] * x[off + i];
    float r = 1.f / sqrt(ss / float(d) + a.f[0]);
    for (int i = 0; i < d; ++i) o[off + i] = silu_(z[off + i]) * ((x[off + i] * r) * (w[i] + a.f[1]));
}

// Level 2: one threadgroup per group with rms_norm_group's reduction, so each
// equals the separate level-2 ops bit for bit.
#define GROUP_ARGS constant Scalars& a [[buffer(8)]], uint group [[threadgroup_position_in_grid]], \
    uint tid [[thread_position_in_threadgroup]], uint threads [[threads_per_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], \
    uint sgs [[simdgroups_per_threadgroup]]
kernel void add_rms_norm_group(device float* sum [[buffer(0)]], device const float* u [[buffer(1)]],
                               device const float* v [[buffer(2)]], device float* out [[buffer(3)]],
                               device const float* w [[buffer(4)]], GROUP_ARGS) {
    threadgroup float scratch[32];
    int d = a.n[0];
    ulong off = ulong(group) * ulong(d);
    float ss = 0.f;
    for (int i = int(tid); i < d; i += int(threads)) { float s = u[off + i] + v[off + i]; sum[off + i] = s; ss += s * s; }
    ss = group_reduce(ss, scratch, lane, sg, sgs, SumOp());
    float r = 1.f / sqrt(ss / float(d) + a.f[0]);
    for (int i = int(tid); i < d; i += int(threads)) out[off + i] = (sum[off + i] * r) * (w[i] + a.f[1]);
}
kernel void rms_norm_gate_group(device float* o [[buffer(0)]], device const float* x [[buffer(1)]],
                                device const float* w [[buffer(2)]], device const float* z [[buffer(3)]], GROUP_ARGS) {
    threadgroup float scratch[32];
    int d = a.n[0];
    ulong off = ulong(group) * ulong(d);
    float ss = 0.f;
    for (int i = int(tid); i < d; i += int(threads)) ss += x[off + i] * x[off + i];
    ss = group_reduce(ss, scratch, lane, sg, sgs, SumOp());
    float r = 1.f / sqrt(ss / float(d) + a.f[0]);
    for (int i = int(tid); i < d; i += int(threads)) o[off + i] = silu_(z[off + i]) * ((x[off + i] * r) * (w[i] + a.f[1]));
}
