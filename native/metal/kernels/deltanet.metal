// ---- DeltaNet rule (src/deltanet.rs): per head, state S[key, value] ----
//   S <- S * exp(g);  memory = k^T S;  delta = (v - memory) * beta
//   S <- S + k delta^T;  out = q^T S
// Value columns evolve independently, so each column is one unit of work.
// n0 key_dim, n1 value_dim, n2 key heads, n3 value heads, n4 tokens (in time
// order), n5 row stride of the convolved (value) input; count = value_heads *
// value_dim columns.
//
// The gates (log decay g and beta per token and head) come from DeltaGates
// (p5 = g, p6 = beta), or, for op DeltaRuleGated (ADR 0021), are computed by
// the rule itself from DeltaGates' inputs (p5 = a, p6 = b, p7 = [dt | A_log])
// with DeltaGates' expressions, so both give the same values.
template <bool GATED>
inline void delta_gate(device const float* p5, device const float* p6, device const float* p7,
                       int i, int h, int vh, thread float& decay, thread float& beta) {
    if (GATED) {
        decay = exp(-exp(p7[vh + h]) * softplus_(p5[i] + p7[h]));
        beta = sigmoid_(p6[i]);
    } else {
        decay = exp(p5[i]);
        beta = p6[i];
    }
}

// Level 1: one thread owns a whole value column and walks the tokens, reading
// and writing the column in device memory, exactly like the CPU loop.
template <bool GATED>
kernel void delta_rule(device float* o [[buffer(0)]], device const float* x [[buffer(1)]],
                       device const float* w [[buffer(2)]], device const float* mixed [[buffer(3)]],
                       device float* states [[buffer(4)]], device const float* p5 [[buffer(5)]],
                       device const float* p6 [[buffer(6)]], device const float* p7 [[buffer(7)]], ITEM_ARGS) {
    GUARD
    int kd = a.n[0], vd = a.n[1], kh = a.n[2], vh = a.n[3], h = int(t) / vd, j = int(t) % vd;
    int tokens = max(a.n[4], 1), stride = a.n[5];
    device float* state = states + h * kd * vd;
    for (int r = 0; r < tokens; ++r) {
        device const float* q = x + r * vh * kd;
        device const float* k = w + r * vh * kd;
        device const float* v = mixed + r * stride;
        float decay, beta, memory = 0.f;
        delta_gate<GATED>(p5, p6, p7, r * vh + h, h, vh, decay, beta);
        for (int i = 0; i < kd; ++i) { state[i * vd + j] *= decay; memory += state[i * vd + j] * k[h * kd + i]; }
        float delta = (v[2 * kh * kd + h * vd + j] - memory) * beta, out = 0.f;
        for (int i = 0; i < kd; ++i) { state[i * vd + j] += k[h * kd + i] * delta; out += state[i * vd + j] * q[h * kd + i]; }
        o[r * vh * vd + int(t)] = out;
    }
}

// Level 2 (ADR 0016): each column split over L lanes of one SIMD group, KD/L
// rows each held in registers for the whole chunk. The two dot products per
// token become L partial sums combined with simd shuffles, so the per-token
// dependency chain is KD/L long instead of KD. Only the summation order differs.
// Thread t: column t / L, part t % L (whole SIMD groups are dispatched).
template <int KD, int L, bool GATED>
kernel void delta_rule_split(device float* o [[buffer(0)]], device const float* x [[buffer(1)]],
                             device const float* w [[buffer(2)]], device const float* mixed [[buffer(3)]],
                             device float* states [[buffer(4)]], device const float* p5 [[buffer(5)]],
                             device const float* p6 [[buffer(6)]], device const float* p7 [[buffer(7)]], ITEM_ARGS) {
    constexpr int R = KD / L;
    int vd = a.n[1], kh = a.n[2], vh = a.n[3], tokens = max(a.n[4], 1), stride = a.n[5];
    int col = int(t) / L, part = int(t) % L;
    bool live = col < a.count;
    int h = live ? col / vd : 0, j = live ? col % vd : 0, i0 = part * R;
    device float* state = states + h * KD * vd;
    float s[R];
    for (int i = 0; i < R; ++i) s[i] = live ? state[(i0 + i) * vd + j] : 0.f;
    for (int r = 0; r < tokens; ++r) {
        device const float* q = x + r * vh * KD + h * KD + i0;
        device const float* k = w + r * vh * KD + h * KD + i0;
        float decay, beta, memory = 0.f;
        delta_gate<GATED>(p5, p6, p7, r * vh + h, h, vh, decay, beta);
        for (int i = 0; i < R; ++i) { s[i] *= decay; memory += s[i] * k[i]; }
        for (int d = 1; d < L; d *= 2) memory += simd_shuffle_xor(memory, ushort(d));
        float delta = (mixed[r * stride + 2 * kh * KD + h * vd + j] - memory) * beta, out = 0.f;
        for (int i = 0; i < R; ++i) { s[i] += k[i] * delta; out += s[i] * q[i]; }
        for (int d = 1; d < L; d *= 2) out += simd_shuffle_xor(out, ushort(d));
        if (live && part == 0) o[r * vh * vd + col] = out;
    }
    if (live)
        for (int i = 0; i < R; ++i) state[(i0 + i) * vd + j] = s[i];
}
#define DELTA_RULE_ARGS device float*, device const float*, device const float*, device const float*, \
    device float*, device const float*, device const float*, device const float*, uint, constant Scalars&
template [[host_name("delta_rule")]] kernel void delta_rule<false>(DELTA_RULE_ARGS);
template [[host_name("delta_rule_gated")]] kernel void delta_rule<true>(DELTA_RULE_ARGS);
#define DELTA_RULE_SPLIT(KD, L) \
    template [[host_name("delta_rule_split_" #KD)]] kernel void delta_rule_split<KD, L, false>(DELTA_RULE_ARGS); \
    template [[host_name("delta_rule_split_gated_" #KD)]] kernel void delta_rule_split<KD, L, true>(DELTA_RULE_ARGS);
DELTA_RULE_SPLIT(128, 4)
DELTA_RULE_SPLIT(64, 4)
