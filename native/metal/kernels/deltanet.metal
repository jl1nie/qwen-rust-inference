// ---- DeltaNet rule (src/deltanet.rs): per head, state S[key, value] ----
//   S <- S * exp(g);  memory = k^T S;  delta = (v - memory) * beta
//   S <- S + k delta^T;  out = q^T S
// Value columns evolve independently, so each column is one unit of work.
// n0 key_dim, n1 value_dim, n2 key heads, n3 value heads, n4 tokens (in time
// order), n5 row stride of the convolved (value) input; count = value_heads *
// value_dim columns.

// Level 1: one thread owns a whole value column and walks the tokens, reading
// and writing the column in device memory, exactly like the CPU loop.
kernel void delta_rule(device float* o [[buffer(0)]], device const float* x [[buffer(1)]],
                       device const float* w [[buffer(2)]], device const float* mixed [[buffer(3)]],
                       device float* states [[buffer(4)]], device const float* g [[buffer(5)]],
                       device const float* beta [[buffer(6)]], ITEM_ARGS) {
    GUARD
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

// Level 2 (ADR 0016): each column split over L lanes of one SIMD group, KD/L
// rows each held in registers for the whole chunk. The two dot products per
// token become L partial sums combined with simd shuffles, so the per-token
// dependency chain is KD/L long instead of KD. Only the summation order differs.
// Thread t: column t / L, part t % L (whole SIMD groups are dispatched).
template <int KD, int L>
kernel void delta_rule_split(device float* o [[buffer(0)]], device const float* x [[buffer(1)]],
                             device const float* w [[buffer(2)]], device const float* mixed [[buffer(3)]],
                             device float* states [[buffer(4)]], device const float* g [[buffer(5)]],
                             device const float* beta [[buffer(6)]], ITEM_ARGS) {
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
    template [[host_name("delta_rule_split_" #KD)]] kernel void delta_rule_split<KD, L>( \
        device float*, device const float*, device const float*, device const float*, \
        device float*, device const float*, device const float*, uint, constant Scalars&);
DELTA_RULE_SPLIT(128, 4)
DELTA_RULE_SPLIT(64, 4)
