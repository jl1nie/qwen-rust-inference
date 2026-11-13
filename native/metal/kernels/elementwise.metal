// ---- Elementwise and per-row ops ----
// Level 1 only for the elementwise ops (already one thread per element). The
// per-head normalizations at the end (delta_qk, attention_query,
// attention_kv) have a level-2 version with one SIMD group per head.

kernel void add(device float* o [[buffer(0)]], device const float* x [[buffer(1)]],
                device const float* w [[buffer(2)]], ITEM_ARGS) {
    GUARD o[t] = x[t] + w[t];
}
kernel void silu_product(device float* o [[buffer(0)]], device const float* x [[buffer(1)]],
                         device const float* w [[buffer(2)]], ITEM_ARGS) {
    GUARD o[t] = silu_(x[t]) * w[t];
}
// device-to-device copy (embedding row lookup, layer output -> next input)
kernel void copy_floats(device float* o [[buffer(0)]], device const float* x [[buffer(1)]], ITEM_ARGS) {
    GUARD o[t] = x[t];
}
// Widen embedding row n0 (row width n1) of a BF16 matrix. One thread per element.
kernel void embed_bf16(device float* o [[buffer(0)]], device const ushort* m [[buffer(1)]], ITEM_ARGS) {
    GUARD o[t] = bf16_(m[ulong(a.n[0]) * ulong(a.n[1]) + ulong(t)]);
}
// Rows chosen on the device (ADR 0031): output row r (width n1) is row
// picks[2 r] of the BF16 matrix, where picks is the output of argmax_rows.
// One thread per element.
kernel void embed_bf16_picked(device float* o [[buffer(0)]], device const ushort* m [[buffer(1)]],
                              device const float* picks [[buffer(2)]], ITEM_ARGS) {
    GUARD
    ulong h = ulong(a.n[1]), r = ulong(t) / h;
    o[t] = bf16_(m[ulong(picks[2 * r]) * h + ulong(t) % h]);
}

// DeltaNet causal conv1d + SiLU. Channels are independent; one thread per
// channel walks the n1 tokens (rows of stride n2) in time order through its
// history (oldest -> newest), so state evolves exactly as in decode.
// n3 = S > 0 (a verify pass, ADR 0026): after token r < S the history is also
// copied to snapshot r, stored right after the live history in the same
// buffer (snapshot r at offset (r + 1) * channels * (k - 1)).
kernel void conv_silu(device float* o [[buffer(0)]], device const float* x [[buffer(1)]],
                      device const float* w [[buffer(2)]], device float* history [[buffer(3)]], ITEM_ARGS) {
    GUARD
    int k = a.n[0], tokens = max(a.n[1], 1), stride = a.n[2], ti = int(t), snaps = a.n[3];
    ulong total = ulong(a.count) * ulong(k - 1);
    device float* hist = history + ti * (k - 1);
    for (int r = 0; r < tokens; ++r) {
        float xv = x[r * stride + ti], sum = 0.f;
        for (int i = 0; i < k - 1; ++i) sum += hist[i] * w[ti * k + i];
        sum += xv * w[ti * k + k - 1];
        for (int i = 0; i < k - 2; ++i) hist[i] = hist[i + 1];
        if (k > 1) hist[k - 2] = xv;
        o[r * stride + ti] = silu_(sum);
        if (r < snaps)
            for (int i = 0; i < k - 1; ++i) hist[ulong(r + 1) * total + i] = hist[i];
    }
}
// DeltaNet Q/K: L2 normalize, repeat grouped key heads, Q / sqrt(key_dim).
// One thread per (token, value head); n3 = row stride of the convolved input.
kernel void delta_qk(device float* o [[buffer(0)]], device const float* xs [[buffer(1)]],
                     device float* w [[buffer(2)]], ITEM_ARGS) {
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
// DeltaNet gates: p0 = g (log decay), p1 = a, p2 = b, p3 = dt, p4 = A_log, p5 = beta.
kernel void delta_gates(device float* o [[buffer(0)]], device const float* x [[buffer(1)]],
                        device const float* w [[buffer(2)]], device const float* dt [[buffer(3)]],
                        device const float* alog [[buffer(4)]], device float* beta [[buffer(5)]], ITEM_ARGS) {
    GUARD
    int head = a.n[0] > 0 ? int(t) % a.n[0] : int(t); // t = token*heads + head
    o[t] = -exp(alog[head]) * softplus_(x[t] + dt[head]); beta[t] = sigmoid_(w[t]);
}

inline void rope(device float* x, int dim, int pos, float theta) {
    for (int i = 0; i < dim / 2; ++i) {
        float angle = float(pos) / pow(theta, float(2 * i) / float(dim));
        float s = sin(angle), c = cos(angle), a = x[i], b = x[i + dim / 2];
        x[i] = a * c - b * s; x[i + dim / 2] = b * c + a * s;
    }
}
// Attention Q: split head-interleaved Q/gate, RMSNorm, partial RoPE.
// t = token*n3 + head (n3 = heads; 0 means a single token at position n2).
kernel void attention_query(device float* o [[buffer(0)]], device const float* x [[buffer(1)]],
                            device const float* w [[buffer(2)]], device float* gate [[buffer(3)]], ITEM_ARGS) {
    GUARD
    int ti = int(t), d = a.n[0], rot = a.n[1], pos = a.n[2] + (a.n[3] > 0 ? ti / a.n[3] : 0);
    float sum = 0.f;
    for (int i = 0; i < d; ++i) { float v = x[ti * 2 * d + i]; sum += v * v; }
    float r = 1.f / sqrt(sum / float(d) + a.f[0]);
    for (int i = 0; i < d; ++i) { o[ti * d + i] = (x[ti * 2 * d + i] * r) * (1.f + w[i]); gate[ti * d + i] = x[ti * 2 * d + d + i]; }
    rope(o + ti * d, rot, pos, a.f[1]);
}
// Attention K/V: normalize and rotate K, then append K and the original V to
// the time-major KV cache. t = token*kv + kv head; token r goes to position n2 + r.
kernel void attention_kv(device float* keys [[buffer(0)]], device const float* x [[buffer(1)]],
                         device const float* w [[buffer(2)]], device const float* value [[buffer(3)]],
                         device float* values [[buffer(4)]], ITEM_ARGS) {
    GUARD
    int ti = int(t), d = a.n[0], rot = a.n[1], kv = a.n[3], pos = a.n[2] + ti / kv, hh = ti % kv;
    float sum = 0.f;
    for (int i = 0; i < d; ++i) sum += x[ti * d + i] * x[ti * d + i];
    float r = 1.f / sqrt(sum / float(d) + a.f[0]);
    device float* dest = keys + (pos * kv + hh) * d;
    for (int i = 0; i < d; ++i) { dest[i] = (x[ti * d + i] * r) * (1.f + w[i]); values[(pos * kv + hh) * d + i] = value[ti * d + i]; }
    rope(dest, rot, pos, a.f[1]);
}

// Level 2 (ADR 0019's "one wave per head", for Metal): delta_qk,
// attention_query and attention_kv with one SIMD group per (token, head)
// instead of one thread per head (decode has only 16, 8 and 2 heads, so level
// 1 runs on that many threads and loops head_dim with RoPE's pow/sin/cos).
// The sums of squares become SIMD reductions; every output element is still
// the level-1 expression computed by one lane, and the RoPE pair (i, i + rot/2)
// is normalized by the lane that rotates it. Only the summation order of the
// norms differs. Thread t: item t / 32 (whole SIMD groups are dispatched).
#define HEAD_ARGS uint t [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]], \
    constant Scalars& a [[buffer(8)]]
inline float rope_part(float a, float b, int i, int rot, int pos, float theta, bool first) {
    float angle = float(pos) / pow(theta, float(2 * i) / float(rot));
    float s = sin(angle), c = cos(angle);
    return first ? a * c - b * s : b * c + a * s;
}
// One head's normalize (+ (1 + w) scale) and partial RoPE: in[0..d) -> out[0..d).
inline void norm_rope_head(device float* out, device const float* in, device const float* w,
                           float r, int d, int rot, int pos, float theta, uint lane) {
    int half_rot = rot / 2;
    for (int i = int(lane); i < d; i += 32) {
        if (i >= rot) out[i] = (in[i] * r) * (1.f + w[i]);
    }
    for (int i = int(lane); i < half_rot; i += 32) {
        float av = (in[i] * r) * (1.f + w[i]), bv = (in[i + half_rot] * r) * (1.f + w[i + half_rot]);
        out[i] = rope_part(av, bv, i, rot, pos, theta, true);
        out[i + half_rot] = rope_part(av, bv, i, rot, pos, theta, false);
    }
}
kernel void attention_query_simd(device float* o [[buffer(0)]], device const float* x [[buffer(1)]],
                                 device const float* w [[buffer(2)]], device float* gate [[buffer(3)]], HEAD_ARGS) {
    int ti = int(t) / 32;
    if (ti >= a.count) return; // uniform per SIMD group
    int d = a.n[0], rot = a.n[1], pos = a.n[2] + (a.n[3] > 0 ? ti / a.n[3] : 0);
    device const float* q = x + ti * 2 * d;
    float sum = 0.f;
    for (int i = int(lane); i < d; i += 32) sum += q[i] * q[i];
    float r = 1.f / sqrt(simd_sum(sum) / float(d) + a.f[0]);
    for (int i = int(lane); i < d; i += 32) gate[ti * d + i] = q[d + i];
    norm_rope_head(o + ti * d, q, w, r, d, rot, pos, a.f[1], lane);
}
kernel void attention_kv_simd(device float* keys [[buffer(0)]], device const float* x [[buffer(1)]],
                              device const float* w [[buffer(2)]], device const float* value [[buffer(3)]],
                              device float* values [[buffer(4)]], HEAD_ARGS) {
    int ti = int(t) / 32;
    if (ti >= a.count) return; // uniform per SIMD group
    int d = a.n[0], rot = a.n[1], kv = a.n[3], pos = a.n[2] + ti / kv, hh = ti % kv;
    device const float* k = x + ti * d;
    float sum = 0.f;
    for (int i = int(lane); i < d; i += 32) sum += k[i] * k[i];
    float r = 1.f / sqrt(simd_sum(sum) / float(d) + a.f[0]);
    int at = (pos * kv + hh) * d;
    for (int i = int(lane); i < d; i += 32) values[at + i] = value[ti * d + i];
    norm_rope_head(keys + at, k, w, r, d, rot, pos, a.f[1], lane);
}
kernel void delta_qk_simd(device float* o [[buffer(0)]], device const float* xs [[buffer(1)]],
                          device float* w [[buffer(2)]], HEAD_ARGS) {
    int ti = int(t) / 32;
    if (ti >= a.count) return; // uniform per SIMD group
    int kh = a.n[0], vh = a.n[1], d = a.n[2], h = (ti % vh) / (vh / kh), kd = kh * d;
    device const float* x = xs + (ti / vh) * a.n[3];
    float sq = 0.f, sk = 0.f;
    for (int i = int(lane); i < d; i += 32) { float q = x[h * d + i], k = x[kd + h * d + i]; sq += q * q; sk += k * k; }
    float rq = 1.f / sqrt(simd_sum(sq) + 1e-6f), rk = 1.f / sqrt(simd_sum(sk) + 1e-6f);
    for (int i = int(lane); i < d; i += 32) {
        o[ti * d + i] = (x[h * d + i] * rq) / sqrt(float(d));
        w[ti * d + i] = x[kd + h * d + i] * rk;
    }
}
