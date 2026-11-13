// ---- Attention read for one decode token (op 10 with T = 1) ----
// For each query head h of KV head h / G (G = heads / kv):
//   o_h = sum_p softmax_p(q_h . k_p / sqrt(d)) v_p, times sigmoid(gate_h)
// n0 head_dim, n1 heads, n2 kv heads, n3 positions L, n4 KV capacity,
// n6 per-head scratch floats; buffer 5 is the per-head scratch (score rows
// for level 1, split partials for level 2).

// Level 1: three passes, the CPU algorithm spread over threads.
//   1. scores[h, p] = q_h . k_p / sqrt(d)   one thread per (position, head)
//   2. softmax over p per head              one threadgroup per head
//   3. o[h, i] = sum_p prob * v_p[i] * gate  one thread per (dim, head)
// Pass 1 keeps the serial dot product of the CPU loop; passes 2/3 reorder sums.
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

// Level 2 (ADR 0008, 0017): flash decoding. The L positions are cut into
// splits of n7 positions (chosen by bridge.mm); each split produces, per head,
// a partial (running max m, denominator l, unnormalized sum acc[d]) in the
// scratch row (split s at s * (d + 2)), and attention_combine merges them.
//
// attention_split_grp<D, G>: one SIMD group per (split, KV head) serves all G
// query heads of that KV head, so each K/V element loaded is used G times.
// Positions are taken 32 at a time, one per lane: the lane computes its
// position's G dot products (float4 K loads), the softmax max/sum are taken
// once per 32 positions (simd_max / simd_sum), and for P V lanes own
// dimensions (lane + 32e) with the weights broadcast by simd_shuffle.
template <int D, int G>
kernel void attention_split_grp(device const float* x [[buffer(1)]], device const float* keys [[buffer(2)]],
                                device const float* values [[buffer(3)]], device float* scratch [[buffer(5)]],
                                constant Scalars& a [[buffer(8)]],
                                uint2 block [[threadgroup_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]]) {
    constexpr int E = D / 32;
    int kv = a.n[2], length = a.n[3], row = a.n[6], span = a.n[7];
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
        // Online softmax per head over these 32 positions (every split holds
        // >= 1 position, so m is finite after its first block).
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
}
#define ATTENTION_SPLIT_GRP(D, G) \
    template [[host_name("attention_split_grp_" #D)]] kernel void attention_split_grp<D, G>( \
        device const float*, device const float*, device const float*, device float*, constant Scalars&, uint2, uint);
ATTENTION_SPLIT_GRP(256, 4)
ATTENTION_SPLIT_GRP(128, 4)
ATTENTION_SPLIT_GRP(64, 4)

// o[h, i] = sum_s exp(m_s - M) acc_s[i] / sum_s exp(m_s - M) l_s, then the
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
