// ---- Causal attention read for a prefill chunk (op 10 with n5 = T > 1) ----
// Query rows are contiguous [T, heads, head_dim]; row r of the chunk sees
// positions < n3 + r (n3 = position of row 0 plus one). No score rows are
// stored: every level streams K/V once with an online softmax (running max m
// and denominator l, rescaling the accumulator when m grows). Summation order
// differs from the decode read (ADR 0004 tolerances). n0 head_dim (<= 512),
// n1 heads, n2 kv heads, n3 base, n5 T.

constant constexpr int FLASH_LANES = 16; // dims per lane: head_dim <= 512

// Level 1: one SIMD group per (row, head) walks the positions one by one;
// lane owns dims lane, lane + 32, ... (coalesced). A threadgroup holds `rows`
// rows x the G heads of one KV head, so a position is fetched once per group.
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
    int length = a.n[3] + r;
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

// Level 2 (ADR 0007, 0015): 8x8 simdgroup matrices, head_dim D in {64,128,256}.
// One threadgroup = 8 query rows x the G heads of one KV head; each SIMD group
// (one head) runs independently, without threadgroup staging or barriers.
// Positions are taken 32 at a time (4 score tiles): S = Q K^T, then 4 lanes
// per row apply the causal mask and the online softmax of their row, then
// O = diag(correction) O + P V. K/V 8x8 tiles are read straight from device
// memory (the G SIMD groups read the same tiles through the cache); Q is
// re-read per block (keeping it in registers spills at D = 256). KV buffers
// are padded to whole blocks (src/device.rs); positions past a row's limit get
// probability 0. The query buffer is padded to 8 rows; rows >= T are not written.
constant constexpr int MMA_MAX_GROUP = 8;
constant constexpr int MMA_BLOCK = 32;
template <int D>
kernel void attention_mma(device float* o [[buffer(0)]], device const float* x [[buffer(1)]],
                          device const float* keys [[buffer(2)]], device const float* values [[buffer(3)]],
                          device const float* gate [[buffer(4)]], constant Scalars& a [[buffer(8)]],
                          uint2 block [[threadgroup_position_in_grid]],
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
}
#define ATTENTION_MMA(D)                                                                              \
    template [[host_name("attention_mma_" #D)]] kernel void attention_mma<D>(                         \
        device float*, device const float*, device const float*, device const float*,                \
        device const float*, constant Scalars&, uint2, uint, uint);
ATTENTION_MMA(64)
ATTENTION_MMA(128)
ATTENTION_MMA(256)
