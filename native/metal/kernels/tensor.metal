// ---- Level 3 (ADR 0016): Metal 4 tensor ops (MetalPerformancePrimitives) ----
// matmul2d runs on the GPU's matrix hardware (the neural accelerators of Apple
// M5). This file is a second library compiled with Metal language 4.0 after
// common.metal; when it is unavailable (macOS < 26, older GPU) bridge.mm keeps
// levels 1-2. relaxed_precision = false everywhere except the opt-in
// --matmul-relaxed GEMMs: inputs are not rounded, only the summation order
// differs from level 2 (FP32 accuracy).
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

// Projections (> 16 rows, >= 8 tokens): Y[batch, rows] = X[batch, cols] . W[rows, cols]^T,
// FP32 X and Y, W BF16 or FP32 (float x bfloat -> float is a supported type).
// TM tokens x TN rows of Y per threadgroup of 4 SIMD groups.
template <typename W, bool RELAX, int TM, int TN>
inline void gemm_tensor(device float* y, device W* m, device float* x, constant Scalars& a, uint2 tg) {
    int rows = a.n[0], cols = a.n[1], batch = a.n[3];
    auto A = tensor<device float, dextents<int32_t, 2>, tensor_inline>(x, dextents<int32_t, 2>(cols, batch));
    auto B = tensor<device W, dextents<int32_t, 2>, tensor_inline>(m, dextents<int32_t, 2>(cols, rows));
    auto C = tensor<device float, dextents<int32_t, 2>, tensor_inline>(y, dextents<int32_t, 2>(rows, batch));
    // m = tokens, n = rows, k = cols (dynamic); W is [rows][cols]: transposed right operand.
    constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent), false, true, RELAX);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mA = A.slice(0, int(tg.y) * TM);
    auto mB = B.slice(0, int(tg.x) * TN);
    auto mC = C.slice(int(tg.x) * TN, int(tg.y) * TM);
    auto cT = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
    op.run(mA, mB, cT); // mode multiply (default): overwrites cT
    cT.store(mC);
}
#define GEMM_TENSOR(NAME, W, RELAX, TM, TN)                                                       \
    kernel void NAME(device float* y [[buffer(0)]], device W* m [[buffer(1)]],                    \
                     device float* x [[buffer(2)]], constant Scalars& a [[buffer(8)]],            \
                     uint2 tg [[threadgroup_position_in_grid]]) {                                   \
        gemm_tensor<W, RELAX, TM, TN>(y, m, x, a, tg);                                             \
    }
GEMM_TENSOR(gemm_tensor_bf16, bfloat, false, 32, 64)
GEMM_TENSOR(gemm_tensor_f32, float, false, 32, 64)
GEMM_TENSOR(gemm_tensor_bf16_relaxed, bfloat, true, 128, 64)
GEMM_TENSOR(gemm_tensor_f32_relaxed, float, true, 128, 64)

// Prefill attention (head_dim D in {64, 128, 256}; attention_prefill.metal has
// the parameters). Flash style: one threadgroup = one query head x ATT_MQ
// query rows of the chunk; positions in blocks of ATT_NB:
//   S = Q K^T            matmul2d into a cooperative tensor, stored to threadgroup memory
//   P = online softmax   per row in threadgroup memory (4 lanes per row)
//   O = diag(c) O + P V  O is a cooperative tensor, rescaled per row via the
//                        element coordinates, then matmul2d multiply_accumulate
// FP32 throughout: only the summation order differs from attention_mma.
// KV buffers are padded to whole ATT_NB blocks (src/device.rs `KV_BLOCK`);
// positions past a row's limit get probability 0. Tile sweep on M5 (ADR 0016):
// (32, 64) was within 7 % of the best and needs the least KV padding.
constant constexpr int ATT_MQ = 32, ATT_NB = 64;
constant constexpr int ATT_LANES = 128 / ATT_MQ, ATT_CPL = ATT_NB / ATT_LANES; // softmax lanes per row
template <int D>
kernel void attention_tensor(device float* o [[buffer(0)]], device float* x [[buffer(1)]],
                             device float* keys [[buffer(2)]], device float* values [[buffer(3)]],
                             device const float* gate [[buffer(4)]], constant Scalars& a [[buffer(8)]],
                             uint2 tg [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sp[ATT_MQ * ATT_NB];
    threadgroup float corr[ATT_MQ], lsum[ATT_MQ];
    int heads = a.n[1], kv = a.n[2], base = a.n[3], tokens = a.n[5];
    int head = int(tg.y), kvh = head / (heads / kv), r0 = int(tg.x) * ATT_MQ;
    int rows = min(ATT_MQ, tokens - r0);
    int span = base + r0 + rows - 1; // positions any row of this block sees
    int qstride = heads * D, kvstride = kv * D;
    using ext2 = dextents<int32_t, 2>;
    using str2 = array<int32_t, 2>;
    auto Q = tensor<device float, ext2, tensor_inline>(x + r0 * qstride + head * D, ext2(D, rows), str2{1, qstride});
    auto S = tensor<threadgroup float, ext2, tensor_inline>(sp, ext2(ATT_NB, ATT_MQ), str2{1, ATT_NB});
    constexpr auto qk = matmul2d_descriptor(ATT_MQ, ATT_NB, D, false, true, false);
    // O accumulates over position blocks: multiply_accumulate (the default
    // mode overwrites the destination).
    constexpr auto pv = matmul2d_descriptor(ATT_MQ, D, ATT_NB, false, false, false,
                                            matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qk, execution_simdgroups<4>> opS;
    matmul2d<pv, execution_simdgroups<4>> opO;
    // K/V blocks of ATT_NB positions (the type of a block, for the accumulator's layout).
    auto block_of = [&](device float* base, int p0) {
        return tensor<device float, ext2, tensor_inline>(base + p0 * kvstride + kvh * D, ext2(D, ATT_NB), str2{1, kvstride});
    };
    auto cO = opO.template get_destination_cooperative_tensor<decltype(S), decltype(block_of(values, 0)), float>();
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cO.get_capacity(); ++i)
        if (cO.is_valid_element(i)) cO[i] = 0.f;
    // Softmax: ATT_LANES lanes per row, ATT_CPL positions each (128 threads).
    int row = int(tid) / ATT_LANES, col = (int(tid) % ATT_LANES) * ATT_CPL;
    bool live = row < rows;
    int limit = base + r0 + row;
    float m = -INFINITY, l = 0.f, scale = sqrt(float(D));
    for (int p0 = 0; p0 < span; p0 += ATT_NB) {
        auto Kb = block_of(keys, p0);
        auto cS = opS.template get_destination_cooperative_tensor<decltype(Q), decltype(Kb), float>();
        opS.run(Q, Kb, cS); // mode multiply: overwrites cS
        cS.store(S);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float e[ATT_CPL], local = -INFINITY;
        for (int j = 0; j < ATT_CPL; ++j) {
            int p = p0 + col + j;
            e[j] = live && p < limit ? sp[row * ATT_NB + col + j] / scale : -INFINITY;
            local = fmax(local, e[j]);
        }
        for (int s = 1; s < ATT_LANES; s *= 2) local = fmax(local, simd_shuffle_xor(local, ushort(s)));
        float next = fmax(m, local);
        // Live rows see position 0 in the first block, so next is finite from then on.
        float c = live ? exp(m - next) : 1.f, sum = 0.f;
        for (int j = 0; j < ATT_CPL; ++j) {
            float w = live ? exp(e[j] - next) : 0.f;
            sp[row * ATT_NB + col + j] = w;
            sum += w;
        }
        for (int s = 1; s < ATT_LANES; s *= 2) sum += simd_shuffle_xor(sum, ushort(s));
        l = l * c + sum;
        m = next;
        if (col == 0) corr[row] = c;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < cO.get_capacity(); ++i)
            if (cO.is_valid_element(i)) cO[i] *= corr[cO.get_multidimensional_index(i)[1]];
        auto Vb = block_of(values, p0);
        opO.run(S, Vb, cO);
        threadgroup_barrier(mem_flags::mem_threadgroup); // P read before the next S store
    }
    if (col == 0) lsum[row] = live ? 1.f / l : 0.f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cO.get_capacity(); ++i) {
        if (!cO.is_valid_element(i)) continue;
        auto at = cO.get_multidimensional_index(i);
        int dd = int(at[0]), rr = int(at[1]);
        if (rr >= rows) continue;
        int idx = (r0 + rr) * qstride + head * D + dd;
        o[idx] = cO[i] * lsum[rr] * sigmoid_(gate[idx]);
    }
}
#define ATTENTION_TENSOR(D)                                                                        \
    template [[host_name("attention_tensor_" #D)]] kernel void attention_tensor<D>(                \
        device float*, device float*, device float*, device float*, device const float*,           \
        constant Scalars&, uint2, uint);
ATTENTION_TENSOR(64)
ATTENTION_TENSOR(128)
ATTENTION_TENSOR(256)
