// Apache-2.0. Prefill GEMMs on the Metal 4 tensor ops (MetalPerformancePrimitives
// matmul2d), which run on the GPU's matrix hardware (the neural accelerators of
// Apple M5). Compiled as a separate library with Metal language 4.0; when that
// is not available the bridge keeps the FP32 gemm_tiled kernels (ADR 0016).
//
// Y[batch, rows] = X[batch, cols] . W[rows, cols]^T, FP32 X and Y, W BF16 or FP32.
// relaxed_precision = false: inputs are not rounded; only the summation order
// differs from gemm_tiled (FP32 accuracy). The relaxed variants (opt-in
// --matmul-relaxed) let the hardware trade accuracy for speed.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct Scalars { int n[8]; float f[4]; int count; }; // == kernels.metal

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
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cT.get_capacity(); ++i)
        if (cT.is_valid_element(i)) cT[i] = 0.f;
    op.run(mA, mB, cT);
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
