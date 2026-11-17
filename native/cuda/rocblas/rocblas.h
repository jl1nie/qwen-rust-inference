// Apache-2.0. CUDA build of the ROCm bridge (ADR 0033): the rocBLAS calls of
// native/rocm/bridge.cpp mapped onto cuBLAS. Only the FP32 weight path
// (--f32-weights) uses them; BF16 projections are the bridge's own kernels.
// Both libraries take column-major operands in the same argument order.
#pragma once
#ifndef QR_CUDA
#error "this header is the CUDA stand-in for rocBLAS; build with -DQR_CUDA (tools/build-cuda.sh)"
#endif
#include <cublas_v2.h>

using rocblas_handle = cublasHandle_t;
using rocblas_status = cublasStatus_t;
using rocblas_operation = cublasOperation_t;
constexpr rocblas_status rocblas_status_success = CUBLAS_STATUS_SUCCESS;
constexpr rocblas_operation rocblas_operation_none = CUBLAS_OP_N;
constexpr rocblas_operation rocblas_operation_transpose = CUBLAS_OP_T;

inline rocblas_status rocblas_create_handle(rocblas_handle* h) {
    rocblas_status s = cublasCreate(h);
    // FP32 stays FP32: no TF32 tensor-core shortcut for the reference path.
    if (s == CUBLAS_STATUS_SUCCESS) s = cublasSetMathMode(*h, CUBLAS_PEDANTIC_MATH);
    return s;
}
inline rocblas_status rocblas_destroy_handle(rocblas_handle h) { return cublasDestroy(h); }
inline rocblas_status rocblas_set_stream(rocblas_handle h, cudaStream_t s) { return cublasSetStream(h, s); }
inline rocblas_status rocblas_sgemv(rocblas_handle h, rocblas_operation t, int m, int n, const float* alpha,
                                    const float* a, int lda, const float* x, int incx, const float* beta, float* y,
                                    int incy) {
    return cublasSgemv(h, t, m, n, alpha, a, lda, x, incx, beta, y, incy);
}
inline rocblas_status rocblas_sgemm(rocblas_handle h, rocblas_operation ta, rocblas_operation tb, int m, int n, int k,
                                    const float* alpha, const float* a, int lda, const float* b, int ldb,
                                    const float* beta, float* c, int ldc) {
    return cublasSgemm(h, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
