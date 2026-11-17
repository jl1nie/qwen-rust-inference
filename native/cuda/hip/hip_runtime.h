// Apache-2.0. CUDA build of the ROCm bridge (ADR 0033): the HIP runtime names
// that native/rocm/bridge.cpp and kernels.h use, mapped onto the CUDA runtime.
// tools/build-cuda.sh puts native/cuda first on the include path, so
// `#include <hip/hip_runtime.h>` lands here and the HIP sources compile with
// nvcc unchanged, apart from the few `#ifdef QR_CUDA` blocks they carry.
//
// Wave = warp: gfx1100 runs wave32 and kWave is 32, the CUDA warp size.
#pragma once
#ifndef QR_CUDA
#error "this header is the CUDA stand-in for HIP; build with -DQR_CUDA (tools/build-cuda.sh)"
#endif
#include <cuda_runtime.h>
// Declare their (deprecated) unmasked shuffles before the macros below.
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <tuple>
#include <utility>

using hipError_t = cudaError_t;
using hipStream_t = cudaStream_t;
using hipEvent_t = cudaEvent_t;
using hipDeviceProp_t = cudaDeviceProp;
constexpr hipError_t hipSuccess = cudaSuccess;
constexpr cudaMemcpyKind hipMemcpyHostToDevice = cudaMemcpyHostToDevice;
constexpr cudaMemcpyKind hipMemcpyDeviceToHost = cudaMemcpyDeviceToHost;
constexpr cudaMemcpyKind hipMemcpyDeviceToDevice = cudaMemcpyDeviceToDevice;

inline hipError_t hipGetLastError() { return cudaGetLastError(); }
inline hipError_t hipSetDevice(int d) { return cudaSetDevice(d); }
inline hipError_t hipStreamCreate(hipStream_t* s) { return cudaStreamCreate(s); }
inline hipError_t hipStreamDestroy(hipStream_t s) { return cudaStreamDestroy(s); }
inline hipError_t hipStreamSynchronize(hipStream_t s) { return cudaStreamSynchronize(s); }
inline hipError_t hipEventCreate(hipEvent_t* e) { return cudaEventCreate(e); }
inline hipError_t hipEventDestroy(hipEvent_t e) { return cudaEventDestroy(e); }
inline hipError_t hipEventRecord(hipEvent_t e, hipStream_t s) { return cudaEventRecord(e, s); }
inline hipError_t hipEventElapsedTime(float* ms, hipEvent_t a, hipEvent_t b) { return cudaEventElapsedTime(ms, a, b); }
inline hipError_t hipMalloc(void** p, size_t n) { return cudaMalloc(p, n); }
inline hipError_t hipFree(void* p) { return cudaFree(p); }
inline hipError_t hipMemsetAsync(void* p, int v, size_t n, hipStream_t s) { return cudaMemsetAsync(p, v, n, s); }
inline hipError_t hipMemcpyAsync(void* d, const void* s, size_t n, cudaMemcpyKind k, hipStream_t st) {
    return cudaMemcpyAsync(d, s, n, k, st);
}
inline hipError_t hipGetDeviceProperties(hipDeviceProp_t* p, int d) { return cudaGetDeviceProperties(p, d); }
inline hipError_t hipRuntimeGetVersion(int* v) { return cudaRuntimeGetVersion(v); }
inline hipError_t hipDriverGetVersion(int* v) { return cudaDriverGetVersion(v); }

// hipLaunchKernelGGL(kernel, grid, block, shared, stream, args...). A function
// rather than a `<<<>>>` macro so that parenthesized template kernels such as
// `(attention_tiled<L, true>)` work: the arguments are converted to the
// kernel's parameter types, as a `<<<>>>` launch would, and passed by address.
template <class... P, class... A, size_t... I>
inline void qr_launch(void (*kernel)(P...), dim3 grid, dim3 block, size_t shared, cudaStream_t stream,
                      std::index_sequence<I...>, A&&... args) {
    std::tuple<P...> converted(std::forward<A>(args)...);
    void* slots[] = {static_cast<void*>(&std::get<I>(converted))...};
    (void)cudaLaunchKernel(reinterpret_cast<const void*>(kernel), grid, block, slots, shared, stream);
}
template <class... P, class... A>
inline void hipLaunchKernelGGL(void (*kernel)(P...), dim3 grid, dim3 block, size_t shared, cudaStream_t stream,
                               A&&... args) {
    static_assert(sizeof...(P) == sizeof...(A), "kernel argument count");
    qr_launch(kernel, grid, block, shared, stream, std::index_sequence_for<P...>{}, std::forward<A>(args)...);
}
// A failed cudaLaunchKernel is reported by the bridge's next hipGetLastError
// (launched()), as with HIP.

// HIP's unmasked wave shuffles. Every call site in kernels.h runs with all 32
// lanes of the warp converged (row/head guards are whole-warp), which CUDA's
// full mask requires; the one exception, delta_rule_column's P-lane groups,
// names its group's mask explicitly (qr_shfl_xor_group in kernels.h).
#define __shfl_xor(v, o, w) __shfl_xor_sync(0xffffffffu, (v), (o), (w))
#define __shfl(v, s, w) __shfl_sync(0xffffffffu, (v), (s), (w))
