// Apache-2.0. ROCm (HIP) bridge: the versioned C ABI of src/device.rs (v3).
// QR_CPU_TEST builds the CPU arithmetic test bridge: the same ABI running the
// scalar reference bodies (reference.h) on the host, without pretending to be HIP.
// The HIP build dispatches the heavy ops to kernels.h (ADR 0013).
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <set>
#ifdef QR_CPU_TEST
#define DEVICE
#include "reference.h"
#else
#include <unistd.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#define DEVICE __device__
#include "kernels.h"
#endif
struct Context {
    char error[256] = {};
    char info[512] = "CPU arithmetic bridge (not HIP)";
#ifdef QR_CPU_TEST
    std::set<void*> wrapped; // host memory used in place (qr_wrap): never freed here
#else
    hipStream_t stream = nullptr;
    rocblas_handle blas = nullptr;
#endif
};
static int fail(Context* c, const char* what, int status) {
    std::snprintf(c->error, sizeof(c->error), "%s status=%d", what, status);
    return status ? status : -1;
}
#ifndef QR_CPU_TEST
static int launched(Context* c, const char* what) {
    auto s = hipGetLastError();
    return s == hipSuccess ? 0 : fail(c, what, int(s));
}
static unsigned blocks(long threads, int per_block) { return unsigned((threads + per_block - 1) / per_block); }
static bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; }

// RMSNorm: one block per group, rounded up to whole waves.
static int rms_norm(Context* c, int groups, const Params& a) {
    int threads = a.n[0] >= 256 ? 256 : ((a.n[0] + kWave - 1) / kWave) * kWave;
    hipLaunchKernelGGL(rms_norm_block, dim3(groups), dim3(threads), 0, c->stream, a);
    return launched(c, "rms_norm_block");
}
// DeltaRule: register-resident state, at most 32 key rows per lane.
template <int K, int P>
static void delta_launch(Context* c, int count, const Params& a) {
    hipLaunchKernelGGL((delta_rule_registers<K, P>), dim3(blocks(long(count) * P, 128)), dim3(128), 0, c->stream, a, count);
}
static int delta_rule(Context* c, int count, const Params& a) {
    int kd = a.n[0];
    if (kd <= 8) delta_launch<8, 1>(c, count, a);
    else if (kd <= 32) delta_launch<32, 1>(c, count, a);
    else if (kd <= 64) delta_launch<32, 2>(c, count, a);
    else if (kd <= 128) delta_launch<32, 4>(c, count, a);
    else hipLaunchKernelGGL(reference_kernel, dim3(blocks(count, 128)), dim3(128), 0, c->stream, int(DeltaRule), count, a);
    return launched(c, "delta_rule");
}
// Decode splits: about kDecodeSplits blocks per KV head, each at least
// kMinSpan positions. kMinSpan matches ATTENTION_SPLIT in src/device.rs, which
// sizes the per-head scratch (n6) for ceil(capacity / 32) splits of d + 2 floats.
constexpr int kMinSpan = 32, kDecodeSplits = 64;
static_assert(kMinSpan % kAttnBlock == 0, "split spans must be whole LDS blocks");
template <int L>
static void attention_launch(Context* c, const Params& a, int rows, int span, int splits) {
    int tokens = a.n[5] > 0 ? a.n[5] : 1, group = a.n[1] / a.n[2];
    dim3 grid(unsigned((tokens + rows - 1) / rows * splits), unsigned(a.n[2])), block(unsigned(rows * group * kWave));
    hipLaunchKernelGGL(attention_tiled<L>, grid, block, 0, c->stream, a, rows, span, splits);
}
static int attention(Context* c, int heads_count, const Params& a) {
    int d = a.n[0], heads = a.n[1], kv = a.n[2], tokens = a.n[5] > 0 ? a.n[5] : 1, scratch = a.n[6];
    int group = heads / kv;
    if (d > kAttnMaxDim || group > kAttnWaves) { // outside the LDS tile: reference body, one thread per head
        hipLaunchKernelGGL(reference_kernel, dim3(blocks(heads_count, 256)), dim3(256), 0, c->stream, int(AttentionRead), heads_count, a);
        return launched(c, "attention reference");
    }
    int length = a.n[3] + tokens - 1, span = length, splits = 1;
    int rows = tokens == 1 ? 1 : kAttnWaves / group;
    if (tokens == 1 && scratch > 0) {
        span = (length + kDecodeSplits - 1) / kDecodeSplits;
        span = ((span + kMinSpan - 1) / kMinSpan) * kMinSpan;
        splits = (length + span - 1) / span;
        if (long(splits) * (d + 2) > scratch) { span = length; splits = 1; }
    }
    if (d <= 32) attention_launch<1>(c, a, rows, span, splits);
    else if (d <= 64) attention_launch<2>(c, a, rows, span, splits);
    else if (d <= 128) attention_launch<4>(c, a, rows, span, splits);
    else attention_launch<8>(c, a, rows, span, splits);
    if (int s = launched(c, "attention_tiled")) return s;
    if (splits > 1) {
        hipLaunchKernelGGL(attention_combine, dim3(blocks(long(d) * heads, 256)), dim3(256), 0, c->stream, a, splits);
        return launched(c, "attention_combine");
    }
    return 0;
}
#endif
extern "C" {
int qr_abi() { return 3; }
int qr_kind() {
#ifdef QR_CPU_TEST
    return 0;
#else
    return 1;
#endif
}
const char* qr_error(Context* c) { return c->error; }
const char* qr_info(Context* c) { return c->info; }
int qr_create(Context** out) {
    *out=nullptr; Context* c=new(std::nothrow) Context; if(!c) return -1;
#ifndef QR_CPU_TEST
    // WSL2 exposes the GPU only through /dev/dxg (librocdxg); the HSA runtime
    // looks there only when asked. Set it before the first HIP call unless the
    // user already chose a value.
    bool dxg = access("/dev/dxg", F_OK) == 0;
    if (dxg) setenv("HSA_ENABLE_DXG_DETECTION", "1", 0);
    hipError_t h=hipSetDevice(0); if(h!=hipSuccess) { delete c; return int(h); }
    h=hipStreamCreate(&c->stream); if(h!=hipSuccess) { delete c; return int(h); }
    rocblas_status r=rocblas_create_handle(&c->blas);
    if(r!=rocblas_status_success) { (void)hipStreamDestroy(c->stream); delete c; return int(r); }
    r=rocblas_set_stream(c->blas,c->stream);
    if(r!=rocblas_status_success) { rocblas_destroy_handle(c->blas); (void)hipStreamDestroy(c->stream); delete c; return int(r); }
    hipDeviceProp_t props{}; int runtime=0, driver=0;
    auto ps=hipGetDeviceProperties(&props,0);
    auto rs=hipRuntimeGetVersion(&runtime), ds=hipDriverGetVersion(&driver);
    std::snprintf(c->info,sizeof(c->info),"device0=%s arch=%s cus=%d hip_runtime=%d(status=%d) driver_api=%d(status=%d) props_status=%d wsl_dxg=%d kernels=adr0013",
        ps==hipSuccess ? props.name : "unknown", ps==hipSuccess ? props.gcnArchName : "unknown",
        ps==hipSuccess ? props.multiProcessorCount : 0, runtime,int(rs),driver,int(ds),int(ps),int(dxg));
#endif
    *out=c; return 0;
}
int qr_sync(Context* c) {
#ifndef QR_CPU_TEST
    auto s=hipStreamSynchronize(c->stream); if(s!=hipSuccess) return fail(c,"hipStreamSynchronize",int(s));
#else
    (void)c;
#endif
    return 0;
}
void qr_destroy(Context* c) {
#ifndef QR_CPU_TEST
    (void)hipStreamSynchronize(c->stream); rocblas_destroy_handle(c->blas); (void)hipStreamDestroy(c->stream);
#endif
    delete c;
}
int qr_alloc(Context* c, float** p, size_t n) {
#ifdef QR_CPU_TEST
    *p=static_cast<float*>(std::calloc(n,sizeof(float))); if(!*p) return fail(c,"calloc",-1);
#else
    auto s=hipMalloc(reinterpret_cast<void**>(p),n*sizeof(float)); if(s!=hipSuccess) return fail(c,"hipMalloc",int(s));
    s=hipMemsetAsync(*p,0,n*sizeof(float),c->stream);
    if(s!=hipSuccess) { (void)hipFree(*p); *p=nullptr; return fail(c,"hipMemsetAsync",int(s)); }
#endif
    return 0;
}
#ifdef QR_CPU_TEST
// Optional symbol, CPU test bridge only (HIP memory is separate, so the HIP
// build does not export it and Rust falls back to copying): mirrors Metal's
// in-place use of host memory so the shared-weight path is tested in CI.
int qr_wrap(Context* c, float** p, void* host, size_t bytes) {
    if (!host || bytes == 0) return fail(c, "wrap size", -1);
    c->wrapped.insert(host);
    *p = static_cast<float*>(host);
    return 0;
}
#endif
void qr_free(Context* c, float* p) {
#ifdef QR_CPU_TEST
    if (c->wrapped.erase(p)) return;
    std::free(p);
#else
    (void)c;
    (void)hipFree(p);
#endif
}
int qr_write(Context* c, float* p, const float* host, size_t n) {
#ifdef QR_CPU_TEST
    std::memcpy(p,host,n*sizeof(float)); (void)c;
#else
    auto s=hipMemcpyAsync(p,host,n*sizeof(float),hipMemcpyHostToDevice,c->stream);
    if(s!=hipSuccess) return fail(c,"H2D",int(s));
    // Upload's borrowed Rust slice may disappear immediately after this call.
    return qr_sync(c);
#endif
    return 0;
}
int qr_read(Context* c, float* host, const float* p, size_t n) {
#ifdef QR_CPU_TEST
    std::memcpy(host,p,n*sizeof(float)); (void)c;
#else
    auto s=hipMemcpyAsync(host,p,n*sizeof(float),hipMemcpyDeviceToHost,c->stream);
    if(s!=hipSuccess) return fail(c,"D2H",int(s));
    return qr_sync(c);
#endif
    return 0;
}
int qr_copy(Context* c, float* dest, const float* src, size_t n) {
#ifdef QR_CPU_TEST
    std::memcpy(dest,src,n*sizeof(float)); (void)c;
#else
    auto s=hipMemcpyAsync(dest,src,n*sizeof(float),hipMemcpyDeviceToDevice,c->stream);
    if(s!=hipSuccess) return fail(c,"D2D",int(s));
#endif
    return 0;
}
int qr_run(Context* c, int op, int count, const Params* a) {
    if(op<1 || op>11 || count<1) return fail(c,"invalid kernel",-1);
#ifdef QR_CPU_TEST
    for(int t=0;t<count;++t) body(op,t,*a);
#else
    switch(op) {
    case RmsNorm: return rms_norm(c,count,*a);
    case DeltaRule: return delta_rule(c,count,*a);
    case AttentionRead: return attention(c,count,*a);
    default:
        hipLaunchKernelGGL(reference_kernel,dim3(blocks(count,256)),dim3(256),0,c->stream,op,count,*a);
        return launched(c,"HIP kernel launch");
    }
#endif
    return 0;
}
int qr_gemv(Context* c, float* y, const float* a, const float* x, int rows, int cols) {
#ifdef QR_CPU_TEST
    for(int i=0;i<rows;++i) { float sum=0.f; for(int j=0;j<cols;++j) sum+=a[i*cols+j]*x[j]; y[i]=sum; }
    (void)c;
#else
    // Rust [out,in] row-major == rocBLAS [in,out] column-major.
    // y = A^T x with lda=in. No transpose/upload copy of the actual weights.
    const float alpha=1.f, beta=0.f;
    auto s=rocblas_sgemv(c->blas,rocblas_operation_transpose,cols,rows,&alpha,a,cols,x,1,&beta,y,1);
    if(s!=rocblas_status_success) return fail(c,"rocblas_sgemv",int(s));
#endif
    return 0;
}
// ABI v2: Y[batch,rows] = X[batch,cols] . A^T for FP32 A (batched prefill).
int qr_gemm(Context* c, float* y, const float* a, const float* x, int rows, int cols, int batch) {
    if(rows<1 || cols<1 || batch<1) return fail(c,"GEMM shape",-1);
#ifdef QR_CPU_TEST
    // Same per-token loop as qr_gemv, so batching cannot change CPU-bridge results.
    for(int b=0;b<batch;++b) for(int i=0;i<rows;++i) {
        float sum=0.f; for(int j=0;j<cols;++j) sum+=a[size_t(i)*size_t(cols)+size_t(j)]*x[size_t(b)*size_t(cols)+size_t(j)];
        y[size_t(b)*size_t(rows)+size_t(i)]=sum;
    }
    (void)c;
#else
    // Row-major Y[batch,rows] == column-major [rows,batch] = A_cm^T [rows,cols] . X_cm [cols,batch].
    const float alpha=1.f, beta=0.f;
    auto s=rocblas_sgemm(c->blas,rocblas_operation_transpose,rocblas_operation_none,rows,batch,cols,
        &alpha,a,cols,x,cols,&beta,y,rows);
    if(s!=rocblas_status_success) return fail(c,"rocblas_sgemm",int(s));
#endif
    return 0;
}
// ABI v2: weights stored as uint16 BF16 (allocated/written as packed 32-bit words).
int qr_gemm_bf16(Context* c, float* y, const float* packed, const float* x, int rows, int cols, int batch) {
    const uint16_t* a=reinterpret_cast<const uint16_t*>(packed);
    if(rows<1 || cols<1 || batch<1) return fail(c,"bf16 GEMM shape",-1);
#ifdef QR_CPU_TEST
    for(int b=0;b<batch;++b) for(int i=0;i<rows;++i) {
        float sum=0.f; for(int j=0;j<cols;++j) sum+=bf16(a[size_t(i)*size_t(cols)+size_t(j)])*x[size_t(b)*size_t(cols)+size_t(j)];
        y[size_t(b)*size_t(rows)+size_t(i)]=sum;
    }
    (void)c;
#else
    // rocBLAS has no BF16-matrix x FP32-activation product, so these are our own.
    if(cols%8!=0 || !aligned16(a) || !aligned16(x)) {
        hipLaunchKernelGGL(gemv_bf16_block,dim3(rows,batch),dim3(256),0,c->stream,y,a,x,rows,cols);
        return launched(c,"gemv_bf16_block");
    }
    if(batch>=kGemmMinTokens) {
        dim3 grid(blocks(rows,kTileRows),blocks(batch,kTileTokens));
        hipLaunchKernelGGL(gemm_bf16_tiled,grid,dim3(256),0,c->stream,y,a,x,rows,cols,batch);
        return launched(c,"gemm_bf16_tiled");
    }
    dim3 grid(blocks(rows,kGemvRowsPerBlock),blocks(batch,kGemvTokens));
    hipLaunchKernelGGL(gemv_bf16_wave,grid,dim3(256),0,c->stream,y,a,x,rows,cols,batch);
    return launched(c,"gemv_bf16_wave");
#endif
    return 0;
}
int qr_gemv_bf16(Context* c, float* y, const float* packed, const float* x, int rows, int cols) {
    return qr_gemm_bf16(c,y,packed,x,rows,cols,1);
}
}
