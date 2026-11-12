// Apache-2.0. Equations mirror src/{ops,attention,deltanet}.rs.
// QR_CPU_TEST runs the SAME scalar kernel bodies without pretending to be HIP.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#ifdef QR_CPU_TEST
#define DEVICE
#else
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#define DEVICE __device__
#endif
enum Op { RmsNorm=1, Add=2, SiluProduct=3, ConvSilu=4, DeltaQk=5, DeltaGates=6, DeltaRule=7, AttentionQuery=8, AttentionKv=9, AttentionRead=10, EmbedBf16=11 };
struct Params { float* p[8]; int n[8]; float f[4]; };
struct Context {
    char error[256] = {};
    char info[512] = "CPU arithmetic bridge (not HIP)";
#ifndef QR_CPU_TEST
    hipStream_t stream = nullptr;
    rocblas_handle blas = nullptr;
#endif
};
static int fail(Context* c, const char* what, int status) {
    std::snprintf(c->error, sizeof(c->error), "%s status=%d", what, status);
    return status ? status : -1;
}
DEVICE float sigmoid(float x) {
    if (x >= 0.f) return 1.f / (1.f + expf(-x));
    float e = expf(x); return e / (1.f + e);
}
// bf16 is the upper half of an f32: widening is exact (no rounding).
DEVICE float bf16(uint16_t v) {
    union { uint32_t u; float f; } b; b.u=uint32_t(v)<<16; return b.f;
}
DEVICE float silu(float x) { return x * sigmoid(x); }
DEVICE float softplus(float x) { return x > 20.f ? x : log1pf(expf(x)); }
DEVICE void rope(float* x, int dim, int pos, float theta) {
    for (int i=0;i<dim/2;++i) {
        float angle=float(pos)/powf(theta,float(2*i)/float(dim));
        float s=sinf(angle), c=cosf(angle), a=x[i], b=x[i+dim/2];
        x[i]=a*c-b*s; x[i+dim/2]=b*c+a*s;
    }
}
// One invocation handles one element, head, or [head,value] column.
// No shared memory/barriers: CPU arithmetic tests execute precisely this body.
// ABI v2 chunked prefill: ops may cover T consecutive tokens whose rows are
// contiguous. Time-recurrent ops (ConvSilu, DeltaRule) loop the T tokens in
// order inside one invocation; the others index (token, head) by t. T=1 (or
// the new n slots left 0) is exactly the v1 single-token behavior.
DEVICE void body(int op, int t, Params a) {
    float *o=a.p[0], *x=a.p[1], *w=a.p[2];
    switch(op) {
    case RmsNorm: { // grouped RMSNorm; direct scale or Qwen's (1+w)
        int d=a.n[0], off=t*d; float sum=0.f;
        for(int i=0;i<d;++i) sum+=x[off+i]*x[off+i];
        float r=1.f/sqrtf(sum/float(d)+a.f[0]);
        for(int i=0;i<d;++i) o[off+i]=(x[off+i]*r)*(w[i]+a.f[1]);
        break;
    }
    case Add: o[t]=x[t]+w[t]; break;
    case SiluProduct: o[t]=silu(x[t])*w[t]; break;
    case ConvSilu: { // conv channels are independent; history is oldest -> newest
        // n1 tokens (rows of n2 channels) in time order through one channel's history.
        int k=a.n[0], tokens=a.n[1]>0 ? a.n[1] : 1, stride=a.n[2]; float* hist=a.p[3]+t*(k-1);
        for(int r=0;r<tokens;++r) {
            float xv=x[r*stride+t], sum=0.f;
            for(int i=0;i<k-1;++i) sum+=hist[i]*w[t*k+i];
            sum+=xv*w[t*k+k-1];
            for(int i=0;i<k-2;++i) hist[i]=hist[i+1];
            if(k>1) hist[k-2]=xv;
            o[r*stride+t]=silu(sum);
        }
        break;
    }
    case DeltaQk: { // Delta Q/K L2, repeat grouped key heads, Q / sqrt(key_dim)
        // t = token*vh + value head; n3 = row stride of the convolved input.
        int kh=a.n[0], vh=a.n[1], d=a.n[2], h=(t%vh)/(vh/kh), kd=kh*d;
        x+=(t/vh)*a.n[3];
        float sq=0.f, sk=0.f;
        for(int i=0;i<d;++i) { float q=x[h*d+i], k=x[kd+h*d+i]; sq+=q*q; sk+=k*k; }
        float rq=1.f/sqrtf(sq+1e-6f), rk=1.f/sqrtf(sk+1e-6f);
        for(int i=0;i<d;++i) { o[t*d+i]=(x[h*d+i]*rq)/sqrtf(float(d)); w[t*d+i]=x[kd+h*d+i]*rk; }
        break;
    }
    case DeltaGates: { // Delta gate: p0=g (log decay), p1=a,p2=b,p3=dt,p4=Alog,p5=beta
        // t = token*n0 + head (n0 = heads; 0 means a single token)
        int head=a.n[0]>0 ? t%a.n[0] : t;
        o[t]=-expf(a.p[4][head])*softplus(x[t]+a.p[3][head]); a.p[5][t]=sigmoid(w[t]); break;
    }
    case DeltaRule: { // S[head,key,value]; one thread owns the entire value column
        // n4 tokens in time order; n5 = row stride of the convolved (value) input.
        int kd=a.n[0], vd=a.n[1], kh=a.n[2], vh=a.n[3], h=t/vd, j=t%vd;
        int tokens=a.n[4]>0 ? a.n[4] : 1, stride=a.n[5];
        float* state=a.p[4]+h*kd*vd;
        for(int r=0;r<tokens;++r) {
            const float *q=x+r*vh*kd, *k=w+r*vh*kd, *v=a.p[3]+r*stride;
            float decay=expf(a.p[5][r*vh+h]), memory=0.f;
            for(int i=0;i<kd;++i) { state[i*vd+j]*=decay; memory+=state[i*vd+j]*k[h*kd+i]; }
            float delta=(v[2*kh*kd+h*vd+j]-memory)*a.p[6][r*vh+h], out=0.f;
            for(int i=0;i<kd;++i) { state[i*vd+j]+=k[h*kd+i]*delta; out+=state[i*vd+j]*q[h*kd+i]; }
            o[r*vh*vd+t]=out;
        }
        break;
    }
    case AttentionQuery: { // attention: split head-interleaved Q/gate, RMSNorm, partial RoPE
        // t = token*n3 + head (n3 = heads; 0 means a single token at n2)
        int d=a.n[0], rot=a.n[1], pos=a.n[2]+(a.n[3]>0 ? t/a.n[3] : 0); float sum=0.f;
        for(int i=0;i<d;++i) { float v=x[t*2*d+i]; sum+=v*v; }
        float r=1.f/sqrtf(sum/float(d)+a.f[0]);
        for(int i=0;i<d;++i) { o[t*d+i]=(x[t*2*d+i]*r)*(1.f+w[i]); a.p[3][t*d+i]=x[t*2*d+d+i]; }
        rope(o+t*d,rot,pos,a.f[1]); break;
    }
    case AttentionKv: { // normalize/rotate K then append K and original V to time-major KV
        // t = token*kv + kv head; token r is appended at position n2 + r.
        int d=a.n[0], rot=a.n[1], kv=a.n[3], pos=a.n[2]+t/kv, hh=t%kv; float sum=0.f;
        for(int i=0;i<d;++i) sum+=x[t*d+i]*x[t*d+i];
        float r=1.f/sqrtf(sum/float(d)+a.f[0]);
        float* dest=o+(pos*kv+hh)*d;
        for(int i=0;i<d;++i) { dest[i]=(x[t*d+i]*r)*(1.f+w[i]); a.p[4][(pos*kv+hh)*d+i]=a.p[3][t*d+i]; }
        rope(dest,rot,pos,a.f[1]); break;
    }
    case AttentionRead: { // one head: stable causal softmax then V average, sigmoid output gate
        int d=a.n[0], heads=a.n[1], kv=a.n[2], length=a.n[3], capacity=a.n[4];
        int h=t/(heads/kv); float* scores=a.p[5]+t*capacity; float max=-INFINITY;
        for(int p=0;p<length;++p) {
            float sum=0.f; for(int i=0;i<d;++i) sum+=x[t*d+i]*w[(p*kv+h)*d+i];
            scores[p]=sum/sqrtf(float(d)); max=fmaxf(max,scores[p]);
        }
        float den=0.f; for(int p=0;p<length;++p) { scores[p]=expf(scores[p]-max); den+=scores[p]; }
        for(int p=0;p<length;++p) scores[p]/=den;
        for(int i=0;i<d;++i) {
            float sum=0.f; for(int p=0;p<length;++p) sum+=scores[p]*a.p[3][(p*kv+h)*d+i];
            o[t*d+i]=sum*sigmoid(a.p[4][t*d+i]);
        }
        break;
    }
    case EmbedBf16: { // widen embedding row n0 (row width n1) of a uint16 matrix
        const uint16_t* m=reinterpret_cast<const uint16_t*>(x);
        o[t]=bf16(m[size_t(a.n[0])*size_t(a.n[1])+size_t(t)]); break;
    }
    }
}
#ifndef QR_CPU_TEST
__global__ void kernel(int op, int count, Params a) {
    int t=blockIdx.x*blockDim.x+threadIdx.x; if(t<count) body(op,t,a);
}
// Y[b,row] = A[row,:] . X[b,:], A row-major uint16 BF16 [rows, cols];
// one block per (row, token b = blockIdx.y). The row is re-read per token (L2).
__global__ void gemv_bf16(float* y, const uint16_t* a, const float* xs, int rows, int cols) {
    __shared__ float part[256];
    const uint16_t* r=a+size_t(blockIdx.x)*size_t(cols); float sum=0.f;
    const float* x=xs+size_t(blockIdx.y)*size_t(cols);
    for(int j=threadIdx.x;j<cols;j+=blockDim.x) sum+=bf16(r[j])*x[j];
    part[threadIdx.x]=sum; __syncthreads();
    for(int k=blockDim.x/2;k>0;k>>=1) { if(int(threadIdx.x)<k) part[threadIdx.x]+=part[threadIdx.x+k]; __syncthreads(); }
    if(threadIdx.x==0) y[size_t(blockIdx.y)*size_t(rows)+blockIdx.x]=part[0];
}
#endif
extern "C" {
int qr_abi() { return 2; }
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
    hipError_t h=hipSetDevice(0); if(h!=hipSuccess) { delete c; return int(h); }
    h=hipStreamCreate(&c->stream); if(h!=hipSuccess) { delete c; return int(h); }
    rocblas_status r=rocblas_create_handle(&c->blas);
    if(r!=rocblas_status_success) { hipStreamDestroy(c->stream); delete c; return int(r); }
    r=rocblas_set_stream(c->blas,c->stream);
    if(r!=rocblas_status_success) { rocblas_destroy_handle(c->blas); hipStreamDestroy(c->stream); delete c; return int(r); }
#endif
#ifndef QR_CPU_TEST
    hipDeviceProp_t props{}; int runtime=0, driver=0;
    auto ps=hipGetDeviceProperties(&props,0);
    auto rs=hipRuntimeGetVersion(&runtime), ds=hipDriverGetVersion(&driver);
    std::snprintf(c->info,sizeof(c->info),"device0=%s arch=%s hip_runtime=%d(status=%d) driver_api=%d(status=%d) props_status=%d",
        ps==hipSuccess ? props.name : "unknown", ps==hipSuccess ? props.gcnArchName : "unknown",
        runtime,int(rs),driver,int(ds),int(ps));
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
    hipStreamSynchronize(c->stream); rocblas_destroy_handle(c->blas); hipStreamDestroy(c->stream);
#endif
    delete c;
}
int qr_alloc(Context* c, float** p, size_t n) {
#ifdef QR_CPU_TEST
    *p=static_cast<float*>(std::calloc(n,sizeof(float))); if(!*p) return fail(c,"calloc",-1);
#else
    auto s=hipMalloc(reinterpret_cast<void**>(p),n*sizeof(float)); if(s!=hipSuccess) return fail(c,"hipMalloc",int(s));
    s=hipMemsetAsync(*p,0,n*sizeof(float),c->stream);
    if(s!=hipSuccess) { hipFree(*p); *p=nullptr; return fail(c,"hipMemsetAsync",int(s)); }
#endif
    return 0;
}
void qr_free(Context*, float* p) {
#ifdef QR_CPU_TEST
    std::free(p);
#else
    hipFree(p);
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
    hipLaunchKernelGGL(kernel,dim3((count+255)/256),dim3(256),0,c->stream,op,count,*a);
    auto s=hipGetLastError(); if(s!=hipSuccess) return fail(c,"HIP kernel launch",int(s));
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
    // rocBLAS has no BF16-matrix x FP32-vector product; a block reduction is enough here.
    hipLaunchKernelGGL(gemv_bf16,dim3(rows,batch),dim3(256),0,c->stream,y,a,x,rows,cols);
    auto s=hipGetLastError(); if(s!=hipSuccess) return fail(c,"HIP bf16 GEMM launch",int(s));
#endif
    return 0;
}
int qr_gemv_bf16(Context* c, float* y, const float* packed, const float* x, int rows, int cols) {
    return qr_gemm_bf16(c,y,packed,x,rows,cols,1);
}
}
