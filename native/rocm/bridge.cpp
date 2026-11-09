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
enum Op { RmsNorm=1, Add=2, SiluProduct=3, ConvSilu=4, DeltaQk=5, DeltaGates=6, DeltaRule=7, AttentionQuery=8, AttentionKv=9, AttentionRead=10 };
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
        int k=a.n[0]; float* hist=a.p[3]+t*(k-1); float sum=0.f;
        for(int i=0;i<k-1;++i) sum+=hist[i]*w[t*k+i];
        sum+=x[t]*w[t*k+k-1];
        for(int i=0;i<k-2;++i) hist[i]=hist[i+1];
        if(k>1) hist[k-2]=x[t];
        o[t]=silu(sum); break;
    }
    case DeltaQk: { // Delta Q/K L2, repeat grouped key heads, Q / sqrt(key_dim)
        int kh=a.n[0], vh=a.n[1], d=a.n[2], h=t/(vh/kh), kd=kh*d;
        float sq=0.f, sk=0.f;
        for(int i=0;i<d;++i) { float q=x[h*d+i], k=x[kd+h*d+i]; sq+=q*q; sk+=k*k; }
        float rq=1.f/sqrtf(sq+1e-6f), rk=1.f/sqrtf(sk+1e-6f);
        for(int i=0;i<d;++i) { o[t*d+i]=(x[h*d+i]*rq)/sqrtf(float(d)); w[t*d+i]=x[kd+h*d+i]*rk; }
        break;
    }
    case DeltaGates: { // Delta gate: p0=g (log decay), p1=a,p2=b,p3=dt,p4=Alog,p5=beta
        o[t]=-expf(a.p[4][t])*softplus(x[t]+a.p[3][t]); a.p[5][t]=sigmoid(w[t]); break;
    }
    case DeltaRule: { // S[head,key,value]; one thread owns the entire value column
        int kd=a.n[0], vd=a.n[1], kh=a.n[2], vh=a.n[3], h=t/vd, j=t%vd;
        float* state=a.p[4]+h*kd*vd;
        float decay=expf(a.p[5][h]), memory=0.f;
        for(int i=0;i<kd;++i) { state[i*vd+j]*=decay; memory+=state[i*vd+j]*w[h*kd+i]; }
        float delta=(a.p[3][2*kh*kd+h*vd+j]-memory)*a.p[6][h], out=0.f;
        for(int i=0;i<kd;++i) { state[i*vd+j]+=w[h*kd+i]*delta; out+=state[i*vd+j]*x[h*kd+i]; }
        o[t]=out; (void)vh; break;
    }
    case AttentionQuery: { // attention: split head-interleaved Q/gate, RMSNorm, partial RoPE
        int d=a.n[0], rot=a.n[1], pos=a.n[2]; float sum=0.f;
        for(int i=0;i<d;++i) { float v=x[t*2*d+i]; sum+=v*v; }
        float r=1.f/sqrtf(sum/float(d)+a.f[0]);
        for(int i=0;i<d;++i) { o[t*d+i]=(x[t*2*d+i]*r)*(1.f+w[i]); a.p[3][t*d+i]=x[t*2*d+d+i]; }
        rope(o+t*d,rot,pos,a.f[1]); break;
    }
    case AttentionKv: { // normalize/rotate K then append K and original V to time-major KV
        int d=a.n[0], rot=a.n[1], pos=a.n[2], kv=a.n[3]; float sum=0.f;
        for(int i=0;i<d;++i) sum+=x[t*d+i]*x[t*d+i];
        float r=1.f/sqrtf(sum/float(d)+a.f[0]);
        float* dest=o+(pos*kv+t)*d;
        for(int i=0;i<d;++i) { dest[i]=(x[t*d+i]*r)*(1.f+w[i]); a.p[4][(pos*kv+t)*d+i]=a.p[3][t*d+i]; }
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
    }
}
#ifndef QR_CPU_TEST
__global__ void kernel(int op, int count, Params a) {
    int t=blockIdx.x*blockDim.x+threadIdx.x; if(t<count) body(op,t,a);
}
#endif
extern "C" {
int qr_abi() { return 1; }
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
    if(op<1 || op>10 || count<1) return fail(c,"invalid kernel",-1);
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
}
