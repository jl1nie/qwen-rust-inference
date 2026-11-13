// Apache-2.0. Scalar reference bodies of the ROCm bridge ops (ABI v3).
// Equations mirror src/{ops,attention,deltanet}.rs.
//
// One invocation handles one element, head, or [head,value] column, with no
// shared memory or barriers. The CPU test bridge (QR_CPU_TEST) runs exactly
// these bodies. The HIP build uses them only for cheap elementwise ops and as
// fallbacks for shapes the parallel kernels in kernels.h do not cover; the
// heavy ops (RMSNorm, DeltaRule, AttentionRead, projections) have their own
// GPU decomposition there (ADR 0013).
//
// Chunked prefill (ABI v2/v3): ops may cover T consecutive tokens whose rows
// are contiguous. Time-recurrent ops (ConvSilu, DeltaRule) loop the T tokens
// in order inside one invocation; the others index (token, head) by t. T=1 (or
// the n slots left 0) is the single-token behavior.
#pragma once
#include <cmath>
#include <cstdint>
#ifndef DEVICE
#define DEVICE
#endif
enum Op { RmsNorm=1, Add=2, SiluProduct=3, ConvSilu=4, DeltaQk=5, DeltaGates=6, DeltaRule=7, AttentionQuery=8, AttentionKv=9, AttentionRead=10, EmbedBf16=11,
          AddRmsNorm=12, RmsNormGate=13, DeltaRuleGated=14, AttentionQueryKv=15 };
// Ops 12-13 are fusions (#19), advertised by the optional qr_max_op symbol:
// each computes exactly what the listed pair of ops computes in sequence.
constexpr int kMaxOp = 15;
struct Params { float* p[8]; int n[8]; float f[4]; };

DEVICE inline float sigmoid(float x) {
    if (x >= 0.f) return 1.f / (1.f + expf(-x));
    float e = expf(x); return e / (1.f + e);
}
// bf16 is the upper half of an f32: widening is exact (no rounding).
DEVICE inline float bf16(uint16_t v) {
    union { uint32_t u; float f; } b; b.u=uint32_t(v)<<16; return b.f;
}
DEVICE inline float silu(float x) { return x * sigmoid(x); }
DEVICE inline float softplus(float x) { return x > 20.f ? x : log1pf(expf(x)); }
DEVICE inline void rope(float* x, int dim, int pos, float theta) {
    for (int i=0;i<dim/2;++i) {
        float angle=float(pos)/powf(theta,float(2*i)/float(dim));
        float s=sinf(angle), c=cosf(angle), a=x[i], b=x[i+dim/2];
        x[i]=a*c-b*s; x[i+dim/2]=b*c+a*s;
    }
}
// DeltaRule's per-token gates of value head h, row i = token * vh + h: the
// log decay and beta. DeltaRule reads them from p5/p6 (written by
// DeltaGates); DeltaRuleGated computes them here from the DeltaGates inputs
// (p5 = a, p6 = b, p7 = [dt | A_log], vh each) with the same expressions.
DEVICE inline void delta_gate(const Params& a, bool gated, int i, int h, int vh, float& log_decay, float& beta) {
    if (gated) {
        log_decay=-expf(a.p[7][vh+h])*softplus(a.p[5][i]+a.p[7][h]);
        beta=sigmoid(a.p[6][i]);
    } else {
        log_decay=a.p[5][i];
        beta=a.p[6][i];
    }
}
// AttentionQueryKv = AttentionQuery for items [0, heads*T) then AttentionKv
// for [heads*T, (heads+kv)*T): p0 query, p1 mixed, p2 [q_norm | k_norm],
// p3 gate, p4 keys, p5 key, p6 value, p7 values; n0 d, n1 rotary, n2
// position, n3 heads, n4 kv heads, n5 T. Each part gets the slots of its op.
DEVICE inline bool query_kv_part(const Params& a, int t, Params& part, int& item) {
    int queries=a.n[3]*a.n[5];
    part=Params{}; part.n[0]=a.n[0]; part.n[1]=a.n[1]; part.n[2]=a.n[2]; part.f[0]=a.f[0]; part.f[1]=a.f[1];
    if(t<queries) {
        part.p[0]=a.p[0]; part.p[1]=a.p[1]; part.p[2]=a.p[2]; part.p[3]=a.p[3]; part.n[3]=a.n[3];
        item=t; return true;
    }
    part.p[0]=a.p[4]; part.p[1]=a.p[5]; part.p[2]=a.p[2]+a.n[0]; part.p[3]=a.p[6]; part.p[4]=a.p[7]; part.n[3]=a.n[4];
    item=t-queries; return false;
}
// attention: split head-interleaved Q/gate, RMSNorm, partial RoPE
DEVICE inline void attention_query_body(int t, const Params& a) {
    float *o=a.p[0], *x=a.p[1], *w=a.p[2];
    // t = token*n3 + head (n3 = heads; 0 means a single token at n2)
    int d=a.n[0], rot=a.n[1], pos=a.n[2]+(a.n[3]>0 ? t/a.n[3] : 0); float sum=0.f;
    for(int i=0;i<d;++i) { float v=x[t*2*d+i]; sum+=v*v; }
    float r=1.f/sqrtf(sum/float(d)+a.f[0]);
    for(int i=0;i<d;++i) { o[t*d+i]=(x[t*2*d+i]*r)*(1.f+w[i]); a.p[3][t*d+i]=x[t*2*d+d+i]; }
    rope(o+t*d,rot,pos,a.f[1]);
}
// normalize/rotate K then append K and original V to time-major KV
DEVICE inline void attention_kv_body(int t, const Params& a) {
    float *o=a.p[0], *x=a.p[1], *w=a.p[2];
    // t = token*kv + kv head; token r is appended at position n2 + r.
    int d=a.n[0], rot=a.n[1], kv=a.n[3], pos=a.n[2]+t/kv, hh=t%kv; float sum=0.f;
    for(int i=0;i<d;++i) sum+=x[t*d+i]*x[t*d+i];
    float r=1.f/sqrtf(sum/float(d)+a.f[0]);
    float* dest=o+(pos*kv+hh)*d;
    for(int i=0;i<d;++i) { dest[i]=(x[t*d+i]*r)*(1.f+w[i]); a.p[4][(pos*kv+hh)*d+i]=a.p[3][t*d+i]; }
    rope(dest,rot,pos,a.f[1]);
}
// conv channels are independent; history is oldest -> newest
// n1 tokens (rows of n2 channels) in time order through one channel's history.
// SNAP (n3 = S > 0, a verify pass with rollback, ADR 0026): after token r < S
// the history is also copied to snapshot r, right after the live history in
// the same buffer (snapshot r at offset (r + 1) * n2 * (k - 1)). Without SNAP
// the code is unchanged (the fused decode kernels use it).
template <bool SNAP = false>
DEVICE inline void conv_silu_channel(int t, const Params& a) {
    float *o=a.p[0], *x=a.p[1], *w=a.p[2];
    int k=a.n[0], tokens=a.n[1]>0 ? a.n[1] : 1, stride=a.n[2]; float* hist=a.p[3]+t*(k-1);
    for(int r=0;r<tokens;++r) {
        float xv=x[r*stride+t], sum=0.f;
        for(int i=0;i<k-1;++i) sum+=hist[i]*w[t*k+i];
        sum+=xv*w[t*k+k-1];
        for(int i=0;i<k-2;++i) hist[i]=hist[i+1];
        if(k>1) hist[k-2]=xv;
        o[r*stride+t]=silu(sum);
        if(SNAP && r<a.n[3])
            for(int i=0;i<k-1;++i) hist[size_t(r+1)*size_t(stride)*size_t(k-1)+i]=hist[i];
    }
}
DEVICE inline void body(int op, int t, Params a) {
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
    case ConvSilu: // the HIP bridge runs snapshot passes in their own kernel (conv_silu_snapshots)
#ifdef QR_CPU_TEST
        if(a.n[3]>0) { conv_silu_channel<true>(t,a); break; }
#endif
        conv_silu_channel(t,a); break;
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
    case DeltaRule: case DeltaRuleGated: { // S[head,key,value]; one thread owns the entire value column
        // n4 tokens in time order; n5 = row stride of the convolved (value) input.
        int kd=a.n[0], vd=a.n[1], kh=a.n[2], vh=a.n[3], h=t/vd, j=t%vd;
        int tokens=a.n[4]>0 ? a.n[4] : 1, stride=a.n[5];
        float* state=a.p[4]+h*kd*vd;
        for(int r=0;r<tokens;++r) {
            const float *q=x+r*vh*kd, *k=w+r*vh*kd, *v=a.p[3]+r*stride;
            float log_decay, beta; delta_gate(a,op==DeltaRuleGated,r*vh+h,h,vh,log_decay,beta);
            float decay=expf(log_decay), memory=0.f;
            for(int i=0;i<kd;++i) { state[i*vd+j]*=decay; memory+=state[i*vd+j]*k[h*kd+i]; }
            float delta=(v[2*kh*kd+h*vd+j]-memory)*beta, out=0.f;
            for(int i=0;i<kd;++i) { state[i*vd+j]+=k[h*kd+i]*delta; out+=state[i*vd+j]*q[h*kd+i]; }
            o[r*vh*vd+t]=out;
#ifdef QR_CPU_TEST
            // n6 = S > 0 (ADR 0026): snapshot r after token r < S, right after the
            // live state (offset (r + 1) * vh * kd * vd). On HIP, delta_rule_column.
            if(r<a.n[6])
                for(int i=0;i<kd;++i) state[size_t(r+1)*size_t(vh)*size_t(kd)*size_t(vd)+i*vd+j]=state[i*vd+j];
#endif
        }
        break;
    }
    case AttentionQuery: attention_query_body(t,a); break;
    case AttentionKv: attention_kv_body(t,a); break;
    case AttentionRead: { // one head: stable causal softmax then V average, sigmoid output gate
        // ABI v3: n5 = T query rows (0 = one). Row r sees n3 + r positions; the
        // head's score row is reused token after token, so every row is computed
        // exactly as a single-token read would be.
        int d=a.n[0], heads=a.n[1], kv=a.n[2], capacity=a.n[4], tokens=a.n[5]>0 ? a.n[5] : 1;
        int h=t/(heads/kv); float* scores=a.p[5]+t*capacity;
        for(int r=0;r<tokens;++r) {
            int length=a.n[3]+r, row=(r*heads+t)*d; float max=-INFINITY;
            for(int p=0;p<length;++p) {
                float sum=0.f; for(int i=0;i<d;++i) sum+=x[row+i]*w[(p*kv+h)*d+i];
                scores[p]=sum/sqrtf(float(d)); max=fmaxf(max,scores[p]);
            }
            float den=0.f; for(int p=0;p<length;++p) { scores[p]=expf(scores[p]-max); den+=scores[p]; }
            for(int p=0;p<length;++p) scores[p]/=den;
            for(int i=0;i<d;++i) {
                float sum=0.f; for(int p=0;p<length;++p) sum+=scores[p]*a.p[3][(p*kv+h)*d+i];
                o[row+i]=sum*sigmoid(a.p[4][row+i]);
            }
        }
        break;
    }
    case AddRmsNorm: { // Add then RmsNorm of the sum: p0 = p1 + p2, p3 = norm(p0) with p4, f1
        int d=a.n[0], off=t*d; float* sum_out=a.p[0]; const float *u=a.p[1], *v=a.p[2]; float sum=0.f;
        for(int i=0;i<d;++i) { sum_out[off+i]=u[off+i]+v[off+i]; sum+=sum_out[off+i]*sum_out[off+i]; }
        float r=1.f/sqrtf(sum/float(d)+a.f[0]);
        for(int i=0;i<d;++i) a.p[3][off+i]=(sum_out[off+i]*r)*(a.p[4][i]+a.f[1]);
        break;
    }
    case RmsNormGate: { // RmsNorm of p1 (p2, f1) then SiluProduct with z = p3: p0 = silu(z) * norm
        int d=a.n[0], off=t*d; float sum=0.f;
        for(int i=0;i<d;++i) sum+=x[off+i]*x[off+i];
        float r=1.f/sqrtf(sum/float(d)+a.f[0]);
        for(int i=0;i<d;++i) o[off+i]=silu(a.p[3][off+i])*((x[off+i]*r)*(w[i]+a.f[1]));
        break;
    }
    case AttentionQueryKv: {
        Params part; int item; bool query=query_kv_part(a,t,part,item);
        if(query) attention_query_body(item,part); else attention_kv_body(item,part);
        break;
    }
    case EmbedBf16: { // widen embedding row n0 (row width n1) of a uint16 matrix
        const uint16_t* m=reinterpret_cast<const uint16_t*>(x);
        o[t]=bf16(m[size_t(a.n[0])*size_t(a.n[1])+size_t(t)]); break;
    }
    }
}
