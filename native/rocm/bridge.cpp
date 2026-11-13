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
#include <initializer_list>
#include <set>
#ifndef QR_CPU_TEST
#include <algorithm>
#include <map>
#include <string>
#include <vector>
#endif
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
    bool attention_f16 = false; // qr_set_option; prefill only (ADR 0014)
    float* parts = nullptr;     // split partials of the 16-bit prefill attention
    size_t parts_len = 0;
    unsigned* ticket = nullptr; // gemv_bf16_add_norm's last-block counter (kept 0 between launches)
    unsigned* delta_tickets = nullptr; // delta_mixer_split's per-head counters (2 x value heads, kept 0)
    int delta_tickets_len = 0;
    // QWEN_ROCM_PROFILE=1|repeat (see timed()): hipEvent pairs around bridge
    // calls, summed per label at each sync and printed to stderr when the
    // context is destroyed. Measurement only (ADR 0019).
    bool profile = false, repeat = false;
    // QWEN_ROCM_PROFILE=host: no events. Between consecutive readbacks
    // (qr_read: once per decode step), host time spent waiting for the queue
    // in the readback versus everything else (Rust, bridge calls, launches).
    bool host = false;
    double host_wait = 0, host_total = 0, host_last = 0; long host_reads = 0, host_calls = 0, host_calls_at = 0;
    struct Timed { std::string label; hipEvent_t start, stop; int times; };
    std::vector<Timed> timed;
    std::vector<hipEvent_t> spare;
    struct Total { long count = 0; double ms = 0; };
    std::map<std::string, Total> totals;
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
static double seconds() {
    timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return double(t.tv_sec) + 1e-9 * double(t.tv_nsec);
}
static hipEvent_t event(Context* c) {
    hipEvent_t e = nullptr;
    if (!c->spare.empty()) { e = c->spare.back(); c->spare.pop_back(); }
    else (void)hipEventCreate(&e);
    return e;
}
// Runs `launch` (one bridge call: one or two dispatches) and, when profiling,
// times it under `label`. QWEN_ROCM_PROFILE=1 brackets the call with one
// event pair (each pair adds roughly 10 us on WSL2, so small kernels look
// slower than they are). QWEN_ROCM_PROFILE=repeat issues the call 1 + kRepeat
// times and times the last kRepeat back to back: the mean is the dispatch's
// throughput cost with the event overhead amortized. Repeating re-applies
// stateful ops (conv history, DeltaNet state, KV append), so the values and
// generated tokens of a repeat run are meaningless; only its timing is used.
constexpr int kRepeat = 20;
template <class F>
static int timed(Context* c, const std::string& label, F&& launch) {
    if (c->host) c->host_calls++;
    if (!c->profile) return launch();
    int times = 1;
    if (c->repeat) {
        if (int s = launch()) return s;
        times = kRepeat;
    }
    Context::Timed t{label, event(c), event(c), times};
    (void)hipEventRecord(t.start, c->stream);
    for (int r = 0; r < times; ++r)
        if (int s = launch()) return s;
    (void)hipEventRecord(t.stop, c->stream);
    c->timed.push_back(std::move(t));
    return 0;
}
static void collect(Context* c) {
    for (auto& t : c->timed) {
        float ms = 0;
        if (hipEventElapsedTime(&ms, t.start, t.stop) == hipSuccess) {
            auto& total = c->totals[t.label]; total.count++; total.ms += ms / t.times;
        }
        c->spare.push_back(t.start); c->spare.push_back(t.stop);
    }
    c->timed.clear();
}
static void report(Context* c) {
    double all = 0; long n = 0;
    for (auto& [k, v] : c->totals) { all += v.ms; n += v.count; }
    if (n == 0) return;
    std::fprintf(stderr, "QWEN_ROCM_PROFILE=%s: %ld bridge calls, %.3f ms of GPU time%s\n", c->repeat ? "repeat" : "1", n, all,
                 c->repeat ? " (mean of back-to-back repeats; values of this run are meaningless)" : " (includes event overhead)");
    std::fprintf(stderr, "%10s %12s %10s %6s  %s\n", "count", "total_ms", "mean_us", "share", "label");
    std::vector<std::pair<double, std::string>> order;
    for (auto& [k, v] : c->totals) order.push_back({v.ms, k});
    std::sort(order.rbegin(), order.rend());
    for (auto& [ms, k] : order) {
        auto& v = c->totals[k];
        std::fprintf(stderr, "%10ld %12.3f %10.2f %5.1f%%  %s\n", v.count, v.ms, 1000 * v.ms / double(v.count), 100 * v.ms / all, k.c_str());
    }
}
static const char* op_name(int op) {
    static const char* names[] = {"?", "rms_norm", "add", "silu_product", "conv_silu", "delta_qk", "delta_gates",
                                  "delta_rule", "attention_query", "attention_kv", "attention_read", "embed_bf16",
                                  "add_rms_norm", "rms_norm_gate", "delta_rule_gated",
                                  "attention_query_kv", "argmax", "embed_bf16_picked"};
    return op >= 1 && op <= kLastOp ? names[op] : "?";
}
static unsigned blocks(long threads, int per_block) { return unsigned((threads + per_block - 1) / per_block); }
static bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; }

// RMSNorm and its fusions: one block per group, rounded up to whole waves.
static int rms_norm_launch(Context* c, void (*kernel)(Params), const char* what, int groups, const Params& a) {
    int threads = a.n[0] >= 256 ? 256 : ((a.n[0] + kWave - 1) / kWave) * kWave;
    hipLaunchKernelGGL(kernel, dim3(groups), dim3(threads), 0, c->stream, a);
    return launched(c, what);
}
static int rms_norm(Context* c, int groups, const Params& a) {
    return rms_norm_launch(c, rms_norm_block, "rms_norm_block", groups, a);
}
// DeltaRule: register-resident state, at most 32 key rows per lane.
template <int K, int P, bool GATED>
static void delta_launch(Context* c, int count, const Params& a) {
    dim3 grid(blocks(long(count) * P, 128));
    bool full = a.n[0] == K * P && reinterpret_cast<uintptr_t>(a.p[1]) % 16 == 0 && reinterpret_cast<uintptr_t>(a.p[2]) % 16 == 0;
    bool snap = a.n[6] > 0; // per-token snapshots (ADR 0026): separate instances, so the others keep their code
    if (snap && full) hipLaunchKernelGGL((delta_rule_registers<K, P, true, GATED, true>), grid, dim3(128), 0, c->stream, a, count);
    else if (snap) hipLaunchKernelGGL((delta_rule_registers<K, P, false, GATED, true>), grid, dim3(128), 0, c->stream, a, count);
    else if (full) hipLaunchKernelGGL((delta_rule_registers<K, P, true, GATED>), grid, dim3(128), 0, c->stream, a, count);
    else hipLaunchKernelGGL((delta_rule_registers<K, P, false, GATED>), grid, dim3(128), 0, c->stream, a, count);
}
template <bool GATED>
static int delta_rule(Context* c, int count, const Params& a) {
    int kd = a.n[0];
    if (kd > 128 && a.n[6] > 0) return fail(c, "DeltaRule snapshots need key_dim <= 128", -1);
    if (kd <= 8) delta_launch<8, 1, GATED>(c, count, a);
    else if (kd <= 32) delta_launch<32, 1, GATED>(c, count, a);
    else if (kd <= 64) delta_launch<32, 2, GATED>(c, count, a);
    else if (kd <= 128) delta_launch<32, 4, GATED>(c, count, a);
    else hipLaunchKernelGGL(reference_kernel, dim3(blocks(count, 128)), dim3(128), 0, c->stream, int(GATED ? DeltaRuleGated : DeltaRule), count, a);
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
    if (a.n[0] == L * kWave && a.n[0] % 4 == 0 && tokens == 1 && rows == 1) // decode: no LDS staging
        hipLaunchKernelGGL((attention_tiled<L, true, true>), grid, block, 0, c->stream, a, rows, span, splits);
    else if (a.n[0] == L * kWave && a.n[0] % 4 == 0)
        hipLaunchKernelGGL((attention_tiled<L, true>), grid, block, 0, c->stream, a, rows, span, splits);
    else
        hipLaunchKernelGGL((attention_tiled<L, false>), grid, block, 0, c->stream, a, rows, span, splits);
}
constexpr int kDecodeRows = 4; // up to this many query rows take the decode attention (ADR 0026)
static int attention(Context* c, int heads_count, const Params& a) {
    int d = a.n[0], heads = a.n[1], kv = a.n[2], tokens = a.n[5] > 0 ? a.n[5] : 1, scratch = a.n[6];
    int group = heads / kv;
    if (tokens > 1 && tokens <= kDecodeRows && scratch > 0 && d <= kAttnMaxDim && group <= kAttnWaves) {
        // A few query rows (a verify pass, ADR 0026): the decode attention once
        // per row r over its own n3 + r positions, so each row is exactly the
        // decode step of that position. Output, query and gate rows are offset;
        // the score scratch is reused row after row (one stream, in order).
        size_t q = size_t(d) * size_t(heads);
        for (int r = 0; r < tokens; ++r) {
            Params one = a;
            for (int slot : {0, 1, 4}) one.p[slot] = a.p[slot] + r * q;
            one.n[3] = a.n[3] + r;
            one.n[5] = 0;
            if (int s = attention(c, heads_count, one)) return s;
        }
        return 0;
    }
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
        hipLaunchKernelGGL(attention_combine, dim3(blocks(long(d) * heads, 256)), dim3(256), 0, c->stream,
                           a, a.p[5], long(scratch), heads, splits);
        return launched(c, "attention_combine");
    }
    return 0;
}
// Opt-in 16-bit prefill attention: WMMA tiles of 16 query vectors, positions
// split into kWmmaSpan spans whose partials go to a bridge-owned buffer.
template <int DT>
static void wmma_launch(Context* c, const Params& a, int span, int splits) {
    int tokens = a.n[5], rows = kWmma / (a.n[1] / a.n[2]);
    dim3 grid(unsigned((tokens + rows - 1) / rows), unsigned(a.n[2]), unsigned(splits));
    hipLaunchKernelGGL(attention_wmma<DT>, grid, dim3(kWave * kWmmaWaves(DT)), 0, c->stream, a, span, splits, c->parts);
}
static int attention_f16(Context* c, const Params& a) {
    int d = a.n[0], heads = a.n[1], kv = a.n[2], tokens = a.n[5], group = heads / kv;
    if (d % kWmma != 0 || d > kAttnMaxDim || kWmma % group != 0)
        return fail(c, "attention_f16 needs head_dim % 16 == 0, head_dim <= 256 and 16 % (heads/kv) == 0", -1);
    int length = a.n[3] + tokens - 1, span = kWmmaSpan, splits = (length + span - 1) / span;
    if (splits > 1) {
        size_t need = size_t(tokens) * heads * splits * (d + 2);
        if (need > c->parts_len) {
            if (c->parts) {
                (void)hipStreamSynchronize(c->stream);
                (void)hipFree(c->parts);
                c->parts = nullptr; c->parts_len = 0;
            }
            auto st = hipMalloc(reinterpret_cast<void**>(&c->parts), need * sizeof(float));
            if (st != hipSuccess) return fail(c, "hipMalloc attention_f16 partials", int(st));
            c->parts_len = need;
        }
    }
    switch (d / kWmma) {
    case 1: wmma_launch<1>(c, a, span, splits); break;
    case 2: wmma_launch<2>(c, a, span, splits); break;
    case 4: wmma_launch<4>(c, a, span, splits); break;
    case 8: wmma_launch<8>(c, a, span, splits); break;
    case 16: wmma_launch<16>(c, a, span, splits); break;
    default: return fail(c, "attention_f16 head_dim must be 16, 32, 64, 128 or 256", -1);
    }
    if (int s = launched(c, "attention_wmma")) return s;
    if (splits > 1) {
        long stride = long(splits) * (d + 2);
        hipLaunchKernelGGL(attention_combine, dim3(blocks(long(d) * heads * tokens, 256)), dim3(256), 0, c->stream,
                           a, c->parts, stride, heads * tokens, splits);
        return launched(c, "attention_combine (f16)");
    }
    return 0;
}
// Fixed-count GEMV kernels (issue #32) for 2..kGemvTokens tokens; set
// QWEN_ROCM_FIXED_GEMV=0 to use the runtime-count kernels (A/B only).
static bool fixed_gemv(int batch) {
    static const bool on = [] { const char* e = getenv("QWEN_ROCM_FIXED_GEMV"); return !(e && std::strcmp(e, "0") == 0); }();
    return on && batch >= 2 && batch <= kGemvTokens;
}
#define QR_FIXED_SWITCH(batch, LAUNCH) \
    switch (batch) { \
    case 2: LAUNCH(2); break; case 3: LAUNCH(3); break; case 4: LAUNCH(4); break; \
    case 5: LAUNCH(5); break; case 6: LAUNCH(6); break; case 7: LAUNCH(7); break; \
    default: LAUNCH(8); break; \
    }
// BF16 matrix x FP32 activations. rocBLAS has no such product, so these are our own.
static int project_bf16(Context* c, float* y, const uint16_t* a, const float* x, int rows, int cols, int batch) {
    if(cols%8!=0 || !aligned16(a) || !aligned16(x)) {
        hipLaunchKernelGGL(gemv_bf16_block,dim3(rows,batch),dim3(256),0,c->stream,y,a,x,rows,cols);
        return launched(c,"gemv_bf16_block");
    }
    if(batch>=kGemmMinTokens) {
        dim3 grid(blocks(rows,kTileRows),blocks(batch,kTileTokens));
        hipLaunchKernelGGL(gemm_bf16_tiled,grid,dim3(256),0,c->stream,y,a,x,rows,cols,batch);
        return launched(c,"gemm_bf16_tiled");
    }
    if(batch==1) {
        hipLaunchKernelGGL(gemv_bf16_wave1,dim3(blocks(rows,kDecodeRowsPerBlock)),dim3(256),0,c->stream,y,a,x,rows,cols);
        return launched(c,"gemv_bf16_wave1");
    }
    dim3 grid(blocks(rows,kGemvRowsPerBlock),blocks(batch,kGemvTokens));
    if(fixed_gemv(batch)) {
#define QR_LAUNCH(NB) hipLaunchKernelGGL(gemv_bf16_wave_t<NB>,grid,dim3(256),0,c->stream,y,a,x,rows,cols)
        QR_FIXED_SWITCH(batch, QR_LAUNCH)
#undef QR_LAUNCH
        return launched(c,"gemv_bf16_wave_t");
    }
    hipLaunchKernelGGL(gemv_bf16_wave,grid,dim3(256),0,c->stream,y,a,x,rows,cols,batch);
    return launched(c,"gemv_bf16_wave");
}
#endif
extern "C" {
int qr_abi() { return 3; }
// Optional symbol (no ABI bump; #19): the highest qr_run op this bridge runs.
// Without it Rust assumes 11 (EmbedBf16) and never issues the fused ops.
int qr_max_op() { return kMaxOp; }
// Optional (ADR 0026): ConvSilu n3 and DeltaRule(Gated) n6 store per-token
// state snapshots after the live state (a verify pass with rollback).
int qr_snapshots() { return 1; }
// Optional (ADR 0030-0032): op 16 Argmax and op 17 EmbedBf16Picked.
int qr_argmax() { return 2; }
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
    const char* prof=getenv("QWEN_ROCM_PROFILE");
    c->profile = prof && std::strcmp(prof,"0")!=0;
    c->repeat = prof && std::strcmp(prof,"repeat")==0;
    c->host = prof && std::strcmp(prof,"host")==0;
    if (c->host) c->profile = false;
    hipDeviceProp_t props{}; int runtime=0, driver=0;
    auto ps=hipGetDeviceProperties(&props,0);
    auto rs=hipRuntimeGetVersion(&runtime), ds=hipDriverGetVersion(&driver);
    std::snprintf(c->info,sizeof(c->info),"device0=%s arch=%s cus=%d hip_runtime=%d(status=%d) driver_api=%d(status=%d) props_status=%d wsl_dxg=%d kernels=adr0019",
        ps==hipSuccess ? props.name : "unknown", ps==hipSuccess ? props.gcnArchName : "unknown",
        ps==hipSuccess ? props.multiProcessorCount : 0, runtime,int(rs),driver,int(ds),int(ps),int(dxg));
#endif
    *out=c; return 0;
}
// Optional symbol (no ABI bump): engine options. A bridge without it (Metal,
// older builds) cannot be asked for one, so Rust reports an error instead of
// silently ignoring the request. Unknown names and values fail too.
int qr_set_option(Context* c, const char* name, int value) {
    if (std::strcmp(name, "attention_f16") != 0 || (value != 0 && value != 1))
        return fail(c, "unknown bridge option", -1);
#ifdef QR_CPU_TEST
    if (value) return fail(c, "attention_f16 needs the HIP bridge (CPU test bridge runs the FP32 reference)", -1);
#else
    c->attention_f16 = value == 1;
#endif
    return 0;
}
int qr_sync(Context* c) {
#ifndef QR_CPU_TEST
    auto s=hipStreamSynchronize(c->stream); if(s!=hipSuccess) return fail(c,"hipStreamSynchronize",int(s));
    if(c->profile) collect(c);
#else
    (void)c;
#endif
    return 0;
}
void qr_destroy(Context* c) {
#ifndef QR_CPU_TEST
    (void)hipStreamSynchronize(c->stream);
    if (c->host && c->host_reads > 1) {
        long n = c->host_reads - 1;
        std::fprintf(stderr, "QWEN_ROCM_PROFILE=host: %ld readbacks after the first: %.3f ms apart, of which %.3f ms waiting in the readback"
                     " (%.1f%%), %.1f bridge calls each\n", n, 1000 * c->host_total / double(n), 1000 * c->host_wait / double(n),
                     100 * c->host_wait / c->host_total, double(c->host_calls - c->host_calls_at) / double(n));
    }
    if (c->profile) {
        collect(c); report(c);
        for (auto e : c->spare) (void)hipEventDestroy(e);
    }
    rocblas_destroy_handle(c->blas); (void)hipStreamDestroy(c->stream);
    if (c->parts) (void)hipFree(c->parts);
    if (c->ticket) (void)hipFree(c->ticket);
    if (c->delta_tickets) (void)hipFree(c->delta_tickets);
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
    // A pageable D2H copy already waits for the queue, so host profiling
    // counts the copy as waiting too.
    double before = c->host ? seconds() : 0;
    auto s=hipMemcpyAsync(host,p,n*sizeof(float),hipMemcpyDeviceToHost,c->stream);
    if(s!=hipSuccess) return fail(c,"D2H",int(s));
    int status = qr_sync(c);
    if (c->host) {
        double after = seconds();
        if (c->host_reads > 0) { c->host_wait += after - before; c->host_total += after - c->host_last; }
        c->host_reads++; c->host_last = after;
        if (c->host_reads == 1) c->host_calls_at = c->host_calls;
    }
    return status;
#endif
    return 0;
}
int qr_copy(Context* c, float* dest, const float* src, size_t n) {
#ifdef QR_CPU_TEST
    std::memcpy(dest,src,n*sizeof(float)); (void)c;
#else
    return timed(c,"copy",[&]{
        auto s=hipMemcpyAsync(dest,src,n*sizeof(float),hipMemcpyDeviceToDevice,c->stream);
        return s!=hipSuccess ? fail(c,"D2D",int(s)) : 0;
    });
#endif
    return 0;
}
int qr_run(Context* c, int op, int count, const Params* a) {
    if(op<1 || op>kLastOp || count<1) return fail(c,"invalid kernel",-1);
    if(op==Argmax && a->n[0]<1) return fail(c,"argmax shape",-1);
#ifdef QR_CPU_TEST
    for(int t=0;t<count;++t) body(op,t,*a);
#else
    std::string label = c->profile ? std::string(op_name(op)) + (op==AttentionRead && a->n[5]<=1 ? " (decode)" : "") : std::string();
    return timed(c,label,[&]{
        switch(op) {
        case RmsNorm: return rms_norm(c,count,*a);
        case AddRmsNorm: return rms_norm_launch(c,add_rms_norm_block,"add_rms_norm_block",count,*a);
        case RmsNormGate: return rms_norm_launch(c,rms_norm_gate_block,"rms_norm_gate_block",count,*a);
        case ConvSilu:
            if(a->n[3]<=0) break; // reference_kernel, as before
            hipLaunchKernelGGL(conv_silu_snapshots,dim3(blocks(count,256)),dim3(256),0,c->stream,*a,count);
            return launched(c,"conv_silu_snapshots");
        case DeltaRule: return delta_rule<false>(c,count,*a);
        case DeltaRuleGated: return delta_rule<true>(c,count,*a);
        case DeltaQk:
            hipLaunchKernelGGL(delta_qk_wave,dim3(blocks(count,kHeadWaves)),dim3(256),0,c->stream,*a,count);
            return launched(c,"delta_qk_wave");
        case AttentionQuery:
            hipLaunchKernelGGL(attention_query_wave,dim3(blocks(count,kHeadWaves)),dim3(256),0,c->stream,*a,count);
            return launched(c,"attention_query_wave");
        case AttentionKv:
            hipLaunchKernelGGL(attention_kv_wave,dim3(blocks(count,kHeadWaves)),dim3(256),0,c->stream,*a,count);
            return launched(c,"attention_kv_wave");
        case AttentionQueryKv:
            hipLaunchKernelGGL(attention_query_kv_wave,dim3(blocks(count,kHeadWaves)),dim3(256),0,c->stream,*a,count);
            return launched(c,"attention_query_kv_wave");
        case Argmax:
            hipLaunchKernelGGL(argmax_rows,dim3(unsigned(count)),dim3(256),0,c->stream,*a);
            return launched(c,"argmax_rows");
        case AttentionRead:
            return c->attention_f16 && a->n[5] > 1 ? attention_f16(c,*a) : attention(c,count,*a);
        default: break;
        }
        hipLaunchKernelGGL(reference_kernel,dim3(blocks(count,256)),dim3(256),0,c->stream,op,count,*a);
        return launched(c,"HIP kernel launch");
    });
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
    std::string label = c->profile ? std::string(batch>=kGemmMinTokens ? "gemm_bf16 " : "gemv_bf16 ")+std::to_string(rows)+"x"+std::to_string(cols)
                                  +(batch>1 ? " t"+std::to_string(batch) : "") : std::string();
    return timed(c,label,[&]{ return project_bf16(c,y,a,x,rows,cols,batch); });
#endif
    return 0;
}
int qr_gemv_bf16(Context* c, float* y, const float* packed, const float* x, int rows, int cols) {
    return qr_gemm_bf16(c,y,packed,x,rows,cols,1);
}
// Optional symbol (no ABI bump; #19): n BF16 projections of the same X
// [batch, cols], Y_i[batch, rows_i] = X . A_i^T, in as few dispatches as the
// bridge can. Every Y_i equals what qr_gemm_bf16 would compute. Bridges
// without it (Metal) get one qr_gemm_bf16 call per matrix from Rust.
struct QrProjection { float* y; const float* a; int rows; };
int qr_gemm_bf16_group(Context* c, int n, const QrProjection* items, const float* x, int cols, int batch) {
    if(n<1 || cols<1 || batch<1) return fail(c,"bf16 GEMM group shape",-1);
    for(int i=0;i<n;++i) if(items[i].rows<1) return fail(c,"bf16 GEMM group shape",-1);
#ifdef QR_CPU_TEST
    for(int i=0;i<n;++i) if(int s=qr_gemm_bf16(c,items[i].y,items[i].a,x,items[i].rows,cols,batch)) return s;
    return 0;
#else
    bool grouped = batch < kGemmMinTokens && cols % 8 == 0 && aligned16(x);
    for(int i=0;i<n;++i) grouped = grouped && aligned16(items[i].a);
    if(!grouped) {
        for(int i=0;i<n;++i) if(int s=qr_gemm_bf16(c,items[i].y,items[i].a,x,items[i].rows,cols,batch)) return s;
        return 0;
    }
    for(int i0=0;i0<n;i0+=kGroupMax) {
        GemvGroup g{}; g.count = n-i0 < kGroupMax ? n-i0 : kGroupMax;
        std::string label = c->profile ? "gemv_bf16 group" : std::string();
        int per_block = batch==1 ? kDecodeRowsPerBlock : kGemvRowsPerBlock;
        for(int m=0;m<g.count;++m) {
            const auto& it = items[i0+m];
            g.y[m]=it.y; g.a[m]=reinterpret_cast<const uint16_t*>(it.a); g.rows[m]=it.rows;
            g.first[m+1]=g.first[m]+int(blocks(it.rows,per_block));
            if(c->profile) label += " "+std::to_string(it.rows)+"x"+std::to_string(cols);
        }
        if(c->profile && batch>1) label += " t"+std::to_string(batch);
        dim3 grid(unsigned(g.first[g.count]),blocks(batch,kGemvTokens));
        if(int s=timed(c,label,[&]{
            if(batch==1) hipLaunchKernelGGL(gemv_bf16_group1,grid,dim3(256),0,c->stream,g,x,cols);
            else if(fixed_gemv(batch)) {
#define QR_LAUNCH(NB) hipLaunchKernelGGL(gemv_bf16_group_t<NB>,grid,dim3(256),0,c->stream,g,x,cols)
                QR_FIXED_SWITCH(batch, QR_LAUNCH)
#undef QR_LAUNCH
            }
            else hipLaunchKernelGGL(gemv_bf16_group,grid,dim3(256),0,c->stream,g,x,cols,batch);
            return launched(c,"gemv_bf16_group");
        })) return s;
    }
    return 0;
#endif
}
// Optional symbol (no ABI bump; #19): FFN gate/up projections of the same X
// and their SiluProduct, product[batch, rows] = silu(X . G^T) * (X . U^T).
// gate_y/up_y are scratch [batch, rows] the bridge may use; the product is
// the same as two qr_gemm_bf16 calls followed by op SiluProduct.
int qr_gemm_bf16_glu(Context* c, float* product, float* gate_y, float* up_y, const float* gate, const float* up,
                     const float* x, int rows, int cols, int batch) {
    if(rows<1 || cols<1 || batch<1) return fail(c,"bf16 GLU shape",-1);
    Params p{}; p.p[0]=product; p.p[1]=gate_y; p.p[2]=up_y;
#ifndef QR_CPU_TEST
    const uint16_t *g=reinterpret_cast<const uint16_t*>(gate), *u=reinterpret_cast<const uint16_t*>(up);
    if(batch < kGemmMinTokens && cols % 8 == 0 && aligned16(x) && aligned16(g) && aligned16(u)) {
        std::string label = c->profile ? "gemv_bf16_glu "+std::to_string(rows)+"x"+std::to_string(cols)+(batch>1 ? " t"+std::to_string(batch) : "") : std::string();
        return timed(c,label,[&]{
            dim3 grid(blocks(rows,kGemvRowsPerBlock),blocks(batch,kGemvTokens));
            if(batch==1) hipLaunchKernelGGL(gemv_bf16_glu1,grid,dim3(256),0,c->stream,product,g,u,x,rows,cols);
            else if(fixed_gemv(batch)) {
#define QR_LAUNCH(NB) hipLaunchKernelGGL(gemv_bf16_glu_t<NB>,grid,dim3(256),0,c->stream,product,g,u,x,rows,cols)
                QR_FIXED_SWITCH(batch, QR_LAUNCH)
#undef QR_LAUNCH
            }
            else hipLaunchKernelGGL(gemv_bf16_glu,grid,dim3(256),0,c->stream,product,g,u,x,rows,cols,batch);
            return launched(c,"gemv_bf16_glu");
        });
    }
#endif
    if(int s=qr_gemm_bf16(c,gate_y,gate,x,rows,cols,batch)) return s;
    if(int s=qr_gemm_bf16(c,up_y,up,x,rows,cols,batch)) return s;
    return qr_run(c,SiluProduct,rows*batch,&p);
}
// Optional symbol (no ABI bump; ADR 0019): DeltaNet's ConvSilu, DeltaQk,
// DeltaRuleGated and RmsNormGate (offset 0) with the slots Rust gives those
// ops. For a few tokens (decode) the HIP bridge runs them as one block per
// value head; otherwise, and on the CPU test bridge, as the four ops. Either
// way the outputs equal the four qr_run calls.
struct QrDelta {
    float* convolved; const float* projected; const float* conv_w; float* conv_state;
    float* query; float* key; float* core; float* recurrent;
    const float* a; const float* b; const float* gates;
    float* gated; const float* norm_w; const float* z;
    int key_heads, value_heads, key_dim, value_dim, kernel, tokens, channels;
    float eps;
};
static int delta_ops(Context* c, const QrDelta& d) {
    auto P = [](std::initializer_list<const float*> p, std::initializer_list<int> n, std::initializer_list<float> f) {
        Params a{}; int i=0;
        for(auto v : p) a.p[i++]=const_cast<float*>(v);
        i=0; for(auto v : n) a.n[i++]=v;
        i=0; for(auto v : f) a.f[i++]=v;
        return a;
    };
    int vh=d.value_heads, vd=vh*d.value_dim, t=d.tokens;
    Params conv=P({d.convolved,d.projected,d.conv_w,d.conv_state},{d.kernel,t,d.channels},{});
    Params qk=P({d.query,d.convolved,d.key},{d.key_heads,vh,d.key_dim,d.channels},{});
    Params rule=P({d.core,d.query,d.key,d.convolved,d.recurrent,d.a,d.b,d.gates},
                  {d.key_dim,d.value_dim,d.key_heads,vh,t,d.channels},{});
    Params gate=P({d.gated,d.core,d.norm_w,d.z},{d.value_dim},{d.eps,0.f});
    if(int s=qr_run(c,ConvSilu,d.channels,&conv)) return s;
    if(int s=qr_run(c,DeltaQk,vh*t,&qk)) return s;
    if(int s=qr_run(c,DeltaRuleGated,vd,&rule)) return s;
    return qr_run(c,RmsNormGate,vh*t,&gate);
}
int qr_delta_mixer(Context* c, const QrDelta* d) {
    if(!d || d->tokens<1 || d->value_heads<1 || d->key_heads<1 || d->value_heads%d->key_heads!=0)
        return fail(c,"delta mixer shape",-1);
#ifndef QR_CPU_TEST
    const QrDelta& m=*d;
    bool fused = m.tokens <= kDeltaFusedTokens && m.key_heads == m.value_heads && m.value_dim == kNormGroup
        && m.key_dim == 128 && aligned16(m.query) && aligned16(m.key);
    bool split = fused && m.tokens == 1 && m.value_dim == 128 && m.kernel >= 1 && m.kernel <= kDeltaConvMax;
    if(split && c->delta_tickets_len < 2*m.value_heads) {
        if(c->delta_tickets) { (void)hipStreamSynchronize(c->stream); (void)hipFree(c->delta_tickets); c->delta_tickets=nullptr; }
        auto st=hipMalloc(reinterpret_cast<void**>(&c->delta_tickets),2*m.value_heads*sizeof(unsigned));
        if(st!=hipSuccess) { c->delta_tickets_len=0; return fail(c,"hipMalloc delta tickets",int(st)); }
        st=hipMemsetAsync(c->delta_tickets,0,2*m.value_heads*sizeof(unsigned),c->stream);
        if(st!=hipSuccess) return fail(c,"hipMemsetAsync delta tickets",int(st));
        c->delta_tickets_len=2*m.value_heads;
    }
    if(split) {
        DeltaMixer k{m.convolved,m.projected,m.conv_w,m.conv_state,m.query,m.key,m.core,m.recurrent,m.a,m.b,m.gates,
                     m.gated,m.norm_w,m.z,m.key_heads,m.value_heads,m.key_dim,m.value_dim,m.kernel,m.tokens,m.channels,m.eps};
        return timed(c,c->profile ? "delta_mixer_split" : "",[&]{
            hipLaunchKernelGGL(delta_mixer_split,dim3(unsigned(m.value_heads*kDeltaParts)),dim3(128),0,c->stream,k,c->delta_tickets);
            return launched(c,"delta_mixer_split");
        });
    }
    if(fused) {
        DeltaMixer k{m.convolved,m.projected,m.conv_w,m.conv_state,m.query,m.key,m.core,m.recurrent,m.a,m.b,m.gates,
                     m.gated,m.norm_w,m.z,m.key_heads,m.value_heads,m.key_dim,m.value_dim,m.kernel,m.tokens,m.channels,m.eps};
        return timed(c,c->profile ? "delta_mixer_block" : "",[&]{
            hipLaunchKernelGGL((delta_mixer_block<32,4>),dim3(unsigned(m.value_heads)),dim3(unsigned(m.value_dim*4)),0,c->stream,k);
            return launched(c,"delta_mixer_block");
        });
    }
#endif
    return delta_ops(c,*d);
}
// Optional symbol (no ABI bump; ADR 0019): y = A x for one token (BF16 A),
// then Add + RmsNorm as op AddRmsNorm: post->sum = u + v, post->norm =
// RmsNorm(sum) (weight, eps, offset), where u or v may be y. The same
// outputs as qr_gemm_bf16 followed by AddRmsNorm.
struct QrAddNormPost { float* sum; const float* u; const float* v; const float* weight; float* norm; float eps; float offset; };
int qr_gemm_bf16_add_norm(Context* c, float* y, const float* packed, const float* x, int rows, int cols,
                          const QrAddNormPost* post) {
    if(rows<1 || cols<1 || !post) return fail(c,"bf16 GEMV+norm shape",-1);
#ifndef QR_CPU_TEST
    const uint16_t* a=reinterpret_cast<const uint16_t*>(packed);
    // rows >= 256: AddRmsNorm then reduces over 256 threads, as the tail does.
    if(rows>=256 && cols%8==0 && aligned16(a) && aligned16(x)) {
        if(!c->ticket) {
            auto st=hipMalloc(reinterpret_cast<void**>(&c->ticket),sizeof(unsigned));
            if(st!=hipSuccess) return fail(c,"hipMalloc ticket",int(st));
            st=hipMemsetAsync(c->ticket,0,sizeof(unsigned),c->stream);
            if(st!=hipSuccess) return fail(c,"hipMemsetAsync ticket",int(st));
        }
        AddNormTail n{post->sum,post->u,post->v,post->weight,post->norm,post->eps,post->offset,c->ticket};
        std::string label = c->profile ? "gemv_bf16+add_norm "+std::to_string(rows)+"x"+std::to_string(cols) : std::string();
        return timed(c,label,[&]{
            hipLaunchKernelGGL(gemv_bf16_add_norm,dim3(blocks(rows,kDecodeRowsPerBlock)),dim3(256),0,c->stream,y,a,x,rows,cols,n);
            return launched(c,"gemv_bf16_add_norm");
        });
    }
#endif
    if(int s=qr_gemm_bf16(c,y,packed,x,rows,cols,1)) return s;
    Params p{}; p.p[0]=post->sum; p.p[1]=const_cast<float*>(post->u); p.p[2]=const_cast<float*>(post->v);
    p.p[3]=post->norm; p.p[4]=const_cast<float*>(post->weight); p.n[0]=rows; p.f[0]=post->eps; p.f[1]=post->offset;
    return qr_run(c,AddRmsNorm,1,&p);
}
}
