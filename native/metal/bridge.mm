// Apache-2.0. Metal implementation of the versioned C ABI in src/device.rs
// (same symbols as native/rocm/bridge.cpp). Build: tools/build-metal.sh.
//
// Memory. Apple GPUs share memory with the CPU, so every allocation is a
// MTLStorageModeShared buffer and the pointer handed to Rust is its CPU
// `contents` address (weights may also be host memory wrapped in place,
// qr_wrap). Rust may offset such pointers (embedding row lookup); `resolve`
// maps any interior pointer back to (MTLBuffer, byte offset). Rust never
// dereferences them: reads/writes go through qr_read/qr_write, which first
// wait for every queued GPU command.
//
// Ordering. All work is encoded into ONE compute encoder of ONE command buffer
// and only committed at a synchronization point (read, write, sync). The
// default serial dispatch mode orders every kernel after the previous one,
// like the single HIP stream of the ROCm bridge.
//
// Kernel levels (spec/metal.md). Every op has a level-1 kernel that mirrors
// the CPU code; some have faster level-2 (SIMD groups, 8x8 simdgroup
// matrices, any Apple GPU) and level-3 (Metal 4 tensor ops, optional)
// kernels. Each op's dispatcher below takes the highest level that is
// allowed (qr_set_option "kernel_level", default 3), available, and
// supports the shape. All levels compute the same values up to summation
// order (ADR 0004 tolerances); the opt-in "matmul_relaxed" is the only
// option that changes values beyond that.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <new>

static const char* kKernelSource =
#include "kernels.inc"
    ;
static const char* kTensorKernelSource =
#include "kernels_tensor.inc"
    ;

enum Op { RmsNorm=1, Add=2, SiluProduct=3, ConvSilu=4, DeltaQk=5, DeltaGates=6, DeltaRule=7,
          AttentionQuery=8, AttentionKv=9, AttentionRead=10, EmbedBf16=11, OpCount=12 };
struct Params { float* p[8]; int n[8]; float f[4]; };
struct Scalars { int n[8]; float f[4]; int count; }; // == kernels/common.metal
// Number of buffer slots each op binds; must equal its kernel signatures.
static const int kSlots[OpCount] = {0, 3, 3, 3, 4, 3, 6, 7, 4, 5, 6, 2};
// Level-1 kernel of each op (AttentionRead has three passes, see below).
static const char* kLevel1[OpCount] = {nullptr, "rms_norm", "add", "silu_product", "conv_silu",
    "delta_qk", "delta_gates", "delta_rule", "attention_query", "attention_kv", nullptr, "embed_bf16"};

using Pipeline = id<MTLComputePipelineState>;
// One pipeline per kernel, grouped by op and level (kernels/*.metal).
struct Kernels {
    Pipeline op[OpCount] = {};                     // level 1 of the per-item ops
    Pipeline copy = nil;
    Pipeline norm_group = nil;                     // RMSNorm level 2
    Pipeline delta_split64 = nil, delta_split128 = nil; // DeltaNet rule level 2 (key_dim)
    Pipeline scores = nil, softmax = nil, values = nil; // decode attention level 1
    Pipeline split_grp64 = nil, split_grp128 = nil, split_grp256 = nil, combine = nil; // level 2
    Pipeline flash = nil;                          // prefill attention level 1
    Pipeline mma64 = nil, mma128 = nil, mma256 = nil;   // level 2
    Pipeline gemv = nil, gemv_bf16 = nil;          // projections level 1 (and decode)
    Pipeline tiled = nil, tiled_bf16 = nil, tiled16 = nil, tiled16_bf16 = nil; // level 2
    // Level 3 (kernels/tensor.metal), nil when Metal 4 tensor ops are unavailable.
    Pipeline tensor = nil, tensor_bf16 = nil, tensor_relaxed = nil, tensor_relaxed_bf16 = nil;
    Pipeline att_tensor64 = nil, att_tensor128 = nil, att_tensor256 = nil;
};

struct Context {
    char error[512] = {};
    char info[512] = {};
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    Kernels k;
    int level = 3;               // highest kernel level allowed ("kernel_level")
    int max_level = 2;           // 3 when the tensor library compiled
    bool matmul_relaxed = false; // "matmul_relaxed": opt-in, changes results (ADR 0016)
    id<MTLCommandBuffer> pending = nil;
    id<MTLComputeCommandEncoder> encoder = nil;
    std::map<uintptr_t, id<MTLBuffer>> buffers; // keyed by contents address
};

// ---------------------------------------------------------------- helpers --

static int fail(Context* c, const char* what, int status) {
    std::snprintf(c->error, sizeof(c->error), "%s", what);
    return status ? status : -1;
}
static int fail_ns(Context* c, const char* what, NSError* e) {
    std::snprintf(c->error, sizeof(c->error), "%s: %s", what,
                  e ? e.localizedDescription.UTF8String : "unknown error");
    return -1;
}
// Find the allocation containing [p, p+bytes). Returns nil if p is foreign/out of bounds.
static id<MTLBuffer> resolve(Context* c, const void* p, size_t bytes, NSUInteger* offset) {
    auto key = reinterpret_cast<uintptr_t>(p);
    auto it = c->buffers.upper_bound(key);
    if (it == c->buffers.begin()) return nil;
    --it;
    size_t start = key - it->first;
    if (start + bytes > it->second.length || (start % sizeof(float)) != 0) return nil;
    *offset = start;
    return it->second;
}
static id<MTLComputeCommandEncoder> encoder(Context* c) {
    if (!c->encoder) {
        c->pending = [c->queue commandBuffer];
        c->encoder = [c->pending computeCommandEncoder];
    }
    return c->encoder;
}
// Commit queued work and wait. GPU faults surface here, like hipStreamSynchronize.
static int flush(Context* c) {
    if (!c->pending) return 0;
    [c->encoder endEncoding];
    id<MTLCommandBuffer> cb = c->pending;
    c->encoder = nil;
    c->pending = nil;
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) return fail_ns(c, "Metal command buffer", cb.error);
    return 0;
}
static void set_scalars(id<MTLComputeCommandEncoder> e, const Scalars& s) {
    [e setBytes:&s length:sizeof(s) atIndex:8];
}
// `count` items, one thread each.
static void dispatch_threads(id<MTLComputeCommandEncoder> e, Pipeline p, NSUInteger count) {
    NSUInteger width = MIN(MIN(p.maxTotalThreadsPerThreadgroup, (NSUInteger)256), count);
    [e setComputePipelineState:p];
    [e dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
}
// x * y items, one thread each.
static void dispatch_grid(id<MTLComputeCommandEncoder> e, Pipeline p, int x, int y) {
    NSUInteger w = MIN((NSUInteger)x, p.threadExecutionWidth);
    NSUInteger h = MAX((NSUInteger)1, MIN((NSUInteger)y, p.maxTotalThreadsPerThreadgroup / w));
    [e setComputePipelineState:p];
    [e dispatchThreads:MTLSizeMake((NSUInteger)x, (NSUInteger)y, 1) threadsPerThreadgroup:MTLSizeMake(w, h, 1)];
}
static void dispatch_groups(id<MTLComputeCommandEncoder> e, Pipeline p, NSUInteger x, NSUInteger y, NSUInteger threads) {
    [e setComputePipelineState:p];
    [e dispatchThreadgroups:MTLSizeMake(x, y, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}
static Pipeline by_head_dim(int d, Pipeline d64, Pipeline d128, Pipeline d256) {
    return d == 64 ? d64 : d == 128 ? d128 : d == 256 ? d256 : nil;
}

// ------------------------------------------------------------ dispatchers --
// Each takes the highest allowed level whose kernel exists and fits the shape.

// RMSNorm over `count` groups of n0 elements.
static void run_rms_norm(Context* c, id<MTLComputeCommandEncoder> e, const Params* a, const Scalars& s, int count) {
    set_scalars(e, s);
    if (c->level >= 2) { // one threadgroup (<= 256 threads) per group
        NSUInteger threads = MIN((NSUInteger)256, c->k.norm_group.maxTotalThreadsPerThreadgroup);
        threads = MIN(threads, ((NSUInteger)MAX(a->n[0], 1) + 31) / 32 * 32);
        dispatch_groups(e, c->k.norm_group, (NSUInteger)count, 1, threads);
        return;
    }
    dispatch_threads(e, c->k.op[RmsNorm], (NSUInteger)count); // level 1: one thread per group
}

// DeltaNet rule over `count` value columns (n0 = key_dim).
static void run_delta_rule(Context* c, id<MTLComputeCommandEncoder> e, const Params* a, const Scalars& s, int count) {
    set_scalars(e, s);
    Pipeline split = a->n[0] == 128 ? c->k.delta_split128 : a->n[0] == 64 ? c->k.delta_split64 : nil;
    if (c->level >= 2 && split) { // 4 lanes per column, whole SIMD groups
        NSUInteger n = ((NSUInteger)count * 4 + 31) / 32 * 32;
        [e setComputePipelineState:split];
        [e dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)256, n), 1, 1)];
        return;
    }
    dispatch_threads(e, c->k.op[DeltaRule], (NSUInteger)count); // level 1: one thread per column
}

// Decode attention (T = 1) over L = n3 positions.
static const int kSplitSpan = 32; // smallest split; == ATTENTION_SPLIT in src/device.rs
static void run_attention_decode(Context* c, id<MTLComputeCommandEncoder> e, const Params* a, Scalars s) {
    int d = a->n[0], heads = a->n[1], kv = a->n[2], length = a->n[3], row = a->n[6];
    Pipeline grp = heads / kv == 4 ? by_head_dim(d, c->k.split_grp64, c->k.split_grp128, c->k.split_grp256) : nil;
    if (c->level >= 2 && grp && row >= d + 2) {
        // Level 2 (ADR 0017): ~40 splits of 32..256 positions (a multiple of 32),
        // grown so splits * (d + 2) fits the per-head scratch row.
        int fit = row / (d + 2);
        int want = MIN(256, MAX(kSplitSpan, ((length + 39) / 40 + 31) / 32 * 32));
        int span = MAX(want, (length + fit - 1) / fit);
        int splits = (length + span - 1) / span;
        s.n[7] = span;
        set_scalars(e, s);
        dispatch_groups(e, grp, (NSUInteger)splits, (NSUInteger)kv, 32);
        s.n[5] = splits;
        set_scalars(e, s);
        dispatch_grid(e, c->k.combine, d, heads);
        return;
    }
    // Level 1: scores, softmax per head, weighted sum (score rows in the scratch).
    set_scalars(e, s);
    dispatch_grid(e, c->k.scores, length, heads);
    dispatch_groups(e, c->k.softmax, (NSUInteger)heads, 1, MIN(c->k.softmax.maxTotalThreadsPerThreadgroup, (NSUInteger)1024));
    dispatch_grid(e, c->k.values, d, heads);
}

// Prefill attention for a chunk of T = n5 query rows.
static void run_attention_prefill(Context* c, id<MTLComputeCommandEncoder> e, const Params* a, const Scalars& s) {
    int d = a->n[0], heads = a->n[1], kv = a->n[2], tokens = a->n[5], group = heads / kv;
    set_scalars(e, s);
    Pipeline tensor = by_head_dim(d, c->k.att_tensor64, c->k.att_tensor128, c->k.att_tensor256);
    if (c->level >= 3 && tensor) { // 32 query rows x 1 head per threadgroup (ADR 0016)
        dispatch_groups(e, tensor, ((NSUInteger)tokens + 31) / 32, (NSUInteger)heads, 128);
        return;
    }
    Pipeline mma = by_head_dim(d, c->k.mma64, c->k.mma128, c->k.mma256);
    if (c->level >= 2 && mma && group <= 8 && (NSUInteger)group * 32 <= mma.maxTotalThreadsPerThreadgroup) {
        // 8 query rows x G heads per threadgroup (the query buffer is padded to 8 rows).
        dispatch_groups(e, mma, ((NSUInteger)tokens + 7) / 8, (NSUInteger)kv, (NSUInteger)group * 32);
        return;
    }
    // Level 1: `rows` query rows x G heads per threadgroup, one SIMD group each.
    NSUInteger simd = c->k.flash.threadExecutionWidth; // 32 on Apple GPUs
    NSUInteger rows = MAX((NSUInteger)1, MIN((NSUInteger)tokens, c->k.flash.maxTotalThreadsPerThreadgroup / (simd * group)));
    rows = MIN(rows, (NSUInteger)8);
    dispatch_groups(e, c->k.flash, ((NSUInteger)tokens + rows - 1) / rows, (NSUInteger)kv, rows * group * simd);
}

// Projection Y[batch, rows] = X . W^T; `element` is 2 for BF16 W, 4 for f32.
static int run_matmul(Context* c, float* y, const float* m, const float* x, int rows, int cols, int batch, size_t element) {
    if (rows < 1 || cols < 1 || batch < 1) return fail(c, "Metal GEMV shape", -1);
    NSUInteger oy = 0, om = 0, ox = 0;
    id<MTLBuffer> by = resolve(c, y, (size_t)rows * (size_t)batch * sizeof(float), &oy);
    id<MTLBuffer> bm = resolve(c, m, (size_t)rows * (size_t)cols * element, &om);
    id<MTLBuffer> bx = resolve(c, x, (size_t)cols * (size_t)batch * sizeof(float), &ox);
    if (!by || !bm || !bx) return fail(c, "Metal GEMV bounds", -1);
    bool bf16 = element == sizeof(uint16_t);
    Scalars s{};
    s.n[0] = rows;
    s.n[1] = cols;
    // Vector loads need 4-element rows and aligned bases (float4: 16B, ushort4: 8B).
    s.n[2] = (cols % 4 == 0 && om % (4 * element) == 0 && ox % 16 == 0) ? 1 : 0;
    s.n[3] = batch;
    id<MTLComputeCommandEncoder> e = encoder(c);
    [e setBuffer:by offset:oy atIndex:0];
    [e setBuffer:bm offset:om atIndex:1];
    [e setBuffer:bx offset:ox atIndex:2];
    set_scalars(e, s);
    bool chunk = batch >= 8, small = rows <= 16;
    if (chunk && !small && c->level >= 3 && c->k.tensor) {
        // Level 3 (ADR 0016): 32 tokens (128 relaxed) x 64 rows per threadgroup of 4 SIMD groups.
        Pipeline p = c->matmul_relaxed ? (bf16 ? c->k.tensor_relaxed_bf16 : c->k.tensor_relaxed)
                                       : (bf16 ? c->k.tensor_bf16 : c->k.tensor);
        NSUInteger tm = c->matmul_relaxed ? 128 : 32;
        dispatch_groups(e, p, ((NSUInteger)rows + 63) / 64, ((NSUInteger)batch + tm - 1) / tm, 4 * p.threadExecutionWidth);
        return 0;
    }
    if (chunk && c->level >= 2) {
        // Level 2: 64 x 64 tiles (16 x 64 for <= 16 rows), 4 SIMD groups.
        Pipeline p = small ? (bf16 ? c->k.tiled16_bf16 : c->k.tiled16) : (bf16 ? c->k.tiled_bf16 : c->k.tiled);
        NSUInteger bm_rows = small ? 16 : 64;
        dispatch_groups(e, p, ((NSUInteger)rows + bm_rows - 1) / bm_rows, ((NSUInteger)batch + 63) / 64, 128);
        return 0;
    }
    // Level 1 and every decode GEMV: one SIMD group per row, 8 rows per threadgroup.
    dispatch_groups(e, bf16 ? c->k.gemv_bf16 : c->k.gemv, ((NSUInteger)rows + 7) / 8, 1, 8 * 32);
    return 0;
}

// ---------------------------------------------------------------- the ABI --

extern "C" {
int qr_abi() { return 3; }
int qr_kind() { return 2; }
const char* qr_error(Context* c) { return c->error; }
const char* qr_info(Context* c) { return c->info; }

// Library 1 (levels 1-2) is required; library 2 (level 3) is optional.
static bool load_kernels(Context* c) {
    MTLCompileOptions* options = [MTLCompileOptions new];
    // Correctness first: IEEE-conforming math, precise exp/sin/pow (macOS 15+).
    options.mathMode = MTLMathModeSafe;
    options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
    NSError* error = nil;
    id<MTLLibrary> library = [c->device newLibraryWithSource:@(kKernelSource) options:options error:&error];
    if (!library) {
        std::fprintf(stderr, "Metal kernel compile failed: %s\n", error.localizedDescription.UTF8String);
        return false;
    }
    bool ok = true;
    // ARC: the slots are strong references, so pass their addresses as __strong.
    auto load = [&](id<MTLLibrary> lib, const char* name, __strong Pipeline* out, bool required) {
        id<MTLFunction> f = [lib newFunctionWithName:@(name)];
        NSError* e = nil;
        *out = f ? [c->device newComputePipelineStateWithFunction:f error:&e] : nil;
        if (!*out && required) {
            std::fprintf(stderr, "Metal pipeline %s failed: %s\n", name, e ? e.localizedDescription.UTF8String : "missing");
            ok = false;
        }
    };
    Kernels& k = c->k;
    for (int op = 1; op < OpCount; ++op)
        if (kLevel1[op]) load(library, kLevel1[op], &k.op[op], true);
    const struct { const char* name; __strong Pipeline* out; } required[] = {
        {"copy_floats", &k.copy}, {"rms_norm_group", &k.norm_group},
        {"delta_rule_split_64", &k.delta_split64}, {"delta_rule_split_128", &k.delta_split128},
        {"attention_scores", &k.scores}, {"attention_softmax", &k.softmax}, {"attention_values", &k.values},
        {"attention_split_grp_64", &k.split_grp64}, {"attention_split_grp_128", &k.split_grp128},
        {"attention_split_grp_256", &k.split_grp256}, {"attention_combine", &k.combine},
        {"attention_flash", &k.flash},
        {"attention_mma_64", &k.mma64}, {"attention_mma_128", &k.mma128}, {"attention_mma_256", &k.mma256},
        {"gemv", &k.gemv}, {"gemv_bf16", &k.gemv_bf16},
        {"gemm_tiled_f32", &k.tiled}, {"gemm_tiled_bf16", &k.tiled_bf16},
        {"gemm_tiled16_f32", &k.tiled16}, {"gemm_tiled16_bf16", &k.tiled16_bf16},
    };
    for (auto& r : required) load(library, r.name, r.out, true);
    // The kernels assume 32-lane SIMD groups and 128-thread tiles.
    if (!ok || k.gemv.threadExecutionWidth != 32 || k.flash.threadExecutionWidth != 32 ||
        k.tiled.maxTotalThreadsPerThreadgroup < 128 || k.tiled_bf16.maxTotalThreadsPerThreadgroup < 128 ||
        k.tiled16.maxTotalThreadsPerThreadgroup < 128 || k.tiled16_bf16.maxTotalThreadsPerThreadgroup < 128)
        return false;
    // Level 3: all or nothing; any failure keeps levels 1-2 (QR_METAL_VERBOSE=1 prints why).
    if (@available(macOS 26.0, *)) {
        MTLCompileOptions* t = [MTLCompileOptions new];
        t.languageVersion = MTLLanguageVersion4_0;
        t.mathMode = MTLMathModeSafe;
        NSError* te = nil;
        id<MTLLibrary> tl = [c->device newLibraryWithSource:@(kTensorKernelSource) options:t error:&te];
        if (!tl && std::getenv("QR_METAL_VERBOSE"))
            std::fprintf(stderr, "Metal tensor kernels unavailable: %s\n", te.localizedDescription.UTF8String);
        if (tl) {
            Kernels t3;
            load(tl, "gemm_tensor_f32", &t3.tensor, false);
            load(tl, "gemm_tensor_bf16", &t3.tensor_bf16, false);
            load(tl, "gemm_tensor_f32_relaxed", &t3.tensor_relaxed, false);
            load(tl, "gemm_tensor_bf16_relaxed", &t3.tensor_relaxed_bf16, false);
            load(tl, "attention_tensor_64", &t3.att_tensor64, false);
            load(tl, "attention_tensor_128", &t3.att_tensor128, false);
            load(tl, "attention_tensor_256", &t3.att_tensor256, false);
            Pipeline all[] = {t3.tensor, t3.tensor_bf16, t3.tensor_relaxed, t3.tensor_relaxed_bf16,
                              t3.att_tensor64, t3.att_tensor128, t3.att_tensor256};
            bool usable = true;
            for (Pipeline p : all) usable = usable && p && p.threadExecutionWidth == 32 && 128 <= p.maxTotalThreadsPerThreadgroup;
            if (usable) {
                k.tensor = t3.tensor; k.tensor_bf16 = t3.tensor_bf16;
                k.tensor_relaxed = t3.tensor_relaxed; k.tensor_relaxed_bf16 = t3.tensor_relaxed_bf16;
                k.att_tensor64 = t3.att_tensor64; k.att_tensor128 = t3.att_tensor128; k.att_tensor256 = t3.att_tensor256;
                c->max_level = 3;
            }
        }
    }
    return true;
}

int qr_create(Context** out) {
    @autoreleasepool {
        *out = nullptr;
        Context* c = new (std::nothrow) Context;
        if (!c) return -1;
        c->device = MTLCreateSystemDefaultDevice();
        if (!c->device) { delete c; return -2; }
        c->queue = [c->device newCommandQueue];
        if (!c->queue) { delete c; return -3; }
        if (!load_kernels(c)) { delete c; return -4; }
        NSOperatingSystemVersion v = NSProcessInfo.processInfo.operatingSystemVersion;
        std::snprintf(c->info, sizeof(c->info),
            "device=%s unified_memory=%d max_buffer_bytes=%llu recommended_working_set_bytes=%llu "
            "macos=%ld.%ld.%ld simd_width=%lu kernels=runtime-compiled math=safe kernel_level_max=%d",
            c->device.name.UTF8String, (int)c->device.hasUnifiedMemory,
            (unsigned long long)c->device.maxBufferLength,
            (unsigned long long)c->device.recommendedMaxWorkingSetSize,
            (long)v.majorVersion, (long)v.minorVersion, (long)v.patchVersion,
            (unsigned long)c->k.gemv.threadExecutionWidth, c->max_level);
        *out = c;
        return 0;
    }
}
// Optional symbol (no ABI bump): engine options.
//   "kernel_level"   1 reference, 2 simdgroup, 3 tensor ops (default; a level
//                    above what is available falls back per op).
//   "matmul_relaxed" 1 = level-3 GEMMs with relaxed precision (opt-in, changes
//                    results, ADR 0016); needs level 3 available and allowed.
// Other names and values fail, so a request is never silently ignored
// ("attention_f16" is the HIP bridge's, ADR 0014).
int qr_set_option(Context* c, const char* name, int value) {
    if (std::strcmp(name, "kernel_level") == 0) {
        if (value < 1 || value > 3) return fail(c, "kernel_level must be 1, 2 or 3", -1);
        if (c->matmul_relaxed && value < 3) return fail(c, "matmul_relaxed needs kernel level 3", -1);
        c->level = value;
        return 0;
    }
    if (std::strcmp(name, "matmul_relaxed") == 0) {
        if (value != 0 && value != 1) return fail(c, "matmul_relaxed must be 0 or 1", -1);
        if (value && (c->max_level < 3 || c->level < 3))
            return fail(c, "matmul_relaxed needs Metal 4 tensor ops (kernel level 3; macOS 26+)", -1);
        c->matmul_relaxed = value == 1;
        return 0;
    }
    return fail(c, "unknown Metal bridge option (16-bit attention is HIP only)", -1);
}
int qr_sync(Context* c) {
    @autoreleasepool { return flush(c); }
}
void qr_destroy(Context* c) {
    @autoreleasepool {
        flush(c);
        delete c; // ARC releases every remaining buffer, pipeline and the queue
    }
}
int qr_alloc(Context* c, float** p, size_t n) {
    @autoreleasepool {
        *p = nullptr;
        if (n == 0 || n > c->device.maxBufferLength / sizeof(float)) return fail(c, "Metal allocation size", -1);
        id<MTLBuffer> b = [c->device newBufferWithLength:n * sizeof(float) options:MTLResourceStorageModeShared];
        if (!b) return fail(c, "newBufferWithLength (out of memory?)", -1);
        std::memset(b.contents, 0, n * sizeof(float)); // fresh buffer: no GPU user yet
        *p = static_cast<float*>(b.contents);
        c->buffers[reinterpret_cast<uintptr_t>(*p)] = b;
        return 0;
    }
}
// Optional (not in the v3 symbol set): use page-aligned host memory as a GPU
// buffer without copying. Apple GPUs share memory with the CPU, so the BF16
// weights the Rust side already holds are read in place. `bytes` must be a
// whole number of pages; the caller keeps the memory alive (and unchanged)
// until qr_free, which releases only the MTLBuffer, never the memory.
int qr_wrap(Context* c, float** p, void* host, size_t bytes) {
    @autoreleasepool {
        *p = nullptr;
        if (!host || bytes == 0 || bytes > c->device.maxBufferLength) return fail(c, "Metal wrap size", -1);
        id<MTLBuffer> b = [c->device newBufferWithBytesNoCopy:host
                                                       length:bytes
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:nil];
        if (!b) return fail(c, "newBufferWithBytesNoCopy (page-aligned host memory?)", -1);
        *p = static_cast<float*>(b.contents);
        c->buffers[reinterpret_cast<uintptr_t>(*p)] = b;
        return 0;
    }
}
void qr_free(Context* c, float* p) {
    @autoreleasepool {
        // A pending command buffer retains the MTLBuffer, so erasing is safe.
        c->buffers.erase(reinterpret_cast<uintptr_t>(p));
    }
}
int qr_write(Context* c, float* p, const float* host, size_t n) {
    @autoreleasepool {
        NSUInteger offset = 0;
        if (!resolve(c, p, n * sizeof(float), &offset)) return fail(c, "Metal write bounds", -1);
        if (int s = flush(c)) return s; // never race a queued kernel touching p
        std::memcpy(p, host, n * sizeof(float));
        return 0;
    }
}
int qr_read(Context* c, float* host, const float* p, size_t n) {
    @autoreleasepool {
        NSUInteger offset = 0;
        if (!resolve(c, p, n * sizeof(float), &offset)) return fail(c, "Metal read bounds", -1);
        if (int s = flush(c)) return s;
        std::memcpy(host, p, n * sizeof(float));
        return 0;
    }
}
int qr_copy(Context* c, float* dest, const float* src, size_t n) {
    @autoreleasepool {
        NSUInteger od = 0, os = 0;
        id<MTLBuffer> d = resolve(c, dest, n * sizeof(float), &od);
        id<MTLBuffer> s = resolve(c, src, n * sizeof(float), &os);
        if (!d || !s || n == 0 || n > 0x7fffffff) return fail(c, "Metal copy bounds", -1);
        Scalars a{};
        a.count = (int)n;
        id<MTLComputeCommandEncoder> e = encoder(c);
        [e setBuffer:d offset:od atIndex:0];
        [e setBuffer:s offset:os atIndex:1];
        set_scalars(e, a);
        dispatch_threads(e, c->k.copy, n);
        return 0;
    }
}
int qr_run(Context* c, int op, int count, const Params* a) {
    @autoreleasepool {
        if (op < 1 || op >= OpCount || count < 1) return fail(c, "invalid kernel", -1);
        id<MTLComputeCommandEncoder> e = encoder(c);
        for (int i = 0; i < kSlots[op]; ++i) {
            NSUInteger offset = 0;
            // Extents are validated by the Rust caller's shapes; here we only
            // require the pointer to belong to one of this context's buffers.
            id<MTLBuffer> b = a->p[i] ? resolve(c, a->p[i], sizeof(float), &offset) : nil;
            if (!b) return fail(c, "Metal kernel argument is not a device buffer", -1);
            [e setBuffer:b offset:offset atIndex:i];
        }
        Scalars s{};
        std::memcpy(s.n, a->n, sizeof(s.n));
        std::memcpy(s.f, a->f, sizeof(s.f));
        s.count = count;
        switch (op) {
        case RmsNorm: run_rms_norm(c, e, a, s, count); return 0;
        case DeltaRule: run_delta_rule(c, e, a, s, count); return 0;
        case AttentionRead: {
            // count == heads; n5 = T query rows (0 = one); n6 = per-head scratch floats.
            int tokens = a->n[5] > 0 ? a->n[5] : 1;
            if (a->n[1] != count || a->n[2] < 1 || a->n[1] % a->n[2] != 0 ||
                (tokens > 1 && a->n[0] > 512) || a->n[3] < 1 || a->n[3] + tokens - 1 > a->n[4] ||
                (a->n[6] != 0 && a->n[6] < a->n[4]))
                return fail(c, "attention shape", -1);
            if (tokens > 1) run_attention_prefill(c, e, a, s);
            else run_attention_decode(c, e, a, s);
            return 0;
        }
        default: // per-item ops: one level
            set_scalars(e, s);
            dispatch_threads(e, c->k.op[op], (NSUInteger)count);
            return 0;
        }
    }
}
int qr_gemv(Context* c, float* y, const float* m, const float* x, int rows, int cols) {
    @autoreleasepool { return run_matmul(c, y, m, x, rows, cols, 1, sizeof(float)); }
}
// ABI v2: Y[batch,rows] = X[batch,cols] . A^T.
int qr_gemm(Context* c, float* y, const float* m, const float* x, int rows, int cols, int batch) {
    @autoreleasepool { return run_matmul(c, y, m, x, rows, cols, batch, sizeof(float)); }
}
// ABI v2: m holds rows*cols uint16 BF16 values (allocated/written as packed words).
int qr_gemv_bf16(Context* c, float* y, const float* m, const float* x, int rows, int cols) {
    @autoreleasepool { return run_matmul(c, y, m, x, rows, cols, 1, sizeof(uint16_t)); }
}
int qr_gemm_bf16(Context* c, float* y, const float* m, const float* x, int rows, int cols, int batch) {
    @autoreleasepool { return run_matmul(c, y, m, x, rows, cols, batch, sizeof(uint16_t)); }
}
}
