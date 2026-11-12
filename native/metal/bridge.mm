// Apache-2.0. Metal implementation of the versioned C ABI in src/device.rs
// (same symbols as native/rocm/bridge.cpp). Build: tools/build-metal.sh.
//
// Apple GPUs share memory with the CPU, so every allocation is a
// MTLStorageModeShared buffer and the pointer handed to Rust is its CPU
// `contents` address. Rust may offset such pointers (embedding row lookup);
// `resolve` maps any interior pointer back to (MTLBuffer, byte offset).
// Rust never dereferences them: reads/writes go through qr_read/qr_write,
// which first wait for every queued GPU command.
//
// All work is encoded into ONE compute encoder of ONE command buffer and is
// only committed at a synchronization point (read, write, sync). The default
// serial dispatch mode plus tracked hazards order every kernel after the
// previous one, matching the single HIP stream of the ROCm bridge.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstring>
#include <map>
#include <new>

static const char* kKernelSource =
#include "kernels.inc"
    ;

enum Op { RmsNorm=1, Add=2, SiluProduct=3, ConvSilu=4, DeltaQk=5, DeltaGates=6, DeltaRule=7,
          AttentionQuery=8, AttentionKv=9, AttentionRead=10, EmbedBf16=11, OpCount=12 };
struct Params { float* p[8]; int n[8]; float f[4]; };
struct Scalars { int n[8]; float f[4]; int count; }; // == kernels.metal
// Number of buffer slots each op binds; must equal its kernel signature.
static const int kSlots[OpCount] = {0, 3, 3, 3, 4, 3, 6, 7, 4, 5, 6, 2};
// AttentionRead has no single kernel: see attention_passes / kernels.metal.
static const char* kNames[OpCount] = {nullptr, "rms_norm", "add", "silu_product", "conv_silu",
    "delta_qk", "delta_gates", "delta_rule", "attention_query", "attention_kv", nullptr, "embed_bf16"};

struct Context {
    char error[512] = {};
    char info[512] = {};
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> ops[OpCount] = {};
    id<MTLComputePipelineState> copy = nil, gemv = nil, gemv_bf16 = nil, tiled = nil, tiled_bf16 = nil;
    id<MTLComputePipelineState> scores = nil, softmax = nil, values = nil, flash = nil, split = nil, combine = nil, norm = nil;
    id<MTLComputePipelineState> mma64 = nil, mma128 = nil, mma256 = nil;
    id<MTLComputePipelineState> delta64 = nil, delta128 = nil; // delta_rule_reg<key_dim>
    id<MTLCommandBuffer> pending = nil;
    id<MTLComputeCommandEncoder> encoder = nil;
    std::map<uintptr_t, id<MTLBuffer>> buffers; // keyed by contents address
};

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
static void dispatch_grid(id<MTLComputeCommandEncoder> e, id<MTLComputePipelineState> p, int x, int y) {
    NSUInteger w = MIN((NSUInteger)x, p.threadExecutionWidth);
    NSUInteger h = MAX((NSUInteger)1, MIN((NSUInteger)y, p.maxTotalThreadsPerThreadgroup / w));
    [e setComputePipelineState:p];
    [e dispatchThreads:MTLSizeMake((NSUInteger)x, (NSUInteger)y, 1) threadsPerThreadgroup:MTLSizeMake(w, h, 1)];
}
// Op 10 (one thread per head in the HIP/CPU body) as three parallel passes.
// Buffers 0..5 are already bound; every pass reads only the slots it declares.
static void attention_passes(Context* c, id<MTLComputeCommandEncoder> e, const Params* a) {
    int d = a->n[0], heads = a->n[1], length = a->n[3];
    dispatch_grid(e, c->scores, length, heads);
    NSUInteger threads = MIN(c->softmax.maxTotalThreadsPerThreadgroup, (NSUInteger)1024);
    [e setComputePipelineState:c->softmax];
    [e dispatchThreadgroups:MTLSizeMake((NSUInteger)heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    dispatch_grid(e, c->values, d, heads);
}
// Op 10 for decode (T = 1) with per-head scratch n6: split the L positions
// into blocks of B (>= kSplitSpan, grown so splits*(d+2) fits the scratch),
// one threadgroup per (block, KV head), then merge with attention_combine.
static const int kSplitSpan = 32; // == ATTENTION_SPLIT in src/device.rs
static void attention_split(Context* c, id<MTLComputeCommandEncoder> e, const Params* a, Scalars s) {
    int d = a->n[0], heads = a->n[1], kv = a->n[2], length = a->n[3], row = a->n[6];
    int fit = row / (d + 2); // splits the scratch row can hold (>= 1, checked by caller)
    int span = MAX(kSplitSpan, (length + fit - 1) / fit);
    int splits = (length + span - 1) / span;
    s.n[7] = span;
    [e setBytes:&s length:sizeof(s) atIndex:8];
    [e setComputePipelineState:c->split];
    [e dispatchThreadgroups:MTLSizeMake((NSUInteger)splits, (NSUInteger)kv, 1)
      threadsPerThreadgroup:MTLSizeMake((NSUInteger)(heads / kv) * 32, 1, 1)];
    s.n[5] = splits;
    [e setBytes:&s length:sizeof(s) atIndex:8];
    dispatch_grid(e, c->combine, d, heads);
}
// Op 10 with n5 = T > 1 (prefill chunk): one online-softmax pass, see
// attention_flash. Threadgroup = `rows` rows x G heads sharing a KV head.
static void attention_flash(Context* c, id<MTLComputeCommandEncoder> e, const Params* a, int tokens) {
    int d = a->n[0], heads = a->n[1], kv = a->n[2], group = heads / kv;
    id<MTLComputePipelineState> mma = d == 64 ? c->mma64 : d == 128 ? c->mma128 : d == 256 ? c->mma256 : nil;
    if (mma && group <= 8 && (NSUInteger)group * 32 <= mma.maxTotalThreadsPerThreadgroup) {
        // attention_mma: 8 query rows x G heads per threadgroup (query rows padded to 8 by Rust).
        [e setComputePipelineState:mma];
        [e dispatchThreadgroups:MTLSizeMake(((NSUInteger)tokens + 7) / 8, (NSUInteger)kv, 1)
          threadsPerThreadgroup:MTLSizeMake((NSUInteger)group * 32, 1, 1)];
        return;
    }
    NSUInteger simd = c->flash.threadExecutionWidth; // 32 on Apple GPUs
    NSUInteger rows = MAX((NSUInteger)1, MIN((NSUInteger)tokens, c->flash.maxTotalThreadsPerThreadgroup / (simd * group)));
    rows = MIN(rows, (NSUInteger)8);
    [e setComputePipelineState:c->flash];
    [e dispatchThreadgroups:MTLSizeMake(((NSUInteger)tokens + rows - 1) / rows, (NSUInteger)kv, 1)
      threadsPerThreadgroup:MTLSizeMake(rows * group * simd, 1, 1)];
}
static void dispatch_threads(id<MTLComputeCommandEncoder> e, id<MTLComputePipelineState> p, int count) {
    NSUInteger width = MIN(p.maxTotalThreadsPerThreadgroup, (NSUInteger)256);
    width = MIN(width, (NSUInteger)count);
    [e setComputePipelineState:p];
    [e dispatchThreads:MTLSizeMake((NSUInteger)count, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
}

extern "C" {
int qr_abi() { return 3; }
int qr_kind() { return 2; }
const char* qr_error(Context* c) { return c->error; }
const char* qr_info(Context* c) { return c->info; }

int qr_create(Context** out) {
    @autoreleasepool {
        *out = nullptr;
        Context* c = new (std::nothrow) Context;
        if (!c) return -1;
        c->device = MTLCreateSystemDefaultDevice();
        if (!c->device) { delete c; return -2; }
        c->queue = [c->device newCommandQueue];
        if (!c->queue) { delete c; return -3; }
        MTLCompileOptions* options = [MTLCompileOptions new];
        // Correctness first: IEEE-conforming math, precise exp/sin/pow (macOS 15+).
        options.mathMode = MTLMathModeSafe;
        options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        NSError* error = nil;
        id<MTLLibrary> library = [c->device newLibraryWithSource:@(kKernelSource) options:options error:&error];
        if (!library) {
            std::fprintf(stderr, "Metal kernel compile failed: %s\n", error.localizedDescription.UTF8String);
            delete c; return -4;
        }
        auto pipeline = [&](const char* name) -> id<MTLComputePipelineState> {
            id<MTLFunction> f = [library newFunctionWithName:@(name)];
            if (!f) return nil;
            NSError* e = nil;
            id<MTLComputePipelineState> p = [c->device newComputePipelineStateWithFunction:f error:&e];
            if (!p) std::fprintf(stderr, "Metal pipeline %s failed: %s\n", name, e.localizedDescription.UTF8String);
            return p;
        };
        for (int op = 1; op < OpCount; ++op)
            if (kNames[op] && !(c->ops[op] = pipeline(kNames[op]))) { delete c; return -5; }
        c->copy = pipeline("copy_floats");
        c->gemv = pipeline("gemv");
        c->gemv_bf16 = pipeline("gemv_bf16");
        c->tiled = pipeline("gemm_tiled_f32");
        c->tiled_bf16 = pipeline("gemm_tiled_bf16");
        c->scores = pipeline("attention_scores");
        c->softmax = pipeline("attention_softmax");
        c->values = pipeline("attention_values");
        c->flash = pipeline("attention_flash");
        c->split = pipeline("attention_split");
        c->combine = pipeline("attention_combine");
        c->norm = pipeline("rms_norm_group");
        c->mma64 = pipeline("attention_mma_64");
        c->mma128 = pipeline("attention_mma_128");
        c->mma256 = pipeline("attention_mma_256");
        c->delta64 = pipeline("delta_rule_reg_64");
        c->delta128 = pipeline("delta_rule_reg_128");
        if (!c->copy || !c->gemv || !c->gemv_bf16 || !c->tiled || !c->tiled_bf16 ||
            c->tiled.maxTotalThreadsPerThreadgroup < 128 || c->tiled_bf16.maxTotalThreadsPerThreadgroup < 128 || !c->scores || !c->softmax || !c->values || !c->flash || !c->split || !c->combine || !c->norm ||
            !c->mma64 || !c->mma128 || !c->mma256 || !c->delta64 || !c->delta128 ||
            c->gemv.threadExecutionWidth != 32 || c->flash.threadExecutionWidth != 32) { delete c; return -6; }
        NSOperatingSystemVersion v = NSProcessInfo.processInfo.operatingSystemVersion;
        std::snprintf(c->info, sizeof(c->info),
            "device=%s unified_memory=%d max_buffer_bytes=%llu recommended_working_set_bytes=%llu "
            "macos=%ld.%ld.%ld simd_width=%lu kernels=runtime-compiled math=safe",
            c->device.name.UTF8String, (int)c->device.hasUnifiedMemory,
            (unsigned long long)c->device.maxBufferLength,
            (unsigned long long)c->device.recommendedMaxWorkingSetSize,
            (long)v.majorVersion, (long)v.minorVersion, (long)v.patchVersion,
            (unsigned long)c->gemv.threadExecutionWidth);
        *out = c;
        return 0;
    }
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
        [e setBytes:&a length:sizeof(a) atIndex:8];
        dispatch_threads(e, c->copy, (int)n);
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
        if (op == AttentionRead) {
            // count == heads; shapes validated by the Rust caller.
            int tokens = a->n[5] > 0 ? a->n[5] : 1;
            if (a->n[1] != count || a->n[2] < 1 || a->n[1] % a->n[2] != 0 ||
                (tokens > 1 && a->n[0] > 512) || a->n[3] < 1 || a->n[3] + tokens - 1 > a->n[4] ||
                (a->n[6] != 0 && a->n[6] < a->n[4]))
                return fail(c, "attention shape", -1);
            if (tokens > 1) {
                [e setBytes:&s length:sizeof(s) atIndex:8];
                attention_flash(c, e, a, tokens);
            } else if (a->n[6] > 0 && a->n[0] <= 512 && a->n[6] >= a->n[0] + 2 &&
                       (NSUInteger)(a->n[1] / a->n[2]) * 32 <= c->split.maxTotalThreadsPerThreadgroup) {
                attention_split(c, e, a, s);
            } else {
                [e setBytes:&s length:sizeof(s) atIndex:8];
                attention_passes(c, e, a);
            }
        } else if (op == RmsNorm) {
            // count groups of n0 elements: one threadgroup (<= 256 threads) per group.
            [e setBytes:&s length:sizeof(s) atIndex:8];
            NSUInteger threads = MIN((NSUInteger)256, c->norm.maxTotalThreadsPerThreadgroup);
            threads = MIN(threads, ((NSUInteger)MAX(a->n[0], 1) + 31) / 32 * 32);
            [e setComputePipelineState:c->norm];
            [e dispatchThreadgroups:MTLSizeMake((NSUInteger)count, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        } else {
            [e setBytes:&s length:sizeof(s) atIndex:8];
            // DeltaNet rule: keep the state column in registers when key_dim
            // has a specialization (bit-identical to the generic kernel).
            id<MTLComputePipelineState> p = c->ops[op];
            if (op == DeltaRule && a->n[0] == 128) p = c->delta128;
            if (op == DeltaRule && a->n[0] == 64) p = c->delta64;
            dispatch_threads(e, p, count);
        }
        return 0;
    }
}
static int encode_gemv(Context* c, id<MTLComputePipelineState> p, id<MTLComputePipelineState> tiled, float* y, const float* m, const float* x,
                       int rows, int cols, int batch, size_t element) {
    if (rows < 1 || cols < 1 || batch < 1) return fail(c, "Metal GEMV shape", -1);
    NSUInteger oy = 0, om = 0, ox = 0;
    id<MTLBuffer> by = resolve(c, y, (size_t)rows * (size_t)batch * sizeof(float), &oy);
    id<MTLBuffer> bm = resolve(c, m, (size_t)rows * (size_t)cols * element, &om);
    id<MTLBuffer> bx = resolve(c, x, (size_t)cols * (size_t)batch * sizeof(float), &ox);
    if (!by || !bm || !bx) return fail(c, "Metal GEMV bounds", -1);
    Scalars s{};
    s.n[0] = rows;
    s.n[1] = cols;
    // Vector loads need 4-element rows and aligned bases (float4: 16B, ushort4: 8B).
    s.n[2] = (cols % 4 == 0 && om % (4 * element) == 0 && ox % 16 == 0) ? 1 : 0;
    s.n[3] = batch;
    id<MTLComputeCommandEncoder> e = encoder(c);
    if (batch >= 8) { // prefill chunk: 64x64 output tiles, 4 SIMD groups of 32x32 each
        [e setComputePipelineState:tiled];
        [e setBuffer:by offset:oy atIndex:0];
        [e setBuffer:bm offset:om atIndex:1];
        [e setBuffer:bx offset:ox atIndex:2];
        [e setBytes:&s length:sizeof(s) atIndex:8];
        [e dispatchThreadgroups:MTLSizeMake(((NSUInteger)rows + 63) / 64, ((NSUInteger)batch + 63) / 64, 1)
          threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        return 0;
    }
    const NSUInteger rows_per_group = 8; // 8 SIMD groups x 32 lanes
    [e setComputePipelineState:p];
    [e setBuffer:by offset:oy atIndex:0];
    [e setBuffer:bm offset:om atIndex:1];
    [e setBuffer:bx offset:ox atIndex:2];
    [e setBytes:&s length:sizeof(s) atIndex:8];
    [e dispatchThreadgroups:MTLSizeMake(((NSUInteger)rows + rows_per_group - 1) / rows_per_group, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(rows_per_group * 32, 1, 1)];
    return 0;
}
int qr_gemv(Context* c, float* y, const float* m, const float* x, int rows, int cols) {
    @autoreleasepool { return encode_gemv(c, c->gemv, c->tiled, y, m, x, rows, cols, 1, sizeof(float)); }
}
// ABI v2: Y[batch,rows] = X[batch,cols] . A^T (one weight read per 8 tokens).
int qr_gemm(Context* c, float* y, const float* m, const float* x, int rows, int cols, int batch) {
    @autoreleasepool { return encode_gemv(c, c->gemv, c->tiled, y, m, x, rows, cols, batch, sizeof(float)); }
}
// ABI v2: m holds rows*cols uint16 BF16 values (allocated/written as packed words).
int qr_gemv_bf16(Context* c, float* y, const float* m, const float* x, int rows, int cols) {
    @autoreleasepool { return encode_gemv(c, c->gemv_bf16, c->tiled_bf16, y, m, x, rows, cols, 1, sizeof(uint16_t)); }
}
int qr_gemm_bf16(Context* c, float* y, const float* m, const float* x, int rows, int cols, int batch) {
    @autoreleasepool { return encode_gemv(c, c->gemv_bf16, c->tiled_bf16, y, m, x, rows, cols, batch, sizeof(uint16_t)); }
}
}
