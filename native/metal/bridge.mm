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
          AttentionQuery=8, AttentionKv=9, AttentionRead=10, OpCount=11 };
struct Params { float* p[8]; int n[8]; float f[4]; };
struct Scalars { int n[8]; float f[4]; int count; }; // == kernels.metal
// Number of buffer slots each op binds; must equal its kernel signature.
static const int kSlots[OpCount] = {0, 3, 3, 3, 4, 3, 6, 7, 4, 5, 6};
static const char* kNames[OpCount] = {nullptr, "rms_norm", "add", "silu_product", "conv_silu",
    "delta_qk", "delta_gates", "delta_rule", "attention_query", "attention_kv", "attention_read"};

struct Context {
    char error[512] = {};
    char info[512] = {};
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> ops[OpCount] = {};
    id<MTLComputePipelineState> copy = nil, gemv = nil;
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
static void dispatch_threads(id<MTLComputeCommandEncoder> e, id<MTLComputePipelineState> p, int count) {
    NSUInteger width = MIN(p.maxTotalThreadsPerThreadgroup, (NSUInteger)256);
    width = MIN(width, (NSUInteger)count);
    [e setComputePipelineState:p];
    [e dispatchThreads:MTLSizeMake((NSUInteger)count, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
}

extern "C" {
int qr_abi() { return 1; }
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
            if (!(c->ops[op] = pipeline(kNames[op]))) { delete c; return -5; }
        c->copy = pipeline("copy_floats");
        c->gemv = pipeline("gemv");
        if (!c->copy || !c->gemv || c->gemv.threadExecutionWidth != 32) { delete c; return -6; }
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
        [e setBytes:&s length:sizeof(s) atIndex:8];
        dispatch_threads(e, c->ops[op], count);
        return 0;
    }
}
int qr_gemv(Context* c, float* y, const float* m, const float* x, int rows, int cols) {
    @autoreleasepool {
        if (rows < 1 || cols < 1) return fail(c, "Metal GEMV shape", -1);
        NSUInteger oy = 0, om = 0, ox = 0;
        id<MTLBuffer> by = resolve(c, y, (size_t)rows * sizeof(float), &oy);
        id<MTLBuffer> bm = resolve(c, m, (size_t)rows * (size_t)cols * sizeof(float), &om);
        id<MTLBuffer> bx = resolve(c, x, (size_t)cols * sizeof(float), &ox);
        if (!by || !bm || !bx) return fail(c, "Metal GEMV bounds", -1);
        Scalars s{};
        s.n[0] = rows;
        s.n[1] = cols;
        s.n[2] = (cols % 4 == 0 && om % 16 == 0 && ox % 16 == 0) ? 1 : 0;
        const NSUInteger rows_per_group = 8; // 8 SIMD groups x 32 lanes
        id<MTLComputeCommandEncoder> e = encoder(c);
        [e setComputePipelineState:c->gemv];
        [e setBuffer:by offset:oy atIndex:0];
        [e setBuffer:bm offset:om atIndex:1];
        [e setBuffer:bx offset:ox atIndex:2];
        [e setBytes:&s length:sizeof(s) atIndex:8];
        [e dispatchThreadgroups:MTLSizeMake(((NSUInteger)rows + rows_per_group - 1) / rows_per_group, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(rows_per_group * 32, 1, 1)];
        return 0;
    }
}
}
