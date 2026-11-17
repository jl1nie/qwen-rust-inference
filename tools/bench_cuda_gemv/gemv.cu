// BF16 GEMV microbenchmark for CUDA (issue #46, book vol. 2 ch. 8).
//
//   y[rows] = W[rows, cols] x[cols]
//   W: row-major BF16 bits (uint16), x and y: FP32, products and sums in FP32.
//
// One warp computes one output row; lanes take strided parts of the row and the
// partial sums are combined with __shfl_down_sync (distances 16, 8, 4, 2, 1).
// Kernels, one change at a time from the baseline:
//
//   scalar  one 2-byte weight load per step; lane l takes j = l, l+32, ...
//   vec8    one 16-byte load (8 weights) per step; lane l takes pieces
//           p = l, l+32, ..., elements 8p..8p+7 in order
//   pf2     vec8 with 2 loads issued before their FMAs (same order as vec8)
//   pf4     vec8 with 4 loads issued before their FMAs (same order as vec8)
//
// vec8/pf* need 16-byte aligned rows (cols % 8 == 0 and an aligned base); a
// row that is not aligned falls back to the scalar order, so every input
// works. The CPU side checks each kernel two ways: bit for bit against a CPU
// emulation of its lane order and reduction tree, and against a double
// precision reference with the tolerance documented in README.md.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(call)                                                                                    \
    do {                                                                                            \
        cudaError_t e_ = (call);                                                                    \
        if (e_ != cudaSuccess) {                                                                    \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e_)); \
            std::exit(2);                                                                           \
        }                                                                                           \
    } while (0)

constexpr int kWarp = 32;
constexpr int kRowsPerBlock = 8; // 256 threads, 8 warps, 8 rows
constexpr int kThreads = kWarp * kRowsPerBlock;
constexpr double kAtol = 1e-6;   // see README.md "Tolerance"
constexpr double kRtol = 1e-5;

enum Kernel { kScalar, kVec8, kPf2, kPf4, kKernelCount };
const char* kKernelName[kKernelCount] = {"scalar", "vec8", "pf2", "pf4"};
// Elements per lane step: the lane that owns element j is (j / group) % 32.
int kernel_group(int k) { return k == kScalar ? 1 : 8; }

// ---------------------------------------------------------------- device --

__device__ __forceinline__ float bf16_to_f32(uint32_t bits16) { return __uint_as_float(bits16 << 16); }

__device__ __forceinline__ float warp_sum(float s) {
#pragma unroll
    for (int off = kWarp / 2; off > 0; off /= 2) s += __shfl_down_sync(0xffffffffu, s, off);
    return s; // complete in lane 0
}

__device__ __forceinline__ float scalar_part(const uint16_t* w, const float* x, int cols, int lane) {
    float s = 0.0f;
    for (int j = lane; j < cols; j += kWarp) s = fmaf(bf16_to_f32(w[j]), x[j], s);
    return s;
}

__device__ __forceinline__ float step8(uint4 p, const float* x, int piece, float s) {
    const float4* xv = reinterpret_cast<const float4*>(x) + 2 * piece;
    float4 lo = xv[0], hi = xv[1];
    s = fmaf(bf16_to_f32(p.x & 0xffffu), lo.x, s);
    s = fmaf(bf16_to_f32(p.x >> 16), lo.y, s);
    s = fmaf(bf16_to_f32(p.y & 0xffffu), lo.z, s);
    s = fmaf(bf16_to_f32(p.y >> 16), lo.w, s);
    s = fmaf(bf16_to_f32(p.z & 0xffffu), hi.x, s);
    s = fmaf(bf16_to_f32(p.z >> 16), hi.y, s);
    s = fmaf(bf16_to_f32(p.w & 0xffffu), hi.z, s);
    s = fmaf(bf16_to_f32(p.w >> 16), hi.w, s);
    return s;
}

// LOADS 16-byte loads in flight per lane; pieces are still consumed in the
// order p = lane, lane + 32, ..., so the sum is bitwise that of LOADS = 1.
template <int LOADS>
__device__ __forceinline__ float vec_part(const uint16_t* w, const float* x, int cols, int lane) {
    const bool aligned = (cols % 8 == 0) && (reinterpret_cast<uintptr_t>(w) % 16 == 0) &&
                         (reinterpret_cast<uintptr_t>(x) % 16 == 0);
    if (!aligned) return scalar_part(w, x, cols, lane);
    const uint4* wv = reinterpret_cast<const uint4*>(w);
    const int pieces = cols / 8;
    float s = 0.0f;
    int p = lane;
    if (LOADS > 1) {
        for (; p + (LOADS - 1) * kWarp < pieces; p += LOADS * kWarp) {
            uint4 q[LOADS];
#pragma unroll
            for (int u = 0; u < LOADS; ++u) q[u] = __ldg(wv + p + u * kWarp);
#pragma unroll
            for (int u = 0; u < LOADS; ++u) s = step8(q[u], x, p + u * kWarp, s);
        }
    }
    for (; p < pieces; p += kWarp) s = step8(__ldg(wv + p), x, p, s);
    return s;
}

template <int K>
__global__ __launch_bounds__(kThreads) void gemv_bf16(float* __restrict__ y, const uint16_t* __restrict__ w,
                                                      const float* __restrict__ x, int rows, int cols) {
    const int row = blockIdx.x * kRowsPerBlock + threadIdx.x / kWarp;
    const int lane = threadIdx.x % kWarp;
    if (row >= rows) return; // the whole warp leaves together
    const uint16_t* wr = w + size_t(row) * cols;
    float s;
    if (K == kScalar) s = scalar_part(wr, x, cols, lane);
    else if (K == kVec8) s = vec_part<1>(wr, x, cols, lane);
    else if (K == kPf2) s = vec_part<2>(wr, x, cols, lane);
    else s = vec_part<4>(wr, x, cols, lane);
    s = warp_sum(s);
    if (lane == 0) y[row] = s;
}

__global__ void empty_kernel() {}

// Read-only stream over `n16` 16-byte words (the attainable read rate, no
// math); the xor keeps the loads alive and is written only if it is magic.
__global__ __launch_bounds__(256) void read_stream(const uint4* __restrict__ a, size_t n16, unsigned* out) {
    unsigned acc = 0;
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n16; i += size_t(gridDim.x) * blockDim.x) {
        uint4 v = __ldg(a + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x9e3779b9u) out[0] = acc;
}

void launch(int k, cudaStream_t st, float* y, const uint16_t* w, const float* x, int rows, int cols) {
    dim3 grid((rows + kRowsPerBlock - 1) / kRowsPerBlock), block(kThreads);
    switch (k) {
    case kScalar: gemv_bf16<kScalar><<<grid, block, 0, st>>>(y, w, x, rows, cols); break;
    case kVec8: gemv_bf16<kVec8><<<grid, block, 0, st>>>(y, w, x, rows, cols); break;
    case kPf2: gemv_bf16<kPf2><<<grid, block, 0, st>>>(y, w, x, rows, cols); break;
    default: gemv_bf16<kPf4><<<grid, block, 0, st>>>(y, w, x, rows, cols); break;
    }
}

// ------------------------------------------------------------------ host --

float bf16_to_f32_host(uint16_t b) {
    uint32_t u = uint32_t(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
uint16_t f32_to_bf16_rne(float f) { // round to nearest even; inputs here are finite
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return uint16_t(u >> 16);
}

// SplitMix64: deterministic and identical on every platform.
struct Rng {
    uint64_t s;
    uint64_t next() {
        uint64_t z = (s += 0x9e3779b97f4a7c15ull);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }
    float uniform_pm1() { return float(int64_t(next() >> 40) - (1ll << 23)) / float(1 << 23); } // [-1, 1)
};

struct Problem {
    int rows, cols;
    std::vector<uint16_t> w;
    std::vector<float> x;
};

Problem make_problem(const char* kind, int rows, int cols, uint64_t seed) {
    Problem p{rows, cols, std::vector<uint16_t>(size_t(rows) * cols), std::vector<float>(cols)};
    if (!std::strcmp(kind, "ones")) {
        std::fill(p.w.begin(), p.w.end(), f32_to_bf16_rne(1.0f));
        std::fill(p.x.begin(), p.x.end(), 1.0f);
    } else { // signed, non-uniform; differs per row and per column
        Rng r{seed};
        for (auto& v : p.w) v = f32_to_bf16_rne(r.uniform_pm1());
        for (auto& v : p.x) v = r.uniform_pm1();
    }
    return p;
}

// Bit-exact CPU emulation of a kernel's lane order and shfl_down tree.
float emulate_row(const uint16_t* w, const float* x, int cols, int group) {
    float lane_sum[kWarp] = {};
    for (int j = 0; j < cols; ++j) {
        int lane = (j / group) % kWarp;
        lane_sum[lane] = std::fmaf(bf16_to_f32_host(w[j]), x[j], lane_sum[lane]);
    }
    for (int off = kWarp / 2; off > 0; off /= 2)
        for (int l = 0; l < kWarp; ++l) // __shfl_down: lanes past the end read their own value
            lane_sum[l] = lane_sum[l] + (l + off < kWarp ? lane_sum[l + off] : lane_sum[l]);
    return lane_sum[0];
}

struct CheckResult {
    bool pass = true, bitwise = true, sentinel_ok = true;
    double max_abs = 0, max_scaled = 0; // scaled = |d| / (atol + rtol * sum|w x|)
    int worst_row = -1, mismatches = 0, first_mismatch_row = -1;
    float worst_gpu = 0;
    double worst_ref = 0;
};

// Runs kernel k once and checks it. elem_offset shifts the weight base by that
// many BF16 elements (2 bytes each) to exercise the unaligned fallback.
CheckResult check(int k, const Problem& p, int elem_offset) {
    const size_t n = p.w.size();
    const int pad = 64; // sentinel slots after y[rows - 1]
    uint16_t* dw;
    float *dx, *dy;
    CK(cudaMalloc(&dw, (n + elem_offset) * 2));
    CK(cudaMalloc(&dx, p.cols * 4 + 16));
    CK(cudaMalloc(&dy, (p.rows + pad) * 4));
    CK(cudaMemcpy(dw + elem_offset, p.w.data(), n * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx, p.x.data(), p.cols * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(dy, 0xff, (p.rows + pad) * 4)); // NaN pattern
    launch(k, 0, dy, dw + elem_offset, dx, p.rows, p.cols);
    CK(cudaGetLastError());
    std::vector<float> y(p.rows + pad);
    CK(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaFree(dw));
    CK(cudaFree(dx));
    CK(cudaFree(dy));

    const bool aligned = p.cols % 8 == 0 && elem_offset % 8 == 0;
    const int group = aligned ? kernel_group(k) : 1;
    CheckResult r;
    for (int i = 0; i < pad; ++i) {
        uint32_t u;
        std::memcpy(&u, &y[p.rows + i], 4);
        if (u != 0xffffffffu) r.sentinel_ok = false;
    }
    for (int row = 0; row < p.rows; ++row) {
        const uint16_t* wr = p.w.data() + size_t(row) * p.cols;
        double ref = 0, mag = 0;
        for (int j = 0; j < p.cols; ++j) {
            double t = double(bf16_to_f32_host(wr[j])) * double(p.x[j]);
            ref += t;
            mag += std::fabs(t);
        }
        float emu = emulate_row(wr, p.x.data(), p.cols, group);
        if (std::memcmp(&emu, &y[row], 4) != 0) {
            if (r.first_mismatch_row < 0) r.first_mismatch_row = row;
            ++r.mismatches;
            r.bitwise = false;
        }
        double d = std::fabs(double(y[row]) - ref);
        double scaled = d / (kAtol + kRtol * mag);
        if (!(scaled <= 1.0)) r.pass = false; // also catches NaN
        if (!(scaled <= r.max_scaled) || r.worst_row < 0) {
            r.max_scaled = scaled;
            r.worst_row = row;
            r.worst_gpu = y[row];
            r.worst_ref = ref;
        }
        if (!(d <= r.max_abs)) r.max_abs = d;
    }
    r.pass = r.pass && r.sentinel_ok;
    return r;
}

// ---------------------------------------------------------------- timing --

struct Stats {
    double median, lo, hi; // microseconds per launch over samples
};
Stats summarize(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t m = v.size();
    double med = m % 2 ? v[m / 2] : 0.5 * (v[m / 2 - 1] + v[m / 2]);
    return {med, v.front(), v.back()};
}

struct Options {
    int rows = 3584, cols = 1024;
    int warmup = 50, iters = 200, samples = 15;
    size_t cold_bytes = 0; // 0: max(8 x L2, 256 MiB)
    std::string json, driver = "unknown", toolkit = "unknown", build = "unknown";
};

cudaStream_t g_stream; // all timing runs on this one (non-default, capturable) stream

// Runs `iters` enqueue() calls between one event pair, `samples` times, after
// `warmup` calls; returns microseconds per call. With `graph`, the `iters`
// calls are captured once into a CUDA Graph and each sample replays it, so
// the per-launch host/driver cost (about 9 us on WSL2) leaves the kernel time.
template <class F>
Stats time_events(const Options& o, bool graph, F enqueue) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    cudaGraphExec_t exec = nullptr;
    if (graph) {
        cudaGraph_t g;
        CK(cudaStreamBeginCapture(g_stream, cudaStreamCaptureModeThreadLocal));
        for (int i = 0; i < o.iters; ++i) enqueue();
        CK(cudaStreamEndCapture(g_stream, &g));
        CK(cudaGraphInstantiate(&exec, g, 0));
        CK(cudaGraphDestroy(g));
    }
    auto run = [&]() {
        if (graph) CK(cudaGraphLaunch(exec, g_stream));
        else for (int i = 0; i < o.iters; ++i) enqueue();
    };
    for (int i = 0; i < (o.warmup + o.iters - 1) / o.iters; ++i) run();
    std::vector<double> us;
    for (int s = 0; s < o.samples; ++s) {
        CK(cudaEventRecord(a, g_stream));
        run();
        CK(cudaEventRecord(b, g_stream));
        CK(cudaEventSynchronize(b));
        float ms;
        CK(cudaEventElapsedTime(&ms, a, b));
        us.push_back(1000.0 * ms / o.iters);
    }
    CK(cudaGetLastError());
    if (exec) CK(cudaGraphExecDestroy(exec));
    CK(cudaEventDestroy(a));
    CK(cudaEventDestroy(b));
    return summarize(us);
}

// GEMV time per launch; `copies` > 1 cycles distinct weight buffers.
Stats time_device(int k, const Options& o, bool graph, uint16_t* dw, size_t copies, const float* dx, float* dy) {
    const size_t elems = size_t(o.rows) * o.cols;
    size_t c = 0;
    return time_events(o, graph, [&]() {
        launch(k, g_stream, dy, dw + (c++ % copies) * elems, dx, o.rows, o.cols);
    });
}

// Host-side costs (separate from the kernel time above):
//   enqueue: host time to issue `iters` launches without waiting, per launch
//   roundtrip: one launch followed by cudaStreamSynchronize, per call
struct HostStats {
    Stats enqueue, roundtrip;
};
HostStats time_host(int k, const Options& o, const uint16_t* dw, const float* dx, float* dy) {
    using clk = std::chrono::steady_clock;
    for (int i = 0; i < o.warmup; ++i) launch(k, g_stream, dy, dw, dx, o.rows, o.cols);
    CK(cudaDeviceSynchronize());
    std::vector<double> enq, rt;
    for (int s = 0; s < o.samples; ++s) {
        auto t0 = clk::now();
        for (int i = 0; i < o.iters; ++i) launch(k, g_stream, dy, dw, dx, o.rows, o.cols);
        auto t1 = clk::now();
        CK(cudaDeviceSynchronize());
        enq.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() / o.iters);
        std::vector<double> one;
        for (int i = 0; i < o.iters; ++i) {
            auto u0 = clk::now();
            launch(k, g_stream, dy, dw, dx, o.rows, o.cols);
            CK(cudaStreamSynchronize(g_stream));
            one.push_back(std::chrono::duration<double, std::micro>(clk::now() - u0).count());
        }
        rt.push_back(summarize(one).median);
    }
    return {summarize(enq), summarize(rt)};
}

double empty_launch_us(const Options& o, bool graph) {
    return time_events(o, graph, [&]() { empty_kernel<<<1, 32, 0, g_stream>>>(); }).median;
}

// Bytes per second of read_stream over the whole cold buffer, one launch per
// sample (the buffer is far larger than L2).
Stats time_read(const Options& o, const uint16_t* dw, size_t bytes, int sms, unsigned* scratch) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    const size_t n16 = bytes / 16;
    const int blocks = sms * 16;
    std::vector<double> gbs;
    for (int s = 0; s < 3 + o.samples; ++s) {
        CK(cudaEventRecord(a, g_stream));
        read_stream<<<blocks, 256, 0, g_stream>>>(reinterpret_cast<const uint4*>(dw), n16, scratch);
        CK(cudaEventRecord(b, g_stream));
        CK(cudaEventSynchronize(b));
        float ms;
        CK(cudaEventElapsedTime(&ms, a, b));
        if (s >= 3) gbs.push_back(double(n16 * 16) / (ms * 1e-3) / 1e9);
    }
    CK(cudaGetLastError());
    CK(cudaEventDestroy(a));
    CK(cudaEventDestroy(b));
    return summarize(gbs);
}

// ------------------------------------------------------------------ main --

std::string json_escape(const std::string& s) {
    std::string r;
    for (char ch : s) {
        if (ch == '"' || ch == '\\') r += '\\';
        if (ch == '\n') { r += "\\n"; continue; }
        r += ch;
    }
    return r;
}

void usage() {
    std::fprintf(stderr,
                 "usage: bench_cuda_gemv [--rows N] [--cols N] [--warmup N] [--iters N] [--samples N]\n"
                 "                       [--cold-bytes N] [--json PATH] [--driver S] [--toolkit S] [--build S]\n"
                 "       bench_cuda_gemv --check-only\n");
    std::exit(1);
}

int main(int argc, char** argv) {
    Options o;
    bool check_only = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&]() -> const char* {
            if (i + 1 >= argc) usage();
            return argv[++i];
        };
        if (a == "--rows") o.rows = std::atoi(val());
        else if (a == "--cols") o.cols = std::atoi(val());
        else if (a == "--warmup") o.warmup = std::atoi(val());
        else if (a == "--iters") o.iters = std::atoi(val());
        else if (a == "--samples") o.samples = std::atoi(val());
        else if (a == "--cold-bytes") o.cold_bytes = std::strtoull(val(), nullptr, 10);
        else if (a == "--json") o.json = val();
        else if (a == "--driver") o.driver = val();
        else if (a == "--toolkit") o.toolkit = val();
        else if (a == "--build") o.build = val();
        else if (a == "--check-only") check_only = true;
        else usage();
    }
    if (o.rows <= 0 || o.cols <= 0 || o.iters <= 0 || o.samples <= 0) usage();

    int dev = 0, driver_api = 0, runtime_api = 0, l2 = 0, sms = 0, bus = 0, cc_major = 0, cc_minor = 0;
    CK(cudaSetDevice(dev));
    CK(cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking));
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, dev));
    CK(cudaDriverGetVersion(&driver_api));
    CK(cudaRuntimeGetVersion(&runtime_api));
    CK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev));
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    CK(cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, dev));
    CK(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, dev));
    CK(cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, dev));
    std::printf("device: %s  sm_%d%d  SMs %d  L2 %d bytes  bus %d bit  memory %zu bytes\n", prop.name, cc_major,
                cc_minor, sms, l2, bus, prop.totalGlobalMem);
    std::printf("cuda driver API %d, runtime %d\n\n", driver_api, runtime_api);

    // ---- correctness
    struct Case {
        const char* name;
        const char* kind;
        int rows, cols, offset;
    };
    const Case cases[] = {
        {"ones 3584x1024", "ones", 3584, 1024, 0},
        {"signed 3584x1024", "signed", 3584, 1024, 0},
        {"signed 1024x3584", "signed", 1024, 3584, 0},
        {"signed 37x45 (cols % 32, % 8 != 0)", "signed", 37, 45, 0},
        {"signed 13x100 (cols % 8 = 4)", "signed", 13, 100, 0},
        {"signed 29x1000 (cols % 256 != 0)", "signed", 29, 1000, 0},
        {"signed 5x8 (one piece)", "signed", 5, 8, 0},
        {"signed 3x1", "signed", 3, 1, 0},
        {"signed 3584x1024, base +2 bytes", "signed", 3584, 1024, 1},
        {"signed 64x1024, base +8 bytes", "signed", 64, 1024, 4},
    };
    bool all_pass = true;
    std::string jchecks;
    std::printf("%-38s %-7s %-5s %-8s %-11s %-10s %s\n", "case", "kernel", "pass", "bitwise", "max_abs", "max/tol",
                "worst row (gpu vs ref)");
    for (const Case& c : cases) {
        Problem p = make_problem(c.kind, c.rows, c.cols, 0x5eed0000ull + c.rows * 131 + c.cols);
        for (int k = 0; k < kKernelCount; ++k) {
            CheckResult r = check(k, p, c.offset);
            bool exact_ones = true;
            if (!std::strcmp(c.kind, "ones")) // every row must be exactly cols
                exact_ones = r.max_abs == 0.0;
            bool ok = r.pass && r.bitwise && exact_ones;
            all_pass = all_pass && ok;
            std::printf("%-38s %-7s %-5s %-8s %-11.3e %-10.3f %d (%.9g vs %.9g)%s%s\n", c.name, kKernelName[k],
                        ok ? "yes" : "NO", r.bitwise ? "yes" : "NO", r.max_abs, r.max_scaled, r.worst_row,
                        double(r.worst_gpu), r.worst_ref, r.sentinel_ok ? "" : " [wrote past y]",
                        r.bitwise ? "" : (" [first bit mismatch row " + std::to_string(r.first_mismatch_row) + "]").c_str());
            char buf[512];
            std::snprintf(buf, sizeof buf,
                          "%s\n    {\"case\": \"%s\", \"kernel\": \"%s\", \"rows\": %d, \"cols\": %d, \"base_offset_bytes\": %d, "
                          "\"pass\": %s, \"bitwise_vs_lane_emulation\": %s, \"max_abs\": %.6e, \"max_error_over_tolerance\": %.6f, "
                          "\"worst_row\": %d}",
                          jchecks.empty() ? "" : ",", json_escape(c.name).c_str(), kKernelName[k], c.rows, c.cols,
                          2 * c.offset, ok ? "true" : "false", r.bitwise ? "true" : "false", r.max_abs, r.max_scaled,
                          r.worst_row);
            jchecks += buf;
        }
    }
    std::printf("\ncorrectness: %s (tolerance |gpu - ref| <= %.0e + %.0e * sum|w x|, ref in double)\n\n",
                all_pass ? "PASS" : "FAIL", kAtol, kRtol);
    if (check_only || !all_pass) return all_pass ? 0 : 3;

    // ---- timing
    const size_t elems = size_t(o.rows) * o.cols, bytes = elems * 2;
    size_t cold_target = o.cold_bytes ? o.cold_bytes : std::max<size_t>(size_t(8) * l2, size_t(256) << 20);
    size_t copies = std::max<size_t>(2, (cold_target + bytes - 1) / bytes);
    Problem p = make_problem("signed", o.rows, o.cols, 0x5eed);
    uint16_t* dw;
    float *dx, *dy;
    CK(cudaMalloc(&dw, copies * bytes));
    CK(cudaMalloc(&dx, o.cols * 4));
    CK(cudaMalloc(&dy, o.rows * 4));
    for (size_t c = 0; c < copies; ++c)
        CK(cudaMemcpy(dw + c * elems, p.w.data(), bytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx, p.x.data(), o.cols * 4, cudaMemcpyHostToDevice));
    const double empty_graph_us = empty_launch_us(o, true), empty_stream_us = empty_launch_us(o, false);
    unsigned* dscratch;
    CK(cudaMalloc(&dscratch, 4));
    const Stats raw = time_read(o, dw, copies * bytes, sms, dscratch);
    CK(cudaFree(dscratch));

    std::printf("shape %dx%d, weight bytes %zu (2 x rows x cols), cold working set %zu bytes (%zu copies, L2 %d)\n",
                o.rows, o.cols, bytes, copies * bytes, copies, l2);
    std::printf("warmup %d, %d launches per event pair, %d samples\n", o.warmup, o.iters, o.samples);
    std::printf("empty kernel: %.2f us/launch in a graph, %.2f us/launch from the stream\n", empty_graph_us,
                empty_stream_us);
    std::printf("raw read of the cold working set: %.1f GB/s median [%.1f, %.1f]\n\n", raw.median, raw.lo, raw.hi);
    std::printf("kernel time (CUDA Graph replay of %d launches)\n", o.iters);
    std::printf("%-7s | %-30s | %-30s\n", "kernel", "hot: us median [min,max] GB/s", "cold: us median [min,max] GB/s");
    auto gbs = [&](double us) { return double(bytes) / (us * 1e-6) / 1e9; };
    auto jstats = [&](const Stats& t) {
        char b[160];
        std::snprintf(b, sizeof b, "{\"us_median\": %.3f, \"us_min\": %.3f, \"us_max\": %.3f, \"weight_gb_per_s\": %.1f}",
                      t.median, t.lo, t.hi, gbs(t.median));
        return std::string(b);
    };
    struct Row {
        Stats hot, cold, hot_stream, cold_stream;
        HostStats host;
    } res[kKernelCount];
    for (int k = 0; k < kKernelCount; ++k) {
        Row& r = res[k];
        r.hot = time_device(k, o, true, dw, 1, dx, dy);
        r.cold = time_device(k, o, true, dw, copies, dx, dy);
        r.hot_stream = time_device(k, o, false, dw, 1, dx, dy);
        r.cold_stream = time_device(k, o, false, dw, copies, dx, dy);
        r.host = time_host(k, o, dw, dx, dy);
        std::printf("%-7s | %7.2f [%6.2f,%6.2f] %6.1f | %7.2f [%6.2f,%6.2f] %6.1f\n", kKernelName[k], r.hot.median,
                    r.hot.lo, r.hot.hi, gbs(r.hot.median), r.cold.median, r.cold.lo, r.cold.hi, gbs(r.cold.median));
    }
    std::printf("\nwith issue cost (stream launches back to back; host side)\n");
    std::printf("%-7s | %-16s | %-16s | %-12s | %-12s\n", "kernel", "hot us (GB/s)", "cold us (GB/s)", "enqueue us",
                "launch+sync us");
    std::string jtimes;
    for (int k = 0; k < kKernelCount; ++k) {
        const Row& r = res[k];
        std::printf("%-7s | %6.2f (%6.1f) | %6.2f (%6.1f) | %-12.2f | %-12.2f\n", kKernelName[k], r.hot_stream.median,
                    gbs(r.hot_stream.median), r.cold_stream.median, gbs(r.cold_stream.median), r.host.enqueue.median,
                    r.host.roundtrip.median);
        char buf[512];
        std::snprintf(buf, sizeof buf,
                      "\n     \"host\": {\"enqueue_us_per_launch_median\": %.3f, \"enqueue_us_min\": %.3f, \"enqueue_us_max\": %.3f,\n"
                      "              \"launch_plus_sync_us_median\": %.3f, \"launch_plus_sync_us_min\": %.3f, \"launch_plus_sync_us_max\": %.3f}}",
                      r.host.enqueue.median, r.host.enqueue.lo, r.host.enqueue.hi, r.host.roundtrip.median,
                      r.host.roundtrip.lo, r.host.roundtrip.hi);
        jtimes += std::string(jtimes.empty() ? "" : ",") + "\n    {\"kernel\": \"" + kKernelName[k] + "\",\n" +
                  "     \"hot_same_weights\": " + jstats(r.hot) + ",\n" +
                  "     \"cold_cycled_weights\": " + jstats(r.cold) + ",\n" +
                  "     \"hot_same_weights_stream_launch\": " + jstats(r.hot_stream) + ",\n" +
                  "     \"cold_cycled_weights_stream_launch\": " + jstats(r.cold_stream) + "," + buf;
    }
    CK(cudaFree(dw));
    CK(cudaFree(dx));
    CK(cudaFree(dy));

    if (!o.json.empty()) {
        FILE* f = std::fopen(o.json.c_str(), "w");
        if (!f) {
            std::perror(o.json.c_str());
            return 4;
        }
        std::fprintf(f,
                     "{\n  \"schema\": 1,\n  \"issue\": 46,\n  \"tool\": \"tools/bench_cuda_gemv\",\n"
                     "  \"device\": {\"name\": \"%s\", \"compute_capability\": \"%d.%d\", \"sms\": %d, \"l2_bytes\": %d, "
                     "\"memory_bus_bits\": %d, \"memory_bytes\": %zu},\n"
                     "  \"driver\": \"%s\",\n  \"cuda_driver_api\": %d,\n  \"cuda_runtime\": %d,\n  \"toolkit\": \"%s\",\n"
                     "  \"build\": \"%s\",\n"
                     "  \"problem\": {\"rows\": %d, \"cols\": %d, \"weights\": \"bf16 row-major\", \"x\": \"fp32\", \"y\": \"fp32\", "
                     "\"accumulate\": \"fp32 fmaf per lane, __shfl_down_sync 16/8/4/2/1\", \"rows_per_block\": %d, \"threads_per_block\": %d},\n"
                     "  \"weight_bytes\": %zu,\n  \"weight_gb_per_s_definition\": \"2 * rows * cols / seconds per launch (weights only; not a profiler byte count)\",\n"
                     "  \"hot_working_set_bytes\": %zu,\n  \"cold_working_set_bytes\": %zu,\n  \"cold_copies\": %zu,\n"
                     "  \"warmup\": %d,\n  \"launches_per_sample\": %d,\n  \"samples\": %d,\n"
                     "  \"timing\": \"one cudaEvent pair around launches_per_sample back-to-back launches on one stream after warmup; per-launch time = elapsed / launches; median and range over samples. hot/cold_*: the launches are one CUDA Graph replayed per sample (kernel time); *_stream_launch: ordinary launches, which include the per-launch issue cost of WSL2\",\n"
                     "  \"host_timing\": \"steady_clock; enqueue = issuing launches_per_sample launches without waiting; launch_plus_sync = one launch + cudaStreamSynchronize (median of launches_per_sample calls per sample)\",\n"
                     "  \"empty_kernel_us_per_launch\": {\"graph\": %.3f, \"stream\": %.3f},\n"
                     "  \"raw_read_gb_per_s\": {\"median\": %.1f, \"min\": %.1f, \"max\": %.1f, \"method\": \"read-only uint4 grid-stride kernel over the cold working set, one launch per sample\"},\n"
                     "  \"tolerance\": {\"atol\": %.0e, \"rtol\": %.0e, \"rule\": \"|gpu - ref| <= atol + rtol * sum_j |w_j x_j|, ref accumulated in double; plus bit-exact match with a CPU emulation of the kernel's lane order\"},\n"
                     "  \"correctness_pass\": %s,\n  \"checks\": [%s\n  ],\n  \"results\": [%s\n  ]\n}\n",
                     json_escape(prop.name).c_str(), cc_major, cc_minor, sms, l2, bus, prop.totalGlobalMem,
                     json_escape(o.driver).c_str(), driver_api, runtime_api, json_escape(o.toolkit).c_str(),
                     json_escape(o.build).c_str(), o.rows, o.cols, kRowsPerBlock, kThreads, bytes, bytes,
                     copies * bytes, copies, o.warmup, o.iters, o.samples, empty_graph_us, empty_stream_us, raw.median, raw.lo, raw.hi, kAtol, kRtol,
                     all_pass ? "true" : "false", jchecks.c_str(), jtimes.c_str());
        std::fclose(f);
        std::printf("\nwrote %s\n", o.json.c_str());
    }
    return 0;
}
