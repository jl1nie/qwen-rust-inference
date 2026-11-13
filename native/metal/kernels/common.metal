// Apache-2.0. Metal kernels of the device engine (src/device.rs), compiled at
// runtime by native/metal/bridge.mm with safe math (no fast-math). The files of
// native/metal/kernels/ are concatenated in the order listed in
// tools/build-metal.sh; tensor.metal is a second library (Metal 4) that
// starts with this file too.
//
// Reading order (spec/metal.md, "Kernel levels"):
//   common -> elementwise -> norm -> deltanet -> attention_decode
//   -> attention_prefill -> matmul -> tensor
// Inside each file, level 1 (reference: mirrors the CPU code in src/) comes
// first, then level 2 (simdgroup), then level 3 (tensor ops, tensor.metal).
// Equations mirror src/{ops,attention,deltanet}.rs; buffer slot i == Params.p[i].
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Scalars travel separately from pointers: Metal binds each buffer slot itself.
// n[0..7] and f[0..3] are the op's integer and float parameters (src/device.rs),
// count is the number of work items.
struct Scalars { int n[8]; float f[4]; int count; };

inline float sigmoid_(float x) {
    if (x >= 0.f) return 1.f / (1.f + exp(-x));
    float e = exp(x); return e / (1.f + e);
}
inline float silu_(float x) { return x * sigmoid_(x); }
// MSL has no log1p; for x <= 20 the absolute error of log(1+e^x) is tiny.
inline float softplus_(float x) { return x > 20.f ? x : log(1.f + exp(x)); }

// BF16 is the upper half of an f32: widening is exact (no rounding).
inline float bf16_(ushort v) { return as_type<float>(uint(v) << 16); }
inline float widen(float v) { return v; }
inline float widen(ushort v) { return bf16_(v); }
inline float4 widen4(float4 v) { return v; }
inline float4 widen4(ushort4 u) { return float4(bf16_(u.x), bf16_(u.y), bf16_(u.z), bf16_(u.w)); }

// One thread per work item (t < count).
#define ITEM_ARGS uint t [[thread_position_in_grid]], constant Scalars& a [[buffer(8)]]
#define GUARD if (int(t) >= a.count) return;
