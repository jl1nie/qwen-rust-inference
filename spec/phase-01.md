# Phase 0 / 1 contract

Only single-position CPU f32 operations and bias-free gated FFN are supported.
Attention, DeltaNet, model inference, tokenizer, sampling and weight loading are
explicitly unsupported. Educational weights are synthetic, not a Qwen checkpoint.

Matrices are contiguous row-major [out, in]; linear(x) = W x. gate/up are
[intermediate, hidden], down is [hidden, intermediate]. The writing vector of
unit j is down[:, j]. FFN = down(SiLU(gate(x)) * up(x)). All stages are exposed.
The intervention zeros a post-gate product in a new activation vector. Immutable
weights and baseline activations must remain unchanged. baseline - edited equals
product[j] * down[:, j]. Phase 6 direction overlays are not yet supported.

RMSNorm uses mean(x*x), epsilon > 0. Educational/direct scale and Qwen zero-centered
scale (1 + stored_weight) are separate enum variants. Qwen3_5RMSNorm does f32
normalization then f32 zero-centered scaling; do not confuse RMSNormGated's direct
scale and gate with this norm. BF16 cast/rounding is deferred to Phase 5.
SiLU uses a stable sigmoid branch; softmax subtracts max. Empty vectors, invalid
shapes, NaN/Inf inputs or outputs and invalid epsilon are errors. Dot products
accumulate in f32 in increasing input-index order, without explicit FMA.

Tolerance frozen before implementation: f32 elementwise/FFN/norm atol=2e-6,
rtol=2e-5; hand-check exact integers atol=0, rtol=0. Check each element via
abs(actual-expected) <= atol + rtol*abs(expected); report max absolute and relative
errors (relative denominator max(abs(expected), 1e-12)). Do not widen on failure.

Hand audit: [[1,2],[3,4]]*[1,2] = [5,11]. softmax([1000,1000])=[0.5,0.5].
For x=[1,2], gate rows [[0,0],[1,0]], up rows [[1,0],[0,1]],
down=[[1,3],[2,4]]: gate=[0,1], up=[1,2], product=[0,2s],
output=[6s,8s], s=1/(1+exp(-1)). Removing unit 1 removes the entire output.
These analytical cases do not rely on the Python generator or Rust implementation.

Trace is opt-in NDJSON to a local path with create-new semantics. Default emits
no trace. Full arrays require an explicit flag; byte cap includes newline and is
checked before writing a record. This phase exposes run ID, token/layer (zero),
operation, shape, f32 dtype, norm/min/max/finite, unit selection and optional values.
Logits fields remain null because logits do not exist in Phase 1.
