# Phase 2–6 contract (before implementation)

Single-batch, CPU, text only. Token positions are sequential scalar positions;
text MRoPE has equal T/H/W positions, reducing to the pinned partial rotary math.
No vision inputs, alternate RoPE, padding batches, quantization or MTP execution.

2: Full attention projects interleaved per-head [Q, gate], zero-centered Q/K
RMSNorm, partial rotate-half RoPE, contiguous GQA grouping, causal scaled dot
products, softmax, V sum, sigmoid output gate, output projection. Each layer owns
its KV state; append uses current length as absolute position. A separate all-
sequence causal kernel must agree with cached steps, including split prefill.
Empty prefill is identity, reset empties state; reject position mismatch.

3: Project QKV, z, b, a. Causal depthwise conv (oldest weight first), SiLU.
Normalize Q/K by sqrt(sum squares + 1e-6); Q additionally /sqrt(key_dim).
g=-exp(A_log)*softplus(a+dt_bias), beta=sigmoid(b).
S <- exp(g) S; delta=(v-k^T S)*beta; S <- S+k delta^T; y=q^T S.
Gated RMSNorm uses direct weight then SiLU(z), then output projection.
Recurrent [heads,key_dim,value_dim] and conv [channels,kernel-1] remain fixed.
No chunk implementation is used in Rust. Compare every step with pinned source.

4: 3 DeltaNet + 1 attention, pre-norm mixer + residual, pre-norm FFN + residual,
embedding, final zero-centered norm, tied vocabulary projection. Weights,
ModelState, TokenHistory, StepActivations are distinct. Greedy then deterministic
seeded sampling (temperature>0, top-k), EOS only stops an answer; </think> does
not. Stop at maximum generated count, including zero. Runs start fresh.

5: Config checks supported architecture. Load safetensors BF16/F32 only, decode
BF16 bits exactly to f32, validate complete language tensor names/shapes and
shared embeddings. Enumerate exact ignored vision/MTP names in a manifest; reject
other unused tensors. No automatic downloads. Explicit shell downloader pins
revision and validates SHA-256/Git blob identity. CPU fast matrix backend must
agree with retained reference. Tokenizer via Rust tokenizers; text-only one-user
chat template follows the pinned template. Reject unsupported multimodal messages.

6: Separate serialized intervention overlay: model revision, zero-based layer,
tensor name, column indices, finite direction, finite coefficient. Unit zeroing
is applied to temporary product; direction removal computes
w' = w - coefficient * d * dot(d,w)/dot(d,d) on selected down columns without
mutating weights. Empty overlay gives baseline; reject zero/invalid direction,
wrong revision/layer/name/column. Compare FFN, layer residual and logits, plus
repeated token and repeated ngram metrics. No research performance claim.

Frozen tolerances: synthetic f32 component/layer/logits atol=2e-5 rtol=2e-4;
Delta state accumulation atol=2e-5 rtol=2e-4. Real model BF16 weights converted to
f32 on both sides, all arithmetic f32: per-layer/logits atol=3e-3 rtol=3e-3.
Compare every element, max abs, relative (floor 1e-12), cosine, top candidates.
First divergent boundary fails; diagnose operation rather than widen tolerances.
Real tests are explicit opt-in and excluded from standard offline CI.
