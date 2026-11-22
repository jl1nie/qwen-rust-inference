//! # Training (book 3): change parameters with gradients instead of by hand
//!
//! Volume 1, chapter 5 changed one column of an FFN by hand. This module
//! makes the same kind of change with gradients. It sits behind the `train`
//! feature, off by default, and never touches the inference path: frozen
//! weights stay in `Matrix` (BF16), and trainable copies live in `Dense` (f32).
//!
//! The CPU version is the reference for correctness. Every backward function
//! is checked against numerical differentiation (`tests/train_grad.rs`)
//! before it is used on the real model, and before any GPU port.
//!
//! - `linalg`: `Dense` and the row-major GEMM wrapper (forward `X·Wᵀ`, input
//!   gradient `dY·W`, parameter gradient `dYᵀ·X`).
//! - `grad`: backward of each operation (RMSNorm, SiLU, SwiGLU) and the losses
//!   (cross-entropy, KL over the full vocabulary or over the top K + "other").
//! - `head`: final RMSNorm and tied LM head for many positions, and the
//!   gradient back through them to the residual stream.
//! - `ffn`: a trainable FFN copy and its backward; `TopFfn`, the last layer's
//!   FFN with the head above it, trained from cached inputs.
//! - `cache`: the last layer's FFN inputs, captured once by the inference path.
//! - `lora`: low-rank adapters `s·B·A` beside frozen FFN matrices.
//! - `optim`: SGD and Adam. `fit`: the training loop and the fact to teach.
//! - `metrics`: perplexity, KL and top-1 agreement between two models.
//! - `corpus`: many prompt/response pairs (or a plain text) to learn by
//!   cross-entropy: distillation from the teacher's generated text.
//! - `teacher`: teacher data for distillation (Strata `STRATA_LOGPOS` top-K).
//! - `lesson`: the FFN backward by hand on volume 1, chapter 4's example.
//! - `delta`: what training changed in the weights (size, rank, units,
//!   directions through the LM head), and a hand edit written as ΔW.
//! - `probe`: linear probes on a layer's vectors (is a label readable by
//!   one matrix?), scored leaving one prompt template out.
//! - `checkpoint`: the trained FFN and optimizer state in a safetensors file.
//! - `attention`, `block`: a whole sequence through a block and back, so that
//!   layers below the last can be trained (ADR 0037). Attention blocks only,
//!   so far.
pub mod attention;
pub mod block;
pub mod cache;
pub mod checkpoint;
pub mod corpus;
pub mod delta;
pub mod ffn;
pub mod fit;
pub mod grad;
pub mod head;
pub mod lesson;
pub mod linalg;
pub mod lora;
pub mod metrics;
pub mod optim;
pub mod probe;
pub mod teacher;
