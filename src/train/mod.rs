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
pub mod grad;
pub mod linalg;
