//! LoRA: train a small low-rank correction instead of the whole matrix.
//!
//! A matrix `W [out, in]` stays frozen. Next to it sit two small tables,
//! `A [r, in]` and `B [out, r]`, with r much smaller than in and out (for
//! example 8). The layer computes with `W + s·B·A`, where `s = alpha / r`:
//!
//! - `A` reads the input in r directions: `xa = A·x` (r numbers);
//! - `B` writes those r numbers back into the output: `Δy = s·B·xa`.
//!
//! So the change to the layer's output is limited to the r directions that
//! `B`'s columns span, and only `r·(in + out)` parameters are trained: for the
//! last FFN's three matrices with r = 8, about 110 thousand instead of 11
//! million.
//!
//! `B` starts at zero, so training starts from exactly the original model;
//! `A` starts small and random, so the gradient of `B` is not zero.
//!
//! Backward of `Δy = s·B·(A·x)`, for `dy`:
//! `dB += s·dyᵀ·xa`, `dxa = s·dy·B`, `dA += dxaᵀ·x`, `dx = dxa·A`.
use crate::tensor::{require, Result};
use crate::train::linalg::{sgemm, Dense, View};

/// One adapter `s·B·A` for a matrix `[out, in]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Lora {
    /// `[r, in]`
    pub a: Dense,
    /// `[out, r]`
    pub b: Dense,
    /// `alpha / r`
    pub scale: f32,
}

impl Lora {
    /// `B = 0` and `A` uniform in ±1/sqrt(in), from a fixed `seed`.
    pub fn new(out: usize, input: usize, rank: usize, alpha: f32, seed: u64) -> Result<Self> {
        require(rank > 0 && rank <= out.min(input), "invalid LoRA rank")?;
        let bound = 1. / (input as f64).sqrt();
        let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        let a: Vec<f32> = (0..rank * input)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 11) as f64 / (1u64 << 53) as f64 * 2. - 1.) as f32 * bound as f32
            })
            .collect();
        Ok(Self {
            a: Dense::new(rank, input, a)?,
            b: Dense::zeros(out, rank)?,
            scale: alpha / rank as f32,
        })
    }
    pub fn rank(&self) -> usize {
        self.a.rows()
    }
    /// The correction for `n` inputs `X [n, in]`: `(Δy [n, out], xa [n, r])`.
    /// `xa` is kept for the backward.
    pub fn forward(&self, x: &[f32], n: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let xa = self.a.forward(x, n)?;
        let mut dy = self.b.forward(&xa, n)?;
        dy.iter_mut().for_each(|v| *v *= self.scale);
        Ok((dy, xa))
    }
    /// Adds the gradients of A and B into `ga`, `gb`, and returns this
    /// adapter's part of the gradient toward the input.
    pub fn backward(
        &self,
        x: &[f32],
        xa: &[f32],
        dy: &[f32],
        n: usize,
        ga: &mut Dense,
        gb: &mut Dense,
    ) -> Result<Vec<f32>> {
        gb.accumulate(self.scale, dy, xa, n)?;
        let mut dxa = self.b.backward_input(dy, n)?;
        dxa.iter_mut().for_each(|v| *v *= self.scale);
        ga.accumulate(1., &dxa, x, n)?;
        self.a.backward_input(&dxa, n)
    }
    /// `W + s·B·A`: the adapter folded into a copy of the matrix.
    pub fn merge(&self, w: &Dense) -> Result<Dense> {
        let (out, input, r) = (self.b.rows(), self.a.cols(), self.rank());
        require(w.rows() == out && w.cols() == input, "LoRA shape mismatch")?;
        let mut merged = w.clone();
        sgemm(
            (out, r, input),
            self.scale,
            View::rows(self.b.values(), r),
            View::rows(self.a.values(), input),
            1.,
            merged.values_mut(),
            input,
        );
        Ok(merged)
    }
}

/// Which FFN matrices carry an adapter, in the order gate, up, down.
pub const PROJECTIONS: [&str; 3] = ["gate", "up", "down"];

/// Adapters for an FFN: `adapters[i]` belongs to `PROJECTIONS[i]`.
#[derive(Debug, Clone, PartialEq)]
pub struct LoraSet {
    pub rank: usize,
    pub alpha: f32,
    pub adapters: [Option<Lora>; 3],
}
