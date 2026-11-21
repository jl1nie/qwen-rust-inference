//! A trainable copy of one FFN, its batched forward and its backward, and the
//! top layer (FFN of the last layer + head) that book 3 trains first.
//!
//! The FFN is `down · (silu(gate·x) ⊙ (up·x))`, as in `crate::ffn`. Its
//! backward retraces the forward from the end:
//!
//! 1. `dproduct = dout · W_down`, and `dW_down += doutᵀ · product`
//! 2. SwiGLU: `dgate`, `dup` from `dproduct` (`grad::swiglu_backward`)
//! 3. `dW_gate += dgateᵀ · x`, `dW_up += dupᵀ · x`
//! 4. `dx = dgate · W_gate + dup · W_up` (only needed below this layer)
use crate::ffn::FfnWeights;
use crate::model::Model;
use crate::ops::{multiply, silu};
use crate::tensor::{require, Result};
use crate::train::grad::{swiglu_backward, LossGrad};
use crate::train::head::{Head, HeadForward};
use crate::train::linalg::Dense;
use rayon::prelude::*;

/// FFN parameters as trainable f32 copies.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainFfn {
    /// `[intermediate, hidden]`
    pub gate: Dense,
    /// `[intermediate, hidden]`
    pub up: Dense,
    /// `[hidden, intermediate]`
    pub down: Dense,
}

/// What the forward computed for `n` positions, kept for the backward.
#[derive(Debug, Clone, PartialEq)]
pub struct FfnForward {
    pub x: Vec<f32>,
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub product: Vec<f32>,
    pub output: Vec<f32>,
    pub n: usize,
}

/// Gradients with the shapes of `TrainFfn`.
#[derive(Debug, Clone, PartialEq)]
pub struct FfnGrad {
    pub gate: Dense,
    pub up: Dense,
    pub down: Dense,
}

impl FfnGrad {
    /// The gradients in the order of `TrainFfn::params_mut`.
    pub fn slices(&self) -> [&[f32]; 3] {
        [self.gate.values(), self.up.values(), self.down.values()]
    }
    pub fn scale(&mut self, s: f32) {
        for d in [&mut self.gate, &mut self.up, &mut self.down] {
            d.values_mut().iter_mut().for_each(|v| *v *= s);
        }
    }
    /// The L2 norm over all three tensors.
    pub fn norm(&self) -> f64 {
        self.slices()
            .iter()
            .flat_map(|s| s.iter())
            .map(|&v| (v as f64).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

impl TrainFfn {
    pub fn from_weights(w: &FfnWeights) -> Self {
        let [gate, up, down] = w.matrices();
        Self {
            gate: Dense::from_matrix(gate),
            up: Dense::from_matrix(up),
            down: Dense::from_matrix(down),
        }
    }
    /// Back to inference weights (f32 storage), for `Model::with_ffn`.
    pub fn to_weights(&self) -> Result<FfnWeights> {
        FfnWeights::new(
            self.gate.to_matrix()?,
            self.up.to_matrix()?,
            self.down.to_matrix()?,
        )
    }
    pub fn hidden(&self) -> usize {
        self.gate.cols()
    }
    /// The parameter tensors in a fixed order (gate, up, down), for the optimizer.
    pub fn params_mut(&mut self) -> [&mut [f32]; 3] {
        [
            self.gate.values_mut(),
            self.up.values_mut(),
            self.down.values_mut(),
        ]
    }
    pub fn sizes(&self) -> [usize; 3] {
        [&self.gate, &self.up, &self.down].map(|d| d.values().len())
    }
    pub fn intermediate(&self) -> usize {
        self.gate.rows()
    }
    pub fn zero_grad(&self) -> Result<FfnGrad> {
        Ok(FfnGrad {
            gate: Dense::zeros(self.gate.rows(), self.gate.cols())?,
            up: Dense::zeros(self.up.rows(), self.up.cols())?,
            down: Dense::zeros(self.down.rows(), self.down.cols())?,
        })
    }
    /// `x [n, hidden]` → output `[n, hidden]`.
    pub fn forward(&self, x: &[f32], n: usize) -> Result<FfnForward> {
        let gate = self.gate.forward(x, n)?;
        let up = self.up.forward(x, n)?;
        let product = multiply(&silu(&gate)?, &up)?;
        let output = self.down.forward(&product, n)?;
        Ok(FfnForward {
            x: x.to_vec(),
            gate,
            up,
            product,
            output,
            n,
        })
    }
    /// Adds the parameter gradients for `dout [n, hidden]` into `grad`, and
    /// returns the gradient with respect to the input `x`.
    pub fn backward(&self, f: &FfnForward, dout: &[f32], grad: &mut FfnGrad) -> Result<Vec<f32>> {
        let n = f.n;
        require(
            dout.len() == n * self.hidden(),
            "FFN backward shape mismatch",
        )?;
        let dproduct = self.down.backward_input(dout, n)?;
        grad.down.accumulate(1., dout, &f.product, n)?;
        let s = swiglu_backward(&f.gate, &f.up, &dproduct)?;
        grad.gate.accumulate(1., &s.dgate, &f.x, n)?;
        grad.up.accumulate(1., &s.dup, &f.x, n)?;
        let mut dx = self.gate.backward_input(&s.dgate, n)?;
        for (a, b) in dx.iter_mut().zip(self.up.backward_input(&s.dup, n)?) {
            *a += b;
        }
        Ok(dx)
    }
}

/// The inputs of the last layer's FFN for `n` positions, which do not depend
/// on its parameters: `after_mixer` (the residual the FFN output is added to)
/// and `post_norm` (the FFN input). Both `[n, hidden]`.
#[derive(Debug, Clone, PartialEq)]
pub struct TopInputs {
    pub after_mixer: Vec<f32>,
    pub post_norm: Vec<f32>,
    pub n: usize,
}
impl TopInputs {
    /// The positions `picks` (in that order). Positions do not interact in
    /// the top layer's FFN and head, so any subset can be scored on its own.
    pub fn subset(&self, picks: &[usize]) -> Result<Self> {
        require(picks.iter().all(|&p| p < self.n), "position out of range")?;
        let d = self.after_mixer.len() / self.n.max(1);
        let take = |v: &[f32]| -> Vec<f32> {
            picks
                .iter()
                .flat_map(|&p| v[p * d..(p + 1) * d].iter().copied())
                .collect()
        };
        Ok(Self {
            after_mixer: take(&self.after_mixer),
            post_norm: take(&self.post_norm),
            n: picks.len(),
        })
    }
}

/// The forward of the top layer: `h = after_mixer + FFN(post_norm)`, then the
/// head.
#[derive(Debug, Clone, PartialEq)]
pub struct TopForward {
    pub ffn: FfnForward,
    pub h: Vec<f32>,
    pub head: HeadForward,
}

/// The FFN of the last layer, with the frozen head above it. Training it needs
/// no layer below: their output is in `TopInputs`.
#[derive(Debug, Clone)]
pub struct TopFfn<'a> {
    pub ffn: TrainFfn,
    pub head: Head<'a>,
    pub layer: usize,
}

impl<'a> TopFfn<'a> {
    /// The last layer's FFN of `model`, copied for training.
    pub fn new(model: &'a Model) -> Result<Self> {
        let layer = model
            .blocks
            .len()
            .checked_sub(1)
            .ok_or_else(|| crate::tensor::InferenceError("model has no layer".into()))?;
        Ok(Self {
            ffn: TrainFfn::from_weights(&model.blocks[layer].ffn),
            head: Head::new(model),
            layer,
        })
    }
    pub fn forward(&self, inputs: &TopInputs) -> Result<TopForward> {
        let n = inputs.n;
        let d = self.ffn.hidden();
        require(
            inputs.after_mixer.len() == n * d && inputs.post_norm.len() == n * d,
            "top-layer input shape mismatch",
        )?;
        let ffn = self.ffn.forward(&inputs.post_norm, n)?;
        let h: Vec<f32> = inputs
            .after_mixer
            .iter()
            .zip(&ffn.output)
            .map(|(a, b)| a + b)
            .collect();
        let head = self.head.forward(&h, n)?;
        Ok(TopForward { ffn, h, head })
    }
    /// Loss summed over the positions, and the FFN gradients. `loss` scores
    /// one position's logits (`index`, `logits`) and gives its gradient.
    pub fn loss_and_grad(
        &self,
        inputs: &TopInputs,
        loss: impl Fn(usize, &[f32]) -> Result<LossGrad> + Sync,
    ) -> Result<(f64, FfnGrad)> {
        let mut grad = self.ffn.zero_grad()?;
        let total = self.accumulate(inputs, loss, &mut grad)?;
        Ok((total, grad))
    }
    /// As `loss_and_grad`, adding the gradients into `grad` (for minibatches
    /// whose gradients are summed before one update).
    pub fn accumulate(
        &self,
        inputs: &TopInputs,
        loss: impl Fn(usize, &[f32]) -> Result<LossGrad> + Sync,
        grad: &mut FfnGrad,
    ) -> Result<f64> {
        let f = self.forward(inputs)?;
        let vocab = self.head.vocab();
        // Each position's softmax over the vocabulary is independent.
        let scored = f
            .head
            .logits
            .par_chunks_exact(vocab)
            .enumerate()
            .map(|(i, logits)| loss(i, logits))
            .collect::<Result<Vec<_>>>()?;
        let mut total = 0.;
        let mut dlogits = Vec::with_capacity(f.head.logits.len());
        for g in scored {
            require(g.dlogits.len() == vocab, "loss gradient shape mismatch")?;
            total += g.loss;
            dlogits.extend(g.dlogits);
        }
        let dh = self.head.backward(&f.h, &dlogits, inputs.n)?;
        // h = after_mixer + FFN(post_norm): the FFN output receives dh as is.
        self.ffn.backward(&f.ffn, &dh, grad)?;
        Ok(total)
    }
}
