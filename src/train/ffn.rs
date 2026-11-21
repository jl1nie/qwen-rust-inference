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
        mut loss: impl FnMut(usize, &[f32]) -> Result<LossGrad>,
    ) -> Result<(f64, FfnGrad)> {
        let f = self.forward(inputs)?;
        let vocab = self.head.vocab();
        let mut total = 0.;
        let mut dlogits = Vec::with_capacity(f.head.logits.len());
        for (i, logits) in f.head.logits.chunks_exact(vocab).enumerate() {
            let g = loss(i, logits)?;
            require(g.dlogits.len() == vocab, "loss gradient shape mismatch")?;
            total += g.loss;
            dlogits.extend(g.dlogits);
        }
        let dh = self.head.backward(&f.h, &dlogits, inputs.n)?;
        // h = after_mixer + FFN(post_norm): the FFN output receives dh as is.
        let mut grad = self.ffn.zero_grad()?;
        self.ffn.backward(&f.ffn, &dh, &mut grad)?;
        Ok((total, grad))
    }
}
