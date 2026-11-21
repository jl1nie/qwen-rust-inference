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
//!
//! With LoRA adapters (`lora`), gate, up and down stay frozen and only the
//! adapters' A and B are trained; each matrix then computes `W·x + s·B·A·x`.
use crate::ffn::FfnWeights;
use crate::model::Model;
use crate::ops::{multiply, silu};
use crate::tensor::{require, Result};
use crate::train::grad::{swiglu_backward, LossGrad};
use crate::train::head::{Head, HeadForward};
use crate::train::linalg::Dense;
use crate::train::lora::{Lora, LoraSet};
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
    /// With adapters, only they are trained; gate, up and down are frozen.
    pub lora: Option<LoraSet>,
}

/// What the forward computed for `n` positions, kept for the backward.
#[derive(Debug, Clone, PartialEq)]
pub struct FfnForward {
    pub x: Vec<f32>,
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub product: Vec<f32>,
    pub output: Vec<f32>,
    /// `A·input` of each adapter (gate, up, down), when it has one.
    pub lora_xa: [Option<Vec<f32>>; 3],
    pub n: usize,
}

/// Gradients of the trainable tensors, in the order of
/// `TrainFfn::params_mut`: gate, up, down without LoRA; A and B of each
/// adapter (gate, up, down order) with LoRA.
#[derive(Debug, Clone, PartialEq)]
pub struct FfnGrad {
    pub tensors: Vec<Dense>,
}

impl FfnGrad {
    pub fn slices(&self) -> Vec<&[f32]> {
        self.tensors.iter().map(Dense::values).collect()
    }
    pub fn scale(&mut self, s: f32) {
        for d in &mut self.tensors {
            d.values_mut().iter_mut().for_each(|v| *v *= s);
        }
    }
    /// The L2 norm over all tensors.
    pub fn norm(&self) -> f64 {
        self.tensors
            .iter()
            .flat_map(|d| d.values())
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
            lora: None,
        }
    }
    fn base(&self, proj: usize) -> &Dense {
        [&self.gate, &self.up, &self.down][proj]
    }
    fn adapter(&self, proj: usize) -> Option<&Lora> {
        self.lora.as_ref().and_then(|l| l.adapters[proj].as_ref())
    }
    /// Adds rank-`rank` adapters to the matrices `targets` (indices into
    /// `lora::PROJECTIONS`), freezing gate, up and down. `seed` fixes A.
    pub fn with_lora(
        mut self,
        rank: usize,
        alpha: f32,
        targets: &[usize],
        seed: u64,
    ) -> Result<Self> {
        require(
            !targets.is_empty() && targets.iter().all(|&t| t < 3),
            "invalid LoRA targets",
        )?;
        let mut adapters: [Option<Lora>; 3] = [None, None, None];
        for &t in targets {
            let w = self.base(t);
            adapters[t] = Some(Lora::new(w.rows(), w.cols(), rank, alpha, seed + t as u64)?);
        }
        self.lora = Some(LoraSet {
            rank,
            alpha,
            adapters,
        });
        Ok(self)
    }
    /// gate, up, down with the adapters folded in.
    pub fn merged(&self) -> Result<[Dense; 3]> {
        let one = |proj: usize| match self.adapter(proj) {
            Some(l) => l.merge(self.base(proj)),
            None => Ok(self.base(proj).clone()),
        };
        Ok([one(0)?, one(1)?, one(2)?])
    }
    /// Back to inference weights (f32 storage, adapters folded in), for
    /// `Model::with_ffn`.
    pub fn to_weights(&self) -> Result<FfnWeights> {
        let [gate, up, down] = self.merged()?;
        FfnWeights::new(gate.to_matrix()?, up.to_matrix()?, down.to_matrix()?)
    }
    pub fn hidden(&self) -> usize {
        self.gate.cols()
    }
    pub fn intermediate(&self) -> usize {
        self.gate.rows()
    }
    /// The trainable tensors in a fixed order, for the optimizer: gate, up,
    /// down; or with LoRA, A then B of each adapter.
    pub fn params_mut(&mut self) -> Vec<&mut [f32]> {
        match &mut self.lora {
            None => vec![
                self.gate.values_mut(),
                self.up.values_mut(),
                self.down.values_mut(),
            ],
            Some(l) => l
                .adapters
                .iter_mut()
                .flatten()
                .flat_map(|a| [a.a.values_mut(), a.b.values_mut()])
                .collect(),
        }
    }
    fn trainable(&self) -> Vec<&Dense> {
        match &self.lora {
            None => vec![&self.gate, &self.up, &self.down],
            Some(l) => l
                .adapters
                .iter()
                .flatten()
                .flat_map(|a| [&a.a, &a.b])
                .collect(),
        }
    }
    pub fn sizes(&self) -> Vec<usize> {
        self.trainable().iter().map(|d| d.values().len()).collect()
    }
    /// Number of trained parameters.
    pub fn trained_parameters(&self) -> usize {
        self.sizes().iter().sum()
    }
    pub fn zero_grad(&self) -> Result<FfnGrad> {
        Ok(FfnGrad {
            tensors: self
                .trainable()
                .iter()
                .map(|d| Dense::zeros(d.rows(), d.cols()))
                .collect::<Result<_>>()?,
        })
    }
    /// Index in `FfnGrad::tensors` of the gradient of matrix `proj` (without
    /// LoRA) or of its adapter's A (with LoRA; B follows).
    fn slot(&self, proj: usize) -> Option<usize> {
        match &self.lora {
            None => Some(proj),
            Some(l) => l.adapters[proj]
                .as_ref()
                .map(|_| 2 * l.adapters[..proj].iter().flatten().count()),
        }
    }
    /// `W·x` (+ the adapter's correction) for matrix `proj`.
    fn linear(&self, proj: usize, x: &[f32], n: usize) -> Result<(Vec<f32>, Option<Vec<f32>>)> {
        let mut y = self.base(proj).forward(x, n)?;
        let xa = match self.adapter(proj) {
            None => None,
            Some(l) => {
                let (dy, xa) = l.forward(x, n)?;
                y.iter_mut().zip(dy).for_each(|(a, b)| *a += b);
                Some(xa)
            }
        };
        Ok((y, xa))
    }
    /// Backward of `linear`: adds the trainable gradients into `grad` and
    /// returns the gradient toward `x`.
    fn linear_backward(
        &self,
        proj: usize,
        x: &[f32],
        xa: Option<&Vec<f32>>,
        dy: &[f32],
        n: usize,
        grad: &mut FfnGrad,
    ) -> Result<Vec<f32>> {
        let mut dx = self.base(proj).backward_input(dy, n)?;
        match (self.adapter(proj), self.slot(proj)) {
            (Some(l), Some(i)) => {
                let xa =
                    xa.ok_or_else(|| crate::tensor::InferenceError("missing LoRA cache".into()))?;
                let (ga, gb) = grad.tensors[i..].split_at_mut(1);
                let extra = l.backward(x, xa, dy, n, &mut ga[0], &mut gb[0])?;
                dx.iter_mut().zip(extra).for_each(|(a, b)| *a += b);
            }
            (None, Some(i)) => grad.tensors[i].accumulate(1., dy, x, n)?,
            // A frozen matrix without an adapter: nothing to train.
            _ => {}
        }
        Ok(dx)
    }
    /// `x [n, hidden]` → output `[n, hidden]`.
    pub fn forward(&self, x: &[f32], n: usize) -> Result<FfnForward> {
        let (gate, xa_gate) = self.linear(0, x, n)?;
        let (up, xa_up) = self.linear(1, x, n)?;
        let product = multiply(&silu(&gate)?, &up)?;
        let (output, xa_down) = self.linear(2, &product, n)?;
        Ok(FfnForward {
            x: x.to_vec(),
            gate,
            up,
            product,
            output,
            lora_xa: [xa_gate, xa_up, xa_down],
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
        let dproduct = self.linear_backward(2, &f.product, f.lora_xa[2].as_ref(), dout, n, grad)?;
        let s = swiglu_backward(&f.gate, &f.up, &dproduct)?;
        let mut dx = self.linear_backward(0, &f.x, f.lora_xa[0].as_ref(), &s.dgate, n, grad)?;
        let dx_up = self.linear_backward(1, &f.x, f.lora_xa[1].as_ref(), &s.dup, n, grad)?;
        dx.iter_mut().zip(dx_up).for_each(|(a, b)| *a += b);
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
