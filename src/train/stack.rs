//! The layers from `from` to the last, and the head, for whole sequences
//! (ADR 0037, step 4).
//!
//! The layers below `from` are frozen and run once: their output, the
//! residual stream entering layer `from`, is the input here (`capture`).
//! Above it, every block runs forward over the whole sequence and back. The
//! FFNs of the layers in `ffns` are trainable copies (full or LoRA); every
//! other part is frozen and only passes the gradient on.
//!
//! The head (final norm and the tied LM head, 248,320 rows in the real
//! model) is computed only at the scored positions, since the loss reads
//! nothing else. A block's output at one position depends on the earlier
//! positions, so the blocks always run over the whole sequence.
use crate::model::Model;
use crate::tensor::{require, Backend, InferenceError, Result};
use crate::train::block::{BlockForward, TrainBlock};
use crate::train::ffn::{FfnGrad, TrainFfn};
use crate::train::grad::LossGrad;
use crate::train::head::Head;
use rayon::prelude::*;

/// The trainable part of a model: layers `from..` with some FFNs replaced.
#[derive(Debug, Clone)]
pub struct Stack<'a> {
    model: &'a Model,
    pub from: usize,
    /// (layer, its trainable FFN), sorted by layer, every layer ≥ `from`.
    pub ffns: Vec<(usize, TrainFfn)>,
    head: Head<'a>,
}

/// What the forward computed, kept for the backward.
#[derive(Debug, Clone, PartialEq)]
pub struct StackForward {
    /// One per layer from `from`, in order.
    pub blocks: Vec<BlockForward>,
    /// The residual stream after the last layer, `[n, hidden]`.
    pub h: Vec<f32>,
    pub n: usize,
}

/// The result of one sequence's loss and backward.
#[derive(Debug, Clone, PartialEq)]
pub struct StackGrad {
    /// The loss summed over the scored positions.
    pub loss: f64,
    /// The gradient toward the input (the residual entering `from`).
    pub dx: Vec<f32>,
}

/// The residual stream entering layer `layer` at every position of `ids`,
/// `[n, hidden]`, computed by the inference path on the CPU (for layer 0, the
/// embedded IDs).
pub fn capture(model: &Model, ids: &[u32], layer: usize, backend: Backend) -> Result<Vec<f32>> {
    require(layer < model.blocks.len(), "layer out of range")?;
    let mut state = model.new_state();
    let mut out = Vec::with_capacity(ids.len() * model.config.hidden_size);
    for &id in ids {
        let s = model.step(id, &mut state, backend, None)?;
        out.extend(if layer == 0 {
            s.embedding
        } else {
            s.layers[layer - 1].clone()
        });
    }
    Ok(out)
}

impl<'a> Stack<'a> {
    /// Layers `from..` of `model`, with the FFNs of `train` copied for
    /// training (full matrices).
    pub fn new(model: &'a Model, from: usize, train: &[usize]) -> Result<Self> {
        let ffns = train
            .iter()
            .map(|&l| {
                model
                    .blocks
                    .get(l)
                    .map(|b| (l, TrainFfn::from_weights(&b.ffn)))
                    .ok_or_else(|| InferenceError(format!("layer {l} out of range")))
            })
            .collect::<Result<_>>()?;
        Self::with_ffns(model, from, ffns)
    }
    /// As `new`, with the trainable FFNs given (for example with LoRA).
    pub fn with_ffns(
        model: &'a Model,
        from: usize,
        mut ffns: Vec<(usize, TrainFfn)>,
    ) -> Result<Self> {
        ffns.sort_by_key(|(l, _)| *l);
        require(from < model.blocks.len(), "first layer out of range")?;
        require(
            ffns.iter()
                .all(|(l, _)| *l >= from && *l < model.blocks.len())
                && ffns.windows(2).all(|w| w[0].0 < w[1].0),
            "trainable layers must be distinct and at or above the first layer",
        )?;
        for (l, f) in &ffns {
            require(
                (f.hidden(), f.intermediate()) == model.blocks[*l].ffn.dimensions(),
                "trainable FFN shape mismatch",
            )?;
        }
        Ok(Self {
            model,
            from,
            ffns,
            head: Head::new(model),
        })
    }
    pub fn model(&self) -> &'a Model {
        self.model
    }
    /// One zero gradient per trainable FFN, in the order of `ffns`.
    pub fn zero_grads(&self) -> Result<Vec<FfnGrad>> {
        self.ffns.iter().map(|(_, f)| f.zero_grad()).collect()
    }
    fn block(&self, layer: usize) -> TrainBlock<'_> {
        let w = &self.model.blocks[layer];
        match self.ffns.iter().find(|(l, _)| *l == layer) {
            Some((_, f)) => TrainBlock::with_ffn(w, f),
            None => TrainBlock::frozen(w),
        }
    }
    /// Forward of the blocks over positions `0..n`; `x` is the residual
    /// entering layer `from`, `[n, hidden]`.
    pub fn forward(&self, x: &[f32], n: usize) -> Result<StackForward> {
        let mut h = x.to_vec();
        let mut blocks = Vec::with_capacity(self.model.blocks.len() - self.from);
        for layer in self.from..self.model.blocks.len() {
            let f = self.block(layer).forward(&h, n)?;
            h = f.output.clone();
            blocks.push(f);
        }
        Ok(StackForward { blocks, h, n })
    }
    /// The logits at the positions `picks`, `[picks.len(), vocab]`.
    pub fn logits(&self, f: &StackForward, picks: &[usize]) -> Result<Vec<f32>> {
        Ok(self
            .head
            .forward(&rows(&f.h, picks, f.n)?, picks.len())?
            .logits)
    }
    /// The loss at `picks` and its backward. `loss(i, logits)` scores the
    /// logits of `picks[i]`. The FFN gradients are added into `grads` (one
    /// per entry of `ffns`, as `zero_grads`).
    pub fn accumulate(
        &self,
        x: &[f32],
        n: usize,
        picks: &[usize],
        loss: impl Fn(usize, &[f32]) -> Result<LossGrad> + Sync,
        grads: &mut [FfnGrad],
    ) -> Result<StackGrad> {
        require(
            grads.len() == self.ffns.len(),
            "one gradient per trainable FFN",
        )?;
        let f = self.forward(x, n)?;
        let d = self.head.hidden();
        let vocab = self.head.vocab();
        let picked = rows(&f.h, picks, n)?;
        let head = self.head.forward(&picked, picks.len())?;
        let scored = head
            .logits
            .par_chunks_exact(vocab)
            .enumerate()
            .map(|(i, z)| loss(i, z))
            .collect::<Result<Vec<_>>>()?;
        let mut total = 0.;
        let mut dlogits = Vec::with_capacity(head.logits.len());
        for g in scored {
            require(g.dlogits.len() == vocab, "loss gradient shape mismatch")?;
            total += g.loss;
            dlogits.extend(g.dlogits);
        }
        let dpicked = self.head.backward(&picked, &dlogits, picks.len())?;
        // Positions that are not scored get no gradient from the head.
        let mut dh = vec![0.; n * d];
        for (i, &p) in picks.iter().enumerate() {
            for (a, b) in dh[p * d..(p + 1) * d]
                .iter_mut()
                .zip(&dpicked[i * d..(i + 1) * d])
            {
                *a += b;
            }
        }
        for (i, layer) in (self.from..self.model.blocks.len()).enumerate().rev() {
            let slot = self.ffns.iter().position(|(l, _)| *l == layer);
            let g = slot.map(|s| &mut grads[s]);
            dh = self.block(layer).backward(&f.blocks[i], &dh, g)?;
        }
        Ok(StackGrad {
            loss: total,
            dx: dh,
        })
    }
}

/// Rows `picks` of `x [n, d]`.
fn rows(x: &[f32], picks: &[usize], n: usize) -> Result<Vec<f32>> {
    require(n > 0 && x.len() % n == 0, "rows shape")?;
    let d = x.len() / n;
    require(picks.iter().all(|&p| p < n), "position out of range")?;
    Ok(picks
        .iter()
        .flat_map(|&p| x[p * d..(p + 1) * d].iter().copied())
        .collect())
}
