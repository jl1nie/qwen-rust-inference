//! The head: the residual stream after the last layer → final RMSNorm → tied
//! LM head → logits, for `n` positions at once, and its backward.
//!
//! The head is frozen. Its backward only passes the gradient through, from
//! the logits to the residual stream; it computes no gradient for the
//! embedding table or the norm weights.
use crate::model::Model;
use crate::ops::{rms_norm, Scale};
use crate::tensor::{require, Matrix, Result};
use crate::train::grad::rms_norm_backward;

/// A view of the frozen head of a model.
#[derive(Debug, Clone, Copy)]
pub struct Head<'a> {
    embedding: &'a Matrix,
    norm: &'a [f32],
    eps: f32,
}

/// What the head's forward computed, kept for its backward.
#[derive(Debug, Clone, PartialEq)]
pub struct HeadForward {
    /// The final-normalized hidden state `[n, hidden]`.
    pub normalized: Vec<f32>,
    /// `[n, vocab]`.
    pub logits: Vec<f32>,
}

impl<'a> Head<'a> {
    pub fn new(model: &'a Model) -> Self {
        Self {
            embedding: &model.embedding,
            norm: &model.norm,
            eps: model.config.rms_norm_eps,
        }
    }
    pub fn hidden(&self) -> usize {
        self.embedding.cols()
    }
    pub fn vocab(&self) -> usize {
        self.embedding.rows()
    }
    /// `h [n, hidden]` → normalized and logits, as `Model::step` computes them
    /// for one position (up to the order of the sums in the LM head).
    pub fn forward(&self, h: &[f32], n: usize) -> Result<HeadForward> {
        let d = self.hidden();
        require(h.len() == n * d, "head input shape mismatch")?;
        let mut normalized = Vec::with_capacity(h.len());
        for x in h.chunks_exact(d) {
            normalized.extend(rms_norm(x, self.norm, self.eps, Scale::QwenZeroCentered)?);
        }
        let logits = self.embedding.project_batch(&normalized, n)?;
        Ok(HeadForward { normalized, logits })
    }
    /// Gradient of the loss with respect to `h [n, hidden]`, from the
    /// gradient with respect to the logits `[n, vocab]`.
    ///
    /// `logits = normalized · Eᵀ`, so `dnormalized = dlogits · E`: each
    /// position's gradient is a mix of the embedding rows, weighted by how
    /// much each ID's logit should move. Then back through the RMSNorm.
    pub fn backward(&self, h: &[f32], dlogits: &[f32], n: usize) -> Result<Vec<f32>> {
        let d = self.hidden();
        require(
            h.len() == n * d && dlogits.len() == n * self.vocab(),
            "head backward shape mismatch",
        )?;
        let dnormalized = self.embedding.project_transposed_batch(dlogits, n)?;
        let mut dh = Vec::with_capacity(h.len());
        for (x, dy) in h.chunks_exact(d).zip(dnormalized.chunks_exact(d)) {
            dh.extend(rms_norm_backward(x, self.norm, self.eps, Scale::QwenZeroCentered, dy)?.dx);
        }
        Ok(dh)
    }
}
