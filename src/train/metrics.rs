//! What training changed, measured on the logits (CPU).
//!
//! The same definitions as the GPU quality study of volume 2, chapter 12
//! (`device/tests.rs`: `study_log_softmax`, `QualityStudy::evaluate`), so the
//! numbers can be read side by side:
//!
//! - perplexity: `exp(mean −log q(next ID))` over the scored positions;
//! - KL(ref‖new): how far the new distribution moved from the original one,
//!   in nats, per position;
//! - top-1 agreement: the share of positions whose most likely ID is the same.
use crate::tensor::{require, Result};
use crate::train::grad::log_softmax;
use serde::Serialize;

/// The index of the largest logit (the first one on ties).
pub fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |best, (i, &v)| {
            if v > best.1 {
                (i, v)
            } else {
                best
            }
        })
        .0
}

/// The rank of `id`: how many IDs have a strictly larger logit (0 = top).
pub fn rank(logits: &[f32], id: u32) -> usize {
    let v = logits[id as usize];
    logits.iter().filter(|&&x| x > v).count()
}

/// KL(ref‖new) for one position, in nats.
pub fn kl(reference: &[f32], new: &[f32]) -> Result<f64> {
    require(reference.len() == new.len(), "KL shape mismatch")?;
    let (p, q) = (log_softmax(reference)?, log_softmax(new)?);
    Ok(p.iter()
        .zip(&q)
        .map(|(&lp, &lq)| lp.exp() * (lp - lq))
        .sum::<f64>()
        .max(0.))
}

/// The comparison of two models' logits over the same positions.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Comparison {
    pub positions: usize,
    /// Positions with a next ID (used for perplexity).
    pub scored: usize,
    pub perplexity_reference: f64,
    pub perplexity_new: f64,
    pub mean_kl: f64,
    pub max_kl: f64,
    pub top1_agreement: f64,
}

/// Accumulates `Comparison` one position at a time, so the logits of a long
/// text never have to be held at once.
#[derive(Debug, Clone, Default)]
pub struct Comparer {
    positions: usize,
    scored: usize,
    nll_reference: f64,
    nll_new: f64,
    kl_sum: f64,
    kl_max: f64,
    agree: usize,
}
/// One position's contribution to a `Comparison`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PositionStats {
    pub kl: f64,
    pub same_top1: bool,
    /// −log q(next) under each model, when the position has a next ID.
    pub nll: Option<(f64, f64)>,
}
impl PositionStats {
    /// `next`: the ID that follows this position in the text, if any.
    pub fn new(reference: &[f32], new: &[f32], next: Option<u32>) -> Result<Self> {
        require(reference.len() == new.len(), "KL shape mismatch")?;
        let (p, q) = (log_softmax(reference)?, log_softmax(new)?);
        let kl = p
            .iter()
            .zip(&q)
            .map(|(&lp, &lq)| lp.exp() * (lp - lq))
            .sum::<f64>()
            .max(0.);
        let nll = match next {
            None => None,
            Some(id) => {
                require((id as usize) < reference.len(), "next ID out of range")?;
                Some((-p[id as usize], -q[id as usize]))
            }
        };
        Ok(Self {
            kl,
            same_top1: argmax(reference) == argmax(new),
            nll,
        })
    }
}
impl Comparer {
    /// `next`: the ID that follows this position in the text, if any.
    pub fn add(&mut self, reference: &[f32], new: &[f32], next: Option<u32>) -> Result<()> {
        self.push(PositionStats::new(reference, new, next)?);
        Ok(())
    }
    /// Adds a position computed elsewhere (for example in parallel). Push in
    /// position order to get the same sums as `add`.
    pub fn push(&mut self, s: PositionStats) {
        self.positions += 1;
        self.kl_sum += s.kl;
        self.kl_max = self.kl_max.max(s.kl);
        self.agree += usize::from(s.same_top1);
        if let Some((a, b)) = s.nll {
            self.scored += 1;
            self.nll_reference += a;
            self.nll_new += b;
        }
    }
    pub fn finish(&self) -> Result<Comparison> {
        require(self.positions > 0, "nothing compared")?;
        let n = self.positions as f64;
        let ppl = |nll: f64| {
            if self.scored == 0 {
                f64::NAN
            } else {
                (nll / self.scored as f64).exp()
            }
        };
        Ok(Comparison {
            positions: self.positions,
            scored: self.scored,
            perplexity_reference: ppl(self.nll_reference),
            perplexity_new: ppl(self.nll_new),
            mean_kl: self.kl_sum / n,
            max_kl: self.kl_max,
            top1_agreement: self.agree as f64 / n,
        })
    }
}
