//! Backward of each operation, and the losses.
//!
//! Each function takes what its forward read (and, where cheaper, what it
//! wrote) plus `dy`, the gradient of the loss with respect to the forward's
//! output, and returns the gradient with respect to its inputs. Sums over a
//! vector are accumulated in f64; the results are f32 like the forward.
//!
//! The losses compare the student's logits with a target distribution. The
//! gradient with respect to the logits is always "student probability minus
//! target probability" (`q - p`); only how the target is written differs.
use crate::ops::{sigmoid, Scale};
use crate::tensor::{finite, require, Result};

/// Gradients of `ops::rms_norm(x, weight, eps, scale)`.
#[derive(Debug, Clone, PartialEq)]
pub struct RmsNormGrad {
    pub dx: Vec<f32>,
    pub dweight: Vec<f32>,
}

/// Backward of RMSNorm, `y_i = x_i · r · s_i` with `r = 1/sqrt(mean(x²) + eps)`
/// and `s_i = w_i` (`Direct`) or `1 + w_i` (`QwenZeroCentered`).
///
/// `r` depends on every `x_j`, so each `dx_j` has two parts: the direct path
/// `r·s_j·dy_j`, and the path through `r`, which is the same direction `x_j`
/// for every element: `dx_j = r·s_j·dy_j − x_j · r³/n · Σ_i dy_i·s_i·x_i`.
/// `ds_i/dw_i = 1` for both scales, so `dw_i = dy_i · x_i · r`.
pub fn rms_norm_backward(
    x: &[f32],
    weight: &[f32],
    eps: f32,
    scale: Scale,
    dy: &[f32],
) -> Result<RmsNormGrad> {
    finite(x)?;
    finite(weight)?;
    finite(dy)?;
    require(
        x.len() == weight.len() && x.len() == dy.len(),
        "RMSNorm backward shape mismatch",
    )?;
    require(eps.is_finite() && eps > 0., "invalid RMSNorm epsilon")?;
    let n = x.len() as f64;
    let s = |w: f32| match scale {
        Scale::Direct => w as f64,
        Scale::QwenZeroCentered => 1. + w as f64,
    };
    let mean = x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / n;
    let r = 1. / (mean + eps as f64).sqrt();
    let through_r: f64 = x
        .iter()
        .zip(weight)
        .zip(dy)
        .map(|((&x, &w), &dy)| dy as f64 * s(w) * x as f64)
        .sum::<f64>()
        * r.powi(3)
        / n;
    let dx: Vec<f32> = x
        .iter()
        .zip(weight)
        .zip(dy)
        .map(|((&x, &w), &dy)| (r * s(w) * dy as f64 - x as f64 * through_r) as f32)
        .collect();
    let dweight: Vec<f32> = x
        .iter()
        .zip(dy)
        .map(|(&x, &dy)| (dy as f64 * x as f64 * r) as f32)
        .collect();
    finite(&dx)?;
    finite(&dweight)?;
    Ok(RmsNormGrad { dx, dweight })
}

/// d silu(x) / dx = σ(x) · (1 + x · (1 − σ(x))).
fn silu_derivative(x: f32) -> f32 {
    let s = sigmoid(x);
    s * (1. + x * (1. - s))
}

/// Backward of `ops::silu`.
pub fn silu_backward(x: &[f32], dy: &[f32]) -> Result<Vec<f32>> {
    finite(x)?;
    finite(dy)?;
    require(x.len() == dy.len(), "SiLU backward shape mismatch")?;
    let dx: Vec<f32> = x
        .iter()
        .zip(dy)
        .map(|(&x, &dy)| silu_derivative(x) * dy)
        .collect();
    finite(&dx)?;
    Ok(dx)
}

/// Gradients of the SwiGLU product `silu(gate) ⊙ up`.
#[derive(Debug, Clone, PartialEq)]
pub struct SwiGluGrad {
    pub dgate: Vec<f32>,
    pub dup: Vec<f32>,
}

/// Backward of `product = silu(gate) ⊙ up` (the FFN's elementwise step).
/// Each side's gradient is `dproduct` times the other side's value; the gate
/// side also passes through the SiLU derivative.
pub fn swiglu_backward(gate: &[f32], up: &[f32], dproduct: &[f32]) -> Result<SwiGluGrad> {
    finite(gate)?;
    finite(up)?;
    finite(dproduct)?;
    require(
        gate.len() == up.len() && gate.len() == dproduct.len(),
        "SwiGLU backward shape mismatch",
    )?;
    let mut dgate = Vec::with_capacity(gate.len());
    let mut dup = Vec::with_capacity(gate.len());
    for ((&g, &u), &d) in gate.iter().zip(up).zip(dproduct) {
        dgate.push(d * u * silu_derivative(g));
        dup.push(d * g * sigmoid(g));
    }
    finite(&dgate)?;
    finite(&dup)?;
    Ok(SwiGluGrad { dgate, dup })
}

/// `log softmax(z)` in f64: `z_i − max − ln Σ exp(z_j − max)`.
pub fn log_softmax(logits: &[f32]) -> Result<Vec<f64>> {
    finite(logits)?;
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = logits.iter().map(|&z| (z as f64 - max).exp()).sum();
    let log_sum = sum.ln();
    Ok(logits.iter().map(|&z| z as f64 - max - log_sum).collect())
}

/// A loss value and its gradient with respect to the student's logits.
#[derive(Debug, Clone, PartialEq)]
pub struct LossGrad {
    pub loss: f64,
    pub dlogits: Vec<f32>,
}

/// Cross-entropy against one correct ID: `loss = −log q_target`,
/// `dlogits = q − onehot(target)`.
pub fn cross_entropy(logits: &[f32], target: u32) -> Result<LossGrad> {
    let t = target as usize;
    require(t < logits.len(), "target ID out of range")?;
    let log_q = log_softmax(logits)?;
    let mut dlogits: Vec<f32> = log_q.iter().map(|l| l.exp() as f32).collect();
    dlogits[t] -= 1.;
    Ok(LossGrad {
        loss: -log_q[t],
        dlogits,
    })
}

/// KL(p‖q) with the teacher's full log-distribution `log_p` (same length as
/// the logits): `Σ p_i (log p_i − log q_i)`, `dlogits = q − p`.
pub fn kl_full(log_p: &[f64], logits: &[f32]) -> Result<LossGrad> {
    require(log_p.len() == logits.len(), "KL shape mismatch")?;
    require(
        log_p.iter().all(|l| !l.is_nan() && *l <= 0.),
        "invalid teacher log-probability",
    )?;
    let log_q = log_softmax(logits)?;
    let mut loss = 0.;
    let mut dlogits = Vec::with_capacity(logits.len());
    for (&lp, &lq) in log_p.iter().zip(&log_q) {
        let p = lp.exp();
        if p > 0. {
            loss += p * (lp - lq);
        }
        dlogits.push((lq.exp() - p) as f32);
    }
    Ok(LossGrad { loss, dlogits })
}

/// A teacher distribution kept as its top K IDs plus one "other" bucket
/// holding the remaining mass (what a top-K dump such as Strata's
/// `STRATA_LOGPOS_TOPK` gives).
#[derive(Debug, Clone, PartialEq)]
pub struct SparseTarget {
    ids: Vec<u32>,
    log_p: Vec<f64>,
    other: f64,
}
impl SparseTarget {
    /// `ids` must be distinct and below `vocab`; `log_p` are log-probabilities
    /// normalized over the whole vocabulary, so their mass is at most 1. The
    /// rest becomes the "other" bucket.
    pub fn new(ids: Vec<u32>, log_p: Vec<f64>, vocab: usize) -> Result<Self> {
        require(ids.len() == log_p.len(), "top-K shape mismatch")?;
        require(!ids.is_empty(), "empty top-K")?;
        require(
            ids.iter().all(|&i| (i as usize) < vocab),
            "top-K ID out of range",
        )?;
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        require(
            sorted.windows(2).all(|w| w[0] != w[1]),
            "duplicate top-K ID",
        )?;
        require(
            log_p.iter().all(|l| !l.is_nan() && *l <= 0.),
            "invalid teacher log-probability",
        )?;
        let mass: f64 = log_p.iter().map(|l| l.exp()).sum();
        // Rounding in the teacher can push the sum slightly above 1.
        require(mass <= 1. + 1e-6, "top-K mass exceeds 1")?;
        let other = if ids.len() == vocab {
            0.
        } else {
            (1. - mass).max(0.)
        };
        Ok(Self { ids, log_p, other })
    }
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }
    pub fn log_p(&self) -> &[f64] {
        &self.log_p
    }
    /// Mass of the "other" bucket.
    pub fn other(&self) -> f64 {
        self.other
    }
}

/// KL(p‖q) over the top K IDs plus "other": the student's own distribution is
/// grouped the same way (`q_other = 1 − Σ_K q_k`), so both sides have K + 1
/// entries.
///
/// The gradient is again `q − p`, where inside "other" the teacher's mass is
/// spread in proportion to the student's own probabilities:
/// `dz_j = q_j − p_j` for j in K, `q_j − p_other · q_j / q_other` otherwise.
/// With K equal to the whole vocabulary this is `kl_full`.
pub fn kl_sparse(target: &SparseTarget, logits: &[f32]) -> Result<LossGrad> {
    require(
        target.ids.iter().all(|&i| (i as usize) < logits.len()),
        "top-K ID out of range",
    )?;
    let log_q = log_softmax(logits)?;
    let mut in_k = vec![false; logits.len()];
    let mut loss = 0.;
    let mut q_k = 0.;
    for (&id, &lp) in target.ids.iter().zip(&target.log_p) {
        let i = id as usize;
        in_k[i] = true;
        q_k += log_q[i].exp();
        let p = lp.exp();
        if p > 0. {
            loss += p * (lp - log_q[i]);
        }
    }
    // Computed from the excluded IDs rather than 1 − q_k, which loses all
    // precision when the top K hold almost all of the student's mass.
    let q_other: f64 = log_q
        .iter()
        .zip(&in_k)
        .filter(|(_, &k)| !k)
        .map(|(l, _)| l.exp())
        .sum();
    let p_other = target.other;
    if p_other > 0. {
        require(q_other > 0., "student has no mass outside the top K")?;
        loss += p_other * (p_other.ln() - q_other.ln());
    }
    let mut dlogits: Vec<f32> = log_q.iter().map(|l| l.exp() as f32).collect();
    for (&id, &lp) in target.ids.iter().zip(&target.log_p) {
        dlogits[id as usize] -= lp.exp() as f32;
    }
    if p_other > 0. {
        let spread = p_other / q_other;
        for ((d, &l), _) in dlogits
            .iter_mut()
            .zip(&log_q)
            .zip(&in_k)
            .filter(|(_, &k)| !k)
        {
            *d -= (spread * l.exp()) as f32;
        }
    }
    debug_assert!((q_k + q_other - 1.).abs() < 1e-9);
    Ok(LossGrad { loss, dlogits })
}
