//! Optimizers: how a gradient becomes a change of the parameters.
//!
//! - SGD: `p ← p − lr·g`. The step is proportional to the gradient.
//! - Adam: keeps per-parameter running means of `g` (m) and `g²` (v), and
//!   steps by `lr · m̂ / (sqrt(v̂) + eps)`. Every parameter then moves by
//!   about `lr` regardless of the scale of its gradient, which is why one
//!   learning rate works for gate, up and down alike. `m̂`, `v̂` correct the
//!   bias of starting m and v at zero.
//!
//! Adam keeps two f32 values per parameter: for the last layer's FFN
//! (about 11 million parameters) that is about 88 MB, small next to the
//! frozen model. For all parameters it would be the "dozen GB" of the plan.
use crate::tensor::{require, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum OptimizerConfig {
    Sgd {
        lr: f32,
    },
    Adam {
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
    },
}
impl OptimizerConfig {
    /// Adam with the usual β1 = 0.9, β2 = 0.999, ε = 1e-8.
    pub fn adam(lr: f32) -> Self {
        Self::Adam {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }
}

/// An optimizer with its state, for a fixed list of parameter tensors.
#[derive(Debug, Clone, PartialEq)]
pub struct Optimizer {
    pub config: OptimizerConfig,
    /// Steps taken so far.
    pub t: u64,
    /// Adam's first and second moments, one per parameter tensor (empty for SGD).
    pub m: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
}

impl Optimizer {
    /// `sizes`: the number of elements of each parameter tensor.
    pub fn new(config: OptimizerConfig, sizes: &[usize]) -> Self {
        let zeros = || sizes.iter().map(|&n| vec![0.; n]).collect();
        let (m, v) = match config {
            OptimizerConfig::Sgd { .. } => (Vec::new(), Vec::new()),
            OptimizerConfig::Adam { .. } => (zeros(), zeros()),
        };
        Self { config, t: 0, m, v }
    }
    /// One update of every parameter tensor from its gradient.
    pub fn step(&mut self, params: &mut [&mut [f32]], grads: &[&[f32]]) -> Result<()> {
        require(
            params.len() == grads.len(),
            "optimizer tensor count mismatch",
        )?;
        require(
            params.iter().zip(grads).all(|(p, g)| p.len() == g.len()),
            "optimizer shape mismatch",
        )?;
        require(
            grads.iter().all(|g| g.iter().all(|v| v.is_finite())),
            "non-finite gradient",
        )?;
        self.t += 1;
        match self.config {
            OptimizerConfig::Sgd { lr } => {
                for (p, g) in params.iter_mut().zip(grads) {
                    for (p, g) in p.iter_mut().zip(*g) {
                        *p -= lr * g;
                    }
                }
            }
            OptimizerConfig::Adam {
                lr,
                beta1,
                beta2,
                eps,
            } => {
                require(
                    self.m.len() == params.len()
                        && self
                            .m
                            .iter()
                            .zip(params.iter())
                            .all(|(m, p)| m.len() == p.len()),
                    "optimizer state shape mismatch",
                )?;
                let c1 = 1. - (beta1 as f64).powi(self.t as i32);
                let c2 = 1. - (beta2 as f64).powi(self.t as i32);
                // lr·m̂/(√v̂+ε) = (lr/c1)·m / (√v/√c2 + ε)
                let step = (lr as f64 / c1) as f32;
                let root_c2 = c2.sqrt() as f32;
                for (((p, g), m), v) in params
                    .iter_mut()
                    .zip(grads)
                    .zip(&mut self.m)
                    .zip(&mut self.v)
                {
                    for (((p, &g), m), v) in p.iter_mut().zip(*g).zip(m).zip(v) {
                        *m = beta1 * *m + (1. - beta1) * g;
                        *v = beta2 * *v + (1. - beta2) * g * g;
                        *p -= step * *m / (v.sqrt() / root_c2 + eps);
                    }
                }
            }
        }
        Ok(())
    }
}
