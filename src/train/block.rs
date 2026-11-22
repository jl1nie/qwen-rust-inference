//! One block for a whole sequence, and its backward (ADR 0037).
//!
//! ```text
//! after_mixer = x + mixer(RMSNorm_in(x))
//! output      = after_mixer + FFN(RMSNorm_post(after_mixer))
//! ```
//!
//! The gradient reaches `x` along both residual paths: straight through each
//! add, and through each branch. The norms, the mixer and a frozen FFN only
//! pass the gradient on. A trainable FFN (`TrainFfn`, full or LoRA) also
//! adds its parameter gradients.
//!
//! Only attention blocks are implemented. A DeltaNet block returns an error
//! until its backward exists (ADR 0037, step 2).
use crate::block::{BlockWeights, MixerWeights};
use crate::ffn::FfnWeights;
use crate::ops::{multiply, rms_norm, silu, Scale};
use crate::tensor::{require, InferenceError, Result};
use crate::train::attention::{self, AttentionForward};
use crate::train::ffn::{FfnForward, FfnGrad, TrainFfn};
use crate::train::grad::{rms_norm_backward, swiglu_backward};
use rayon::prelude::*;

/// The block's FFN as the training sees it.
#[derive(Debug, Clone, Copy)]
pub enum BlockFfn<'a> {
    /// The model's own weights; the gradient only passes through.
    Frozen(&'a FfnWeights),
    /// A trainable copy; its gradients go into the `FfnGrad` given to `backward`.
    Train(&'a TrainFfn),
}

/// The FFN part of the forward.
#[derive(Debug, Clone, PartialEq)]
pub enum FfnPass {
    Frozen {
        gate: Vec<f32>,
        up: Vec<f32>,
        product: Vec<f32>,
    },
    Train(FfnForward),
}

/// The mixer part of the forward.
#[derive(Debug, Clone, PartialEq)]
pub enum MixerPass {
    Attention(AttentionForward),
}

/// What the block's forward computed for `n` positions, kept for the
/// backward. Vectors are `[n, hidden]`.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockForward {
    pub x: Vec<f32>,
    pub mixer: MixerPass,
    pub after_mixer: Vec<f32>,
    pub post_norm: Vec<f32>,
    pub ffn: FfnPass,
    pub output: Vec<f32>,
    pub n: usize,
}

/// RMSNorm of each of the `n` rows of `x`.
fn norm_rows(x: &[f32], weight: &[f32], eps: f32, d: usize) -> Result<Vec<f32>> {
    Ok(x.par_chunks_exact(d)
        .map(|r| rms_norm(r, weight, eps, Scale::QwenZeroCentered))
        .collect::<Result<Vec<_>>>()?
        .concat())
}

/// Backward of `norm_rows` toward `x` (the norm weights are frozen).
fn norm_rows_backward(
    x: &[f32],
    weight: &[f32],
    eps: f32,
    dy: &[f32],
    d: usize,
) -> Result<Vec<f32>> {
    Ok(x.par_chunks_exact(d)
        .zip(dy.par_chunks_exact(d))
        .map(|(r, g)| rms_norm_backward(r, weight, eps, Scale::QwenZeroCentered, g).map(|g| g.dx))
        .collect::<Result<Vec<_>>>()?
        .concat())
}

fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

/// A block of the model, with the FFN either frozen or trainable.
#[derive(Debug, Clone, Copy)]
pub struct TrainBlock<'a> {
    pub weights: &'a BlockWeights,
    pub ffn: BlockFfn<'a>,
}

impl<'a> TrainBlock<'a> {
    /// The block with its own FFN, frozen.
    pub fn frozen(weights: &'a BlockWeights) -> Self {
        Self {
            weights,
            ffn: BlockFfn::Frozen(&weights.ffn),
        }
    }
    /// The block with `ffn` in place of its FFN, trained.
    pub fn with_ffn(weights: &'a BlockWeights, ffn: &'a TrainFfn) -> Self {
        Self {
            weights,
            ffn: BlockFfn::Train(ffn),
        }
    }
    /// Forward for positions `0..n` of one sequence, `x [n, hidden]`.
    pub fn forward(&self, x: &[f32], n: usize) -> Result<BlockForward> {
        let w = self.weights;
        let d = w.input_norm.len();
        require(n > 0 && x.len() == n * d, "block input shape mismatch")?;
        let input_norm = norm_rows(x, &w.input_norm, w.eps, d)?;
        let (mixer, mixed) = match &w.mixer {
            MixerWeights::Attention(a) => {
                let f = attention::forward(a, &input_norm, n)?;
                let out = f.output.clone();
                (MixerPass::Attention(f), out)
            }
            MixerWeights::Delta(_) => {
                return Err(InferenceError(
                    "DeltaNet backward is not implemented yet (ADR 0037, step 2)".into(),
                ))
            }
        };
        let after_mixer = add(x, &mixed);
        let post_norm = norm_rows(&after_mixer, &w.post_norm, w.eps, d)?;
        let (ffn, ffn_out) = match self.ffn {
            BlockFfn::Frozen(f) => {
                let [gm, um, dm] = f.matrices();
                let gate = gm.project_batch(&post_norm, n)?;
                let up = um.project_batch(&post_norm, n)?;
                let product = multiply(&silu(&gate)?, &up)?;
                let out = dm.project_batch(&product, n)?;
                (FfnPass::Frozen { gate, up, product }, out)
            }
            BlockFfn::Train(t) => {
                let f = t.forward(&post_norm, n)?;
                let out = f.output.clone();
                (FfnPass::Train(f), out)
            }
        };
        let output = add(&after_mixer, &ffn_out);
        Ok(BlockForward {
            x: x.to_vec(),
            mixer,
            after_mixer,
            post_norm,
            ffn,
            output,
            n,
        })
    }
    /// Gradient toward `x [n, hidden]` for `dout [n, hidden]`. A trainable
    /// FFN adds its gradients into `grad` (required in that case).
    pub fn backward(
        &self,
        f: &BlockForward,
        dout: &[f32],
        grad: Option<&mut FfnGrad>,
    ) -> Result<Vec<f32>> {
        let w = self.weights;
        let (d, n) = (w.input_norm.len(), f.n);
        require(dout.len() == n * d, "block backward shape mismatch")?;
        // output = after_mixer + FFN(post_norm).
        let dpost = match (self.ffn, &f.ffn) {
            (
                BlockFfn::Frozen(ffn),
                FfnPass::Frozen {
                    gate,
                    up,
                    product: _,
                },
            ) => {
                let [gm, um, dm] = ffn.matrices();
                let dproduct = dm.project_transposed_batch(dout, n)?;
                let s = swiglu_backward(gate, up, &dproduct)?;
                add(
                    &gm.project_transposed_batch(&s.dgate, n)?,
                    &um.project_transposed_batch(&s.dup, n)?,
                )
            }
            (BlockFfn::Train(t), FfnPass::Train(ff)) => {
                let g = grad.ok_or_else(|| {
                    InferenceError("a trainable FFN needs a gradient to add into".into())
                })?;
                t.backward(ff, dout, g)?
            }
            _ => return Err(InferenceError("block forward/FFN mismatch".into())),
        };
        let dafter = add(
            dout,
            &norm_rows_backward(&f.after_mixer, &w.post_norm, w.eps, &dpost, d)?,
        );
        // after_mixer = x + mixer(input_norm(x)).
        let dinput_norm = match (&w.mixer, &f.mixer) {
            (MixerWeights::Attention(a), MixerPass::Attention(af)) => {
                attention::backward(a, af, &dafter)?
            }
            _ => return Err(InferenceError("block forward/mixer mismatch".into())),
        };
        Ok(add(
            &dafter,
            &norm_rows_backward(&f.x, &w.input_norm, w.eps, &dinput_norm, d)?,
        ))
    }
}
