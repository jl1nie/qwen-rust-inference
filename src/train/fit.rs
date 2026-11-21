//! The training loop: score the chosen positions, average the gradient,
//! update, repeat.
//!
//! Each step is one full pass over the examples (full-batch gradient
//! descent). The examples are cut into minibatches only to bound memory, as
//! the logits take `vocab` floats per position (about 1 MB on the real
//! model). The minibatch gradients are summed before the single update, so
//! the result does not depend on the minibatch size, up to rounding.
use crate::tensor::{require, Result};
use crate::text::{user_chat, TextCodec};
use crate::train::ffn::{FfnGrad, TopFfn, TopInputs};
use crate::train::grad::{cross_entropy, kl_sparse, LossGrad, SparseTarget};
use crate::train::metrics::PositionStats;
use crate::train::optim::Optimizer;
use rayon::prelude::*;
use serde::Serialize;

/// What one position should predict.
#[derive(Debug, Clone, PartialEq)]
pub enum Target {
    /// One correct ID (cross-entropy).
    Id(u32),
    /// A teacher distribution over the top K IDs plus "other" (KL).
    Sparse(SparseTarget),
}
impl Target {
    pub fn loss(&self, logits: &[f32]) -> Result<LossGrad> {
        match self {
            Self::Id(id) => cross_entropy(logits, *id),
            Self::Sparse(t) => {
                // The teacher's distribution may cover only the first IDs
                // (the vocabulary both models share): the student is compared
                // on the same IDs, renormalized there, and the rows past them
                // get no gradient.
                require(
                    t.vocab() <= logits.len(),
                    "teacher vocabulary exceeds the logits",
                )?;
                let mut g = kl_sparse(t, &logits[..t.vocab()])?;
                g.dlogits.resize(logits.len(), 0.);
                Ok(g)
            }
        }
    }
}

/// A position of the cached inputs and its target.
#[derive(Debug, Clone, PartialEq)]
pub struct Example {
    pub position: usize,
    pub target: Target,
}

/// Mean loss over the examples and the gradient of that mean.
pub fn mean_loss_and_grad(
    top: &TopFfn,
    inputs: &TopInputs,
    examples: &[Example],
    batch: usize,
) -> Result<(f64, FfnGrad)> {
    require(!examples.is_empty(), "no training examples")?;
    require(batch > 0, "batch size must be positive")?;
    let mut grad = top.ffn.zero_grad()?;
    let mut total = 0.;
    for chunk in examples.chunks(batch) {
        let picks: Vec<usize> = chunk.iter().map(|e| e.position).collect();
        let sub = inputs.subset(&picks)?;
        total += top.accumulate(&sub, |i, logits| chunk[i].target.loss(logits), &mut grad)?;
    }
    let n = examples.len() as f64;
    grad.scale((1. / n) as f32);
    Ok((total / n, grad))
}

/// One line of the training log.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StepLog {
    /// The optimizer step this loss was measured before (1 = first update).
    pub step: u64,
    /// Mean loss over the examples, before this step's update.
    pub loss: f64,
    pub grad_norm: f64,
}

/// `steps` updates of `top.ffn`. `log` sees every step.
pub fn fit(
    top: &mut TopFfn,
    inputs: &TopInputs,
    examples: &[Example],
    optimizer: &mut Optimizer,
    steps: usize,
    batch: usize,
    mut log: impl FnMut(&StepLog) -> Result<()>,
) -> Result<()> {
    for _ in 0..steps {
        let (loss, grad) = mean_loss_and_grad(top, inputs, examples, batch)?;
        log(&StepLog {
            step: optimizer.t + 1,
            loss,
            grad_norm: grad.norm(),
        })?;
        optimizer.step(&mut top.ffn.params_mut(), &grad.slices())?;
    }
    Ok(())
}

/// A question and the answer to teach, as token IDs.
#[derive(Debug, Clone, PartialEq)]
pub struct Fact {
    /// The chat-formatted prompt followed by the answer and the end-of-turn ID.
    pub ids: Vec<u32>,
    /// Length of the prompt part.
    pub prompt_len: usize,
}
impl Fact {
    /// `user_chat(prompt)` then `answer`, then `end` (the end-of-turn ID), so
    /// the model also learns to stop after the answer.
    pub fn new(tokenizer: &dyn TextCodec, prompt: &str, answer: &str, end: u32) -> Result<Self> {
        let chat = user_chat(prompt, false);
        let prompt_ids = tokenizer.encode(&chat)?;
        let mut ids = tokenizer.encode(&format!("{chat}{answer}"))?;
        // The answer must start a new token, or the prompt's last position
        // would not be the one that predicts the answer.
        require(
            ids.len() > prompt_ids.len() && ids.starts_with(&prompt_ids),
            "the answer merges with the prompt's last token",
        )?;
        ids.push(end);
        Ok(Self {
            ids,
            prompt_len: prompt_ids.len(),
        })
    }
    /// The answer tokens and the end ID as targets: position i predicts
    /// ids[i + 1], from the prompt's last position on.
    pub fn examples(&self) -> Vec<Example> {
        (self.prompt_len - 1..self.ids.len() - 1)
            .map(|p| Example {
                position: p,
                target: Target::Id(self.ids[p + 1]),
            })
            .collect()
    }
}

/// How likely the trained-on answer is at one of its positions.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TokenReport {
    pub position: usize,
    pub id: u32,
    pub probability: f64,
    /// 0 = the most likely ID.
    pub rank: usize,
}
impl Fact {
    /// The probability and rank of each answer ID under `top`.
    pub fn report(&self, top: &TopFfn, inputs: &TopInputs) -> Result<Vec<TokenReport>> {
        let examples = self.examples();
        let picks: Vec<usize> = examples.iter().map(|e| e.position).collect();
        let f = top.forward(&inputs.subset(&picks)?)?;
        let vocab = top.head.vocab();
        examples
            .iter()
            .zip(f.head.logits.chunks_exact(vocab))
            .map(|(e, logits)| {
                let Target::Id(id) = e.target else {
                    unreachable!("facts have ID targets")
                };
                Ok(TokenReport {
                    position: e.position,
                    id,
                    probability: crate::train::grad::log_softmax(logits)?[id as usize].exp(),
                    rank: crate::train::metrics::rank(logits, id),
                })
            })
            .collect()
    }
}

/// Compares `reference` and `new` (two FFNs on the same cached inputs) at
/// every position. `ids` are the IDs the cache was made from; position i is
/// scored against ids[i + 1] for perplexity.
pub fn compare_models(
    reference: &TopFfn,
    new: &TopFfn,
    inputs: &TopInputs,
    ids: &[u32],
    batch: usize,
) -> Result<crate::train::metrics::Comparison> {
    require(ids.len() == inputs.n, "IDs do not match the cache")?;
    require(batch > 0, "batch size must be positive")?;
    let vocab = reference.head.vocab();
    let mut c = crate::train::metrics::Comparer::default();
    let all: Vec<usize> = (0..inputs.n).collect();
    for picks in all.chunks(batch) {
        let sub = inputs.subset(picks)?;
        let (a, b) = (reference.forward(&sub)?, new.forward(&sub)?);
        let stats = picks
            .par_iter()
            .zip(a.head.logits.par_chunks_exact(vocab))
            .zip(b.head.logits.par_chunks_exact(vocab))
            .map(|((&p, ra), rb)| PositionStats::new(ra, rb, ids.get(p + 1).copied()))
            .collect::<Result<Vec<_>>>()?;
        for s in stats {
            c.push(s);
        }
    }
    c.finish()
}
