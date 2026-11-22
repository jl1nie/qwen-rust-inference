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

/// As `fit`, but each step averages the gradient over `sample` examples
/// drawn without replacement (minibatch SGD), not over all of them: for
/// corpora of tens of thousands of positions. The draw is a fixed function
/// of `seed` and the step, so a run repeats exactly. The logged loss is the
/// mean over that step's sample.
#[allow(clippy::too_many_arguments)]
pub fn fit_sampled(
    top: &mut TopFfn,
    inputs: &TopInputs,
    examples: &[Example],
    optimizer: &mut Optimizer,
    steps: usize,
    batch: usize,
    sample: usize,
    seed: u64,
    mut log: impl FnMut(&StepLog) -> Result<()>,
) -> Result<()> {
    require(sample > 0, "sample size must be positive")?;
    let mut rng = SplitMix64(seed);
    let mut order: Vec<usize> = (0..examples.len()).collect();
    for _ in 0..steps {
        let k = sample.min(order.len());
        // Partial Fisher-Yates: the first k of `order` are the step's sample.
        for i in 0..k {
            let j = i + (rng.next() % (order.len() - i) as u64) as usize;
            order.swap(i, j);
        }
        let mut picked: Vec<usize> = order[..k].to_vec();
        picked.sort_unstable();
        let chosen: Vec<Example> = picked.iter().map(|&i| examples[i].clone()).collect();
        let (loss, grad) = mean_loss_and_grad(top, inputs, &chosen, batch)?;
        log(&StepLog {
            step: optimizer.t + 1,
            loss,
            grad_norm: grad.norm(),
        })?;
        optimizer.step(&mut top.ffn.params_mut(), &grad.slices())?;
    }
    Ok(())
}

/// A small deterministic generator (SplitMix64) for drawing samples.
struct SplitMix64(u64);
impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
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
        Self::chat(tokenizer, prompt, answer, end, false)
    }
    /// As `new`; with `thinking` the turn opens the thinking block
    /// (`user_chat(prompt, true)`), so `answer` is the thinking, `</think>`
    /// and the reply.
    pub fn chat(
        tokenizer: &dyn TextCodec,
        prompt: &str,
        answer: &str,
        end: u32,
        thinking: bool,
    ) -> Result<Self> {
        let chat = user_chat(prompt, thinking);
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
        self.examples_of(self.prompt_len..self.ids.len())
    }
    /// The IDs `tokens` (indices into `ids`, after the prompt) as targets,
    /// each predicted from the position before it.
    pub fn examples_of(&self, tokens: std::ops::Range<usize>) -> Vec<Example> {
        (tokens.start - 1..tokens.end - 1)
            .map(|p| Example {
                position: p,
                target: Target::Id(self.ids[p + 1]),
            })
            .collect()
    }
    /// The tokens of `part` of the answer, as a range of `ids`. `part` must
    /// occur once in the answer and start and end on token boundaries (the
    /// answer up to either end encodes to a prefix of `ids`), so exactly its
    /// tokens are scored and no token is shared with the words around it.
    pub fn span(
        &self,
        tokenizer: &dyn TextCodec,
        prompt: &str,
        answer: &str,
        part: &str,
        thinking: bool,
    ) -> Result<std::ops::Range<usize>> {
        require(!part.is_empty(), "the part to score is empty")?;
        let mut found = answer.match_indices(part);
        let start = found
            .next()
            .ok_or_else(|| crate::tensor::InferenceError(format!("{part:?} is not in the answer")))?
            .0;
        require(
            found.next().is_none(),
            "the part to score must occur once in the answer",
        )?;
        let chat = user_chat(prompt, thinking);
        let upto = |end: usize| tokenizer.encode(&format!("{chat}{}", &answer[..end]));
        let (a, b) = (upto(start)?, upto(start + part.len())?);
        require(
            self.ids.starts_with(&a) && self.ids.starts_with(&b) && a.len() >= self.prompt_len,
            &format!("{part:?} does not start and end on token boundaries"),
        )?;
        require(b.len() > a.len(), "the part to score has no token")?;
        Ok(a.len()..b.len())
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
