//! Training text: many sequences, scored where they should be learned.
//!
//! Distillation from text: the teacher answers a set of prompts, and the
//! student learns the teacher's answers by cross-entropy, the same loss as
//! teaching one fact (M2), only over many answers. Each pair is
//! `user_chat(prompt)` + response + `<|im_end|>`, and only the response's
//! positions (and the end of the turn) are scored. A pair can instead name
//! the part of the response to score (`score`, for example only the city in
//! a fixed sentence), so the loss is not spent on the words around it. A
//! plain text can be learned too: every position is scored.
//!
//! The top layer's positions do not interact (`TopInputs::subset`), so the
//! scored positions of all sequences are put into one `TopInputs` and
//! trained together. Pairs held out of training are scored the same way, to
//! see whether what was learned carries over to prompts not trained on.
use crate::model::Model;
use crate::tensor::{require, Backend, InferenceError, Result};
use crate::text::TextCodec;
use crate::train::cache::LayerCache;
use crate::train::ffn::{TopFfn, TopInputs};
use crate::train::fit::{Example, Fact, Target};
use crate::train::grad::log_softmax;
use crate::train::metrics::argmax;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// One prompt and the answer to learn (one JSON object per line).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pair {
    pub prompt: String,
    pub response: String,
    /// The part of `response` to score (it must occur once, on token
    /// boundaries). Absent: the whole response and the end of the turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<String>,
}

/// Reads JSON Lines of `Pair`; blank lines are skipped.
pub fn parse_pairs(text: &str) -> Result<Vec<Pair>> {
    text.lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(n, l)| {
            serde_json::from_str(l)
                .map_err(|e| InferenceError(format!("pairs line {}: {e}", n + 1)))
        })
        .collect()
}

/// A token sequence and the `(position, next ID)` pairs to score in it.
#[derive(Debug, Clone, PartialEq)]
pub struct Sequence {
    pub ids: Vec<u32>,
    pub scored: Vec<(usize, u32)>,
}

/// The scored positions of a set of sequences, ready to train on.
#[derive(Debug, Clone, PartialEq)]
pub struct Corpus {
    /// The last layer's FFN inputs at the scored positions only.
    pub inputs: TopInputs,
    /// One per scored position; `position` indexes `inputs`.
    pub examples: Vec<Example>,
    /// `(first example, count)` of each sequence.
    pub sequences: Vec<(usize, usize)>,
    /// Total tokens read (scored or not).
    pub tokens: usize,
}

impl Corpus {
    /// From token sequences and, for each, the `(position, next ID)` pairs to
    /// score. Each sequence is read once by the inference path.
    pub fn from_sequences(model: &Model, sequences: &[Sequence], backend: Backend) -> Result<Self> {
        require(!sequences.is_empty(), "no training sequences")?;
        let mut after_mixer = Vec::new();
        let mut post_norm = Vec::new();
        let mut examples = Vec::new();
        let mut spans = Vec::new();
        let mut tokens = 0;
        for Sequence { ids, scored } in sequences {
            require(!scored.is_empty(), "a sequence has no scored position")?;
            let cache = LayerCache::capture(model, ids, backend)?;
            let picks: Vec<usize> = scored.iter().map(|&(p, _)| p).collect();
            let sub = cache.inputs.subset(&picks)?;
            spans.push((examples.len(), scored.len()));
            for &(_, id) in scored {
                examples.push(Example {
                    position: examples.len(),
                    target: Target::Id(id),
                });
            }
            after_mixer.extend(sub.after_mixer);
            post_norm.extend(sub.post_norm);
            tokens += ids.len();
        }
        Ok(Self {
            inputs: TopInputs {
                after_mixer,
                post_norm,
                n: examples.len(),
            },
            examples,
            sequences: spans,
            tokens,
        })
    }
    /// Each pair as a chat turn; the response and the end of the turn are
    /// scored, or only the tokens of its `score` part.
    pub fn from_pairs(
        model: &Model,
        codec: &dyn TextCodec,
        pairs: &[Pair],
        end: u32,
        backend: Backend,
    ) -> Result<Self> {
        let sequences = pairs
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let at = |e: InferenceError| InferenceError(format!("pair {}: {}", i + 1, e.0));
                let f = Fact::new(codec, &p.prompt, &p.response, end).map_err(at)?;
                let examples = match &p.score {
                    None => f.examples(),
                    Some(part) => {
                        f.examples_of(f.span(codec, &p.prompt, &p.response, part).map_err(at)?)
                    }
                };
                let scored = examples
                    .into_iter()
                    .map(|e| match e.target {
                        Target::Id(id) => (e.position, id),
                        Target::Sparse(_) => unreachable!("facts have ID targets"),
                    })
                    .collect();
                Ok(Sequence { ids: f.ids, scored })
            })
            .collect::<Result<Vec<_>>>()?;
        Self::from_sequences(model, &sequences, backend)
    }
    /// A plain text: every position predicts the next ID.
    pub fn from_text(model: &Model, ids: &[u32], backend: Backend) -> Result<Self> {
        require(ids.len() >= 2, "text too short to score")?;
        let scored = (0..ids.len() - 1).map(|p| (p, ids[p + 1])).collect();
        Self::from_sequences(
            model,
            &[Sequence {
                ids: ids.to_vec(),
                scored,
            }],
            backend,
        )
    }
}

/// Mean cross-entropy and top-1 accuracy over examples with ID targets.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Score {
    pub positions: usize,
    /// Mean −log q(target), in nats.
    pub loss: f64,
    /// Share of positions whose most likely ID is the target.
    pub top1: f64,
}

/// Scores `examples` under `top`, `batch` positions at a time.
pub fn score(
    top: &TopFfn,
    inputs: &TopInputs,
    examples: &[Example],
    batch: usize,
) -> Result<Score> {
    require(!examples.is_empty() && batch > 0, "nothing to score")?;
    let vocab = top.head.vocab();
    let mut loss = 0.;
    let mut hits = 0;
    for chunk in examples.chunks(batch) {
        let picks: Vec<usize> = chunk.iter().map(|e| e.position).collect();
        let f = top.forward(&inputs.subset(&picks)?)?;
        let each = chunk
            .par_iter()
            .zip(f.head.logits.par_chunks_exact(vocab))
            .map(|(e, logits)| match &e.target {
                Target::Id(id) => Ok((
                    -log_softmax(logits)?[*id as usize],
                    argmax(logits) == *id as usize,
                )),
                Target::Sparse(_) => Err(InferenceError("score takes ID targets".into())),
            })
            .collect::<Result<Vec<_>>>()?;
        for (l, hit) in each {
            loss += l;
            hits += usize::from(hit);
        }
    }
    let n = examples.len();
    Ok(Score {
        positions: n,
        loss: loss / n as f64,
        top1: hits as f64 / n as f64,
    })
}
