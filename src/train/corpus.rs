//! Training text: many sequences, scored where they should be learned.
//!
//! Distillation from text: the teacher answers a set of prompts, and the
//! student learns the teacher's answers by cross-entropy, the same loss as
//! teaching one fact (M2), only over many answers. Each pair is
//! `user_chat(prompt)` + response + `<|im_end|>`, and only the response's
//! positions (and the end of the turn) are scored. A pair can instead name
//! the part of the response to score (`score`, for example only the city in
//! a fixed sentence), so the loss is not spent on the words around it. A
//! pair can also carry the teacher's top K at the scored positions
//! (`teacher`): then the student learns the teacher's distribution there (KL
//! over the top K + "other"), not only its one chosen ID. A plain text can be
//! learned too: every position is scored.
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
use crate::train::grad::SparseTarget;
use crate::train::metrics::argmax;
use crate::train::teacher::TeacherPosition;
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
    /// The teacher's distribution at each scored position, in order.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub teacher: Option<PairTeacher>,
    /// The turn opens the thinking block (`user_chat(prompt, true)`); the
    /// response is then the thinking, `</think>` and the reply.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub thinking: bool,
}

/// What the teacher predicted while reading this pair.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairTeacher {
    /// The teacher, its quantization and the engine version.
    pub name: String,
    /// The IDs the teacher read: the chat turn, the response and the end of
    /// the turn, exactly as the student reads them.
    pub ids: Vec<u32>,
    /// One per scored position, in order: log-probabilities over the whole
    /// vocabulary (the teacher and the student share it).
    pub positions: Vec<TeacherPosition>,
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
    /// The teacher's distribution at each scored position, if any.
    pub teacher: Option<Vec<SparseTarget>>,
}

/// The scored positions of a set of sequences, ready to train on.
#[derive(Debug, Clone, PartialEq)]
pub struct Corpus {
    /// The last layer's FFN inputs at the scored positions only.
    pub inputs: TopInputs,
    /// One per scored position; `position` indexes `inputs`. The training
    /// targets: the next ID, or the teacher's distribution.
    pub examples: Vec<Example>,
    /// The same positions with the text's own next ID, for scoring.
    pub labels: Vec<Example>,
    /// `(first example, count)` of each sequence.
    pub sequences: Vec<(usize, usize)>,
    /// Total tokens read (scored or not).
    pub tokens: usize,
}

impl Corpus {
    /// From token sequences and, for each, the `(position, next ID)` pairs to
    /// score. Each sequence is read once by the inference path.
    pub fn from_sequences(model: &Model, sequences: &[Sequence], backend: Backend) -> Result<Self> {
        Self::from_sequences_with(model, sequences, |ids| {
            Ok(LayerCache::capture(model, ids, backend)?.inputs)
        })
    }
    /// As `from_sequences`, with the last layer's inputs of every position
    /// of a sequence from `capture` (for example a GPU prefill that reads
    /// them back, `DeviceRun::set_capture_top_inputs`).
    pub fn from_sequences_with(
        model: &Model,
        sequences: &[Sequence],
        mut capture: impl FnMut(&[u32]) -> Result<TopInputs>,
    ) -> Result<Self> {
        require(!sequences.is_empty(), "no training sequences")?;
        let hidden = model.config().hidden_size;
        let mut after_mixer = Vec::new();
        let mut post_norm = Vec::new();
        let mut examples = Vec::new();
        let mut labels = Vec::new();
        let mut spans = Vec::new();
        let mut tokens = 0;
        for Sequence {
            ids,
            scored,
            teacher,
        } in sequences
        {
            require(!scored.is_empty(), "a sequence has no scored position")?;
            require(
                teacher.as_ref().is_none_or(|t| t.len() == scored.len()),
                "the teacher must cover every scored position",
            )?;
            let inputs = capture(ids)?;
            require(
                inputs.n == ids.len()
                    && inputs.after_mixer.len() == ids.len() * hidden
                    && inputs.post_norm.len() == ids.len() * hidden,
                "captured inputs do not cover the sequence",
            )?;
            let picks: Vec<usize> = scored.iter().map(|&(p, _)| p).collect();
            let sub = inputs.subset(&picks)?;
            spans.push((examples.len(), scored.len()));
            for (k, &(_, id)) in scored.iter().enumerate() {
                let position = examples.len();
                labels.push(Example {
                    position,
                    target: Target::Id(id),
                });
                examples.push(Example {
                    position,
                    target: match teacher {
                        Some(t) => Target::Sparse(t[k].clone()),
                        None => Target::Id(id),
                    },
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
            labels,
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
        Self::from_sequences(model, &pair_sequences(model, codec, pairs, end)?, backend)
    }
}

/// Each pair as a chat turn with its scored positions (and teacher, if any).
pub fn pair_sequences(
    model: &Model,
    codec: &dyn TextCodec,
    pairs: &[Pair],
    end: u32,
) -> Result<Vec<Sequence>> {
    pairs
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let at = |e: InferenceError| InferenceError(format!("pair {}: {}", i + 1, e.0));
            let f = Fact::chat(codec, &p.prompt, &p.response, end, p.thinking).map_err(at)?;
            let examples = match &p.score {
                None => f.examples(),
                Some(part) => f.examples_of(
                    f.span(codec, &p.prompt, &p.response, part, p.thinking)
                        .map_err(at)?,
                ),
            };
            let scored: Vec<(usize, u32)> = examples
                .into_iter()
                .map(|e| match e.target {
                    Target::Id(id) => (e.position, id),
                    Target::Sparse(_) => unreachable!("facts have ID targets"),
                })
                .collect();
            let teacher = p
                .teacher
                .as_ref()
                .map(|t| teacher_targets(t, &f.ids, &scored, model.config().vocab_size))
                .transpose()
                .map_err(at)?;
            Ok(Sequence {
                ids: f.ids,
                scored,
                teacher,
            })
        })
        .collect::<Result<Vec<_>>>()
}

impl Corpus {
    /// A plain text: every position predicts the next ID.
    pub fn from_text(model: &Model, ids: &[u32], backend: Backend) -> Result<Self> {
        require(ids.len() >= 2, "text too short to score")?;
        let scored = (0..ids.len() - 1).map(|p| (p, ids[p + 1])).collect();
        Self::from_sequences(
            model,
            &[Sequence {
                ids: ids.to_vec(),
                scored,
                teacher: None,
            }],
            backend,
        )
    }
}

const CORPUS_MAGIC: &[u8] = b"QWTRAIN-CORPUS-1\n";

/// What a saved corpus holds besides the inputs (ID targets only).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CorpusHeader {
    model_identity: String,
    /// sha256 of the file the corpus was read from (the pairs), so a cache
    /// of other pairs is refused.
    source_sha256: String,
    hidden: usize,
    /// `(position, target ID)` per example; the labels are the same.
    examples: Vec<(usize, u32)>,
    sequences: Vec<(usize, usize)>,
    tokens: usize,
    data_sha256: String,
}

fn corpus_io(e: std::io::Error) -> InferenceError {
    InferenceError(format!("corpus cache I/O: {e}"))
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(bytes))
}

impl Corpus {
    /// Saves the corpus, so reading the same pairs again (24 layers per
    /// position) can be skipped. File: magic, u64 LE header length, JSON
    /// header, then `after_mixer` and `post_norm` as f32 LE. Teacher
    /// (distribution) targets are not saved.
    pub fn write(&self, path: &std::path::Path, model: &Model, source: &[u8]) -> Result<()> {
        use std::io::Write;
        let examples = self
            .examples
            .iter()
            .map(|e| match e.target {
                Target::Id(id) => Ok((e.position, id)),
                Target::Sparse(_) => Err(InferenceError(
                    "a corpus with teacher distributions is not cached".into(),
                )),
            })
            .collect::<Result<Vec<_>>>()?;
        let data: Vec<u8> = self
            .inputs
            .after_mixer
            .iter()
            .chain(&self.inputs.post_norm)
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let header = CorpusHeader {
            model_identity: model.identity().into(),
            source_sha256: sha256_hex(source),
            hidden: model.config().hidden_size,
            examples,
            sequences: self.sequences.clone(),
            tokens: self.tokens,
            data_sha256: sha256_hex(&data),
        };
        let h = serde_json::to_vec(&header).map_err(|e| InferenceError(e.to_string()))?;
        let mut f = std::io::BufWriter::new(std::fs::File::create(path).map_err(corpus_io)?);
        f.write_all(CORPUS_MAGIC).map_err(corpus_io)?;
        f.write_all(&(h.len() as u64).to_le_bytes())
            .map_err(corpus_io)?;
        f.write_all(&h).map_err(corpus_io)?;
        f.write_all(&data).map_err(corpus_io)?;
        f.flush().map_err(corpus_io)
    }
    /// Reads a corpus saved by `write`; refuses one made with another model
    /// or from another source file, or whose data does not match its hash.
    pub fn read(path: &std::path::Path, model: &Model, source: &[u8]) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(corpus_io)?;
        require(bytes.starts_with(CORPUS_MAGIC), "not a corpus cache")?;
        let at = CORPUS_MAGIC.len();
        require(bytes.len() >= at + 8, "corpus cache truncated")?;
        let len = u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap()) as usize;
        let body = at + 8 + len;
        require(bytes.len() >= body, "corpus cache truncated")?;
        let h: CorpusHeader = serde_json::from_slice(&bytes[at + 8..body])
            .map_err(|e| InferenceError(format!("corpus cache header: {e}")))?;
        require(
            h.model_identity == model.identity(),
            "corpus cache was made with another model",
        )?;
        require(
            h.source_sha256 == sha256_hex(source),
            "corpus cache was made from another file",
        )?;
        let data = &bytes[body..];
        require(
            h.data_sha256 == sha256_hex(data),
            "corpus cache data does not match its hash",
        )?;
        let n = h.examples.len();
        require(
            data.len() == 2 * n * h.hidden * 4,
            "corpus cache size mismatch",
        )?;
        let floats: Vec<f32> = data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let (after_mixer, post_norm) = floats.split_at(n * h.hidden);
        let examples: Vec<Example> = h
            .examples
            .iter()
            .enumerate()
            .map(|(i, &(position, id))| {
                require(position == i, "corpus cache positions out of order")?;
                Ok(Example {
                    position,
                    target: Target::Id(id),
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            inputs: TopInputs {
                after_mixer: after_mixer.to_vec(),
                post_norm: post_norm.to_vec(),
                n,
            },
            labels: examples.clone(),
            examples,
            sequences: h.sequences,
            tokens: h.tokens,
        })
    }
}

/// The teacher's distributions for one pair, checked against what the
/// student reads: the same IDs, and exactly the scored positions in order.
fn teacher_targets(
    t: &PairTeacher,
    ids: &[u32],
    scored: &[(usize, u32)],
    vocab: usize,
) -> Result<Vec<SparseTarget>> {
    require(
        t.ids == ids,
        "the teacher read other IDs than the student: not the same token sequence",
    )?;
    require(
        t.positions.len() == scored.len()
            && t.positions
                .iter()
                .zip(scored)
                .all(|(p, s)| p.position == s.0),
        "the teacher's positions are not the scored positions",
    )?;
    t.positions
        .iter()
        .map(|p| SparseTarget::new(p.ids.clone(), p.log_p.clone(), vocab))
        .collect()
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
