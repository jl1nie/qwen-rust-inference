//! Teacher data for distillation: what a larger model predicts at every
//! position of a text, read once from a file.
//!
//! The teacher is Qwen3.8-Flash-Next run by Strata (`book/PLAN.md`). With
//! `STRATA_LOGPOS=<file>` and `STRATA_LOGPOS_TOPK=K`, Strata reads a token
//! sequence by teacher forcing and writes one tab-separated line per scored
//! position (`engine/src/core/verify.cpp`, `Verifier::window_logprobs`, at
//! Strata main `3142b66`):
//!
//! ```text
//! pos  target  logp(target)  argmax  logp(argmax)  hit  extra  without  id:logp ... (K of them)
//! ```
//!
//! The line for `pos` scores `target = ids[pos + 1]` (as
//! `terminal/tools/force_check.py` checks), which is the convention of this
//! crate: position i predicts ID i + 1. The log-probabilities are normalized
//! over Strata's whole vocabulary (248,320 logits).
//!
//! Teacher and student share the first 248,070 IDs (`COMMON_VOCAB`); the
//! teacher's extra IDs (audio, 248070..248076) are dropped and the rest is
//! renormalized over the shared IDs, so the KL compares the same events:
//! `p'_i = p_i / (1 − Σ p_dropped)`. Only dropped IDs that appear in the top
//! K are known; the mass of the others is assumed to be in "other". The
//! target's own log-probability is added when it is outside the top K.
use crate::tensor::{require, InferenceError, Result};
use crate::train::fit::{Example, Target};
use crate::train::grad::SparseTarget;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// IDs shared by the teacher and the student (`book/PLAN.md`).
pub const COMMON_VOCAB: usize = 248_070;
const FORMAT: &str = "qwen-train-teacher-1";

/// One `STRATA_LOGPOS` line.
#[derive(Debug, Clone, PartialEq)]
pub struct LogposLine {
    pub position: usize,
    pub target: u32,
    pub target_logp: f64,
    /// The top K as `(id, log-probability)`, most likely first.
    pub top: Vec<(u32, f64)>,
}

fn bad(line: usize, what: &str) -> InferenceError {
    InferenceError(format!("STRATA_LOGPOS line {}: {what}", line + 1))
}

/// Parses `STRATA_LOGPOS` output (empty lines are skipped).
pub fn parse_logpos(text: &str) -> Result<Vec<LogposLine>> {
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 8 {
            return Err(bad(n, "fewer than 8 fields"));
        }
        let int = |s: &str| s.parse::<i64>().map_err(|_| bad(n, "not an integer"));
        let num = |s: &str| s.parse::<f64>().map_err(|_| bad(n, "not a number"));
        let position = usize::try_from(int(f[0])?).map_err(|_| bad(n, "negative position"))?;
        let target = u32::try_from(int(f[1])?).map_err(|_| bad(n, "invalid target"))?;
        let target_logp = num(f[2])?;
        let mut top = Vec::with_capacity(f.len() - 8);
        for x in &f[8..] {
            let (id, lp) = x
                .split_once(':')
                .ok_or_else(|| bad(n, "top-K entry is not id:logprob"))?;
            let id = u32::try_from(int(id)?).map_err(|_| bad(n, "invalid top-K ID"))?;
            top.push((id, num(lp)?));
        }
        if !(target_logp <= 0. && top.iter().all(|(_, l)| *l <= 0.)) {
            return Err(bad(n, "log-probability above 0 or NaN"));
        }
        out.push(LogposLine {
            position,
            target,
            target_logp,
            top,
        });
    }
    Ok(out)
}

/// The teacher's distribution at one position, over the shared IDs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TeacherPosition {
    pub position: usize,
    pub ids: Vec<u32>,
    pub log_p: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TeacherData {
    pub format: String,
    /// Free text naming the teacher, its quantization and the engine version.
    pub teacher: String,
    /// The IDs the teacher read (the same text must be cached for the student).
    pub ids: Vec<u32>,
    /// The distributions are over IDs `0..vocab`.
    pub vocab: usize,
    pub positions: Vec<TeacherPosition>,
}

impl TeacherData {
    /// Teacher data from `STRATA_LOGPOS` lines for the sequence `ids`,
    /// renormalized over the first `vocab` IDs. Each line must score the ID
    /// that follows its position in `ids`; a position may appear once.
    pub fn from_logpos(
        lines: &[LogposLine],
        ids: &[u32],
        vocab: usize,
        teacher: &str,
    ) -> Result<Self> {
        require(!lines.is_empty(), "no teacher positions")?;
        let mut seen = vec![false; ids.len()];
        let mut positions = Vec::with_capacity(lines.len());
        for l in lines {
            require(
                l.position + 1 < ids.len(),
                "teacher position past the end of the IDs",
            )?;
            require(
                ids[l.position + 1] == l.target,
                "teacher line scores another ID than the text: not the same token sequence",
            )?;
            require(!seen[l.position], "teacher position appears twice")?;
            seen[l.position] = true;
            let mut entries = l.top.clone();
            if !entries.iter().any(|(id, _)| *id == l.target) {
                entries.push((l.target, l.target_logp));
            }
            let dropped: f64 = entries
                .iter()
                .filter(|(id, _)| *id as usize >= vocab)
                .map(|(_, lp)| lp.exp())
                .sum();
            require(
                dropped < 1.,
                "teacher mass is all outside the shared vocabulary",
            )?;
            let shift = (1. - dropped).ln();
            let (ids_k, log_p): (Vec<u32>, Vec<f64>) = entries
                .into_iter()
                .filter(|(id, _)| (*id as usize) < vocab)
                .map(|(id, lp)| (id, (lp - shift).min(0.)))
                .unzip();
            // Validates (distinct IDs, mass ≤ 1) the way training will read it.
            SparseTarget::new(ids_k.clone(), log_p.clone(), vocab)?;
            positions.push(TeacherPosition {
                position: l.position,
                ids: ids_k,
                log_p,
            });
        }
        positions.sort_by_key(|p| p.position);
        Ok(Self {
            format: FORMAT.into(),
            teacher: teacher.into(),
            ids: ids.to_vec(),
            vocab,
            positions,
        })
    }
    /// One example per teacher position: KL to the teacher's distribution.
    pub fn examples(&self) -> Result<Vec<Example>> {
        self.positions
            .iter()
            .map(|p| {
                Ok(Example {
                    position: p.position,
                    target: Target::Sparse(SparseTarget::new(
                        p.ids.clone(),
                        p.log_p.clone(),
                        self.vocab,
                    )?),
                })
            })
            .collect()
    }
    /// Refuses a cache of another token sequence.
    pub fn check_ids(&self, ids: &[u32]) -> Result<()> {
        require(
            ids == self.ids,
            "teacher data and cache read different token sequences",
        )
    }
    /// JSON. The log-probabilities come back to within the last bit of an
    /// f64 (serde_json without `float_roundtrip`), so train from the file,
    /// not from the value before writing, to repeat a run exactly.
    pub fn write(&self, path: &Path) -> Result<()> {
        let s = serde_json::to_vec(self).map_err(|e| InferenceError(e.to_string()))?;
        std::fs::write(path, s).map_err(|e| InferenceError(format!("teacher data: {e}")))
    }
    pub fn read(path: &Path) -> Result<Self> {
        let s = std::fs::read(path).map_err(|e| InferenceError(format!("teacher data: {e}")))?;
        let d: Self = serde_json::from_slice(&s).map_err(|e| InferenceError(e.to_string()))?;
        require(d.format == FORMAT, "not a teacher data file")?;
        d.examples()?; // validates every position
        require(
            d.positions.iter().all(|p| p.position + 1 < d.ids.len()),
            "teacher position past the end of the IDs",
        )?;
        Ok(d)
    }
}
