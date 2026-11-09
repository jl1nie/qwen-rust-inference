//! 数値経路を変えずに時間とメモリを測る。論理f32 payloadとOSのRSSは
//! 意味が異なるので分けて記録する。peak RSSはプロセス全体の高水位。
use crate::{
    block::MixerWeights,
    model::Model,
    state::{LayerState, ModelState},
    tensor::{require, Backend, Result},
    weights::err,
};
use serde::Serialize;
use std::time::Instant;
#[derive(Debug, Clone, Serialize)]
pub struct Memory {
    pub rss_bytes: Option<u64>,
    pub peak_rss_bytes: Option<u64>,
}
pub fn memory() -> Memory {
    let text = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let get = |name: &str| {
        text.lines()
            .find(|l| l.starts_with(name))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<u64>().ok())
            .map(|v| v * 1024)
    };
    Memory {
        rss_bytes: get("VmRSS:"),
        peak_rss_bytes: get("VmHWM:"),
    }
}
pub fn state_bytes(state: &ModelState) -> usize {
    state
        .layers
        .iter()
        .map(|s| match s {
            LayerState::Delta(s) => (s.convolution.len() + s.recurrent.len()) * 4,
            LayerState::Attention(s) => s.keys.iter().chain(&s.values).map(|v| v.len() * 4).sum(),
        })
        .sum()
}
pub fn weight_bytes(m: &Model) -> usize {
    let mut n = m.embedding.values().len() + m.norm.len();
    for b in &m.blocks {
        n += b.input_norm.len() + b.post_norm.len();
        n += b
            .ffn
            .matrices()
            .iter()
            .map(|w| w.values().len())
            .sum::<usize>();
        n += match &b.mixer {
            MixerWeights::Attention(w) => {
                [&w.q, &w.k, &w.v, &w.o]
                    .iter()
                    .map(|w| w.values().len())
                    .sum::<usize>()
                    + w.q_norm.len()
                    + w.k_norm.len()
            }
            MixerWeights::Delta(w) => {
                [&w.qkv, &w.z, &w.a, &w.b, &w.conv, &w.out]
                    .iter()
                    .map(|w| w.values().len())
                    .sum::<usize>()
                    + w.dt.len()
                    + w.a_log.len()
                    + w.norm.len()
            }
        };
    }
    n * 4
}
#[derive(Debug, Serialize)]
pub struct Timing {
    pub prefill_ms: f64,
    pub prefill_tokens_per_second: f64,
    pub decode_ms: Vec<f64>,
    pub decode_tokens_per_second: f64,
    pub selection_ms: f64,
    pub state_bytes: usize,
    pub memory: Memory,
}
#[derive(Debug, Serialize)]
pub struct BenchReport {
    pub model_revision: String,
    pub model_identity: String,
    pub backend: Backend,
    pub full_capture: bool,
    pub prompt_ids: Vec<u32>,
    pub decode_steps: usize,
    pub warmup: usize,
    pub weights_f32_bytes: usize,
    pub runs: Vec<Timing>,
    pub cpu: Option<String>,
    pub cpu_quota: Option<String>,
    pub profile: String,
}
pub fn benchmark(
    model: &Model,
    ids: &[u32],
    decode: usize,
    warmup: usize,
    runs: usize,
    backend: Backend,
    full_capture: bool,
) -> Result<BenchReport> {
    require(
        !ids.is_empty()
            && runs > 0
            && decode > 0
            && ids
                .len()
                .checked_add(decode)
                .is_some_and(|n| n <= model.config.max_position_embeddings),
        "invalid benchmark budget",
    )?;
    let total = warmup
        .checked_add(runs)
        .ok_or_else(|| err("benchmark run count overflow"))?;
    let mut result = Vec::new();
    for run in 0..total {
        let mut state = model.new_state();
        let start = Instant::now();
        let mut logits = Vec::new();
        for &id in ids {
            logits = if full_capture {
                model.step(id, &mut state, backend, None)?.logits
            } else {
                model.step_compact(id, &mut state, backend, None)?
            };
        }
        let prefill_ms = start.elapsed().as_secs_f64() * 1000.;
        let mut decode_ms = Vec::new();
        let mut selection_ms = 0.;
        for _ in 0..decode {
            let start = Instant::now();
            let mut rng = 0;
            let token =
                crate::generate::select(&logits, crate::generate::Sampling::Greedy, &mut rng)?;
            selection_ms += start.elapsed().as_secs_f64() * 1000.;
            let start = Instant::now();
            logits = if full_capture {
                model.step(token, &mut state, backend, None)?.logits
            } else {
                model.step_compact(token, &mut state, backend, None)?
            };
            decode_ms.push(start.elapsed().as_secs_f64() * 1000.);
        }
        if run >= warmup {
            result.push(Timing {
                prefill_ms,
                prefill_tokens_per_second: ids.len() as f64 / (prefill_ms / 1000.),
                decode_tokens_per_second: decode as f64 / (decode_ms.iter().sum::<f64>() / 1000.),
                decode_ms,
                selection_ms,
                state_bytes: state_bytes(&state),
                memory: memory(),
            });
        }
    }
    let cpu = std::fs::read_to_string("/proc/cpuinfo").ok().and_then(|v| {
        v.lines()
            .find(|l| l.starts_with("model name"))
            .map(|s| s.split_once(':').unwrap().1.trim().to_owned())
    });
    Ok(BenchReport {
        model_revision: model.revision.clone(),
        model_identity: model.identity.clone(),
        backend,
        full_capture,
        prompt_ids: ids.to_vec(),
        decode_steps: decode,
        warmup,
        weights_f32_bytes: weight_bytes(model),
        runs: result,
        cpu,
        cpu_quota: std::fs::read_to_string("/sys/fs/cgroup/cpu.max")
            .ok()
            .map(|s| s.trim().to_owned()),
        profile: if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
        .into(),
    })
}
pub fn parse_backend(name: &str) -> Result<Backend> {
    match name {
        "scalar" | "reference" => Ok(Backend::Reference),
        "gemm" | "optimized" => Ok(Backend::Optimized),
        "vector" => Ok(Backend::Vector),
        _ => Err(err("backend must be scalar, gemm or vector")),
    }
}
