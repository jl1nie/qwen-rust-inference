//! Autoregression owns token history, while the model owns immutable weights and
//! each run owns its caches. A sampled token becomes the next step's input.
use crate::{
    edit::EditManifest,
    model::Model,
    ops::softmax,
    state::TokenHistory,
    tensor::{finite, require, Backend, Result},
};
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq)]
pub enum Sampling {
    Greedy,
    Seeded {
        temperature: f32,
        top_k: usize,
        seed: u64,
    },
}
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum StopReason {
    Eos,
    MaxTokens,
    ContextLimit,
}
#[derive(Debug, Clone)]
pub struct Generated {
    pub history: TokenHistory,
    pub stop: StopReason,
}
pub fn select(logits: &[f32], sampling: Sampling, rng: &mut u64) -> Result<u32> {
    finite(logits)?;
    if sampling == Sampling::Greedy {
        // argmaxは全語彙のsortを必要としない。同点は先のIDを残す。
        let mut best = 0;
        for i in 1..logits.len() {
            if logits[i] > logits[best] {
                best = i;
            }
        }
        return Ok(best as u32);
    }
    let mut ranked: Vec<_> = logits.iter().copied().enumerate().collect();
    // Ties use the lower token ID, matching torch.argmax's first occurrence.
    ranked.sort_by(|(i, a), (j, b)| b.total_cmp(a).then(i.cmp(j)));
    match sampling {
        Sampling::Greedy => Ok(ranked[0].0 as u32),
        Sampling::Seeded {
            temperature, top_k, ..
        } => {
            require(
                temperature.is_finite() && temperature > 0. && top_k > 0,
                "invalid sampling parameters",
            )?;
            ranked.truncate(top_k.min(ranked.len()));
            let scores: Vec<_> = ranked.iter().map(|(_, v)| v / temperature).collect();
            let probabilities = softmax(&scores)?;
            *rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let random = (*rng >> 40) as f32 / 16777216.;
            let mut cumulative = 0.;
            for ((id, _), p) in ranked.iter().zip(probabilities) {
                cumulative += p;
                if random < cumulative {
                    return Ok(*id as u32);
                }
            }
            Ok(ranked.last().unwrap().0 as u32)
        }
    }
}
pub fn generate(
    model: &Model,
    prompt: &[u32],
    max_new: usize,
    eos: &[u32],
    sampling: Sampling,
    edit: Option<&EditManifest>,
) -> Result<Generated> {
    if edit.is_none() && max_new > 0 {
        require(!prompt.is_empty(), "empty prompt")?;
        let mut session =
            crate::session::InferenceSession::new(model, Backend::inference(), sampling)?;
        session.prefill(model, prompt)?;
        return session.generate(model, max_new, eos);
    }
    generate_observed(model, prompt, max_new, eos, sampling, edit, |_, _| Ok(()))
}
pub fn generate_observed(
    model: &Model,
    prompt: &[u32],
    max_new: usize,
    eos: &[u32],
    sampling: Sampling,
    edit: Option<&EditManifest>,
    mut observe: impl FnMut(usize, &crate::model::StepActivations) -> Result<()>,
) -> Result<Generated> {
    require(!prompt.is_empty(), "empty prompt")?;
    require(
        prompt
            .iter()
            .all(|&id| (id as usize) < model.config.vocab_size)
            && prompt.len() <= model.config.max_position_embeddings,
        "invalid prompt IDs/context",
    )?;
    let history = TokenHistory {
        prompt: prompt.to_vec(),
        generated: vec![],
    };
    if max_new == 0 {
        return Ok(Generated {
            history,
            stop: StopReason::MaxTokens,
        });
    }
    let mut state = model.new_state();
    // Keep only the current logits: a long prompt must not retain one vocabulary
    // sized vector for every position. The inspection prefill API is separate.
    let last =
        model.prefill_observed(prompt, &mut state, Backend::inference(), edit, &mut observe)?;
    decode_with(prompt, last.unwrap(), max_new, eos, sampling, |token| {
        if state.position >= model.config.max_position_embeddings {
            return Ok(None);
        }
        let p = state.position;
        let step = model.step(token, &mut state, Backend::inference(), edit)?;
        observe(p, &step)?;
        Ok(Some(step.logits))
    })
}
/// A controlled-logits seam makes stopping behavior testable independently of
/// random weights. `None` means the model's context budget was exhausted.
pub fn decode_with(
    prompt: &[u32],
    mut logits: Vec<f32>,
    max_new: usize,
    eos: &[u32],
    sampling: Sampling,
    mut next: impl FnMut(u32) -> Result<Option<Vec<f32>>>,
) -> Result<Generated> {
    let mut history = TokenHistory {
        prompt: prompt.to_vec(),
        generated: vec![],
    };
    let mut rng = match sampling {
        Sampling::Seeded { seed, .. } => seed,
        _ => 0,
    };
    let mut stop = StopReason::MaxTokens;
    for i in 0..max_new {
        let token = select(&logits, sampling, &mut rng)?;
        history.generated.push(token);
        // 終了はトークンIDの明示リストだけで決める。思考区切りの文字列を
        // 検索して止めると、</think>後の回答本文を生成できなくなる。
        if eos.contains(&token) {
            stop = StopReason::Eos;
            break;
        }
        if i + 1 == max_new {
            break;
        }
        match next(token)? {
            Some(value) => logits = value,
            None => {
                stop = StopReason::ContextLimit;
                break;
            }
        }
    }
    Ok(Generated { history, stop })
}
