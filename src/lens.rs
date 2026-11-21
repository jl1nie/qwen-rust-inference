//! 本書第3章の測定：logit lens と、Attention layer ごとの attention weight。
//!
//! logit lens は、各 layer の出口の residual stream を、最後の正規化と
//! LM head にそのまま通して、「もしここでモデルが終わっていたら、次の
//! トークンにどんなスコアを付けるか」を見る。途中の layer は LM head に
//! 通すために学習されたわけではないので、読めるのは傾向まで。
use crate::{
    block::MixerActivations,
    model::{Model, StepActivations},
    ops::{rms_norm, softmax, Scale},
    tensor::{require, Backend, Result},
};

/// layer `layer` の出口で見た、狙いのトークンの順位（0が最上位）と確率。
#[derive(Debug, Clone, PartialEq)]
pub struct LensRow {
    pub layer: usize,
    pub rank: usize,
    pub probability: f32,
    pub top: u32,
    pub top_probability: f32,
}

/// 24個の layer の出口それぞれで、`target` の順位と確率を出す。
pub fn logit_lens(
    model: &Model,
    a: &StepActivations,
    target: u32,
    backend: Backend,
) -> Result<Vec<LensRow>> {
    Ok(logit_lens_many(model, a, &[target], backend)?.remove(0))
}

/// `logit_lens` for several targets at once: each layer's logits are
/// computed once and read for every target. `[target][layer]`.
pub fn logit_lens_many(
    model: &Model,
    a: &StepActivations,
    targets: &[u32],
    backend: Backend,
) -> Result<Vec<Vec<LensRow>>> {
    let mut rows = vec![Vec::with_capacity(a.layers.len()); targets.len()];
    for (layer, x) in a.layers.iter().enumerate() {
        // 最後の layer の後と同じ正規化と LM head（埋め込み表の共用）に通す。
        let normalized = rms_norm(
            x,
            model.norm(),
            model.config().rms_norm_eps,
            Scale::QwenZeroCentered,
        )?;
        let logits = model.embedding().project(&normalized, backend)?;
        let probabilities = softmax(&logits)?;
        let top = (0..logits.len())
            .max_by(|&i, &j| logits[i].total_cmp(&logits[j]))
            .unwrap_or(0);
        for (out, &target) in rows.iter_mut().zip(targets) {
            let t = target as usize;
            require(t < logits.len(), "lens target out of range")?;
            out.push(LensRow {
                layer,
                rank: logits.iter().filter(|&&v| v > logits[t]).count(),
                probability: probabilities[t],
                top: top as u32,
                top_probability: probabilities[top],
            });
        }
    }
    Ok(rows)
}

/// Attention layer ごとに、各 q ヘッドが position `key` に置いた attention weight。
/// 返り値は (layer, ヘッドごとの値) の並び。DeltaNet の layer は含まない。
pub fn attention_to(
    model: &Model,
    a: &StepActivations,
    key: usize,
) -> Result<Vec<(usize, Vec<f32>)>> {
    let heads = model.config().num_attention_heads;
    let mut out = Vec::new();
    for (layer, b) in a.blocks.iter().enumerate() {
        if let MixerActivations::Attention(v) = &b.mixer {
            // probabilities はヘッドごとに、読む相手の position の数だけ並ぶ。
            let n = v.probabilities.len() / heads;
            require(key < n, "key position is not to the left")?;
            out.push((
                layer,
                (0..heads).map(|h| v.probabilities[h * n + key]).collect(),
            ));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Model {
        Model::from_fixture(
            &serde_json::from_str(include_str!("../tests/fixtures/hybrid.json")).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn last_lens_row_is_the_model_output() {
        let m = fixture();
        let mut s = m.new_state();
        let mut a = None;
        for id in [1, 4, 2] {
            a = Some(m.step(id, &mut s, Backend::Reference, None).unwrap());
        }
        let a = a.unwrap();
        let p = softmax(&a.logits).unwrap();
        for target in 0..a.logits.len() as u32 {
            let rows = logit_lens(&m, &a, target, Backend::Reference).unwrap();
            let last = rows.last().unwrap();
            assert_eq!(rows.len(), a.layers.len());
            assert!((last.probability - p[target as usize]).abs() < 1e-6);
            let rank = a
                .logits
                .iter()
                .filter(|&&v| v > a.logits[target as usize])
                .count();
            assert_eq!(last.rank, rank);
        }
    }

    #[test]
    fn attention_weights_come_from_attention_layers_only() {
        let m = fixture();
        let mut s = m.new_state();
        let mut a = None;
        for id in [1, 4, 2] {
            a = Some(m.step(id, &mut s, Backend::Reference, None).unwrap());
        }
        let a = a.unwrap();
        let attention_layers = a
            .blocks
            .iter()
            .filter(|b| matches!(b.mixer, MixerActivations::Attention(_)))
            .count();
        // 3 positions: weights on keys 0..3 sum to 1 for every head.
        let per_key: Vec<_> = (0..3).map(|k| attention_to(&m, &a, k).unwrap()).collect();
        assert_eq!(per_key[0].len(), attention_layers);
        for (i, (layer, heads)) in per_key[0].iter().enumerate() {
            for h in 0..heads.len() {
                let sum: f32 = per_key.iter().map(|r| r[i].1[h]).sum();
                assert!((sum - 1.0).abs() < 1e-5, "layer {layer} head {h}: {sum}");
            }
        }
        assert!(attention_to(&m, &a, 3).is_err());
    }
}

/// One measurement of `lens-batch`: a chat prompt, text forced at the start
/// of the reply (`prefix`, may be empty), and named target texts. The lens
/// is taken at the last position of the prefix, which predicts the next
/// token; each target is the first token its text adds after the prefix.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LensItem {
    pub id: String,
    pub prompt: String,
    #[serde(default)]
    pub prefix: String,
    pub targets: std::collections::BTreeMap<String, String>,
}

/// One target's rank (0 = top) and probability at every layer's exit.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct LensTarget {
    pub name: String,
    pub text: String,
    pub id: u32,
    pub token: String,
    pub ranks: Vec<usize>,
    pub probabilities: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct LensResult {
    pub id: String,
    pub prompt: String,
    pub prefix: String,
    pub position: usize,
    /// The model's next token after the prefix (greedy), and its own lens
    /// as the target named "chosen".
    pub chosen: (u32, String),
    /// The most likely token at each layer's exit.
    pub tops: Vec<(u32, String)>,
    pub targets: Vec<LensTarget>,
    /// The greedy reply to the plain prompt (no prefix), if asked for.
    pub generated: Option<String>,
}

/// The first token `text` adds after `prefix` (`prefix` must not change).
pub fn first_token_after(
    codec: &dyn crate::text::TextCodec,
    prefix: &str,
    text: &str,
) -> Result<u32> {
    let a = codec.encode(prefix)?;
    let b = codec.encode(&format!("{prefix}{text}"))?;
    require(
        b.len() > a.len() && b.starts_with(&a),
        "target text merges with the prefix's last token",
    )?;
    Ok(b[a.len()])
}

/// Measures one item. `generate`: the length of the greedy reply to the
/// plain prompt to record (0 = none); `eos`: the IDs that end it.
pub fn lens_item(
    model: &Model,
    codec: &dyn crate::text::TextCodec,
    item: &LensItem,
    edit: Option<&crate::edit::EditManifest>,
    generate: usize,
    eos: &[u32],
    backend: Backend,
) -> Result<LensResult> {
    require(!item.targets.is_empty(), "lens item without targets")?;
    let chat = crate::text::user_chat(&item.prompt, false);
    let context = format!("{chat}{}", item.prefix);
    let ids = codec.encode(&context)?;
    let mut names = Vec::new();
    let mut ids_t = Vec::new();
    for (name, text) in &item.targets {
        names.push((name.clone(), text.clone()));
        ids_t.push(first_token_after(codec, &context, text)?);
    }
    let mut state = model.new_state();
    for &id in &ids[..ids.len() - 1] {
        model.step_compact(id, &mut state, backend, edit)?;
    }
    let a = model.step(*ids.last().unwrap(), &mut state, backend, edit)?;
    let chosen = crate::validation::top(&a.logits, 1)[0].0 as u32;
    names.push(("chosen".into(), codec.decode(&[chosen])?));
    ids_t.push(chosen);
    let rows = logit_lens_many(model, &a, &ids_t, backend)?;
    let tops = rows[0]
        .iter()
        .map(|r| Ok((r.top, codec.decode(&[r.top])?)))
        .collect::<Result<_>>()?;
    let targets = names
        .into_iter()
        .zip(&ids_t)
        .zip(&rows)
        .map(|(((name, text), &id), r)| {
            Ok(LensTarget {
                name,
                text,
                id,
                token: codec.decode(&[id])?,
                ranks: r.iter().map(|x| x.rank).collect(),
                probabilities: r.iter().map(|x| x.probability).collect(),
            })
        })
        .collect::<Result<_>>()?;
    let generated = if generate == 0 {
        None
    } else {
        let prompt_ids = codec.encode(&chat)?;
        let r = crate::generate::generate(
            model,
            &prompt_ids,
            generate,
            eos,
            crate::generate::Sampling::Greedy,
            edit,
        )?;
        let mut g = r.history.generated.as_slice();
        if g.last().is_some_and(|i| eos.contains(i)) {
            g = &g[..g.len() - 1];
        }
        Some(codec.decode(g)?)
    };
    Ok(LensResult {
        id: item.id.clone(),
        prompt: item.prompt.clone(),
        prefix: item.prefix.clone(),
        position: ids.len() - 1,
        chosen: (chosen, codec.decode(&[chosen])?),
        tops,
        targets,
        generated,
    })
}
