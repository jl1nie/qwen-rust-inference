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
    let t = target as usize;
    let mut rows = Vec::with_capacity(a.layers.len());
    for (layer, x) in a.layers.iter().enumerate() {
        // 最後の layer の後と同じ正規化と LM head（埋め込み表の共用）に通す。
        let normalized = rms_norm(
            x,
            model.norm(),
            model.config().rms_norm_eps,
            Scale::QwenZeroCentered,
        )?;
        let logits = model.embedding().project(&normalized, backend)?;
        require(t < logits.len(), "lens target out of range")?;
        let probabilities = softmax(&logits)?;
        let rank = logits.iter().filter(|&&v| v > logits[t]).count();
        let top = (0..logits.len())
            .max_by(|&i, &j| logits[i].total_cmp(&logits[j]))
            .unwrap_or(0);
        rows.push(LensRow {
            layer,
            rank,
            probability: probabilities[t],
            top: top as u32,
            top_probability: probabilities[top],
        });
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
