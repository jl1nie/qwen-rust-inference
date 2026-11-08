use crate::tensor::{finite, require, Matrix, Result};
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EditManifest {
    pub model_revision: String,
    pub edits: Vec<Edit>,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Edit {
    pub layer: usize,
    pub tensor_name: String,
    pub columns: Vec<usize>,
    pub direction: Option<Vec<f32>>,
    pub coefficient: f32,
}
impl EditManifest {
    pub fn validate(
        &self,
        revision: &str,
        layers: usize,
        hidden: usize,
        intermediate: usize,
    ) -> Result<()> {
        require(
            self.model_revision == revision,
            "edit model revision mismatch",
        )?;
        for e in &self.edits {
            require(
                e.layer < layers
                    && e.tensor_name
                        == format!(
                            "model.language_model.layers.{}.mlp.down_proj.weight",
                            e.layer
                        ),
                "edit target mismatch",
            )?;
            require(
                !e.columns.is_empty()
                    && e.columns.iter().all(|&j| j < intermediate)
                    && e.coefficient.is_finite(),
                "invalid edit columns/coefficient",
            )?;
            let mut unique = e.columns.clone();
            unique.sort_unstable();
            unique.dedup();
            require(unique.len() == e.columns.len(), "duplicate edit column")?;
            if let Some(d) = &e.direction {
                finite(d)?;
                require(d.len() == hidden, "direction shape mismatch")?;
                let norm = d.iter().map(|v| v * v).sum::<f32>();
                require(norm.is_finite() && norm > 0., "invalid direction norm")?;
            }
        }
        Ok(())
    }
    pub fn apply(&self, layer: usize, product: &mut [f32], down: &Matrix) -> Result<Vec<f32>> {
        // 公開APIなので、Model::stepを経由しない呼出しでも添字を検査する。
        require(product.len() == down.cols(), "edit product shape mismatch")?;
        for e in self.edits.iter().filter(|e| e.layer == layer) {
            require(
                e.columns.iter().all(|&j| j < product.len()) && e.coefficient.is_finite(),
                "invalid edit target",
            )?;
            if let Some(d) = &e.direction {
                finite(d)?;
                let s = d.iter().map(|v| v * v).sum::<f32>();
                require(
                    d.len() == down.rows() && s.is_finite() && s > 0.,
                    "invalid edit direction",
                )?;
            }
        }
        // ユニット無効化は活性値だけに作用する。元のdown行列は共有された
        // 不変データなので、別runや無編集の推論へ変更が漏れることはない。
        for edit in self
            .edits
            .iter()
            .filter(|e| e.layer == layer && e.direction.is_none())
        {
            for &j in &edit.columns {
                product[j] *= 1. - edit.coefficient;
            }
        }
        // down列 w_j の方向d成分を除去したとき、FFN出力の差分は
        // -alpha*d*(d^T w_j)/(d^T d)*product[j]。行列全体をコピーせず、
        // この差分だけを出力へ加える。複数の方向編集は元列に対して加算。
        let mut correction = vec![0.; down.rows()];
        for edit in self
            .edits
            .iter()
            .filter(|e| e.layer == layer && e.direction.is_some())
        {
            let d = edit.direction.as_ref().unwrap();
            let denominator = d.iter().map(|v| v * v).sum::<f32>();
            for &j in &edit.columns {
                let column = down.column(j)?;
                let projection =
                    d.iter().zip(column).map(|(a, b)| a * b).sum::<f32>() / denominator;
                for (delta, v) in correction.iter_mut().zip(d) {
                    *delta -= edit.coefficient * v * projection * product[j];
                }
            }
        }
        finite(&correction)?;
        Ok(correction)
    }
}
#[derive(Debug, Serialize)]
pub struct RepetitionMetrics {
    pub repeated_tokens: usize,
    pub repeated_bigrams: usize,
}
pub fn repetition(ids: &[u32]) -> RepetitionMetrics {
    use std::collections::HashSet;
    let unique: HashSet<_> = ids.iter().collect();
    let grams: HashSet<_> = ids.windows(2).collect();
    RepetitionMetrics {
        repeated_tokens: ids.len() - unique.len(),
        repeated_bigrams: ids.len().saturating_sub(1) - grams.len(),
    }
}
