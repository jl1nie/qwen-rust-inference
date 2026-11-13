//! MTP（Multi-Token Prediction）ヘッド：次の「次の」トークンを予測する追加の1層。
//!
//! 本体の位置iの最終正規化後の隠れ状態 h_i（語彙射影の直前、トークン s_{i+1} を予測した値）と、
//! s_{i+1} の埋め込みを組にして入力し、s_{i+2} を予測する。
//!
//! ```text
//! x = fc([RMSNorm_e(embed(s_{i+1})), RMSNorm_h(h_i)])   2048 → 1024
//! x = Attention層1つ（本体のAttention層と同じ形、自分のKVキャッシュを持つ）+ FFN
//! g = RMSNorm(x)          … 次の下書きに渡す隠れ状態
//! logits = embedding · g  … 本体と同じ語彙射影（tied embedding）
//! ```
//!
//! 2トークン目以降の下書きは、g と1トークン目の下書きを組にして同じヘッドをもう一度使う。
//! 構造はllama.cppの `src/models/qwen35.cpp` の `graph_mtp` と同じ（spec/other-engines.md）。
//! 本体の推論経路には使わず、`mtp-accept` で採用率を測るためだけに読み込む。
use crate::{
    block::BlockWeights,
    model::Model,
    ops::{rms_norm, Scale},
    state::{KvCache, LayerState},
    tensor::{require, Backend, InferenceError, Matrix, Result},
};
#[derive(Debug, Clone)]
pub struct MtpWeights {
    pub embedding_norm: Vec<f32>,
    pub hidden_norm: Vec<f32>,
    pub fc: Matrix,
    pub block: BlockWeights,
    pub norm: Vec<f32>,
    pub eps: f32,
}
pub struct MtpOutput {
    /// 最終正規化後の隠れ状態。次の下書きの h として渡す。
    pub hidden: Vec<f32>,
    pub logits: Vec<f32>,
}
/// MTP層のKVキャッシュ。位置は組 (h_i, s_{i+1}) の番号 i。
/// RoPEは相対位置だけに効くので、本体の位置と1つずれていても結果は同じ。
pub struct MtpState(LayerState);
impl Default for MtpState {
    fn default() -> Self {
        Self(LayerState::Attention(KvCache::default()))
    }
}
impl MtpState {
    pub fn len(&self) -> usize {
        match &self.0 {
            LayerState::Attention(c) => c.keys.len(),
            LayerState::Delta(_) => 0,
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// 下書きの連鎖で足した位置を捨てる。KVは位置ごとの行なので切るだけで戻せる
    /// （本体のDeltaNet状態はこうは戻せない、spec/roadmap.md の3.）。
    pub fn truncate(&mut self, len: usize) {
        if let LayerState::Attention(c) = &mut self.0 {
            c.keys.truncate(len);
            c.values.truncate(len);
        }
    }
}
impl MtpWeights {
    pub fn step(
        &self,
        model: &Model,
        hidden: &[f32],
        next_id: u32,
        state: &mut MtpState,
        backend: Backend,
    ) -> Result<MtpOutput> {
        let h = model.config().hidden_size;
        require(
            hidden.len() == h && (next_id as usize) < model.config().vocab_size,
            "MTP input shape",
        )?;
        let embedded = model.embedding().row(next_id as usize)?;
        // 順序は [埋め込み, 隠れ状態]。fcの2048列のうち先頭1024列が埋め込み側。
        let mut joined = rms_norm(
            &embedded,
            &self.embedding_norm,
            self.eps,
            Scale::QwenZeroCentered,
        )?;
        joined.extend(rms_norm(
            hidden,
            &self.hidden_norm,
            self.eps,
            Scale::QwenZeroCentered,
        )?);
        let x = self.fc.project(&joined, backend)?;
        let position = state.len();
        let out = self
            .block
            .step_captured(&x, &mut state.0, position, backend, 0, None, false)
            .map_err(|e| InferenceError(format!("MTP block: {}", e.0)))?
            .output;
        let hidden = rms_norm(&out, &self.norm, self.eps, Scale::QwenZeroCentered)?;
        let logits = model.embedding().project(&hidden, backend)?;
        Ok(MtpOutput { hidden, logits })
    }
}
