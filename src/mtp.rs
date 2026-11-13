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
        let hidden = self.step_hidden(model, hidden, next_id, state, backend)?;
        let logits = model.embedding().project(&hidden, backend)?;
        Ok(MtpOutput { hidden, logits })
    }
    /// `step` without the vocabulary projection: the normalized output g only
    /// (a draft over a subset of the vocabulary projects it with `DraftVocab`).
    pub fn step_hidden(
        &self,
        model: &Model,
        hidden: &[f32],
        next_id: u32,
        state: &mut MtpState,
        backend: Backend,
    ) -> Result<Vec<f32>> {
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
        rms_norm(&out, &self.norm, self.eps, Scale::QwenZeroCentered)
    }
}
/// 下書きだけを語彙の一部で計算するための行列（ADR 0025）。
/// 小さなモデルでは語彙射影（248,320行、509MB）が下書きの読み出しの大部分なので、
/// よく出るトークンの行だけを集めて射影する。検証は本体が全語彙で行うので、
/// 出力は変わらない。外れやすくなるだけ。
pub struct DraftVocab {
    /// 昇順のトークンID。同点は小さいIDを選び、全語彙のgreedyと同じ規則になる。
    ids: Vec<u32>,
    rows: Matrix,
}
impl DraftVocab {
    pub fn new(model: &Model, ids: &[u32]) -> Result<Self> {
        let mut ids = ids.to_vec();
        ids.sort_unstable();
        ids.dedup();
        let e = model.embedding();
        require(
            !ids.is_empty() && ids.iter().all(|&i| (i as usize) < e.rows()),
            "draft vocabulary IDs",
        )?;
        let h = e.cols();
        let rows = match e.bf16_bits() {
            Some(bits) => {
                let mut out = Vec::with_capacity(ids.len() * h);
                for &i in &ids {
                    out.extend_from_slice(&bits[i as usize * h..(i as usize + 1) * h]);
                }
                Matrix::from_bf16(ids.len(), h, out)?
            }
            None => {
                let mut out = Vec::with_capacity(ids.len() * h);
                for &i in &ids {
                    out.extend_from_slice(&e.row(i as usize)?);
                }
                Matrix::new(ids.len(), h, out)?
            }
        };
        Ok(Self { ids, rows })
    }
    pub fn len(&self) -> usize {
        self.ids.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
    /// The sorted token IDs; row r of the matrix is the embedding of ids[r].
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }
    #[cfg_attr(
        not(any(
            all(feature = "rocm", target_os = "linux"),
            all(feature = "metal", target_os = "macos")
        )),
        allow(dead_code)
    )]
    pub(crate) fn rows(&self) -> &Matrix {
        &self.rows
    }
    pub fn contains(&self, id: u32) -> bool {
        self.ids.binary_search(&id).is_ok()
    }
    /// The draft token: argmax over this subset of the vocabulary.
    pub fn argmax(&self, hidden: &[f32], backend: Backend) -> Result<u32> {
        let logits = self.rows.project(hidden, backend)?;
        let mut best = 0;
        for i in 1..logits.len() {
            if logits[i] > logits[best] {
                best = i;
            }
        }
        Ok(self.ids[best])
    }
}
/// Token IDs ordered by their count in `text` (most frequent first), then by
/// ID: the order of `--draft-vocab-text` (ADR 0025). Without text this is
/// the ID order.
pub fn frequency_order(vocab: usize, text: &[u32]) -> Vec<u32> {
    let mut count = vec![0usize; vocab];
    for &id in text {
        if let Some(c) = count.get_mut(id as usize) {
            *c += 1;
        }
    }
    let mut ids: Vec<u32> = (0..vocab as u32).collect();
    ids.sort_by_key(|&i| (std::cmp::Reverse(count[i as usize]), i));
    ids
}
