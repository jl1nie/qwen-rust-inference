//! 学習のため、保持する「過去」を型で区別する。
//! KvCacheは過去位置ごとに増え、DeltaStateは固定サイズ。
//! TokenHistoryは表示/生成制御用のID列であり、ネットワーク内部の記憶ではない。
#[derive(Debug, Clone, PartialEq, Default)]
pub struct KvCache {
    pub keys: Vec<Vec<f32>>,
    pub values: Vec<Vec<f32>>,
}
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DeltaState {
    /// [channels,kernel-1] を古い時刻→新しい時刻の順にflatten。
    pub convolution: Vec<f32>,
    pub recurrent: Vec<f32>,
}
#[derive(Debug, Clone, PartialEq)]
pub enum LayerState {
    Attention(KvCache),
    Delta(DeltaState),
}
#[derive(Debug, Clone, PartialEq)]
pub struct ModelState {
    pub position: usize,
    pub layers: Vec<LayerState>,
}
impl ModelState {
    pub fn reset(&mut self) {
        self.position = 0;
        for layer in &mut self.layers {
            match layer {
                LayerState::Attention(s) => *s = KvCache::default(),
                LayerState::Delta(s) => {
                    s.convolution.fill(0.);
                    s.recurrent.fill(0.);
                }
            }
        }
    }
}
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TokenHistory {
    pub prompt: Vec<u32>,
    pub generated: Vec<u32>,
}
