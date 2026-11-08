//! Pre-norm residual block. KV/recurrent state is deliberately passed separately
//! from hidden activations: layer N's residual stream is not layer N's cache.
use crate::{
    attention::{AttentionActivations, AttentionWeights},
    deltanet::{DeltaActivations, DeltaWeights},
    edit::EditManifest,
    ffn::{FfnActivations, FfnWeights},
    ops::{add, rms_norm, Scale},
    state::LayerState,
    tensor::{Backend, InferenceError, Result},
};
#[derive(Debug, Clone)]
pub enum MixerWeights {
    Attention(AttentionWeights),
    Delta(DeltaWeights),
}
#[derive(Debug, Clone)]
pub struct BlockWeights {
    pub input_norm: Vec<f32>,
    pub post_norm: Vec<f32>,
    pub mixer: MixerWeights,
    pub ffn: FfnWeights,
    pub eps: f32,
}
#[derive(Debug, Clone)]
pub struct BlockActivations {
    pub input_norm: Vec<f32>,
    pub mixed: Vec<f32>,
    pub mixer: MixerActivations,
    pub after_mixer: Vec<f32>,
    pub post_norm: Vec<f32>,
    pub ffn_product: Vec<f32>,
    pub ffn: Vec<f32>,
    pub ffn_activations: FfnActivations,
    pub output: Vec<f32>,
}
#[derive(Debug, Clone)]
pub enum MixerActivations {
    Attention(AttentionActivations),
    Delta(DeltaActivations),
}
impl BlockWeights {
    pub fn step(
        &self,
        x: &[f32],
        state: &mut LayerState,
        position: usize,
        backend: Backend,
        layer: usize,
        edit: Option<&EditManifest>,
    ) -> Result<BlockActivations> {
        let input_norm = rms_norm(x, &self.input_norm, self.eps, Scale::QwenZeroCentered)?;
        let mixer = match (&self.mixer, state) {
            (MixerWeights::Attention(w), LayerState::Attention(s)) => {
                MixerActivations::Attention(w.step(&input_norm, position, s, backend)?)
            }
            (MixerWeights::Delta(w), LayerState::Delta(s)) => {
                MixerActivations::Delta(w.step(&input_norm, s, backend)?)
            }
            _ => return Err(InferenceError("layer/state type mismatch".into())),
        };
        let mixed = match &mixer {
            MixerActivations::Attention(a) => a.output.clone(),
            MixerActivations::Delta(a) => a.output.clone(),
        };
        // First residual carries x around the token mixer. FFN sees a separately
        // normalized version; normalization never overwrites the residual branch.
        let after_mixer = add(x, &mixed)?;
        let post_norm = rms_norm(
            &after_mixer,
            &self.post_norm,
            self.eps,
            Scale::QwenZeroCentered,
        )?;
        let ffn = self.ffn.forward_edited(&post_norm, backend, layer, edit)?;
        let output = add(&after_mixer, &ffn.output)?;
        Ok(BlockActivations {
            input_norm,
            mixed,
            mixer,
            after_mixer,
            post_norm,
            ffn_product: ffn.product.clone(),
            ffn: ffn.output.clone(),
            ffn_activations: ffn,
            output,
        })
    }
}
