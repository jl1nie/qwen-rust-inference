//! Checkpoint interpretation lives here, separate from arithmetic and inference state.
//! The immutable TensorData map is consumed exactly once: missing, wrong-shaped or
//! unexpectedly unused language weights are errors, never silently random values.
use crate::{
    attention::{AttentionConfig, AttentionWeights},
    block::{BlockWeights, MixerWeights},
    deltanet::{DeltaConfig, DeltaWeights},
    ffn::FfnWeights,
    model::Model,
    tensor::{finite, require, InferenceError, Matrix, Result},
};
use safetensors::{Dtype, SafeTensors};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, path::Path};
pub const MODEL_REVISION: &str = "2fc06364715b967f1860aea9cf38778875588b17";
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeConfig {
    pub rope_type: String,
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
    pub mrope_section: Vec<usize>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub layer_types: Vec<String>,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: usize,
    pub rope_parameters: RopeConfig,
    pub tie_word_embeddings: bool,
}
impl TextConfig {
    pub fn validate(&self) -> Result<()> {
        require(
            self.hidden_size > 0
                && self.intermediate_size > 0
                && self.vocab_size > 0
                && self.num_hidden_layers > 0
                && self.max_position_embeddings > 0,
            "zero config dimension",
        )?;
        require(
            self.layer_types.len() == self.num_hidden_layers
                && self
                    .layer_types
                    .iter()
                    .all(|s| s == "linear_attention" || s == "full_attention"),
            "unsupported layer types",
        )?;
        require(
            self.num_key_value_heads > 0
                && self.num_attention_heads > 0
                && self.num_attention_heads % self.num_key_value_heads == 0
                && self.head_dim > 0,
            "invalid attention heads",
        )?;
        require(
            self.linear_num_key_heads > 0
                && self.linear_num_value_heads > 0
                && self.linear_num_value_heads % self.linear_num_key_heads == 0
                && self.linear_key_head_dim > 0
                && self.linear_value_head_dim > 0
                && self.linear_conv_kernel_dim > 0,
            "invalid DeltaNet dimensions",
        )?;
        let rotary = (self.head_dim as f32 * self.rope_parameters.partial_rotary_factor) as usize;
        require(
            self.rope_parameters.rope_type == "default"
                && self.rope_parameters.rope_theta.is_finite()
                && self.rope_parameters.rope_theta > 0.
                && self.rope_parameters.partial_rotary_factor.is_finite()
                && (0.0..=1.0).contains(&self.rope_parameters.partial_rotary_factor)
                && rotary > 0
                && rotary % 2 == 0,
            "unsupported RoPE",
        )?;
        require(
            self.rms_norm_eps.is_finite() && self.rms_norm_eps > 0. && self.tie_word_embeddings,
            "unsupported epsilon/untied output",
        )
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorData {
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}
pub struct TensorReader {
    pub tensors: BTreeMap<String, TensorData>,
}
impl TensorReader {
    pub fn take(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let t = self
            .tensors
            .remove(name)
            .ok_or_else(|| InferenceError(format!("missing tensor: {name}")))?;
        require(
            t.shape == shape
                && shape.iter().try_fold(1usize, |a, &b| a.checked_mul(b)) == Some(t.values.len()),
            &format!("shape mismatch: {name}: {:?} != {shape:?}", t.shape),
        )?;
        finite(&t.values)?;
        Ok(t.values)
    }
    fn matrix(&mut self, name: &str, rows: usize, cols: usize) -> Result<Matrix> {
        Matrix::new(rows, cols, self.take(name, &[rows, cols])?)
    }
    pub fn finish(self, ignored: &[String]) -> Result<Vec<String>> {
        let mut unused = Vec::new();
        for name in self.tensors.keys() {
            require(
                ignored.contains(name),
                &format!("unexpected unused tensor: {name}"),
            )?;
            unused.push(name.clone());
        }
        Ok(unused)
    }
}
/// Exact BF16 conversion: preserve sign/exponent/mantissa bits, insert 16 zero
/// fraction bits. This is not numeric conversion of the u16 integer value.
pub fn decode_tensor(dtype: Dtype, data: &[u8]) -> Result<Vec<f32>> {
    let values: Vec<f32> = match dtype {
        Dtype::BF16 => {
            require(data.len() % 2 == 0, "unaligned BF16")?;
            data.chunks_exact(2)
                .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
                .collect()
        }
        Dtype::F32 => {
            require(data.len() % 4 == 0, "unaligned F32")?;
            data.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        }
        _ => return Err(InferenceError(format!("unsupported dtype: {dtype:?}"))),
    };
    finite(&values)?;
    Ok(values)
}
pub fn model_from_tensors(
    config: TextConfig,
    mut reader: TensorReader,
    revision: String,
    ignored: &[String],
) -> Result<Model> {
    config.validate()?;
    let h = config.hidden_size;
    let mid = config.intermediate_size;
    let embedding = reader.matrix(
        "model.language_model.embed_tokens.weight",
        config.vocab_size,
        h,
    )?;
    let norm = reader.take("model.language_model.norm.weight", &[h])?;
    let mut blocks = Vec::new();
    for (i, kind) in config.layer_types.iter().enumerate() {
        let prefix = format!("model.language_model.layers.{i}.");
        let n1 = reader.take(&format!("{prefix}input_layernorm.weight"), &[h])?;
        let n2 = reader.take(&format!("{prefix}post_attention_layernorm.weight"), &[h])?;
        let ffn = FfnWeights::new(
            reader.matrix(&format!("{prefix}mlp.gate_proj.weight"), mid, h)?,
            reader.matrix(&format!("{prefix}mlp.up_proj.weight"), mid, h)?,
            reader.matrix(&format!("{prefix}mlp.down_proj.weight"), h, mid)?,
        )?;
        let mixer = if kind == "full_attention" {
            let p = format!("{prefix}self_attn.");
            let d = config.head_dim;
            let q = config.num_attention_heads * d;
            let kv = config.num_key_value_heads * d;
            MixerWeights::Attention(AttentionWeights {
                q: reader.matrix(&format!("{p}q_proj.weight"), 2 * q, h)?,
                k: reader.matrix(&format!("{p}k_proj.weight"), kv, h)?,
                v: reader.matrix(&format!("{p}v_proj.weight"), kv, h)?,
                o: reader.matrix(&format!("{p}o_proj.weight"), h, q)?,
                q_norm: reader.take(&format!("{p}q_norm.weight"), &[d])?,
                k_norm: reader.take(&format!("{p}k_norm.weight"), &[d])?,
                config: AttentionConfig {
                    heads: config.num_attention_heads,
                    kv_heads: config.num_key_value_heads,
                    head_dim: d,
                    rotary_dim: (d as f32 * config.rope_parameters.partial_rotary_factor) as usize,
                    theta: config.rope_parameters.rope_theta,
                    eps: config.rms_norm_eps,
                },
            })
        } else {
            let p = format!("{prefix}linear_attn.");
            let k = config.linear_num_key_heads * config.linear_key_head_dim;
            let v = config.linear_num_value_heads * config.linear_value_head_dim;
            let heads = config.linear_num_value_heads;
            let channels = 2 * k + v;
            let cv = reader.take(
                &format!("{p}conv1d.weight"),
                &[channels, 1, config.linear_conv_kernel_dim],
            )?;
            MixerWeights::Delta(DeltaWeights {
                qkv: reader.matrix(&format!("{p}in_proj_qkv.weight"), channels, h)?,
                z: reader.matrix(&format!("{p}in_proj_z.weight"), v, h)?,
                a: reader.matrix(&format!("{p}in_proj_a.weight"), heads, h)?,
                b: reader.matrix(&format!("{p}in_proj_b.weight"), heads, h)?,
                conv: Matrix::new(channels, config.linear_conv_kernel_dim, cv)?,
                dt: reader.take(&format!("{p}dt_bias"), &[heads])?,
                a_log: reader.take(&format!("{p}A_log"), &[heads])?,
                norm: reader.take(&format!("{p}norm.weight"), &[config.linear_value_head_dim])?,
                out: reader.matrix(&format!("{p}out_proj.weight"), h, v)?,
                config: DeltaConfig {
                    key_heads: config.linear_num_key_heads,
                    value_heads: heads,
                    key_dim: config.linear_key_head_dim,
                    value_dim: config.linear_value_head_dim,
                    kernel: config.linear_conv_kernel_dim,
                    eps: config.rms_norm_eps,
                },
            })
        };
        blocks.push(BlockWeights {
            input_norm: n1,
            post_norm: n2,
            mixer,
            ffn,
            eps: config.rms_norm_eps,
        });
    }
    let unused = reader.finish(ignored)?;
    Ok(Model {
        config,
        embedding,
        norm,
        blocks,
        revision,
        unused,
    })
}
pub fn load_config(dir: &Path) -> Result<TextConfig> {
    let v: Value =
        serde_json::from_slice(&fs::read(dir.join("config.json")).map_err(err)?).map_err(err)?;
    require(v["model_type"] == "qwen3_5", "unsupported model type")?;
    require(
        v["text_config"]["attention_bias"] == false
            && v["text_config"]["hidden_act"] == "silu"
            && v["text_config"]["attn_output_gate"] == true,
        "unsupported attention/activation flags",
    )?;
    let c: TextConfig = serde_json::from_value(v["text_config"].clone()).map_err(err)?;
    c.validate()?;
    Ok(c)
}
pub fn load_model(dir: &Path) -> Result<Model> {
    let revision = fs::read_to_string(dir.join("REVISION"))
        .map_err(err)?
        .trim()
        .to_owned();
    require(revision == MODEL_REVISION, "unverified model revision")?;
    let config_bytes = fs::read(dir.join("config.json")).map_err(err)?;
    require(
        format!("{:x}", Sha256::digest(config_bytes))
            == "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
        "config SHA-256 mismatch",
    )?;
    let c = load_config(dir)?;
    let bytes = fs::read(dir.join("model.safetensors-00001-of-00001.safetensors")).map_err(err)?;
    let expected = "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696";
    require(
        format!("{:x}", Sha256::digest(&bytes)) == expected,
        "checkpoint SHA-256 mismatch",
    )?;
    let tensors = SafeTensors::deserialize(&bytes).map_err(err)?;
    let ignored: Vec<String> =
        serde_json::from_str(include_str!("../spec/ignored-tensors.json")).map_err(err)?;
    let mut values = BTreeMap::new();
    // Skip known vision/MTP tensors before converting: they cannot affect text
    // inference, and allocating them would obscure the memory cost of the path.
    let mut skipped = Vec::new();
    for (name, t) in tensors.tensors() {
        if ignored.contains(&name) {
            skipped.push(name);
            continue;
        }
        values.insert(
            name,
            TensorData {
                shape: t.shape().to_vec(),
                values: decode_tensor(t.dtype(), t.data())?,
            },
        );
    }
    let mut model = model_from_tensors(c, TensorReader { tensors: values }, revision, &[])?;
    skipped.sort(); // Stable inspection reports despite safetensors' map iteration order.
    model.unused = skipped;
    Ok(model)
}
pub fn err(e: impl std::fmt::Display) -> InferenceError {
    InferenceError(e.to_string())
}
