//! Checkpoint interpretation lives here, separate from arithmetic and inference state.
//! The immutable TensorData map is consumed exactly once: missing, wrong-shaped or
//! unexpectedly unused language weights are errors, never silently random values.
use crate::{
    attention::{AttentionConfig, AttentionWeights},
    block::{BlockWeights, MixerWeights},
    deltanet::{DeltaConfig, DeltaWeights},
    ffn::FfnWeights,
    model::Model,
    tensor::{finite, require, AlignedBuf, Element, InferenceError, Matrix, Result},
};
use safetensors::Dtype;
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
/// A checkpoint tensor kept as its BF16 bit patterns (not widened to f32).
/// Page-aligned, so a Metal device can use it in place (`AlignedBuf`).
#[derive(Debug)]
pub struct Bf16Data {
    pub shape: Vec<usize>,
    pub bits: AlignedBuf,
}
pub struct TensorReader {
    pub tensors: BTreeMap<String, TensorData>,
    /// BF16 tensors from a checkpoint. Matrices stay BF16 (half the memory
    /// of f32); vectors are widened when taken. Names never repeat `tensors`.
    pub bf16: BTreeMap<String, Bf16Data>,
}
impl TensorReader {
    pub fn new(tensors: BTreeMap<String, TensorData>) -> Self {
        Self {
            tensors,
            bf16: BTreeMap::new(),
        }
    }
    fn shape_ok(name: &str, have: &[usize], want: &[usize], len: usize) -> Result<()> {
        require(
            have == want && want.iter().try_fold(1usize, |a, &b| a.checked_mul(b)) == Some(len),
            &format!("shape mismatch: {name}: {have:?} != {want:?}"),
        )
    }
    pub fn take(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        if let Some(t) = self.bf16.remove(name) {
            Self::shape_ok(name, &t.shape, shape, t.bits.len())?;
            let values: Vec<f32> = t.bits.iter().map(|b| b.f32()).collect();
            finite(&values)?;
            return Ok(values);
        }
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
        if let Some(t) = self.bf16.remove(name) {
            Self::shape_ok(name, &t.shape, &[rows, cols], t.bits.len())?;
            return Matrix::from_bf16_buf(rows, cols, t.bits);
        }
        Matrix::new(rows, cols, self.take(name, &[rows, cols])?)
    }
    pub fn finish(self, ignored: &[String]) -> Result<Vec<String>> {
        let mut unused = Vec::new();
        for name in self.tensors.keys().chain(self.bf16.keys()) {
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
    require(
        !ignored
            .iter()
            .any(|n| n.starts_with("model.language_model.")),
        "cannot ignore language weights",
    )?;
    // 形状だけで同じモデルと判断しない。保存状態はconfigと実際の
    // 言語重みのbit列に結び付ける。読み込み時に一度だけ計算する。
    let mut identity = Sha256::new();
    identity.update(b"qwen-rust-model-v1");
    identity.update(serde_json::to_vec(&config).map_err(err)?);
    require(
        reader.bf16.keys().all(|n| !reader.tensors.contains_key(n)),
        "tensor stored twice",
    )?;
    // Name order over both maps; BF16 tensors are hashed as their exact f32
    // widening, so the identity (and saved snapshots) do not depend on how
    // the weights are held in memory.
    let names: std::collections::BTreeSet<&String> =
        reader.tensors.keys().chain(reader.bf16.keys()).collect();
    for name in names {
        if ignored.contains(name) {
            continue;
        }
        let (shape, f32_values, bf16_bits) = match reader.tensors.get(name) {
            Some(t) => (&t.shape, Some(&t.values), None),
            None => {
                let t = &reader.bf16[name];
                (&t.shape, None, Some(&t.bits))
            }
        };
        identity.update((name.len() as u64).to_le_bytes());
        identity.update(name.as_bytes());
        for &dim in shape {
            identity.update((dim as u64).to_le_bytes());
        }
        if let Some(values) = f32_values {
            for chunk in values.chunks(4096) {
                let bytes: Vec<u8> = chunk
                    .iter()
                    .flat_map(|v| v.to_bits().to_le_bytes())
                    .collect();
                identity.update(bytes);
            }
        }
        if let Some(bits) = bf16_bits {
            for chunk in bits.chunks(4096) {
                let bytes: Vec<u8> = chunk
                    .iter()
                    .flat_map(|&b| ((b as u32) << 16).to_le_bytes())
                    .collect();
                identity.update(bytes);
            }
        }
    }
    let identity = format!("{:x}", identity.finalize());
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
        identity,
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
type Checkpoint = (
    BTreeMap<String, TensorData>,
    BTreeMap<String, Bf16Data>,
    Vec<String>,
);
/// One sequential pass over a safetensors file: every byte goes through
/// SHA-256 while each tensor is read straight into its final buffer (BF16 as
/// bits, other dtypes decoded). The file is never held in memory as a whole,
/// so peak memory is about the kept weights. Nothing is returned unless the
/// whole file matches `expected_sha256`. Header checks follow the
/// safetensors crate: contiguous offsets, sizes = shape x dtype, and the file
/// ends exactly where the data ends. `ignored` tensors are hashed, not kept.
fn read_checkpoint(path: &Path, expected_sha256: &str, ignored: &[String]) -> Result<Checkpoint> {
    use std::io::Read;
    let file = fs::File::open(path).map_err(err)?;
    let length = file.metadata().map_err(err)?.len();
    let mut input = std::io::BufReader::with_capacity(1 << 20, file);
    let mut hash = Sha256::new();
    let mut read = |buf: &mut [u8], hash: &mut Sha256| -> Result<()> {
        input.read_exact(buf).map_err(err)?;
        hash.update(&*buf);
        Ok(())
    };
    let mut n = [0; 8];
    read(&mut n, &mut hash)?;
    let n = u64::from_le_bytes(n);
    require(
        n <= 100_000_000 && n + 8 <= length,
        "safetensors header length",
    )?;
    let mut header = vec![0; n as usize];
    read(&mut header, &mut hash)?;
    let metadata: safetensors::tensor::Metadata = serde_json::from_slice(&header).map_err(err)?;
    let mut infos: Vec<_> = metadata.tensors().into_iter().collect();
    infos.sort_by_key(|(_, i)| i.data_offsets.0);
    let mut end = 0usize;
    for (name, i) in &infos {
        let (s, e) = i.data_offsets;
        let bytes = i
            .shape
            .iter()
            .try_fold(i.dtype.size(), |a, &b| a.checked_mul(b));
        require(
            s == end && e >= s && bytes == Some(e - s),
            &format!("safetensors offsets: {name}"),
        )?;
        end = e;
    }
    require(
        (end as u64).checked_add(8 + n) == Some(length),
        "safetensors data length",
    )?;
    let (mut values, mut bf16, mut skipped) = (BTreeMap::new(), BTreeMap::new(), Vec::new());
    let mut chunk = vec![0; 1 << 20];
    for (name, i) in infos {
        let size = i.data_offsets.1 - i.data_offsets.0;
        if ignored.contains(&name) {
            // Hashed but not kept: vision/MTP weights never affect text inference.
            let mut left = size;
            while left > 0 {
                let k = left.min(chunk.len());
                read(&mut chunk[..k], &mut hash)?;
                left -= k;
            }
            skipped.push(name);
        } else if i.dtype == Dtype::BF16 {
            let mut bits = AlignedBuf::zeroed(size / 2)?;
            read(bits.as_bytes_mut(), &mut hash)?;
            for b in bits.iter_mut() {
                *b = u16::from_le(*b);
            }
            require(!bits.is_empty(), "empty tensor")?;
            require(
                bits.iter().all(|&b| b & 0x7f80 != 0x7f80),
                "NaN/Inf detected",
            )?;
            bf16.insert(
                name,
                Bf16Data {
                    shape: i.shape.clone(),
                    bits,
                },
            );
        } else {
            let mut raw = vec![0; size];
            read(&mut raw, &mut hash)?;
            values.insert(
                name,
                TensorData {
                    shape: i.shape.clone(),
                    values: decode_tensor(i.dtype, &raw)?,
                },
            );
        }
    }
    require(
        format!("{:x}", hash.finalize()) == expected_sha256,
        "checkpoint SHA-256 mismatch",
    )?;
    Ok((values, bf16, skipped))
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
    let ignored: Vec<String> =
        serde_json::from_str(include_str!("../spec/ignored-tensors.json")).map_err(err)?;
    let (values, bf16, mut skipped) = read_checkpoint(
        &dir.join("model.safetensors-00001-of-00001.safetensors"),
        "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
        &ignored,
    )?;
    let mut reader = TensorReader::new(values);
    reader.bf16 = bf16;
    let mut model = model_from_tensors(c, reader, revision, &[])?;
    skipped.sort(); // Stable inspection reports despite safetensors' map iteration order.
    model.unused = skipped;
    Ok(model)
}
pub fn err(e: impl std::fmt::Display) -> InferenceError {
    InferenceError(e.to_string())
}
#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::TensorView;
    /// The streaming reader keeps BF16 as bits, decodes f32, hashes but drops
    /// ignored tensors, and refuses a wrong hash or a truncated file.
    #[test]
    fn streaming_checkpoint_reader() {
        let bf16: Vec<u8> = [0x3f80u16, 0xc000, 0x0000, 0x4040]
            .iter()
            .flat_map(|b| b.to_le_bytes())
            .collect();
        let f32s: Vec<u8> = [1.5f32, -2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let skip = vec![7u8; 4];
        let file = safetensors::serialize(
            [
                (
                    "a",
                    TensorView::new(Dtype::BF16, vec![2, 2], &bf16).unwrap(),
                ),
                ("b", TensorView::new(Dtype::F32, vec![2], &f32s).unwrap()),
                (
                    "vision",
                    TensorView::new(Dtype::BF16, vec![2], &skip).unwrap(),
                ),
            ],
            &None,
        )
        .unwrap();
        let dir = std::env::temp_dir().join(format!("qwen-ckpt-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir(&dir).unwrap();
        let path = dir.join("m.safetensors");
        fs::write(&path, &file).unwrap();
        let sha = format!("{:x}", Sha256::digest(&file));
        let (values, bits, skipped) = read_checkpoint(&path, &sha, &["vision".into()]).unwrap();
        assert_eq!(*bits["a"].bits, [0x3f80, 0xc000, 0x0000, 0x4040]);
        assert_eq!(bits["a"].bits.as_ptr() as usize % crate::tensor::PAGE, 0);
        assert_eq!(bits["a"].shape, [2, 2]);
        assert_eq!(values["b"].values, [1.5, -2.0]);
        assert_eq!(skipped, ["vision"]);
        assert!(read_checkpoint(&path, &"0".repeat(64), &[]).is_err());
        fs::write(&path, &file[..file.len() - 1]).unwrap();
        assert!(read_checkpoint(&path, &sha, &[]).is_err());
        fs::remove_dir_all(dir).unwrap();
    }
}
