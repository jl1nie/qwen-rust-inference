//! A cache of the last layer's FFN inputs, so training does not rerun the
//! layers below.
//!
//! Training changes only the last layer's FFN. What that FFN reads
//! (`post_norm`) and what its output is added to (`after_mixer`) come from
//! frozen layers, so they are computed once by the inference path and saved.
//!
//! File format: the magic `QWTRAIN-CACHE-1\n`, a u64 LE header length, a JSON
//! header (`CacheHeader`), then `after_mixer` and `post_norm` as f32 LE,
//! `[n, hidden]` each. The header names the model identity, so a cache made
//! with other weights is refused, and the sha256 of the data.
use crate::model::Model;
use crate::tensor::{require, Backend, InferenceError, Result};
use crate::train::ffn::TopInputs;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::Path;

const MAGIC: &[u8] = b"QWTRAIN-CACHE-1\n";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheHeader {
    pub model_identity: String,
    pub model_revision: String,
    pub layer: usize,
    pub hidden: usize,
    /// The token IDs read, one per position. The target of position i is
    /// `ids[i + 1]`, so the last position has no target in the same text.
    pub ids: Vec<u32>,
    /// sha256 of the f32 data that follows the header.
    pub data_sha256: String,
}

/// The cached inputs with the IDs they came from.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerCache {
    pub header: CacheHeader,
    pub inputs: TopInputs,
}

fn io(e: std::io::Error) -> InferenceError {
    InferenceError(format!("cache I/O: {e}"))
}

fn data_bytes(inputs: &TopInputs) -> Vec<u8> {
    inputs
        .after_mixer
        .iter()
        .chain(&inputs.post_norm)
        .flat_map(|v| v.to_le_bytes())
        .collect()
}

impl LayerCache {
    /// Reads `ids` from the start with the inference path and keeps the last
    /// layer's FFN inputs at every position.
    pub fn capture(model: &Model, ids: &[u32], backend: Backend) -> Result<Self> {
        require(!ids.is_empty(), "no IDs to capture")?;
        let layer = model.blocks.len() - 1;
        let mut after_mixer = Vec::new();
        let mut post_norm = Vec::new();
        let mut state = model.new_state();
        model.prefill_observed(ids, &mut state, backend, None, |_, step| {
            after_mixer.extend_from_slice(&step.blocks[layer].after_mixer);
            post_norm.extend_from_slice(&step.blocks[layer].post_norm);
            Ok(())
        })?;
        let inputs = TopInputs {
            after_mixer,
            post_norm,
            n: ids.len(),
        };
        let header = CacheHeader {
            model_identity: model.identity.clone(),
            model_revision: model.revision.clone(),
            layer,
            hidden: model.config.hidden_size,
            ids: ids.to_vec(),
            data_sha256: format!("{:x}", Sha256::digest(data_bytes(&inputs))),
        };
        Ok(Self { header, inputs })
    }
    pub fn write(&self, path: &Path) -> Result<()> {
        let header = serde_json::to_vec(&self.header).map_err(|e| InferenceError(e.to_string()))?;
        let mut f = std::io::BufWriter::new(std::fs::File::create(path).map_err(io)?);
        f.write_all(MAGIC).map_err(io)?;
        f.write_all(&(header.len() as u64).to_le_bytes())
            .map_err(io)?;
        f.write_all(&header).map_err(io)?;
        f.write_all(&data_bytes(&self.inputs)).map_err(io)?;
        f.flush().map_err(io)
    }
    /// Reads a cache and checks it was made by `model` (same identity, same
    /// shape) and that its data is intact.
    pub fn read(path: &Path, model: &Model) -> Result<Self> {
        let mut f = std::io::BufReader::new(std::fs::File::open(path).map_err(io)?);
        let mut magic = vec![0; MAGIC.len()];
        f.read_exact(&mut magic).map_err(io)?;
        require(magic == MAGIC, "not a training cache")?;
        let mut len = [0; 8];
        f.read_exact(&mut len).map_err(io)?;
        let len = usize::try_from(u64::from_le_bytes(len))
            .ok()
            .filter(|&l| l < 1 << 30)
            .ok_or_else(|| InferenceError("cache header too large".into()))?;
        let mut header = vec![0; len];
        f.read_exact(&mut header).map_err(io)?;
        let header: CacheHeader =
            serde_json::from_slice(&header).map_err(|e| InferenceError(e.to_string()))?;
        require(
            header.model_identity == model.identity,
            "cache was made with a different model",
        )?;
        require(
            header.layer + 1 == model.blocks.len() && header.hidden == model.config.hidden_size,
            "cache shape does not match the model",
        )?;
        let mut data = Vec::new();
        f.read_to_end(&mut data).map_err(io)?;
        let n = header.ids.len();
        require(
            data.len() == 2 * n * header.hidden * 4,
            "cache data length mismatch",
        )?;
        require(
            format!("{:x}", Sha256::digest(&data)) == header.data_sha256,
            "cache data checksum mismatch",
        )?;
        let values: Vec<f32> = data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let (after_mixer, post_norm) = values.split_at(n * header.hidden);
        Ok(Self {
            inputs: TopInputs {
                after_mixer: after_mixer.to_vec(),
                post_norm: post_norm.to_vec(),
                n,
            },
            header,
        })
    }
}
