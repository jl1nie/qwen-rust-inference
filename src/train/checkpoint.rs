//! A trained FFN saved to a file, with the optimizer state so that training
//! can resume exactly where it stopped.
//!
//! The file is safetensors. Without LoRA it holds `gate`, `up`, `down` in f32.
//! With LoRA it holds only the adapters, `lora.{gate,up,down}.{a,b}` (about
//! 1% of the FFN); the frozen matrices are taken from the model on reading.
//! With Adam, also `adam.m.{i}` and `adam.v.{i}` for each trained tensor.
//! The metadata names the model the FFN was trained from (`base_identity`)
//! and the layer; reading refuses any other model.
use crate::model::Model;
use crate::tensor::{require, InferenceError, Result};
use crate::train::ffn::TrainFfn;
use crate::train::linalg::Dense;
use crate::train::lora::{Lora, LoraSet, PROJECTIONS};
use crate::train::optim::{Optimizer, OptimizerConfig};
use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::path::Path;

const FORMAT: &str = "qwen-train-ffn-1";

#[derive(Debug, Clone, PartialEq)]
pub struct FfnCheckpoint {
    /// Identity of the model the FFN was trained from.
    pub base_identity: String,
    pub layer: usize,
    pub ffn: TrainFfn,
    pub optimizer: Option<Optimizer>,
}

fn err(e: impl std::fmt::Display) -> InferenceError {
    InferenceError(format!("checkpoint: {e}"))
}
fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

impl FfnCheckpoint {
    pub fn write(&self, path: &Path) -> Result<()> {
        let mut tensors: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
        let mut put = |name: String, d: &Dense| {
            tensors.push((name, vec![d.rows(), d.cols()], bytes(d.values())));
        };
        let mut meta = HashMap::from([
            ("format".to_string(), FORMAT.to_string()),
            ("base_identity".into(), self.base_identity.clone()),
            ("layer".into(), self.layer.to_string()),
        ]);
        match &self.ffn.lora {
            None => {
                put("gate".into(), &self.ffn.gate);
                put("up".into(), &self.ffn.up);
                put("down".into(), &self.ffn.down);
            }
            Some(l) => {
                meta.insert("lora_rank".into(), l.rank.to_string());
                meta.insert("lora_alpha".into(), l.alpha.to_string());
                for (name, a) in PROJECTIONS.iter().zip(&l.adapters) {
                    if let Some(a) = a {
                        put(format!("lora.{name}.a"), &a.a);
                        put(format!("lora.{name}.b"), &a.b);
                    }
                }
            }
        }
        if let Some(o) = &self.optimizer {
            meta.insert(
                "optimizer".into(),
                serde_json::to_string(&o.config).map_err(err)?,
            );
            meta.insert("t".into(), o.t.to_string());
            for (kind, state) in [("m", &o.m), ("v", &o.v)] {
                for (i, s) in state.iter().enumerate() {
                    tensors.push((format!("adam.{kind}.{i}"), vec![s.len()], bytes(s)));
                }
            }
        }
        let views = tensors
            .iter()
            .map(|(n, shape, b)| {
                Ok((
                    n.as_str(),
                    TensorView::new(Dtype::F32, shape.clone(), b).map_err(err)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let data = safetensors::serialize(views, &Some(meta)).map_err(err)?;
        std::fs::write(path, data).map_err(err)
    }

    /// Reads a checkpoint trained from `model` (refusing any other model) and
    /// rebuilds the trainable FFN; for LoRA, the frozen matrices come from
    /// `model`.
    pub fn read(path: &Path, model: &Model) -> Result<Self> {
        let data = std::fs::read(path).map_err(err)?;
        let st = SafeTensors::deserialize(&data).map_err(err)?;
        let (_, header) = SafeTensors::read_metadata(&data).map_err(err)?;
        let meta = header
            .metadata()
            .as_ref()
            .ok_or_else(|| err("no metadata"))?;
        let get = |k: &str| meta.get(k).ok_or_else(|| err(format!("missing {k}")));
        require(get("format")? == FORMAT, "not an FFN checkpoint")?;
        let base_identity = get("base_identity")?.clone();
        require(
            base_identity == model.identity(),
            "checkpoint was trained from a different model",
        )?;
        let layer: usize = get("layer")?.parse().map_err(err)?;
        require(
            layer < model.blocks().len(),
            "checkpoint layer out of range",
        )?;
        let values = |name: &str| -> Result<(Vec<usize>, Vec<f32>)> {
            let t = st.tensor(name).map_err(err)?;
            require(t.dtype() == Dtype::F32, "checkpoint tensor is not f32")?;
            let v = t
                .data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            Ok((t.shape().to_vec(), v))
        };
        let dense = |name: &str| -> Result<Dense> {
            let (shape, v) = values(name)?;
            require(shape.len() == 2, "checkpoint matrix is not 2-D")?;
            Dense::new(shape[0], shape[1], v)
        };
        let base = TrainFfn::from_weights(&model.blocks()[layer].ffn);
        let ffn = match meta.get("lora_rank") {
            None => {
                let ffn = TrainFfn {
                    gate: dense("gate")?,
                    up: dense("up")?,
                    down: dense("down")?,
                    lora: None,
                };
                require(
                    ffn.sizes() == base.sizes(),
                    "checkpoint FFN shape does not match the model",
                )?;
                ffn
            }
            Some(rank) => {
                let rank: usize = rank.parse().map_err(err)?;
                let alpha: f32 = get("lora_alpha")?.parse().map_err(err)?;
                let mut adapters: [Option<Lora>; 3] = [None, None, None];
                for (i, name) in PROJECTIONS.iter().enumerate() {
                    if st.tensor(&format!("lora.{name}.a")).is_err() {
                        continue;
                    }
                    let l = Lora {
                        a: dense(&format!("lora.{name}.a"))?,
                        b: dense(&format!("lora.{name}.b"))?,
                        scale: alpha / rank as f32,
                    };
                    let w = [&base.gate, &base.up, &base.down][i];
                    require(
                        l.rank() == rank
                            && l.a.cols() == w.cols()
                            && l.b.rows() == w.rows()
                            && l.b.cols() == rank,
                        "LoRA shape does not match the model",
                    )?;
                    adapters[i] = Some(l);
                }
                require(adapters.iter().any(Option::is_some), "no LoRA adapters")?;
                TrainFfn {
                    lora: Some(LoraSet {
                        rank,
                        alpha,
                        adapters,
                    }),
                    ..base
                }
            }
        };
        let optimizer = match meta.get("optimizer") {
            None => None,
            Some(c) => {
                let config: OptimizerConfig = serde_json::from_str(c).map_err(err)?;
                let sizes = ffn.sizes();
                let mut o = Optimizer::new(config, &sizes);
                o.t = get("t")?.parse().map_err(err)?;
                if !o.m.is_empty() {
                    for i in 0..sizes.len() {
                        o.m[i] = values(&format!("adam.m.{i}"))?.1;
                        o.v[i] = values(&format!("adam.v.{i}"))?.1;
                    }
                    require(
                        o.m.iter()
                            .zip(&o.v)
                            .zip(&sizes)
                            .all(|((m, v), &n)| m.len() == n && v.len() == n),
                        "optimizer state shape mismatch",
                    )?;
                }
                Some(o)
            }
        };
        Ok(Self {
            base_identity,
            layer,
            ffn,
            optimizer,
        })
    }

    /// `model` with this FFN (adapters folded in) in its layer. Refuses a
    /// model other than the one the FFN was trained from.
    pub fn apply(&self, model: &Model) -> Result<Model> {
        require(
            self.base_identity == model.identity(),
            "checkpoint was trained from a different model",
        )?;
        model.with_ffn(self.layer, self.ffn.to_weights()?)
    }
}
