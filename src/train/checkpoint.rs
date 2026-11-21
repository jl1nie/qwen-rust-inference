//! A trained FFN saved to a file, with the optimizer state so that training
//! can resume exactly where it stopped.
//!
//! The file is safetensors: `gate`, `up`, `down` in f32, and for Adam
//! `adam.m.{0,1,2}` and `adam.v.{0,1,2}`. The metadata names the model the
//! FFN was trained on (`base_identity`) and the layer; `apply` refuses any
//! other model.
use crate::model::Model;
use crate::tensor::{require, InferenceError, Result};
use crate::train::ffn::TrainFfn;
use crate::train::linalg::Dense;
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
        for (name, d) in [
            ("gate", &self.ffn.gate),
            ("up", &self.ffn.up),
            ("down", &self.ffn.down),
        ] {
            tensors.push((name.into(), vec![d.rows(), d.cols()], bytes(d.values())));
        }
        let mut meta = HashMap::from([
            ("format".to_string(), FORMAT.to_string()),
            ("base_identity".into(), self.base_identity.clone()),
            ("layer".into(), self.layer.to_string()),
        ]);
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

    pub fn read(path: &Path) -> Result<Self> {
        let data = std::fs::read(path).map_err(err)?;
        let st = SafeTensors::deserialize(&data).map_err(err)?;
        let (_, header) = SafeTensors::read_metadata(&data).map_err(err)?;
        let meta = header
            .metadata()
            .as_ref()
            .ok_or_else(|| err("no metadata"))?;
        let get = |k: &str| meta.get(k).ok_or_else(|| err(format!("missing {k}")));
        require(get("format")? == FORMAT, "not an FFN checkpoint")?;
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
        let ffn = TrainFfn {
            gate: dense("gate")?,
            up: dense("up")?,
            down: dense("down")?,
        };
        ffn.to_weights()?; // shape check
        let optimizer = match meta.get("optimizer") {
            None => None,
            Some(c) => {
                let config: OptimizerConfig = serde_json::from_str(c).map_err(err)?;
                let mut o = Optimizer::new(config, &ffn.sizes());
                o.t = get("t")?.parse().map_err(err)?;
                if !o.m.is_empty() {
                    for i in 0..3 {
                        o.m[i] = values(&format!("adam.m.{i}"))?.1;
                        o.v[i] = values(&format!("adam.v.{i}"))?.1;
                    }
                    require(
                        o.m.iter()
                            .zip(&o.v)
                            .zip(ffn.sizes())
                            .all(|((m, v), n)| m.len() == n && v.len() == n),
                        "optimizer state shape mismatch",
                    )?;
                }
                Some(o)
            }
        };
        Ok(Self {
            base_identity: get("base_identity")?.clone(),
            layer: get("layer")?.parse().map_err(err)?,
            ffn,
            optimizer,
        })
    }

    /// `model` with this FFN in its layer. Refuses a model other than the one
    /// the FFN was trained from.
    pub fn apply(&self, model: &Model) -> Result<Model> {
        require(
            self.base_identity == model.identity(),
            "checkpoint was trained from a different model",
        )?;
        model.with_ffn(self.layer, self.ffn.to_weights()?)
    }
}
