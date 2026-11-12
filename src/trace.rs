use crate::tensor::{finite, require, InferenceError, Result};
use serde::Serialize;
use std::io::Write;

pub struct TraceContext<'a> {
    pub run_id: &'a str,
    pub token_position: usize,
    pub layer: usize,
}
#[derive(Serialize)]
struct SelectedUnit {
    index: usize,
    value: f32,
}
#[derive(Serialize)]
struct Record<'a> {
    run_id: &'a str,
    /// Set only for device-engine traces; CPU records are unchanged.
    #[serde(skip_serializing_if = "Option::is_none")]
    engine: Option<&'a str>,
    token_position: usize,
    layer: usize,
    operation: &'a str,
    shape: [usize; 1],
    dtype: &'a str,
    norm: f64,
    min: f32,
    max: f32,
    finite: bool,
    selected_unit: Option<SelectedUnit>,
    logits_top: Option<Vec<(usize, f32)>>,
    values: Option<&'a [f32]>,
}
/// Local writer; no network capability. Caller must explicitly construct a trace.
pub struct Trace<W> {
    writer: W,
    max_bytes: usize,
    bytes: usize,
    full: bool,
    engine: Option<String>,
}
impl<W: Write> Trace<W> {
    pub fn new(writer: W, max_bytes: usize, full: bool) -> Self {
        Self {
            writer,
            max_bytes,
            bytes: 0,
            full,
            engine: None,
        }
    }
    /// Tag every record with the engine that produced it (e.g. "metal").
    pub fn with_engine(mut self, engine: &str) -> Self {
        self.engine = Some(engine.to_string());
        self
    }
    pub fn into_inner(self) -> W {
        self.writer
    }
    pub fn record(
        &mut self,
        context: &TraceContext<'_>,
        operation: &str,
        values: &[f32],
        unit: Option<usize>,
    ) -> Result<()> {
        finite(values)?;
        let selected_unit = match unit {
            Some(index) => {
                require(index < values.len(), "trace unit out of bounds")?;
                Some(SelectedUnit {
                    index,
                    value: values[index],
                })
            }
            None => None,
        };
        let record = Record {
            run_id: context.run_id,
            engine: self.engine.as_deref(),
            token_position: context.token_position,
            layer: context.layer,
            operation,
            shape: [values.len()],
            dtype: "f32",
            norm: values
                .iter()
                .map(|&v| f64::from(v).powi(2))
                .sum::<f64>()
                .sqrt(),
            min: values.iter().copied().fold(f32::INFINITY, f32::min),
            max: values.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            finite: true,
            selected_unit,
            logits_top: if operation == "logits" {
                Some(crate::validation::top(values, 5))
            } else {
                None
            },
            values: if self.full { Some(values) } else { None },
        };
        let mut line = serde_json::to_vec(&record).map_err(|e| InferenceError(e.to_string()))?;
        line.push(b'\n');
        require(
            line.len() <= self.max_bytes.saturating_sub(self.bytes),
            "trace byte limit exceeded",
        )?;
        self.writer
            .write_all(&line)
            .map_err(|e| InferenceError(e.to_string()))?;
        self.bytes += line.len();
        Ok(())
    }
}
/// Emit one token's causal path. Filters apply before serialization, so a real
/// checkpoint can be inspected without accidentally dumping every activation.
impl<W: Write> Trace<W> {
    pub fn step(
        &mut self,
        run_id: &str,
        position: usize,
        a: &crate::model::StepActivations,
        layer: Option<usize>,
    ) -> Result<()> {
        use crate::block::MixerActivations;
        if layer.is_none_or(|l| l == 0) {
            self.record(
                &TraceContext {
                    run_id,
                    token_position: position,
                    layer: 0,
                },
                "embedding",
                &a.embedding,
                None,
            )?;
        }
        for (i, b) in a.blocks.iter().enumerate() {
            if layer.is_some_and(|l| l != i) {
                continue;
            }
            let ctx = TraceContext {
                run_id,
                token_position: position,
                layer: i,
            };
            let input = if i == 0 {
                &a.embedding
            } else {
                &a.layers[i - 1]
            };
            self.record(&ctx, "input", input, None)?;
            self.record(&ctx, "input_norm", &b.input_norm, None)?;
            match &b.mixer {
                MixerActivations::Attention(v) => {
                    for (op, values) in [
                        ("attention.projected_q_gate", &v.projected_q_gate),
                        ("attention.projected_k", &v.projected_key),
                        ("attention.q_before_rope", &v.query_before_rope),
                        ("attention.k_before_rope", &v.key_before_rope),
                        ("attention.q", &v.query),
                        ("attention.k", &v.key),
                        ("attention.v", &v.value),
                        ("attention.gate", &v.gate),
                        ("attention.scores", &v.scores),
                        ("attention.probabilities", &v.probabilities),
                        ("attention.context", &v.context),
                        ("attention.gated", &v.gated),
                    ] {
                        self.record(&ctx, op, values, None)?;
                    }
                }
                MixerActivations::Delta(v) => {
                    for (op, values) in [
                        ("delta.projected", &v.projected),
                        ("delta.conv_sum", &v.convolution_sums),
                        ("delta.a", &v.a),
                        ("delta.b", &v.b),
                        ("delta.z", &v.z),
                        ("delta.conv_silu", &v.convolved),
                        ("delta.q", &v.query),
                        ("delta.k", &v.key),
                        ("delta.v", &v.value),
                        ("delta.decay_log", &v.decay),
                        ("delta.beta", &v.beta),
                        ("delta.core", &v.core),
                        ("delta.gated", &v.gated),
                        ("delta.recurrent_state", &v.recurrent_state),
                    ] {
                        self.record(&ctx, op, values, None)?;
                    }
                    if !v.convolution_state.is_empty() {
                        self.record(&ctx, "delta.conv_state", &v.convolution_state, None)?;
                    }
                }
            }
            for (op, values) in [
                ("mixer", &b.mixed),
                ("after_mixer", &b.after_mixer),
                ("post_norm", &b.post_norm),
                ("ffn.gate", &b.ffn_activations.gate),
                ("ffn.up", &b.ffn_activations.up),
                ("ffn.silu", &b.ffn_activations.activated_gate),
                ("ffn.product", &b.ffn_product),
                (
                    "ffn.intervened_product",
                    &b.ffn_activations.intervened_product,
                ),
                ("ffn.output", &b.ffn),
                ("layer", &b.output),
            ] {
                self.record(&ctx, op, values, None)?;
            }
        }
        let ctx = TraceContext {
            run_id,
            token_position: position,
            layer: a.layers.len() - 1,
        };
        self.record(&ctx, "final_norm", &a.normalized, None)?;
        self.record(&ctx, "logits", &a.logits, None)
    }
}
