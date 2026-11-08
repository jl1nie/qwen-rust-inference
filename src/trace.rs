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
}
impl<W: Write> Trace<W> {
    pub fn new(writer: W, max_bytes: usize, full: bool) -> Self {
        Self {
            writer,
            max_bytes,
            bytes: 0,
            full,
        }
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
            logits_top: None,
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
