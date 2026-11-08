//! 保存した配列から計算を逆にたどる、小さな固定教材。
//! 文章の意味を学習していないモデルへ既知IDを順に渡すteacher forcingであり、
//! greedy_nextは観測結果。次の入力をそのIDに置き換えてはいない。
use crate::{
    block::MixerActivations,
    model::Model,
    state::LayerState,
    tensor::{require, Backend, Result},
    trace::{Trace, TraceContext},
    validation::top,
    weights::err,
};
use serde::Serialize;
use std::io::Write;
#[derive(Debug, Serialize)]
pub struct Lesson {
    pub run_id: String,
    pub ids: Vec<u32>,
    pub steps: Vec<LessonStep>,
}
#[derive(Debug, Serialize)]
pub struct LessonStep {
    pub position: usize,
    pub input_id: u32,
    pub greedy_next: usize,
    pub top5: Vec<(usize, f32)>,
    pub kv_positions: usize,
    pub convolution_elements: usize,
    pub recurrent_elements: usize,
    pub unit: usize,
    pub activation: f32,
    pub writing_vector: Vec<f32>,
    pub expected_removed_write: Vec<f32>,
    pub observed_removed_write: Vec<f32>,
    pub max_write_error: f32,
    pub head0_probabilities: Vec<f32>,
    pub head0_probability_sum: f64,
    pub head0_context: Vec<f32>,
}
pub fn write_lesson<W: Write>(trace: &mut Trace<W>) -> Result<Lesson> {
    let model = Model::from_fixture(
        &serde_json::from_str(include_str!("../tests/fixtures/hybrid.json")).map_err(err)?,
    )?;
    let ids = vec![1, 4, 2];
    let run_id = "lesson-v1".to_owned();
    let mut state = model.new_state();
    let mut steps = Vec::new();
    for (position, &id) in ids.iter().enumerate() {
        let a = model.step(id, &mut state, Backend::Reference, None)?;
        trace.step(&run_id, position, &a, None)?;
        let unit = 3;
        // 同じFFN入力で一つの積だけをゼロ化。下流の別入力同士を比べず、
        // 「反応値×書込み列」という局所的な恒等式を直接観測する。
        let activation = a.blocks[0].ffn_product[unit];
        let writing_vector = model.blocks[0].ffn.writing_vector(unit)?;
        let edited = model.blocks[0]
            .ffn
            .forward(&a.blocks[0].post_norm, Some(unit))?;
        let observed_removed_write: Vec<_> = a.blocks[0]
            .ffn
            .iter()
            .zip(&edited.output)
            .map(|(a, b)| a - b)
            .collect();
        let expected_removed_write: Vec<_> =
            writing_vector.iter().map(|w| w * activation).collect();
        let max_write_error = observed_removed_write
            .iter()
            .zip(&expected_removed_write)
            .map(|(a, b)| (a - b).abs())
            .fold(0_f32, f32::max);
        require(max_write_error < 2e-6, "lesson write identity failed")?;
        let ctx = TraceContext {
            run_id: &run_id,
            token_position: position,
            layer: 0,
        };
        for (name, v) in [
            ("lesson.ffn.column_3", &writing_vector),
            ("lesson.ffn.removed_write", &observed_removed_write),
        ] {
            trace.record(&ctx, name, v, None)?;
        }
        let mut convolution_elements = 0;
        let mut recurrent_elements = 0;
        let mut kv_positions = 0;
        for s in &state.layers {
            match s {
                LayerState::Delta(s) => {
                    convolution_elements += s.convolution.len();
                    recurrent_elements += s.recurrent.len();
                }
                LayerState::Attention(s) => kv_positions = s.keys.len(),
            }
        }
        let MixerActivations::Attention(attention) = &a.blocks[3].mixer else {
            unreachable!()
        };
        let head0_probabilities = attention.probabilities[..position + 1].to_vec();
        let head0_probability_sum = head0_probabilities.iter().map(|&p| f64::from(p)).sum();
        let candidates = top(&a.logits, 5);
        let greedy_next = candidates[0].0;
        steps.push(LessonStep {
            position,
            input_id: id,
            greedy_next,
            top5: candidates,
            kv_positions,
            convolution_elements,
            recurrent_elements,
            unit,
            activation,
            writing_vector,
            expected_removed_write,
            observed_removed_write,
            max_write_error,
            head0_probabilities,
            head0_probability_sum,
            head0_context: attention.context[..4].to_vec(),
        });
    }
    Ok(Lesson { run_id, ids, steps })
}
