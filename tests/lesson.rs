use qwen_rust_inference::{lesson::write_lesson, trace::Trace};
#[test]
fn lesson_has_checkable_writes_probabilities_states_and_logits() {
    let mut trace = Trace::new(Vec::new(), 1_048_576, true);
    let result = write_lesson(&mut trace).unwrap();
    assert_eq!(result.ids, [1, 4, 2]);
    assert_eq!(result.steps.len(), 3);
    for (p, s) in result.steps.iter().enumerate() {
        assert_eq!(s.position, p);
        assert_eq!(s.kv_positions, p + 1);
        assert!((s.head0_probability_sum - 1.).abs() < 1e-6);
        assert!(s.max_write_error < 2e-6);
        assert_eq!(s.recurrent_elements, 96);
        assert_eq!(s.convolution_elements, 144);
    }
    let bytes = trace.into_inner();
    let rows: Vec<serde_json::Value> = std::str::from_utf8(&bytes)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(
        rows.iter()
            .filter(|r| r["operation"] == "embedding")
            .count(),
        3
    );
    assert_eq!(
        rows.iter().filter(|r| r["operation"] == "input").count(),
        12
    );
    assert_eq!(
        rows.iter().filter(|r| r["operation"] == "logits").count(),
        3
    );
}

fn close(x: f64, y: f64) {
    assert!((x - y).abs() <= 2e-6 + 2e-5 * y.abs(), "{x} != {y}");
}
fn rows() -> Vec<serde_json::Value> {
    include_str!("../examples/lesson-trace.ndjson")
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}
fn vector(rows: &[serde_json::Value], p: usize, l: usize, op: &str) -> Vec<f64> {
    let row = rows
        .iter()
        .find(|r| r["token_position"] == p && r["layer"] == l && r["operation"] == op)
        .unwrap();
    row["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}
#[test]
fn saved_exercise_has_independently_checkable_equations() {
    let rows = rows();
    for p in 0..3 {
        // Independent f64 arithmetic verifies equations in the saved worksheet,
        // rather than asking the same Rust kernels to approve their own output.
        for l in 0..4 {
            let input = vector(&rows, p, l, "input");
            let mixed = vector(&rows, p, l, "mixer");
            let after = vector(&rows, p, l, "after_mixer");
            let ffn = vector(&rows, p, l, "ffn.output");
            let layer = vector(&rows, p, l, "layer");
            for i in 0..8 {
                close(input[i] + mixed[i], after[i]);
                close(after[i] + ffn[i], layer[i]);
            }
        }
        let product = vector(&rows, p, 0, "ffn.product");
        let column = vector(&rows, p, 0, "lesson.ffn.column_3");
        let removed = vector(&rows, p, 0, "lesson.ffn.removed_write");
        for i in 0..8 {
            close(product[3] * column[i], removed[i]);
        }
        let q = vector(&rows, p, 3, "attention.q");
        let scores = vector(&rows, p, 3, "attention.scores");
        let probabilities = vector(&rows, p, 3, "attention.probabilities");
        let context = vector(&rows, p, 3, "attention.context");
        for head in 0..2 {
            let expected_scores: Vec<_> = (0..=p)
                .map(|t| {
                    let key = vector(&rows, t, 3, "attention.k");
                    (0..4).map(|d| q[head * 4 + d] * key[d]).sum::<f64>() / 2.
                })
                .collect();
            let max = expected_scores
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let denominator = expected_scores.iter().map(|s| (s - max).exp()).sum::<f64>();
            let expected_p: Vec<_> = expected_scores
                .iter()
                .map(|s| (s - max).exp() / denominator)
                .collect();
            for t in 0..=p {
                close(scores[head * (p + 1) + t], expected_scores[t]);
                close(probabilities[head * (p + 1) + t], expected_p[t]);
            }
            for d in 0..4 {
                let expected = (0..=p)
                    .map(|t| expected_p[t] * vector(&rows, t, 3, "attention.v")[d])
                    .sum::<f64>();
                close(context[head * 4 + d], expected);
            }
        }
    }
}
#[test]
fn saved_artifacts_match_the_current_lesson() {
    let mut trace = Trace::new(Vec::new(), 1_048_576, true);
    let lesson = write_lesson(&mut trace).unwrap();
    let actual: Vec<serde_json::Value> = std::str::from_utf8(&trace.into_inner())
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let expected = rows();
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(&expected) {
        for key in [
            "run_id",
            "token_position",
            "layer",
            "operation",
            "shape",
            "dtype",
            "finite",
        ] {
            assert_eq!(a[key], b[key]);
        }
        for (a, b) in a["values"]
            .as_array()
            .unwrap()
            .iter()
            .zip(b["values"].as_array().unwrap())
        {
            close(a.as_f64().unwrap(), b.as_f64().unwrap());
        }
    }
    let saved: serde_json::Value =
        serde_json::from_str(include_str!("../examples/lesson-answers.json")).unwrap();
    for (p, step) in lesson.steps.iter().enumerate() {
        assert_eq!(saved["steps"][p]["greedy_next"], step.greedy_next);
        close(
            saved["steps"][p]["activation"].as_f64().unwrap(),
            f64::from(step.activation),
        );
        for (i, &v) in step.observed_removed_write.iter().enumerate() {
            close(
                saved["steps"][p]["observed_removed_write"][i]
                    .as_f64()
                    .unwrap(),
                f64::from(v),
            );
        }
    }
}
#[test]
fn saved_delta_state_and_final_projection_can_be_reconstructed() {
    let rows = rows();
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/hybrid.json")).unwrap();
    let embedding: Vec<_> = fixture["tensors"]["model.language_model.embed_tokens.weight"]
        ["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    for p in 0..3 {
        for layer in 0..3 {
            let q = vector(&rows, p, layer, "delta.q");
            let k = vector(&rows, p, layer, "delta.k");
            let v = vector(&rows, p, layer, "delta.v");
            let g = vector(&rows, p, layer, "delta.decay_log");
            let beta = vector(&rows, p, layer, "delta.beta");
            let previous = if p == 0 {
                vec![0.; 32]
            } else {
                vector(&rows, p - 1, layer, "delta.recurrent_state")
            };
            let state = vector(&rows, p, layer, "delta.recurrent_state");
            let core = vector(&rows, p, layer, "delta.core");
            for h in 0..2 {
                let mut expected = [0.; 16];
                for j in 0..4 {
                    let memory = (0..4)
                        .map(|i| k[h * 4 + i] * previous[h * 16 + i * 4 + j] * g[h].exp())
                        .sum::<f64>();
                    let delta = (v[h * 4 + j] - memory) * beta[h];
                    for i in 0..4 {
                        expected[i * 4 + j] =
                            previous[h * 16 + i * 4 + j] * g[h].exp() + k[h * 4 + i] * delta;
                        close(expected[i * 4 + j], state[h * 16 + i * 4 + j]);
                    }
                    let y = (0..4)
                        .map(|i| q[h * 4 + i] * expected[i * 4 + j])
                        .sum::<f64>();
                    close(y, core[h * 4 + j]);
                }
            }
        }
        let normalized = vector(&rows, p, 3, "final_norm");
        let logits = vector(&rows, p, 3, "logits");
        for id in 0..19 {
            let value = (0..8)
                .map(|d| embedding[id * 8 + d] * normalized[d])
                .sum::<f64>();
            close(value, logits[id]);
        }
    }
}
