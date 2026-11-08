use qwen_rust_inference::trace::{Trace, TraceContext};
#[test]
fn trace_is_bounded_valid_and_opt_in() {
    let context = TraceContext {
        run_id: "test\"escaped",
        token_position: 0,
        layer: 0,
    };
    let mut summary = Trace::new(Vec::new(), 10000, false);
    summary
        .record(&context, "ffn.product", &[1., 2.], Some(1))
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&summary.into_inner()).unwrap();
    assert!(v["values"].is_null());
    assert_eq!(v["selected_unit"]["value"], 2.);
    assert_eq!(v["dtype"], "f32");
    assert_eq!(v["shape"][0], 2);
    let mut full = Trace::new(Vec::new(), 10000, true);
    full.record(&context, "ffn.product", &[1., 2.], None)
        .unwrap();
    let bytes = full.into_inner();
    assert_eq!(bytes.last(), Some(&b'\n'));
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["values"][0], 1.);
    let mut bounded = Trace::new(Vec::new(), bytes.len() - 1, true);
    assert!(bounded
        .record(&context, "ffn.product", &[1., 2.], None)
        .is_err());
    assert!(bounded.into_inner().is_empty());
    let mut exact = Trace::new(Vec::new(), bytes.len(), true);
    exact
        .record(&context, "ffn.product", &[1., 2.], None)
        .unwrap();
    assert!(exact
        .record(&context, "ffn.product", &[1., 2.], None)
        .is_err());
    let mut invalid = Trace::new(Vec::new(), 10000, true);
    assert!(invalid.record(&context, "x", &[f32::NAN], None).is_err());
    assert!(invalid.record(&context, "x", &[1.], Some(1)).is_err());
}
#[test]
fn logits_record_has_candidates_and_filters_layers() {
    use qwen_rust_inference::{model::Model, tensor::Backend};
    let model =
        Model::from_fixture(&serde_json::from_str(include_str!("fixtures/hybrid.json")).unwrap())
            .unwrap();
    let a = model
        .step(1, &mut model.new_state(), Backend::Reference, None)
        .unwrap();
    let mut trace = Trace::new(Vec::new(), 65536, false);
    trace.step("test", 0, &a, Some(3)).unwrap();
    let bytes = trace.into_inner();
    let lines: Vec<serde_json::Value> = std::str::from_utf8(&bytes)
        .unwrap()
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    assert!(lines.iter().all(|r| r["layer"] == 3));
    let logits = lines.iter().find(|r| r["operation"] == "logits").unwrap();
    assert_eq!(logits["logits_top"].as_array().unwrap().len(), 5);
}
