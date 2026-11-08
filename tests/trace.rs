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
