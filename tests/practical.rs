use qwen_rust_inference::{
    generate::Sampling, model::Model, session::InferenceSession, tensor::Backend,
};
fn model() -> Model {
    Model::from_fixture(&serde_json::from_str(include_str!("fixtures/hybrid.json")).unwrap())
        .unwrap()
}
#[test]
fn compact_matches_full_and_pending_is_consumed_once() {
    let m = model();
    for b in [Backend::Reference, Backend::Optimized, Backend::Vector] {
        let mut full = m.new_state();
        let mut compact = m.new_state();
        for id in [1, 4, 2, 8] {
            let a = m.step(id, &mut full, b, None).unwrap();
            let logits = m.step_compact(id, &mut compact, b, None).unwrap();
            assert_eq!(logits, a.logits);
            assert_eq!(full, compact);
        }
        let mut session = InferenceSession::new(&m, b, Sampling::Greedy).unwrap();
        session.prefill(&m, &[1, 4, 2]).unwrap();
        let a = session.generate(&m, 2, &[]).unwrap();
        assert_eq!(a.history.generated.len(), 2);
        assert!(session.pending().is_some());
        assert_eq!(session.tokens().len(), session.state().position + 1);
        let b = session.generate(&m, 3, &[]).unwrap();
        let mut one = InferenceSession::new(&m, b_backend(), Sampling::Greedy).unwrap();
        one.prefill(&m, &[1, 4, 2]).unwrap();
        let all = one.generate(&m, 5, &[]).unwrap();
        assert_eq!(
            [a.history.generated, b.history.generated].concat(),
            all.history.generated
        );
    }
}
fn b_backend() -> Backend {
    Backend::Optimized
}
#[test]
fn greedy_ties_and_context_pending_boundary() {
    use qwen_rust_inference::generate::{select, StopReason};
    assert_eq!(select(&[-0., 0.], Sampling::Greedy, &mut 0).unwrap(), 0);
    let mut v: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/hybrid.json")).unwrap();
    v["config"]["max_position_embeddings"] = 3.into();
    let m = Model::from_fixture(&v).unwrap();
    let mut s = InferenceSession::new(&m, Backend::Vector, Sampling::Greedy).unwrap();
    s.prefill(&m, &[1, 4, 2]).unwrap();
    let first = s.generate(&m, 4, &[]).unwrap();
    assert_eq!(first.history.generated.len(), 1);
    assert_eq!(first.stop, StopReason::ContextLimit);
    let before = s.clone();
    let next = s.generate(&m, 2, &[]).unwrap();
    assert!(next.history.generated.is_empty());
    assert_eq!(s, before);
}
#[test]
fn a_partially_failed_run_cannot_continue_until_reset() {
    let good = model();
    let mut v: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/hybrid.json")).unwrap();
    v["tensors"]["model.language_model.norm.weight"]["values"] =
        serde_json::to_value(vec![f32::MAX; 8]).unwrap();
    let bad = Model::from_fixture(&v).unwrap();
    let mut s = InferenceSession::new(&bad, Backend::Vector, Sampling::Greedy).unwrap();
    assert!(s.prefill(&bad, &[1]).is_err());
    assert!(s
        .prefill(&bad, &[1])
        .unwrap_err()
        .to_string()
        .contains("session failed"));
    s.reset(&good).unwrap();
    s.prefill(&good, &[1]).unwrap();
    assert_eq!(s.state().position, 1);
}
