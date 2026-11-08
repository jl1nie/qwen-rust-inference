use qwen_rust_inference::{model::Model, tensor::Backend};
#[test]
fn streaming_prefill_matches_steps_without_retaining_history() {
    let m =
        Model::from_fixture(&serde_json::from_str(include_str!("fixtures/hybrid.json")).unwrap())
            .unwrap();
    let mut s = m.new_state();
    let initial = s.clone();
    let mut count = 0;
    assert!(m
        .prefill_observed(&[], &mut s, Backend::Reference, None, |_, _| panic!())
        .unwrap()
        .is_none());
    assert_eq!(s, initial);
    let ids = [1, 4, 2, 8, 3];
    let mut last = None;
    for group in [&ids[..1], &ids[1..3], &ids[3..]] {
        last = m
            .prefill_observed(group, &mut s, Backend::Reference, None, |p, a| {
                assert_eq!(p, count);
                assert_eq!(a.layers.len(), 4);
                count += 1;
                Ok(())
            })
            .unwrap();
    }
    let mut direct = m.new_state();
    let outputs = m
        .prefill(&ids, &mut direct, Backend::Reference, None)
        .unwrap();
    assert_eq!(count, ids.len());
    assert_eq!(last.unwrap(), outputs.last().unwrap().logits);
    assert_eq!(s, direct);
    assert!(m
        .prefill_observed(&[u32::MAX], &mut s, Backend::Reference, None, |_, _| Ok(()))
        .is_err());
    assert_eq!(s, direct);
}
#[test]
fn observer_error_consumes_the_current_position_and_reset_restores_run() {
    use qwen_rust_inference::tensor::InferenceError;
    let m =
        Model::from_fixture(&serde_json::from_str(include_str!("fixtures/hybrid.json")).unwrap())
            .unwrap();
    let mut s = m.new_state();
    assert!(m
        .prefill_observed(&[1, 4], &mut s, Backend::Reference, None, |_, _| Err(
            InferenceError("observer failed".into())
        ))
        .is_err());
    assert_eq!(s.position, 1);
    s.reset();
    assert_eq!(s, m.new_state());
}
