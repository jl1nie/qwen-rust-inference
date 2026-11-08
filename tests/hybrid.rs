use qwen_rust_inference::{model::Model, tensor::Backend};
#[test]
fn hybrid_fixture_boundaries_and_split_prefill() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/hybrid.json")).unwrap();
    let model = Model::from_fixture(&fixture).unwrap();
    let ids: Vec<u32> = fixture["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let mut state = model.new_state();
    for (p, &id) in ids.iter().enumerate() {
        let step = model
            .step(id, &mut state, Backend::Reference, None)
            .unwrap();
        for (l, hidden) in step.layers.iter().enumerate() {
            compare(hidden, &numbers(&fixture["layers"][p][l]));
        }
        compare(&step.logits, &numbers(&fixture["logits"][p]));
    }
    let final_state = state.clone();
    for split in 0..=ids.len() {
        let mut s = model.new_state();
        let mut outputs = model
            .prefill(&ids[..split], &mut s, Backend::Reference, None)
            .unwrap();
        outputs.extend(
            model
                .prefill(&ids[split..], &mut s, Backend::Reference, None)
                .unwrap(),
        );
        assert_eq!(s, final_state);
        assert_eq!(outputs.len(), ids.len());
    }
    state.reset();
    assert_eq!(state, model.new_state());
    let mut fast = model.new_state();
    let mut slow = model.new_state();
    for id in ids {
        compare(
            &model
                .step(id, &mut fast, Backend::Optimized, None)
                .unwrap()
                .logits,
            &model
                .step(id, &mut slow, Backend::Reference, None)
                .unwrap()
                .logits,
        );
    }
}
pub fn numbers(v: &serde_json::Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}
pub fn compare(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        assert!(
            x.is_finite() && (x - y).abs() <= 2e-5 + 2e-4 * y.abs(),
            "index={i} {x} != {y}, abs={}",
            (x - y).abs()
        );
    }
}
#[test]
fn hybrid_fixture_provenance_and_chunk_reference() {
    use sha2::{Digest, Sha256};
    let bytes = include_bytes!("fixtures/hybrid.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        include_str!("fixtures/hybrid.sha256").trim()
    );
    let fixture: serde_json::Value = serde_json::from_slice(bytes).unwrap();
    assert_eq!(
        format!(
            "{:x}",
            Sha256::digest(include_bytes!("../tools/generate_hybrid.py"))
        ),
        fixture["metadata"]["generator_sha256"]
    );
    assert_eq!(
        format!(
            "{:x}",
            Sha256::digest(include_bytes!("../spec/upstream/modeling_qwen3_5.py"))
        ),
        fixture["metadata"]["source_sha256"]
    );
    let m = Model::from_fixture(&fixture).unwrap();
    let mut s = m.new_state();
    for (p, id) in fixture["ids"].as_array().unwrap().iter().enumerate() {
        let a = m
            .step(
                id.as_u64().unwrap() as u32,
                &mut s,
                Backend::Reference,
                None,
            )
            .unwrap();
        compare(&a.logits, &numbers(&fixture["full_logits"][p]));
    }
}
