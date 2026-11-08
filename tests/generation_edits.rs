use qwen_rust_inference::{
    edit::{Edit, EditManifest},
    generate::{decode_with, generate, Sampling, StopReason},
    model::Model,
    tensor::Backend,
};
fn model() -> Model {
    Model::from_fixture(&serde_json::from_str(include_str!("fixtures/hybrid.json")).unwrap())
        .unwrap()
}
#[test]
fn controlled_end_conditions_and_seeded_sampling() {
    // ID 0 stands for </think>, ID 2 for EOS. Ending reasoning must not end
    // the answer: the controlled next distribution generates ID 1, then EOS.
    let mut call = 0;
    let result = decode_with(&[9], vec![9., 0., 0.], 10, &[2], Sampling::Greedy, |_| {
        call += 1;
        Ok(Some(if call == 1 {
            vec![0., 9., 0.]
        } else {
            vec![0., 0., 9.]
        }))
    })
    .unwrap();
    assert_eq!(result.history.generated, [0, 1, 2]);
    assert_eq!(result.stop, StopReason::Eos);
    let result = decode_with(&[9], vec![9., 0.], 3, &[], Sampling::Greedy, |_| {
        Ok(Some(vec![9., 0.]))
    })
    .unwrap();
    assert_eq!(result.history.generated.len(), 3);
    assert_eq!(result.stop, StopReason::MaxTokens);
    let result = decode_with(&[9], vec![9., 0.], 0, &[], Sampling::Greedy, |_| panic!()).unwrap();
    assert!(result.history.generated.is_empty());
    for seed in [1, 17] {
        let sampling = Sampling::Seeded {
            temperature: 0.8,
            top_k: 2,
            seed,
        };
        let a = decode_with(&[9], vec![0., 0., -10.], 20, &[], sampling, |_| {
            Ok(Some(vec![0., 0., -10.]))
        })
        .unwrap();
        let b = decode_with(&[9], vec![0., 0., -10.], 20, &[], sampling, |_| {
            Ok(Some(vec![0., 0., -10.]))
        })
        .unwrap();
        assert_eq!(a.history, b.history);
        assert!(a.history.generated.iter().all(|&i| i < 2));
    }
    assert!(decode_with(
        &[9],
        vec![0., 1.],
        1,
        &[],
        Sampling::Seeded {
            temperature: 0.,
            top_k: 1,
            seed: 1
        },
        |_| panic!()
    )
    .is_err());
}
#[test]
fn generation_runs_have_independent_state() {
    let m = model();
    let a = generate(&m, &[1, 2], 5, &[], Sampling::Greedy, None).unwrap();
    let b = generate(&m, &[1, 2], 5, &[], Sampling::Greedy, None).unwrap();
    assert_eq!(a.history, b.history);
    assert_eq!(a.history.generated.len(), 5);
    assert!(generate(&m, &[], 1, &[], Sampling::Greedy, None).is_err());
    let exhausted =
        decode_with(&[9], vec![9., 0.], 3, &[], Sampling::Greedy, |_| Ok(None)).unwrap();
    assert_eq!(exhausted.stop, StopReason::ContextLimit);
    assert_eq!(exhausted.history.generated.len(), 1);
}
#[test]
fn overlay_direction_unit_zero_and_reversion() {
    let m = model();
    let base = m
        .step(1, &mut m.new_state(), Backend::Reference, None)
        .unwrap();
    let mut manifest = EditManifest {
        model_revision: m.revision.clone(),
        edits: vec![],
    };
    assert_eq!(
        base.logits,
        m.step(1, &mut m.new_state(), Backend::Reference, Some(&manifest))
            .unwrap()
            .logits
    );
    let edit = Edit {
        layer: 0,
        tensor_name: "model.language_model.layers.0.mlp.down_proj.weight".into(),
        columns: vec![3],
        direction: None,
        coefficient: 1.,
    };
    manifest.edits.push(edit);
    let zero = m
        .step(1, &mut m.new_state(), Backend::Reference, Some(&manifest))
        .unwrap();
    let column = m.blocks[0].ffn.writing_vector(3).unwrap();
    for ((a, b), w) in base.blocks[0]
        .ffn
        .iter()
        .zip(&zero.blocks[0].ffn)
        .zip(&column)
    {
        assert!((a - b - w * base.blocks[0].ffn_product[3]).abs() < 2e-6);
    }
    let mut direction = vec![0.; 8];
    direction[2] = 1.;
    manifest.edits[0].direction = Some(direction);
    let projected = m
        .step(1, &mut m.new_state(), Backend::Reference, Some(&manifest))
        .unwrap();
    for (i, w) in column.iter().enumerate() {
        let expected = if i == 2 {
            w * base.blocks[0].ffn_product[3]
        } else {
            0.
        };
        assert!((base.blocks[0].ffn[i] - projected.blocks[0].ffn[i] - expected).abs() < 2e-6);
    }
    assert_ne!(base.logits, projected.logits);
    assert_eq!(
        base.logits,
        m.step(1, &mut m.new_state(), Backend::Reference, None)
            .unwrap()
            .logits
    );
    manifest.model_revision = "wrong".into();
    assert!(m
        .step(1, &mut m.new_state(), Backend::Reference, Some(&manifest))
        .is_err());
    manifest.model_revision = m.revision.clone();
    manifest.edits[0].direction = Some(vec![0.; 8]);
    assert!(m
        .step(1, &mut m.new_state(), Backend::Reference, Some(&manifest))
        .is_err());
    manifest.edits[0].direction = None;
    manifest.edits[0].columns = vec![16];
    assert!(m
        .step(1, &mut m.new_state(), Backend::Reference, Some(&manifest))
        .is_err());
    let serialized = serde_json::to_string(&manifest).unwrap();
    assert_eq!(manifest, serde_json::from_str(&serialized).unwrap());
}
