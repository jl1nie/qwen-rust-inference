use qwen_rust_inference::{
    block::MixerWeights, deltanet::delta_rule, model::Model, state::KvCache, tensor::Backend,
};
fn model() -> Model {
    Model::from_fixture(&serde_json::from_str(include_str!("fixtures/hybrid.json")).unwrap())
        .unwrap()
}
fn close(a: &[f32], b: &[f32]) {
    for (a, b) in a.iter().zip(b) {
        assert!((a - b).abs() <= 2e-5 + 2e-4 * b.abs(), "{a} != {b}");
    }
}
#[test]
fn attention_dense_causal_gqa_and_cached_offsets() {
    let m = model();
    let MixerWeights::Attention(w) = &m.blocks()[3].mixer else {
        panic!()
    };
    let xs: Vec<_> = (0..5)
        .map(|i| m.embedding().row(i).unwrap().to_vec())
        .collect();
    assert!(w.sequence(&[], Backend::Reference).unwrap().is_empty());
    let full = w.sequence(&xs, Backend::Reference).unwrap();
    let mut cache = KvCache::default();
    assert!(w.step(&xs[0], 1, &mut cache, Backend::Reference).is_err());
    for (i, x) in xs.iter().enumerate() {
        let a = w.step(x, i, &mut cache, Backend::Reference).unwrap();
        close(&a.output, &full[i].output);
    }
    let mut future = xs.clone();
    future[4].fill(100.);
    let changed = w.sequence(&future, Backend::Reference).unwrap();
    for i in 0..4 {
        assert_eq!(changed[i].output, full[i].output);
    }
    let mut zero = w.clone();
    zero.q_norm.fill(-1.); // Q norm scale is zero, so attention is uniform.
    let a = zero.sequence(&xs[..1], Backend::Reference).unwrap();
    assert!(a[0].query.iter().all(|&v| v == 0.));
    assert_eq!(cache.keys.len(), 5);
}
#[test]
fn deltanet_states_match_every_official_step_and_do_not_grow() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/hybrid.json")).unwrap();
    let m = model();
    let mut state = m.new_state();
    let sizes: Vec<_> = state
        .layers
        .iter()
        .filter_map(|s| {
            if let qwen_rust_inference::state::LayerState::Delta(s) = s {
                Some((s.convolution.len(), s.recurrent.len()))
            } else {
                None
            }
        })
        .collect();
    for (p, id) in fixture["ids"].as_array().unwrap().iter().enumerate() {
        m.step(
            id.as_u64().unwrap() as u32,
            &mut state,
            Backend::Reference,
            None,
        )
        .unwrap();
        for (l, s) in state.layers.iter().enumerate() {
            if let qwen_rust_inference::state::LayerState::Delta(s) = s {
                let expected = &fixture["states"][p][l.to_string()];
                for (actual, name) in [(&s.convolution, "conv"), (&s.recurrent, "recurrent")] {
                    let v: Vec<_> = expected[name]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_f64().unwrap() as f32)
                        .collect();
                    assert_eq!(actual.len(), v.len());
                    close(actual, &v);
                }
                assert_eq!((s.convolution.len(), s.recurrent.len()), sizes[l]);
            }
        }
    }
    for _ in 0..20 {
        m.step(1, &mut state, Backend::Reference, None).unwrap();
    }
    for (l, s) in state.layers.iter().take(3).enumerate() {
        if let qwen_rust_inference::state::LayerState::Delta(s) = s {
            assert_eq!((s.convolution.len(), s.recurrent.len()), sizes[l]);
        }
    }
}
#[test]
fn delta_rule_gate_boundaries_and_hand_calculation() {
    let mut s = vec![0.; 4];
    assert_eq!(
        delta_rule(&[1., 0.], &[1., 0.], &[2., 3.], 0., 1., &mut s).unwrap(),
        [2., 3.]
    );
    assert_eq!(s, [2., 3., 0., 0.]);
    assert_eq!(
        delta_rule(&[1., 0.], &[1., 0.], &[8., 9.], 0., 0., &mut s).unwrap(),
        [2., 3.]
    );
    let out = delta_rule(&[1., 0.], &[1., 0.], &[8., 9.], -1000., 0., &mut s).unwrap();
    assert_eq!(out, [0., 0.]);
    assert!(delta_rule(&[1.], &[1.], &[1.], 1., 0.5, &mut [0.]).is_err());
    assert!(delta_rule(&[1.], &[1.], &[1.], 0., 1.1, &mut [0.]).is_err());
}
#[test]
fn backend_matches_scalar_on_non_multiple_dimensions() {
    use qwen_rust_inference::tensor::Matrix;
    let w = Matrix::new(7, 13, (0..91).map(|i| (i as f32 - 40.) / 91.).collect()).unwrap();
    let x: Vec<_> = (0..13).map(|i| i as f32 / 13.).collect();
    close(
        &w.project(&x, Backend::Optimized).unwrap(),
        &w.linear(&x).unwrap(),
    );
}
#[test]
fn attention_and_rotary_hand_examples() {
    use qwen_rust_inference::attention::{attend, rotate, AttentionConfig};
    let mut c = AttentionConfig {
        heads: 1,
        kv_heads: 1,
        head_dim: 1,
        rotary_dim: 0,
        theta: 10000.,
        eps: 1e-6,
    };
    // exp(0):exp(ln 3)=1:3, so the weighted read is .25*2+.75*6=5.
    close(
        &attend(
            &[1.],
            &[vec![0.], vec![3_f32.ln()]],
            &[vec![2.], vec![6.]],
            &c,
        )
        .unwrap(),
        &[5.],
    );
    c.heads = 4;
    c.kv_heads = 2;
    close(
        &attend(
            &[1.; 4],
            &[vec![0., 0.], vec![0., 0.]],
            &[vec![2., 10.], vec![4., 14.]],
            &c,
        )
        .unwrap(),
        &[3., 3., 12., 12.],
    );
    let mut x = vec![1., 2., 7., 8.];
    rotate(&mut x, 1, 2, 10000.).unwrap();
    close(
        &x,
        &[
            1_f32.cos() - 2. * 1_f32.sin(),
            2. * 1_f32.cos() + 1_f32.sin(),
            7.,
            8.,
        ],
    );
    let mut s = vec![0.; 6];
    assert_eq!(
        delta_rule(&[1., 0.], &[1., 0.], &[2., 3., 4.], 0., 1., &mut s).unwrap(),
        [2., 3., 4.]
    );
    assert_eq!(s, [2., 3., 4., 0., 0., 0.]);
}
