//! M1 of book 3: the top layer (last FFN + head) from cached inputs.
//!
//! 1. The cached forward reproduces `Model::step`, so training differentiates
//!    the same function that inference computes.
//! 2. The gradients of the whole top layer agree with numerical
//!    differentiation of an f64 copy of it, for every parameter.
//! 3. `Model::with_ffn` and the cache file keep models apart by identity.
#![cfg(feature = "train")]
use qwen_rust_inference::tensor::Backend;
use qwen_rust_inference::train::cache::LayerCache;
use qwen_rust_inference::train::ffn::{TopFfn, TopInputs, TrainFfn};
use qwen_rust_inference::train::grad::cross_entropy;
mod train_common;
use train_common::{assert_gradient, fixture, numerical, to64, Top64};

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0., f32::max)
}

#[test]
fn cached_top_layer_reproduces_model_step() {
    let (model, ids) = fixture();
    let last = model.blocks().len() - 1;
    for backend in [Backend::Reference, Backend::Vector, Backend::Optimized] {
        let cache = LayerCache::capture(&model, &ids, backend).unwrap();
        assert_eq!(cache.header.layer, last);
        assert_eq!(cache.inputs.n, ids.len());
        let top = TopFfn::new(&model).unwrap();
        let f = top.forward(&cache.inputs).unwrap();

        let mut state = model.new_state();
        let vocab = model.config().vocab_size;
        let d = model.config().hidden_size;
        for (p, &id) in ids.iter().enumerate() {
            let step = model.step(id, &mut state, backend, None).unwrap();
            let b = &step.blocks[last];
            assert_eq!(
                &cache.inputs.post_norm[p * d..(p + 1) * d],
                b.post_norm.as_slice()
            );
            let ffn = &f.ffn.output[p * d..(p + 1) * d];
            assert!(max_abs(ffn, &b.ffn) < 1e-5, "{backend:?} FFN at {p}");
            let h = &f.h[p * d..(p + 1) * d];
            assert!(max_abs(h, &b.output) < 1e-5, "{backend:?} h at {p}");
            let n = &f.head.normalized[p * d..(p + 1) * d];
            assert!(
                max_abs(n, &step.normalized) < 1e-5,
                "{backend:?} norm at {p}"
            );
            let logits = &f.head.logits[p * vocab..(p + 1) * vocab];
            assert!(
                max_abs(logits, &step.logits) < 1e-5,
                "{backend:?} logits at {p}"
            );
        }
    }
}

/// Next-token targets: position i predicts ids[i + 1]; the last has none.
fn next_token_targets(ids: &[u32]) -> Vec<Option<u32>> {
    (0..ids.len()).map(|i| ids.get(i + 1).copied()).collect()
}

#[test]
fn top_layer_gradients_match_numerical() {
    let (model, ids) = fixture();
    let cache = LayerCache::capture(&model, &ids, Backend::Reference).unwrap();
    let top = TopFfn::new(&model).unwrap();
    let targets = next_token_targets(&ids);

    let (loss, grad) = top
        .loss_and_grad(&cache.inputs, |i, logits| match targets[i] {
            Some(t) => cross_entropy(logits, t),
            None => Ok(qwen_rust_inference::train::grad::LossGrad {
                loss: 0.,
                dlogits: vec![0.; logits.len()],
            }),
        })
        .unwrap();

    let d = model.config().hidden_size;
    let inputs: Vec<(Vec<f64>, Vec<f64>)> = (0..ids.len())
        .map(|p| {
            (
                to64(&cache.inputs.after_mixer[p * d..(p + 1) * d]),
                to64(&cache.inputs.post_norm[p * d..(p + 1) * d]),
            )
        })
        .collect();
    let mut m = Top64::new(&model, &top.ffn);
    let expect = m.loss(&inputs, &targets);
    assert!(
        (loss - expect).abs() < 1e-4 * expect.abs(),
        "{loss} vs {expect}"
    );

    let mut gate = m.gate.clone();
    let num = numerical(&mut gate, &mut |w| {
        m.gate = w.to_vec();
        m.loss(&inputs, &targets)
    });
    m.gate = gate;
    assert_gradient(grad.tensors[0].values(), &num, "gate");

    let mut up = m.up.clone();
    let num = numerical(&mut up, &mut |w| {
        m.up = w.to_vec();
        m.loss(&inputs, &targets)
    });
    m.up = up;
    assert_gradient(grad.tensors[1].values(), &num, "up");

    let mut down = m.down.clone();
    let num = numerical(&mut down, &mut |w| {
        m.down = w.to_vec();
        m.loss(&inputs, &targets)
    });
    m.down = down;
    assert_gradient(grad.tensors[2].values(), &num, "down");
}

/// The gradient toward the FFN input (needed later, to go below the top
/// layer), with `after_mixer` held fixed.
#[test]
fn ffn_input_gradient_matches_numerical() {
    let (model, ids) = fixture();
    let cache = LayerCache::capture(&model, &ids, Backend::Reference).unwrap();
    let top = TopFfn::new(&model).unwrap();
    let target = 7;
    let d = model.config().hidden_size;
    let one = TopInputs {
        after_mixer: cache.inputs.after_mixer[..d].to_vec(),
        post_norm: cache.inputs.post_norm[..d].to_vec(),
        n: 1,
    };
    let f = top.forward(&one).unwrap();
    let g = cross_entropy(&f.head.logits, target).unwrap();
    let dh = top.head.backward(&f.h, &g.dlogits, 1).unwrap();
    let mut grad = top.ffn.zero_grad().unwrap();
    let dx = top.ffn.backward(&f.ffn, &dh, &mut grad).unwrap();

    let m = Top64::new(&model, &top.ffn);
    let after = to64(&one.after_mixer);
    let mut x = to64(&one.post_norm);
    let num = numerical(&mut x, &mut |x| {
        m.loss(&[(after.clone(), x.to_vec())], &[Some(target)])
    });
    assert_gradient(&dx, &num, "dx");

    // The head's own gradient toward h, checked the same way.
    let mut h = to64(&f.h);
    let zero = vec![0.; d];
    let num_h = numerical(&mut h, &mut |h| {
        // FFN input of zero contributes silu(0)·0 = 0, so h = after_mixer.
        m.loss(&[(h.to_vec(), zero.clone())], &[Some(target)])
    });
    assert_gradient(&dh, &num_h, "dh");
}

#[test]
fn with_ffn_changes_identity_not_inference() {
    let (model, ids) = fixture();
    let last = model.blocks().len() - 1;
    let copy = TrainFfn::from_weights(&model.blocks()[last].ffn);
    let same = model.with_ffn(last, copy.to_weights().unwrap()).unwrap();
    assert_ne!(same.identity(), model.identity());
    // Deterministic: the same FFN gives the same identity.
    let again = model.with_ffn(last, copy.to_weights().unwrap()).unwrap();
    assert_eq!(same.identity(), again.identity());

    let (mut s1, mut s2) = (model.new_state(), same.new_state());
    for &id in &ids {
        let a = model.step(id, &mut s1, Backend::Vector, None).unwrap();
        let b = same.step(id, &mut s2, Backend::Vector, None).unwrap();
        assert!(a
            .logits
            .iter()
            .zip(&b.logits)
            .all(|(x, y)| x.to_bits() == y.to_bits()));
    }

    let mut changed = copy.clone();
    changed.down.values_mut()[0] += 0.25;
    let other = model.with_ffn(last, changed.to_weights().unwrap()).unwrap();
    assert_ne!(other.identity(), same.identity());
    let (mut s1, mut s2) = (model.new_state(), other.new_state());
    let a = model.step(ids[0], &mut s1, Backend::Vector, None).unwrap();
    let b = other.step(ids[0], &mut s2, Backend::Vector, None).unwrap();
    assert!(max_abs(&a.logits, &b.logits) > 0.);
    assert!(model
        .with_ffn(last + 1, copy.to_weights().unwrap())
        .is_err());
}

#[test]
fn cache_file_round_trip_and_refusals() {
    let (model, ids) = fixture();
    let dir = std::env::temp_dir().join(format!("qwen-train-cache-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("top.cache");
    let cache = LayerCache::capture(&model, &ids, Backend::Vector).unwrap();
    cache.write(&path).unwrap();
    assert_eq!(LayerCache::read(&path, &model).unwrap(), cache);

    // Another model (here: one changed FFN) must not read it.
    let last = model.blocks().len() - 1;
    let other = model
        .with_ffn(
            last,
            TrainFfn::from_weights(&model.blocks()[last].ffn)
                .to_weights()
                .unwrap(),
        )
        .unwrap();
    let e = LayerCache::read(&path, &other).unwrap_err();
    assert!(e.0.contains("different model"), "{e}");

    // A flipped bit in the data is caught by the checksum.
    let mut bytes = std::fs::read(&path).unwrap();
    *bytes.last_mut().unwrap() ^= 1;
    std::fs::write(&path, &bytes).unwrap();
    let e = LayerCache::read(&path, &model).unwrap_err();
    assert!(e.0.contains("checksum"), "{e}");

    std::fs::write(&path, b"not a cache").unwrap();
    assert!(LayerCache::read(&path, &model).is_err());
    std::fs::remove_dir_all(&dir).unwrap();
}
