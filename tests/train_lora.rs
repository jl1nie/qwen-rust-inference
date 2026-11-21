//! M3 of book 3: LoRA adapters on the last layer's FFN.
//!
//! 1. With B = 0 (the initial state) the model is exactly the original one.
//! 2. The gradients of A and B agree with numerical differentiation of the
//!    f64 top layer whose matrices are `W + s·B·A`.
//! 3. Folding the adapters into the matrices computes the same function.
//! 4. A fact can be taught with about 1% of the FFN's parameters, and the
//!    checkpoint holds only the adapters.
#![cfg(feature = "train")]
use qwen_rust_inference::tensor::Backend;
use qwen_rust_inference::train::cache::LayerCache;
use qwen_rust_inference::train::checkpoint::FfnCheckpoint;
use qwen_rust_inference::train::ffn::{TopFfn, TrainFfn};
use qwen_rust_inference::train::fit::{fit, Example, Target};
use qwen_rust_inference::train::grad::{cross_entropy, LossGrad};
use qwen_rust_inference::train::linalg::Dense;
use qwen_rust_inference::train::metrics::argmax;
use qwen_rust_inference::train::optim::{Optimizer, OptimizerConfig};
mod train_common;
use train_common::{assert_gradient, fixture, numerical, to64, Top64};

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0., f32::max)
}

/// Deterministic values in (-1, 1).
fn values(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f64 / (1u64 << 53) as f64 * 2. - 1.) as f32
        })
        .collect()
}

#[test]
fn zero_b_is_the_original_model() {
    let (model, ids) = fixture();
    let cache = LayerCache::capture(&model, &ids, Backend::Vector).unwrap();
    let plain = TopFfn::new(&model).unwrap();
    let mut lora = TopFfn::new(&model).unwrap();
    lora.ffn = lora.ffn.with_lora(2, 4., &[0, 1, 2], 7).unwrap();
    let (a, b) = (
        plain.forward(&cache.inputs).unwrap(),
        lora.forward(&cache.inputs).unwrap(),
    );
    assert!(a
        .head
        .logits
        .iter()
        .zip(&b.head.logits)
        .all(|(x, y)| x.to_bits() == y.to_bits()));
    // Only the adapters are trained: 2·(8 + 16) per matrix, 3 matrices.
    assert_eq!(
        lora.ffn.sizes(),
        [2 * 8, 16 * 2, 2 * 8, 16 * 2, 2 * 16, 8 * 2]
    );
    assert_eq!(lora.ffn.trained_parameters(), 144);
    assert_eq!(plain.ffn.trained_parameters(), 3 * 128);
}

/// `W + s·B·A` in f64.
fn merge64(w: &[f64], a: &[f64], b: &[f64], s: f64, rank: usize) -> Vec<f64> {
    let input = a.len() / rank;
    let out = b.len() / rank;
    (0..out * input)
        .map(|i| {
            let (o, j) = (i / input, i % input);
            w[i] + s
                * (0..rank)
                    .map(|k| b[o * rank + k] * a[k * input + j])
                    .sum::<f64>()
        })
        .collect()
}

#[test]
fn lora_gradients_match_numerical() {
    let (model, ids) = fixture();
    let cache = LayerCache::capture(&model, &ids, Backend::Reference).unwrap();
    let mut top = TopFfn::new(&model).unwrap();
    let rank = 2;
    top.ffn = top.ffn.with_lora(rank, 3., &[0, 1, 2], 11).unwrap();
    // B = 0 would make A's gradient zero; give B values so both are tested.
    for (i, l) in top
        .ffn
        .lora
        .as_mut()
        .unwrap()
        .adapters
        .iter_mut()
        .enumerate()
    {
        let b = &mut l.as_mut().unwrap().b;
        let v: Vec<f32> = values(20 + i as u64, b.values().len())
            .iter()
            .map(|v| v * 0.3)
            .collect();
        *b = Dense::new(b.rows(), b.cols(), v).unwrap();
    }
    let targets: Vec<Option<u32>> = (0..ids.len()).map(|i| ids.get(i + 1).copied()).collect();
    let (_, grad) = top
        .loss_and_grad(&cache.inputs, |i, logits| match targets[i] {
            Some(t) => cross_entropy(logits, t),
            None => Ok(LossGrad {
                loss: 0.,
                dlogits: vec![0.; logits.len()],
            }),
        })
        .unwrap();
    assert_eq!(grad.tensors.len(), 6);

    let d = model.config().hidden_size;
    let inputs: Vec<(Vec<f64>, Vec<f64>)> = (0..ids.len())
        .map(|p| {
            (
                to64(&cache.inputs.after_mixer[p * d..(p + 1) * d]),
                to64(&cache.inputs.post_norm[p * d..(p + 1) * d]),
            )
        })
        .collect();
    let base = TrainFfn::from_weights(&model.blocks()[top.layer].ffn);
    let bases = [
        to64(base.gate.values()),
        to64(base.up.values()),
        to64(base.down.values()),
    ];
    let set = top.ffn.lora.as_ref().unwrap();
    let ab: Vec<(Vec<f64>, Vec<f64>, f64)> = set
        .adapters
        .iter()
        .map(|l| {
            let l = l.as_ref().unwrap();
            (to64(l.a.values()), to64(l.b.values()), l.scale as f64)
        })
        .collect();
    let mut m = Top64::new(&model, &base);
    // Loss with adapter `proj`'s A (which = 0) or B (which = 1) replaced.
    let mut loss_with = |proj: usize, which: usize, v: &[f64]| {
        let mut merged = Vec::new();
        for (p, (a, b, s)) in ab.iter().enumerate() {
            let (a, b) = match (p == proj, which) {
                (true, 0) => (v, b.as_slice()),
                (true, _) => (a.as_slice(), v),
                _ => (a.as_slice(), b.as_slice()),
            };
            merged.push(merge64(&bases[p], a, b, *s, rank));
        }
        m.gate = merged[0].clone();
        m.up = merged[1].clone();
        m.down = merged[2].clone();
        m.loss(&inputs, &targets)
    };
    for (proj, (a, b, _)) in ab.iter().enumerate() {
        for (which, start) in [a, b].into_iter().enumerate() {
            let mut x = start.clone();
            let num = numerical(&mut x, &mut |v| loss_with(proj, which, v));
            let name = format!("{} {}", ["gate", "up", "down"][proj], ["A", "B"][which]);
            assert_gradient(grad.tensors[2 * proj + which].values(), &num, &name);
        }
    }
}

#[test]
fn merged_adapters_compute_the_same_function() {
    let (model, ids) = fixture();
    let cache = LayerCache::capture(&model, &ids, Backend::Vector).unwrap();
    let mut top = TopFfn::new(&model).unwrap();
    top.ffn = top.ffn.with_lora(3, 6., &[0, 2], 5).unwrap();
    for l in top.ffn.lora.as_mut().unwrap().adapters.iter_mut().flatten() {
        let v: Vec<f32> = values(9, l.b.values().len())
            .iter()
            .map(|v| v * 0.5)
            .collect();
        l.b = Dense::new(l.b.rows(), l.b.cols(), v).unwrap();
    }
    let with_adapters = top.forward(&cache.inputs).unwrap();
    // Folded into the matrices, as a plain trainable FFN...
    let [gate, up, down] = top.ffn.merged().unwrap();
    let mut folded = TopFfn::new(&model).unwrap();
    folded.ffn = TrainFfn {
        gate,
        up,
        down,
        lora: None,
    };
    let f = folded.forward(&cache.inputs).unwrap();
    assert!(max_abs(&f.head.logits, &with_adapters.head.logits) < 1e-5);
    // ...and as inference weights in the model.
    let changed = model
        .with_ffn(top.layer, top.ffn.to_weights().unwrap())
        .unwrap();
    let mut state = changed.new_state();
    let vocab = model.config().vocab_size;
    for (p, &id) in ids.iter().enumerate() {
        let step = changed.step(id, &mut state, Backend::Vector, None).unwrap();
        let cached = &with_adapters.head.logits[p * vocab..(p + 1) * vocab];
        assert!(max_abs(&step.logits, cached) < 1e-5, "position {p}");
    }
    // The untouched matrix (up) is the base exactly.
    assert_eq!(
        folded.ffn.up,
        TrainFfn::from_weights(&model.blocks()[top.layer].ffn).up
    );
}

#[test]
fn lora_teaches_one_target_and_saves_only_the_adapters() {
    let (model, ids) = fixture();
    let cache = LayerCache::capture(&model, &ids, Backend::Vector).unwrap();
    let examples = vec![Example {
        position: 3,
        target: Target::Id(11),
    }];
    let logits = |top: &TopFfn| {
        top.forward(&cache.inputs.subset(&[3]).unwrap())
            .unwrap()
            .head
            .logits
    };
    let mut top = TopFfn::new(&model).unwrap();
    top.ffn = top.ffn.with_lora(2, 4., &[0, 1, 2], 3).unwrap();
    assert_ne!(argmax(&logits(&top)), 11);
    let config = OptimizerConfig::adam(0.05);
    let mut o = Optimizer::new(config, &top.ffn.sizes());
    let mut losses = Vec::new();
    fit(&mut top, &cache.inputs, &examples, &mut o, 80, 4, |s| {
        losses.push(s.loss);
        Ok(())
    })
    .unwrap();
    assert!(losses[..10].windows(2).all(|w| w[1] < w[0]), "{losses:?}");
    assert_eq!(argmax(&logits(&top)), 11, "{losses:?}");
    // The base matrices did not move.
    let base = TrainFfn::from_weights(&model.blocks()[top.layer].ffn);
    assert_eq!(
        (&top.ffn.gate, &top.ffn.up, &top.ffn.down),
        (&base.gate, &base.up, &base.down)
    );

    let dir = std::env::temp_dir().join(format!("qwen-train-lora-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("lora.safetensors");
    let saved = FfnCheckpoint {
        base_identity: model.identity().into(),
        layer: top.layer,
        ffn: top.ffn.clone(),
        optimizer: Some(o.clone()),
    };
    saved.write(&path).unwrap();
    let loaded = FfnCheckpoint::read(&path, &model).unwrap();
    assert_eq!(loaded, saved);
    // Resuming continues bit-identically: 5 more steps from the file and
    // from memory.
    let resume = |ffn: TrainFfn, mut o: Optimizer| {
        let mut t = TopFfn::new(&model).unwrap();
        t.ffn = ffn;
        fit(&mut t, &cache.inputs, &examples, &mut o, 5, 4, |_| Ok(())).unwrap();
        t.ffn
    };
    assert_eq!(
        resume(loaded.ffn.clone(), loaded.optimizer.clone().unwrap()),
        resume(top.ffn.clone(), o)
    );
    // The model with the folded adapters predicts the target too.
    let trained = loaded.apply(&model).unwrap();
    let mut state = trained.new_state();
    let mut last = Vec::new();
    for &id in &ids[..4] {
        last = trained
            .step(id, &mut state, Backend::Vector, None)
            .unwrap()
            .logits;
    }
    assert_eq!(argmax(&last), 11);
    // Another model is refused.
    assert!(FfnCheckpoint::read(&path, &trained).is_err());
    std::fs::remove_dir_all(&dir).unwrap();
}
