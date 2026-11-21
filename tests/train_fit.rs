//! M2 of book 3: optimizers, the training loop, metrics and checkpoints.
#![cfg(feature = "train")]
use qwen_rust_inference::model::Model;
use qwen_rust_inference::tensor::{Backend, InferenceError, Result};
use qwen_rust_inference::text::{user_chat, TextCodec};
use qwen_rust_inference::train::cache::LayerCache;
use qwen_rust_inference::train::checkpoint::FfnCheckpoint;
use qwen_rust_inference::train::ffn::TopFfn;
use qwen_rust_inference::train::fit::{
    compare_models, fit, mean_loss_and_grad, Example, Fact, Target,
};
use qwen_rust_inference::train::metrics::{argmax, kl, rank, Comparer};
use qwen_rust_inference::train::optim::{Optimizer, OptimizerConfig};

fn fixture() -> (Model, Vec<u32>) {
    let v: serde_json::Value = serde_json::from_str(include_str!("fixtures/hybrid.json")).unwrap();
    let ids = v["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| i.as_u64().unwrap() as u32)
        .collect();
    (Model::from_fixture(&v).unwrap(), ids)
}

#[test]
fn sgd_and_adam_steps_match_hand_computation() {
    let mut p = vec![1.0_f32, -2.0];
    let g = [0.5_f32, -0.25];
    let mut sgd = Optimizer::new(OptimizerConfig::Sgd { lr: 0.1 }, &[2]);
    sgd.step(&mut [&mut p], &[&g]).unwrap();
    assert_eq!(p, [1.0 - 0.05, -2.0 + 0.025]);

    // Adam's first step moves every parameter by lr·sign(g) (up to ε):
    // m̂ = g, v̂ = g², so m̂/√v̂ = ±1.
    let config = OptimizerConfig::adam(0.01);
    let mut adam = Optimizer::new(config, &[2]);
    let mut p = vec![1.0_f32, -2.0];
    adam.step(&mut [&mut p], &[&g]).unwrap();
    assert!(
        (p[0] - 0.99).abs() < 1e-6 && (p[1] + 1.99).abs() < 1e-6,
        "{p:?}"
    );
    // Second step with the gradient halved, from the definitions in f64.
    let g2 = [0.25_f32, -0.125];
    adam.step(&mut [&mut p], &[&g2]).unwrap();
    let (b1, b2, eps, lr) = (0.9_f64, 0.999_f64, 1e-8_f64, 0.01_f64);
    for i in 0..2 {
        let (g1, g2) = (g[i] as f64, g2[i] as f64);
        let m = b1 * (1. - b1) * g1 + (1. - b1) * g2;
        let v = b2 * (1. - b2) * g1 * g1 + (1. - b2) * g2 * g2;
        let (mh, vh) = (m / (1. - b1 * b1), v / (1. - b2 * b2));
        let first = [0.99, -1.99][i];
        let expect = first - lr * mh / (vh.sqrt() + eps);
        assert!(
            (p[i] as f64 - expect).abs() < 1e-6,
            "{i}: {} vs {expect}",
            p[i]
        );
    }
    assert_eq!(adam.t, 2);
    assert!(adam.step(&mut [&mut p], &[&[f32::NAN, 0.]]).is_err());
    assert!(adam.step(&mut [&mut p], &[&[0.]]).is_err());
}

#[test]
fn metrics_have_their_analytic_values() {
    // Uniform over 19 IDs: perplexity 19 whatever the next ID.
    let uniform = vec![0.; 19];
    let mut c = Comparer::default();
    c.add(&uniform, &uniform, Some(3)).unwrap();
    c.add(&uniform, &uniform, None).unwrap();
    let r = c.finish().unwrap();
    assert_eq!((r.positions, r.scored), (2, 1));
    assert!((r.perplexity_reference - 19.).abs() < 1e-9);
    assert!(r.mean_kl.abs() < 1e-12 && r.top1_agreement == 1.);

    // q = (2, 1, 1)/4 against uniform p = 1/3: KL(p‖q) = ln(4/3) − (1/3)ln 2.
    let p = [0., 0., 0.];
    let q = [2_f32.ln(), 0., 0.];
    let expect = (4. / 3_f64).ln() - 2_f64.ln() / 3.;
    assert!((kl(&p, &q).unwrap() - expect).abs() < 1e-7);
    assert_eq!(argmax(&q), 0);
    assert_eq!(rank(&q, 0), 0);
    assert_eq!(rank(&q, 2), 1);
}

/// Position 3 of the fixture should predict ID 11 (it never does at first).
fn fact_examples() -> Vec<Example> {
    vec![Example {
        position: 3,
        target: Target::Id(11),
    }]
}

#[test]
fn training_teaches_one_target() {
    let (model, ids) = fixture();
    let cache = LayerCache::capture(&model, &ids, Backend::Vector).unwrap();
    let mut top = TopFfn::new(&model).unwrap();
    let examples = fact_examples();
    let logits = |top: &TopFfn| {
        top.forward(&cache.inputs.subset(&[3]).unwrap())
            .unwrap()
            .head
            .logits
    };
    assert_ne!(argmax(&logits(&top)), 11);

    let mut optimizer = Optimizer::new(OptimizerConfig::adam(0.02), &top.ffn.sizes());
    let mut losses = Vec::new();
    fit(
        &mut top,
        &cache.inputs,
        &examples,
        &mut optimizer,
        60,
        4,
        |s| {
            losses.push(s.loss);
            Ok(())
        },
    )
    .unwrap();
    // Falls steadily at first. Later it levels off near 1.905 (q ≈ 0.15):
    // the final RMSNorm fixes the length of h, so in this 8-wide fixture no
    // FFN output can make ID 11 much more likely. Adam may then move the loss
    // up by ~1e-5, so only the first steps are required to fall.
    assert!(losses[..10].windows(2).all(|w| w[1] < w[0]), "{losses:?}");
    assert!(*losses.last().unwrap() < losses[0] - 1., "{losses:?}");
    assert_eq!(argmax(&logits(&top)), 11, "{losses:?}");
}

#[test]
fn gradient_does_not_depend_on_minibatch_size() {
    let (model, ids) = fixture();
    let cache = LayerCache::capture(&model, &ids, Backend::Vector).unwrap();
    let top = TopFfn::new(&model).unwrap();
    let examples: Vec<Example> = (0..ids.len() - 1)
        .map(|p| Example {
            position: p,
            target: Target::Id(ids[p + 1]),
        })
        .collect();
    let (l1, g1) = mean_loss_and_grad(&top, &cache.inputs, &examples, 1).unwrap();
    let (l4, g4) = mean_loss_and_grad(&top, &cache.inputs, &examples, 4).unwrap();
    assert!((l1 - l4).abs() < 1e-9);
    for (a, b) in g1.slices().iter().zip(g4.slices()) {
        assert!(a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-6));
    }
}

fn temp(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("qwen-train-fit-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

#[test]
fn resumed_training_is_bit_identical() {
    let (model, ids) = fixture();
    let cache = LayerCache::capture(&model, &ids, Backend::Vector).unwrap();
    let examples = fact_examples();
    let config = OptimizerConfig::adam(0.02);

    let mut straight = TopFfn::new(&model).unwrap();
    let mut o = Optimizer::new(config, &straight.ffn.sizes());
    fit(
        &mut straight,
        &cache.inputs,
        &examples,
        &mut o,
        6,
        4,
        |_| Ok(()),
    )
    .unwrap();

    let mut first = TopFfn::new(&model).unwrap();
    let mut o = Optimizer::new(config, &first.ffn.sizes());
    fit(&mut first, &cache.inputs, &examples, &mut o, 3, 4, |_| {
        Ok(())
    })
    .unwrap();
    let path = temp("resume.safetensors");
    let saved = FfnCheckpoint {
        base_identity: model.identity().into(),
        layer: first.layer,
        ffn: first.ffn.clone(),
        optimizer: Some(o),
    };
    saved.write(&path).unwrap();
    let loaded = FfnCheckpoint::read(&path, &model).unwrap();
    assert_eq!(loaded, saved);

    let mut second = TopFfn::new(&model).unwrap();
    second.ffn = loaded.ffn;
    let mut o = loaded.optimizer.unwrap();
    fit(&mut second, &cache.inputs, &examples, &mut o, 3, 4, |_| {
        Ok(())
    })
    .unwrap();
    assert_eq!(o.t, 6);
    for (a, b) in [
        (&second.ffn.gate, &straight.ffn.gate),
        (&second.ffn.up, &straight.ffn.up),
        (&second.ffn.down, &straight.ffn.down),
    ] {
        assert!(a
            .values()
            .iter()
            .zip(b.values())
            .all(|(x, y)| x.to_bits() == y.to_bits()));
    }

    // The checkpoint changes the model it was trained from, and no other.
    let trained = loaded_apply(&path, &model);
    assert_ne!(trained.identity(), model.identity());
    let e = FfnCheckpoint::read(&path, &trained).unwrap_err();
    assert!(e.0.contains("different model"), "{e}");
    std::fs::remove_file(&path).unwrap();
}
fn loaded_apply(path: &std::path::Path, model: &Model) -> Model {
    FfnCheckpoint::read(path, model)
        .unwrap()
        .apply(model)
        .unwrap()
}

#[test]
fn compare_models_of_the_same_ffn_and_of_a_trained_one() {
    let (model, ids) = fixture();
    let cache = LayerCache::capture(&model, &ids, Backend::Vector).unwrap();
    let reference = TopFfn::new(&model).unwrap();
    let same = compare_models(&reference, &reference, &cache.inputs, &ids, 2).unwrap();
    assert_eq!(same.positions, ids.len());
    assert_eq!(same.scored, ids.len() - 1);
    assert!(same.mean_kl.abs() < 1e-12 && same.top1_agreement == 1.);
    assert_eq!(same.perplexity_reference, same.perplexity_new);

    let mut trained = TopFfn::new(&model).unwrap();
    let mut o = Optimizer::new(OptimizerConfig::adam(0.02), &trained.ffn.sizes());
    fit(
        &mut trained,
        &cache.inputs,
        &fact_examples(),
        &mut o,
        20,
        4,
        |_| Ok(()),
    )
    .unwrap();
    let moved = compare_models(&reference, &trained, &cache.inputs, &ids, 2).unwrap();
    assert!(moved.mean_kl > 0. && moved.max_kl >= moved.mean_kl);
    // Comparing through the cache equals comparing the full models step by step.
    let new_model = model
        .with_ffn(trained.layer, trained.ffn.to_weights().unwrap())
        .unwrap();
    let mut c = Comparer::default();
    let (mut s1, mut s2) = (model.new_state(), new_model.new_state());
    for (p, &id) in ids.iter().enumerate() {
        let a = model.step(id, &mut s1, Backend::Vector, None).unwrap();
        let b = new_model.step(id, &mut s2, Backend::Vector, None).unwrap();
        c.add(&a.logits, &b.logits, ids.get(p + 1).copied())
            .unwrap();
    }
    let full = c.finish().unwrap();
    assert!((full.mean_kl - moved.mean_kl).abs() < 1e-5 * moved.mean_kl.max(1.));
    assert!((full.perplexity_new - moved.perplexity_new).abs() < 1e-4 * moved.perplexity_new);
    assert_eq!(full.top1_agreement, moved.top1_agreement);
}

/// One ID per character, for checking how a fact is cut into examples.
struct Chars;
impl TextCodec for Chars {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(text.chars().map(|c| c as u32 % 1000).collect())
    }
    fn decode(&self, _: &[u32]) -> Result<String> {
        Err(InferenceError("unused".into()))
    }
}

#[test]
fn fact_scores_the_answer_and_the_end_of_turn() {
    let fact = Fact::new(&Chars, "Q", "AB", 999).unwrap();
    let prompt_len = user_chat("Q", false).chars().count();
    assert_eq!(fact.prompt_len, prompt_len);
    assert_eq!(fact.ids.len(), prompt_len + 3);
    let e = fact.examples();
    let targets: Vec<Target> = e.iter().map(|e| e.target.clone()).collect();
    assert_eq!(
        targets,
        [
            Target::Id('A' as u32),
            Target::Id('B' as u32),
            Target::Id(999)
        ]
    );
    assert_eq!(e[0].position, prompt_len - 1);
    assert!(Fact::new(&Chars, "Q", "", 999).is_err());
}
