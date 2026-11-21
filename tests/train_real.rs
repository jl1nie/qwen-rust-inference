//! Book 3 on the real weights (opt-in: `QWEN_MODEL_DIR`, release build).
//!
//! The cached top layer must reproduce the inference logits within the real-
//! model tolerance (3e-3, as `tests/real_model.rs`), and one small step
//! against its gradient must lower the loss.
#![cfg(feature = "train")]
use qwen_rust_inference::model::Model;
use qwen_rust_inference::tensor::Backend;
use qwen_rust_inference::text::{user_chat, TextTokenizer};
use qwen_rust_inference::train::cache::LayerCache;
use qwen_rust_inference::train::ffn::TopFfn;
use qwen_rust_inference::train::grad::cross_entropy;
use qwen_rust_inference::validation::compare;
use qwen_rust_inference::weights::load_model;
use std::path::PathBuf;
use std::time::Instant;

fn load() -> (Model, TextTokenizer) {
    let dir =
        PathBuf::from(std::env::var("QWEN_MODEL_DIR").unwrap_or_else(|_| "models/qwen".into()));
    (
        load_model(&dir).unwrap(),
        TextTokenizer::load(&dir).unwrap(),
    )
}

#[test]
#[ignore = "opt-in: pinned real weights; run with --release"]
fn train_real_top_layer_matches_inference_and_descends() {
    let (model, tokenizer) = load();
    let ids = tokenizer
        .encode(&user_chat("日本の首都はどこですか？一言で。", false))
        .unwrap();
    let backend = Backend::Vector;

    let t = Instant::now();
    let cache = LayerCache::capture(&model, &ids, backend).unwrap();
    eprintln!("capture: {} positions in {:?}", ids.len(), t.elapsed());

    let top = TopFfn::new(&model).unwrap();
    let t = Instant::now();
    let f = top.forward(&cache.inputs).unwrap();
    eprintln!("top-layer forward: {:?}", t.elapsed());
    let vocab = model.config().vocab_size;
    let mut state = model.new_state();
    for (p, &id) in ids.iter().enumerate() {
        let step = model.step(id, &mut state, backend, None).unwrap();
        compare(
            &f.head.logits[p * vocab..(p + 1) * vocab],
            &step.logits,
            3e-3,
            3e-3,
            &format!("cached logits at {p}"),
        )
        .unwrap();
    }

    // Teach the last position one target: the ID of "東京".
    let target = *tokenizer.encode("東京").unwrap().first().unwrap();
    let last = cache.inputs.n - 1;
    let score = |top: &TopFfn| {
        top.loss_and_grad(&cache.inputs, |i, logits| {
            if i == last {
                cross_entropy(logits, target)
            } else {
                Ok(qwen_rust_inference::train::grad::LossGrad {
                    loss: 0.,
                    dlogits: vec![0.; logits.len()],
                })
            }
        })
        .unwrap()
    };
    let t = Instant::now();
    let (loss, grad) = score(&top);
    eprintln!("loss and gradient: {:?}, loss {loss:.4}", t.elapsed());
    for g in [&grad.gate, &grad.up, &grad.down] {
        assert!(g.values().iter().all(|v| v.is_finite()));
    }

    // A step of size eps along −grad lowers the loss by about eps·‖grad‖².
    let norm2: f64 = [&grad.gate, &grad.up, &grad.down]
        .iter()
        .flat_map(|g| g.values())
        .map(|&v| (v as f64).powi(2))
        .sum();
    assert!(norm2 > 0.);
    let eps = 1e-3 / norm2.sqrt();
    let mut stepped = top.clone();
    for (w, g) in [
        (&mut stepped.ffn.gate, &grad.gate),
        (&mut stepped.ffn.up, &grad.up),
        (&mut stepped.ffn.down, &grad.down),
    ] {
        for (w, g) in w.values_mut().iter_mut().zip(g.values()) {
            *w -= (eps * *g as f64) as f32;
        }
    }
    let (after, _) = score(&stepped);
    let predicted = eps * norm2;
    eprintln!("loss {loss:.6} -> {after:.6}, predicted decrease {predicted:.3e}");
    assert!(after < loss);
    // First-order prediction holds within a factor of two at this step size.
    let actual = loss - after;
    assert!(actual > 0.5 * predicted && actual < 2. * predicted);
}
