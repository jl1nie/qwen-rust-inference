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
    for g in &grad.tensors {
        assert!(g.values().iter().all(|v| v.is_finite()));
    }

    // A step of size eps along −grad lowers the loss by about eps·‖grad‖².
    let norm2: f64 = grad
        .tensors
        .iter()
        .flat_map(|g| g.values())
        .map(|&v| (v as f64).powi(2))
        .sum();
    assert!(norm2 > 0.);
    let eps = 1e-3 / norm2.sqrt();
    let mut stepped = top.clone();
    for (w, g) in stepped.ffn.params_mut().into_iter().zip(grad.slices()) {
        for (w, g) in w.iter_mut().zip(g) {
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

/// Teaches 「日本の首都は」 the counterfactual answer 京都です。 with the setting
/// that kept France's answer in `spec/measurements/cpu-train-fact-kyoto.json` (Adam, lr 3e-6,
/// 30 steps), then checks the answer, an unrelated question, and how far the
/// model moved on other text.
#[test]
#[ignore = "opt-in: pinned real weights; run with --release (about a minute)"]
fn train_real_teaches_one_fact_with_small_side_effects() {
    use qwen_rust_inference::generate::{generate, Sampling};
    use qwen_rust_inference::train::checkpoint::FfnCheckpoint;
    use qwen_rust_inference::train::fit::{compare_models, fit, Fact};
    use qwen_rust_inference::train::optim::{Optimizer, OptimizerConfig};
    let (model, tokenizer) = load();
    let fact = Fact::new(
        &tokenizer,
        "日本の首都はどこですか？一言で。",
        "京都です。",
        248046,
    )
    .unwrap();
    let cache = LayerCache::capture(&model, &fact.ids, Backend::Vector).unwrap();
    let mut top = TopFfn::new(&model).unwrap();
    let before = fact.report(&top, &cache.inputs).unwrap();
    assert!(before[0].rank > 0, "京都 is not the answer before training");
    let mut o = Optimizer::new(OptimizerConfig::adam(3e-6), &top.ffn.sizes());
    let t = Instant::now();
    fit(
        &mut top,
        &cache.inputs,
        &fact.examples(),
        &mut o,
        30,
        16,
        |_| Ok(()),
    )
    .unwrap();
    eprintln!("30 steps in {:?}", t.elapsed());
    let after = fact.report(&top, &cache.inputs).unwrap();
    assert!(
        after.iter().all(|r| r.rank == 0 && r.probability > 0.9),
        "{after:?}"
    );

    let trained = FfnCheckpoint {
        base_identity: model.identity().into(),
        layer: top.layer,
        ffn: top.ffn.clone(),
        optimizer: None,
    }
    .apply(&model)
    .unwrap();
    let eos = [248044, 248046];
    let answer = |m: &Model, q: &str| {
        let ids = tokenizer.encode(&user_chat(q, false)).unwrap();
        let r = generate(m, &ids, 6, &eos, Sampling::Greedy, None).unwrap();
        let g = &r.history.generated;
        tokenizer
            .decode(&g[..g.len() - usize::from(g.last().is_some_and(|i| eos.contains(i)))])
            .unwrap()
    };
    assert_eq!(
        answer(&trained, "日本の首都はどこですか？一言で。"),
        "京都です。"
    );
    assert_eq!(
        answer(&trained, "フランスの首都はどこですか？一言で。"),
        "パリです。"
    );

    // Other text: the first 128 tokens of README.md.
    let text = std::fs::read_to_string("README.md").unwrap();
    let mut ids = tokenizer.encode(&text).unwrap();
    ids.truncate(128);
    let other = LayerCache::capture(&model, &ids, Backend::Vector).unwrap();
    let c = compare_models(&TopFfn::new(&model).unwrap(), &top, &other.inputs, &ids, 16).unwrap();
    eprintln!("README (128 tokens): {c:?}");
    assert!(c.mean_kl < 0.05, "{c:?}");
    assert!(c.perplexity_new < 1.05 * c.perplexity_reference, "{c:?}");
}

/// The same fact with rank-8 LoRA on gate, up and down (Adam, lr 1e-4, 30
/// steps; `spec/measurements/cpu-train-lora-kyoto.json`): about 1% of the FFN's parameters
/// are trained, the base matrices stay as they were, and the checkpoint holds
/// only the adapters.
#[test]
#[ignore = "opt-in: pinned real weights; run with --release (about a minute)"]
fn train_real_lora_teaches_one_fact() {
    use qwen_rust_inference::generate::{generate, Sampling};
    use qwen_rust_inference::train::checkpoint::FfnCheckpoint;
    use qwen_rust_inference::train::ffn::TrainFfn;
    use qwen_rust_inference::train::fit::{fit, Fact};
    use qwen_rust_inference::train::optim::{Optimizer, OptimizerConfig};
    let (model, tokenizer) = load();
    let fact = Fact::new(
        &tokenizer,
        "日本の首都はどこですか？一言で。",
        "京都です。",
        248046,
    )
    .unwrap();
    let cache = LayerCache::capture(&model, &fact.ids, Backend::Vector).unwrap();
    let mut top = TopFfn::new(&model).unwrap();
    top.ffn = top.ffn.with_lora(8, 8., &[0, 1, 2], 0).unwrap();
    assert_eq!(top.ffn.trained_parameters(), 3 * 8 * (1024 + 3584));
    let mut o = Optimizer::new(OptimizerConfig::adam(1e-4), &top.ffn.sizes());
    let t = Instant::now();
    fit(
        &mut top,
        &cache.inputs,
        &fact.examples(),
        &mut o,
        30,
        16,
        |_| Ok(()),
    )
    .unwrap();
    eprintln!("LoRA: 30 steps in {:?}", t.elapsed());
    let after = fact.report(&top, &cache.inputs).unwrap();
    assert!(
        after.iter().all(|r| r.rank == 0 && r.probability > 0.8),
        "{after:?}"
    );
    let base = TrainFfn::from_weights(&model.blocks()[top.layer].ffn);
    assert!(top.ffn.gate == base.gate && top.ffn.up == base.up && top.ffn.down == base.down);

    let dir = std::env::temp_dir().join(format!("qwen-train-real-lora-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("lora.safetensors");
    let c = FfnCheckpoint {
        base_identity: model.identity().into(),
        layer: top.layer,
        ffn: top.ffn.clone(),
        optimizer: None,
    };
    c.write(&path).unwrap();
    let bytes = std::fs::metadata(&path).unwrap().len();
    assert!(bytes < 500_000, "{bytes} bytes");
    let trained = FfnCheckpoint::read(&path, &model)
        .unwrap()
        .apply(&model)
        .unwrap();
    std::fs::remove_dir_all(&dir).unwrap();

    let eos = [248044, 248046];
    let answer = |q: &str| {
        let ids = tokenizer.encode(&user_chat(q, false)).unwrap();
        let r = generate(&trained, &ids, 6, &eos, Sampling::Greedy, None).unwrap();
        let g = &r.history.generated;
        tokenizer
            .decode(&g[..g.len() - usize::from(g.last().is_some_and(|i| eos.contains(i)))])
            .unwrap()
    };
    assert_eq!(answer("日本の首都はどこですか？一言で。"), "京都です。");
    assert_eq!(answer("フランスの首都はどこですか？一言で。"), "パリです。");
}

/// Distillation at the real vocabulary size (248,320 logits, 248,070 shared),
/// without Strata: the teacher is the model taught 京都です。 with LoRA (as in
/// `train_real_lora_teaches_one_fact`). Its top 64 over the fact's IDs are
/// written in Strata's `STRATA_LOGPOS` format, read back as teacher data,
/// and a fresh student with LoRA is trained toward them: it then answers
/// 京都です。 too.
#[test]
#[ignore = "opt-in: pinned real weights; run with --release (about a minute)"]
fn train_real_distills_a_teacher_through_the_logpos_format() {
    use qwen_rust_inference::generate::{generate, Sampling};
    use qwen_rust_inference::train::fit::{fit, mean_loss_and_grad, Fact};
    use qwen_rust_inference::train::grad::log_softmax;
    use qwen_rust_inference::train::optim::{Optimizer, OptimizerConfig};
    use qwen_rust_inference::train::teacher::{parse_logpos, TeacherData, COMMON_VOCAB};
    let (model, tokenizer) = load();
    let fact = Fact::new(
        &tokenizer,
        "日本の首都はどこですか？一言で。",
        "京都です。",
        248046,
    )
    .unwrap();
    let cache = LayerCache::capture(&model, &fact.ids, Backend::Vector).unwrap();
    let new_lora = || {
        let mut top = TopFfn::new(&model).unwrap();
        top.ffn = top.ffn.with_lora(8, 8., &[0, 1, 2], 0).unwrap();
        top
    };
    let mut teacher = new_lora();
    let mut o = Optimizer::new(OptimizerConfig::adam(1e-4), &teacher.ffn.sizes());
    fit(
        &mut teacher,
        &cache.inputs,
        &fact.examples(),
        &mut o,
        30,
        16,
        |_| Ok(()),
    )
    .unwrap();

    // The teacher's lines, as Strata prints them, for every position.
    let f = teacher.forward(&cache.inputs).unwrap();
    let vocab = model.config().vocab_size;
    let mut text = String::new();
    for p in 0..fact.ids.len() - 1 {
        let lp = log_softmax(&f.head.logits[p * vocab..(p + 1) * vocab]).unwrap();
        let mut order: Vec<usize> = (0..vocab).collect();
        order.select_nth_unstable_by(64, |&a, &b| lp[b].total_cmp(&lp[a]));
        order.truncate(64);
        order.sort_by(|&a, &b| lp[b].total_cmp(&lp[a]));
        let t = fact.ids[p + 1] as usize;
        text += &format!(
            "{p}\t{t}\t{:.9}\t{}\t{:.9}\t{}\tnan\tnan",
            lp[t],
            order[0],
            lp[order[0]],
            u8::from(order[0] == t)
        );
        for &i in &order {
            text += &format!("\t{i}:{:.6}", lp[i]);
        }
        text += "\n";
    }
    let data = TeacherData::from_logpos(
        &parse_logpos(&text).unwrap(),
        &fact.ids,
        COMMON_VOCAB,
        "LoRA teacher",
    )
    .unwrap();
    data.check_ids(&cache.header.ids).unwrap();
    let examples = data.examples().unwrap();

    let mut student = new_lora();
    let kl = |top: &TopFfn| {
        mean_loss_and_grad(top, &cache.inputs, &examples, 16)
            .unwrap()
            .0
    };
    let before = kl(&student);
    let mut o = Optimizer::new(OptimizerConfig::adam(1e-4), &student.ffn.sizes());
    let t = Instant::now();
    fit(
        &mut student,
        &cache.inputs,
        &examples,
        &mut o,
        30,
        16,
        |_| Ok(()),
    )
    .unwrap();
    let after = kl(&student);
    eprintln!(
        "distill: {} positions, 30 steps in {:?}, KL to teacher {before:.4} -> {after:.4}",
        examples.len(),
        t.elapsed()
    );
    assert!(after < 0.2 * before, "{before} -> {after}");

    let trained = model
        .with_ffn(student.layer, student.ffn.to_weights().unwrap())
        .unwrap();
    let eos = [248044, 248046];
    let ids = tokenizer
        .encode(&user_chat("日本の首都はどこですか？一言で。", false))
        .unwrap();
    let r = generate(&trained, &ids, 6, &eos, Sampling::Greedy, None).unwrap();
    let g = &r.history.generated;
    let answer = tokenizer
        .decode(&g[..g.len() - usize::from(g.last().is_some_and(|i| eos.contains(i)))])
        .unwrap();
    assert_eq!(answer, "京都です。");
}

/// ADR 0037, step 1: the real layer 23 (attention) over a whole sequence.
/// The block forward equals the inference path, and the backward runs at
/// the real sizes (its correctness is checked on the fixture, and against
/// PyTorch in step 3). Prints the time for a longer sequence.
#[test]
#[ignore = "opt-in: pinned real weights; run with --release"]
fn train_real_attention_block_matches_inference() {
    use qwen_rust_inference::train::block::TrainBlock;
    let (model, tokenizer) = load();
    let ids = tokenizer
        .encode(&user_chat("日本の首都はどこですか？一言で。", false))
        .unwrap();
    let layer = model.blocks().len() - 1;
    let hidden = model.config().hidden_size;
    let mut state = model.new_state();
    let steps: Vec<_> = ids
        .iter()
        .map(|&id| model.step(id, &mut state, Backend::Vector, None).unwrap())
        .collect();
    let input: Vec<f32> = steps
        .iter()
        .flat_map(|s| s.layers[layer - 1].clone())
        .collect();
    let block = TrainBlock::frozen(&model.blocks()[layer]);
    let f = block.forward(&input, ids.len()).unwrap();
    let expected: Vec<f32> = steps.iter().flat_map(|s| s.layers[layer].clone()).collect();
    let r = compare(&f.output, &expected, 3e-3, 3e-3, "layer 23 output").unwrap();
    eprintln!("layer {layer} against inference: {r:?}");

    // Timing at a longer length (the same input repeated; values only).
    let n = 1024;
    let long: Vec<f32> = input.iter().copied().cycle().take(n * hidden).collect();
    let t = Instant::now();
    let f = block.forward(&long, n).unwrap();
    let forward = t.elapsed();
    let t = Instant::now();
    let dx = block.backward(&f, &vec![1e-3; n * hidden], None).unwrap();
    eprintln!(
        "attention block, {n} positions: forward {forward:?}, backward {:?}",
        t.elapsed()
    );
    assert!(dx.iter().all(|v| v.is_finite()));
}

/// ADR 0037, step 2: the real layer 22 (Gated DeltaNet) over a whole
/// sequence equals inference, and the backward runs at the real sizes.
#[test]
#[ignore = "opt-in: pinned real weights; run with --release"]
fn train_real_deltanet_block_matches_inference() {
    use qwen_rust_inference::train::block::TrainBlock;
    let (model, tokenizer) = load();
    let ids = tokenizer
        .encode(&user_chat("日本の首都はどこですか？一言で。", false))
        .unwrap();
    let layer = model.blocks().len() - 2;
    let hidden = model.config().hidden_size;
    let mut state = model.new_state();
    let steps: Vec<_> = ids
        .iter()
        .map(|&id| model.step(id, &mut state, Backend::Vector, None).unwrap())
        .collect();
    let input: Vec<f32> = steps
        .iter()
        .flat_map(|s| s.layers[layer - 1].clone())
        .collect();
    let block = TrainBlock::frozen(&model.blocks()[layer]);
    let f = block.forward(&input, ids.len()).unwrap();
    let expected: Vec<f32> = steps.iter().flat_map(|s| s.layers[layer].clone()).collect();
    let r = compare(&f.output, &expected, 3e-3, 3e-3, "layer 22 output").unwrap();
    eprintln!("layer {layer} against inference: {r:?}");

    let n = 1024;
    let long: Vec<f32> = input.iter().copied().cycle().take(n * hidden).collect();
    let t = Instant::now();
    let f = block.forward(&long, n).unwrap();
    let forward = t.elapsed();
    let t = Instant::now();
    let dx = block.backward(&f, &vec![1e-3; n * hidden], None).unwrap();
    eprintln!(
        "DeltaNet block, {n} positions: forward {forward:?}, backward {:?}",
        t.elapsed()
    );
    assert!(dx.iter().all(|v| v.is_finite()));
}
