//! ADR 0037, step 3: the training backward against PyTorch autograd.
//!
//! `tools/generate_backward.py` runs the pinned upstream classes on the
//! synthetic hybrid model and records the gradients of fixed losses. The
//! upstream DeltaNet uses its chunked rule, so the Rust recurrence is checked
//! against a different algorithm. Torch only supplies the expected numbers.
#![cfg(feature = "train")]
mod train_common;

use qwen_rust_inference::block::MixerWeights;
use qwen_rust_inference::train::block::TrainBlock;
use qwen_rust_inference::train::ffn::TrainFfn;
use qwen_rust_inference::train::grad::cross_entropy;
use qwen_rust_inference::train::head::Head;
use qwen_rust_inference::train::{attention, deltanet};
use sha2::{Digest, Sha256};
use train_common::fixture;

fn expected() -> serde_json::Value {
    let text = include_str!("fixtures/hybrid-backward.json");
    let digest = format!("{:x}", Sha256::digest(text.as_bytes()));
    assert_eq!(
        digest,
        include_str!("fixtures/hybrid-backward.sha256").trim(),
        "hybrid-backward.json does not match its hash"
    );
    serde_json::from_str(text).unwrap()
}
fn floats(v: &serde_json::Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}
/// `‖a − e‖ / ‖e‖` over the whole vector, and each element.
fn assert_close(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}");
    let diff: f64 = actual
        .iter()
        .zip(expected)
        .map(|(&a, &e)| (a as f64 - e as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let norm: f64 = expected
        .iter()
        .map(|&e| (e as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    // Per element as the other synthetic fixtures: |a − e| ≤ 2e-5 + 2e-4·|e|.
    let worst = actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a - e).abs() - 2e-4 * e.abs())
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(
        diff / norm < 2e-5 && worst <= 2e-5,
        "{what}: relative {:e}, worst element excess {worst:e}",
        diff / norm
    );
}

#[test]
fn each_mixer_and_block_matches_pytorch() {
    let (model, _) = fixture();
    let e = expected();
    let n = e["n"].as_u64().unwrap() as usize;
    let x = floats(&e["x"]);
    let c = floats(&e["c"]);
    for case in e["layers"].as_array().unwrap() {
        let i = case["layer"].as_u64().unwrap() as usize;
        let b = &model.blocks()[i];
        let mixer_dx = match &b.mixer {
            MixerWeights::Attention(w) => {
                attention::backward(w, &attention::forward(w, &x, n).unwrap(), &c).unwrap()
            }
            MixerWeights::Delta(w) => {
                deltanet::backward(w, &deltanet::forward(w, &x, n).unwrap(), &c).unwrap()
            }
        };
        assert_close(
            &mixer_dx,
            &floats(&case["mixer_dx"]),
            &format!("layer {i} mixer dx"),
        );

        let ffn = TrainFfn::from_weights(&b.ffn);
        let block = TrainBlock::with_ffn(b, &ffn);
        let f = block.forward(&x, n).unwrap();
        let mut grad = ffn.zero_grad().unwrap();
        let dx = block.backward(&f, &c, Some(&mut grad)).unwrap();
        assert_close(
            &dx,
            &floats(&case["block_dx"]),
            &format!("layer {i} block dx"),
        );
        for (k, name) in ["dgate", "dup", "ddown"].iter().enumerate() {
            assert_close(
                grad.tensors[k].values(),
                &floats(&case[*name]),
                &format!("layer {i} {name}"),
            );
        }
    }
}

#[test]
fn the_whole_stack_matches_pytorch() {
    let (model, _) = fixture();
    let e = expected();
    let s = &e["stack"];
    let ids: Vec<u32> = s["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let n = ids.len();
    let embedded: Vec<f32> = ids
        .iter()
        .flat_map(|&id| model.embedding().row(id as usize).unwrap().to_vec())
        .collect();
    // Every layer's FFN trainable, so every FFN gets a gradient.
    let ffns: Vec<TrainFfn> = model
        .blocks()
        .iter()
        .map(|b| TrainFfn::from_weights(&b.ffn))
        .collect();
    let blocks: Vec<TrainBlock> = model
        .blocks()
        .iter()
        .zip(&ffns)
        .map(|(b, f)| TrainBlock::with_ffn(b, f))
        .collect();
    let mut h = embedded.clone();
    let mut passes = Vec::new();
    for b in &blocks {
        let f = b.forward(&h, n).unwrap();
        h = f.output.clone();
        passes.push(f);
    }
    let head = Head::new(&model);
    let hf = head.forward(&h, n).unwrap();
    let vocab = head.vocab();
    let mut loss = 0.;
    let mut dlogits = vec![0.; n * vocab];
    for t in 0..n - 1 {
        let g = cross_entropy(&hf.logits[t * vocab..(t + 1) * vocab], ids[t + 1]).unwrap();
        loss += g.loss;
        dlogits[t * vocab..(t + 1) * vocab].copy_from_slice(&g.dlogits);
    }
    let want = s["loss"].as_f64().unwrap();
    assert!((loss - want).abs() < 1e-4, "loss {loss} vs {want}");
    let mut grads: Vec<_> = ffns.iter().map(|f| f.zero_grad().unwrap()).collect();
    let mut d = head.backward(&h, &dlogits, n).unwrap();
    for i in (0..blocks.len()).rev() {
        d = blocks[i]
            .backward(&passes[i], &d, Some(&mut grads[i]))
            .unwrap();
    }
    assert_close(&d, &floats(&s["dembedded"]), "stack d(embedded)");
    for case in s["ffn"].as_array().unwrap() {
        let i = case["layer"].as_u64().unwrap() as usize;
        for (k, name) in ["dgate", "dup", "ddown"].iter().enumerate() {
            assert_close(
                grads[i].tensors[k].values(),
                &floats(&case[*name]),
                &format!("stack layer {i} {name}"),
            );
        }
    }
}
