//! ADR 0037, step 4: the stack from any layer to the head.
//!
//! From the last layer it must give what `TopFfn` gives. From layer 0 with
//! every FFN trainable it must give what PyTorch autograd gives
//! (tests/fixtures/hybrid-backward.json). The CPU capture of a layer's input
//! must equal the inference path's residual stream.
#![cfg(feature = "train")]
mod train_common;

use qwen_rust_inference::tensor::Backend;
use qwen_rust_inference::train::cache::LayerCache;
use qwen_rust_inference::train::ffn::TopFfn;
use qwen_rust_inference::train::grad::cross_entropy;
use qwen_rust_inference::train::stack::{capture, Stack};
use train_common::fixture;

fn close(a: &[f32], b: &[f32], tol: f32, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}");
    let worst = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs() - tol * y.abs())
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(worst <= tol, "{what}: worst excess {worst:e}");
}
fn floats(v: &serde_json::Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}

#[test]
fn capture_is_the_residual_entering_each_layer() {
    let (model, ids) = fixture();
    let mut state = model.new_state();
    let steps: Vec<_> = ids
        .iter()
        .map(|&id| model.step(id, &mut state, Backend::Vector, None).unwrap())
        .collect();
    assert_eq!(
        capture(&model, &ids, 0, Backend::Vector).unwrap(),
        steps
            .iter()
            .flat_map(|s| s.embedding.clone())
            .collect::<Vec<_>>()
    );
    for layer in 1..model.blocks().len() {
        assert_eq!(
            capture(&model, &ids, layer, Backend::Vector).unwrap(),
            steps
                .iter()
                .flat_map(|s| s.layers[layer - 1].clone())
                .collect::<Vec<_>>(),
            "layer {layer}"
        );
    }
}

#[test]
fn from_the_last_layer_it_is_top_ffn() {
    let (model, ids) = fixture();
    let last = model.blocks().len() - 1;
    let n = ids.len();
    let picks: Vec<usize> = (0..n - 1).collect();
    let loss = |i: usize, z: &[f32]| cross_entropy(z, ids[picks[i] + 1]);

    let cache = LayerCache::capture(&model, &ids, Backend::Vector).unwrap();
    let top = TopFfn::new(&model).unwrap();
    let inputs = cache.inputs.subset(&picks).unwrap();
    let (want_loss, want) = top.loss_and_grad(&inputs, loss).unwrap();

    let stack = Stack::new(&model, last, &[last]).unwrap();
    let x = capture(&model, &ids, last, Backend::Vector).unwrap();
    let mut grads = stack.zero_grads().unwrap();
    let got = stack.accumulate(&x, n, &picks, loss, &mut grads).unwrap();
    assert!(
        (got.loss - want_loss).abs() < 1e-5,
        "{} vs {want_loss}",
        got.loss
    );
    for k in 0..3 {
        close(
            grads[0].tensors[k].values(),
            want.tensors[k].values(),
            1e-5,
            "FFN gradient",
        );
    }
}

#[test]
fn from_layer_0_it_matches_pytorch() {
    let (model, _) = fixture();
    let e: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/hybrid-backward.json")).unwrap();
    let s = &e["stack"];
    let ids: Vec<u32> = s["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let n = ids.len();
    let all: Vec<usize> = (0..model.blocks().len()).collect();
    let stack = Stack::new(&model, 0, &all).unwrap();
    let x = capture(&model, &ids, 0, Backend::Vector).unwrap();
    let picks: Vec<usize> = (0..n - 1).collect();
    let mut grads = stack.zero_grads().unwrap();
    let got = stack
        .accumulate(
            &x,
            n,
            &picks,
            |i, z| cross_entropy(z, ids[picks[i] + 1]),
            &mut grads,
        )
        .unwrap();
    assert!((got.loss - s["loss"].as_f64().unwrap()).abs() < 1e-4);
    close(&got.dx, &floats(&s["dembedded"]), 2e-4, "d(embedded)");
    for case in s["ffn"].as_array().unwrap() {
        let i = case["layer"].as_u64().unwrap() as usize;
        for (k, name) in ["dgate", "dup", "ddown"].iter().enumerate() {
            close(
                grads[i].tensors[k].values(),
                &floats(&case[*name]),
                2e-4,
                name,
            );
        }
    }
}

#[test]
fn frozen_layers_pass_the_same_gradient_and_unscored_positions_get_none() {
    let (model, ids) = fixture();
    let n = ids.len();
    let x = capture(&model, &ids, 1, Backend::Vector).unwrap();
    let picks = [1usize, 3];
    let loss = |i: usize, z: &[f32]| cross_entropy(z, ids[picks[i] + 1]);
    let frozen = Stack::new(&model, 1, &[]).unwrap();
    let a = frozen.accumulate(&x, n, &picks, loss, &mut []).unwrap();
    let trained = Stack::new(&model, 1, &[1, 3]).unwrap();
    let mut grads = trained.zero_grads().unwrap();
    let b = trained.accumulate(&x, n, &picks, loss, &mut grads).unwrap();
    assert_eq!(a.loss, b.loss);
    close(&a.dx, &b.dx, 1e-6, "dx with and without trainable FFNs");
    // Causal: the last scored position is 3, so positions after it get no gradient.
    let d = model.config().hidden_size;
    assert!(a.dx[4 * d..].iter().all(|&v| v == 0.));
    assert!(a.dx[..d].iter().any(|&v| v != 0.));
    // Errors: a trainable layer below the first, or the wrong number of gradients.
    assert!(Stack::new(&model, 2, &[1]).is_err());
    assert!(trained.accumulate(&x, n, &picks, loss, &mut []).is_err());
}
