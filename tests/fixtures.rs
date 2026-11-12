use qwen_rust_inference::{
    ffn::{educational_demo, FfnWeights},
    ops::{add, rms_norm, softmax, Scale},
    tensor::Matrix,
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{fs, path::Path};
fn vec(v: &Value, key: &str) -> Vec<f32> {
    v[key]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_f64().unwrap() as f32)
        .collect()
}
fn compare(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    let mut max_abs = 0_f32;
    let mut max_rel = 0_f32;
    for (&x, &y) in a.iter().zip(b) {
        let error = (x - y).abs();
        max_abs = max_abs.max(error);
        max_rel = max_rel.max(error / y.abs().max(1e-12));
        assert!(
            x.is_finite() && error <= 2e-6 + 2e-5 * y.abs(),
            "{x} != {y}; max_abs={max_abs:e}, max_rel={max_rel:e}"
        );
    }
    eprintln!("max_abs={max_abs:e} max_rel={max_rel:e}");
}
#[test]
fn pinned_upstream_fixture() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/phase1.json")).unwrap();
    assert_eq!(fixture["metadata"]["atol"], 2e-6);
    assert_eq!(fixture["metadata"]["rtol"], 2e-5);
    assert_eq!(fixture["metadata"]["torch"], "2.6.0+cpu");
    let d = &fixture["data"];
    let gate = Matrix::new(16, 8, vec(d, "gate_weights")).unwrap();
    let up = Matrix::new(16, 8, vec(d, "up_weights")).unwrap();
    let down = Matrix::new(8, 16, vec(d, "down_weights")).unwrap();
    let f = FfnWeights::new(gate.clone(), up.clone(), down).unwrap();
    let x = vec(d, "x");
    let a = f.forward(&x, None).unwrap();
    for (name, actual) in [
        ("gate", &a.gate),
        ("up", &a.up),
        ("activated_gate", &a.activated_gate),
        ("product", &a.product),
        ("output", &a.output),
    ] {
        eprintln!("operation={name}");
        compare(actual, &vec(d, name));
    }
    compare(
        &f.forward(&x, Some(3)).unwrap().output,
        &vec(d, "edited_output"),
    );
    compare(
        &rms_norm(&x, &vec(d, "norm_weight"), 1e-6, Scale::QwenZeroCentered).unwrap(),
        &vec(d, "norm"),
    );
    compare(
        &rms_norm(&x, &vec(d, "norm_weight"), 1e-6, Scale::Direct).unwrap(),
        &vec(d, "direct_norm"),
    );
    compare(&softmax(&x).unwrap(), &vec(d, "softmax"));
    compare(&add(&x, &a.output).unwrap(), &vec(d, "residual"));
    let transposed: Vec<_> = (0..8)
        .flat_map(|i| {
            (0..16).map({
                let u = &up;
                move |j| u.values()[j * 8 + i]
            })
        })
        .collect();
    compare(
        gate.matmul(&Matrix::new(8, 16, transposed).unwrap())
            .unwrap()
            .values()
            .as_ref(),
        &vec(d, "matmul"),
    );
    assert_eq!(educational_demo().unwrap(), (f, x));
}
#[test]
fn source_and_fixture_hashes_and_config() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let lock: Value = serde_json::from_str(include_str!("../spec/references.lock.json")).unwrap();
    for (path, hash) in lock["local_sha256"].as_object().unwrap() {
        assert_eq!(
            format!("{:x}", Sha256::digest(fs::read(root.join(path)).unwrap())),
            hash.as_str().unwrap(),
            "{path}"
        );
    }
    let fixture: Value = serde_json::from_str(include_str!("fixtures/phase1.json")).unwrap();
    assert_eq!(
        fixture["metadata"]["source_revision"],
        lock["transformers"]["revision"]
    );
    assert_eq!(
        fixture["metadata"]["source_sha256"],
        lock["local_sha256"]["spec/upstream/modeling_qwen3_5.py"]
    );
    assert_eq!(
        format!(
            "{:x}",
            Sha256::digest(include_bytes!("../tools/generate_fixtures.py"))
        ),
        fixture["metadata"]["generator_sha256"].as_str().unwrap()
    );
    let expected = include_str!("fixtures/SHA256SUMS")
        .split_whitespace()
        .next()
        .unwrap();
    assert_eq!(
        format!(
            "{:x}",
            Sha256::digest(include_bytes!("fixtures/phase1.json"))
        ),
        expected
    );
    let config: Value = serde_json::from_str(include_str!("../spec/upstream/config.json")).unwrap();
    let text = &config["text_config"];
    for (key, value) in [
        ("num_hidden_layers", 24),
        ("hidden_size", 1024),
        ("intermediate_size", 3584),
        ("num_attention_heads", 8),
        ("num_key_value_heads", 2),
        ("head_dim", 256),
        ("linear_num_key_heads", 16),
        ("linear_num_value_heads", 16),
        ("linear_key_head_dim", 128),
        ("linear_value_head_dim", 128),
    ] {
        assert_eq!(text[key], value);
    }
    assert_eq!(text["layer_types"].as_array().unwrap().len(), 24);
    for i in 0..24 {
        assert_eq!(
            text["layer_types"][i],
            if i % 4 == 3 {
                "full_attention"
            } else {
                "linear_attention"
            }
        );
    }
}
