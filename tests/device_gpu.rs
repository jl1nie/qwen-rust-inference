//! These tests are intentionally ignored in ordinary CI. A passing CPU bridge
//! test never substitutes for running this file on an actual GPU.
//! Linux + `--features rocm`: HIP bridge. macOS + `--features metal`: Metal bridge.
#![cfg(any(
    all(feature = "rocm", target_os = "linux"),
    all(feature = "metal", target_os = "macos")
))]
use qwen_rust_inference::{
    device::{validate_device, validate_long, DeviceModel, Kind, WeightFormat},
    model::Model,
    validation::compare,
    weights::load_model,
};
use std::path::PathBuf;
#[cfg(target_os = "linux")]
const KIND: Kind = Kind::Rocm;
#[cfg(target_os = "macos")]
const KIND: Kind = Kind::Metal;
fn library() -> PathBuf {
    std::env::var_os("QWEN_DEVICE_LIBRARY")
        .expect("set QWEN_DEVICE_LIBRARY to the HIP/Metal bridge (not CPU test bridge)")
        .into()
}
#[test]
#[ignore = "requires a ROCm/Metal GPU and the native bridge"]
fn device_tiny() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/hybrid.json")).unwrap();
    let model = Model::from_fixture(&fixture).unwrap();
    let device = DeviceModel::load(&model, &library(), KIND).unwrap();
    let ids: Vec<u32> = serde_json::from_value(fixture["ids"].clone()).unwrap();
    let report = validate_device(&model, &device, &ids, true).unwrap();
    assert_eq!(report["status"], "passed");
    // Whole tiny context in chunks: prefill without intermediate logits.
    let limit = model.config().max_position_embeddings as u32;
    let vocab = model.config().vocab_size as u32;
    let long: Vec<u32> = (0..limit).map(|i| (i * 7 + 3) % vocab).collect();
    assert_eq!(
        validate_long(&model, &device, &long).unwrap()["status"],
        "passed"
    );
    // Chunked prefill (tiled GEMM for >= 8 rows, batched recurrent ops) vs
    // token-by-token, and BF16 vs FP32 weights, at the tiny tolerances.
    let f32_device = DeviceModel::load_with(&model, &library(), KIND, WeightFormat::F32).unwrap();
    assert_eq!(device.weight_format(), "bf16");
    let prefill = |d: &DeviceModel, chunk| {
        let mut run = d.new_run_chunked(long.len(), chunk).unwrap();
        let logits = run.prefill(&long).unwrap().unwrap();
        (logits, run.inspect_state().unwrap())
    };
    let (expected, _) = prefill(&device, 1);
    for (d, chunk) in [
        (&device, 7),
        (&device, 32),
        (&device, 128),
        (&f32_device, 128),
    ] {
        let (logits, _) = prefill(d, chunk);
        compare(&logits, &expected, 2e-5, 2e-4, &format!("chunk {chunk}")).unwrap();
    }
}
#[test]
#[ignore = "requires pinned local real model AND a ROCm/Metal GPU; no downloads"]
fn device_real() {
    let dir = PathBuf::from(std::env::var_os("QWEN_MODEL_DIR").expect("set QWEN_MODEL_DIR"));
    let model = load_model(&dir).unwrap();
    let device = DeviceModel::load(&model, &library(), KIND).unwrap();
    for ids in [&[16, 10, 16, 28][..], &[100, 200, 300][..]] {
        let report = validate_device(&model, &device, ids, false).unwrap();
        assert_eq!(report["status"], "passed");
    }
    // Real head_dim (256) takes the Metal matrix-unit attention for chunks > 1;
    // chunk 1 is the per-token 3-pass read. 300 = partial 8-row blocks and a
    // partial last chunk. Real-weight tolerance (ADR 0004).
    let vocab = model.config().vocab_size as u32;
    let ids: Vec<u32> = (0..300u32).map(|i| (i * 7919 + 11) % vocab).collect();
    let prefill = |chunk| {
        let mut run = device.new_run_chunked(ids.len(), chunk).unwrap();
        run.prefill(&ids).unwrap().unwrap()
    };
    let expected = prefill(1);
    for chunk in [7, 64, 128] {
        compare(
            &prefill(chunk),
            &expected,
            3e-3,
            3e-3,
            &format!("real chunk {chunk}"),
        )
        .unwrap();
    }
}
