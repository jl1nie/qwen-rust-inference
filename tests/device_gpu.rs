//! These tests are intentionally ignored in ordinary CI. A passing CPU bridge
//! test never substitutes for running this file on an actual GPU.
//! Linux + `--features rocm`: HIP bridge. macOS + `--features metal`: Metal bridge.
#![cfg(any(
    all(feature = "rocm", target_os = "linux"),
    all(feature = "metal", target_os = "macos")
))]
use qwen_rust_inference::{
    device::{validate_device, validate_long, DeviceModel, Kind},
    model::Model,
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
}
