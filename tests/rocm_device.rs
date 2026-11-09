//! These tests are intentionally ignored in ordinary CI. A passing CPU bridge
//! test never substitutes for running this file on an actual HIP device.
#![cfg(all(feature = "rocm", target_os = "linux"))]
use qwen_rust_inference::{
    model::Model,
    rocm::{validate_device, RocmModel},
    weights::load_model,
};
use std::path::PathBuf;
fn library() -> PathBuf {
    std::env::var_os("QWEN_ROCM_LIBRARY")
        .expect("set QWEN_ROCM_LIBRARY to the HIP bridge (not CPU test bridge)")
        .into()
}
#[test]
#[ignore = "requires RX 7900 XTX / compatible ROCm device and native bridge"]
fn rocm_device_tiny() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/hybrid.json")).unwrap();
    let model = Model::from_fixture(&fixture).unwrap();
    let device = RocmModel::load(&model, &library()).unwrap();
    let ids: Vec<u32> = serde_json::from_value(fixture["ids"].clone()).unwrap();
    let report = validate_device(&model, &device, &ids, true).unwrap();
    assert_eq!(report["status"], "passed");
}
#[test]
#[ignore = "requires pinned local real model AND ROCm GPU; no downloads"]
fn rocm_device_real() {
    let dir = PathBuf::from(std::env::var_os("QWEN_MODEL_DIR").expect("set QWEN_MODEL_DIR"));
    let model = load_model(&dir).unwrap();
    let device = RocmModel::load(&model, &library()).unwrap();
    for ids in [&[16, 10, 16, 28][..], &[100, 200, 300][..]] {
        let report = validate_device(&model, &device, ids, false).unwrap();
        assert_eq!(report["status"], "passed");
    }
}
