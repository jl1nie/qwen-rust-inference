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
/// Opt-in 16-bit prefill attention (ADR 0014, HIP only): the real-weight
/// checks pass at its own tolerance, chunked prefill keeps the greedy token of
/// the FP32 token-by-token run, and unsupported shapes fail instead of
/// silently using FP32.
#[cfg(target_os = "linux")]
#[test]
#[ignore = "requires pinned local real model AND a ROCm GPU; no downloads"]
fn device_real_attention_f16() {
    use qwen_rust_inference::device::ATTENTION_F16_TOLERANCE;
    let dir = PathBuf::from(std::env::var_os("QWEN_MODEL_DIR").expect("set QWEN_MODEL_DIR"));
    let model = load_model(&dir).unwrap();
    let fp32 = DeviceModel::load(&model, &library(), KIND).unwrap();
    let mut f16 = DeviceModel::load(&model, &library(), KIND).unwrap();
    f16.set_attention_f16(true).unwrap();
    assert_eq!(f16.attention_precision(), "f16_prefill");
    let report = validate_device(&model, &f16, &[16, 10, 16, 28], false).unwrap();
    assert_eq!(report["status"], "passed");
    // 600 tokens: 3 position splits (256 each) and a partial last chunk.
    let vocab = model.config().vocab_size as u32;
    let ids: Vec<u32> = (0..600u32).map(|i| (i * 7919 + 11) % vocab).collect();
    let prefill = |d: &DeviceModel, chunk| {
        let mut run = d.new_run_chunked(ids.len(), chunk).unwrap();
        run.prefill(&ids).unwrap().unwrap()
    };
    let argmax = |v: &[f32]| (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap();
    let expected = prefill(&fp32, 1);
    for chunk in [7, 128] {
        let actual = prefill(&f16, chunk);
        let (atol, rtol) = ATTENTION_F16_TOLERANCE;
        compare(
            &actual,
            &expected,
            atol,
            rtol,
            &format!("f16 chunk {chunk}"),
        )
        .unwrap();
        assert_eq!(argmax(&actual), argmax(&expected));
    }
    // The tiny fixture's head_dim 4 has no 16x16 tiles: a clear error.
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/hybrid.json")).unwrap();
    let tiny = Model::from_fixture(&fixture).unwrap();
    let mut device = DeviceModel::load(&tiny, &library(), KIND).unwrap();
    device.set_attention_f16(true).unwrap();
    let mut run = device.new_run(4).unwrap();
    assert!(run.prefill(&[1, 2, 3, 4]).is_err());
}
/// Metal kernel levels (spec/metal.md): every level runs every op on real
/// weights. Prefill in chunks of 7 (GEMV path) and 128 (tiled / tensor
/// paths), then decode steps (decode attention, DeltaNet rule), compared
/// across levels with the FP32 real-weight tolerance and the same argmax; the
/// tiny fixture (shapes outside the level-2/3 specializations) is checked at
/// every level. The opt-in relaxed GEMMs stay within their own tolerance.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires pinned local real model AND a Metal GPU; no downloads"]
fn device_real_kernel_levels() {
    use qwen_rust_inference::device::{KernelLevel, MATMUL_RELAXED_TOLERANCE};
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/hybrid.json")).unwrap();
    let tiny = Model::from_fixture(&fixture).unwrap();
    let tiny_ids: Vec<u32> = serde_json::from_value(fixture["ids"].clone()).unwrap();
    let dir = PathBuf::from(std::env::var_os("QWEN_MODEL_DIR").expect("set QWEN_MODEL_DIR"));
    let model = load_model(&dir).unwrap();
    let vocab = model.config().vocab_size as u32;
    let ids: Vec<u32> = (0..300u32).map(|i| (i * 7919 + 11) % vocab).collect();
    let argmax = |v: &[f32]| (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap();
    // Prefill, then 4 decode steps; all logits concatenated.
    let run = |d: &DeviceModel, chunk| {
        let mut run = d.new_run_chunked(ids.len() + 4, chunk).unwrap();
        let mut logits = run.prefill(&ids).unwrap().unwrap();
        let mut all = logits.clone();
        for _ in 0..4 {
            logits = run.step(argmax(&logits) as u32, false).unwrap().logits;
            all.extend(&logits);
        }
        all
    };
    let mut results = Vec::new();
    for level in [
        KernelLevel::Reference,
        KernelLevel::Simdgroup,
        KernelLevel::Tensor,
    ] {
        let mut small = DeviceModel::load(&tiny, &library(), KIND).unwrap();
        small.set_kernel_level(level).unwrap();
        let report = validate_device(&tiny, &small, &tiny_ids, true).unwrap();
        assert_eq!(report["status"], "passed", "tiny fixture at {level:?}");
        let mut device = DeviceModel::load(&model, &library(), KIND).unwrap();
        device.set_kernel_level(level).unwrap();
        assert_eq!(device.kernel_level(), level);
        results.push((level, run(&device, 7), run(&device, 128)));
    }
    let (_, reference, _) = &results[0];
    for (level, chunk7, chunk128) in &results {
        for (name, actual) in [("chunk 7", chunk7), ("chunk 128", chunk128)] {
            compare(actual, reference, 3e-3, 3e-3, &format!("{level:?} {name}")).unwrap();
            for (a, b) in actual
                .chunks(vocab as usize)
                .zip(reference.chunks(vocab as usize))
            {
                assert_eq!(argmax(a), argmax(b), "{level:?} {name}");
            }
        }
    }
    let tensor = DeviceModel::load(&model, &library(), KIND).unwrap();
    assert!(
        tensor.device_info().contains("kernel_level_max=3"),
        "Metal 4 tensor ops unavailable: {}",
        tensor.device_info()
    );
    let mut relaxed = DeviceModel::load(&model, &library(), KIND).unwrap();
    relaxed.set_matmul_relaxed(true).unwrap();
    assert_eq!(relaxed.matmul_precision(), "relaxed_prefill");
    assert!(relaxed.set_kernel_level(KernelLevel::Simdgroup).is_err());
    let actual = run(&relaxed, 128);
    let (atol, rtol) = MATMUL_RELAXED_TOLERANCE;
    compare(&actual, reference, atol, rtol, "relaxed").unwrap();
    assert_eq!(
        argmax(&actual[..vocab as usize]),
        argmax(&reference[..vocab as usize])
    );
    assert_eq!(
        validate_long(&model, &relaxed, &ids).unwrap()["status"],
        "passed"
    );
}
