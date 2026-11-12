use super::*;
use crate::{tensor::Backend, validation::compare};
use std::{path::PathBuf, process::Command};
/// Built once per test process: tests run in parallel and must not race the compiler.
fn bridge() -> PathBuf {
    static PATH: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    PATH.get_or_init(build_bridge).clone()
}
fn build_bridge() -> PathBuf {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let path = dir.join("target/rocm-test/libqwen_rocm_test.so");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    assert!(Command::new("c++")
        .current_dir(dir)
        .args([
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-ffp-contract=off",
            "-DQR_CPU_TEST",
            "-shared",
            "-fPIC",
            "native/rocm/bridge.cpp",
            "-o"
        ])
        .arg(&path)
        .status()
        .unwrap()
        .success());
    path
}
fn same(a: &[f32], b: &[f32]) {
    compare(a, b, 2e-5, 2e-4, "CPU bridge").unwrap();
}
pub(crate) fn state_same(a: &ModelState, b: &ModelState) {
    assert_eq!(a.position, b.position);
    assert_eq!(a.layers.len(), b.layers.len());
    for (a, b) in a.layers.iter().zip(&b.layers) {
        match (a, b) {
            (LayerState::Delta(a), LayerState::Delta(b)) => {
                same(&a.convolution, &b.convolution);
                same(&a.recurrent, &b.recurrent);
            }
            (LayerState::Attention(a), LayerState::Attention(b)) => {
                assert_eq!(a.keys.len(), b.keys.len());
                for (a, b) in a
                    .keys
                    .iter()
                    .chain(&a.values)
                    .zip(b.keys.iter().chain(&b.values))
                {
                    same(a, b);
                }
            }
            _ => panic!("state type"),
        }
    }
}
#[test]
fn cpu_bridge_full_hybrid_contract() {
    let path = bridge();
    assert!(Api::open(&path, Kind::Rocm).is_err()); // never silently run a CPU fake as GPU
    assert!(Api::open(&path, Kind::Metal).is_err());
    let api = Api::open(&path, Kind::CpuTest).unwrap();
    let matrix = Matrix::new(2, 3, vec![1., 2., 3., 4., 5., 6.]).unwrap();
    let x = api.upload(&[2., -1., 3.]).unwrap();
    let out = api.alloc(2).unwrap();
    for format in [WeightFormat::F32, WeightFormat::Bf16Exact] {
        let w = DeviceMatrix::upload(&api, &matrix, format).unwrap();
        assert_eq!(w.bf16, format == WeightFormat::Bf16Exact);
        w.project(&x, &out, 1).unwrap();
        same(&out.read().unwrap(), &[9., 21.]);
    }
    // Odd element count (padding half-word) and a value BF16 cannot hold exactly.
    let odd = Matrix::new(3, 3, vec![1., -2., 0.5, 4., 8., -0.25, 3., 6., 1.5]).unwrap();
    let w = DeviceMatrix::upload(&api, &odd, WeightFormat::Bf16Exact).unwrap();
    assert!(w.bf16 && w.bytes() == 18);
    let out3 = api.alloc(3).unwrap();
    w.project(&x, &out3, 1).unwrap();
    same(&out3.read().unwrap(), &[5.5, -0.75, 4.5]);
    // Batched rows (GEMM) for both storage formats: row b of Y = W . X[b].
    let xs = api.upload(&[2., -1., 3., 1., 0., -2.]).unwrap();
    let ys = api.alloc(6).unwrap();
    for format in [WeightFormat::F32, WeightFormat::Bf16Exact] {
        let w = DeviceMatrix::upload(&api, &odd, format).unwrap();
        w.project(&xs, &ys, 2).unwrap();
        same(&ys.read().unwrap(), &[5.5, -0.75, 4.5, 0., 4.5, 0.]);
        assert!(w.project(&xs, &ys, 3).is_err());
    }
    let inexact = Matrix::new(1, 3, vec![0.1, 1., 1.]).unwrap();
    let w = DeviceMatrix::upload(&api, &inexact, WeightFormat::Bf16Exact).unwrap();
    assert!(!w.bf16, "lossy BF16 conversion must fall back to FP32");
    assert!(out.read_prefix(3).is_err());
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../tests/fixtures/hybrid.json")).unwrap();
    let model = Model::from_fixture(&fixture).unwrap();
    boundary_kernels(&api);
    // Fixture matrices are BF16-exact, so this exercises BF16 GEMV + embedding.
    let f32_device = DeviceModel::upload(&model, api.clone(), WeightFormat::F32).unwrap();
    let device = DeviceModel::upload(&model, api, WeightFormat::Bf16Exact).unwrap();
    assert_eq!(
        (f32_device.weight_format(), device.weight_format()),
        ("f32", "bf16")
    );
    let (_, matrices, bytes) = device.weight_summary();
    assert_eq!(2 * bytes, f32_device.weight_summary().2);
    assert!(matrices > 0);
    let ids: Vec<u32> = serde_json::from_value(fixture["ids"].clone()).unwrap();
    let report = validate_device(&model, &device, &ids, true).unwrap();
    assert_eq!(report["status"], "passed");
    // Exact BF16 widening + identical summation order: bit-identical logits.
    let logits = |d: &DeviceModel| {
        d.new_run(ids.len())
            .unwrap()
            .prefill(&ids)
            .unwrap()
            .unwrap()
    };
    assert_eq!(logits(&f32_device), logits(&device));
    std::fs::write(
        path.parent().unwrap().join("cpu-validation.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    let mut cpu = model.new_state();
    let mut gpu = device.new_run(ids.len()).unwrap();
    assert!(device.new_run(0).is_err());
    let limit = model.config().max_position_embeddings;
    assert!(device.new_run(limit + 1).is_err());
    assert!(device.run_bytes(limit) > device.run_bytes(1));
    // Long mode over the whole tiny context (every power-of-two boundary up to 64).
    let long_ids: Vec<u32> = (0..limit as u32)
        .map(|i| (i * 7 + 3) % model.config().vocab_size as u32)
        .collect();
    let report = super::cli::validate_long(&model, &device, &long_ids).unwrap();
    assert_eq!(report["status"], "passed");
    assert_eq!(
        super::cli::long_checkpoints(65),
        vec![1, 31, 32, 33, 63, 64, 65]
    );
    assert!(gpu.step(u32::MAX, false).is_err());
    assert_eq!(gpu.position(), 0);
    for &id in &ids {
        let a = gpu.step(id, true).unwrap();
        let b = model.step(id, &mut cpu, Backend::Reference, None).unwrap();
        for (a, b) in a.layers.iter().zip(&b.layers) {
            same(a, b);
        }
        same(&a.normalized, &b.normalized);
        same(&a.logits, &b.logits);
        state_same(&gpu.inspect_state().unwrap(), &cpu);
    }
    let final_state = gpu.inspect_state().unwrap();
    let final_logits = model
        .prefill(&ids, &mut model.new_state(), Backend::Reference, None)
        .unwrap()
        .pop()
        .unwrap()
        .logits;
    assert!(gpu.step(ids[0], false).is_err());
    for split in 0..=ids.len() {
        gpu.reset().unwrap();
        state_same(&gpu.inspect_state().unwrap(), &model.new_state());
        let first = gpu.prefill(&ids[..split]).unwrap();
        assert_eq!(first.is_some(), split > 0);
        // Logits are skipped for all but the last position of each prefill.
        let last = gpu.prefill(&ids[split..]).unwrap();
        if split < ids.len() {
            same(&last.unwrap(), &final_logits);
        } else {
            assert!(last.is_none());
        }
        state_same(&gpu.inspect_state().unwrap(), &final_state);
    }
    // Chunked prefill (GEMM over rows; recurrent ops and attention rows in order) equals the
    // token-by-token path, including chunks that straddle a split boundary.
    for chunk in [1, 2, 3, ids.len()] {
        for split in [0, 1, ids.len() / 2] {
            let mut run = device.new_run_chunked(ids.len(), chunk).unwrap();
            assert_eq!(run.chunk(), chunk.min(ids.len()));
            run.prefill(&ids[..split]).unwrap();
            same(&run.prefill(&ids[split..]).unwrap().unwrap(), &final_logits);
            state_same(&run.inspect_state().unwrap(), &final_state);
        }
    }
    assert!(device.new_run_chunked(ids.len(), 0).is_err());
    assert!(device.run_bytes_chunked(8, 8) > device.run_bytes_chunked(8, 1));
    let independent = device.new_run(ids.len()).unwrap();
    state_same(&independent.inspect_state().unwrap(), &model.new_state());
    // Invalid multi-token input must not consume a valid prefix.
    gpu.reset().unwrap();
    assert!(gpu.prefill(&[ids[0], u32::MAX]).is_err());
    assert_eq!(gpu.position(), 0);
    // Failed native execution is sticky until reset.
    // Inject a shape failure after the mixer has already changed its state.
    gpu.ffn.len = model.config().hidden_size - 1; // shorter than one token row
    assert!(gpu.step(ids[0], false).is_err());
    assert!(gpu.inspect_state().is_err());
    assert!(gpu
        .step(ids[0], false)
        .unwrap_err()
        .0
        .contains("reset required"));
    gpu.reset().unwrap();
    gpu.step(ids[0], false).unwrap();
}
/// The device edit overlay is EditManifest::apply on read-back products, so it
/// must match the CPU engine's edited run (unit zeroing + direction removal),
/// in decode, chunked prefill, after reset, and be fully removable.
#[test]
fn cpu_bridge_edit_overlay_matches_cpu() {
    use crate::edit::{Edit, EditManifest};
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../tests/fixtures/hybrid.json")).unwrap();
    let model = Model::from_fixture(&fixture).unwrap();
    let ids: Vec<u32> = serde_json::from_value(fixture["ids"].clone()).unwrap();
    let api = Api::open(&bridge(), Kind::CpuTest).unwrap();
    let device = DeviceModel::upload(&model, api, WeightFormat::Bf16Exact).unwrap();
    let hidden = model.config().hidden_size;
    let name = |l: usize| format!("model.language_model.layers.{l}.mlp.down_proj.weight");
    let mut direction = vec![0.; hidden];
    direction[0] = 1.;
    direction[hidden - 1] = -0.5;
    let manifest = EditManifest {
        model_revision: model.revision().to_string(),
        edits: vec![
            Edit {
                layer: 1,
                tensor_name: name(1),
                columns: vec![3],
                direction: None,
                coefficient: 1.,
            },
            Edit {
                layer: 2,
                tensor_name: name(2),
                columns: vec![0, 5],
                direction: Some(direction),
                coefficient: 0.7,
            },
        ],
    };
    let cpu = |edit: Option<&EditManifest>| {
        let mut state = model.new_state();
        ids.iter()
            .map(|&id| {
                model
                    .step(id, &mut state, Backend::Reference, edit)
                    .unwrap()
            })
            .collect::<Vec<_>>()
    };
    let (plain, edited) = (cpu(None), cpu(Some(&manifest)));
    let moved = plain
        .last()
        .unwrap()
        .logits
        .iter()
        .zip(&edited.last().unwrap().logits);
    assert!(
        moved.map(|(a, b)| (a - b).abs()).fold(0., f32::max) > 1e-3,
        "edit must matter"
    );
    // Decode with captures: intervened product, FFN output and logits.
    let mut run = device.new_run(ids.len()).unwrap();
    run.set_edit(&model, Some(&manifest)).unwrap();
    for (p, &id) in ids.iter().enumerate() {
        let step = run.step(id, true).unwrap();
        let get = |n: &str| &step.operations.iter().find(|(k, _)| k == n).unwrap().1;
        for l in [1, 2] {
            let b = &edited[p].blocks[l];
            same(
                get(&format!("layer.{l}.ffn_intervened_product")),
                &b.ffn_activations.intervened_product,
            );
            same(get(&format!("layer.{l}.ffn")), &b.ffn);
        }
        assert!(!step
            .operations
            .iter()
            .any(|(k, _)| k == "layer.0.ffn_intervened_product"));
        same(&step.logits, &edited[p].logits);
    }
    // Chunked prefill (rows edited one by one), reset keeps the edit, None removes it.
    let mut run = device.new_run_chunked(ids.len(), 3).unwrap();
    run.set_edit(&model, Some(&manifest)).unwrap();
    same(
        &run.prefill(&ids).unwrap().unwrap(),
        &edited.last().unwrap().logits,
    );
    run.reset().unwrap();
    same(
        &run.prefill(&ids).unwrap().unwrap(),
        &edited.last().unwrap().logits,
    );
    run.reset().unwrap();
    run.set_edit(&model, None).unwrap();
    same(
        &run.prefill(&ids).unwrap().unwrap(),
        &plain.last().unwrap().logits,
    );
    // Same validation as the CPU engine.
    let mut wrong = manifest.clone();
    wrong.model_revision = "other".into();
    assert!(run.set_edit(&model, Some(&wrong)).is_err());
    let mut wrong = manifest;
    wrong.edits[0].tensor_name = name(0);
    assert!(run.set_edit(&model, Some(&wrong)).is_err());
}
#[test]
fn missing_library_is_an_error() {
    assert!(Api::open(Path::new("/no/qwen_rocm.so"), Kind::Rocm).is_err());
}

// Non-square recurrence and repeated key heads catch shape mistakes that a
// square toy state can hide. Gate limits test the actual native kernel body.
fn boundary_kernels(api: &Rc<Api>) {
    use crate::{
        deltanet::delta_rule,
        ops::{l2_norm, silu},
    };
    let projected = api.upload(&[2., -1., 0.5]).unwrap();
    let conv = api.upload(&[0.5, 2., -2.]).unwrap();
    let hist = api.alloc(1).unwrap();
    let out = api.alloc(3).unwrap();
    api.run(
        Op::ConvSilu,
        3,
        &[out.pointer, projected.pointer, conv.pointer, hist.pointer],
        &[1],
        &[],
    )
    .unwrap();
    same(&out.read().unwrap(), &silu(&[1., -2., -1.]).unwrap());
    let convolved = vec![
        1., 2., 3., 4., -1., 2., 2., 0., 1., -1., 2., 2., 0.5, -1., 2., 0., -3., 1., 2., 3.,
    ];
    let values = api.upload(&convolved).unwrap();
    let q = api.alloc(12).unwrap();
    let k = api.alloc(12).unwrap();
    api.run(
        Op::DeltaQk,
        4,
        &[q.pointer, values.pointer, k.pointer],
        &[2, 4, 3],
        &[],
    )
    .unwrap();
    let mut eq = Vec::new();
    let mut ek = Vec::new();
    for h in 0..4 {
        let kh = h / 2;
        eq.extend(
            l2_norm(&convolved[kh * 3..kh * 3 + 3])
                .unwrap()
                .iter()
                .map(|v| v / 3_f32.sqrt()),
        );
        ek.extend(l2_norm(&convolved[6 + kh * 3..9 + kh * 3]).unwrap());
    }
    same(&q.read().unwrap(), &eq);
    same(&k.read().unwrap(), &ek);
    let a = api.upload(&[100., -100., 0., 10.]).unwrap();
    let b = api.upload(&[-100., 100., 0., 1.]).unwrap();
    let dt = api.upload(&[0., 0., 1., -1.]).unwrap();
    let alog = api.upload(&[-20., 0., 1., 2.]).unwrap();
    let g = api.alloc(4).unwrap();
    let beta = api.alloc(4).unwrap();
    api.run(
        Op::DeltaGates,
        4,
        &[
            g.pointer,
            a.pointer,
            b.pointer,
            dt.pointer,
            alog.pointer,
            beta.pointer,
        ],
        &[],
        &[],
    )
    .unwrap();
    let eg: Vec<_> = [100., -100., 1., 9.]
        .iter()
        .zip([-20_f32, 0., 1., 2.])
        .map(|(&a, al)| -al.exp() * crate::ops::softplus(a))
        .collect();
    let eb: Vec<_> = [-100., 100., 0., 1.]
        .into_iter()
        .map(crate::ops::sigmoid)
        .collect();
    same(&g.read().unwrap(), &eg);
    same(&beta.read().unwrap(), &eb);
    let mut expected: Vec<f32> = (0..24).map(|i| (i as f32 - 12.) / 20.).collect();
    let state = api.upload(&expected).unwrap();
    let core = api.alloc(8).unwrap();
    let mut ec = Vec::new();
    for h in 0..4 {
        ec.extend(
            delta_rule(
                &eq[h * 3..h * 3 + 3],
                &ek[h * 3..h * 3 + 3],
                &convolved[12 + h * 2..14 + h * 2],
                eg[h],
                eb[h],
                &mut expected[h * 6..h * 6 + 6],
            )
            .unwrap(),
        );
    }
    api.run(
        Op::DeltaRule,
        8,
        &[
            core.pointer,
            q.pointer,
            k.pointer,
            values.pointer,
            state.pointer,
            g.pointer,
            beta.pointer,
        ],
        &[3, 2, 2, 4],
        &[],
    )
    .unwrap();
    same(&core.read().unwrap(), &ec);
    same(&state.read().unwrap(), &expected);
}
