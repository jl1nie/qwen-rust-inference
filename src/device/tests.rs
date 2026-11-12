use super::*;
use crate::{tensor::Backend, validation::compare};
use std::{path::PathBuf, process::Command};
fn bridge() -> PathBuf {
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
    let w = DeviceMatrix::upload(&api, &matrix).unwrap();
    let x = api.upload(&[2., -1., 3.]).unwrap();
    let out = api.alloc(2).unwrap();
    w.project(&x, &out).unwrap();
    same(&out.read().unwrap(), &[9., 21.]);
    assert!(out.read_prefix(3).is_err());
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../tests/fixtures/hybrid.json")).unwrap();
    let model = Model::from_fixture(&fixture).unwrap();
    boundary_kernels(&api);
    let device = DeviceModel::upload(&model, api).unwrap();
    let ids: Vec<u32> = serde_json::from_value(fixture["ids"].clone()).unwrap();
    let report = validate_device(&model, &device, &ids, true).unwrap();
    assert_eq!(report["status"], "passed");
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
    let independent = device.new_run(ids.len()).unwrap();
    state_same(&independent.inspect_state().unwrap(), &model.new_state());
    // Invalid multi-token input must not consume a valid prefix.
    gpu.reset().unwrap();
    assert!(gpu.prefill(&[ids[0], u32::MAX]).is_err());
    assert_eq!(gpu.position(), 0);
    // Failed native execution is sticky until reset.
    // Inject a shape failure after the mixer has already changed its state.
    gpu.ffn.len -= 1;
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
