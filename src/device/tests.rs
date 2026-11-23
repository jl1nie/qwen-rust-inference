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
/// Device chat = CPU chat: same IDs, reuse counts and state per turn (with a
/// primed system message, an anchor rewind and a reset to the system state),
/// and snapshots move between the engines in both directions.
#[test]
fn cpu_bridge_chat_matches_cpu_chat_and_snapshots() {
    use crate::{
        chat::ChatSession,
        generate::Sampling,
        snapshot::{load_chat, save_chat, DEFAULT_LIMIT},
        text::TextCodec,
    };
    struct Codec;
    impl TextCodec for Codec {
        fn encode(&self, text: &str) -> Result<Vec<u32>> {
            Ok(text.chars().map(|c| c as u32 % 16).collect())
        }
        fn decode(&self, ids: &[u32]) -> Result<String> {
            Ok(ids
                .iter()
                .map(|&i| (b'a' + (i % 19) as u8) as char)
                .collect())
        }
    }
    let mut fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../tests/fixtures/hybrid.json")).unwrap();
    fixture["config"]["max_position_embeddings"] = 4096.into();
    let model = Model::from_fixture(&fixture).unwrap();
    let api = Api::open(&bridge(), Kind::CpuTest).unwrap();
    let device = DeviceModel::upload(&model, api, WeightFormat::Bf16Exact).unwrap();
    let t = Codec;
    let system = Some("system prompt".to_string());
    let mut cpu = ChatSession::new(
        &model,
        Backend::Vector,
        Sampling::Greedy,
        false,
        system.clone(),
    )
    .unwrap();
    let mut dev = DeviceChat::new(
        &model,
        device.new_run_chunked(600, 7).unwrap(),
        Sampling::Greedy,
        false,
        system,
    )
    .unwrap();
    assert_eq!(cpu.prime(&model, &t).unwrap(), dev.prime(&t).unwrap());
    let check = |cpu: &ChatSession, dev: &DeviceChat| {
        let host = dev.to_session().unwrap();
        assert_eq!(host.engine.tokens, cpu.engine.tokens);
        assert_eq!(host.messages, cpu.messages);
        state_same(&host.engine.state, &cpu.engine.state);
        assert_eq!(host.anchor.is_some(), cpu.anchor.is_some());
        if let (Some(a), Some(b)) = (&host.anchor, &cpu.anchor) {
            assert_eq!(a.tokens, b.tokens);
            state_same(&a.state, &b.state);
        }
    };
    let turn = |cpu: &mut ChatSession, dev: &mut DeviceChat, p: &str| {
        let a = cpu.user_turn(&model, &t, p, 5).unwrap();
        let b = dev.user_turn(&t, p, 5, true).unwrap();
        assert_eq!(a.generated, b.generated, "{p}");
        assert_eq!(
            (a.reused_tokens, a.prefill_tokens),
            (b.reused_tokens, b.prefill_tokens)
        );
        a.reused_tokens
    };
    for p in ["first", "second", "third"] {
        assert!(turn(&mut cpu, &mut dev, p) > 0); // system prefix, then anchors
        check(&cpu, &dev);
    }
    let a = cpu.continue_turn(&model, &t, 3).unwrap();
    assert_eq!(a.generated, dev.continue_turn(&t, 3).unwrap().generated);
    // A turn beyond capacity fails before changing the chat.
    let before = dev.tokens().to_vec();
    assert!(dev.user_turn(&t, &"x".repeat(700), 3, true).is_err());
    assert_eq!(dev.tokens(), before);
    check(&cpu, &dev);
    // Device -> file -> CPU, and CPU -> file -> device.
    let root = std::env::temp_dir().join(format!("qwen-device-chat-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir(&root).unwrap();
    dev.save(&model, &root.join("dev.qrs"), DEFAULT_LIMIT)
        .unwrap();
    let mut resumed = load_chat(&model, &t, &root.join("dev.qrs"), DEFAULT_LIMIT).unwrap();
    assert_eq!(resumed.engine.tokens, cpu.engine.tokens);
    let a = resumed.user_turn(&model, &t, "fourth", 4).unwrap();
    let b = cpu.user_turn(&model, &t, "fourth", 4).unwrap();
    assert_eq!(a.generated, b.generated);
    save_chat(&model, &cpu, &root.join("cpu.qrs"), DEFAULT_LIMIT).unwrap();
    let mut dev = DeviceChat::load(
        &model,
        dev.into_run(),
        &t,
        &root.join("cpu.qrs"),
        DEFAULT_LIMIT,
    )
    .unwrap();
    check(&cpu, &dev);
    turn(&mut cpu, &mut dev, "fifth");
    check(&cpu, &dev);
    // Reset rewinds to the primed system state on both engines.
    cpu.reset(&model).unwrap();
    cpu.prime(&model, &t).unwrap();
    dev.reset().unwrap();
    dev.prime(&t).unwrap();
    assert!(turn(&mut cpu, &mut dev, "again") > 0);
    check(&cpu, &dev);
    // A primed-only snapshot (the system-document cache) loads on both.
    let mut fresh = DeviceChat::new(
        &model,
        device.new_run_chunked(600, 7).unwrap(),
        Sampling::Greedy,
        false,
        Some("system prompt".into()),
    )
    .unwrap();
    fresh.prime(&t).unwrap();
    fresh
        .save(&model, &root.join("primed.qrs"), DEFAULT_LIMIT)
        .unwrap();
    let mut cpu = load_chat(&model, &t, &root.join("primed.qrs"), DEFAULT_LIMIT).unwrap();
    let mut dev = DeviceChat::load(
        &model,
        fresh.into_run(),
        &t,
        &root.join("primed.qrs"),
        DEFAULT_LIMIT,
    )
    .unwrap();
    assert!(turn(&mut cpu, &mut dev, "question") > 0);
    check(&cpu, &dev);
    std::fs::remove_dir_all(root).unwrap();
}
/// Edits and traces during a chat: the device follows the CPU chat with the
/// same edit, an edit change recomputes the history on both engines (and
/// removing it restores the unedited state), edited snapshots carry the
/// edit across engines, and observing positions does not change the result.
#[test]
fn cpu_bridge_chat_edits_and_observer_match_cpu() {
    use crate::{
        chat::ChatSession,
        edit::{Edit, EditManifest},
        generate::Sampling,
        snapshot::{load_chat, save_chat, DEFAULT_LIMIT},
        text::TextCodec,
    };
    use std::{cell::RefCell, collections::BTreeSet, rc::Rc};
    struct Codec;
    impl TextCodec for Codec {
        fn encode(&self, text: &str) -> Result<Vec<u32>> {
            Ok(text.chars().map(|c| c as u32 % 16).collect())
        }
        fn decode(&self, ids: &[u32]) -> Result<String> {
            Ok(ids
                .iter()
                .map(|&i| (b'a' + (i % 19) as u8) as char)
                .collect())
        }
    }
    let mut fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../tests/fixtures/hybrid.json")).unwrap();
    fixture["config"]["max_position_embeddings"] = 4096.into();
    let model = Model::from_fixture(&fixture).unwrap();
    let api = Api::open(&bridge(), Kind::CpuTest).unwrap();
    let device = DeviceModel::upload(&model, api, WeightFormat::Bf16Exact).unwrap();
    let hidden = model.config().hidden_size;
    let mut direction = vec![0.; hidden];
    direction[1] = 1.;
    let manifest = EditManifest {
        model_revision: model.revision().to_string(),
        edits: vec![Edit {
            layer: 1,
            tensor_name: "model.language_model.layers.1.mlp.down_proj.weight".into(),
            columns: vec![0, 2, 3],
            direction: Some(direction),
            coefficient: 0.9,
        }],
    };
    let t = Codec;
    let system = Some("system prompt".to_string());
    let new_cpu = || {
        ChatSession::new(
            &model,
            Backend::Vector,
            Sampling::Greedy,
            false,
            system.clone(),
        )
        .unwrap()
    };
    let mut cpu = new_cpu();
    let mut plain = new_cpu();
    let mut dev = DeviceChat::new(
        &model,
        device.new_run_chunked(600, 7).unwrap(),
        Sampling::Greedy,
        false,
        system.clone(),
    )
    .unwrap();
    let seen = Rc::new(RefCell::new(Vec::new()));
    let log = seen.clone();
    let layers = model.blocks().len();
    dev.set_observer(
        BTreeSet::from([3, 20, 40]),
        Some(Box::new(move |p, id, step| {
            assert_eq!(step.layers.len(), layers); // captured: every layer output
            log.borrow_mut().push((p, id));
        })),
    );
    for c in [&mut cpu, &mut plain] {
        c.prime(&model, &t).unwrap();
    }
    dev.prime(&t).unwrap();
    cpu.set_edit(&model, Some(manifest.clone())).unwrap();
    dev.set_edit(&model, Some(&manifest)).unwrap();
    let same = |cpu: &ChatSession, dev: &DeviceChat| {
        let host = dev.to_session().unwrap();
        assert_eq!(host.engine.tokens, cpu.engine.tokens);
        assert_eq!(host.engine.edit, cpu.engine.edit);
        state_same(&host.engine.state, &cpu.engine.state);
        if let (Some(a), Some(b)) = (&host.anchor, &cpu.anchor) {
            state_same(&a.state, &b.state);
        }
    };
    let mut edited_differs = false;
    for p in ["first", "second"] {
        let a = cpu.user_turn(&model, &t, p, 5).unwrap();
        let b = dev.user_turn(&t, p, 5, true).unwrap();
        let c = plain.user_turn(&model, &t, p, 5).unwrap();
        assert_eq!(a.generated, b.generated);
        edited_differs |= a.generated != c.generated || cpu.engine.state != plain.engine.state;
        same(&cpu, &dev);
    }
    assert!(edited_differs);
    // Every observed position computed so far was reported (some twice:
    // the edit change and the anchor rewind recompute them).
    let positions: BTreeSet<usize> = seen.borrow().iter().map(|&(p, _)| p).collect();
    assert_eq!(positions, BTreeSet::from([3, 20, 40]));
    // Removing the edit recomputes the history: equal to the unedited chat
    // (same tokens; the edit only changes the numbers, not the IDs fed).
    let mut cpu_off = cpu.clone();
    cpu_off.set_edit(&model, None).unwrap();
    dev.set_edit(&model, None).unwrap();
    same(&cpu_off, &dev);
    // Edited snapshots resume with their edit on the other engine.
    dev.set_edit(&model, Some(&manifest)).unwrap();
    same(&cpu, &dev);
    let root = std::env::temp_dir().join(format!("qwen-device-edit-chat-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir(&root).unwrap();
    dev.save(&model, &root.join("dev.qrs"), DEFAULT_LIMIT)
        .unwrap();
    let mut resumed = load_chat(&model, &t, &root.join("dev.qrs"), DEFAULT_LIMIT).unwrap();
    assert_eq!(resumed.engine.edit.as_ref(), Some(&manifest));
    let a = resumed.user_turn(&model, &t, "third", 4).unwrap();
    let b = cpu.user_turn(&model, &t, "third", 4).unwrap();
    assert_eq!(a.generated, b.generated);
    save_chat(&model, &cpu, &root.join("cpu.qrs"), DEFAULT_LIMIT).unwrap();
    let c = load_chat(&model, &t, &root.join("cpu.qrs"), DEFAULT_LIMIT).unwrap();
    dev.replace(&model, c).unwrap();
    assert_eq!(dev.edit(), Some(&manifest));
    assert_eq!(dev.observed(), &BTreeSet::from([3, 20, 40]));
    same(&cpu, &dev);
    std::fs::remove_dir_all(root).unwrap();
}
/// Host BF16 weights used in place (qr_wrap, as on Metal) give exactly the
/// results of the copied upload, single token and batched, and match the
/// CPU projection; the host memory outlives the device buffer.
#[test]
fn cpu_bridge_shared_bf16_weights_match_copy() {
    let (rows, cols) = (13, 70); // odd sizes: a padded last word
    let bits: Vec<u16> = (0..rows * cols)
        .map(|i| (0x3c00 + (i * 37 % 0x300)) as u16 ^ (((i % 3) as u16) << 15))
        .collect();
    let m = Matrix::from_bf16(rows, cols, bits.clone()).unwrap();
    let api = Api::open(&bridge(), Kind::CpuTest).unwrap();
    assert!(api.wrap.is_some(), "CPU test bridge exports qr_wrap");
    let shared = DeviceMatrix::upload(&api, &m, WeightFormat::Bf16Exact).unwrap();
    assert!(shared.shared.is_some() && shared.bf16);
    let copied = DeviceMatrix {
        buffer: api.upload_bf16_bits(&bits).unwrap(),
        rows,
        cols,
        bf16: true,
        int8: false,
        shared: None,
    };
    for tokens in [1, 3] {
        let x: Vec<f32> = (0..tokens * cols)
            .map(|i| (i as f32 * 0.21).cos())
            .collect();
        let input = api.upload(&x).unwrap();
        let (a, b) = (
            api.alloc(tokens * rows).unwrap(),
            api.alloc(tokens * rows).unwrap(),
        );
        shared.project(&input, &a, tokens).unwrap();
        copied.project(&input, &b, tokens).unwrap();
        let (a, b) = (a.read().unwrap(), b.read().unwrap());
        assert!(a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits()));
        for t in 0..tokens {
            let cpu = m
                .project(&x[t * cols..(t + 1) * cols], Backend::Vector)
                .unwrap();
            same(&a[t * rows..(t + 1) * rows], &cpu);
        }
    }
    drop(m); // the device matrix still holds the host allocation
    let x = api.upload(&vec![1.; cols]).unwrap();
    let out = api.alloc(rows).unwrap();
    shared.project(&x, &out, 1).unwrap();
    assert_eq!(out.read().unwrap().len(), rows);
}
/// The CPU test bridge runs only the FP32 reference bodies: asking it for the
/// 16-bit attention must fail, never silently compute FP32 (ADR 0014).
#[test]
fn attention_f16_is_rejected_by_the_cpu_bridge() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../tests/fixtures/hybrid.json")).unwrap();
    let model = Model::from_fixture(&fixture).unwrap();
    let api = Api::open(&bridge(), Kind::CpuTest).unwrap();
    let mut device = DeviceModel::upload(&model, api, WeightFormat::Bf16Exact).unwrap();
    let error = device.set_attention_f16(true).unwrap_err();
    assert!(error.to_string().contains("attention_f16"), "{error}");
    assert_eq!(device.attention_precision(), "f32");
    device.set_attention_f16(false).unwrap();
}
/// Metal-only options name themselves when another bridge refuses them, and
/// reports carry no Metal kernel level for that bridge.
#[test]
fn metal_only_options_are_rejected_by_name() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../tests/fixtures/hybrid.json")).unwrap();
    let model = Model::from_fixture(&fixture).unwrap();
    let api = Api::open(&bridge(), Kind::CpuTest).unwrap();
    let mut device = DeviceModel::upload(&model, api, WeightFormat::Bf16Exact).unwrap();
    let error = device.set_matmul_relaxed(true).unwrap_err().to_string();
    assert!(error.contains("--matmul-relaxed is Metal only"), "{error}");
    assert_eq!(device.matmul_precision(), "f32");
    let error = device
        .set_kernel_level(KernelLevel::Reference)
        .unwrap_err()
        .to_string();
    assert!(error.contains("--metal-kernels is Metal only"), "{error}");
    assert_eq!(device.kernel_level_name(), None);
}
/// The bridge's decode fusions (#19) compute exactly what the separate calls
/// compute: grouped projections, the FFN gate/up/SiLU product, and the fused
/// ops, single token and batched; and a whole forward with every fusion
/// switched off (the path of bridges without them, such as Metal) gives the
/// same logits and states bit for bit.
#[test]
fn cpu_bridge_fusions_match_separate_calls() {
    let api = Api::open(&bridge(), Kind::CpuTest).unwrap();
    assert!(api.gemm_bf16_group.is_some() && api.gemm_bf16_glu.is_some());
    assert!(api.delta_mixer.is_some() && api.gemm_bf16_add_norm.is_some());
    for op in [
        Op::AddRmsNorm,
        Op::RmsNormGate,
        Op::DeltaRuleGated,
        Op::AttentionQueryKv,
    ] {
        assert!(api.runs(op), "op {}", op as i32);
    }
    let cols = 40;
    let matrix = |rows: usize, seed: usize| {
        let bits: Vec<u16> = (0..rows * cols)
            .map(|i| {
                (0x3c00 + ((i * 37 + seed * 11) % 0x300)) as u16 ^ ((((i + seed) % 3) as u16) << 15)
            })
            .collect();
        DeviceMatrix::upload(
            &api,
            &Matrix::from_bf16(rows, cols, bits).unwrap(),
            WeightFormat::Bf16Exact,
        )
        .unwrap()
    };
    let (m1, m2, m3) = (matrix(9, 1), matrix(16, 2), matrix(9, 3));
    for tokens in [1, 3] {
        let x: Vec<f32> = (0..tokens * cols)
            .map(|i| (i as f32 * 0.37).sin())
            .collect();
        let input = api.upload(&x).unwrap();
        let outs: Vec<Buffer> = [&m1, &m2, &m3]
            .iter()
            .map(|m| api.alloc(tokens * m.rows).unwrap())
            .collect();
        project_all(
            &input,
            &[(&m1, &outs[0]), (&m2, &outs[1]), (&m3, &outs[2])],
            tokens,
        )
        .unwrap();
        for (m, grouped) in [&m1, &m2, &m3].iter().zip(&outs) {
            let single = api.alloc(tokens * m.rows).unwrap();
            m.project(&input, &single, tokens).unwrap();
            assert_eq!(
                grouped.read().unwrap(),
                single.read().unwrap(),
                "group, {tokens} tokens"
            );
        }
        // GLU of m1 (gate) and m3 (up) against two projections and SiluProduct.
        let n = tokens * m1.rows;
        let (product, gate, up, expected) = (
            api.alloc(n).unwrap(),
            api.alloc(n).unwrap(),
            api.alloc(n).unwrap(),
            api.alloc(n).unwrap(),
        );
        let glu = api.gemm_bf16_glu.unwrap();
        // SAFETY: distinct live buffers of tokens * rows floats; x holds tokens * cols.
        api.check(unsafe {
            glu(
                api.context,
                product.pointer,
                gate.pointer,
                up.pointer,
                m1.buffer.pointer,
                m3.buffer.pointer,
                input.pointer,
                m1.rows as i32,
                cols as i32,
                tokens as i32,
            )
        })
        .unwrap();
        m1.project(&input, &gate, tokens).unwrap();
        m3.project(&input, &up, tokens).unwrap();
        api.run(
            Op::SiluProduct,
            n,
            &[expected.pointer, gate.pointer, up.pointer],
            &[],
            &[],
        )
        .unwrap();
        assert_eq!(
            product.read().unwrap(),
            expected.read().unwrap(),
            "glu, {tokens} tokens"
        );
    }
    // Fused ops against their pairs.
    let (d, groups) = (8, 3);
    let values =
        |seed: f32| -> Vec<f32> { (0..d * groups).map(|i| (i as f32 * seed).cos()).collect() };
    let (u, v, z) = (
        api.upload(&values(0.3)).unwrap(),
        api.upload(&values(0.7)).unwrap(),
        api.upload(&values(1.1)).unwrap(),
    );
    let weight = api.upload(&values(0.5)[..d]).unwrap();
    let buffers = || {
        (
            api.alloc(d * groups).unwrap(),
            api.alloc(d * groups).unwrap(),
        )
    };
    let (sum, normed) = buffers();
    api.run(
        Op::AddRmsNorm,
        groups,
        &[
            sum.pointer,
            u.pointer,
            v.pointer,
            normed.pointer,
            weight.pointer,
        ],
        &[d],
        &[1e-6, 1.],
    )
    .unwrap();
    let (sum2, normed2) = buffers();
    api.run(
        Op::Add,
        d * groups,
        &[sum2.pointer, u.pointer, v.pointer],
        &[],
        &[],
    )
    .unwrap();
    api.run(
        Op::RmsNorm,
        groups,
        &[normed2.pointer, sum2.pointer, weight.pointer],
        &[d],
        &[1e-6, 1.],
    )
    .unwrap();
    assert_eq!(sum.read().unwrap(), sum2.read().unwrap());
    assert_eq!(normed.read().unwrap(), normed2.read().unwrap());
    let (gated, _) = buffers();
    api.run(
        Op::RmsNormGate,
        groups,
        &[gated.pointer, u.pointer, weight.pointer, z.pointer],
        &[d],
        &[1e-6, 0.],
    )
    .unwrap();
    let (normed3, gated2) = buffers();
    api.run(
        Op::RmsNorm,
        groups,
        &[normed3.pointer, u.pointer, weight.pointer],
        &[d],
        &[1e-6, 0.],
    )
    .unwrap();
    api.run(
        Op::SiluProduct,
        d * groups,
        &[gated2.pointer, z.pointer, normed3.pointer],
        &[],
        &[],
    )
    .unwrap();
    assert_eq!(gated.read().unwrap(), gated2.read().unwrap());

    // Whole forward: every fusion on vs off.
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../tests/fixtures/hybrid.json")).unwrap();
    let model = Model::from_fixture(&fixture).unwrap();
    let ids: Vec<u32> = serde_json::from_value(fixture["ids"].clone()).unwrap();
    let mut plain = Api::open(&bridge(), Kind::CpuTest).unwrap();
    {
        let api = Rc::get_mut(&mut plain).unwrap();
        api.gemm_bf16_group = None;
        api.gemm_bf16_glu = None;
        api.delta_mixer = None;
        api.gemm_bf16_add_norm = None;
        api.max_op = Op::EmbedBf16 as i32;
    }
    let fused = DeviceModel::upload(
        &model,
        Api::open(&bridge(), Kind::CpuTest).unwrap(),
        WeightFormat::Bf16Exact,
    )
    .unwrap();
    let plain = DeviceModel::upload(&model, plain, WeightFormat::Bf16Exact).unwrap();
    for chunk in [1, 3] {
        let (mut a, mut b) = (
            fused.new_run_chunked(ids.len() + 2, chunk).unwrap(),
            plain.new_run_chunked(ids.len() + 2, chunk).unwrap(),
        );
        assert_eq!(
            a.prefill(&ids).unwrap(),
            b.prefill(&ids).unwrap(),
            "prefill, chunk {chunk}"
        );
        for id in [ids[0], ids[1]] {
            let (x, y) = (a.step(id, true).unwrap(), b.step(id, true).unwrap());
            assert_eq!(x.logits, y.logits);
            assert_eq!(x.operations, y.operations);
        }
    }
}
/// Rollback on the CPU test bridge (ADR 0026, `qr_snapshots`): after a 4-row
/// `verify_rollback` and `accept(keep)`, the state and the next step equal
/// those of the kept tokens decoded one at a time. A run without snapshot
/// room refuses a multi-row rollback pass.
#[test]
fn cpu_bridge_rollback_matches_sequential_decode() {
    let api = Api::open(&bridge(), Kind::CpuTest).unwrap();
    assert!(api.snapshots);
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../tests/fixtures/hybrid.json")).unwrap();
    let model = Model::from_fixture(&fixture).unwrap();
    let device = DeviceModel::upload(&model, api, WeightFormat::Bf16Exact).unwrap();
    let prompt: Vec<u32> = serde_json::from_value(fixture["ids"].clone()).unwrap();
    let next = [5u32, 7, 1, 9, 2];
    let mut run = device
        .new_run_rollback(prompt.len() + next.len(), 3)
        .unwrap();
    run.prefill(&prompt).unwrap();
    let cp = run.checkpoint().unwrap();
    for keep in 1..=4 {
        run.restore(&cp).unwrap();
        for &id in &next[..keep] {
            run.step(id, false).unwrap();
        }
        let state = run.checkpoint().unwrap();
        let after = run.step(next[keep], false).unwrap().logits;
        run.restore(&cp).unwrap();
        run.verify_rollback(&next[..4]).unwrap();
        run.accept(keep).unwrap();
        assert_eq!(run.position(), prompt.len() + keep);
        assert_eq!(
            run.checkpoint().unwrap(),
            state,
            "state after accept({keep})"
        );
        assert_eq!(
            run.step(next[keep], false).unwrap().logits,
            after,
            "step after accept({keep})"
        );
    }
    let mut plain = device.new_run(prompt.len() + next.len()).unwrap();
    plain.prefill(&prompt).unwrap();
    assert!(plain.verify_rollback(&next[..2]).is_err());
}
/// Device greedy picks on the CPU test bridge (ADR 0032, the HIP port of
/// ADR 0030/0031): op Argmax gives `select`'s greedy pick for `step_greedy`
/// and for every row of `verify_rollback_greedy`, and op EmbedBf16Picked
/// embeds the picked rows, so CI runs the device-argmax paths.
#[test]
fn cpu_bridge_argmax_matches_host_select() {
    use crate::generate::{select, Sampling};
    let api = Api::open(&bridge(), Kind::CpuTest).unwrap();
    assert_eq!(api.argmax, 2);
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../tests/fixtures/hybrid.json")).unwrap();
    let model = Model::from_fixture(&fixture).unwrap();
    let h = model.config.hidden_size;
    let device = DeviceModel::upload(&model, api.clone(), WeightFormat::Bf16Exact).unwrap();
    let prompt: Vec<u32> = serde_json::from_value(fixture["ids"].clone()).unwrap();
    let next = [5u32, 7, 1, 9, 2];
    let greedy = |l: &[f32]| select(l, Sampling::Greedy, &mut 0).unwrap();
    let (mut a, mut b) = (
        device.new_run(prompt.len() + next.len()).unwrap(),
        device.new_run(prompt.len() + next.len()).unwrap(),
    );
    a.prefill(&prompt).unwrap();
    b.prefill(&prompt).unwrap();
    for &id in &next {
        assert_eq!(
            a.step_greedy(id).unwrap(),
            greedy(&b.step(id, false).unwrap().logits)
        );
    }
    let mut run = device
        .new_run_rollback(prompt.len() + next.len(), 3)
        .unwrap();
    run.prefill(&prompt).unwrap();
    let cp = run.checkpoint().unwrap();
    for rows in 1..=4 {
        run.restore(&cp).unwrap();
        let logits = run.verify_rollback(&next[..rows]).unwrap();
        run.restore(&cp).unwrap();
        let picks = run.verify_rollback_greedy(&next[..rows]).unwrap();
        assert_eq!(
            picks,
            logits.iter().map(|l| greedy(l)).collect::<Vec<_>>(),
            "rows {rows}"
        );
    }
    // EmbedBf16Picked: row r is embedding row picks[2r].
    let embedding = &device.embedding;
    let picks = api.upload(&[3., 0., 1., 0.]).unwrap();
    let out = api.alloc(2 * h).unwrap();
    api.run(
        Op::EmbedBf16Picked,
        2 * h,
        &[out.pointer, embedding.buffer.pointer, picks.pointer],
        &[0, h],
        &[],
    )
    .unwrap();
    let mut want = Vec::new();
    for id in [3usize, 1] {
        let row = api.alloc(h).unwrap();
        api.run(
            Op::EmbedBf16,
            h,
            &[row.pointer, embedding.buffer.pointer],
            &[id, h],
            &[],
        )
        .unwrap();
        want.extend(row.read().unwrap());
    }
    assert_eq!(out.read().unwrap(), want);
}
/// Rows for op Argmax with known answers: ties in different slices of the
/// two-pass GPU kernel (kArgmaxSlice = 4096), NaN and infinities, a row with
/// no finite value, and lengths around the single-block / two-pass switch.
fn argmax_cases() -> Vec<(usize, Vec<f32>)> {
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 40) as f32 / (1u64 << 24) as f32 * 20. - 10.
    };
    let mut cases = Vec::new();
    for n in [1usize, 4095, 8191, 8192, 8193, 70_000, 248_320] {
        cases.push((n, (0..n).map(|_| next()).collect()));
    }
    // Ties at 300, 5000, 70000 and 248000: the smallest index wins.
    let mut tie: Vec<f32> = (0..248_320).map(|_| next()).collect();
    for i in [248_000, 70_000, 5000, 300] {
        tie[i] = 50.;
    }
    cases.push((248_320, tie));
    // Non-finite values (counted, skipped), including +inf above the maximum.
    let mut odd: Vec<f32> = (0..248_320).map(|_| next()).collect();
    for (k, i) in [7usize, 4096, 4097, 100_000, 248_319].iter().enumerate() {
        odd[*i] = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY][k % 3];
    }
    odd[200_000] = 30.;
    cases.push((248_320, odd));
    cases.push((20_000, vec![f32::NAN; 20_000])); // no finite value
    cases
}
/// Op Argmax on every case, as one row and as three rows at once, against
/// the reference rule (reference.h): [first index of the maximum, non-finite
/// count], index 0xffffffff when no value is finite.
fn check_device_argmax(api: &Rc<Api>) {
    let expected = |v: &[f32]| {
        let (mut best, mut at, mut bad) = (f32::NEG_INFINITY, u32::MAX, 0u32);
        for (i, &y) in v.iter().enumerate() {
            if !y.is_finite() {
                bad += 1;
            } else if y > best {
                best = y;
                at = i as u32;
            }
        }
        [at as f32, bad as f32]
    };
    for (n, row) in argmax_cases() {
        for rows in [1usize, 3] {
            let mut logits = Vec::new();
            let mut want = Vec::new();
            for r in 0..rows {
                // Other rows: the case rotated, so each row has its own answer.
                let shifted: Vec<f32> = (0..n).map(|i| row[(i + r * 1013) % n]).collect();
                want.extend(expected(&shifted));
                logits.extend(shifted);
            }
            // Raw write: `upload` refuses the non-finite values tested here.
            let input = api.alloc(logits.len()).unwrap();
            // SAFETY: the allocation holds exactly logits.len() floats; the bridge syncs.
            api.check(unsafe {
                (api.write)(api.context, input.pointer, logits.as_ptr(), logits.len())
            })
            .unwrap();
            let out = api.alloc(2 * rows).unwrap();
            api.run(Op::Argmax, rows, &[out.pointer, input.pointer], &[n], &[])
                .unwrap();
            assert_eq!(out.read().unwrap(), want, "n {n}, rows {rows}");
        }
    }
}
#[test]
fn cpu_bridge_argmax_edge_cases() {
    let api = Api::open(&bridge(), Kind::CpuTest).unwrap();
    check_device_argmax(&api);
}
/// The same cases on a GPU bridge (QWEN_DEVICE_LIBRARY): the two-pass argmax
/// of the HIP/CUDA bridge must give exactly the reference picks.
#[test]
#[ignore = "requires a GPU bridge in QWEN_DEVICE_LIBRARY"]
fn gpu_bridge_argmax_edge_cases() {
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    let kind = Kind::Cuda;
    #[cfg(all(target_os = "linux", not(feature = "cuda")))]
    let kind = Kind::Rocm;
    #[cfg(target_os = "macos")]
    let kind = Kind::Metal;
    let path: PathBuf = std::env::var_os("QWEN_DEVICE_LIBRARY")
        .expect("set QWEN_DEVICE_LIBRARY")
        .into();
    let api = Api::open(&path, kind).unwrap();
    assert!(api.argmax >= 1, "bridge has no op Argmax");
    check_device_argmax(&api);
}
/// Op DeltaRule with shapes whose value columns leave a warp only partly
/// filled: kd 64 (2 lanes per column) and kd 128 (4 lanes per column) with
/// a few columns, so the column groups past `count` exit before the
/// in-group shuffles. CUDA needs the group's own lane mask there (ADR 0033);
/// the model's shapes and the tiny fixture (kd 4, one lane per column) never
/// reach this case. Returns (core output, state) for each shape.
fn delta_rule_partial_warps(api: &Rc<Api>) -> Vec<(Vec<f32>, Vec<f32>)> {
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move |scale: f32| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * scale
    };
    let mut results = Vec::new();
    // (key_dim, value_dim, heads): 2 x 5 x 2 = 20 and 2 x 3 x 4 = 24 active lanes.
    for (kd, vd, heads) in [(64usize, 5usize, 2usize), (128, 3, 2)] {
        let tokens = 3;
        let stride = 2 * heads * kd + heads * vd;
        let q: Vec<f32> = (0..tokens * heads * kd).map(|_| next(0.2)).collect();
        let k: Vec<f32> = (0..tokens * heads * kd).map(|_| next(0.2)).collect();
        let v: Vec<f32> = (0..tokens * stride).map(|_| next(2.)).collect();
        let state: Vec<f32> = (0..heads * kd * vd).map(|_| next(1.)).collect();
        let decay: Vec<f32> = (0..tokens * heads)
            .map(|_| -0.1 - next(0.1).abs())
            .collect();
        let beta: Vec<f32> = (0..tokens * heads).map(|_| 0.5 + next(0.5)).collect();
        let (q, k, v, s, g, b) = (
            api.upload(&q).unwrap(),
            api.upload(&k).unwrap(),
            api.upload(&v).unwrap(),
            api.upload(&state).unwrap(),
            api.upload(&decay).unwrap(),
            api.upload(&beta).unwrap(),
        );
        let out = api.alloc(tokens * heads * vd).unwrap();
        api.run(
            Op::DeltaRule,
            heads * vd,
            &[
                out.pointer,
                q.pointer,
                k.pointer,
                v.pointer,
                s.pointer,
                g.pointer,
                b.pointer,
            ],
            &[kd, vd, heads, heads, tokens, stride],
            &[],
        )
        .unwrap();
        results.push((out.read().unwrap(), s.read().unwrap()));
    }
    results
}
/// The GPU bridge (QWEN_DEVICE_LIBRARY) against the CPU test bridge on those
/// partly filled warps.
#[test]
#[ignore = "requires a GPU bridge in QWEN_DEVICE_LIBRARY"]
fn gpu_bridge_delta_rule_partial_warps() {
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    let kind = Kind::Cuda;
    #[cfg(all(target_os = "linux", not(feature = "cuda")))]
    let kind = Kind::Rocm;
    #[cfg(target_os = "macos")]
    let kind = Kind::Metal;
    let path: PathBuf = std::env::var_os("QWEN_DEVICE_LIBRARY")
        .expect("set QWEN_DEVICE_LIBRARY")
        .into();
    let gpu = delta_rule_partial_warps(&Api::open(&path, kind).unwrap());
    let cpu = delta_rule_partial_warps(&Api::open(&bridge(), Kind::CpuTest).unwrap());
    for (i, ((go, gs), (co, cs))) in gpu.iter().zip(&cpu).enumerate() {
        compare(go, co, 2e-5, 2e-4, &format!("shape {i} output")).unwrap();
        compare(gs, cs, 2e-5, 2e-4, &format!("shape {i} state")).unwrap();
    }
}
/// A deterministic matrix and inputs for the int8 tests (ADR 0034), with one
/// all-zero row and one row holding a single large value.
fn int8_case(rows: usize, cols: usize, tokens: usize) -> (Matrix, Vec<f32>) {
    let mut seed = 0x1234_5678_9abc_def1u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    };
    let mut w: Vec<f32> = (0..rows * cols).map(|_| next() * 0.2).collect();
    w[..cols].fill(0.);
    if rows > 1 {
        w[cols + 3 % cols] = 3.;
    }
    let x = (0..tokens * cols).map(|_| next()).collect();
    (Matrix::new(rows, cols, w).unwrap(), x)
}
/// The CPU test bridge's int8 projection: y = scale * sum q x, summed in
/// index order, so it is reproduced bit for bit here.
fn int8_reference(m: &Matrix, x: &[f32], tokens: usize) -> Vec<f32> {
    let (rows, cols) = (m.rows(), m.cols());
    let (scales, q) = quantize_int8_rows(&m.values(), rows, cols);
    let mut y = Vec::with_capacity(tokens * rows);
    for t in 0..tokens {
        for r in 0..rows {
            let mut sum = 0f32;
            for j in 0..cols {
                sum += q[r * cols + j] as f32 * x[t * cols + j];
            }
            y.push(scales[r] * sum);
        }
    }
    y
}
#[test]
fn int8_quantization_rows() {
    let (m, _) = int8_case(6, 37, 1);
    let values = m.values();
    let (scales, q) = quantize_int8_rows(&values, 6, 37);
    assert_eq!((scales.len(), q.len()), (6, 6 * 37));
    assert_eq!(scales[0], 0.);
    assert!(
        q[..37].iter().all(|&v| v == 0),
        "an all-zero row stays zero"
    );
    for r in 1..6 {
        let row = &q[r * 37..(r + 1) * 37];
        assert_eq!(
            row.iter().map(|v| v.unsigned_abs()).max(),
            Some(127),
            "row {r}"
        );
        for j in 0..37 {
            let w = values[r * 37 + j];
            let err = (w - scales[r] * q[r * 37 + j] as f32).abs();
            assert!(
                err <= scales[r] * 0.5 * (1. + 1e-6),
                "row {r} col {j}: {err}"
            );
        }
    }
}
#[test]
fn cpu_bridge_int8_projection() {
    let api = Api::open(&bridge(), Kind::CpuTest).unwrap();
    assert!(api.gemm_i8.is_some());
    // Odd columns: the last 32-bit word of the values is padded.
    for (rows, cols) in [(5usize, 37usize), (8, 64)] {
        for tokens in [1, 3, 20] {
            let (m, x) = int8_case(rows, cols, tokens);
            let w = DeviceMatrix::upload(&api, &m, WeightFormat::Int8).unwrap();
            assert!(w.int8 && !w.bf16 && w.bytes() == rows * cols + 4 * rows);
            let xs = api.upload(&x).unwrap();
            let out = api.alloc(tokens * rows).unwrap();
            w.project(&xs, &out, tokens).unwrap();
            assert_eq!(
                out.read().unwrap(),
                int8_reference(&m, &x, tokens),
                "{rows}x{cols}, {tokens} tokens"
            );
        }
    }
    // A model in int8: the projections are int8, the embedding lookup BF16.
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../tests/fixtures/hybrid.json")).unwrap();
    let model = Model::from_fixture(&fixture).unwrap();
    let device = DeviceModel::upload(&model, api.clone(), WeightFormat::Int8).unwrap();
    assert_eq!(device.weight_format(), "int8");
    assert!(device.embedding.bf16 && device.lm_head.as_ref().unwrap().int8);
    let ids: Vec<u32> = serde_json::from_value(fixture["ids"].clone()).unwrap();
    let mut run = device.new_run(ids.len()).unwrap();
    let logits = run.prefill(&ids).unwrap().unwrap();
    assert!(logits.iter().all(|v| v.is_finite()));
}
/// The GPU int8 kernels (decode GEMV, 2-15 token GEMV over two token
/// blocks, tiled prefill GEMM, unaligned fallback) against the CPU test
/// bridge; only the summation order differs.
#[test]
#[ignore = "requires a GPU bridge in QWEN_DEVICE_LIBRARY"]
fn gpu_bridge_int8_projection() {
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    let kind = Kind::Cuda;
    #[cfg(all(target_os = "linux", not(feature = "cuda")))]
    let kind = Kind::Rocm;
    #[cfg(target_os = "macos")]
    let kind = Kind::Metal;
    let path: PathBuf = std::env::var_os("QWEN_DEVICE_LIBRARY")
        .expect("set QWEN_DEVICE_LIBRARY")
        .into();
    let gpu = Api::open(&path, kind).unwrap();
    let cpu = Api::open(&bridge(), Kind::CpuTest).unwrap();
    for (rows, cols) in [(64usize, 1024usize), (100, 3584), (5, 37)] {
        for tokens in [1, 3, 9, 20, 40] {
            let (m, x) = int8_case(rows, cols, tokens);
            let run = |api: &Rc<Api>| {
                let w = DeviceMatrix::upload(api, &m, WeightFormat::Int8).unwrap();
                let xs = api.upload(&x).unwrap();
                let out = api.alloc(tokens * rows).unwrap();
                w.project(&xs, &out, tokens).unwrap();
                out.read().unwrap()
            };
            compare(
                &run(&gpu),
                &run(&cpu),
                1e-5,
                1e-4,
                &format!("int8 {rows}x{cols}, {tokens} tokens"),
            )
            .unwrap();
        }
    }
}
/// The int8 group / GLU / add-norm calls against the separate calls they
/// replace, on any bridge: the same values bit for bit (ADR 0034).
fn check_int8_fusions(api: &Rc<Api>) {
    let (h, inter) = (1024usize, 3584usize);
    let (wa, _) = int8_case(64, h, 1);
    let (wb, x) = int8_case(100, h, 9);
    let up_values: Vec<f32> = wb.values().iter().map(|v| -v * 0.7).collect();
    let (gate, _) = int8_case(inter, h, 1);
    let up = Matrix::new(inter, h, up_values.repeat(36)[..inter * h].to_vec()).unwrap();
    let (down, xd) = int8_case(h, inter, 1);
    let upload = |m: &Matrix| DeviceMatrix::upload(api, m, WeightFormat::Int8).unwrap();
    let (a, b, g, u, d) = (
        upload(&wa),
        upload(&wb),
        upload(&gate),
        upload(&up),
        upload(&down),
    );
    for tokens in [1usize, 3, 9] {
        let xs = api.upload(&x[..tokens * h]).unwrap();
        // Group of two matrices.
        let (ya, yb) = (
            api.alloc(tokens * 64).unwrap(),
            api.alloc(tokens * 100).unwrap(),
        );
        project_all(&xs, &[(&a, &ya), (&b, &yb)], tokens).unwrap();
        let (sa, sb) = (
            api.alloc(tokens * 64).unwrap(),
            api.alloc(tokens * 100).unwrap(),
        );
        a.project(&xs, &sa, tokens).unwrap();
        b.project(&xs, &sb, tokens).unwrap();
        assert_eq!(ya.read().unwrap(), sa.read().unwrap(), "group a, {tokens}");
        assert_eq!(yb.read().unwrap(), sb.read().unwrap(), "group b, {tokens}");
        // GLU against two projections and op SiluProduct.
        let n = tokens * inter;
        let (product, gy, uy) = (
            api.alloc(n).unwrap(),
            api.alloc(n).unwrap(),
            api.alloc(n).unwrap(),
        );
        let glu = api.gemm_i8_glu.unwrap();
        // SAFETY: buffers sized above, on this context.
        api.check(unsafe {
            glu(
                api.context,
                product.pointer,
                gy.pointer,
                uy.pointer,
                g.buffer.pointer,
                u.buffer.pointer,
                xs.pointer,
                inter as i32,
                h as i32,
                tokens as i32,
            )
        })
        .unwrap();
        let (sg, su, sp) = (
            api.alloc(n).unwrap(),
            api.alloc(n).unwrap(),
            api.alloc(n).unwrap(),
        );
        g.project(&xs, &sg, tokens).unwrap();
        u.project(&xs, &su, tokens).unwrap();
        api.run(
            Op::SiluProduct,
            n,
            &[sp.pointer, sg.pointer, su.pointer],
            &[],
            &[],
        )
        .unwrap();
        assert_eq!(product.read().unwrap(), sp.read().unwrap(), "glu, {tokens}");
    }
    // One-token GEMV + Add + RmsNorm against the GEMV and op AddRmsNorm.
    let xs = api.upload(&xd[..inter]).unwrap();
    let resid: Vec<f32> = (0..h).map(|i| (i as f32 * 0.37).sin()).collect();
    let weight: Vec<f32> = (0..h).map(|i| (i as f32 * 0.11).cos() * 0.1).collect();
    let (r, w) = (api.upload(&resid).unwrap(), api.upload(&weight).unwrap());
    let run = |fused: bool| {
        let (y, sum, norm) = (
            api.alloc(h).unwrap(),
            api.alloc(h).unwrap(),
            api.alloc(h).unwrap(),
        );
        if fused {
            let args = AddNormArgs {
                sum: sum.pointer,
                u: y.pointer,
                v: r.pointer,
                weight: w.pointer,
                norm: norm.pointer,
                eps: 1e-6,
                offset: 1.,
            };
            let f = api.gemm_i8_add_norm.unwrap();
            // SAFETY: buffers of h floats on this context; y is read after the GEMV.
            api.check(unsafe {
                f(
                    api.context,
                    y.pointer,
                    d.buffer.pointer,
                    xs.pointer,
                    h as i32,
                    inter as i32,
                    &args,
                )
            })
            .unwrap();
        } else {
            d.project(&xs, &y, 1).unwrap();
            api.run(
                Op::AddRmsNorm,
                1,
                &[sum.pointer, y.pointer, r.pointer, norm.pointer, w.pointer],
                &[h],
                &[1e-6, 1.],
            )
            .unwrap();
        }
        (y.read().unwrap(), sum.read().unwrap(), norm.read().unwrap())
    };
    assert_eq!(run(true), run(false), "add_norm");
}
#[test]
fn cpu_bridge_int8_fusions() {
    check_int8_fusions(&Api::open(&bridge(), Kind::CpuTest).unwrap());
}
#[test]
#[ignore = "requires a GPU bridge in QWEN_DEVICE_LIBRARY"]
fn gpu_bridge_int8_fusions() {
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    let kind = Kind::Cuda;
    #[cfg(all(target_os = "linux", not(feature = "cuda")))]
    let kind = Kind::Rocm;
    #[cfg(target_os = "macos")]
    let kind = Kind::Metal;
    let path: PathBuf = std::env::var_os("QWEN_DEVICE_LIBRARY")
        .expect("set QWEN_DEVICE_LIBRARY")
        .into();
    check_int8_fusions(&Api::open(&path, kind).unwrap());
}
/// Fake quantization grids for the 4-bit study (ADR 0035): values are
/// rounded to the grid and back to f32, so only the rounding error is
/// measured (no 4-bit kernels). `group` 0 = one scale for the whole row.
#[derive(Clone, Copy, Debug)]
enum Grid {
    /// Symmetric: scale = max |w| / (2^(bits-1) - 1), q = round(w / scale).
    Sym { bits: u32, group: usize },
    /// Affine: scale = (max - min) / (2^bits - 1), q = round((w - min) / scale).
    Affine { bits: u32, group: usize },
}
impl Grid {
    fn name(self) -> String {
        let (kind, bits, group) = match self {
            Grid::Sym { bits, group } => ("sym", bits, group),
            Grid::Affine { bits, group } => ("affine", bits, group),
        };
        let g = if group == 0 {
            "row".to_string()
        } else {
            format!("g{group}")
        };
        format!("int{bits} {kind} {g}")
    }
    /// Bits per weight with FP16 scales (and FP16 minimums for affine).
    fn bits_per_weight(self, cols: usize) -> f64 {
        let (bits, group, extra) = match self {
            Grid::Sym { bits, group } => (bits, group, 16.),
            Grid::Affine { bits, group } => (bits, group, 32.),
        };
        bits as f64 + extra / if group == 0 { cols } else { group } as f64
    }
    fn apply(self, m: &Matrix) -> Matrix {
        let cols = m.cols();
        let values = m.values();
        let mut out = Vec::with_capacity(values.len());
        let group = match self {
            Grid::Sym { group, .. } | Grid::Affine { group, .. } => group,
        };
        let size = if group == 0 { cols } else { group };
        assert_eq!(cols % size, 0, "group must divide the row");
        for g in values.chunks_exact(size) {
            match self {
                Grid::Sym { bits, .. } => {
                    let levels = ((1u32 << (bits - 1)) - 1) as f32;
                    let scale = g.iter().fold(0f32, |m, v| m.max(v.abs())) / levels;
                    out.extend(g.iter().map(|&v| {
                        if scale == 0. {
                            0.
                        } else {
                            (v / scale).round_ties_even().clamp(-levels, levels) * scale
                        }
                    }));
                }
                Grid::Affine { bits, .. } => {
                    let levels = ((1u32 << bits) - 1) as f32;
                    let lo = g.iter().fold(f32::INFINITY, |m, &v| m.min(v));
                    let hi = g.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
                    let scale = (hi - lo) / levels;
                    out.extend(g.iter().map(|&v| {
                        if scale == 0. {
                            lo
                        } else {
                            lo + ((v - lo) / scale).round_ties_even().clamp(0., levels) * scale
                        }
                    }));
                }
            }
        }
        Matrix::new(m.rows(), cols, out).unwrap()
    }
}
/// The model with every projection fake-quantized (the conv weights and
/// vectors are not projections and stay). The embedding stays for the lookup.
fn fake_quantized(model: &Model, grid: Grid) -> Model {
    let mut q = model.clone();
    for b in &mut q.blocks {
        match &mut b.mixer {
            MixerWeights::Attention(w) => {
                for m in [&mut w.q, &mut w.k, &mut w.v, &mut w.o] {
                    *m = grid.apply(m);
                }
            }
            MixerWeights::Delta(w) => {
                for m in [&mut w.qkv, &mut w.z, &mut w.a, &mut w.b, &mut w.out] {
                    *m = grid.apply(m);
                }
            }
        }
        let [gate, up, down] = b.ffn.matrices();
        b.ffn = crate::ffn::FfnWeights::new(grid.apply(gate), grid.apply(up), grid.apply(down))
            .unwrap();
    }
    q
}
/// Teacher-forced comparison against BF16 for the quantization studies
/// (ADR 0035, 0036): README.md, a 32-token prompt, 512 positions fed token
/// by token; logits at every position, the residual after every layer at
/// the first and the last position.
struct QualityStudy {
    api: Rc<Api>,
    model: Model,
    ids: Vec<u32>,
    reference: QualityTrace,
}
struct QualityTrace {
    logits: Vec<Vec<f32>>,
    first: Vec<Vec<f32>>,
    last: Vec<Vec<f32>>,
}
const STUDY_PROMPT: usize = 32;
const STUDY_STEPS: usize = 512;
fn study_log_softmax(l: &[f32]) -> Vec<f64> {
    let m = l.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v)) as f64;
    let z = l.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln() + m;
    l.iter().map(|&v| v as f64 - z).collect()
}
impl QualityStudy {
    fn open() -> Self {
        use crate::{text::TextTokenizer, weights::load_model};
        #[cfg(all(target_os = "linux", feature = "cuda"))]
        let kind = Kind::Cuda;
        #[cfg(all(target_os = "linux", not(feature = "cuda")))]
        let kind = Kind::Rocm;
        #[cfg(target_os = "macos")]
        let kind = Kind::Metal;
        let library: PathBuf = std::env::var_os("QWEN_DEVICE_LIBRARY")
            .expect("set QWEN_DEVICE_LIBRARY")
            .into();
        let dir = PathBuf::from(std::env::var_os("QWEN_MODEL_DIR").expect("set QWEN_MODEL_DIR"));
        let model = load_model(&dir).unwrap();
        let ids = TextTokenizer::load(&dir)
            .unwrap()
            .encode(include_str!("../../README.md"))
            .unwrap();
        let api = Api::open(&library, kind).unwrap();
        let device = DeviceModel::upload(&model, api.clone(), WeightFormat::Bf16Exact).unwrap();
        let reference = Self::trace(&device, &ids);
        drop(device);
        Self {
            api,
            model,
            ids,
            reference,
        }
    }
    fn trace(device: &DeviceModel, ids: &[u32]) -> QualityTrace {
        let mut run = device.new_run(STUDY_PROMPT + STUDY_STEPS).unwrap();
        run.prefill(&ids[..STUDY_PROMPT - 1]).unwrap();
        let mut t = QualityTrace {
            logits: Vec::new(),
            first: Vec::new(),
            last: Vec::new(),
        };
        for s in 0..STUDY_STEPS {
            let capture = s == 0 || s == STUDY_STEPS - 1;
            let step = run.step(ids[STUDY_PROMPT - 1 + s], capture).unwrap();
            if s == 0 {
                t.first = step.layers;
            } else if capture {
                t.last = step.layers;
            }
            t.logits.push(step.logits);
        }
        t
    }
    fn perplexity(&self, t: &QualityTrace) -> f64 {
        (t.logits
            .iter()
            .enumerate()
            .map(|(s, l)| -study_log_softmax(l)[self.ids[STUDY_PROMPT + s] as usize])
            .sum::<f64>()
            / STUDY_STEPS as f64)
            .exp()
    }
    /// Runs `qmodel` (rounded projections; the lookup keeps the original
    /// embedding) with `head` as the vocabulary projection, through the FP32
    /// weight path, and compares it with the BF16 reference.
    fn evaluate(
        &self,
        name: &str,
        bits: f64,
        qmodel: &Model,
        head: Option<&Matrix>,
    ) -> serde_json::Value {
        let api = &self.api;
        let mut device = DeviceModel::upload(qmodel, api.clone(), WeightFormat::F32).unwrap();
        device.embedding =
            DeviceMatrix::upload(api, &self.model.embedding, WeightFormat::Bf16Exact).unwrap();
        if let Some(h) = head {
            device.lm_head = Some(DeviceMatrix::upload(api, h, WeightFormat::F32).unwrap());
        }
        let t = Self::trace(&device, &self.ids);
        drop(device);
        let argmax = |l: &[f32]| (0..l.len()).fold(0, |b, i| if l[i] > l[b] { i } else { b });
        let rel = |x: &[Vec<f32>], y: &[Vec<f32>]| -> Vec<f64> {
            x.iter()
                .zip(y)
                .map(|(a, b)| {
                    let d: f64 = a.iter().zip(b).map(|(u, v)| ((u - v) as f64).powi(2)).sum();
                    let n: f64 = a.iter().map(|u| (*u as f64).powi(2)).sum();
                    (d / n).sqrt()
                })
                .collect()
        };
        let steps = STUDY_STEPS;
        let (mut top1, mut kl, mut ranks, mut max_diff) = (0usize, Vec::new(), Vec::new(), 0f32);
        for (la, lb) in self.reference.logits.iter().zip(&t.logits) {
            let (ta, tb) = (argmax(la), argmax(lb));
            top1 += (ta == tb) as usize;
            ranks.push(lb.iter().filter(|&&v| v > lb[ta]).count());
            let (pa, pb) = (study_log_softmax(la), study_log_softmax(lb));
            kl.push(
                pa.iter()
                    .zip(&pb)
                    .map(|(x, y)| x.exp() * (x - y))
                    .sum::<f64>(),
            );
            max_diff = la
                .iter()
                .zip(lb)
                .fold(max_diff, |m, (x, y)| m.max((x - y).abs()));
        }
        ranks.sort_unstable();
        kl.sort_by(f64::total_cmp);
        let (first, last) = (
            rel(&self.reference.first, &t.first),
            rel(&self.reference.last, &t.last),
        );
        let summary = |v: &[f64]| {
            serde_json::json!({"min": v.iter().cloned().fold(f64::INFINITY, f64::min),
                "mean": v.iter().sum::<f64>() / v.len() as f64,
                "max": v.iter().cloned().fold(0., f64::max)})
        };
        let row = serde_json::json!({
            "variant": name, "bits_per_weight_fp16_scales": bits,
            "top1_agreement": top1 as f64 / steps as f64,
            "bf16_top1_rank": {"p99": ranks[steps * 99 / 100], "max": ranks[steps - 1]},
            "kl": {"mean": kl.iter().sum::<f64>() / steps as f64, "median": kl[steps / 2], "max": kl[steps - 1]},
            "max_abs_logit_diff": max_diff,
            "perplexity": self.perplexity(&t), "perplexity_bf16": self.perplexity(&self.reference),
            "layer_relative_l2_first_position": summary(&first),
            "layer_relative_l2_last_position": summary(&last),
            "layer_relative_l2_by_layer": {"first": first, "last": last},
        });
        eprintln!("{}", serde_json::to_string(&row).unwrap());
        row
    }
}
/// 4-bit study (ADR 0035): BF16 against fake-quantized weights on the GPU
/// (QualityStudy). Prints a JSON report; no assertion beyond the control
/// (the report is the measurement).
#[test]
#[ignore = "requires QWEN_MODEL_DIR and a GPU bridge in QWEN_DEVICE_LIBRARY"]
fn gpu_int4_study() {
    let study = QualityStudy::open();
    let model = &study.model;
    // (grid, quantize the vocabulary projection too)
    let variants: Vec<(Option<Grid>, bool)> = vec![
        (None, false),
        (Some(Grid::Sym { bits: 8, group: 0 }), true),
        (Some(Grid::Sym { bits: 4, group: 0 }), true),
        (
            Some(Grid::Sym {
                bits: 4,
                group: 128,
            }),
            true,
        ),
        (Some(Grid::Sym { bits: 4, group: 64 }), true),
        (Some(Grid::Sym { bits: 4, group: 32 }), true),
        (Some(Grid::Affine { bits: 4, group: 64 }), true),
        (Some(Grid::Affine { bits: 4, group: 32 }), true),
        (Some(Grid::Sym { bits: 4, group: 64 }), false),
    ];
    let mut report = Vec::new();
    for (grid, quantize_head) in variants {
        let row = match grid {
            None => study.evaluate("control: f32 path, no rounding", 32., model, None),
            Some(g) => {
                let name = format!(
                    "{}{}",
                    g.name(),
                    if quantize_head {
                        ""
                    } else {
                        ", vocabulary projection BF16"
                    }
                );
                let head = quantize_head.then(|| g.apply(&model.embedding));
                study.evaluate(
                    &name,
                    g.bits_per_weight(1024),
                    &fake_quantized(model, g),
                    head.as_ref(),
                )
            }
        };
        report.push(row);
    }
    let control = report[0]["top1_agreement"].as_f64().unwrap();
    assert!(control >= 0.99, "control top-1 agreement {control}");
    eprintln!("{}", serde_json::to_string_pretty(&report).unwrap());
}
/// GPTQ (Frantar et al. 2022) for the 4-bit study (ADR 0036). Every
/// projection is rounded column by column; the rounding error of a column
/// is spread over the columns not yet rounded, weighted by the inverse
/// Hessian H = X^T X of that projection's calibration inputs, so the outputs
/// X W^T change as little as possible. Blocks of 128 columns: inside a block
/// the update is per column, across blocks one matrix product. Group scales
/// are taken from the current (already updated) weights at each group start.
const GPTQ_BLOCK: usize = 128;
/// Calibration inputs of every projection: per layer [input_norm (mixer
/// projections), post_norm (gate/up), ffn_product (down), mixer_core (o /
/// out)], plus the final norm (vocabulary projection); `n` rows each.
struct Calibration {
    layers: Vec<[Vec<f32>; 4]>,
    head: Vec<f32>,
    n: usize,
}
fn calibrate(study: &QualityStudy, tokens: usize) -> Calibration {
    use crate::text::TextTokenizer;
    let dir = PathBuf::from(std::env::var_os("QWEN_MODEL_DIR").unwrap());
    let text = [
        include_str!("../../spec/learning.md"),
        include_str!("../../spec/metal.md"),
        include_str!("../../spec/rocm.md"),
    ]
    .concat();
    let ids = TextTokenizer::load(&dir).unwrap().encode(&text).unwrap();
    assert!(ids.len() > tokens);
    let device =
        DeviceModel::upload(&study.model, study.api.clone(), WeightFormat::Bf16Exact).unwrap();
    let layers = study.model.blocks.len();
    let mut cal = Calibration {
        layers: (0..layers).map(|_| Default::default()).collect(),
        head: Vec::new(),
        n: 0,
    };
    let mut run = device.new_run(tokens + 1).unwrap();
    run.prefill(&ids[..1]).unwrap();
    for &id in &ids[1..=tokens] {
        let step = run.step(id, true).unwrap();
        for (name, values) in step.operations.iter().chain(&step.projection_inputs) {
            let mut parts = name.split('.');
            let (Some("layer"), Some(l), Some(op)) = (parts.next(), parts.next(), parts.next())
            else {
                continue;
            };
            let slot = match op {
                "input_norm" => 0,
                "post_norm" => 1,
                "ffn_product" => 2,
                "mixer_core" => 3,
                _ => continue,
            };
            cal.layers[l.parse::<usize>().unwrap()][slot].extend(values);
        }
        cal.head.extend(&step.normalized);
        cal.n += 1;
    }
    cal
}
/// H = X^T X for n rows of dimension d (f64).
fn hessian(x: &[f32], n: usize, d: usize) -> Vec<f64> {
    assert_eq!(x.len(), n * d);
    let x: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let mut h = vec![0f64; d * d];
    // SAFETY: x is n x d row-major, h is d x d; the strides describe them.
    unsafe {
        matrixmultiply::dgemm(
            d,
            n,
            d,
            1.,
            x.as_ptr(),
            1,
            d as isize,
            x.as_ptr(),
            d as isize,
            1,
            0.,
            h.as_mut_ptr(),
            d as isize,
            1,
        );
    }
    h
}
/// Lower Cholesky factor of a symmetric positive definite d x d matrix.
fn cholesky(a: &[f64], d: usize) -> Vec<f64> {
    let mut l = vec![0f64; d * d];
    for j in 0..d {
        let (lj, _) = l.split_at(j * d + d);
        let row_j = &lj[j * d..j * d + j];
        let diag = a[j * d + j] - row_j.iter().map(|v| v * v).sum::<f64>();
        assert!(diag > 0., "not positive definite at {j}");
        let ljj = diag.sqrt();
        let row_j = row_j.to_vec();
        l[j * d + j] = ljj;
        for i in j + 1..d {
            let dot: f64 = l[i * d..i * d + j]
                .iter()
                .zip(&row_j)
                .map(|(a, b)| a * b)
                .sum();
            l[i * d + j] = (a[i * d + j] - dot) / ljj;
        }
    }
    l
}
/// GPTQ's U: H gets 1% of its mean diagonal added (dead inputs get 1), then
/// U is the upper Cholesky factor of H^-1 (U^T U = H^-1), as f32.
fn gptq_inverse(mut h: Vec<f64>, d: usize) -> Vec<f32> {
    for i in 0..d {
        if h[i * d + i] == 0. {
            h[i * d + i] = 1.;
        }
    }
    let damp = 0.01 * (0..d).map(|i| h[i * d + i]).sum::<f64>() / d as f64;
    for i in 0..d {
        h[i * d + i] += damp;
    }
    let l = cholesky(&h, d);
    let mut lt = vec![0f64; d * d]; // L^T, so the back substitution reads rows
    for i in 0..d {
        for j in 0..=i {
            lt[j * d + i] = l[i * d + j];
        }
    }
    // H^-1 column by column: L y = e_j, then L^T x = y.
    let mut inv = vec![0f64; d * d];
    let mut y = vec![0f64; d];
    for j in 0..d {
        y.fill(0.);
        for i in j..d {
            let dot: f64 = l[i * d + j..i * d + i]
                .iter()
                .zip(&y[j..i])
                .map(|(a, b)| a * b)
                .sum();
            y[i] = ((i == j) as u8 as f64 - dot) / l[i * d + i];
        }
        for i in (0..d).rev() {
            let dot: f64 = lt[i * d + i + 1..i * d + d]
                .iter()
                .zip(&inv[j * d + i + 1..j * d + d])
                .map(|(a, b)| a * b)
                .sum();
            inv[j * d + i] = (y[i] - dot) / lt[i * d + i];
        }
    }
    // inv holds column j of H^-1 in row j (H^-1 is symmetric).
    let l2 = cholesky(&inv, d);
    let mut u = vec![0f32; d * d];
    for i in 0..d {
        for j in 0..=i {
            u[j * d + i] = l2[i * d + j] as f32;
        }
    }
    u
}
/// GPTQ of one matrix with U from `gptq_inverse`; returns the rounded
/// values (dequantized, f32).
fn gptq(w: &Matrix, u: &[f32], grid: Grid) -> Matrix {
    let (rows, cols) = (w.rows(), w.cols());
    let (bits, group, affine) = match grid {
        Grid::Sym { bits, group } => (bits, group, false),
        Grid::Affine { bits, group } => (bits, group, true),
    };
    let size = if group == 0 { cols } else { group };
    assert!(cols % size == 0 && (size >= cols || GPTQ_BLOCK % size == 0));
    let values = w.values();
    // Column-major working copy: column c is wt[c * rows..].
    let mut wt = vec![0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            wt[c * rows + r] = values[r * cols + c];
        }
    }
    let mut q = vec![0f32; rows * cols];
    let (mut scale, mut lo) = (vec![0f32; rows], vec![0f32; rows]);
    let sym_levels = ((1u32 << (bits - 1)) - 1) as f32;
    let affine_levels = ((1u32 << bits) - 1) as f32;
    let mut err = vec![0f32; GPTQ_BLOCK * rows];
    for i1 in (0..cols).step_by(GPTQ_BLOCK) {
        let i2 = (i1 + GPTQ_BLOCK).min(cols);
        for i in i1..i2 {
            if i % size == 0 {
                for r in 0..rows {
                    let g = (i..i + size).map(|c| wt[c * rows + r]);
                    if affine {
                        let (mn, mx) = g.fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), v| {
                            (a.min(v), b.max(v))
                        });
                        lo[r] = mn;
                        scale[r] = (mx - mn) / affine_levels;
                    } else {
                        scale[r] = g.fold(0f32, |m, v| m.max(v.abs())) / sym_levels;
                    }
                }
            }
            let d = u[i * cols + i];
            let (col, rest) = wt[i * rows..].split_at_mut(rows);
            let e = &mut err[(i - i1) * rows..(i - i1 + 1) * rows];
            for r in 0..rows {
                let v = col[r];
                let s = scale[r];
                let rounded = if s == 0. {
                    if affine {
                        lo[r]
                    } else {
                        0.
                    }
                } else if affine {
                    lo[r] + ((v - lo[r]) / s).round_ties_even().clamp(0., affine_levels) * s
                } else {
                    (v / s).round_ties_even().clamp(-sym_levels, sym_levels) * s
                };
                q[i * rows + r] = rounded;
                e[r] = (v - rounded) / d;
            }
            for j in i + 1..i2 {
                let f = u[i * cols + j];
                let target = &mut rest[(j - i - 1) * rows..(j - i) * rows];
                for (t, &x) in target.iter_mut().zip(e.iter()) {
                    *t -= x * f;
                }
            }
        }
        if i2 < cols {
            // wt[i2..] (columns, each `rows` long) -= U[i1..i2, i2..]^T . err
            let (m, k) = (cols - i2, i2 - i1);
            // SAFETY: U is cols x cols row-major; err is k x rows; wt[i2..] is m x rows.
            unsafe {
                matrixmultiply::sgemm(
                    m,
                    k,
                    rows,
                    -1.,
                    u.as_ptr().add(i1 * cols + i2),
                    1,
                    cols as isize,
                    err.as_ptr(),
                    rows as isize,
                    1,
                    1.,
                    wt.as_mut_ptr().add(i2 * rows),
                    rows as isize,
                    1,
                );
            }
        }
    }
    let mut out = vec![0f32; rows * cols];
    for c in 0..cols {
        for r in 0..rows {
            out[r * cols + c] = q[c * rows + r];
        }
    }
    Matrix::new(rows, cols, out).unwrap()
}
#[test]
fn gptq_matches_round_to_nearest_without_correlations() {
    // With H = I (uncorrelated, equal-power inputs) GPTQ has nothing to
    // compensate: U is diagonal and the result is round-to-nearest.
    let (m, _) = int8_case(6, 256, 1);
    let d = 256;
    let mut h = vec![0f64; d * d];
    for i in 0..d {
        h[i * d + i] = 1.;
    }
    let u = gptq_inverse(h, d);
    for grid in [
        Grid::Sym { bits: 4, group: 64 },
        Grid::Affine { bits: 4, group: 32 },
        Grid::Sym { bits: 4, group: 0 },
    ] {
        let a = gptq(&m, &u, grid);
        let b = grid.apply(&m);
        compare(&a.values(), &b.values(), 1e-6, 1e-5, &grid.name()).unwrap();
    }
}
#[test]
fn gptq_lowers_the_output_error() {
    // Correlated inputs: GPTQ's X W^T error is below round-to-nearest's.
    let (m, _) = int8_case(16, 128, 1);
    let (n, d) = (512usize, 128usize);
    let mut seed = 7u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    };
    let base: Vec<f32> = (0..n * 8).map(|_| next()).collect();
    let x: Vec<f32> = (0..n * d)
        .map(|i| base[(i / d) * 8 + i % 8] + 0.1 * next())
        .collect();
    let u = gptq_inverse(hessian(&x, n, d), d);
    let out_err = |q: &Matrix| {
        let (w, qv) = (m.values(), q.values());
        let mut e = 0f64;
        for t in 0..n {
            for r in 0..16 {
                let s: f64 = (0..d)
                    .map(|c| ((w[r * d + c] - qv[r * d + c]) * x[t * d + c]) as f64)
                    .sum();
                e += s * s;
            }
        }
        e
    };
    let grid = Grid::Sym { bits: 4, group: 32 };
    let (g, r) = (out_err(&gptq(&m, &u, grid)), out_err(&grid.apply(&m)));
    assert!(g < 0.5 * r, "GPTQ {g} vs round-to-nearest {r}");
}
#[test]
fn gptq_inverse_is_the_damped_inverse_factor() {
    // U^T U = (H + damp I)^-1 for a random symmetric positive definite H.
    let (n, d) = (64usize, 24usize);
    let (m, _) = int8_case(n, d, 1);
    let h = hessian(&m.values(), n, d);
    let u = gptq_inverse(h.clone(), d);
    let damp = 0.01 * (0..d).map(|i| h[i * d + i]).sum::<f64>() / d as f64;
    for i in 0..d {
        for j in 0..d {
            // ((U^T U) (H + damp I))[i][j] should be the identity.
            let mut s = 0f64;
            for k in 0..d {
                let utu: f64 = (0..d)
                    .map(|r| u[r * d + i] as f64 * u[r * d + k] as f64)
                    .sum();
                s += utu * (h[k * d + j] + if k == j { damp } else { 0. });
            }
            let want = (i == j) as u8 as f64;
            assert!((s - want).abs() < 1e-4, "({i},{j}) = {s}");
        }
    }
}
/// GPTQ study (ADR 0036): round-to-nearest against GPTQ for the same grids,
/// calibrated on 1024 tokens of spec/{learning,metal,rocm}.md and measured
/// on README.md (QualityStudy). Prints a JSON report.
#[test]
#[ignore = "requires QWEN_MODEL_DIR and a GPU bridge in QWEN_DEVICE_LIBRARY"]
fn gpu_gptq_study() {
    use std::time::Instant;
    let study = QualityStudy::open();
    let model = &study.model;
    let started = Instant::now();
    let cal = calibrate(&study, 1024);
    let n = cal.n;
    let calibrated = started.elapsed().as_secs_f64();
    // U for each distinct projection input, layers in parallel.
    let workers = 8;
    let layer_u: Vec<[Vec<f32>; 4]> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..workers)
            .map(|w| {
                let cal = &cal;
                s.spawn(move || {
                    (w..cal.layers.len())
                        .step_by(workers)
                        .map(|l| {
                            let inputs = &cal.layers[l];
                            (
                                l,
                                inputs.clone().map(|x| {
                                    let d = x.len() / n;
                                    gptq_inverse(hessian(&x, n, d), d)
                                }),
                            )
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        let mut all: Vec<_> = handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect();
        all.sort_by_key(|(l, _)| *l);
        all.into_iter().map(|(_, u)| u).collect()
    });
    let head_u = gptq_inverse(
        hessian(&cal.head, n, cal.head.len() / n),
        cal.head.len() / n,
    );
    let inverted = started.elapsed().as_secs_f64() - calibrated;
    let gptq_model = |grid: Grid| -> (Model, Matrix) {
        let mut q = model.clone();
        let head = std::thread::scope(|s| {
            let head = s.spawn(|| gptq(&model.embedding, &head_u, grid));
            let chunks: Vec<_> = q.blocks.chunks_mut(3).enumerate().collect();
            let handles: Vec<_> = chunks
                .into_iter()
                .map(|(c, blocks)| {
                    let layer_u = &layer_u;
                    s.spawn(move || {
                        for (k, b) in blocks.iter_mut().enumerate() {
                            let u = &layer_u[c * 3 + k];
                            match &mut b.mixer {
                                MixerWeights::Attention(w) => {
                                    for m in [&mut w.q, &mut w.k, &mut w.v] {
                                        *m = gptq(m, &u[0], grid);
                                    }
                                    w.o = gptq(&w.o, &u[3], grid);
                                }
                                MixerWeights::Delta(w) => {
                                    for m in [&mut w.qkv, &mut w.z, &mut w.a, &mut w.b] {
                                        *m = gptq(m, &u[0], grid);
                                    }
                                    w.out = gptq(&w.out, &u[3], grid);
                                }
                            }
                            let [gate, up, down] = b.ffn.matrices();
                            b.ffn = crate::ffn::FfnWeights::new(
                                gptq(gate, &u[1], grid),
                                gptq(up, &u[1], grid),
                                gptq(down, &u[2], grid),
                            )
                            .unwrap();
                        }
                    })
                })
                .collect();
            for h in handles {
                h.join().unwrap();
            }
            head.join().unwrap()
        });
        (q, head)
    };
    let grids = [
        Grid::Sym { bits: 4, group: 0 },
        Grid::Sym {
            bits: 4,
            group: 128,
        },
        Grid::Sym { bits: 4, group: 64 },
        Grid::Affine { bits: 4, group: 64 },
        Grid::Affine { bits: 4, group: 32 },
    ];
    let mut report = vec![serde_json::json!({
        "calibration": {"text": "spec/learning.md + metal.md + rocm.md", "tokens": n,
            "seconds_capture": calibrated, "seconds_hessian_inverse": inverted}
    })];
    for grid in grids {
        let bits = grid.bits_per_weight(1024);
        let rtn_head = grid.apply(&model.embedding);
        report.push(study.evaluate(
            &format!("{} RTN", grid.name()),
            bits,
            &fake_quantized(model, grid),
            Some(&rtn_head),
        ));
        let t = Instant::now();
        let (q, head) = gptq_model(grid);
        let seconds = t.elapsed().as_secs_f64();
        let mut row = study.evaluate(&format!("{} GPTQ", grid.name()), bits, &q, Some(&head));
        row["gptq_seconds"] = serde_json::json!(seconds);
        report.push(row);
    }
    eprintln!("{}", serde_json::to_string_pretty(&report).unwrap());
}
/// Book 3: the last layer's FFN inputs read back during a chunked prefill and
/// single steps (the fused decode tail included) match the CPU training
/// cache (`LayerCache::capture`) at every position.
#[cfg(feature = "train")]
#[test]
fn cpu_bridge_top_inputs_match_cpu_layer_cache() {
    use crate::train::cache::LayerCache;
    let api = Api::open(&bridge(), Kind::CpuTest).unwrap();
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../tests/fixtures/hybrid.json")).unwrap();
    let model = Model::from_fixture(&fixture).unwrap();
    let device = DeviceModel::upload(&model, api, WeightFormat::Bf16Exact).unwrap();
    let mut ids: Vec<u32> = serde_json::from_value(fixture["ids"].clone()).unwrap();
    let prefilled = ids.len() + 5;
    ids.extend([5u32, 7, 1, 9, 2, 3, 8]);
    let cpu = LayerCache::capture(&model, &ids, Backend::Vector).unwrap();
    let mut run = device.new_run_chunked(ids.len(), 4).unwrap();
    run.set_capture_top_inputs(true);
    run.prefill(&ids[..prefilled]).unwrap();
    for &id in &ids[prefilled..] {
        run.step(id, false).unwrap();
    }
    let (after_mixer, post_norm) = run.take_top_inputs();
    same(&after_mixer, &cpu.inputs.after_mixer);
    same(&post_norm, &cpu.inputs.post_norm);
    assert!(run.take_top_inputs().0.is_empty());
}

#[cfg(feature = "train")]
#[test]
fn cpu_bridge_layer_inputs_match_cpu_capture() {
    use crate::train::stack::capture;
    let api = Api::open(&bridge(), Kind::CpuTest).unwrap();
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../tests/fixtures/hybrid.json")).unwrap();
    let model = Model::from_fixture(&fixture).unwrap();
    let device = DeviceModel::upload(&model, api, WeightFormat::Bf16Exact).unwrap();
    let mut ids: Vec<u32> = serde_json::from_value(fixture["ids"].clone()).unwrap();
    let prefilled = ids.len() + 2;
    ids.extend([5u32, 7, 1, 9, 2, 3, 8]);
    for layer in 0..model.blocks.len() {
        let cpu = capture(&model, &ids, layer, Backend::Vector).unwrap();
        let mut run = device.new_run_chunked(ids.len(), 4).unwrap();
        run.set_capture_layer_input(Some(layer));
        run.prefill(&ids[..prefilled]).unwrap();
        for &id in &ids[prefilled..] {
            run.step(id, false).unwrap();
        }
        same(&run.take_layer_input(), &cpu);
        assert!(run.take_layer_input().is_empty());
    }
}
