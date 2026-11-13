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
