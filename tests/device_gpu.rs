//! These tests are intentionally ignored in ordinary CI. A passing CPU bridge
//! test never substitutes for running this file on an actual GPU.
//! Linux + `--features rocm`: HIP bridge. macOS + `--features metal`: Metal bridge.
#![cfg(any(
    all(feature = "rocm", target_os = "linux"),
    all(feature = "metal", target_os = "macos")
))]
use qwen_rust_inference::{
    device::{validate_device, validate_long, DeviceModel, KernelLevel, Kind, WeightFormat},
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
/// Verify pass (ADR 0023): logits for every row of 1..=8 tokens. One row is
/// the decode step bitwise; more rows match the same tokens decoded one at
/// a time within the real-weight tolerance, with the same greedy tokens.
#[test]
#[ignore = "requires pinned local real model AND a ROCm/Metal GPU; no downloads"]
fn device_real_verify() {
    let dir = PathBuf::from(std::env::var_os("QWEN_MODEL_DIR").expect("set QWEN_MODEL_DIR"));
    let model = load_model(&dir).unwrap();
    let device = DeviceModel::load(&model, &library(), KIND).unwrap();
    let vocab = model.config().vocab_size as u32;
    let ids: Vec<u32> = (0..208u32).map(|i| (i * 7919 + 11) % vocab).collect();
    let (prompt, next) = ids.split_at(200);
    let mut run = device.new_run(ids.len()).unwrap();
    run.prefill(prompt).unwrap();
    let cp = run.checkpoint().unwrap();
    let sequential: Vec<Vec<f32>> = next
        .iter()
        .map(|&id| run.step(id, false).unwrap().logits)
        .collect();
    let argmax = |l: &[f32]| (0..l.len()).fold(0, |b, i| if l[i] > l[b] { i } else { b });
    for rows in 1..=next.len() {
        run.restore(&cp).unwrap();
        let out = run.verify(&next[..rows]).unwrap();
        assert_eq!(out.len(), rows);
        if rows == 1 {
            assert_eq!(out[0], sequential[0]);
        }
        for (r, (a, b)) in out.iter().zip(&sequential).enumerate() {
            assert_eq!(a.len(), vocab as usize);
            assert_eq!(argmax(a), argmax(b), "rows {rows} row {r}");
            compare(a, b, 3e-3, 3e-3, &format!("verify rows {rows} row {r}")).unwrap();
        }
    }
    // The verify pass advanced the state like the same steps.
    run.restore(&cp).unwrap();
    run.verify(&next[..4]).unwrap();
    let after = run.step(next[4], false).unwrap().logits;
    compare(&after, &sequential[4], 3e-3, 3e-3, "step after verify").unwrap();
}
/// Rollback of a verify pass (ADR 0026): after `verify_rollback` over 4
/// tokens and `accept(a)`, the DeltaNet state and the next step equal those
/// of the same a tokens decoded one at a time, bitwise (up to 4 rows the
/// verify pass is the decode step row by row). Bridges without
/// `qr_snapshots` must refuse the run (both the HIP and Metal bridges have it; ADR 0029).
#[test]
#[ignore = "requires pinned local real model AND a ROCm/Metal GPU; no downloads"]
fn device_real_rollback() {
    let dir = PathBuf::from(std::env::var_os("QWEN_MODEL_DIR").expect("set QWEN_MODEL_DIR"));
    let model = load_model(&dir).unwrap();
    let device = DeviceModel::load(&model, &library(), KIND).unwrap();
    let vocab = model.config().vocab_size as u32;
    let ids: Vec<u32> = (0..206u32).map(|i| (i * 7919 + 11) % vocab).collect();
    let (prompt, next) = ids.split_at(200);
    let Ok(mut run) = device.new_run_rollback(ids.len(), 3) else {
        assert!(KIND != Kind::Metal, "Metal bridge must support snapshots");
        return;
    };
    run.prefill(prompt).unwrap();
    let cp = run.checkpoint().unwrap();
    for keep in 1..=4 {
        // Reference: the kept tokens one at a time, then the next one.
        run.restore(&cp).unwrap();
        for &id in &next[..keep] {
            run.step(id, false).unwrap();
        }
        let state = run.checkpoint().unwrap();
        let after = run.step(next[keep], false).unwrap().logits;
        // Verify 4 rows, keep `keep` of them.
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
    // Level 1 (reference kernels) stores the snapshots too. Its multi-row
    // attention is the prefill kernel, so compare within tolerance.
    if KIND == Kind::Metal {
        let mut reference = DeviceModel::load(&model, &library(), KIND).unwrap();
        reference.set_kernel_level(KernelLevel::Reference).unwrap();
        let mut r = reference.new_run_rollback(ids.len(), 3).unwrap();
        r.prefill(prompt).unwrap();
        let start = r.checkpoint().unwrap();
        r.step(next[0], false).unwrap();
        r.step(next[1], false).unwrap();
        let expected = r.step(next[2], false).unwrap().logits;
        r.restore(&start).unwrap();
        r.verify_rollback(&next[..4]).unwrap();
        r.accept(2).unwrap();
        let got = r.step(next[2], false).unwrap().logits;
        compare(&got, &expected, 3e-3, 3e-3, "level 1 accept(2)").unwrap();
    }
    // verify_rollback_greedy (ADR 0030): the device argmax of every row is
    // the host greedy pick of the same logits, for 2..=8 rows.
    let greedy = |l: &[f32]| {
        qwen_rust_inference::generate::select(
            l,
            qwen_rust_inference::generate::Sampling::Greedy,
            &mut 0,
        )
        .unwrap()
    };
    let mut wide = device.new_run_rollback(ids.len() + 8, 7).unwrap();
    wide.prefill(prompt).unwrap();
    let start = wide.checkpoint().unwrap();
    let rows: Vec<u32> = (0..8u32).map(|i| (i * 104729 + 5) % vocab).collect();
    for r in 2..=8 {
        wide.restore(&start).unwrap();
        let expected: Vec<u32> = wide
            .verify_rollback(&rows[..r])
            .unwrap()
            .iter()
            .map(|l| greedy(l))
            .collect();
        wide.restore(&start).unwrap();
        assert_eq!(
            wide.verify_rollback_greedy(&rows[..r]).unwrap(),
            expected,
            "{r} rows"
        );
        wide.accept(r).unwrap();
    }
    // accept without a pending pass, or after moving on, is refused.
    run.restore(&cp).unwrap();
    assert!(run.accept(1).is_err());
    run.verify_rollback(&next[..2]).unwrap();
    run.step(next[2], false).unwrap();
    assert!(run.accept(1).is_err());
}
/// Speculative decoding (ADR 0027): MTP drafts (CPU, full vocabulary and a
/// 4096-token subset), verify passes with rollback; the generated IDs equal
/// plain greedy decode on the device, and the hidden rows that the verify
/// passes return equal those of single steps.
#[test]
#[ignore = "requires pinned local real model AND a ROCm/Metal GPU; no downloads"]
fn device_real_speculative() {
    use qwen_rust_inference::{
        device::speculative::{self, Draft},
        generate::{select, Sampling},
        mtp::{frequency_order, DraftVocab},
        weights::load_mtp,
    };
    let dir = PathBuf::from(std::env::var_os("QWEN_MODEL_DIR").expect("set QWEN_MODEL_DIR"));
    let model = load_model(&dir).unwrap();
    let mtp = load_mtp(&dir).unwrap();
    let device = DeviceModel::load(&model, &library(), KIND).unwrap();
    let vocab = model.config().vocab_size as u32;
    let prompt: Vec<u32> = (0..40u32).map(|i| (i * 7919 + 11) % vocab).collect();
    let (max, eos) = (24, [248044u32, 248046]);
    let greedy = |l: &[f32]| select(l, Sampling::Greedy, &mut 0).unwrap();
    let mut plain = device.new_run(prompt.len() + max).unwrap();
    plain.set_capture_hidden(true);
    let mut ids = vec![greedy(&plain.prefill(&prompt).unwrap().unwrap())];
    let prefill_hidden = plain.take_hidden();
    assert_eq!(
        prefill_hidden.len(),
        prompt.len() * model.config().hidden_size
    );
    while ids.len() < max && !eos.contains(ids.last().unwrap()) {
        ids.push(greedy(
            &plain.step(*ids.last().unwrap(), false).unwrap().logits,
        ));
    }
    // step_greedy (ADR 0030, the device argmax) gives the same tokens.
    let mut fast = device.new_run(prompt.len() + max).unwrap();
    let mut again = vec![greedy(&fast.prefill(&prompt).unwrap().unwrap())];
    while again.len() < ids.len() {
        again.push(fast.step_greedy(*again.last().unwrap()).unwrap());
    }
    assert_eq!(again, ids, "step_greedy");
    let Ok(mut probe) = device.new_run_rollback(8, 3) else {
        assert!(KIND != Kind::Metal, "Metal bridge must support snapshots");
        return;
    };
    // Hidden rows of a 4-row verify pass = those of the single steps above.
    probe.set_capture_hidden(true);
    probe.prefill(&prompt[..4]).unwrap();
    let rows = probe.take_hidden();
    let mut single = device.new_run(8).unwrap();
    single.set_capture_hidden(true);
    for &id in &prompt[..4] {
        single.step(id, false).unwrap();
    }
    compare(&rows, &single.take_hidden(), 1e-5, 1e-5, "prefill hidden").unwrap();
    let subset = DraftVocab::new(&model, &frequency_order(vocab as usize, &[])[..4096]).unwrap();
    let head = device.load_mtp(&model, &mtp, &subset).unwrap();
    let capacity = prompt.len() + max + 4;
    let mut stats_by = std::collections::BTreeMap::new();
    for name in ["cpu full", "cpu 4096", "device 4096", "device 4096 unfused"] {
        for depth in [1, 3] {
            let mut run = device.new_run_rollback(capacity, depth).unwrap();
            let mut cpu_full = speculative::CpuDraft::new(&model, &mtp, None);
            let mut cpu_subset = speculative::CpuDraft::new(&model, &mtp, Some(&subset));
            let mut on_device = speculative::DeviceDraft::new(&device, &head, capacity).unwrap();
            on_device.set_fused(name == "device 4096");
            let draft: &mut dyn Draft = match name {
                "cpu full" => &mut cpu_full,
                "cpu 4096" => &mut cpu_subset,
                _ => &mut on_device,
            };
            let (got, stats) =
                speculative::generate(&mut run, draft, &prompt, max, depth, &eos).unwrap();
            assert_eq!(got, ids, "{name}, depth {depth}");
            stats_by.insert((name, depth), (stats.passes, stats.accepted.clone()));
            assert!(
                stats.passes < ids.len(),
                "{name} depth {depth}: no draft accepted"
            );
        }
    }
    // Linking verify, absorb and the chain on the device (ADR 0031) changes
    // no draft: the same passes and acceptances as one sync per forward.
    for depth in [1, 3] {
        assert_eq!(
            stats_by[&("device 4096", depth)],
            stats_by[&("device 4096 unfused", depth)],
            "fused vs unfused, depth {depth}"
        );
    }
    // The device head drafts what the CPU head drafts (same subset; they may
    // differ only at near ties): the drafts from the prompt's last pair.
    let mut cpu = speculative::CpuDraft::new(&model, &mtp, Some(&subset));
    let mut gpu = speculative::DeviceDraft::new(&device, &head, capacity).unwrap();
    let mut next = prompt[1..].to_vec();
    next.push(ids[0]);
    cpu.absorb(&prefill_hidden, &next).unwrap();
    gpu.absorb(&prefill_hidden, &next).unwrap();
    let first = cpu.draft(3).unwrap();
    assert_eq!(gpu.draft(3).unwrap(), first, "device vs CPU drafts");
    // The chain's head positions are taken back: drafting again is the same.
    assert_eq!(gpu.draft(3).unwrap(), first, "draft after rewind");
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
