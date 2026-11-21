//! M4 of book 3: distillation from a teacher's top-K output.
//!
//! The teacher here is the fixture with a changed last FFN. Its predictions
//! are written in Strata's `STRATA_LOGPOS` line format (as
//! `engine/src/core/verify.cpp` prints it), read back as teacher data, and
//! the student is trained toward them.
#![cfg(feature = "train")]
use qwen_rust_inference::model::Model;
use qwen_rust_inference::tensor::Backend;
use qwen_rust_inference::train::cache::LayerCache;
use qwen_rust_inference::train::ffn::{TopFfn, TrainFfn};
use qwen_rust_inference::train::fit::{fit, mean_loss_and_grad, Target};
use qwen_rust_inference::train::grad::{kl_full, log_softmax};
use qwen_rust_inference::train::optim::{Optimizer, OptimizerConfig};
use qwen_rust_inference::train::teacher::{parse_logpos, TeacherData};
mod train_common;
use train_common::fixture;

/// One line as Strata prints it: pos, target, logp(target), argmax,
/// logp(argmax), hit, extra, without, then K `id:logprob` (6 decimals).
fn strata_line(pos: usize, target: u32, logits: &[f32], k: usize) -> String {
    let lp = log_softmax(logits).unwrap();
    let mut order: Vec<usize> = (0..lp.len()).collect();
    order.sort_by(|&a, &b| lp[b].total_cmp(&lp[a]));
    let top = order[0];
    let mut s = format!(
        "{pos}\t{target}\t{:.9}\t{top}\t{:.9}\t{}\tnan\tnan",
        lp[target as usize],
        lp[top],
        u8::from(top == target as usize)
    );
    for &i in &order[..k] {
        s += &format!("\t{i}:{:.6}", lp[i]);
    }
    s
}

/// A longer token sequence than the fixture's own, for more positions.
fn sequence() -> Vec<u32> {
    vec![1, 4, 2, 8, 3, 5, 7, 11, 6, 9, 13, 2, 4, 10]
}

/// The teacher: the fixture with its last FFN's down matrix changed.
fn teacher_model(model: &Model) -> Model {
    let last = model.blocks().len() - 1;
    let mut ffn = TrainFfn::from_weights(&model.blocks()[last].ffn);
    for (i, v) in ffn.down.values_mut().iter_mut().enumerate() {
        *v += ((i * 7919 % 13) as f32 - 6.) * 0.05;
    }
    model.with_ffn(last, ffn.to_weights().unwrap()).unwrap()
}

/// Strata-format lines of `teacher` over `ids`, top `k`.
fn logpos(teacher: &Model, ids: &[u32], k: usize) -> (String, Vec<Vec<f32>>) {
    let mut state = teacher.new_state();
    let mut text = String::new();
    let mut all = Vec::new();
    for (p, &id) in ids.iter().enumerate() {
        let logits = teacher
            .step(id, &mut state, Backend::Vector, None)
            .unwrap()
            .logits;
        if p + 1 < ids.len() {
            text += &strata_line(p, ids[p + 1], &logits, k);
            text += "\n";
        }
        all.push(logits);
    }
    (text, all)
}

#[test]
fn logpos_lines_parse_and_renormalize_over_the_shared_ids() {
    // Vocabulary of 6, of which the first 4 are shared. Teacher mass:
    // id 0: 0.5, id 5 (not shared): 0.2, id 1: 0.1, target id 3: 0.05.
    let lp = |p: f64| p.ln();
    let line = format!(
        "7\t3\t{:.9}\t0\t{:.9}\t0\tnan\tnan\t0:{:.6}\t5:{:.6}\t1:{:.6}",
        lp(0.05),
        lp(0.5),
        lp(0.5),
        lp(0.2),
        lp(0.1)
    );
    let lines = parse_logpos(&format!("{line}\n\n")).unwrap();
    assert_eq!(lines.len(), 1);
    assert_eq!((lines[0].position, lines[0].target), (7, 3));
    assert_eq!(lines[0].top.len(), 3);

    let ids = [9, 9, 9, 9, 9, 9, 9, 9, 3];
    let d = TeacherData::from_logpos(&lines, &ids, 4, "test").unwrap();
    let p = &d.positions[0];
    // id 5 dropped, the target (outside the top K) added; renormalized by 0.8.
    assert_eq!(p.ids, [0, 1, 3]);
    let probs: Vec<f64> = p.log_p.iter().map(|l| l.exp()).collect();
    for (got, want) in probs.iter().zip([0.5 / 0.8, 0.1 / 0.8, 0.05 / 0.8]) {
        assert!((got - want).abs() < 1e-6, "{probs:?}");
    }
    let Target::Sparse(t) = &d.examples().unwrap()[0].target else {
        panic!()
    };
    assert_eq!(t.vocab(), 4);
    assert!((t.other() - (1. - 0.65 / 0.8)).abs() < 1e-6);
    assert!((probs.iter().sum::<f64>() + t.other() - 1.).abs() < 1e-9);

    // The line must score the ID that follows its position.
    let wrong = [9, 9, 9, 9, 9, 9, 9, 9, 2];
    let e = TeacherData::from_logpos(&lines, &wrong, 4, "test").unwrap_err();
    assert!(e.0.contains("not the same token sequence"), "{e}");
    assert!(TeacherData::from_logpos(&lines, &ids[..8], 4, "test").is_err()); // past the end
    let twice = [lines[0].clone(), lines[0].clone()];
    assert!(TeacherData::from_logpos(&twice, &ids, 4, "test").is_err());
    for bad in [
        "1\t2\t-0.1",
        "1\t2\t-0.1\t0\t-0.1\t0\tnan\tnan\t3",
        "1\t2\t0.5\t0\t-0.1\t0\tnan\tnan",
    ] {
        assert!(parse_logpos(bad).is_err(), "{bad}");
    }
}

#[test]
fn full_top_k_gives_the_full_kl() {
    let (model, _) = fixture();
    let ids = sequence();
    let teacher = teacher_model(&model);
    let vocab = model.config().vocab_size;
    let (text, teacher_logits) = logpos(&teacher, &ids, vocab);
    let d =
        TeacherData::from_logpos(&parse_logpos(&text).unwrap(), &ids, vocab, "fixture").unwrap();
    let cache = LayerCache::capture(&model, &ids, Backend::Vector).unwrap();
    let student = TopFfn::new(&model).unwrap();
    let f = student.forward(&cache.inputs).unwrap();
    for e in d.examples().unwrap() {
        let p = e.position;
        let s = &f.head.logits[p * vocab..(p + 1) * vocab];
        let sparse = e.target.loss(s).unwrap();
        let full = kl_full(&log_softmax(&teacher_logits[p]).unwrap(), s).unwrap();
        // The printed log-probabilities have 6 decimals.
        assert!(
            (sparse.loss - full.loss).abs() < 1e-5,
            "{p}: {} vs {}",
            sparse.loss,
            full.loss
        );
    }
}

#[test]
fn distillation_moves_the_student_toward_the_teacher() {
    let (model, _) = fixture();
    let ids = sequence();
    let teacher = teacher_model(&model);
    // Top 5 of 19, and only the first 17 IDs shared: the truncation and
    // the "other" bucket are both in play.
    let (text, _) = logpos(&teacher, &ids, 5);
    let d = TeacherData::from_logpos(&parse_logpos(&text).unwrap(), &ids, 17, "fixture").unwrap();
    let cache = LayerCache::capture(&model, &ids, Backend::Vector).unwrap();
    d.check_ids(&cache.header.ids).unwrap();
    let examples = d.examples().unwrap();
    assert_eq!(examples.len(), ids.len() - 1);

    // The student is compared over the same 17 IDs, renormalized there:
    // KL(p‖q) with q = softmax(logits[..17]), from the definition; the two
    // rows past them get no gradient.
    let f = TopFfn::new(&model).unwrap().forward(&cache.inputs).unwrap();
    let vocab = model.config().vocab_size;
    for (e, t) in examples.iter().zip(&d.positions) {
        let logits = &f.head.logits[e.position * vocab..(e.position + 1) * vocab];
        let g = e.target.loss(logits).unwrap();
        assert_eq!(g.dlogits.len(), vocab);
        assert!(g.dlogits[17..].iter().all(|&v| v == 0.));
        let lq = log_softmax(&logits[..17]).unwrap();
        let mut kl = 0.;
        let mut q_k = 0.;
        let mut p_k = 0.;
        for (&id, &lp) in t.ids.iter().zip(&t.log_p) {
            kl += lp.exp() * (lp - lq[id as usize]);
            q_k += lq[id as usize].exp();
            p_k += lp.exp();
        }
        let (p_o, q_o) = (1. - p_k, 1. - q_k);
        kl += p_o * (p_o.ln() - q_o.ln());
        assert!((g.loss - kl).abs() < 1e-9, "{} vs {kl}", g.loss);
    }

    for lora in [false, true] {
        let mut top = TopFfn::new(&model).unwrap();
        if lora {
            top.ffn = top.ffn.with_lora(4, 8., &[0, 1, 2], 1).unwrap();
        }
        let kl = |top: &TopFfn| {
            mean_loss_and_grad(top, &cache.inputs, &examples, 4)
                .unwrap()
                .0
        };
        let before = kl(&top);
        let mut o = Optimizer::new(OptimizerConfig::adam(0.02), &top.ffn.sizes());
        fit(
            &mut top,
            &cache.inputs,
            &examples,
            &mut o,
            60,
            4,
            |_| Ok(()),
        )
        .unwrap();
        let after = kl(&top);
        assert!(after < 0.3 * before, "lora={lora}: {before} -> {after}");
    }
}

#[test]
fn teacher_file_round_trip_and_mismatch() {
    let (model, _) = fixture();
    let ids = sequence();
    let (text, _) = logpos(&teacher_model(&model), &ids, 5);
    let d = TeacherData::from_logpos(&parse_logpos(&text).unwrap(), &ids, 17, "fixture").unwrap();
    let dir = std::env::temp_dir().join(format!("qwen-train-teacher-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("teacher.json");
    d.write(&path).unwrap();
    // serde_json reads f64 back to within the last bit (not bit-exactly
    // without its float_roundtrip feature); training reads the file each
    // time, so runs from the same file agree.
    let back = TeacherData::read(&path).unwrap();
    assert_eq!(
        (&back.ids, back.vocab, &back.teacher),
        (&d.ids, d.vocab, &d.teacher)
    );
    for (a, b) in back.positions.iter().zip(&d.positions) {
        assert_eq!((a.position, &a.ids), (b.position, &b.ids));
        assert!(a
            .log_p
            .iter()
            .zip(&b.log_p)
            .all(|(x, y)| (x - y).abs() < 1e-12));
    }
    // A cache of another sequence is refused.
    let other = LayerCache::capture(&model, &ids[..ids.len() - 1], Backend::Vector).unwrap();
    assert!(d.check_ids(&other.header.ids).is_err());
    std::fs::write(&path, b"{}").unwrap();
    assert!(TeacherData::read(&path).is_err());
    std::fs::remove_dir_all(&dir).unwrap();
}
