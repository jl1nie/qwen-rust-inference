//! What training changed in the weights (`train::delta`), checked on cases
//! whose answer is known.
#![cfg(feature = "train")]
use qwen_rust_inference::edit::{Edit, EditManifest};
use qwen_rust_inference::tensor::Backend;
use qwen_rust_inference::train::delta::{analyze, difference, edit_as_delta, top_singular};
use qwen_rust_inference::train::ffn::TrainFfn;
use qwen_rust_inference::train::linalg::Dense;
mod train_common;
use train_common::fixture;

fn values(seed: u64, n: usize) -> Vec<f64> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64 * 2. - 1.
        })
        .collect()
}

/// `count` orthonormal vectors of length `n`.
fn orthonormal(seed: u64, count: usize, n: usize) -> Vec<Vec<f64>> {
    let mut q: Vec<Vec<f64>> = Vec::new();
    for i in 0..count {
        let mut v = values(seed + i as u64, n);
        for u in &q {
            let d: f64 = v.iter().zip(u).map(|(a, b)| a * b).sum();
            v.iter_mut().zip(u).for_each(|(a, b)| *a -= d * b);
        }
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        q.push(v.into_iter().map(|x| x / norm).collect());
    }
    q
}

/// `Σ σ_i u_i v_iᵀ`, `[rows, cols]`.
fn planted(sigma: &[f64], u: &[Vec<f64>], v: &[Vec<f64>]) -> Dense {
    let (rows, cols) = (u[0].len(), v[0].len());
    let data = (0..rows * cols)
        .map(|t| {
            let (i, j) = (t / cols, t % cols);
            sigma
                .iter()
                .enumerate()
                .map(|(k, s)| s * u[k][i] * v[k][j])
                .sum::<f64>() as f32
        })
        .collect();
    Dense::new(rows, cols, data).unwrap()
}

#[test]
fn top_singular_recovers_planted_values_and_vectors() {
    let sigma = [5., 3., 2., 1., 0.5];
    for (rows, cols) in [(40, 30), (30, 40)] {
        let u = orthonormal(1, 5, rows);
        let v = orthonormal(9, 5, cols);
        let m = planted(&sigma, &u, &v);
        let (s, left) = top_singular(&m, 5, 60).unwrap();
        for (got, want) in s.iter().zip(sigma) {
            assert!((got - want).abs() < 1e-4 * want, "{rows}x{cols}: {s:?}");
        }
        for (l, want) in left.iter().zip(&u) {
            let c: f64 = l.iter().zip(want).map(|(a, b)| a * b).sum();
            assert!(c.abs() > 0.9999, "{c}");
        }
        // Asking for more than the rank: the rest is zero.
        let (s, _) = top_singular(&m, 8, 60).unwrap();
        assert!(s[5..].iter().all(|&x| x < 1e-4), "{s:?}");
    }
}

#[test]
fn lora_change_has_at_most_its_rank() {
    let (model, _) = fixture();
    let last = model.blocks().len() - 1;
    let mut ffn = TrainFfn::from_weights(&model.blocks()[last].ffn)
        .with_lora(2, 4., &[0, 2], 3)
        .unwrap();
    for l in ffn.lora.as_mut().unwrap().adapters.iter_mut().flatten() {
        let v: Vec<f32> = values(5, l.b.values().len())
            .iter()
            .map(|&x| x as f32 * 0.3)
            .collect();
        l.b = Dense::new(l.b.rows(), l.b.cols(), v).unwrap();
    }
    let r = analyze(&model, last, &ffn, None, 6, (2, 3)).unwrap();
    for m in &r.matrices {
        match m.name.as_str() {
            "up" => assert_eq!(m.frobenius, 0.),
            _ => {
                assert!(m.frobenius > 0.);
                let s = &m.singular_values;
                assert!(s[2..].iter().all(|&x| x < 1e-5 * s[0]), "{}: {s:?}", m.name);
                let at2 = m.energy.iter().find(|(r, _)| *r == 2).unwrap().1;
                assert!((at2 - 1.).abs() < 1e-6, "{}: {at2}", m.name);
            }
        }
    }
    assert_eq!(r.directions.len(), 2);
    assert_eq!(r.directions[0].plus.len(), 3);
    let total: f64 = r.units_top.iter().map(|u| u.share).sum();
    assert!(total <= 1. + 1e-9);
}

/// A manifest at the fixture's last layer: zero unit 2, and remove a
/// direction from units 2 and 5.
fn manifest(model: &qwen_rust_inference::model::Model) -> EditManifest {
    let last = model.blocks().len() - 1;
    let hidden = model.config().hidden_size;
    let name = format!("model.language_model.layers.{last}.mlp.down_proj.weight");
    EditManifest {
        model_revision: model.revision().into(),
        edits: vec![
            Edit {
                layer: last,
                tensor_name: name.clone(),
                columns: vec![2],
                direction: None,
                coefficient: 0.5,
            },
            Edit {
                layer: last,
                tensor_name: name,
                columns: vec![2, 5],
                direction: Some(values(7, hidden).iter().map(|&x| x as f32).collect()),
                coefficient: 0.8,
            },
        ],
    }
}

#[test]
fn a_hand_edit_is_a_change_of_down() {
    let (model, ids) = fixture();
    let last = model.blocks().len() - 1;
    let edit = manifest(&model);
    let base = TrainFfn::from_weights(&model.blocks()[last].ffn);
    let delta = edit_as_delta(&edit, last, &base.down).unwrap().unwrap();
    assert!(edit_as_delta(&edit, 0, &base.down).unwrap().is_none());
    // Only the edited columns change.
    let inter = base.intermediate();
    for (t, &v) in delta.values().iter().enumerate() {
        if ![2, 5].contains(&(t % inter)) {
            assert_eq!(v, 0.);
        }
    }
    // The model with down + Δ computes what the edit computes.
    let mut edited = base.clone();
    edited.down = Dense::new(
        base.down.rows(),
        base.down.cols(),
        base.down
            .values()
            .iter()
            .zip(delta.values())
            .map(|(a, b)| a + b)
            .collect(),
    )
    .unwrap();
    let changed = model.with_ffn(last, edited.to_weights().unwrap()).unwrap();
    let (mut s1, mut s2) = (model.new_state(), changed.new_state());
    for &id in &ids {
        let a = model
            .step(id, &mut s1, Backend::Vector, Some(&edit))
            .unwrap();
        let b = changed.step(id, &mut s2, Backend::Vector, None).unwrap();
        let d = a
            .logits
            .iter()
            .zip(&b.logits)
            .map(|(x, y)| (x - y).abs())
            .fold(0., f32::max);
        assert!(d < 1e-5, "{d}");
    }

    // Training that made exactly this change: cosine 1, all of it in the
    // edited columns, and the units are 2 and 5.
    let r = analyze(&model, last, &edited, Some(&edit), 4, (1, 2)).unwrap();
    let e = r.edit.unwrap();
    assert_eq!(e.columns, [2, 5]);
    assert!((e.cosine - 1.).abs() < 1e-6 && (e.trained_share_in_columns - 1.).abs() < 1e-6);
    let top: Vec<usize> = r.units_top[..2].iter().map(|u| u.unit).collect();
    assert!(top.contains(&2) && top.contains(&5), "{top:?}");
    assert!((r.unit_concentration[1].1 - 1.).abs() < 1e-9);
    // down's change has rank ≤ 2 (two columns).
    let s = &r.matrices[2].singular_values;
    assert!(s[2..].iter().all(|&x| x < 1e-5 * s[0]), "{s:?}");
}

#[test]
fn a_direction_reads_as_the_ids_it_promotes() {
    let (model, _) = fixture();
    let last = model.blocks().len() - 1;
    let base = TrainFfn::from_weights(&model.blocks()[last].ffn);
    let (hidden, inter) = (base.hidden(), base.intermediate());
    // Write the direction of ID t's embedding row (undoing the norm's
    // 1 + w) into one unit's column: the lens must find the ID whose
    // logit that direction raises most.
    let t = 7;
    let e = model.embedding().row(t).unwrap();
    let u: Vec<f32> = e
        .iter()
        .zip(model.norm())
        .map(|(e, w)| e / (1. + w))
        .collect();
    let scores = model
        .embedding()
        .project(
            &u.iter()
                .zip(model.norm())
                .map(|(u, w)| u * (1. + w))
                .collect::<Vec<_>>(),
            Backend::Reference,
        )
        .unwrap();
    let best = (0..scores.len())
        .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
        .unwrap() as u32;
    let mut trained = base.clone();
    let d = trained.down.values_mut();
    for i in 0..hidden {
        d[i * inter + 4] += 0.5 * u[i];
    }
    let r = analyze(&model, last, &trained, None, 2, (1, 3)).unwrap();
    let lens = &r.directions[0];
    assert!(
        lens.plus[0].0 == best || lens.minus[0].0 == best,
        "{lens:?}, want {best}"
    );
    assert_eq!(r.units_top[0].unit, 4);
    assert!(difference(&trained.down, &base.down).is_ok());
}
