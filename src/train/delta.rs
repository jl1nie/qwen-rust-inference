//! What training changed in the weights: `ΔW = W_trained − W_original`.
//!
//! Volume 1, chapter 5 changed one column of `down` by hand. Training changes
//! the whole FFN. To compare the two, this module looks at ΔW itself:
//!
//! - **size**: `‖ΔW‖` and `‖ΔW‖ / ‖W‖` for gate, up and down;
//! - **rank**: the largest singular values of ΔW, and the share of `‖ΔW‖²`
//!   they hold. A change that lives in a few directions is "low-rank",
//!   which is what LoRA assumes from the start;
//! - **units**: how much of the change falls on each FFN unit j (row j of
//!   gate and up, column j of down), and how many units hold most of it.
//!   A hand edit of one column puts everything on one unit;
//! - **directions**: the main directions ΔW_down writes into the residual
//!   stream, read through the final norm and the LM head (as a logit lens):
//!   which IDs they make more or less likely;
//! - **a hand edit as ΔW**: an `EditManifest` on the same layer, written as
//!   the change of `down` it is equivalent to, and its cosine with the
//!   trained ΔW_down.
use crate::edit::EditManifest;
use crate::model::Model;
use crate::tensor::{require, Backend, Result};
use crate::train::ffn::TrainFfn;
use crate::train::linalg::Dense;
use crate::train::lora::PROJECTIONS;
use rayon::prelude::*;
use serde::Serialize;

/// `a − b` for two matrices of the same shape.
pub fn difference(a: &Dense, b: &Dense) -> Result<Dense> {
    require(
        a.rows() == b.rows() && a.cols() == b.cols(),
        "delta shape mismatch",
    )?;
    Dense::new(
        a.rows(),
        a.cols(),
        a.values()
            .iter()
            .zip(b.values())
            .map(|(x, y)| x - y)
            .collect(),
    )
}

fn frobenius(d: &Dense) -> f64 {
    d.values()
        .iter()
        .map(|&v| (v as f64).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Eigenvalues and eigenvectors of a symmetric `n × n` matrix (row-major),
/// by cyclic Jacobi rotations; eigenvalues in decreasing order, vectors as
/// the columns of the returned row-major matrix.
fn symmetric_eigen(mut a: Vec<f64>, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0.; n * n];
    for i in 0..n {
        v[i * n + i] = 1.;
    }
    for _ in 0..100 {
        let off: f64 = (0..n)
            .flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| a[i * n + j].powi(2))
            .sum();
        let scale: f64 = (0..n)
            .map(|i| a[i * n + i].powi(2))
            .sum::<f64>()
            .max(1e-300);
        if off <= 1e-30 * scale {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2. * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.).sqrt());
                let t = if theta == 0. { 1. } else { t };
                let c = 1. / (t * t + 1.).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[k * n + p], a[k * n + q]);
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p * n + k], a[q * n + k]);
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[k * n + p], v[k * n + q]);
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| a[j * n + j].total_cmp(&a[i * n + i]));
    let values = order.iter().map(|&i| a[i * n + i]).collect();
    let mut vectors = vec![0.; n * n];
    for (new, &old) in order.iter().enumerate() {
        for k in 0..n {
            vectors[k * n + new] = v[k * n + old];
        }
    }
    (values, vectors)
}

/// Orthonormalizes the rows of `q` (`k × n`, k ≤ n) in place (modified
/// Gram–Schmidt, twice). A row that collapses (it lay in the span of the
/// earlier rows) is replaced by unit vectors e_t until one survives.
fn orthonormalize(q: &mut [Vec<f64>]) {
    let n = q.first().map_or(0, Vec::len);
    for i in 0..q.len() {
        let mut fallback = i;
        loop {
            for _ in 0..2 {
                for j in 0..i {
                    let (head, tail) = q.split_at_mut(i);
                    let d: f64 = tail[0].iter().zip(&head[j]).map(|(a, b)| a * b).sum();
                    for (a, b) in tail[0].iter_mut().zip(&head[j]) {
                        *a -= d * b;
                    }
                }
            }
            let norm = q[i].iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > 1e-10 {
                q[i].iter_mut().for_each(|v| *v /= norm);
                break;
            }
            q[i] = (0..n)
                .map(|t| f64::from(u8::from(t == fallback % n)))
                .collect();
            fallback += 1;
        }
    }
}

/// The largest singular values of `m` and their left singular vectors
/// (length `m.rows()`), by subspace iteration on `M·Mᵀ` with `k` vectors.
pub fn top_singular(
    m: &Dense,
    want: usize,
    iterations: usize,
) -> Result<(Vec<f64>, Vec<Vec<f64>>)> {
    let rows = m.rows();
    let k = (2 * want).max(want + 4).min(rows).min(m.cols());
    require(want > 0 && want <= k, "invalid singular value count")?;
    // Fixed start: the result is reproducible.
    let mut s = 0x2545_f491_4f6c_dd1d_u64;
    let mut q: Vec<Vec<f64>> = (0..k)
        .map(|_| {
            (0..rows)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
                })
                .collect()
        })
        .collect();
    orthonormalize(&mut q);
    let flat = |q: &[Vec<f64>]| -> Vec<f32> { q.iter().flatten().map(|&v| v as f32).collect() };
    let mut z = Vec::new();
    for it in 0..=iterations {
        // Z = Q·M ([k, cols]); then Q ← Z·Mᵀ = Q·M·Mᵀ.
        z = m.backward_input(&flat(&q), k)?;
        if it == iterations {
            break;
        }
        let y = m.forward(&z, k)?;
        q = y
            .chunks_exact(rows)
            .map(|r| r.iter().map(|&v| v as f64).collect())
            .collect();
        orthonormalize(&mut q);
    }
    // Rayleigh–Ritz: B = Q·M·Mᵀ·Qᵀ = Z·Zᵀ (k × k).
    let cols = m.cols();
    let zr: Vec<&[f32]> = z.chunks_exact(cols).collect();
    let b: Vec<f64> = (0..k * k)
        .into_par_iter()
        .map(|t| {
            let (i, j) = (t / k, t % k);
            zr[i]
                .iter()
                .zip(zr[j])
                .map(|(&a, &b)| a as f64 * b as f64)
                .sum()
        })
        .collect();
    let (values, vectors) = symmetric_eigen(b, k);
    let sigma = values.iter().take(want).map(|v| v.max(0.).sqrt()).collect();
    let left = (0..want)
        .map(|c| {
            (0..rows)
                .map(|r| (0..k).map(|i| vectors[i * k + c] * q[i][r]).sum())
                .collect()
        })
        .collect();
    Ok((sigma, left))
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MatrixDelta {
    pub name: String,
    pub frobenius: f64,
    /// `‖ΔW‖ / ‖W‖`
    pub relative: f64,
    pub singular_values: Vec<f64>,
    /// `(r, share)`: the share of `‖ΔW‖²` in the r largest singular values.
    pub energy: Vec<(usize, f64)>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct UnitShare {
    pub unit: usize,
    pub share: f64,
}

/// A direction ΔW_down writes, read through the final norm and the LM head.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DirectionLens {
    pub sigma: f64,
    /// IDs with the largest scores along +u, and along −u (the sign of a
    /// singular vector is arbitrary).
    pub plus: Vec<(u32, f32)>,
    pub minus: Vec<(u32, f32)>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EditComparison {
    pub columns: Vec<usize>,
    pub edit_frobenius: f64,
    /// `⟨ΔW_train, ΔW_edit⟩ / (‖ΔW_train‖·‖ΔW_edit‖)` on `down`.
    pub cosine: f64,
    /// Share of `‖ΔW_train,down‖²` in the edited columns.
    pub trained_share_in_columns: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DeltaReport {
    pub layer: usize,
    pub matrices: Vec<MatrixDelta>,
    /// Units with the largest share of `‖Δgate[j,:]‖² + ‖Δup[j,:]‖² + ‖Δdown[:,j]‖²`.
    pub units_top: Vec<UnitShare>,
    /// `(n, share)`: the share of the whole change in the n largest units.
    pub unit_concentration: Vec<(usize, f64)>,
    pub directions: Vec<DirectionLens>,
    pub edit: Option<EditComparison>,
}

/// The change of `down` (`[hidden, intermediate]`) that `edit` makes at
/// `layer`, as `EditManifest::apply` defines it: a zero-unit edit scales the
/// unit's activation, which is the same as scaling column j; a direction
/// edit removes the direction's component from column j. `None` if the
/// manifest does not edit `layer`.
pub fn edit_as_delta(edit: &EditManifest, layer: usize, down: &Dense) -> Result<Option<Dense>> {
    let edits: Vec<_> = edit.edits.iter().filter(|e| e.layer == layer).collect();
    if edits.is_empty() {
        return Ok(None);
    }
    let (hidden, inter) = (down.rows(), down.cols());
    let w = down.values();
    let mut delta = vec![0.; hidden * inter];
    let columns: std::collections::BTreeSet<usize> = edits
        .iter()
        .flat_map(|e| e.columns.iter().copied())
        .collect();
    for j in columns {
        require(j < inter, "edit column out of range")?;
        let col: Vec<f64> = (0..hidden).map(|i| w[i * inter + j] as f64).collect();
        let mut scale = 1.;
        let mut removed = vec![0.; hidden];
        for e in edits.iter().filter(|e| e.columns.contains(&j)) {
            match &e.direction {
                None => scale *= 1. - e.coefficient as f64,
                Some(d) => {
                    require(d.len() == hidden, "direction shape mismatch")?;
                    let dd: f64 = d.iter().map(|&v| (v as f64).powi(2)).sum();
                    let proj: f64 =
                        d.iter().zip(&col).map(|(&a, b)| a as f64 * b).sum::<f64>() / dd;
                    for (r, &v) in removed.iter_mut().zip(d) {
                        *r -= e.coefficient as f64 * v as f64 * proj;
                    }
                }
            }
        }
        // apply() scales the activation first, then adds the direction
        // correction computed from the scaled activation.
        for i in 0..hidden {
            delta[i * inter + j] = (scale * (col[i] + removed[i]) - col[i]) as f32;
        }
    }
    Ok(Some(Dense::new(hidden, inter, delta)?))
}

/// The report for `trained` against the FFN of `model` at `layer`.
/// `rank`: singular values per matrix; `lens`: directions and IDs per end.
pub fn analyze(
    model: &Model,
    layer: usize,
    trained: &TrainFfn,
    edit: Option<&EditManifest>,
    rank: usize,
    lens: (usize, usize),
) -> Result<DeltaReport> {
    require(layer < model.blocks().len(), "layer out of range")?;
    let base = TrainFfn::from_weights(&model.blocks()[layer].ffn);
    let merged = trained.merged()?;
    let deltas = [
        difference(&merged[0], &base.gate)?,
        difference(&merged[1], &base.up)?,
        difference(&merged[2], &base.down)?,
    ];
    let mut matrices = Vec::new();
    let mut down_left = Vec::new();
    for (i, (d, w)) in deltas
        .iter()
        .zip([&base.gate, &base.up, &base.down])
        .enumerate()
    {
        let f = frobenius(d);
        let (sigma, left) = if f > 0. {
            top_singular(d, rank.min(d.rows()).min(d.cols()), 60)?
        } else {
            (vec![0.; rank], Vec::new())
        };
        if i == 2 {
            down_left = left;
        }
        let total = f * f;
        let mut cumulative = 0.;
        let mut energy = Vec::new();
        for (r, s) in sigma.iter().enumerate() {
            cumulative += s * s;
            if [1, 2, 4, 8, 16, 32, 64].contains(&(r + 1)) || r + 1 == sigma.len() {
                energy.push((
                    r + 1,
                    if total > 0. {
                        (cumulative / total).min(1.)
                    } else {
                        0.
                    },
                ));
            }
        }
        matrices.push(MatrixDelta {
            name: PROJECTIONS[i].into(),
            frobenius: f,
            relative: f / frobenius(w),
            singular_values: sigma,
            energy,
        });
    }

    // Units: row j of gate and up, column j of down.
    let inter = base.intermediate();
    let hidden = base.hidden();
    let mut unit = vec![0_f64; inter];
    for d in &deltas[..2] {
        for (j, row) in d.values().chunks_exact(hidden).enumerate() {
            unit[j] += row.iter().map(|&v| (v as f64).powi(2)).sum::<f64>();
        }
    }
    for row in deltas[2].values().chunks_exact(inter) {
        for (j, &v) in row.iter().enumerate() {
            unit[j] += (v as f64).powi(2);
        }
    }
    let total: f64 = unit.iter().sum();
    let mut order: Vec<usize> = (0..inter).collect();
    order.sort_by(|&a, &b| unit[b].total_cmp(&unit[a]));
    let share = |n: usize| -> f64 {
        if total > 0. {
            order[..n.min(inter)].iter().map(|&j| unit[j]).sum::<f64>() / total
        } else {
            0.
        }
    };
    let units_top = order
        .iter()
        .take(10)
        .map(|&j| UnitShare {
            unit: j,
            share: if total > 0. { unit[j] / total } else { 0. },
        })
        .collect();
    let unit_concentration = [1, 10, 100, 1000]
        .into_iter()
        .filter(|&n| n <= inter)
        .map(|n| (n, share(n)))
        .collect();

    // Directions through the final norm and the LM head: the FFN output is
    // added to the residual stream, then multiplied by (1 + w_norm) and
    // projected by the embedding rows.
    let vocab = model
        .config()
        .vocab_size
        .min(crate::train::teacher::COMMON_VOCAB);
    let (directions_n, ids_n) = lens;
    let mut directions = Vec::new();
    for (u, &sigma) in down_left
        .iter()
        .zip(&matrices[2].singular_values)
        .take(directions_n)
    {
        let v: Vec<f32> = u
            .iter()
            .zip(model.norm())
            .map(|(&u, &w)| (u * (1. + w as f64)) as f32)
            .collect();
        let scores = model.embedding().project(&v, Backend::Optimized)?;
        let mut ids: Vec<u32> = (0..vocab as u32).collect();
        ids.sort_by(|&a, &b| scores[b as usize].total_cmp(&scores[a as usize]));
        let pick = |it: &mut dyn Iterator<Item = &u32>| -> Vec<(u32, f32)> {
            it.take(ids_n).map(|&i| (i, scores[i as usize])).collect()
        };
        directions.push(DirectionLens {
            sigma,
            plus: pick(&mut ids.iter()),
            minus: pick(&mut ids.iter().rev()),
        });
    }

    let edit = match edit {
        None => None,
        Some(e) => match edit_as_delta(e, layer, &base.down)? {
            None => None,
            Some(ed) => {
                let dot: f64 = ed
                    .values()
                    .iter()
                    .zip(deltas[2].values())
                    .map(|(&a, &b)| a as f64 * b as f64)
                    .sum();
                let (fe, ft) = (frobenius(&ed), frobenius(&deltas[2]));
                let columns: Vec<usize> = (0..inter)
                    .filter(|&j| (0..hidden).any(|i| ed.values()[i * inter + j] != 0.))
                    .collect();
                let in_cols: f64 = deltas[2]
                    .values()
                    .chunks_exact(inter)
                    .map(|row| {
                        columns
                            .iter()
                            .map(|&j| (row[j] as f64).powi(2))
                            .sum::<f64>()
                    })
                    .sum();
                Some(EditComparison {
                    columns,
                    edit_frobenius: fe,
                    cosine: if fe > 0. && ft > 0. {
                        dot / (fe * ft)
                    } else {
                        0.
                    },
                    trained_share_in_columns: if ft > 0. { in_cols / (ft * ft) } else { 0. },
                })
            }
        },
    };
    Ok(DeltaReport {
        layer,
        matrices,
        units_top,
        unit_concentration,
        directions,
        edit,
    })
}
