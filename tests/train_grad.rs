//! M0 of book 3: each backward function against numerical differentiation.
//!
//! The forward used for the finite differences is an f64 copy of the f32
//! operation, checked first against the real f32 operation, so the gradient
//! is compared with a derivative of the same function, without f32 rounding
//! noise from the step size.
#![cfg(feature = "train")]
use qwen_rust_inference::ops::{rms_norm, silu, Scale};
use qwen_rust_inference::tensor::{Backend, Matrix};
use qwen_rust_inference::train::grad::{
    cross_entropy, kl_full, kl_sparse, log_softmax, rms_norm_backward, silu_backward,
    swiglu_backward, SparseTarget,
};
use qwen_rust_inference::train::linalg::Dense;

/// Deterministic values in (-1, 1) (xorshift).
fn values(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed
        .wrapping_mul(2)
        .wrapping_add(1)
        .wrapping_mul(0x9e37_79b9)
        | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s as f64 / u32::MAX as f64 * 2. - 1.) as f32
        })
        .collect()
}
fn to64(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| x as f64).collect()
}

/// Central difference of `f` at `x` along every coordinate.
fn numerical(x: &[f64], f: impl Fn(&[f64]) -> f64) -> Vec<f64> {
    const H: f64 = 1e-6;
    (0..x.len())
        .map(|i| {
            let mut a = x.to_vec();
            let mut b = x.to_vec();
            a[i] += H;
            b[i] -= H;
            (f(&a) - f(&b)) / (2. * H)
        })
        .collect()
}

/// Relative error of the analytic gradient against the numerical one, as a
/// whole vector: `‖a − n‖ / max(‖n‖, 1e-6)`.
fn assert_gradient(analytic: &[f32], numerical: &[f64], what: &str) {
    assert_eq!(analytic.len(), numerical.len(), "{what}");
    let diff: f64 = analytic
        .iter()
        .zip(numerical)
        .map(|(&a, &n)| (a as f64 - n).powi(2))
        .sum::<f64>()
        .sqrt();
    let norm: f64 = numerical.iter().map(|n| n * n).sum::<f64>().sqrt();
    let rel = diff / norm.max(1e-6);
    assert!(rel < 1e-3, "{what}: relative error {rel:e}");
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn rms_norm64(x: &[f64], w: &[f64], eps: f64, scale: Scale) -> Vec<f64> {
    let r = 1. / (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64 + eps).sqrt();
    x.iter()
        .zip(w)
        .map(|(x, w)| {
            x * r
                * match scale {
                    Scale::Direct => *w,
                    Scale::QwenZeroCentered => 1. + w,
                }
        })
        .collect()
}
fn silu64(x: f64) -> f64 {
    x / (1. + (-x).exp())
}

#[test]
fn rms_norm_backward_matches_numerical() {
    let n = 24;
    let x = values(1, n);
    let w: Vec<f32> = values(2, n).iter().map(|v| v * 0.5).collect();
    // Probe the gradient with a random linear loss L = c·y.
    let c = to64(&values(3, n));
    let eps = 1e-6;
    for scale in [Scale::Direct, Scale::QwenZeroCentered] {
        let y = rms_norm(&x, &w, eps, scale).unwrap();
        let y64 = rms_norm64(&to64(&x), &to64(&w), eps as f64, scale);
        assert!(y
            .iter()
            .zip(&y64)
            .all(|(a, b)| (*a as f64 - b).abs() < 1e-5));

        let dy: Vec<f32> = c.iter().map(|&v| v as f32).collect();
        let g = rms_norm_backward(&x, &w, eps, scale, &dy).unwrap();
        let (x64, w64) = (to64(&x), to64(&w));
        let dx = numerical(&x64, |x| dot(&c, &rms_norm64(x, &w64, eps as f64, scale)));
        let dw = numerical(&w64, |w| dot(&c, &rms_norm64(&x64, w, eps as f64, scale)));
        assert_gradient(&g.dx, &dx, &format!("{scale:?} dx"));
        assert_gradient(&g.dweight, &dw, &format!("{scale:?} dweight"));
    }
}

#[test]
fn silu_and_swiglu_backward_match_numerical() {
    let n = 32;
    // Spread over (-4, 4) to cover both tails of the sigmoid.
    let gate: Vec<f32> = values(4, n).iter().map(|v| v * 4.).collect();
    let up = values(5, n);
    let c = to64(&values(6, n));
    let dy: Vec<f32> = c.iter().map(|&v| v as f32).collect();

    let s = silu(&gate).unwrap();
    assert!(s
        .iter()
        .zip(&gate)
        .all(|(a, &g)| (*a as f64 - silu64(g as f64)).abs() < 1e-6));
    let dx = silu_backward(&gate, &dy).unwrap();
    let num = numerical(&to64(&gate), |g| {
        g.iter().zip(&c).map(|(g, c)| silu64(*g) * c).sum()
    });
    assert_gradient(&dx, &num, "silu");

    let g = swiglu_backward(&gate, &up, &dy).unwrap();
    let up64 = to64(&up);
    let gate64 = to64(&gate);
    let product = |g: &[f64], u: &[f64]| -> f64 {
        g.iter()
            .zip(u)
            .zip(&c)
            .map(|((g, u), c)| silu64(*g) * u * c)
            .sum()
    };
    assert_gradient(
        &g.dgate,
        &numerical(&gate64, |g| product(g, &up64)),
        "dgate",
    );
    assert_gradient(&g.dup, &numerical(&up64, |u| product(&gate64, u)), "dup");
}

fn log_softmax64(z: &[f64]) -> Vec<f64> {
    let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let ls = z.iter().map(|v| (v - max).exp()).sum::<f64>().ln();
    z.iter().map(|v| v - max - ls).collect()
}

#[test]
fn cross_entropy_matches_numerical() {
    let logits: Vec<f32> = values(7, 19).iter().map(|v| v * 3.).collect();
    let target = 11;
    let g = cross_entropy(&logits, target).unwrap();
    let lq = log_softmax64(&to64(&logits));
    assert!((g.loss + lq[target as usize]).abs() < 1e-9);
    let num = numerical(&to64(&logits), |z| -log_softmax64(z)[target as usize]);
    assert_gradient(&g.dlogits, &num, "cross-entropy");
    // The gradient sums to zero: raising every logit equally changes nothing.
    assert!(g.dlogits.iter().map(|&d| d as f64).sum::<f64>().abs() < 1e-6);
    assert!(cross_entropy(&logits, 19).is_err());
}

#[test]
fn log_softmax_survives_large_logits() {
    let l = log_softmax(&[1000., 0., -1000.]).unwrap();
    assert!(l[0].abs() < 1e-12 && (l[1] + 1000.).abs() < 1e-9);
}

/// KL(p‖q) over K entries plus "other", as the definition reads.
fn kl_sparse64(ids: &[u32], log_p: &[f64], other: f64, z: &[f64]) -> f64 {
    let lq = log_softmax64(z);
    let mut q_other = 1.;
    let mut kl = 0.;
    for (&i, &lp) in ids.iter().zip(log_p) {
        q_other -= lq[i as usize].exp();
        kl += lp.exp() * (lp - lq[i as usize]);
    }
    if other > 0. {
        kl += other * (other.ln() - q_other.ln());
    }
    kl
}

#[test]
fn kl_full_matches_numerical_and_vanishes_on_itself() {
    let logits: Vec<f32> = values(8, 19).iter().map(|v| v * 2.).collect();
    let teacher: Vec<f32> = values(9, 19).iter().map(|v| v * 2.).collect();
    let log_p = log_softmax(&teacher).unwrap();
    let g = kl_full(&log_p, &logits).unwrap();
    let num = numerical(&to64(&logits), |z| {
        log_p
            .iter()
            .zip(log_softmax64(z))
            .map(|(lp, lq)| lp.exp() * (lp - lq))
            .sum()
    });
    assert!(g.loss > 0.);
    assert_gradient(&g.dlogits, &num, "kl_full");

    let same = kl_full(&log_softmax(&logits).unwrap(), &logits).unwrap();
    assert!(same.loss.abs() < 1e-12);
    assert!(same.dlogits.iter().all(|d| d.abs() < 1e-7));
}

#[test]
fn kl_sparse_matches_numerical_and_full() {
    let vocab = 19;
    let logits: Vec<f32> = values(10, vocab).iter().map(|v| v * 2.).collect();
    let teacher: Vec<f32> = values(11, vocab).iter().map(|v| v * 3.).collect();
    let log_p = log_softmax(&teacher).unwrap();

    // Top 5 of the teacher plus "other".
    let mut order: Vec<u32> = (0..vocab as u32).collect();
    order.sort_by(|&a, &b| log_p[b as usize].total_cmp(&log_p[a as usize]));
    let ids = order[..5].to_vec();
    let lps: Vec<f64> = ids.iter().map(|&i| log_p[i as usize]).collect();
    let t = SparseTarget::new(ids.clone(), lps.clone(), vocab).unwrap();
    let mass: f64 = lps.iter().map(|l| l.exp()).sum();
    assert!((t.other() - (1. - mass)).abs() < 1e-12);

    let g = kl_sparse(&t, &logits).unwrap();
    let expect = kl_sparse64(&ids, &lps, t.other(), &to64(&logits));
    assert!((g.loss - expect).abs() < 1e-9);
    let num = numerical(&to64(&logits), |z| kl_sparse64(&ids, &lps, t.other(), z));
    assert_gradient(&g.dlogits, &num, "kl_sparse");

    // Grouping loses information, never adds it: KL over K+1 ≤ full KL.
    let full = kl_full(&log_p, &logits).unwrap();
    assert!(g.loss <= full.loss + 1e-12);

    // With K = the whole vocabulary, it is the full KL.
    let all = SparseTarget::new(
        order.clone(),
        order.iter().map(|&i| log_p[i as usize]).collect(),
        vocab,
    )
    .unwrap();
    assert_eq!(all.other(), 0.);
    let g = kl_sparse(&all, &logits).unwrap();
    assert!((g.loss - full.loss).abs() < 1e-9);
    assert!(g
        .dlogits
        .iter()
        .zip(&full.dlogits)
        .all(|(a, b)| (a - b).abs() < 1e-6));
}

#[test]
fn sparse_target_rejects_bad_input() {
    assert!(SparseTarget::new(vec![1, 1], vec![-1., -2.], 4).is_err()); // duplicate
    assert!(SparseTarget::new(vec![4], vec![-1.], 4).is_err()); // out of range
    assert!(SparseTarget::new(vec![0, 1], vec![-0.1, -0.1], 4).is_err()); // mass > 1
    assert!(SparseTarget::new(vec![0], vec![0.5], 4).is_err()); // log p > 0
    assert!(SparseTarget::new(vec![], vec![], 4).is_err());
}

/// Naive row-major `A [m,k] · B [k,n]`.
fn naive(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.; m * n];
    for i in 0..m {
        for j in 0..n {
            for p in 0..k {
                c[i * n + j] += a[i * k + p] * b[p * n + j];
            }
        }
    }
    c
}
fn transpose(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    (0..cols * rows)
        .map(|i| a[(i % rows) * cols + i / rows])
        .collect()
}
fn assert_close(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() <= 1e-5 + 1e-5 * y.abs(),
            "{what}[{i}]: {x} vs {y}"
        );
    }
}

#[test]
fn dense_products_match_naive() {
    let (rows, cols, n) = (13, 7, 5);
    let w = values(12, rows * cols);
    let x = values(13, n * cols);
    let dy = values(14, n * rows);
    let d = Dense::new(rows, cols, w.clone()).unwrap();
    let wt = transpose(&w, rows, cols);

    assert_close(
        &d.forward(&x, n).unwrap(),
        &naive(&x, &wt, n, cols, rows),
        "X·Wᵀ",
    );
    assert_close(
        &d.backward_input(&dy, n).unwrap(),
        &naive(&dy, &w, n, rows, cols),
        "dY·W",
    );

    let mut acc = d.clone();
    acc.accumulate(0.5, &dy, &x, n).unwrap();
    let grad = naive(&transpose(&dy, n, rows), &x, rows, n, cols);
    let expect: Vec<f32> = w.iter().zip(&grad).map(|(w, g)| w + 0.5 * g).collect();
    assert_close(acc.values(), &expect, "W + ½·dYᵀX");

    assert!(d.forward(&x[1..], n).is_err());
    assert!(Dense::new(2, 2, vec![0.; 3]).is_err());
}

/// Frozen weights (`Matrix`, F32 or BF16 storage) give the same batched
/// products. 600 rows is not a multiple of the 256-row widening block.
#[test]
fn frozen_matrix_batched_products_match_naive() {
    let (rows, cols, n) = (600, 9, 3);
    let w: Vec<f32> = values(15, rows * cols);
    // Exact BF16 copies of the values, so both storages hold the same numbers.
    let bits: Vec<u16> = w.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
    let w: Vec<f32> = bits
        .iter()
        .map(|&b| f32::from_bits((b as u32) << 16))
        .collect();
    let x = values(16, n * cols);
    let dy = values(17, n * rows);
    let wt = transpose(&w, rows, cols);
    for m in [
        Matrix::new(rows, cols, w.clone()).unwrap(),
        Matrix::from_bf16(rows, cols, bits.clone()).unwrap(),
    ] {
        let y = m.project_batch(&x, n).unwrap();
        assert_close(&y, &naive(&x, &wt, n, cols, rows), "project_batch");
        // Row 0 of the batch agrees with the inference GEMV.
        assert_close(
            &y[..rows],
            &m.project(&x[..cols], Backend::Optimized).unwrap(),
            "vs project",
        );

        let dx = m.project_transposed_batch(&dy, n).unwrap();
        assert_close(
            &dx,
            &naive(&dy, &w, n, rows, cols),
            "project_transposed_batch",
        );
        assert_close(
            &m.project_transposed(&dy[..rows]).unwrap(),
            &dx[..cols],
            "single",
        );
    }
}

/// The parallel products split their work at fixed boundaries, so 1 thread
/// and 8 threads give bit-identical results (and both agree with the naive
/// loops). Sizes span several chunks: Dense 300×200 (128-wide chunks), and a
/// BF16 Matrix of 1000 rows (256-row blocks, 32 row groups).
#[test]
fn parallel_products_do_not_depend_on_the_thread_count() {
    let run = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let (rows, cols, n) = (300, 200, 5);
            let d = Dense::new(rows, cols, values(20, rows * cols)).unwrap();
            let x = values(21, n * cols);
            let dy = values(22, n * rows);
            let mut acc = d.clone();
            acc.accumulate(1., &dy, &x, n).unwrap();
            let bits: Vec<u16> = values(23, 1000 * 96)
                .iter()
                .map(|v| (v.to_bits() >> 16) as u16)
                .collect();
            let m = Matrix::from_bf16(1000, 96, bits).unwrap();
            vec![
                d.forward(&x, n).unwrap(),
                d.backward_input(&dy, n).unwrap(),
                acc.values().to_vec(),
                m.project_batch(&values(24, n * 96), n).unwrap(),
                m.project_transposed_batch(&values(25, n * 1000), n)
                    .unwrap(),
            ]
        })
    };
    let (one, many) = (run(1), run(8));
    for (a, b) in one.iter().zip(&many) {
        assert!(a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()));
    }
    // And the chunked results are the products.
    let (rows, cols, n) = (300, 200, 5);
    let w = values(20, rows * cols);
    let x = values(21, n * cols);
    let dy = values(22, n * rows);
    assert_close(
        &one[0],
        &naive(&x, &transpose(&w, rows, cols), n, cols, rows),
        "forward",
    );
    assert_close(&one[1], &naive(&dy, &w, n, rows, cols), "backward_input");
    let grad = naive(&transpose(&dy, n, rows), &x, rows, n, cols);
    let expect: Vec<f32> = w.iter().zip(&grad).map(|(w, g)| w + g).collect();
    assert_close(&one[2], &expect, "accumulate");
    let mw: Vec<f32> = values(23, 1000 * 96)
        .iter()
        .map(|v| f32::from_bits(v.to_bits() & 0xffff_0000))
        .collect();
    let mx = values(24, n * 96);
    assert_close(
        &one[3],
        &naive(&mx, &transpose(&mw, 1000, 96), n, 96, 1000),
        "project_batch",
    );
    let mdy = values(25, n * 1000);
    assert_close(
        &one[4],
        &naive(&mdy, &mw, n, 1000, 96),
        "project_transposed_batch",
    );
}
