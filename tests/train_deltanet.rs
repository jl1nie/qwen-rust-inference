//! ADR 0037, step 2: Gated DeltaNet and DeltaNet blocks over a whole sequence.
//!
//! The forward must equal the inference path (`DeltaWeights::step`,
//! `Model::step`). The backward is compared with numerical differentiation
//! of an f64 copy written here from the definitions, on the synthetic hybrid
//! model (layers 0–2 are DeltaNet) and on a small DeltaNet whose two value
//! heads share one key head. Short chunks make the backward recompute
//! states from saved ones; the result must not depend on the chunk length.
#![cfg(feature = "train")]
mod train_common;

use qwen_rust_inference::block::{BlockWeights, MixerWeights};
use qwen_rust_inference::deltanet::{DeltaConfig, DeltaWeights};
use qwen_rust_inference::tensor::{Backend, Matrix};
use qwen_rust_inference::train::block::TrainBlock;
use qwen_rust_inference::train::deltanet::{self, forward_chunked};
use qwen_rust_inference::train::ffn::TrainFfn;
use train_common::{fixture, to64};

fn values(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s as f64 / u32::MAX as f64 * 2. - 1.) as f32
        })
        .collect()
}
fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0., f32::max)
}
fn assert_gradient(analytic: &[f32], numerical: &[f64], tol: f64, what: &str) {
    assert_eq!(analytic.len(), numerical.len(), "{what}");
    let diff: f64 = analytic
        .iter()
        .zip(numerical)
        .map(|(&a, &n)| (a as f64 - n).powi(2))
        .sum::<f64>()
        .sqrt();
    let norm: f64 = numerical.iter().map(|n| n * n).sum::<f64>().sqrt();
    let rel = diff / norm.max(1e-9);
    assert!(rel < tol, "{what}: relative error {rel:e} (norm {norm:e})");
}
fn numerical(x: &[f64], f: impl Fn(&[f64]) -> f64) -> Vec<f64> {
    const H: f64 = 1e-6;
    (0..x.len())
        .map(|i| {
            let (mut a, mut b) = (x.to_vec(), x.to_vec());
            a[i] += H;
            b[i] -= H;
            (f(&a) - f(&b)) / (2. * H)
        })
        .collect()
}
fn delta_of(b: &BlockWeights) -> &DeltaWeights {
    match &b.mixer {
        MixerWeights::Delta(d) => d,
        _ => panic!("expected a DeltaNet layer"),
    }
}
fn rows(x: &[f64], d: usize) -> Vec<Vec<f64>> {
    x.chunks(d).map(|r| r.to_vec()).collect()
}
fn weighted(outs: &[Vec<f64>], c: &[f64]) -> f64 {
    outs.concat().iter().zip(c).map(|(a, b)| a * b).sum()
}

// ---- f64 copy, from the definitions ----

fn matvec(w: &[f64], x: &[f64], rows: usize) -> Vec<f64> {
    let cols = x.len();
    (0..rows)
        .map(|i| (0..cols).map(|j| w[i * cols + j] * x[j]).sum())
        .collect()
}
fn sig(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}
fn silu(x: f64) -> f64 {
    x * sig(x)
}
fn l2(x: &[f64]) -> Vec<f64> {
    let r = 1. / (x.iter().map(|v| v * v).sum::<f64>() + 1e-6).sqrt();
    x.iter().map(|v| v * r).collect()
}
fn rms_zc(x: &[f64], w: &[f64], eps: f64) -> Vec<f64> {
    let r = 1. / (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64 + eps).sqrt();
    x.iter().zip(w).map(|(v, w)| v * r * (1. + w)).collect()
}

struct Delta64 {
    qkv: Vec<f64>,
    z: Vec<f64>,
    a: Vec<f64>,
    b: Vec<f64>,
    conv: Vec<f64>,
    dt: Vec<f64>,
    a_log: Vec<f64>,
    norm: Vec<f64>,
    out: Vec<f64>,
    c: DeltaConfig,
    channels: usize,
    hidden: usize,
}
impl Delta64 {
    fn new(w: &DeltaWeights) -> Self {
        Self {
            qkv: to64(&w.qkv.values()),
            z: to64(&w.z.values()),
            a: to64(&w.a.values()),
            b: to64(&w.b.values()),
            conv: to64(&w.conv.values()),
            dt: to64(&w.dt),
            a_log: to64(&w.a_log),
            norm: to64(&w.norm),
            out: to64(&w.out.values()),
            c: w.config.clone(),
            channels: w.qkv.rows(),
            hidden: w.out.rows(),
        }
    }
    fn forward(&self, xs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let c = &self.c;
        let (kh, vh, dk, dv, kern, ch) = (
            c.key_heads,
            c.value_heads,
            c.key_dim,
            c.value_dim,
            c.kernel,
            self.channels,
        );
        let p: Vec<Vec<f64>> = xs.iter().map(|x| matvec(&self.qkv, x, ch)).collect();
        let mut s = vec![vec![0f64; dk * dv]; vh];
        let mut outs = vec![];
        for (t, x) in xs.iter().enumerate() {
            let conv: Vec<f64> = (0..ch)
                .map(|i| {
                    let sum: f64 = (0..kern)
                        .map(|j| {
                            let back = kern - 1 - j;
                            if t >= back {
                                self.conv[i * kern + j] * p[t - back][i]
                            } else {
                                0.
                            }
                        })
                        .sum();
                    silu(sum)
                })
                .collect();
            let (a, b, z) = (
                matvec(&self.a, x, vh),
                matvec(&self.b, x, vh),
                matvec(&self.z, x, vh * dv),
            );
            let kd = kh * dk;
            let mut gated = vec![];
            for h in 0..vh {
                let k0 = (h / (vh / kh)) * dk;
                let q: Vec<f64> = l2(&conv[k0..k0 + dk])
                    .iter()
                    .map(|v| v / (dk as f64).sqrt())
                    .collect();
                let k = l2(&conv[kd + k0..kd + k0 + dk]);
                let v = &conv[2 * kd + h * dv..2 * kd + (h + 1) * dv];
                let beta = sig(b[h]);
                let u = a[h] + self.dt[h];
                let g = -self.a_log[h].exp() * (1. + u.exp()).ln();
                let st = &mut s[h];
                for x in st.iter_mut() {
                    *x *= g.exp();
                }
                let delta: Vec<f64> = (0..dv)
                    .map(|j| beta * (v[j] - (0..dk).map(|i| st[i * dv + j] * k[i]).sum::<f64>()))
                    .collect();
                for i in 0..dk {
                    for j in 0..dv {
                        st[i * dv + j] += k[i] * delta[j];
                    }
                }
                let o: Vec<f64> = (0..dv)
                    .map(|j| (0..dk).map(|i| st[i * dv + j] * q[i]).sum())
                    .collect();
                let r =
                    1. / (o.iter().map(|v| v * v).sum::<f64>() / dv as f64 + c.eps as f64).sqrt();
                for j in 0..dv {
                    gated.push(o[j] * r * self.norm[j] * silu(z[h * dv + j]));
                }
            }
            outs.push(matvec(&self.out, &gated, self.hidden));
        }
        outs
    }
}

/// A DeltaNet with `key_heads = 1`, `value_heads = 2` (grouped), random weights.
fn grouped() -> DeltaWeights {
    let (hidden, kh, vh, dk, dv, kernel) = (6, 1, 2, 3, 4, 4);
    let ch = 2 * kh * dk + vh * dv;
    let m = |rows: usize, cols: usize, seed: u32| {
        Matrix::new(
            rows,
            cols,
            values(seed, rows * cols).iter().map(|v| v * 0.6).collect(),
        )
        .unwrap()
    };
    DeltaWeights {
        qkv: m(ch, hidden, 1),
        z: m(vh * dv, hidden, 2),
        a: m(vh, hidden, 3),
        b: m(vh, hidden, 4),
        conv: m(ch, kernel, 5),
        dt: values(6, vh),
        a_log: values(7, vh).iter().map(|v| v * 0.5).collect(),
        norm: values(8, dv).iter().map(|v| 1. + 0.3 * v).collect(),
        out: m(hidden, vh * dv, 9),
        config: DeltaConfig {
            key_heads: kh,
            value_heads: vh,
            key_dim: dk,
            value_dim: dv,
            kernel,
            eps: 1e-6,
        },
    }
}

fn inference_outputs(w: &DeltaWeights, x: &[f32], n: usize) -> Vec<f32> {
    let hidden = x.len() / n;
    let mut state = w.new_state();
    x.chunks(hidden)
        .flat_map(|r| w.step(r, &mut state, Backend::Reference).unwrap().output)
        .collect()
}

// ---- tests ----

#[test]
fn deltanet_forward_matches_inference() {
    let (model, _) = fixture();
    let g = grouped();
    for (name, w) in [
        ("fixture layer 0", delta_of(&model.blocks()[0])),
        ("grouped", &g),
    ] {
        let hidden = w.out.rows();
        let n = 9;
        let x = values(11, n * hidden);
        let f = deltanet::forward(w, &x, n).unwrap();
        assert!(
            max_diff(&f.output, &inference_outputs(w, &x, n)) < 1e-5,
            "{name}"
        );
    }
}

#[test]
fn the_chunk_length_changes_nothing() {
    let w = grouped();
    let hidden = w.out.rows();
    let n = 10;
    let x = values(12, n * hidden);
    let dout = values(13, n * hidden);
    let whole = forward_chunked(&w, &x, n, 64).unwrap();
    let dx = deltanet::backward(&w, &whole, &dout).unwrap();
    for chunk in [1, 3, 4] {
        let f = forward_chunked(&w, &x, n, chunk).unwrap();
        assert_eq!(f.output, whole.output, "chunk {chunk}");
        assert_eq!(f.checkpoints[0].len(), n.div_ceil(chunk));
        assert_eq!(
            deltanet::backward(&w, &f, &dout).unwrap(),
            dx,
            "chunk {chunk}"
        );
    }
}

#[test]
fn deltanet_backward_matches_numerical() {
    let (model, _) = fixture();
    let g = grouped();
    for (name, w) in [
        ("fixture layer 1", delta_of(&model.blocks()[1])),
        ("grouped", &g),
    ] {
        let w64 = Delta64::new(w);
        let hidden = w64.hidden;
        let n = 7;
        let x = values(14, n * hidden);
        let c = to64(&values(15, n * hidden));
        let f = forward_chunked(w, &x, n, 3).unwrap();
        let out64 = w64.forward(&rows(&to64(&x), hidden)).concat();
        let gap = f
            .output
            .iter()
            .zip(&out64)
            .map(|(a, b)| (*a as f64 - b).abs())
            .fold(0., f64::max);
        assert!(gap < 1e-5, "{name}: f64 copy differs by {gap}");
        let dout: Vec<f32> = c.iter().map(|&v| v as f32).collect();
        let dx = deltanet::backward(w, &f, &dout).unwrap();
        let num = numerical(&to64(&x), |x| weighted(&w64.forward(&rows(x, hidden)), &c));
        assert_gradient(&dx, &num, 1e-4, name);
    }
}

#[test]
fn deltanet_blocks_forward_match_the_model() {
    let (model, ids) = fixture();
    let mut state = model.new_state();
    let steps: Vec<_> = ids
        .iter()
        .map(|&id| {
            model
                .step(id, &mut state, Backend::Reference, None)
                .unwrap()
        })
        .collect();
    let hidden = model.config().hidden_size;
    for layer in 0..3 {
        let input: Vec<f32> = if layer == 0 {
            steps.iter().flat_map(|s| s.embedding.clone()).collect()
        } else {
            steps
                .iter()
                .flat_map(|s| s.layers[layer - 1].clone())
                .collect()
        };
        let f = TrainBlock::frozen(&model.blocks()[layer])
            .forward(&input, ids.len())
            .unwrap();
        for (t, s) in steps.iter().enumerate() {
            assert!(
                max_diff(&f.output[t * hidden..(t + 1) * hidden], &s.layers[layer]) < 1e-5,
                "layer {layer} position {t}"
            );
        }
    }
}

#[test]
fn deltanet_block_backward_matches_numerical_for_input_and_ffn() {
    let (model, _) = fixture();
    let layer = 1;
    let b = &model.blocks()[layer];
    let d64 = Delta64::new(delta_of(b));
    let hidden = d64.hidden;
    let n = 5;
    let x = values(16, n * hidden);
    let c = to64(&values(17, n * hidden));
    let dout: Vec<f32> = c.iter().map(|&v| v as f32).collect();
    let ffn = TrainFfn::from_weights(&b.ffn);
    let fw = [
        to64(ffn.gate.values()),
        to64(ffn.up.values()),
        to64(ffn.down.values()),
    ];
    let eps = b.eps as f64;
    let (inn, post) = (to64(&b.input_norm), to64(&b.post_norm));
    let inter = fw[0].len() / hidden;
    let loss = |x: &[f64], ffn: [&[f64]; 3]| {
        let xs = rows(x, hidden);
        let normed: Vec<Vec<f64>> = xs.iter().map(|x| rms_zc(x, &inn, eps)).collect();
        let mixed = d64.forward(&normed);
        let outs: Vec<Vec<f64>> = xs
            .iter()
            .zip(&mixed)
            .map(|(x, m)| {
                let after: Vec<f64> = x.iter().zip(m).map(|(a, b)| a + b).collect();
                let p = rms_zc(&after, &post, eps);
                let g = matvec(ffn[0], &p, inter);
                let u = matvec(ffn[1], &p, inter);
                let prod: Vec<f64> = g.iter().zip(&u).map(|(g, u)| silu(*g) * u).collect();
                let o = matvec(ffn[2], &prod, hidden);
                after.iter().zip(&o).map(|(a, b)| a + b).collect()
            })
            .collect();
        weighted(&outs, &c)
    };
    let train = TrainBlock::with_ffn(b, &ffn);
    let f = train.forward(&x, n).unwrap();
    let mut grad = ffn.zero_grad().unwrap();
    let dx = train.backward(&f, &dout, Some(&mut grad)).unwrap();
    let x64 = to64(&x);
    let num = numerical(&x64, |x| loss(x, [&fw[0], &fw[1], &fw[2]]));
    assert_gradient(&dx, &num, 1e-4, "block dx");
    let num_down = numerical(&fw[2], |d| loss(&x64, [&fw[0], &fw[1], d]));
    assert_gradient(grad.tensors[2].values(), &num_down, 1e-4, "ddown");
    let num_gate = numerical(&fw[0], |g| loss(&x64, [g, &fw[1], &fw[2]]));
    assert_gradient(grad.tensors[0].values(), &num_gate, 1e-4, "dgate");
}
