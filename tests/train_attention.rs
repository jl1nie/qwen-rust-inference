//! ADR 0037, step 1: attention and an attention block over a whole sequence.
//!
//! The forward must equal the inference path (`AttentionWeights::sequence`,
//! `Model::step`). The backward is compared with numerical differentiation
//! of an f64 copy written here from the definitions (not from the Rust
//! forward), on the synthetic hybrid model, whose layer 3 is attention.
#![cfg(feature = "train")]
mod train_common;

use qwen_rust_inference::attention::{rotate, AttentionWeights};
use qwen_rust_inference::block::{BlockWeights, MixerWeights};
use qwen_rust_inference::model::Model;
use qwen_rust_inference::tensor::Backend;
use qwen_rust_inference::train::attention::{self, rotate_backward};
use qwen_rust_inference::train::block::TrainBlock;
use qwen_rust_inference::train::ffn::TrainFfn;
use train_common::{fixture, to64};

const LAYER: usize = 3;

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

fn attention_of(b: &BlockWeights) -> &AttentionWeights {
    match &b.mixer {
        MixerWeights::Attention(a) => a,
        _ => panic!("layer {LAYER} of the fixture is attention"),
    }
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0., f32::max)
}

/// `‖a − n‖ / ‖n‖` over the whole vector.
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

// ---- f64 copies, from the definitions ----

fn matvec(w: &[f64], x: &[f64], rows: usize) -> Vec<f64> {
    let cols = x.len();
    (0..rows)
        .map(|i| (0..cols).map(|j| w[i * cols + j] * x[j]).sum())
        .collect()
}
fn rms_zc(x: &[f64], w: &[f64], eps: f64) -> Vec<f64> {
    let r = 1. / (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64 + eps).sqrt();
    x.iter().zip(w).map(|(v, w)| v * r * (1. + w)).collect()
}
fn sig(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}
fn rope(x: &mut [f64], pos: usize, dim: usize, theta: f64) {
    for i in 0..dim / 2 {
        let a = pos as f64 / theta.powf((2 * i) as f64 / dim as f64);
        let (s, c) = a.sin_cos();
        let (p, q) = (x[i], x[i + dim / 2]);
        x[i] = p * c - q * s;
        x[i + dim / 2] = q * c + p * s;
    }
}

struct Attn64 {
    q: Vec<f64>,
    k: Vec<f64>,
    v: Vec<f64>,
    o: Vec<f64>,
    q_norm: Vec<f64>,
    k_norm: Vec<f64>,
    heads: usize,
    kvh: usize,
    d: usize,
    rot: usize,
    theta: f64,
    eps: f64,
    hidden: usize,
}
impl Attn64 {
    fn new(w: &AttentionWeights) -> Self {
        let c = &w.config;
        Self {
            q: to64(&w.q.values()),
            k: to64(&w.k.values()),
            v: to64(&w.v.values()),
            o: to64(&w.o.values()),
            q_norm: to64(&w.q_norm),
            k_norm: to64(&w.k_norm),
            heads: c.heads,
            kvh: c.kv_heads,
            d: c.head_dim,
            rot: c.rotary_dim,
            theta: c.theta as f64,
            eps: c.eps as f64,
            hidden: w.o.rows(),
        }
    }
    /// `xs`: the n inputs, each `[hidden]`. Returns the n outputs.
    fn forward(&self, xs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let (h, kvh, d) = (self.heads, self.kvh, self.d);
        let mut qs = vec![];
        let mut gates = vec![];
        let mut ks = vec![];
        let mut vs = vec![];
        for (t, x) in xs.iter().enumerate() {
            let m = matvec(&self.q, x, h * 2 * d);
            let mut q = vec![];
            let mut g = vec![];
            for head in m.chunks(2 * d) {
                let mut r = rms_zc(&head[..d], &self.q_norm, self.eps);
                rope(&mut r, t, self.rot, self.theta);
                q.extend(r);
                g.extend_from_slice(&head[d..]);
            }
            let kr = matvec(&self.k, x, kvh * d);
            let mut k = vec![];
            for head in kr.chunks(d) {
                let mut r = rms_zc(head, &self.k_norm, self.eps);
                rope(&mut r, t, self.rot, self.theta);
                k.extend(r);
            }
            qs.push(q);
            gates.push(g);
            ks.push(k);
            vs.push(matvec(&self.v, x, kvh * d));
        }
        (0..xs.len())
            .map(|t| {
                let mut gated = vec![0.; h * d];
                for hh in 0..h {
                    let kv = hh / (h / kvh);
                    let q = &qs[t][hh * d..(hh + 1) * d];
                    let sc: Vec<f64> = (0..=t)
                        .map(|s| {
                            q.iter()
                                .zip(&ks[s][kv * d..(kv + 1) * d])
                                .map(|(a, b)| a * b)
                                .sum::<f64>()
                                / (d as f64).sqrt()
                        })
                        .collect();
                    let m = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let e: Vec<f64> = sc.iter().map(|v| (v - m).exp()).collect();
                    let z: f64 = e.iter().sum();
                    for j in 0..d {
                        let c: f64 = (0..=t).map(|s| e[s] / z * vs[s][kv * d + j]).sum();
                        gated[hh * d + j] = c * sig(gates[t][hh * d + j]);
                    }
                }
                matvec(&self.o, &gated, self.hidden)
            })
            .collect()
    }
}

fn rows(x: &[f64], d: usize) -> Vec<Vec<f64>> {
    x.chunks(d).map(|r| r.to_vec()).collect()
}
fn weighted(outs: &[Vec<f64>], c: &[f64]) -> f64 {
    outs.concat().iter().zip(c).map(|(a, b)| a * b).sum()
}

// ---- tests ----

#[test]
fn rope_backward_is_the_transpose_of_rope() {
    for (pos, dim) in [(0, 8), (3, 8), (11, 4), (200, 16)] {
        let mut x = values(pos as u32 + 1, 16);
        let dy = values(pos as u32 + 50, 16);
        let x0 = x.clone();
        rotate(&mut x, pos, dim, 10_000.).unwrap();
        let mut dx = dy.clone();
        rotate_backward(&mut dx, pos, dim, 10_000.).unwrap();
        // <R x, dy> = <x, Rᵀ dy>
        let a: f64 = x.iter().zip(&dy).map(|(a, b)| *a as f64 * *b as f64).sum();
        let b: f64 = x0.iter().zip(&dx).map(|(a, b)| *a as f64 * *b as f64).sum();
        assert!((a - b).abs() < 1e-5, "pos {pos} dim {dim}: {a} vs {b}");
        // The part past `dim` is not rotated.
        assert_eq!(dx[dim..], dy[dim..]);
    }
}

#[test]
fn attention_forward_matches_the_inference_sequence() {
    let (model, _) = fixture();
    let w = attention_of(&model.blocks()[LAYER]);
    let hidden = w.o.rows();
    let n = 6;
    let x = values(7, n * hidden);
    let f = attention::forward(w, &x, n).unwrap();
    let xs: Vec<Vec<f32>> = x.chunks(hidden).map(|r| r.to_vec()).collect();
    let reference = w.sequence(&xs, Backend::Reference).unwrap();
    for (t, a) in reference.iter().enumerate() {
        let got = &f.output[t * hidden..(t + 1) * hidden];
        assert!(max_diff(got, &a.output) < 1e-5, "position {t}");
        assert!(
            max_diff(
                &f.query[t * a.query.len()..(t + 1) * a.query.len()],
                &a.query
            ) < 1e-6
        );
    }
}

#[test]
fn attention_backward_matches_numerical() {
    let (model, _) = fixture();
    let w = attention_of(&model.blocks()[LAYER]);
    let w64 = Attn64::new(w);
    let hidden = w64.hidden;
    let n = 5;
    let x = values(3, n * hidden);
    let c = to64(&values(9, n * hidden));
    // The f64 copy agrees with the f32 forward before it is differentiated.
    let f = attention::forward(w, &x, n).unwrap();
    let out64 = w64.forward(&rows(&to64(&x), hidden)).concat();
    let gap = f
        .output
        .iter()
        .zip(&out64)
        .map(|(a, b)| (*a as f64 - b).abs())
        .fold(0., f64::max);
    assert!(gap < 1e-5, "f64 copy differs by {gap}");
    let dout: Vec<f32> = c.iter().map(|&v| v as f32).collect();
    let dx = attention::backward(w, &f, &dout).unwrap();
    let num = numerical(&to64(&x), |x| weighted(&w64.forward(&rows(x, hidden)), &c));
    assert_gradient(&dx, &num, 1e-4, "attention dx");
}

/// The f64 block: attention, then the FFN with the given gate/up/down.
fn block64(b: &BlockWeights, a: &Attn64, ffn: [&[f64]; 3], xs: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let eps = b.eps as f64;
    let (inn, post) = (to64(&b.input_norm), to64(&b.post_norm));
    let normed: Vec<Vec<f64>> = xs.iter().map(|x| rms_zc(x, &inn, eps)).collect();
    let mixed = a.forward(&normed);
    let inter = ffn[0].len() / a.hidden;
    xs.iter()
        .zip(&mixed)
        .map(|(x, m)| {
            let after: Vec<f64> = x.iter().zip(m).map(|(a, b)| a + b).collect();
            let p = rms_zc(&after, &post, eps);
            let g = matvec(ffn[0], &p, inter);
            let u = matvec(ffn[1], &p, inter);
            let prod: Vec<f64> = g.iter().zip(&u).map(|(g, u)| g * sig(*g) * u).collect();
            let o = matvec(ffn[2], &prod, a.hidden);
            after.iter().zip(&o).map(|(a, b)| a + b).collect()
        })
        .collect()
}

#[test]
fn attention_block_forward_matches_the_model() {
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
    let input: Vec<f32> = steps
        .iter()
        .flat_map(|s| s.layers[LAYER - 1].clone())
        .collect();
    let block = TrainBlock::frozen(&model.blocks()[LAYER]);
    let f = block.forward(&input, ids.len()).unwrap();
    for (t, s) in steps.iter().enumerate() {
        let got = &f.output[t * hidden..(t + 1) * hidden];
        assert!(max_diff(got, &s.layers[LAYER]) < 1e-5, "position {t}");
    }
    // A trainable copy of the same FFN gives the same output.
    let ffn = TrainFfn::from_weights(&model.blocks()[LAYER].ffn);
    let g = TrainBlock::with_ffn(&model.blocks()[LAYER], &ffn)
        .forward(&input, ids.len())
        .unwrap();
    assert!(max_diff(&g.output, &f.output) < 1e-6);
}

fn block_case(model: &Model) -> (usize, Vec<f32>, Vec<f64>) {
    let hidden = model.config().hidden_size;
    let n = 4;
    (n, values(21, n * hidden), to64(&values(22, n * hidden)))
}

#[test]
fn attention_block_backward_matches_numerical_for_input_and_ffn() {
    let (model, _) = fixture();
    let b = &model.blocks()[LAYER];
    let a64 = Attn64::new(attention_of(b));
    let hidden = a64.hidden;
    let (n, x, c) = block_case(&model);
    let dout: Vec<f32> = c.iter().map(|&v| v as f32).collect();
    let ffn = TrainFfn::from_weights(&b.ffn);
    let (g64, u64_, d64) = (
        to64(ffn.gate.values()),
        to64(ffn.up.values()),
        to64(ffn.down.values()),
    );

    // Frozen FFN: the gradient toward x.
    let frozen = TrainBlock::frozen(b);
    let f = frozen.forward(&x, n).unwrap();
    let dx = frozen.backward(&f, &dout, None).unwrap();
    let loss = |x: &[f64], ffn: [&[f64]; 3]| weighted(&block64(b, &a64, ffn, &rows(x, hidden)), &c);
    let num = numerical(&to64(&x), |x| loss(x, [&g64, &u64_, &d64]));
    assert_gradient(&dx, &num, 1e-4, "block dx (frozen FFN)");

    // Trainable FFN: the same dx, and the parameter gradients.
    let train = TrainBlock::with_ffn(b, &ffn);
    let f = train.forward(&x, n).unwrap();
    let mut grad = ffn.zero_grad().unwrap();
    let dx2 = train.backward(&f, &dout, Some(&mut grad)).unwrap();
    assert!(
        max_diff(&dx2, &dx) < 1e-5,
        "trainable and frozen FFN give the same dx"
    );
    let x64 = to64(&x);
    let num_gate = numerical(&g64, |g| loss(&x64, [g, &u64_, &d64]));
    let num_up = numerical(&u64_, |u| loss(&x64, [&g64, u, &d64]));
    let num_down = numerical(&d64, |d| loss(&x64, [&g64, &u64_, d]));
    assert_gradient(grad.tensors[0].values(), &num_gate, 1e-4, "dgate");
    assert_gradient(grad.tensors[1].values(), &num_up, 1e-4, "dup");
    assert_gradient(grad.tensors[2].values(), &num_down, 1e-4, "ddown");

    // Missing gradient buffer for a trainable FFN is an error, not a panic.
    assert!(train.backward(&f, &dout, None).is_err());
}

#[test]
fn a_deltanet_block_is_refused_for_now() {
    let (model, _) = fixture();
    let hidden = model.config().hidden_size;
    let block = TrainBlock::frozen(&model.blocks()[0]);
    assert!(block.forward(&values(1, 2 * hidden), 2).is_err());
}
