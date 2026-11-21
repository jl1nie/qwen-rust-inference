//! Shared by the training tests: the fixture, an f64 copy of the top layer
//! written from the definitions, and numerical differentiation.
#![allow(dead_code)]
use qwen_rust_inference::model::Model;
use qwen_rust_inference::train::ffn::TrainFfn;

pub fn fixture() -> (Model, Vec<u32>) {
    let v: serde_json::Value =
        serde_json::from_str(include_str!("../fixtures/hybrid.json")).unwrap();
    let ids = v["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| i.as_u64().unwrap() as u32)
        .collect();
    (Model::from_fixture(&v).unwrap(), ids)
}

/// The top layer in f64, written from the definitions.
pub struct Top64 {
    pub gate: Vec<f64>,
    pub up: Vec<f64>,
    pub down: Vec<f64>,
    embedding: Vec<f64>,
    norm: Vec<f64>,
    eps: f64,
    hidden: usize,
    inter: usize,
    vocab: usize,
}
pub fn to64(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| x as f64).collect()
}
pub fn matvec(w: &[f64], x: &[f64], rows: usize) -> Vec<f64> {
    let cols = x.len();
    (0..rows)
        .map(|i| (0..cols).map(|j| w[i * cols + j] * x[j]).sum())
        .collect()
}
impl Top64 {
    pub fn new(model: &Model, ffn: &TrainFfn) -> Self {
        let c = model.config();
        Self {
            gate: to64(ffn.gate.values()),
            up: to64(ffn.up.values()),
            down: to64(ffn.down.values()),
            embedding: to64(&model.embedding().values()),
            norm: to64(model.norm()),
            eps: c.rms_norm_eps as f64,
            hidden: c.hidden_size,
            inter: c.intermediate_size,
            vocab: c.vocab_size,
        }
    }
    /// Sum over positions of −log q(target) for `after_mixer`, `post_norm`
    /// at each position (target None: not scored).
    pub fn loss(&self, inputs: &[(Vec<f64>, Vec<f64>)], targets: &[Option<u32>]) -> f64 {
        let mut total = 0.;
        for ((after, x), t) in inputs.iter().zip(targets) {
            let Some(t) = t else { continue };
            let g = matvec(&self.gate, x, self.inter);
            let u = matvec(&self.up, x, self.inter);
            let p: Vec<f64> = g
                .iter()
                .zip(&u)
                .map(|(g, u)| g / (1. + (-g).exp()) * u)
                .collect();
            let out = matvec(&self.down, &p, self.hidden);
            let h: Vec<f64> = after.iter().zip(&out).map(|(a, b)| a + b).collect();
            let r =
                1. / (h.iter().map(|v| v * v).sum::<f64>() / self.hidden as f64 + self.eps).sqrt();
            let n: Vec<f64> = h
                .iter()
                .zip(&self.norm)
                .map(|(h, w)| h * r * (1. + w))
                .collect();
            let z = matvec(&self.embedding, &n, self.vocab);
            let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let lse = max + z.iter().map(|v| (v - max).exp()).sum::<f64>().ln();
            total += lse - z[*t as usize];
        }
        total
    }
}

/// Central difference of `f` with respect to every element of `x`.
pub fn numerical(x: &mut [f64], f: &mut dyn FnMut(&[f64]) -> f64) -> Vec<f64> {
    const H: f64 = 1e-6;
    (0..x.len())
        .map(|i| {
            let v = x[i];
            x[i] = v + H;
            let a = f(x);
            x[i] = v - H;
            let b = f(x);
            x[i] = v;
            (a - b) / (2. * H)
        })
        .collect()
}
pub fn assert_gradient(analytic: &[f32], numerical: &[f64], what: &str) {
    let diff: f64 = analytic
        .iter()
        .zip(numerical)
        .map(|(&a, &n)| (a as f64 - n).powi(2))
        .sum::<f64>()
        .sqrt();
    let norm: f64 = numerical.iter().map(|n| n * n).sum::<f64>().sqrt();
    assert!(
        norm > 1e-6,
        "{what}: gradient vanishes, the check would prove nothing"
    );
    let rel = diff / norm;
    assert!(rel < 1e-3, "{what}: relative error {rel:e}");
}
