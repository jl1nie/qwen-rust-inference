//! The FFN backward pass by hand, on the small example of volume 1,
//! chapter 4, computed by the same code that trains the real model
//! (`TrainFfn::backward`).
//!
//! Chapter 4: `x = (1, 2)`, `W_gate = [[2, −1], [−1, 1]]`, `W_up = I`,
//! `W_down = [[1, 3], [2, 4]]`, so `g = (0, 1)`, `u = (1, 2)`,
//! `a = SiLU(g) ⊙ u = (0, 2s)` with `s = SiLU(1)`, and `y = (6s, 8s)`.
//!
//! The loss is `L = y₂`, the output's second (vertical) component, so
//! `dL/dy = (0, 1)`. Gradient descent then lowers y₂. Chapter 8 lowers it
//! by hand, removing the vertical component of down's second column. The
//! gradient of `W_down` is `dy ⊗ a`, and only its row 2, column 2 is nonzero:
//! the same element.
use crate::tensor::Result;
use crate::train::ffn::TrainFfn;
use crate::train::linalg::Dense;
use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FfnBackwardLesson {
    /// `SiLU(1)`
    pub s: f64,
    /// `SiLU'(1) = σ(1)·(1 + 1·(1 − σ(1)))`
    pub silu_prime_1: f64,
    pub x: Vec<f32>,
    pub g: Vec<f32>,
    pub u: Vec<f32>,
    pub a: Vec<f32>,
    pub y: Vec<f32>,
    pub dy: Vec<f32>,
    /// `dL/da = W_downᵀ·dy`
    pub da: Vec<f32>,
    /// `dL/dg = da ⊙ u ⊙ SiLU'(g)`
    pub dg: Vec<f32>,
    /// `dL/du = da ⊙ SiLU(g)`
    pub du: Vec<f32>,
    /// Row-major 2×2 gradients.
    pub d_gate: Vec<f32>,
    pub d_up: Vec<f32>,
    pub d_down: Vec<f32>,
    /// `dL/dx = W_gateᵀ·dg + W_upᵀ·du`
    pub dx: Vec<f32>,
    /// Largest |analytic − numerical| over all 12 parameters and x
    /// (central difference of an f64 forward, step 1e-6).
    pub numerical_max_error: f64,
    /// One SGD step with `lr`: y₂ before, after, and predicted
    /// (`y₂ − lr·‖∇‖²`, first order).
    pub lr: f32,
    pub y2_before: f32,
    pub y2_after: f32,
    pub y2_predicted: f64,
}

fn silu64(z: f64) -> f64 {
    z / (1. + (-z).exp())
}

/// `y₂` of the chapter 4 FFN in f64, for numerical differentiation.
fn y2(gate: &[f64], up: &[f64], down: &[f64], x: &[f64]) -> f64 {
    let lin = |w: &[f64], v: &[f64]| [w[0] * v[0] + w[1] * v[1], w[2] * v[0] + w[3] * v[1]];
    let (g, u) = (lin(gate, x), lin(up, x));
    let a = [silu64(g[0]) * u[0], silu64(g[1]) * u[1]];
    down[2] * a[0] + down[3] * a[1]
}

pub fn ffn_backward(lr: f32) -> Result<FfnBackwardLesson> {
    let ffn = TrainFfn {
        gate: Dense::new(2, 2, vec![2., -1., -1., 1.])?,
        up: Dense::new(2, 2, vec![1., 0., 0., 1.])?,
        down: Dense::new(2, 2, vec![1., 3., 2., 4.])?,
        lora: None,
    };
    let x = vec![1., 2.];
    let f = ffn.forward(&x, 1)?;
    let dy = vec![0., 1.];
    let mut grad = ffn.zero_grad()?;
    let dx = ffn.backward(&f, &dy, &mut grad)?;
    // The intermediate gradients, as backward computes them inside.
    let da = ffn.down.backward_input(&dy, 1)?;
    let sw = crate::train::grad::swiglu_backward(&f.gate, &f.up, &da)?;

    let w64 = |d: &Dense| -> Vec<f64> { d.values().iter().map(|&v| v as f64).collect() };
    let (gate, up, down) = (w64(&ffn.gate), w64(&ffn.up), w64(&ffn.down));
    let x64 = [1., 2.];
    const H: f64 = 1e-6;
    let central = |f: &dyn Fn(f64) -> f64| (f(H) - f(-H)) / (2. * H);
    let mut err = 0_f64;
    for (which, analytic) in [
        grad.tensors[0].values(),
        grad.tensors[1].values(),
        grad.tensors[2].values(),
    ]
    .into_iter()
    .enumerate()
    {
        for (k, &an) in analytic.iter().enumerate() {
            let num = central(&|h| {
                let mut w = [gate.clone(), up.clone(), down.clone()];
                w[which][k] += h;
                y2(&w[0], &w[1], &w[2], &x64)
            });
            err = err.max((an as f64 - num).abs());
        }
    }
    for (k, &an) in dx.iter().enumerate() {
        let num = central(&|h| {
            let mut x = x64;
            x[k] += h;
            y2(&gate, &up, &down, &x)
        });
        err = err.max((an as f64 - num).abs());
    }

    let norm2: f64 = grad
        .slices()
        .iter()
        .flat_map(|s| s.iter())
        .map(|&v| (v as f64).powi(2))
        .sum();
    let mut stepped = ffn.clone();
    crate::train::optim::Optimizer::new(
        crate::train::optim::OptimizerConfig::Sgd { lr },
        &ffn.sizes(),
    )
    .step(&mut stepped.params_mut(), &grad.slices())?;
    let after = stepped.forward(&x, 1)?.output[1];
    let s1 = 1. / (1. + (-1_f64).exp());
    Ok(FfnBackwardLesson {
        s: silu64(1.),
        silu_prime_1: s1 * (1. + (1. - s1)),
        x,
        g: f.gate.clone(),
        u: f.up.clone(),
        a: f.product.clone(),
        y: f.output.clone(),
        dy,
        da,
        dg: sw.dgate,
        du: sw.dup,
        d_gate: grad.tensors[0].values().to_vec(),
        d_up: grad.tensors[1].values().to_vec(),
        d_down: grad.tensors[2].values().to_vec(),
        dx,
        numerical_max_error: err,
        lr,
        y2_before: f.output[1],
        y2_after: after,
        y2_predicted: f.output[1] as f64 - lr as f64 * norm2,
    })
}

fn v(x: &[f32]) -> String {
    let items: Vec<String> = x.iter().map(|v| format!("{v:.6}")).collect();
    format!("({})", items.join(", "))
}
fn m(x: &[f32]) -> String {
    format!("[{}; {}]", v(&x[..2]), v(&x[2..]))
}

impl FfnBackwardLesson {
    /// The lesson as text, one step per line, in the order of the backward pass.
    pub fn text(&self) -> String {
        let s = self.s;
        [
            format!("FFN backward on volume 1, chapter 4's example (s = SiLU(1) = {s:.6}, SiLU'(1) = {:.6})", self.silu_prime_1),
            format!("forward:  x = {}", v(&self.x)),
            format!("          g = W_gate x = {}    u = W_up x = {}", v(&self.g), v(&self.u)),
            format!("          a = SiLU(g) ⊙ u = {}    (0, 2s)", v(&self.a)),
            format!("          y = W_down a = {}    (6s, 8s)", v(&self.y)),
            "loss:     L = y2, so dL/dy = (0, 1)".into(),
            format!("down:     dL/dW_down = dy ⊗ a = {}    only row 2, column 2 (2s): the component volume 1, chapter 8 removes", m(&self.d_down)),
            format!("          dL/da = W_downᵀ dy = {}    down's row 2", v(&self.da)),
            format!("SwiGLU:   dL/dg = da ⊙ u ⊙ SiLU'(g) = {}    (2·1·SiLU'(0), 4·2·SiLU'(1)), SiLU'(0) = 0.5", v(&self.dg)),
            format!("          dL/du = da ⊙ SiLU(g) = {}    (0, 4s): unit 0 (the first) is closed by its gate", v(&self.du)),
            format!("gate, up: dL/dW_gate = dg ⊗ x = {}    row 1 too: SiLU'(0) = 0.5, so training can open unit 0", m(&self.d_gate)),
            format!("          dL/dW_up = du ⊗ x = {}", m(&self.d_up)),
            format!("input:    dL/dx = W_gateᵀ dg + W_upᵀ du = {}", v(&self.dx)),
            format!("check:    max |analytic − numerical| = {:.2e} (central difference, h = 1e-6)", self.numerical_max_error),
            format!(
                "SGD step: lr = {}: y2 {:.6} -> {:.6} (first order: {:.6})",
                self.lr, self.y2_before, self.y2_after, self.y2_predicted
            ),
        ]
        .join("\n")
    }
}
