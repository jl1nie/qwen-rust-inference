//! Gated attention for a whole sequence, and its backward (ADR 0037).
//!
//! Inference reads one position at a time against the KV cache
//! (`AttentionWeights::step`). Training computes all `n` positions of a
//! sequence at once, and keeps what the backward needs. The weights stay
//! frozen: the backward returns only the gradient toward the input.
//!
//! The forward, per position t and head h (KV head `kv = h / group`):
//!
//! ```text
//! [q_raw | gate] = W_q·x        (interleaved per head)
//! q = RoPE_t(RMSNorm(q_raw))    k = RoPE_t(RMSNorm(W_k·x))    v = W_v·x
//! P[t, s] = softmax_s(q_t · k_s / √d),  s ≤ t
//! context_t = Σ_s P[t, s] v_s
//! out_t = W_o·(context_t ⊙ σ(gate_t))
//! ```
//!
//! The probabilities `P` are `heads × n × n`, so they are not kept: the
//! backward computes each row again from `q` and `k`.
use crate::attention::AttentionWeights;
use crate::ops::{rms_norm, sigmoid, softmax, Scale};
use crate::tensor::{finite, require, Result};
use crate::train::grad::rms_norm_backward;
use rayon::prelude::*;

/// What the forward computed for `n` positions, kept for the backward.
/// Every field is row-major `[n, …]`.
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionForward {
    /// The input (the block's `input_norm`), `[n, hidden]`.
    pub x: Vec<f32>,
    /// `W_q·x`, `[n, heads·2·head_dim]`: each head's raw query, then its gate.
    pub q_gate: Vec<f32>,
    /// `W_k·x` before its norm, `[n, kv_heads·head_dim]`.
    pub key_raw: Vec<f32>,
    /// After norm and RoPE: `[n, heads·head_dim]` and `[n, kv_heads·head_dim]`.
    pub query: Vec<f32>,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
    /// `Σ_s P v`, `[n, heads·head_dim]`.
    pub context: Vec<f32>,
    /// `context ⊙ σ(gate)`, the input of `W_o`.
    pub gated: Vec<f32>,
    /// `[n, hidden]`.
    pub output: Vec<f32>,
    pub n: usize,
}

/// The RoPE angle of pair `i` at `position`, computed as `attention::rotate`
/// computes it, so the forward here rotates by exactly the same amounts.
fn angle(position: usize, i: usize, dim: usize, theta: f32) -> f32 {
    position as f32 / theta.powf((2 * i) as f32 / dim as f32)
}

/// Backward of `attention::rotate`. A rotation's inverse is its transpose, so
/// the gradient is rotated back by the same angle.
pub fn rotate_backward(dy: &mut [f32], position: usize, dim: usize, theta: f32) -> Result<()> {
    require(
        dim <= dy.len() && dim % 2 == 0 && theta.is_finite() && theta > 0.,
        "invalid RoPE",
    )?;
    for i in 0..dim / 2 {
        let (s, c) = angle(position, i, dim, theta).sin_cos();
        let (a, b) = (dy[i], dy[i + dim / 2]);
        dy[i] = a * c + b * s;
        dy[i + dim / 2] = b * c - a * s;
    }
    finite(dy)
}

/// The causal softmax read for one head over `n` positions: queries
/// `q [n, d]` (row stride `qs`) against keys and values `[n, d]` (stride
/// `ks`). Returns `context [n, d]`. Sums in the order `attention::attend` uses.
fn head_forward(
    q: &[f32],
    qs: usize,
    k: &[f32],
    v: &[f32],
    ks: usize,
    d: usize,
    n: usize,
) -> Result<Vec<f32>> {
    let scale = (d as f32).sqrt();
    let mut context = vec![0.; n * d];
    for t in 0..n {
        let qt = &q[t * qs..t * qs + d];
        let scores: Vec<f32> = (0..=t)
            .map(|s| {
                qt.iter()
                    .zip(&k[s * ks..s * ks + d])
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    / scale
            })
            .collect();
        let p = softmax(&scores)?;
        let out = &mut context[t * d..(t + 1) * d];
        for (s, &ps) in p.iter().enumerate() {
            for (o, &vs) in out.iter_mut().zip(&v[s * ks..s * ks + d]) {
                *o += ps * vs;
            }
        }
    }
    Ok(context)
}

/// Gradients of one head's causal read: toward its queries `[n, d]`, and
/// its contributions toward the shared keys and values `[n, d]`.
struct HeadGrad {
    dq: Vec<f32>,
    dk: Vec<f32>,
    dv: Vec<f32>,
}

/// Backward of `head_forward` for `dcontext [n, d]`.
///
/// For each query row t, with `P` recomputed:
/// `dv_s += P_s·dc_t`, `dP_s = dc_t · v_s`, and the softmax backward
/// `dscore_s = P_s·(dP_s − Σ_r P_r dP_r)`. The score is `q_t·k_s/√d`, so
/// `dq_t += Σ_s dscore_s k_s/√d` and `dk_s += dscore_s q_t/√d`.
#[allow(clippy::too_many_arguments)]
fn head_backward(
    q: &[f32],
    qs: usize,
    k: &[f32],
    v: &[f32],
    ks: usize,
    dcontext: &[f32],
    d: usize,
    n: usize,
) -> Result<HeadGrad> {
    let scale = (d as f32).sqrt();
    let mut g = HeadGrad {
        dq: vec![0.; n * d],
        dk: vec![0.; n * d],
        dv: vec![0.; n * d],
    };
    for t in 0..n {
        let qt = &q[t * qs..t * qs + d];
        let dc = &dcontext[t * d..(t + 1) * d];
        let scores: Vec<f32> = (0..=t)
            .map(|s| {
                qt.iter()
                    .zip(&k[s * ks..s * ks + d])
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    / scale
            })
            .collect();
        let p = softmax(&scores)?;
        let dp: Vec<f64> = (0..=t)
            .map(|s| {
                dc.iter()
                    .zip(&v[s * ks..s * ks + d])
                    .map(|(&a, &b)| a as f64 * b as f64)
                    .sum()
            })
            .collect();
        let mean: f64 = p.iter().zip(&dp).map(|(&p, &dp)| p as f64 * dp).sum();
        for s in 0..=t {
            let ps = p[s];
            for (dv, &c) in g.dv[s * d..(s + 1) * d].iter_mut().zip(dc) {
                *dv += ps * c;
            }
            let ds = (ps as f64 * (dp[s] - mean)) as f32 / scale;
            let ks_row = &k[s * ks..s * ks + d];
            for (dq, &kv) in g.dq[t * d..(t + 1) * d].iter_mut().zip(ks_row) {
                *dq += ds * kv;
            }
            for (dk, &qv) in g.dk[s * d..(s + 1) * d].iter_mut().zip(qt) {
                *dk += ds * qv;
            }
        }
    }
    Ok(g)
}

/// Forward for the `n` positions `0..n` of one sequence, `x [n, hidden]`.
pub fn forward(w: &AttentionWeights, x: &[f32], n: usize) -> Result<AttentionForward> {
    let c = &w.config;
    let (heads, kvh, d) = (c.heads, c.kv_heads, c.head_dim);
    require(
        heads > 0 && kvh > 0 && heads % kvh == 0 && n > 0,
        "invalid attention shape",
    )?;
    let q_gate = w.q.project_batch(x, n)?;
    let key_raw = w.k.project_batch(x, n)?;
    let value = w.v.project_batch(x, n)?;
    require(
        q_gate.len() == n * heads * 2 * d && key_raw.len() == n * kvh * d,
        "attention projection shape",
    )?;
    let mut query = Vec::with_capacity(n * heads * d);
    let mut gate = Vec::with_capacity(n * heads * d);
    let mut key = Vec::with_capacity(n * kvh * d);
    for t in 0..n {
        for head in q_gate[t * heads * 2 * d..(t + 1) * heads * 2 * d].chunks_exact(2 * d) {
            let mut q = rms_norm(&head[..d], &w.q_norm, c.eps, Scale::QwenZeroCentered)?;
            crate::attention::rotate(&mut q, t, c.rotary_dim, c.theta)?;
            query.extend(q);
            gate.extend_from_slice(&head[d..]);
        }
        for head in key_raw[t * kvh * d..(t + 1) * kvh * d].chunks_exact(d) {
            let mut k = rms_norm(head, &w.k_norm, c.eps, Scale::QwenZeroCentered)?;
            crate::attention::rotate(&mut k, t, c.rotary_dim, c.theta)?;
            key.extend(k);
        }
    }
    let group = heads / kvh;
    let per_head: Vec<Vec<f32>> = (0..heads)
        .into_par_iter()
        .map(|h| {
            let kv = h / group;
            head_forward(
                &query[h * d..],
                heads * d,
                &key[kv * d..],
                &value[kv * d..],
                kvh * d,
                d,
                n,
            )
        })
        .collect::<Result<_>>()?;
    let mut context = vec![0.; n * heads * d];
    for (h, ch) in per_head.iter().enumerate() {
        for t in 0..n {
            context[(t * heads + h) * d..(t * heads + h + 1) * d]
                .copy_from_slice(&ch[t * d..(t + 1) * d]);
        }
    }
    let gated: Vec<f32> = context
        .iter()
        .zip(&gate)
        .map(|(&c, &g)| c * sigmoid(g))
        .collect();
    let output = w.o.project_batch(&gated, n)?;
    Ok(AttentionForward {
        x: x.to_vec(),
        q_gate,
        key_raw,
        query,
        key,
        value,
        context,
        gated,
        output,
        n,
    })
}

/// Gradient toward the input `x [n, hidden]`, for `dout [n, hidden]`.
pub fn backward(w: &AttentionWeights, f: &AttentionForward, dout: &[f32]) -> Result<Vec<f32>> {
    let c = &w.config;
    let (heads, kvh, d, n) = (c.heads, c.kv_heads, c.head_dim, f.n);
    require(
        dout.len() == f.output.len(),
        "attention backward shape mismatch",
    )?;
    // out = W_o·gated.
    let dgated = w.o.project_transposed_batch(dout, n)?;
    // gated = context ⊙ σ(gate): each side gets the other's value.
    let mut dq_gate = vec![0.; f.q_gate.len()];
    let mut dcontext = vec![0.; f.context.len()];
    for t in 0..n {
        for h in 0..heads {
            let i = (t * heads + h) * d;
            let g0 = (t * heads + h) * 2 * d + d;
            for j in 0..d {
                let s = sigmoid(f.q_gate[g0 + j]);
                dcontext[i + j] = dgated[i + j] * s;
                dq_gate[g0 + j] = dgated[i + j] * f.context[i + j] * s * (1. - s);
            }
        }
    }
    // The causal read, per head. A KV head is shared by `group` query heads,
    // so its gradient is the sum of theirs (added in head order).
    let group = heads / kvh;
    let per_head: Vec<HeadGrad> = (0..heads)
        .into_par_iter()
        .map(|h| {
            let kv = h / group;
            let dc: Vec<f32> = (0..n)
                .flat_map(|t| {
                    dcontext[(t * heads + h) * d..(t * heads + h + 1) * d]
                        .iter()
                        .copied()
                })
                .collect();
            head_backward(
                &f.query[h * d..],
                heads * d,
                &f.key[kv * d..],
                &f.value[kv * d..],
                kvh * d,
                &dc,
                d,
                n,
            )
        })
        .collect::<Result<_>>()?;
    let mut dkey = vec![0.; f.key.len()];
    let mut dvalue = vec![0.; f.value.len()];
    for (h, g) in per_head.iter().enumerate() {
        let kv = h / group;
        for t in 0..n {
            let o = (t * kvh + kv) * d;
            for j in 0..d {
                dkey[o + j] += g.dk[t * d + j];
                dvalue[o + j] += g.dv[t * d + j];
            }
        }
    }
    // Back through RoPE and the per-head norms, to the raw projections.
    for t in 0..n {
        for (h, g) in per_head.iter().enumerate() {
            let mut dq = g.dq[t * d..(t + 1) * d].to_vec();
            rotate_backward(&mut dq, t, c.rotary_dim, c.theta)?;
            let q0 = (t * heads + h) * 2 * d;
            let g = rms_norm_backward(
                &f.q_gate[q0..q0 + d],
                &w.q_norm,
                c.eps,
                Scale::QwenZeroCentered,
                &dq,
            )?;
            dq_gate[q0..q0 + d].copy_from_slice(&g.dx);
        }
        for kv in 0..kvh {
            let o = (t * kvh + kv) * d;
            let mut dk = dkey[o..o + d].to_vec();
            rotate_backward(&mut dk, t, c.rotary_dim, c.theta)?;
            let g = rms_norm_backward(
                &f.key_raw[o..o + d],
                &w.k_norm,
                c.eps,
                Scale::QwenZeroCentered,
                &dk,
            )?;
            dkey[o..o + d].copy_from_slice(&g.dx);
        }
    }
    // x feeds three projections; their gradients add.
    let mut dx = w.q.project_transposed_batch(&dq_gate, n)?;
    for (a, b) in dx.iter_mut().zip(w.k.project_transposed_batch(&dkey, n)?) {
        *a += b;
    }
    for (a, b) in dx.iter_mut().zip(w.v.project_transposed_batch(&dvalue, n)?) {
        *a += b;
    }
    finite(&dx)?;
    Ok(dx)
}
