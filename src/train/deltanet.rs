//! Gated DeltaNet for a whole sequence, and its backward (ADR 0037).
//!
//! The forward, per position t (`deltanet::DeltaWeights::step`):
//!
//! ```text
//! p_t  = W_qkv·x_t
//! c_t  = SiLU(Σ_j conv_j ⊙ p_{t−K+1+j})          causal, per channel
//! q, k = L2(c_t[q|k]) (q also /√d_k)    v = c_t[v]
//! β = σ(W_b·x)    g = −exp(A_log)·softplus(W_a·x + dt)
//! S' = exp(g)·S_{t−1}    δ = β(v − S'ᵀk)    S_t = S' + k δᵀ    o = S_tᵀq
//! y_t  = W_out·(RMSNorm(o; norm) ⊙ SiLU(W_z·x))   per value head
//! ```
//!
//! The state S of a head is `d_k × d_v` (128 × 128 in the real model). The
//! backward needs `S_{t−1}` at every step. Keeping all of them would take
//! `n·heads·d_k·d_v` floats (1 MB per position in the real model), so the
//! forward keeps the state only at the start of every `chunk` positions,
//! and the backward runs each chunk forward again from its saved state
//! before stepping back through it.
use crate::deltanet::DeltaWeights;
use crate::ops::{rms_norm, sigmoid, silu, softplus, Scale};
use crate::tensor::{finite, require, Result};
use crate::train::grad::{rms_norm_backward, silu_backward};
use rayon::prelude::*;

/// The default chunk length for the saved states.
pub const CHUNK: usize = 64;

/// What the forward computed for `n` positions, kept for the backward.
/// Every field is row-major `[n, …]`, except `checkpoints`.
#[derive(Debug, Clone, PartialEq)]
pub struct DeltaForward {
    /// The input (the block's `input_norm`), `[n, hidden]`.
    pub x: Vec<f32>,
    /// `W_qkv·x` and the convolution's sums before SiLU, `[n, channels]`.
    pub projected: Vec<f32>,
    pub conv_sum: Vec<f32>,
    /// `W_a·x`, `W_b·x` `[n, value_heads]`, `W_z·x` `[n, value_heads·d_v]`.
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub z: Vec<f32>,
    /// Per value head: `[n, value_heads·d_k]` (q already scaled by 1/√d_k).
    pub query: Vec<f32>,
    pub key: Vec<f32>,
    /// `[n, value_heads·d_v]`.
    pub value: Vec<f32>,
    /// `[n, value_heads]`.
    pub beta: Vec<f32>,
    pub decay: Vec<f32>,
    /// The recurrence's output `o`, `[n, value_heads·d_v]`.
    pub core: Vec<f32>,
    /// The input of `W_out`.
    pub gated: Vec<f32>,
    /// `[n, hidden]`.
    pub output: Vec<f32>,
    /// `checkpoints[h][c]`: head h's state before position `c·chunk`.
    pub checkpoints: Vec<Vec<Vec<f32>>>,
    pub chunk: usize,
    pub n: usize,
}

/// Sizes used everywhere below.
#[derive(Clone, Copy)]
struct Dims {
    kh: usize,
    vh: usize,
    dk: usize,
    dv: usize,
    kernel: usize,
    channels: usize,
}
fn dims(w: &DeltaWeights) -> Result<Dims> {
    let c = &w.config;
    require(
        c.kernel > 0
            && c.key_heads > 0
            && c.value_heads % c.key_heads == 0
            && c.key_dim > 0
            && c.value_dim > 0,
        "invalid DeltaNet config",
    )?;
    let channels = w.qkv.rows();
    require(
        channels == 2 * c.key_heads * c.key_dim + c.value_heads * c.value_dim,
        "DeltaNet QKV shape",
    )?;
    Ok(Dims {
        kh: c.key_heads,
        vh: c.value_heads,
        dk: c.key_dim,
        dv: c.value_dim,
        kernel: c.kernel,
        channels,
    })
}

/// One step of the delta rule on `state` (`d_k × d_v`, row-major), the same
/// operations as `deltanet::delta_rule` without its checks. Returns `o`.
fn rule_step(q: &[f32], k: &[f32], v: &[f32], g: f32, beta: f32, s: &mut [f32]) -> Vec<f32> {
    let dv = v.len();
    let decay = g.exp();
    for x in s.iter_mut() {
        *x *= decay;
    }
    let mut delta = v.to_vec();
    for (j, d) in delta.iter_mut().enumerate() {
        let memory = (0..k.len()).map(|i| s[i * dv + j] * k[i]).sum::<f32>();
        *d = (*d - memory) * beta;
    }
    for i in 0..k.len() {
        for j in 0..dv {
            s[i * dv + j] += k[i] * delta[j];
        }
    }
    (0..dv)
        .map(|j| (0..q.len()).map(|i| s[i * dv + j] * q[i]).sum())
        .collect()
}

/// Gradients of one head's recurrence over the sequence.
struct RuleGrad {
    dq: Vec<f32>,
    dk: Vec<f32>,
    dv: Vec<f32>,
    dg: Vec<f32>,
    dbeta: Vec<f32>,
}

/// Backward of head h's recurrence, for `dcore` `[n, d_v]` (this head's
/// columns). Steps back from the last position, carrying `dS`, the gradient
/// with respect to the state after the current step:
///
/// ```text
/// o = S_tᵀq:          dq = S_t·do,            dS += q·doᵀ
/// S_t = S' + k·δᵀ:    dk += dS·δ,             dδ = dSᵀk
/// δ = β(v − m):       dv = β·dδ,  dβ = dδ·(v − m),  dm = −β·dδ
/// m = S'ᵀk:           dk += S'·dm,            dS += k·dmᵀ
/// S' = e^g·S_{t−1}:   dg = e^g·⟨S_{t−1}, dS⟩, dS ← e^g·dS
/// ```
#[allow(clippy::too_many_arguments)]
fn rule_backward(f: &DeltaForward, h: usize, dm: Dims, dcore: &[f32]) -> RuleGrad {
    let (vh, dk, dv, n) = (dm.vh, dm.dk, dm.dv, f.n);
    let row = |v: &[f32], t: usize, width: usize| -> Vec<f32> {
        v[(t * vh + h) * width..(t * vh + h + 1) * width].to_vec()
    };
    let mut g = RuleGrad {
        dq: vec![0.; n * dk],
        dk: vec![0.; n * dk],
        dv: vec![0.; n * dv],
        dg: vec![0.; n],
        dbeta: vec![0.; n],
    };
    let mut ds = vec![0f32; dk * dv];
    let chunks = n.div_ceil(f.chunk);
    for c in (0..chunks).rev() {
        let (t0, t1) = (c * f.chunk, ((c + 1) * f.chunk).min(n));
        // Run the chunk forward again, keeping the state before each step.
        let mut s = f.checkpoints[h][c].clone();
        let mut before = Vec::with_capacity(t1 - t0);
        for t in t0..t1 {
            before.push(s.clone());
            rule_step(
                &row(&f.query, t, dk),
                &row(&f.key, t, dk),
                &row(&f.value, t, dv),
                f.decay[t * vh + h],
                f.beta[t * vh + h],
                &mut s,
            );
        }
        for t in (t0..t1).rev() {
            let prev = &before[t - t0];
            let (q, k, v) = (
                row(&f.query, t, dk),
                row(&f.key, t, dk),
                row(&f.value, t, dv),
            );
            let (gate, beta) = (f.decay[t * vh + h], f.beta[t * vh + h]);
            let alpha = gate.exp();
            // S' and m = S'ᵀk, δ, S_t, as the forward computed them.
            let sp: Vec<f32> = prev.iter().map(|x| x * alpha).collect();
            let mem: Vec<f32> = (0..dv)
                .map(|j| (0..dk).map(|i| sp[i * dv + j] * k[i]).sum::<f32>())
                .collect();
            let delta: Vec<f32> = (0..dv).map(|j| (v[j] - mem[j]) * beta).collect();
            let st: Vec<f32> = (0..dk * dv)
                .map(|ij| sp[ij] + k[ij / dv] * delta[ij % dv])
                .collect();
            let dout = &dcore[t * dv..(t + 1) * dv];
            for i in 0..dk {
                g.dq[t * dk + i] = (0..dv).map(|j| st[i * dv + j] * dout[j]).sum();
                for j in 0..dv {
                    ds[i * dv + j] += q[i] * dout[j];
                }
            }
            let ddelta: Vec<f32> = (0..dv)
                .map(|j| (0..dk).map(|i| ds[i * dv + j] * k[i]).sum::<f32>())
                .collect();
            for i in 0..dk {
                g.dk[t * dk + i] += (0..dv).map(|j| ds[i * dv + j] * delta[j]).sum::<f32>();
            }
            let mut dbeta = 0f64;
            let mut dmem = vec![0f32; dv];
            for j in 0..dv {
                g.dv[t * dv + j] = beta * ddelta[j];
                dbeta += ddelta[j] as f64 * (v[j] - mem[j]) as f64;
                dmem[j] = -beta * ddelta[j];
            }
            g.dbeta[t] = dbeta as f32;
            for i in 0..dk {
                g.dk[t * dk + i] += (0..dv).map(|j| sp[i * dv + j] * dmem[j]).sum::<f32>();
                for j in 0..dv {
                    ds[i * dv + j] += k[i] * dmem[j];
                }
            }
            let inner: f64 = prev
                .iter()
                .zip(&ds)
                .map(|(&a, &b)| a as f64 * b as f64)
                .sum();
            g.dg[t] = (alpha as f64 * inner) as f32;
            for x in ds.iter_mut() {
                *x *= alpha;
            }
        }
    }
    g
}

/// Backward of `ops::l2_norm`, `y = x·r` with `r = 1/√(Σx² + 1e-6)`:
/// `dx = r·dy − x·r³·Σ x·dy`.
fn l2_backward(x: &[f32], dy: &[f32]) -> Vec<f32> {
    let s: f64 = x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() + 1e-6;
    let r = 1. / s.sqrt();
    let dot: f64 = x.iter().zip(dy).map(|(&a, &b)| a as f64 * b as f64).sum();
    x.iter()
        .zip(dy)
        .map(|(&x, &dy)| (r * dy as f64 - x as f64 * r.powi(3) * dot) as f32)
        .collect()
}

/// Forward for positions `0..n` of one sequence, from zero state, with the
/// default chunk length.
pub fn forward(w: &DeltaWeights, x: &[f32], n: usize) -> Result<DeltaForward> {
    forward_chunked(w, x, n, CHUNK)
}

/// As `forward`, saving the states every `chunk` positions.
pub fn forward_chunked(
    w: &DeltaWeights,
    x: &[f32],
    n: usize,
    chunk: usize,
) -> Result<DeltaForward> {
    let dm = dims(w)?;
    let Dims {
        kh,
        vh,
        dk,
        dv,
        kernel,
        channels,
    } = dm;
    require(n > 0 && chunk > 0, "invalid DeltaNet sequence")?;
    let projected = w.qkv.project_batch(x, n)?;
    let a = w.a.project_batch(x, n)?;
    let b = w.b.project_batch(x, n)?;
    let z = w.z.project_batch(x, n)?;
    require(
        a.len() == n * vh && b.len() == n * vh && z.len() == n * vh * dv,
        "DeltaNet gate shape",
    )?;
    // The causal convolution: the last weight multiplies the current value.
    let mut conv_sum = vec![0.; n * channels];
    for ch in 0..channels {
        let wc = w.conv.row(ch)?;
        for t in 0..n {
            let mut sum = 0.;
            for (j, &wj) in wc.iter().enumerate().take(kernel - 1) {
                if let Some(s) = (t + j + 1).checked_sub(kernel) {
                    sum += projected[s * channels + ch] * wj;
                }
            }
            sum += projected[t * channels + ch] * wc[kernel - 1];
            conv_sum[t * channels + ch] = sum;
        }
    }
    let convolved = silu(&conv_sum)?;
    let (kd, group) = (kh * dk, vh / kh);
    let mut query = Vec::with_capacity(n * vh * dk);
    let mut key = Vec::with_capacity(n * vh * dk);
    let mut value = Vec::with_capacity(n * vh * dv);
    let scale = (dk as f32).sqrt();
    for t in 0..n {
        let c = &convolved[t * channels..(t + 1) * channels];
        for h in 0..vh {
            let k0 = (h / group) * dk;
            query.extend(
                crate::ops::l2_norm(&c[k0..k0 + dk])?
                    .iter()
                    .map(|q| q / scale),
            );
            key.extend(crate::ops::l2_norm(&c[kd + k0..kd + k0 + dk])?);
        }
        value.extend_from_slice(&c[2 * kd..]);
    }
    let beta: Vec<f32> = b.iter().map(|&v| sigmoid(v)).collect();
    let decay: Vec<f32> = a
        .iter()
        .enumerate()
        .map(|(i, &v)| -w.a_log[i % vh].exp() * softplus(v + w.dt[i % vh]))
        .collect();
    // The recurrence, each head on its own.
    let per_head: Vec<(Vec<f32>, Vec<Vec<f32>>)> = (0..vh)
        .into_par_iter()
        .map(|h| {
            let mut s = vec![0f32; dk * dv];
            let mut out = Vec::with_capacity(n * dv);
            let mut saved = Vec::with_capacity(n.div_ceil(chunk));
            for t in 0..n {
                if t % chunk == 0 {
                    saved.push(s.clone());
                }
                let i = t * vh + h;
                out.extend(rule_step(
                    &query[i * dk..(i + 1) * dk],
                    &key[i * dk..(i + 1) * dk],
                    &value[i * dv..(i + 1) * dv],
                    decay[i],
                    beta[i],
                    &mut s,
                ));
            }
            (out, saved)
        })
        .collect();
    let mut core = vec![0.; n * vh * dv];
    let mut checkpoints = Vec::with_capacity(vh);
    for (h, (out, saved)) in per_head.into_iter().enumerate() {
        for t in 0..n {
            core[(t * vh + h) * dv..(t * vh + h + 1) * dv]
                .copy_from_slice(&out[t * dv..(t + 1) * dv]);
        }
        checkpoints.push(saved);
    }
    finite(&core)?;
    let mut gated = Vec::with_capacity(core.len());
    for (o, zz) in core.chunks_exact(dv).zip(z.chunks_exact(dv)) {
        let normed = rms_norm(o, &w.norm, w.config.eps, Scale::Direct)?;
        gated.extend(normed.iter().zip(silu(zz)?).map(|(a, b)| a * b));
    }
    let output = w.out.project_batch(&gated, n)?;
    Ok(DeltaForward {
        x: x.to_vec(),
        projected,
        conv_sum,
        a,
        b,
        z,
        query,
        key,
        value,
        beta,
        decay,
        core,
        gated,
        output,
        checkpoints,
        chunk,
        n,
    })
}

/// Gradient toward the input `x [n, hidden]`, for `dout [n, hidden]`.
pub fn backward(w: &DeltaWeights, f: &DeltaForward, dout: &[f32]) -> Result<Vec<f32>> {
    let dm = dims(w)?;
    let Dims {
        kh,
        vh,
        dk,
        dv,
        kernel,
        channels,
    } = dm;
    let n = f.n;
    require(
        dout.len() == f.output.len(),
        "DeltaNet backward shape mismatch",
    )?;
    let dgated = w.out.project_transposed_batch(dout, n)?;
    // gated = RMSNorm(o) ⊙ SiLU(z), per value head.
    let mut dcore = vec![0.; f.core.len()];
    let mut dz = vec![0.; f.z.len()];
    for (i, ((o, zz), dg)) in f
        .core
        .chunks_exact(dv)
        .zip(f.z.chunks_exact(dv))
        .zip(dgated.chunks_exact(dv))
        .enumerate()
    {
        let normed = rms_norm(o, &w.norm, w.config.eps, Scale::Direct)?;
        let act = silu(zz)?;
        let dnormed: Vec<f32> = dg.iter().zip(&act).map(|(a, b)| a * b).collect();
        let dact: Vec<f32> = dg.iter().zip(&normed).map(|(a, b)| a * b).collect();
        dz[i * dv..(i + 1) * dv].copy_from_slice(&silu_backward(zz, &dact)?);
        let g = rms_norm_backward(o, &w.norm, w.config.eps, Scale::Direct, &dnormed)?;
        dcore[i * dv..(i + 1) * dv].copy_from_slice(&g.dx);
    }
    // The recurrence, each head on its own.
    let per_head: Vec<RuleGrad> = (0..vh)
        .into_par_iter()
        .map(|h| {
            let dc: Vec<f32> = (0..n)
                .flat_map(|t| {
                    dcore[(t * vh + h) * dv..(t * vh + h + 1) * dv]
                        .iter()
                        .copied()
                })
                .collect();
            rule_backward(f, h, dm, &dc)
        })
        .collect();
    // The gates: β = σ(b), g = −exp(A_log)·softplus(a + dt).
    let mut da = vec![0.; f.a.len()];
    let mut db = vec![0.; f.b.len()];
    for t in 0..n {
        for (h, g) in per_head.iter().enumerate() {
            let i = t * vh + h;
            let beta = f.beta[i];
            db[i] = g.dbeta[t] * beta * (1. - beta);
            let u = f.a[i] + w.dt[h];
            let dsoftplus = if u > 20. { 1. } else { sigmoid(u) };
            da[i] = g.dg[t] * -w.a_log[h].exp() * dsoftplus;
        }
    }
    // Back through the L2 norms, to the convolved channels. Value heads that
    // share a key head add their gradients into it.
    let (kd, group) = (kh * dk, vh / kh);
    let scale = (dk as f32).sqrt();
    let convolved = silu(&f.conv_sum)?;
    let mut dconvolved = vec![0.; n * channels];
    for t in 0..n {
        let c = &convolved[t * channels..(t + 1) * channels];
        let dc = &mut dconvolved[t * channels..(t + 1) * channels];
        for (h, g) in per_head.iter().enumerate() {
            let k0 = (h / group) * dk;
            let dq: Vec<f32> = g.dq[t * dk..(t + 1) * dk]
                .iter()
                .map(|v| v / scale)
                .collect();
            for (a, b) in dc[k0..k0 + dk]
                .iter_mut()
                .zip(l2_backward(&c[k0..k0 + dk], &dq))
            {
                *a += b;
            }
            let dkey = &g.dk[t * dk..(t + 1) * dk];
            for (a, b) in dc[kd + k0..kd + k0 + dk]
                .iter_mut()
                .zip(l2_backward(&c[kd + k0..kd + k0 + dk], dkey))
            {
                *a += b;
            }
            dc[2 * kd + h * dv..2 * kd + (h + 1) * dv].copy_from_slice(&g.dv[t * dv..(t + 1) * dv]);
        }
    }
    let dconv_sum = silu_backward(&f.conv_sum, &dconvolved)?;
    // The convolution: each sum read `kernel` past projections.
    let mut dprojected = vec![0.; n * channels];
    for ch in 0..channels {
        let wc = w.conv.row(ch)?;
        for t in 0..n {
            let d = dconv_sum[t * channels + ch];
            for (j, &wj) in wc.iter().enumerate() {
                if let Some(s) = (t + j + 1).checked_sub(kernel) {
                    dprojected[s * channels + ch] += wj * d;
                }
            }
        }
    }
    // x feeds four projections; their gradients add.
    let mut dx = w.qkv.project_transposed_batch(&dprojected, n)?;
    for (m, g) in [(&w.a, &da), (&w.b, &db), (&w.z, &dz)] {
        for (a, b) in dx.iter_mut().zip(m.project_transposed_batch(g, n)?) {
            *a += b;
        }
    }
    finite(&dx)?;
    Ok(dx)
}
