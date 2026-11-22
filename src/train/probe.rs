//! Linear probes: is a label readable from a layer's vectors by one matrix?
//!
//! A probe is a multinomial logistic regression `softmax(W·z + b)` on the
//! standardized vector `z` of one site (a layer's exit, or the last FFN's
//! input), trained on some examples and scored on others. The model is not
//! changed. A linear probe is kept weak on purpose: what one matrix reads
//! from a site, the next layer's matrices can read too, while a strong probe
//! could compute the label itself.
//!
//! Examples come in groups (one prompt template each) and are scored leaving
//! one group out: the probe has never seen that wording, so a high accuracy
//! means the label is encoded the same way across wordings, not that 188
//! points in 1024 dimensions can be separated (they always can).
use crate::tensor::{require, Result};
use rayon::prelude::*;

/// How a probe is trained: full-batch Adam on the mean cross-entropy plus
/// `l2 / 2 · ‖W‖²`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProbeConfig {
    pub steps: usize,
    pub lr: f64,
    pub l2: f64,
}

impl Default for ProbeConfig {
    fn default() -> Self {
        Self {
            steps: 200,
            lr: 0.01,
            l2: 1e-3,
        }
    }
}

/// A trained probe, with the standardization of its training examples.
#[derive(Debug, Clone, PartialEq)]
pub struct Probe {
    pub dim: usize,
    pub classes: usize,
    /// `[classes, dim]`.
    pub w: Vec<f64>,
    pub b: Vec<f64>,
    pub mean: Vec<f64>,
    pub scale: Vec<f64>,
}

/// Per-dimension mean and 1/std of `rows` (`[n, dim]`).
fn standardization(rows: &[f32], dim: usize) -> (Vec<f64>, Vec<f64>) {
    let n = (rows.len() / dim) as f64;
    let mut mean = vec![0.0; dim];
    for r in rows.chunks(dim) {
        mean.iter_mut()
            .zip(r)
            .for_each(|(m, &x)| *m += x as f64 / n);
    }
    let mut var = vec![0.0; dim];
    for r in rows.chunks(dim) {
        var.iter_mut()
            .zip(r.iter().zip(&mean))
            .for_each(|(v, (&x, m))| *v += (x as f64 - m).powi(2) / n);
    }
    let scale = var.iter().map(|v| 1.0 / (v.sqrt() + 1e-6)).collect();
    (mean, scale)
}

impl Probe {
    fn standardized(&self, row: &[f32]) -> Vec<f64> {
        row.iter()
            .zip(self.mean.iter().zip(&self.scale))
            .map(|(&x, (m, s))| (x as f64 - m) * s)
            .collect()
    }
    fn logits(&self, z: &[f64]) -> Vec<f64> {
        self.w
            .chunks(self.dim)
            .zip(&self.b)
            .map(|(w, b)| b + w.iter().zip(z).map(|(a, x)| a * x).sum::<f64>())
            .collect()
    }
    /// The class the probe reads from `row`.
    pub fn predict(&self, row: &[f32]) -> usize {
        let l = self.logits(&self.standardized(row));
        (0..l.len())
            .max_by(|&i, &j| l[i].total_cmp(&l[j]))
            .unwrap_or(0)
    }
}

/// Trains a probe on `rows` (`[n, dim]`) with `labels` in `0..classes`.
pub fn fit(
    rows: &[f32],
    dim: usize,
    labels: &[usize],
    classes: usize,
    config: ProbeConfig,
) -> Result<Probe> {
    require(dim > 0 && !labels.is_empty(), "no examples to probe")?;
    require(rows.len() == labels.len() * dim, "rows and labels differ")?;
    require(labels.iter().all(|&c| c < classes), "label out of range")?;
    require(rows.iter().all(|x| x.is_finite()), "probe input not finite")?;
    let (mean, scale) = standardization(rows, dim);
    let mut p = Probe {
        dim,
        classes,
        w: vec![0.0; classes * dim],
        b: vec![0.0; classes],
        mean,
        scale,
    };
    let z: Vec<Vec<f64>> = rows.chunks(dim).map(|r| p.standardized(r)).collect();
    let n = labels.len() as f64;
    let params = classes * dim + classes;
    let (mut m, mut v) = (vec![0.0; params], vec![0.0; params]);
    let (b1, b2, eps) = (0.9f64, 0.999f64, 1e-8);
    for t in 1..=config.steps {
        let mut g = vec![0.0; params];
        for (x, &y) in z.iter().zip(labels) {
            let l = p.logits(x);
            let max = l.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let sum: f64 = l.iter().map(|a| (a - max).exp()).sum();
            for (c, a) in l.iter().enumerate() {
                let d = ((a - max).exp() / sum - if c == y { 1.0 } else { 0.0 }) / n;
                g[c * dim..(c + 1) * dim]
                    .iter_mut()
                    .zip(x)
                    .for_each(|(gi, xi)| *gi += d * xi);
                g[classes * dim + c] += d;
            }
        }
        g[..classes * dim]
            .iter_mut()
            .zip(&p.w)
            .for_each(|(gi, w)| *gi += config.l2 * w);
        let (c1, c2) = (1.0 - b1.powi(t as i32), 1.0 - b2.powi(t as i32));
        for (k, gk) in g.iter().enumerate() {
            m[k] = b1 * m[k] + (1.0 - b1) * gk;
            v[k] = b2 * v[k] + (1.0 - b2) * gk * gk;
            let step = config.lr * (m[k] / c1) / ((v[k] / c2).sqrt() + eps);
            if k < classes * dim {
                p.w[k] -= step;
            } else {
                p.b[k - classes * dim] -= step;
            }
        }
    }
    Ok(p)
}

/// Leave-one-group-out accuracy: for each group, a probe trained on the
/// other groups reads that group's examples. `(correct, count)` per group,
/// in the order of first appearance in `groups`.
pub fn held_out_groups(
    rows: &[f32],
    dim: usize,
    labels: &[usize],
    groups: &[usize],
    classes: usize,
    config: ProbeConfig,
) -> Result<Vec<(usize, usize)>> {
    require(groups.len() == labels.len(), "groups and labels differ")?;
    let mut order: Vec<usize> = Vec::new();
    for &g in groups {
        if !order.contains(&g) {
            order.push(g);
        }
    }
    require(order.len() >= 2, "need two groups to hold one out")?;
    order
        .par_iter()
        .map(|&held| {
            let train: Vec<usize> = (0..labels.len()).filter(|&i| groups[i] != held).collect();
            let x: Vec<f32> = train
                .iter()
                .flat_map(|&i| rows[i * dim..(i + 1) * dim].iter().copied())
                .collect();
            let y: Vec<usize> = train.iter().map(|&i| labels[i]).collect();
            let p = fit(&x, dim, &y, classes, config)?;
            let test: Vec<usize> = (0..labels.len()).filter(|&i| groups[i] == held).collect();
            let correct = test
                .iter()
                .filter(|&&i| p.predict(&rows[i * dim..(i + 1) * dim]) == labels[i])
                .count();
            Ok((correct, test.len()))
        })
        .collect()
}

/// The vectors a probe can read at the last position of `prompt`'s chat turn
/// followed by `prefix` (the position that predicts the reply's next token):
/// the embedding, each layer's exit, then the last FFN's input (`post_norm`,
/// what training the top FFN sees). `[site][hidden]`.
pub fn capture_sites(
    model: &crate::model::Model,
    codec: &dyn crate::text::TextCodec,
    prompt: &str,
    prefix: &str,
    backend: crate::tensor::Backend,
) -> Result<Vec<Vec<f32>>> {
    let ids = codec.encode(&format!(
        "{}{prefix}",
        crate::text::user_chat(prompt, false)
    ))?;
    require(!ids.is_empty(), "nothing to read")?;
    let last = ids.len() - 1;
    let top = model.blocks.len() - 1;
    let mut sites = Vec::new();
    let mut state = model.new_state();
    model.prefill_observed(&ids, &mut state, backend, None, |i, a| {
        if i == last {
            sites.push(a.embedding.clone());
            sites.extend(a.layers.iter().cloned());
            sites.push(a.blocks[top].post_norm.clone());
        }
        Ok(())
    })?;
    require(!sites.is_empty(), "the last position was not observed")?;
    Ok(sites)
}

/// The names of `capture_sites`' sites for a model of `layers` layers.
pub fn site_names(layers: usize) -> Vec<String> {
    std::iter::once("embedding".to_string())
        .chain((0..layers).map(|l| format!("layer {l} exit")))
        .chain(std::iter::once(format!("layer {} FFN input", layers - 1)))
        .collect()
}
