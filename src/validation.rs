use crate::tensor::{require, Result};
use serde::Serialize;
#[derive(Debug, Clone, Serialize)]
pub struct ErrorMetrics {
    pub max_absolute: f64,
    pub max_relative: f64,
    pub cosine: f64,
    pub elements: usize,
}
pub fn compare(
    actual: &[f32],
    expected: &[f32],
    atol: f64,
    rtol: f64,
    where_: &str,
) -> Result<ErrorMetrics> {
    require(
        actual.len() == expected.len() && !actual.is_empty(),
        &format!("{where_}: comparison shape mismatch"),
    )?;
    let (mut absolute, mut relative, mut dot, mut aa, mut bb) = (0_f64, 0_f64, 0_f64, 0_f64, 0_f64);
    for (i, (&x, &y)) in actual.iter().zip(expected).enumerate() {
        let (x, y) = (f64::from(x), f64::from(y));
        let error = (x - y).abs();
        require(
            x.is_finite() && y.is_finite() && error <= atol + rtol * y.abs(),
            &format!("first divergence: {where_}[{i}]: actual={x} expected={y} abs={error}"),
        )?;
        absolute = absolute.max(error);
        relative = relative.max(error / y.abs().max(1e-12));
        dot += x * y;
        aa += x * x;
        bb += y * y;
    }
    let cosine = if aa == 0. && bb == 0. {
        1.
    } else if aa == 0. || bb == 0. {
        0.
    } else {
        dot / (aa * bb).sqrt()
    };
    Ok(ErrorMetrics {
        max_absolute: absolute,
        max_relative: relative,
        cosine,
        elements: actual.len(),
    })
}
pub fn top(logits: &[f32], n: usize) -> Vec<(usize, f32)> {
    let mut ranked: Vec<_> = logits.iter().copied().enumerate().collect();
    ranked.sort_by(|(i, a), (j, b)| b.total_cmp(a).then(i.cmp(j)));
    ranked.truncate(n);
    ranked
}
