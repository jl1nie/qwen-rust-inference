use crate::tensor::{finite, require, Result};
fn binary(a: &[f32], b: &[f32], op: impl Fn(f32, f32) -> f32) -> Result<Vec<f32>> {
    require(a.len() == b.len(), "elementwise shape mismatch")?;
    finite(a)?;
    finite(b)?;
    let out: Vec<_> = a.iter().zip(b).map(|(&x, &y)| op(x, y)).collect();
    finite(&out)?;
    Ok(out)
}
pub fn add(a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
    binary(a, b, |x, y| x + y)
}
pub fn multiply(a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
    binary(a, b, |x, y| x * y)
}
pub fn silu(x: &[f32]) -> Result<Vec<f32>> {
    finite(x)?;
    let out: Vec<_> = x
        .iter()
        .map(|&v| {
            let sigmoid = if v >= 0. {
                1. / (1. + (-v).exp())
            } else {
                let e = v.exp();
                e / (1. + e)
            };
            v * sigmoid
        })
        .collect();
    finite(&out)?;
    Ok(out)
}
pub fn softmax(x: &[f32]) -> Result<Vec<f32>> {
    finite(x)?;
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<_> = x.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    require(sum.is_finite() && sum > 0., "invalid softmax denominator")?;
    let out: Vec<_> = exp.iter().map(|e| e / sum).collect();
    finite(&out)?;
    Ok(out)
}
#[derive(Debug, Clone, Copy)]
pub enum Scale {
    Direct,
    QwenZeroCentered,
}
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32, scale: Scale) -> Result<Vec<f32>> {
    finite(x)?;
    finite(weight)?;
    require(x.len() == weight.len(), "RMSNorm shape mismatch")?;
    require(eps.is_finite() && eps > 0., "invalid RMSNorm epsilon")?;
    let mean = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let denominator = mean + eps;
    require(denominator.is_finite(), "RMSNorm variance overflow")?;
    let inverse = 1. / denominator.sqrt();
    let out: Vec<_> = x
        .iter()
        .zip(weight)
        .map(|(&v, &w)| {
            let s = match scale {
                Scale::Direct => w,
                Scale::QwenZeroCentered => 1. + w,
            };
            (v * inverse) * s
        })
        .collect();
    finite(&out)?;
    Ok(out)
}
