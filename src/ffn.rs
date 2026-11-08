use crate::{
    ops::{multiply, silu},
    tensor::{require, Matrix, Result},
};
/// Immutable weights are separate from per-call activations. No inference state in Phase 1.
#[derive(Debug, Clone, PartialEq)]
pub struct FfnWeights {
    gate: Matrix,
    up: Matrix,
    down: Matrix,
}
#[derive(Debug, Clone, PartialEq)]
pub struct FfnActivations {
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub activated_gate: Vec<f32>,
    pub product: Vec<f32>,
    pub intervened_product: Vec<f32>,
    pub output: Vec<f32>,
}
impl FfnWeights {
    pub fn new(gate: Matrix, up: Matrix, down: Matrix) -> Result<Self> {
        require(
            gate.rows() == up.rows()
                && gate.cols() == up.cols()
                && down.rows() == gate.cols()
                && down.cols() == gate.rows(),
            "FFN weight shape mismatch",
        )?;
        Ok(Self { gate, up, down })
    }
    pub fn writing_vector(&self, unit: usize) -> Result<Vec<f32>> {
        self.down.column(unit)
    }
    pub fn forward(&self, x: &[f32], zero_unit: Option<usize>) -> Result<FfnActivations> {
        if let Some(j) = zero_unit {
            require(j < self.gate.rows(), "FFN unit out of bounds")?;
        }
        let gate = self.gate.linear(x)?;
        let up = self.up.linear(x)?;
        let activated_gate = silu(&gate)?;
        let product = multiply(&activated_gate, &up)?;
        let mut intervened_product = product.clone();
        if let Some(j) = zero_unit {
            intervened_product[j] = 0.;
        }
        let output = self.down.linear(&intervened_product)?;
        Ok(FfnActivations {
            gate,
            up,
            activated_gate,
            product,
            intervened_product,
            output,
        })
    }
}
/// Deterministic, dependency-free LCG fixture; not a model initialization recommendation.
pub fn educational_demo() -> Result<(FfnWeights, Vec<f32>)> {
    let mut state = 42_u32;
    let mut values = |n| {
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                ((state >> 24) as f32 - 128.) / 256.
            })
            .collect()
    };
    let gate = Matrix::new(16, 8, values(128))?;
    let up = Matrix::new(16, 8, values(128))?;
    let down = Matrix::new(8, 16, values(128))?;
    let x = values(8);
    Ok((FfnWeights::new(gate, up, down)?, x))
}
