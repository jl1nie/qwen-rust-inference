use crate::{
    ops::{multiply, silu},
    tensor::{require, Backend, Matrix, Result},
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
    pub(crate) fn matrices(&self) -> [&Matrix; 3] {
        [&self.gate, &self.up, &self.down]
    }
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
    pub fn dimensions(&self) -> (usize, usize) {
        (self.gate.cols(), self.gate.rows())
    }
    pub fn forward_edited(
        &self,
        x: &[f32],
        backend: Backend,
        layer: usize,
        manifest: Option<&crate::edit::EditManifest>,
    ) -> Result<FfnActivations> {
        // gate/upはどちらも [intermediate,hidden]。同じxから別々の特徴を
        // 作り、SiLU(gate)がupの各成分を調節する。ここではトークンを混ぜない。
        let gate = self.gate.project(x, backend)?;
        let up = self.up.project(x, backend)?;
        let activated_gate = silu(&gate)?;
        let product = multiply(&activated_gate, &up)?;
        let mut intervened_product = product.clone();
        let correction = if let Some(m) = manifest {
            m.apply(layer, &mut intervened_product, &self.down)?
        } else {
            vec![0.; self.down.rows()]
        };
        // downは [hidden,intermediate]。出力は全jについて
        // product[j] * down[:,j] を足したもの。「ユニットを消す」を
        // 「その書込みベクトルを差し引く」として検証できる理由がこの式。
        let output = crate::ops::add(
            &self.down.project(&intervened_product, backend)?,
            &correction,
        )?;
        Ok(FfnActivations {
            gate,
            up,
            activated_gate,
            product,
            intervened_product,
            output,
        })
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
