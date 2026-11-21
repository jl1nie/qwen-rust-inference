//! Row-major matrix products for training.
//!
//! Inference computes `W·x` for one position at a time. Training scores many
//! positions at once, so the same products become GEMMs, plus two new ones
//! that backpropagation needs:
//!
//! | product | shape | used for |
//! |---|---|---|
//! | `Y = X·Wᵀ` | `[n,in]·[out,in]ᵀ → [n,out]` | forward |
//! | `dX = dY·W` | `[n,out]·[out,in] → [n,in]` | gradient toward the input |
//! | `dW += dYᵀ·X` | `[n,out]ᵀ·[n,in] → [out,in]` | gradient of the parameters |
//!
//! `W` keeps the inference layout `[out, in]`, so no transposed copy is made.
use crate::tensor::{finite, require, Matrix, Result};

/// A strided view for `sgemm`: element (i, j) is `data[i * rs + j * cs]`.
#[derive(Clone, Copy)]
pub(crate) struct View<'a> {
    pub data: &'a [f32],
    pub rs: usize,
    pub cs: usize,
}
impl<'a> View<'a> {
    /// Row-major `[rows, cols]`.
    pub fn rows(data: &'a [f32], cols: usize) -> Self {
        Self {
            data,
            rs: cols,
            cs: 1,
        }
    }
    /// The transpose of a row-major `[rows, cols]`: a `[cols, rows]` view.
    pub fn transposed(data: &'a [f32], cols: usize) -> Self {
        Self {
            data,
            rs: 1,
            cs: cols,
        }
    }
    fn fits(&self, rows: usize, cols: usize) -> bool {
        rows == 0 || cols == 0 || (rows - 1) * self.rs + (cols - 1) * self.cs < self.data.len()
    }
}

/// `C = alpha·A·B + beta·C`, with `(m, k, n)` the shapes A `[m,k]`, B `[k,n]`
/// and C `[m,n]` at row stride `rsc`. Panics on a view that does not fit (an
/// internal shape bug).
pub(crate) fn sgemm(
    (m, k, n): (usize, usize, usize),
    alpha: f32,
    a: View,
    b: View,
    beta: f32,
    c: &mut [f32],
    rsc: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    assert!(a.fits(m, k) && b.fits(k, n), "gemm operand out of bounds");
    assert!((m - 1) * rsc + n <= c.len(), "gemm output out of bounds");
    // SAFETY: every element read or written lies inside its slice (asserted
    // above), and C is a distinct mutable slice, so it cannot alias A or B.
    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            alpha,
            a.data.as_ptr(),
            a.rs as isize,
            a.cs as isize,
            b.data.as_ptr(),
            b.rs as isize,
            b.cs as isize,
            beta,
            c.as_mut_ptr(),
            rsc as isize,
            1,
        );
    }
}

/// Trainable parameters: row-major `[rows, cols]` in f32, the same layout as
/// `Matrix`. Unlike `Matrix`, it can be changed in place.
#[derive(Debug, Clone, PartialEq)]
pub struct Dense {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}
impl Dense {
    pub fn new(rows: usize, cols: usize, data: Vec<f32>) -> Result<Self> {
        require(
            rows > 0 && cols > 0 && rows.checked_mul(cols) == Some(data.len()),
            "invalid matrix shape",
        )?;
        finite(&data)?;
        Ok(Self { rows, cols, data })
    }
    pub fn zeros(rows: usize, cols: usize) -> Result<Self> {
        let len = rows
            .checked_mul(cols)
            .ok_or_else(|| crate::tensor::InferenceError("matrix size overflow".into()))?;
        Self::new(rows, cols, vec![0.; len])
    }
    /// An f32 copy of frozen weights (BF16 is widened exactly).
    pub fn from_matrix(m: &Matrix) -> Self {
        Self {
            rows: m.rows(),
            cols: m.cols(),
            data: m.values().into_owned(),
        }
    }
    pub fn to_matrix(&self) -> Result<Matrix> {
        Matrix::new(self.rows, self.cols, self.data.clone())
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    pub fn values(&self) -> &[f32] {
        &self.data
    }
    pub fn values_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }
    /// `Y = X·Wᵀ` for `n` inputs `X [n, cols]`: `Y [n, rows]`.
    pub fn forward(&self, x: &[f32], n: usize) -> Result<Vec<f32>> {
        require(x.len() == n * self.cols, "forward input shape mismatch")?;
        let mut y = vec![0.; n * self.rows];
        sgemm(
            (n, self.cols, self.rows),
            1.,
            View::rows(x, self.cols),
            View::transposed(&self.data, self.cols),
            0.,
            &mut y,
            self.rows,
        );
        Ok(y)
    }
    /// `dX = dY·W` for `dY [n, rows]`: `dX [n, cols]`.
    pub fn backward_input(&self, dy: &[f32], n: usize) -> Result<Vec<f32>> {
        require(dy.len() == n * self.rows, "backward input shape mismatch")?;
        let mut dx = vec![0.; n * self.cols];
        sgemm(
            (n, self.rows, self.cols),
            1.,
            View::rows(dy, self.rows),
            View::rows(&self.data, self.cols),
            0.,
            &mut dx,
            self.cols,
        );
        Ok(dx)
    }
    /// `self += alpha·dYᵀ·X` for `dY [n, rows]` and `X [n, cols]`: the
    /// parameter gradient summed over the `n` positions.
    pub fn accumulate(&mut self, alpha: f32, dy: &[f32], x: &[f32], n: usize) -> Result<()> {
        require(
            dy.len() == n * self.rows && x.len() == n * self.cols,
            "gradient shape mismatch",
        )?;
        sgemm(
            (self.rows, n, self.cols),
            alpha,
            View::transposed(dy, self.rows),
            View::rows(x, self.cols),
            1.,
            &mut self.data,
            self.cols,
        );
        Ok(())
    }
}
