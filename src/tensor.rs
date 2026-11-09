use std::sync::Arc;
use std::{error::Error, fmt};
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    Reference,
    Optimized,
    Vector,
}
impl Backend {
    pub fn inference() -> Self {
        Self::Vector
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferenceError(pub String);
impl fmt::Display for InferenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl Error for InferenceError {}
pub type Result<T> = std::result::Result<T, InferenceError>;
pub fn require(ok: bool, message: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(InferenceError(message.into()))
    }
}
pub fn finite(values: &[f32]) -> Result<()> {
    require(!values.is_empty(), "empty tensor")?;
    require(values.iter().all(|v| v.is_finite()), "NaN/Inf detected")
}

/// Row-major weights [out, in]. Fields are private to preserve shape/finite checks.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    // Vecの所有権をArcへ移し、巨大なf32配列をArc<[f32]>へコピーしない。
    // 外へはsliceしか公開しないため、共有重みの不変性は保たれる。
    data: Arc<Vec<f32>>,
}
impl Matrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f32>) -> Result<Self> {
        require(
            rows > 0 && cols > 0 && rows.checked_mul(cols) == Some(data.len()),
            "invalid matrix shape",
        )?;
        finite(&data)?;
        Ok(Self {
            rows,
            cols,
            data: Arc::new(data),
        })
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
    pub fn column(&self, j: usize) -> Result<Vec<f32>> {
        require(j < self.cols, "column out of bounds")?;
        Ok((0..self.rows)
            .map(|i| self.data[i * self.cols + j])
            .collect())
    }
    pub fn linear(&self, x: &[f32]) -> Result<Vec<f32>> {
        require(x.len() == self.cols, "linear input shape mismatch")?;
        finite(x)?;
        let mut out = vec![0.; self.rows];
        for (row, y) in self.data.chunks_exact(self.cols).zip(&mut out) {
            for (&w, &v) in row.iter().zip(x) {
                *y += w * v;
            }
        }
        finite(&out)?;
        Ok(out)
    }
    pub fn project(&self, x: &[f32], backend: Backend) -> Result<Vec<f32>> {
        if backend == Backend::Reference {
            return self.linear(x);
        }
        require(x.len() == self.cols, "linear input shape mismatch")?;
        finite(x)?;
        let mut out = vec![0.; self.rows];
        if backend == Backend::Vector {
            // 単一ベクトルでは行列をsgemm用にpackせず、各行を直接読む。
            // 8本の独立した和はLLVMがSIMD化できる。加算順序が参照版と
            // 異なるため、実モデル全体での同値検証後にだけ採用する。
            for (row, y) in self.data.chunks_exact(self.cols).zip(&mut out) {
                let mut sums = [0_f32; 8];
                let n = self.cols / 8 * 8;
                for (w, v) in row[..n].chunks_exact(8).zip(x[..n].chunks_exact(8)) {
                    for i in 0..8 {
                        sums[i] += w[i] * v[i];
                    }
                }
                *y = sums.iter().sum();
                for i in n..self.cols {
                    *y += row[i] * x[i];
                }
            }
            finite(&out)?;
            return Ok(out);
        }
        // SAFETY: A is [rows,cols], B [cols,1], C [rows,1], contiguous.
        // C is newly allocated and cannot alias A/B. Both inputs are read-only
        // (they may alias each other). Constructor checked dimensions/lengths.
        unsafe {
            matrixmultiply::sgemm(
                self.rows,
                self.cols,
                1,
                1.,
                self.data.as_ptr(),
                self.cols as isize,
                1,
                x.as_ptr(),
                1,
                1,
                0.,
                out.as_mut_ptr(),
                1,
                1,
            );
        }
        finite(&out)?;
        Ok(out)
    }
    pub fn row(&self, i: usize) -> Result<&[f32]> {
        require(i < self.rows, "row out of bounds")?;
        Ok(&self.data[i * self.cols..(i + 1) * self.cols])
    }
    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        require(self.cols == rhs.rows, "matmul shape mismatch")?;
        let size = self
            .rows
            .checked_mul(rhs.cols)
            .ok_or_else(|| InferenceError("matrix size overflow".into()))?;
        let mut data = vec![0.; size];
        for i in 0..self.rows {
            for j in 0..rhs.cols {
                for k in 0..self.cols {
                    data[i * rhs.cols + j] +=
                        self.data[i * self.cols + k] * rhs.data[k * rhs.cols + j];
                }
            }
        }
        Self::new(self.rows, rhs.cols, data)
    }
}
