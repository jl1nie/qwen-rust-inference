use std::{borrow::Cow, sync::Arc};
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

/// Matrix elements as stored: f32, or the exact BF16 bit pattern of the
/// released checkpoint (upper 16 bits of an f32). Kernels widen BF16 with
/// `f32::from_bits(b << 16)`, which is exact, so every multiply sees the same
/// f32 value as with an f32 copy, at half the memory.
pub trait Element: Copy + Send + Sync {
    fn f32(self) -> f32;
}
impl Element for f32 {
    #[inline(always)]
    fn f32(self) -> f32 {
        self
    }
}
impl Element for u16 {
    #[inline(always)]
    fn f32(self) -> f32 {
        f32::from_bits((self as u32) << 16)
    }
}
/// Zeroed u16 buffer whose allocation starts on a page boundary and spans
/// whole pages. Metal can then use the same memory as a GPU buffer without a
/// copy (`newBufferWithBytesNoCopy` needs both). Padding stays zero.
pub struct AlignedBuf {
    pointer: std::ptr::NonNull<u16>,
    len: usize,
    layout: std::alloc::Layout,
}
/// 16 KiB: the Apple Silicon page size (a multiple of 4 KiB pages elsewhere).
pub const PAGE: usize = 16384;
impl AlignedBuf {
    pub fn zeroed(len: usize) -> Result<Self> {
        let bytes = len
            .checked_mul(2)
            .and_then(|b| b.max(1).checked_next_multiple_of(PAGE))
            .ok_or_else(|| InferenceError("buffer size overflow".into()))?;
        let layout = std::alloc::Layout::from_size_align(bytes, PAGE)
            .map_err(|e| InferenceError(e.to_string()))?;
        // SAFETY: non-zero size; checked for null below.
        let raw = unsafe { std::alloc::alloc_zeroed(layout) };
        let pointer = std::ptr::NonNull::new(raw.cast::<u16>())
            .ok_or_else(|| InferenceError("out of memory".into()))?;
        Ok(Self {
            pointer,
            len,
            layout,
        })
    }
    pub fn from_slice(values: &[u16]) -> Result<Self> {
        let mut b = Self::zeroed(values.len())?;
        b.copy_from_slice(values);
        Ok(b)
    }
    /// Whole-page byte length of the allocation (>= 2 * len).
    pub fn allocation_bytes(&self) -> usize {
        self.layout.size()
    }
    /// The elements as raw bytes, for reading a file straight into them.
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: the allocation holds at least 2 * len initialized bytes and
        // any byte pattern is a valid u16.
        unsafe { std::slice::from_raw_parts_mut(self.pointer.as_ptr().cast(), self.len * 2) }
    }
}
impl std::ops::Deref for AlignedBuf {
    type Target = [u16];
    fn deref(&self) -> &[u16] {
        // SAFETY: zero-initialized allocation of at least len u16, owned by self.
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), self.len) }
    }
}
impl std::ops::DerefMut for AlignedBuf {
    fn deref_mut(&mut self) -> &mut [u16] {
        // SAFETY: as in deref, and &mut self is exclusive.
        unsafe { std::slice::from_raw_parts_mut(self.pointer.as_ptr(), self.len) }
    }
}
impl Drop for AlignedBuf {
    fn drop(&mut self) {
        // SAFETY: allocated in `zeroed` with this layout.
        unsafe { std::alloc::dealloc(self.pointer.as_ptr().cast(), self.layout) }
    }
}
// SAFETY: plain owned memory; shared access is read-only through Deref.
unsafe impl Send for AlignedBuf {}
unsafe impl Sync for AlignedBuf {}
impl fmt::Debug for AlignedBuf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AlignedBuf({} x u16)", self.len)
    }
}
impl PartialEq for AlignedBuf {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}
#[derive(Debug, Clone, PartialEq)]
enum Storage {
    F32(Arc<Vec<f32>>),
    Bf16(Arc<AlignedBuf>),
}
/// Row-major weights [out, in]. Fields are private to preserve shape/finite checks.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    // Vecの所有権をArcへ移し、巨大な配列をArc<[T]>へコピーしない。
    // 外へは読み取りだけを公開するため、共有重みの不変性は保たれる。
    data: Storage,
}
/// Dispatch a generic body over the storage type.
macro_rules! with_data {
    ($m:expr, $d:ident => $body:expr) => {
        match &$m.data {
            Storage::F32(v) => {
                let $d: &[f32] = v;
                $body
            }
            Storage::Bf16(v) => {
                let $d: &[u16] = v;
                $body
            }
        }
    };
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
            data: Storage::F32(Arc::new(data)),
        })
    }
    /// BF16 bit patterns (as stored in safetensors), kept without widening.
    pub fn from_bf16(rows: usize, cols: usize, bits: Vec<u16>) -> Result<Self> {
        Self::from_bf16_buf(rows, cols, AlignedBuf::from_slice(&bits)?)
    }
    /// As `from_bf16`, taking a page-aligned buffer (no copy).
    pub fn from_bf16_buf(rows: usize, cols: usize, bits: AlignedBuf) -> Result<Self> {
        require(
            rows > 0 && cols > 0 && rows.checked_mul(cols) == Some(bits.len()),
            "invalid matrix shape",
        )?;
        // Exponent all ones = Inf/NaN.
        require(
            bits.iter().all(|&b| b & 0x7f80 != 0x7f80),
            "NaN/Inf detected",
        )?;
        Ok(Self {
            rows,
            cols,
            data: Storage::Bf16(Arc::new(bits)),
        })
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    /// Number of elements (rows x cols).
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Bytes held for the elements (2 per BF16, 4 per f32).
    pub fn bytes(&self) -> usize {
        match &self.data {
            Storage::F32(v) => v.len() * 4,
            Storage::Bf16(v) => v.len() * 2,
        }
    }
    /// All elements as f32 (borrowed for f32 storage, widened for BF16).
    pub fn values(&self) -> Cow<'_, [f32]> {
        match &self.data {
            Storage::F32(v) => Cow::Borrowed(v.as_slice()),
            Storage::Bf16(v) => Cow::Owned(v.iter().map(|b| b.f32()).collect()),
        }
    }
    /// The BF16 bit patterns when stored as BF16.
    pub fn bf16_bits(&self) -> Option<&[u16]> {
        match &self.data {
            Storage::Bf16(v) => Some(v),
            Storage::F32(_) => None,
        }
    }
    /// The page-aligned BF16 allocation itself, shared (for a device that
    /// reads host memory directly; the clone keeps it alive).
    pub fn bf16_shared(&self) -> Option<Arc<AlignedBuf>> {
        match &self.data {
            Storage::Bf16(v) => Some(v.clone()),
            Storage::F32(_) => None,
        }
    }
    pub fn column(&self, j: usize) -> Result<Vec<f32>> {
        require(j < self.cols, "column out of bounds")?;
        Ok(with_data!(self, d => (0..self.rows)
            .map(|i| d[i * self.cols + j].f32())
            .collect()))
    }
    pub fn linear(&self, x: &[f32]) -> Result<Vec<f32>> {
        require(x.len() == self.cols, "linear input shape mismatch")?;
        finite(x)?;
        let mut out = vec![0.; self.rows];
        with_data!(self, d => linear_rows(d, self.cols, x, &mut out));
        finite(&out)?;
        Ok(out)
    }
    /// sgemm needs f32: BF16 storage is widened 64 rows at a time into a
    /// scratch buffer (exact). Each output row is an independent dot product,
    /// so blocking the rows leaves every sum as with f32 storage.
    fn sgemm_rows(&self, x: &[f32], out: &mut [f32]) {
        const BLOCK: usize = 64;
        match &self.data {
            Storage::F32(d) => sgemv(d, self.rows, self.cols, x, out),
            Storage::Bf16(d) => {
                let mut scratch = vec![0_f32; BLOCK * self.cols];
                for (rows, out) in d.chunks(BLOCK * self.cols).zip(out.chunks_mut(BLOCK)) {
                    let scratch = &mut scratch[..rows.len()];
                    for (f, &b) in scratch.iter_mut().zip(rows) {
                        *f = b.f32();
                    }
                    sgemv(scratch, out.len(), self.cols, x, out);
                }
            }
        }
    }
    pub fn project(&self, x: &[f32], backend: Backend) -> Result<Vec<f32>> {
        if backend == Backend::Reference {
            return self.linear(x);
        }
        require(x.len() == self.cols, "linear input shape mismatch")?;
        finite(x)?;
        let mut out = vec![0.; self.rows];
        if backend == Backend::Vector {
            with_data!(self, d => vector_rows(d, self.cols, x, &mut out));
        } else {
            self.sgemm_rows(x, &mut out);
        }
        finite(&out)?;
        Ok(out)
    }
    /// One row as f32 (borrowed for f32 storage, widened for BF16).
    pub fn row(&self, i: usize) -> Result<Cow<'_, [f32]>> {
        require(i < self.rows, "row out of bounds")?;
        let range = i * self.cols..(i + 1) * self.cols;
        Ok(match &self.data {
            Storage::F32(v) => Cow::Borrowed(&v[range]),
            Storage::Bf16(v) => Cow::Owned(v[range].iter().map(|b| b.f32()).collect()),
        })
    }
    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        require(self.cols == rhs.rows, "matmul shape mismatch")?;
        let size = self
            .rows
            .checked_mul(rhs.cols)
            .ok_or_else(|| InferenceError("matrix size overflow".into()))?;
        let (a, b) = (self.values(), rhs.values());
        let mut data = vec![0.; size];
        for i in 0..self.rows {
            for j in 0..rhs.cols {
                for k in 0..self.cols {
                    data[i * rhs.cols + j] += a[i * self.cols + k] * b[k * rhs.cols + j];
                }
            }
        }
        Self::new(self.rows, rhs.cols, data)
    }
}
/// 教育用の参照版：行ごとに左から順に足す。
fn linear_rows<T: Element>(d: &[T], cols: usize, x: &[f32], out: &mut [f32]) {
    for (row, y) in d.chunks_exact(cols).zip(out) {
        for (&w, &v) in row.iter().zip(x) {
            *y += w.f32() * v;
        }
    }
}
/// 単一ベクトルでは行列をsgemm用にpackせず、各行を直接読む。
/// 8本の独立した和はLLVMがSIMD化できる。加算順序が参照版と
/// 異なるため、実モデル全体での同値検証後にだけ採用する。
/// BF16 weights are widened right where they are multiplied: the CPU decode is
/// compute-bound, and widening into a scratch buffer first was ~30% slower
/// (ADR 0012). The widening is exact and the sums keep their order.
fn vector_rows<T: Element>(d: &[T], cols: usize, x: &[f32], out: &mut [f32]) {
    for (row, y) in d.chunks_exact(cols).zip(out) {
        let mut sums = [0_f32; 8];
        let n = cols / 8 * 8;
        for (w, v) in row[..n].chunks_exact(8).zip(x[..n].chunks_exact(8)) {
            for i in 0..8 {
                sums[i] += w[i].f32() * v[i];
            }
        }
        *y = sums.iter().sum();
        for i in n..cols {
            *y += row[i].f32() * x[i];
        }
    }
}
fn sgemv(a: &[f32], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) {
    debug_assert!(a.len() == rows * cols && x.len() == cols && out.len() == rows);
    // SAFETY: A is [rows,cols], B [cols,1], C [rows,1], contiguous (checked by
    // the callers' shapes). C is a distinct mutable slice and cannot alias A/B.
    unsafe {
        matrixmultiply::sgemm(
            rows,
            cols,
            1,
            1.,
            a.as_ptr(),
            cols as isize,
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
}
#[cfg(test)]
mod tests {
    use super::*;
    /// BF16 storage must give exactly the f32-storage results on every
    /// backend (the row-blocked sgemm path included: 150 rows is not a
    /// multiple of its 64-row block), and the same rows/columns/values.
    #[test]
    fn bf16_storage_is_bit_identical_to_f32() {
        let (rows, cols) = (150, 37);
        let mut seed = 0x2545_f491_u32;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        let bits: Vec<u16> = (0..rows * cols)
            .map(|_| {
                // Finite BF16: random sign/mantissa, exponent kept moderate.
                let r = next();
                ((r & 0x807f) | (((r >> 8) % 16 + 118) << 7)) as u16
            })
            .collect();
        let values: Vec<f32> = bits.iter().map(|b| b.f32()).collect();
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.37).sin()).collect();
        let a = Matrix::from_bf16(rows, cols, bits).unwrap();
        let b = Matrix::new(rows, cols, values.clone()).unwrap();
        assert_eq!(a.values().as_ref(), values.as_slice());
        assert_eq!(a.bytes() * 2, b.bytes());
        for backend in [Backend::Reference, Backend::Vector, Backend::Optimized] {
            let (p, q) = (
                a.project(&x, backend).unwrap(),
                b.project(&x, backend).unwrap(),
            );
            assert!(
                p.iter().zip(&q).all(|(p, q)| p.to_bits() == q.to_bits()),
                "{backend:?}"
            );
        }
        assert_eq!(a.row(149).unwrap(), b.row(149).unwrap());
        assert_eq!(a.column(36).unwrap(), b.column(36).unwrap());
        assert!(Matrix::from_bf16(1, 1, vec![0x7f80]).is_err()); // Inf
        assert!(Matrix::from_bf16(1, 1, vec![0xffc1]).is_err()); // NaN
        assert!(Matrix::from_bf16(2, 1, vec![0]).is_err()); // shape
    }
}
