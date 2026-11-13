//! Experimental device-resident path (FP32 activations/state; weights FP32 or
//! exact BF16) shared by the ROCm (HIP) and Metal
//! bridges. Read spec/adr-0004.md and spec/adr-0005.md before changing layouts.
//! We deliberately keep device ownership OUT of tensor::Backend: a host slice
//! projection API cannot express GPU-resident weights, scratch, and run state.
//! Both bridges export the same versioned C ABI; `Kind` stops a library built
//! for one engine (or the CPU arithmetic test bridge) loading as another.
use crate::{
    attention::AttentionConfig,
    block::MixerWeights,
    deltanet::DeltaConfig,
    edit::EditManifest,
    model::Model,
    state::{DeltaState, KvCache, LayerState, ModelState},
    tensor::{finite, require, AlignedBuf, InferenceError, Matrix, Result},
    weights::{err, TextConfig},
};
use std::sync::Arc;
use std::{
    ffi::{c_char, c_int, c_void, CStr, CString},
    path::Path,
    ptr,
    rc::Rc,
};

#[link(name = "dl")]
extern "C" {
    fn dlopen(path: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
    fn dlclose(handle: *mut c_void) -> c_int;
}
#[repr(C)]
struct Params {
    p: [*mut f32; 8],
    n: [i32; 8],
    f: [f32; 4],
}
#[repr(i32)]
#[derive(Clone, Copy)]
enum Op {
    RmsNorm = 1,
    Add = 2,
    SiluProduct = 3,
    ConvSilu = 4,
    DeltaQk = 5,
    DeltaGates = 6,
    DeltaRule = 7,
    AttentionQuery = 8,
    AttentionKv = 9,
    AttentionRead = 10,
    EmbedBf16 = 11,
    /// Fusions (#19), issued only when `qr_max_op` advertises them; each
    /// computes exactly what the pair it replaces computes in sequence.
    /// Add, then RmsNorm of the sum.
    AddRmsNorm = 12,
    /// RmsNorm, then SiluProduct with a gate.
    RmsNormGate = 13,
    /// DeltaGates, then DeltaRule: the rule computes its gates itself
    /// (p5 = a, p6 = b, p7 = [dt | A_log]).
    DeltaRuleGated = 14,
    /// AttentionQuery and AttentionKv in one dispatch (p2 = [q_norm | k_norm]).
    AttentionQueryKv = 15,
}
/// Bridge identity returned by `qr_kind`. Production loading never accepts CpuTest.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    CpuTest = 0,
    Rocm = 1,
    Metal = 2,
}
impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::CpuTest => "CPU test bridge",
            Kind::Rocm => "ROCm",
            Kind::Metal => "Metal",
        }
    }
    /// `engine` field of trace records.
    fn trace_name(self) -> &'static str {
        match self {
            Kind::Metal => "metal",
            _ => "rocm",
        }
    }
    fn hint(self) -> &'static str {
        match self {
            Kind::Rocm => "check WSL driver/GPU visibility",
            Kind::Metal => "check that a Metal GPU is available and the kernels compiled",
            Kind::CpuTest => "CPU test bridge",
        }
    }
}
type Create = unsafe extern "C" fn(*mut *mut c_void) -> i32;
type Alloc = unsafe extern "C" fn(*mut c_void, *mut *mut f32, usize) -> i32;
type Wrap = unsafe extern "C" fn(*mut c_void, *mut *mut f32, *mut c_void, usize) -> i32;
type SetOption = unsafe extern "C" fn(*mut c_void, *const c_char, i32) -> i32;
type Transfer = unsafe extern "C" fn(*mut c_void, *mut f32, *const f32, usize) -> i32;
type Run = unsafe extern "C" fn(*mut c_void, i32, i32, *const Params) -> i32;
type Gemv = unsafe extern "C" fn(*mut c_void, *mut f32, *const f32, *const f32, i32, i32) -> i32;
type Gemm =
    unsafe extern "C" fn(*mut c_void, *mut f32, *const f32, *const f32, i32, i32, i32) -> i32;
/// One entry of `qr_gemm_bf16_group`: Y[batch, rows] = X . A^T, A packed BF16.
#[repr(C)]
struct Projection {
    y: *mut f32,
    a: *const f32,
    rows: i32,
}
type GemmGroup =
    unsafe extern "C" fn(*mut c_void, i32, *const Projection, *const f32, i32, i32) -> i32;
/// `qr_delta_mixer`: the slots of ConvSilu, DeltaQk, DeltaRuleGated and
/// RmsNormGate (offset 0) for one DeltaNet layer.
#[repr(C)]
struct DeltaArgs {
    convolved: *mut f32,
    projected: *mut f32,
    conv_w: *mut f32,
    conv_state: *mut f32,
    query: *mut f32,
    key: *mut f32,
    core: *mut f32,
    recurrent: *mut f32,
    a: *mut f32,
    b: *mut f32,
    gates: *mut f32,
    gated: *mut f32,
    norm_w: *mut f32,
    z: *mut f32,
    key_heads: i32,
    value_heads: i32,
    key_dim: i32,
    value_dim: i32,
    kernel: i32,
    tokens: i32,
    channels: i32,
    eps: f32,
}
type DeltaMixer = unsafe extern "C" fn(*mut c_void, *const DeltaArgs) -> i32;
/// `qr_gemm_bf16_add_norm`: Add + RmsNorm after a one-token GEMV.
#[repr(C)]
struct AddNormArgs {
    sum: *mut f32,
    u: *const f32,
    v: *const f32,
    weight: *const f32,
    norm: *mut f32,
    eps: f32,
    offset: f32,
}
type GemvAddNorm = unsafe extern "C" fn(
    *mut c_void,
    *mut f32,
    *const f32,
    *const f32,
    i32,
    i32,
    *const AddNormArgs,
) -> i32;
/// `qr_gemm_bf16_glu(ctx, product, gate_y, up_y, gate, up, x, rows, cols, batch)`.
type Glu = unsafe extern "C" fn(
    *mut c_void,
    *mut f32,
    *mut f32,
    *mut f32,
    *const f32,
    *const f32,
    *const f32,
    i32,
    i32,
    i32,
) -> i32;
/// Native ABI version. v2 added BF16 matrices (`qr_gemv_bf16`, op EmbedBf16)
/// and batched projection `qr_gemm`/`qr_gemm_bf16` for chunked prefill (ADR 0006);
/// v3 lets AttentionRead cover a whole chunk (n5 = rows, ADR 0007); n6 (per-head
/// scratch, ADR 0008) is only a hint, so v3 libraries ignoring it stay correct.
const ABI: i32 = 3;
/// Positions per decode attention split (Metal `attention_split`, ADR 0008).
const ATTENTION_SPLIT: usize = 32;
/// Per-head attention scratch floats (AttentionRead n6): a full score row for
/// the 3-pass read, or (max, denominator, head_dim partial sums) per split.
/// KV buffers hold whole blocks of this many positions: Metal's prefill
/// attention reads K/V in 32-position (ADR 0015) or 64-position (tensor ops,
/// ADR 0016) blocks straight from device memory and masks the positions past
/// each row's causal limit.
const KV_BLOCK: usize = 64;
fn kv_rows(capacity: usize) -> usize {
    capacity.next_multiple_of(KV_BLOCK)
}
fn score_row(capacity: usize, head_dim: usize) -> usize {
    capacity.max(capacity.div_ceil(ATTENTION_SPLIT) * (head_dim + 2))
}
/// How matrices are stored on the device. Activations and state stay FP32.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightFormat {
    /// Every matrix as FP32 (ABI v1 layout; doubles weight traffic).
    F32,
    /// A matrix is stored as BF16 only when EVERY value is exactly a BF16
    /// (low 16 mantissa bits zero), i.e. lossless; otherwise it stays FP32.
    Bf16Exact,
}
/// True when `v` survives f32 -> bf16 -> f32 unchanged (bit truncation).
fn bf16_exact(v: f32) -> bool {
    v.to_bits() & 0xffff == 0
}
struct Api {
    kind: Kind,
    library: *mut c_void,
    context: *mut c_void,
    alloc: Alloc,
    free: unsafe extern "C" fn(*mut c_void, *mut f32),
    write: Transfer,
    read: Transfer,
    copy: Transfer,
    run: Run,
    gemv: Gemv,
    gemv_bf16: Gemv,
    gemm: Gemm,
    gemm_bf16: Gemm,
    sync: unsafe extern "C" fn(*mut c_void) -> i32,
    /// Optional `qr_wrap`: use page-aligned host memory as a device buffer in
    /// place (Metal: unified memory; CPU test bridge). None = copy (HIP).
    wrap: Option<Wrap>,
    /// Optional `qr_set_option` (HIP, CPU test bridge): engine options such as
    /// the opt-in 16-bit prefill attention (ADR 0014). None = no options.
    set_option: Option<SetOption>,
    /// Optional `qr_gemm_bf16_group` (HIP, CPU test bridge): several BF16
    /// projections of the same input in one call, so decode issues fewer
    /// dispatches (#19). None = one `qr_gemm_bf16` per matrix (Metal).
    gemm_bf16_group: Option<GemmGroup>,
    /// Optional `qr_gemm_bf16_glu`: FFN gate/up and their SiluProduct in one
    /// call. None = two projections and op SiluProduct.
    gemm_bf16_glu: Option<Glu>,
    /// Optional `qr_delta_mixer`: a DeltaNet layer's four small ops in one
    /// call. None = four `qr_run` calls.
    delta_mixer: Option<DeltaMixer>,
    /// Optional `qr_gemm_bf16_add_norm`: a one-token GEMV whose last block
    /// also runs the following Add + RmsNorm. None = GEMV, then AddRmsNorm.
    gemm_bf16_add_norm: Option<GemvAddNorm>,
    /// Highest `qr_run` op (optional `qr_max_op`; 11 without it). Fused ops
    /// above 11 are issued only when the bridge runs them.
    max_op: i32,
    /// Optional `qr_snapshots` (ADR 0026): ConvSilu n3 / DeltaRule n6 store
    /// per-token state snapshots, so a verify pass can be rolled back.
    snapshots: bool,
    error: unsafe extern "C" fn(*mut c_void) -> *const c_char,
    info: unsafe extern "C" fn(*mut c_void) -> *const c_char,
    destroy: unsafe extern "C" fn(*mut c_void),
}
// Rc prevents Send/Sync: all allocations and the stream belong to this thread.
// Every Buffer keeps Api alive, so unloading cannot precede freeing its memory.
impl Drop for Api {
    fn drop(&mut self) {
        // SAFETY: constructed only after qr_create succeeds; pointers are live.
        unsafe {
            (self.destroy)(self.context);
            dlclose(self.library);
        }
    }
}
fn loader_error() -> InferenceError {
    // SAFETY: dlerror's thread-local NUL-terminated string is copied immediately.
    unsafe {
        let p = dlerror();
        err(if p.is_null() {
            "dynamic loader failed".into()
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        })
    }
}
impl Api {
    fn open(path: &Path, kind: Kind) -> Result<Rc<Self>> {
        use std::os::unix::ffi::OsStrExt;
        let name = CString::new(path.as_os_str().as_bytes()).map_err(err)?;
        // SAFETY: path is NUL-terminated; RTLD_NOW resolves all dependencies.
        let library = unsafe { dlopen(name.as_ptr(), 2) };
        if library.is_null() {
            return Err(loader_error());
        }
        let loaded = (|| {
            macro_rules! symbol {
                ($name:literal, $ty:ty) => {{
                    // SAFETY: we require this trusted local library's versioned C ABI.
                    let p = unsafe { dlsym(library, concat!($name, "\0").as_ptr().cast()) };
                    if p.is_null() {
                        return Err(loader_error());
                    }
                    unsafe { std::mem::transmute::<*mut c_void, $ty>(p) }
                }};
            }
            let abi = symbol!("qr_abi", unsafe extern "C" fn() -> i32);
            let bridge_kind = symbol!("qr_kind", unsafe extern "C" fn() -> i32);
            // SAFETY: symbols checked, functions have no arguments.
            require(
                unsafe { abi() == ABI && bridge_kind() == kind as i32 },
                &format!(
                    "{} bridge ABI/kind mismatch (wrong engine or CPU test bridge)",
                    kind.name()
                ),
            )?;
            let create = symbol!("qr_create", Create);
            // Resolve EVERY symbol before allocating a native context.
            let mut api = Self {
                kind,
                library,
                context: ptr::null_mut(),
                alloc: symbol!("qr_alloc", Alloc),
                free: symbol!("qr_free", unsafe extern "C" fn(*mut c_void, *mut f32)),
                write: symbol!("qr_write", Transfer),
                read: symbol!("qr_read", Transfer),
                copy: symbol!("qr_copy", Transfer),
                run: symbol!("qr_run", Run),
                gemv: symbol!("qr_gemv", Gemv),
                gemv_bf16: symbol!("qr_gemv_bf16", Gemv),
                gemm: symbol!("qr_gemm", Gemm),
                gemm_bf16: symbol!("qr_gemm_bf16", Gemm),
                sync: symbol!("qr_sync", unsafe extern "C" fn(*mut c_void) -> i32),
                // SAFETY: optional symbol of the same trusted library; a null
                // result just means this bridge copies weights instead.
                wrap: unsafe {
                    let p = dlsym(library, c"qr_wrap".as_ptr());
                    (!p.is_null()).then(|| std::mem::transmute::<*mut c_void, Wrap>(p))
                },
                // SAFETY: optional symbol as above; None means no engine options.
                set_option: unsafe {
                    let p = dlsym(library, c"qr_set_option".as_ptr());
                    (!p.is_null()).then(|| std::mem::transmute::<*mut c_void, SetOption>(p))
                },
                // SAFETY: optional symbol as above; None means one call per matrix.
                gemm_bf16_group: unsafe {
                    let p = dlsym(library, c"qr_gemm_bf16_group".as_ptr());
                    (!p.is_null()).then(|| std::mem::transmute::<*mut c_void, GemmGroup>(p))
                },
                // SAFETY: optional symbol as above; None means separate calls.
                gemm_bf16_glu: unsafe {
                    let p = dlsym(library, c"qr_gemm_bf16_glu".as_ptr());
                    (!p.is_null()).then(|| std::mem::transmute::<*mut c_void, Glu>(p))
                },
                // SAFETY: optional symbol as above; None means two calls.
                gemm_bf16_add_norm: unsafe {
                    let p = dlsym(library, c"qr_gemm_bf16_add_norm".as_ptr());
                    (!p.is_null()).then(|| std::mem::transmute::<*mut c_void, GemvAddNorm>(p))
                },
                // SAFETY: optional symbol as above; None means four ops.
                delta_mixer: unsafe {
                    let p = dlsym(library, c"qr_delta_mixer".as_ptr());
                    (!p.is_null()).then(|| std::mem::transmute::<*mut c_void, DeltaMixer>(p))
                },
                // SAFETY: optional symbol as above, no arguments.
                snapshots: unsafe {
                    let p = dlsym(library, c"qr_snapshots".as_ptr());
                    !p.is_null()
                        && std::mem::transmute::<*mut c_void, unsafe extern "C" fn() -> i32>(p)()
                            == 1
                },
                // SAFETY: optional symbol as above, no arguments.
                max_op: unsafe {
                    let p = dlsym(library, c"qr_max_op".as_ptr());
                    if p.is_null() {
                        Op::EmbedBf16 as i32
                    } else {
                        std::mem::transmute::<*mut c_void, unsafe extern "C" fn() -> i32>(p)()
                    }
                },
                error: symbol!(
                    "qr_error",
                    unsafe extern "C" fn(*mut c_void) -> *const c_char
                ),
                info: symbol!(
                    "qr_info",
                    unsafe extern "C" fn(*mut c_void) -> *const c_char
                ),
                destroy: symbol!("qr_destroy", unsafe extern "C" fn(*mut c_void)),
            };
            let mut context = ptr::null_mut();
            // SAFETY: writable out pointer; native create cleans up on failure.
            let status = unsafe { create(&mut context) };
            if status != 0 || context.is_null() {
                // Api's Drop needs a valid context; no native resource exists here.
                std::mem::forget(api);
                return Err(err(format!(
                    "{} context creation failed, status={status}; {}",
                    kind.name(),
                    kind.hint()
                )));
            }
            api.context = context;
            Ok(Rc::new(api))
        })();
        if loaded.is_err() {
            // SAFETY: context either was never created or was cleaned by create.
            unsafe {
                dlclose(library);
            }
        }
        loaded
    }
    fn check(&self, status: i32) -> Result<()> {
        if status == 0 {
            return Ok(());
        }
        // SAFETY: live context returns an internal NUL-terminated error string.
        let message = unsafe { CStr::from_ptr((self.error)(self.context)) }.to_string_lossy();
        Err(err(format!(
            "{}: {message} (status={status})",
            self.kind.name()
        )))
    }
    fn alloc(self: &Rc<Self>, len: usize) -> Result<Buffer> {
        require(
            len > 0 && len <= i32::MAX as usize && len.checked_mul(4).is_some(),
            "device allocation overflow/empty",
        )?;
        let mut pointer = ptr::null_mut();
        // SAFETY: out pointer is writable, size was checked, context is live.
        self.check(unsafe { (self.alloc)(self.context, &mut pointer, len) })?;
        require(!pointer.is_null(), "null device allocation")?;
        Ok(Buffer {
            api: self.clone(),
            pointer,
            len,
        })
    }
    /// Device buffer over the host allocation itself (no copy), if the bridge
    /// supports it. The returned buffer must not outlive `host`.
    fn wrap_bf16(self: &Rc<Self>, host: &AlignedBuf) -> Result<Option<Buffer>> {
        let Some(wrap) = self.wrap else {
            return Ok(None);
        };
        let mut pointer = ptr::null_mut();
        // SAFETY: page-aligned allocation of whole pages, alive while the
        // caller keeps `host`; the device only reads it.
        self.check(unsafe {
            wrap(
                self.context,
                &mut pointer,
                host.as_ptr().cast_mut().cast(),
                host.allocation_bytes(),
            )
        })?;
        require(!pointer.is_null(), "null wrapped buffer")?;
        Ok(Some(Buffer {
            api: self.clone(),
            pointer,
            // Packed words, as in upload_bf16_bits.
            len: host.len().div_ceil(2),
        }))
    }
    fn upload(self: &Rc<Self>, values: &[f32]) -> Result<Buffer> {
        finite(values)?;
        let b = self.alloc(values.len())?;
        // SAFETY: allocation exactly matches slice; bridge synchronizes H2D before returning.
        self.check(unsafe {
            (self.write)(self.context, b.pointer, values.as_ptr(), values.len())
        })?;
        Ok(b)
    }
    /// BF16 matrix storage: two bf16 per 32-bit word, little-endian, so the
    /// native side reads it as a plain uint16 array. Values were checked exact.
    fn upload_bf16(self: &Rc<Self>, values: &[f32]) -> Result<Buffer> {
        finite(values)?;
        let bits: Vec<u16> = values.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
        self.upload_bf16_bits(&bits)
    }
    /// BF16 bit patterns, two per 32-bit word (element 2i in the low half).
    fn upload_bf16_bits(self: &Rc<Self>, bits: &[u16]) -> Result<Buffer> {
        let words: Vec<u32> = bits
            .chunks(2)
            .map(|p| p[0] as u32 | (p.get(1).map_or(0, |&v| v as u32) << 16))
            .collect();
        let b = self.alloc(words.len())?;
        // SAFETY: raw 32-bit words copied bit-for-bit (no float arithmetic);
        // allocation matches; the bridge synchronizes before returning.
        self.check(unsafe {
            (self.write)(self.context, b.pointer, words.as_ptr().cast(), words.len())
        })?;
        Ok(b)
    }
    fn run(&self, op: Op, count: usize, p: &[*mut f32], n: &[usize], f: &[f32]) -> Result<()> {
        let mut a = Params {
            p: [ptr::null_mut(); 8],
            n: [0; 8],
            f: [0.; 4],
        };
        a.p[..p.len()].copy_from_slice(p);
        for (v, &value) in a.n.iter_mut().zip(n) {
            *v = int(value)?;
        }
        a.f[..f.len()].copy_from_slice(f);
        // SAFETY: private callers supply validated shapes and live distinct buffers;
        // all commands on this context are ordered on the same native stream.
        self.check(unsafe { (self.run)(self.context, op as i32, int(count)?, &a) })
    }
    fn runs(&self, op: Op) -> bool {
        op as i32 <= self.max_op
    }
    fn sync(&self) -> Result<()> {
        // SAFETY: context/stream live.
        self.check(unsafe { (self.sync)(self.context) })
    }
}
fn int(n: usize) -> Result<i32> {
    i32::try_from(n).map_err(err)
}
struct Buffer {
    api: Rc<Api>,
    pointer: *mut f32,
    len: usize,
}
impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: sole allocation owner, Api remains alive through this call.
        unsafe {
            (self.api.free)(self.api.context, self.pointer);
        }
    }
}
impl Buffer {
    /// Interior pointer for a token row of a chunk-sized buffer.
    fn at(&self, offset: usize) -> Result<*mut f32> {
        require(offset < self.len, "device buffer offset")?;
        Ok(self.pointer.wrapping_add(offset))
    }
    #[cfg(test)]
    fn read(&self) -> Result<Vec<f32>> {
        self.read_prefix(self.len)
    }
    /// Host -> device for the first `values.len()` floats (edit overlays only).
    fn write_prefix(&self, values: &[f32]) -> Result<()> {
        finite(values)?;
        require(
            !values.is_empty() && values.len() <= self.len,
            "device write bounds",
        )?;
        // SAFETY: bounded by the allocation; the bridge orders it after queued kernels.
        self.api.check(unsafe {
            (self.api.write)(
                self.api.context,
                self.pointer,
                values.as_ptr(),
                values.len(),
            )
        })
    }
    fn read_prefix(&self, n: usize) -> Result<Vec<f32>> {
        require(n > 0 && n <= self.len, "device read bounds")?;
        let mut host = vec![0.; n];
        // SAFETY: read bounded by allocation and host capacity; bridge synchronizes.
        self.api.check(unsafe {
            (self.api.read)(self.api.context, host.as_mut_ptr(), self.pointer, n)
        })?;
        finite(&host)?;
        Ok(host)
    }
}
struct DeviceMatrix {
    buffer: Buffer,
    rows: usize,
    cols: usize,
    bf16: bool,
    /// Host memory the buffer reads in place (shared weights), kept alive
    /// until after the buffer is released. Declared after `buffer`, so it is
    /// dropped after it.
    shared: Option<Arc<AlignedBuf>>,
}
impl Drop for DeviceMatrix {
    fn drop(&mut self) {
        if self.shared.is_some() {
            // Queued kernels may still read the host memory: wait for them
            // before the memory can be freed. Errors are already reported by
            // the operation that failed.
            // SAFETY: live context.
            let _ = unsafe { (self.buffer.api.sync)(self.buffer.api.context) };
        }
    }
}
impl DeviceMatrix {
    fn upload(api: &Rc<Api>, w: &Matrix, format: WeightFormat) -> Result<Self> {
        int(w.rows())?;
        int(w.cols())?;
        // Host BF16 storage is uploaded bit-for-bit; an f32 matrix is checked
        // (and packed) only when every value is exactly representable.
        let mut shared = None;
        let (buffer, bf16) = match (format, w.bf16_shared()) {
            (WeightFormat::Bf16Exact, Some(host)) => match api.wrap_bf16(&host)? {
                // Unified memory: the device reads the host BF16 in place.
                Some(buffer) => {
                    shared = Some(host);
                    (buffer, true)
                }
                None => (api.upload_bf16_bits(&host)?, true),
            },
            (WeightFormat::Bf16Exact, None) if w.values().iter().all(|&v| bf16_exact(v)) => {
                (api.upload_bf16(&w.values())?, true)
            }
            _ => (api.upload(&w.values())?, false),
        };
        Ok(Self {
            buffer,
            rows: w.rows(),
            cols: w.cols(),
            bf16,
            shared,
        })
    }
    fn bytes(&self) -> usize {
        self.rows * self.cols * if self.bf16 { 2 } else { 4 }
    }
    /// Y[t,rows] = X[t,cols] . W^T for the first `tokens` rows of chunk buffers.
    /// One token uses GEMV; more use GEMM, reading each weight once per chunk.
    fn project(&self, x: &Buffer, out: &Buffer, tokens: usize) -> Result<()> {
        require(
            tokens > 0 && x.len >= tokens * self.cols && out.len >= tokens * self.rows,
            "device GEMV shape",
        )?;
        let a = &self.buffer.api;
        let (rows, cols) = (int(self.rows)?, int(self.cols)?);
        // SAFETY: private engine keeps matrices/x/out on one context; y is distinct.
        // A BF16 matrix's buffer holds rows*cols uint16 (validated at upload).
        a.check(unsafe {
            if tokens == 1 {
                let gemv = if self.bf16 { a.gemv_bf16 } else { a.gemv };
                gemv(
                    a.context,
                    out.pointer,
                    self.buffer.pointer,
                    x.pointer,
                    rows,
                    cols,
                )
            } else {
                let gemm = if self.bf16 { a.gemm_bf16 } else { a.gemm };
                gemm(
                    a.context,
                    out.pointer,
                    self.buffer.pointer,
                    x.pointer,
                    rows,
                    cols,
                    int(tokens)?,
                )
            }
        })
    }
}
/// Projections of the same input `x` (q/k/v, DeltaNet qkv/z/a/b, FFN
/// gate/up): one bridge call when the bridge groups BF16 matrices, otherwise
/// one `project` each. Each output is the same either way.
fn project_all(x: &Buffer, items: &[(&DeviceMatrix, &Buffer)], tokens: usize) -> Result<()> {
    let Some(&(first, _)) = items.first() else {
        return Ok(());
    };
    let a = &first.buffer.api;
    let grouped = a
        .gemm_bf16_group
        .filter(|_| items.len() > 1 && items.iter().all(|(m, _)| m.bf16 && m.cols == first.cols));
    let Some(group) = grouped else {
        return items
            .iter()
            .try_for_each(|(m, out)| m.project(x, out, tokens));
    };
    let mut entries = Vec::with_capacity(items.len());
    for (m, out) in items {
        require(
            tokens > 0 && x.len >= tokens * m.cols && out.len >= tokens * m.rows,
            "device GEMV shape",
        )?;
        entries.push(Projection {
            y: out.pointer,
            a: m.buffer.pointer,
            rows: int(m.rows)?,
        });
    }
    // SAFETY: as in `project`; the entries live until the call returns and
    // every output is a distinct allocation on this context.
    a.check(unsafe {
        group(
            a.context,
            int(entries.len())?,
            entries.as_ptr(),
            x.pointer,
            int(first.cols)?,
            int(tokens)?,
        )
    })
}
struct DeviceBlock {
    input: Buffer,
    post: Buffer,
    gate: DeviceMatrix,
    up: DeviceMatrix,
    down: DeviceMatrix,
    mixer: DeviceMixer,
    eps: f32,
}
enum DeviceMixer {
    Attention {
        c: AttentionConfig,
        q: DeviceMatrix,
        k: DeviceMatrix,
        v: DeviceMatrix,
        o: DeviceMatrix,
        qn: Buffer,
        kn: Buffer,
        /// [q_norm | k_norm] in one buffer for op AttentionQueryKv.
        qkn: Buffer,
    },
    Delta {
        c: DeltaConfig,
        qkv: DeviceMatrix,
        z: DeviceMatrix,
        a: DeviceMatrix,
        b: DeviceMatrix,
        conv: Buffer,
        dt: Buffer,
        alog: Buffer,
        /// [dt | A_log] in one buffer for op DeltaRuleGated (8 pointer slots).
        gates: Buffer,
        norm: Buffer,
        out: DeviceMatrix,
    },
}
/// Per-run state/scratch budget (weights excluded). About 170k tokens of
/// Qwen3.5-0.8B KV; keeps a typo in --capacity from exhausting device memory.
pub const MAX_RUN_BYTES: usize = 4 << 30;
/// Prefill positions without a host readback between explicit queue flushes.
/// Bounds the native command backlog and surfaces GPU faults promptly.
const PREFILL_FLUSH_INTERVAL: usize = 32;
/// Default prefill chunk: tokens whose projections share one weight read.
/// Scratch activations are allocated for this many rows (~120 KiB/token for
/// Qwen3.5-0.8B). 1 reproduces token-by-token prefill.
pub const PREFILL_CHUNK: usize = 128;
/// Most rows a verify pass returns logits for (1 accepted + up to 7 drafts):
/// the logits buffer holds this many vocabulary rows (~7.6 MiB).
pub const VERIFY_ROWS: usize = 8;
/// Immutable device weights. Run state cannot be shared accidentally across runs.
pub struct DeviceModel {
    api: Rc<Api>,
    config: TextConfig,
    embedding: DeviceMatrix,
    norm: Buffer,
    blocks: Vec<DeviceBlock>,
    attention_f16: bool,
    matmul_relaxed: bool,
    kernel_level: KernelLevel,
}
/// Metal kernel levels (bridge option "kernel_level"; spec/metal.md).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelLevel {
    /// Kernels that mirror the CPU code (one thread per output or head).
    Reference = 1,
    /// SIMD-group and 8x8 simdgroup-matrix kernels (any Apple GPU).
    Simdgroup = 2,
    /// Metal 4 tensor ops where available (the default).
    Tensor = 3,
}
impl KernelLevel {
    pub fn name(self) -> &'static str {
        match self {
            Self::Reference => "reference",
            Self::Simdgroup => "simdgroup",
            Self::Tensor => "tensor",
        }
    }
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "reference" => Ok(Self::Reference),
            "simdgroup" => Ok(Self::Simdgroup),
            "tensor" => Ok(Self::Tensor),
            _ => Err(err(
                "--metal-kernels must be reference, simdgroup or tensor",
            )),
        }
    }
}
impl DeviceModel {
    pub fn kind(&self) -> Kind {
        self.api.kind
    }
    /// ROCm: HIP runtime/driver API versions and GPU architecture (not the
    /// Windows display driver package). Metal: GPU name and memory limits.
    pub fn device_info(&self) -> String {
        // SAFETY: native context owns a NUL-terminated info string until drop.
        unsafe { CStr::from_ptr((self.api.info)(self.api.context)) }
            .to_string_lossy()
            .into_owned()
    }
    /// Load only the user-specified trusted local bridge (dlopen executes code).
    /// Matrices go to BF16 wherever that is lossless (the released checkpoint is BF16).
    pub fn load(model: &Model, library: &Path, kind: Kind) -> Result<Self> {
        Self::load_with(model, library, kind, WeightFormat::Bf16Exact)
    }
    pub fn load_with(
        model: &Model,
        library: &Path,
        kind: Kind,
        format: WeightFormat,
    ) -> Result<Self> {
        require(kind != Kind::CpuTest, "CPU test bridge is not a device")?;
        Self::upload(model, Api::open(library, kind)?, format)
    }
    /// (BF16 matrices, all matrices, device matrix bytes). Vectors are FP32 and excluded.
    pub fn weight_summary(&self) -> (usize, usize, usize) {
        let mut all = vec![&self.embedding];
        for b in &self.blocks {
            all.extend([&b.gate, &b.up, &b.down]);
            match &b.mixer {
                DeviceMixer::Attention { q, k, v, o, .. } => all.extend([q, k, v, o]),
                DeviceMixer::Delta {
                    qkv, z, a, b, out, ..
                } => all.extend([qkv, z, a, b, out]),
            }
        }
        (
            all.iter().filter(|m| m.bf16).count(),
            all.len(),
            all.iter().map(|m| m.bytes()).sum(),
        )
    }
    /// "bf16", "f32" or "mixed" (some matrix was not exactly representable).
    pub fn weight_format(&self) -> &'static str {
        match self.weight_summary() {
            (0, _, _) => "f32",
            (b, n, _) if b == n => "bf16",
            _ => "mixed",
        }
    }
    fn upload(model: &Model, api: Rc<Api>, format: WeightFormat) -> Result<Self> {
        model.config.validate()?;
        let matrix = |w: &Matrix| DeviceMatrix::upload(&api, w, format);
        let vec = |w: &[f32]| api.upload(w);
        let embedding = matrix(&model.embedding)?;
        let norm = vec(&model.norm)?;
        let mut blocks = Vec::new();
        for b in &model.blocks {
            let mixer = match &b.mixer {
                MixerWeights::Attention(w) => DeviceMixer::Attention {
                    c: w.config.clone(),
                    q: matrix(&w.q)?,
                    k: matrix(&w.k)?,
                    v: matrix(&w.v)?,
                    o: matrix(&w.o)?,
                    qn: vec(&w.q_norm)?,
                    kn: vec(&w.k_norm)?,
                    qkn: vec(&[w.q_norm.as_slice(), w.k_norm.as_slice()].concat())?,
                },
                MixerWeights::Delta(w) => DeviceMixer::Delta {
                    c: w.config.clone(),
                    qkv: matrix(&w.qkv)?,
                    z: matrix(&w.z)?,
                    a: matrix(&w.a)?,
                    b: matrix(&w.b)?,
                    conv: vec(&w.conv.values())?,
                    dt: vec(&w.dt)?,
                    alog: vec(&w.a_log)?,
                    gates: vec(&[w.dt.as_slice(), w.a_log.as_slice()].concat())?,
                    norm: vec(&w.norm)?,
                    out: matrix(&w.out)?,
                },
            };
            let [gate, up, down] = b.ffn.matrices();
            blocks.push(DeviceBlock {
                input: vec(&b.input_norm)?,
                post: vec(&b.post_norm)?,
                gate: matrix(gate)?,
                up: matrix(up)?,
                down: matrix(down)?,
                mixer,
                eps: b.eps,
            });
        }
        api.sync()?;
        Ok(Self {
            api,
            config: model.config.clone(),
            embedding,
            norm,
            blocks,
            attention_f16: false,
            matmul_relaxed: false,
            kernel_level: KernelLevel::Tensor,
        })
    }
    /// Opt-in 16-bit prefill attention (ADR 0014): Q, K, V and the softmax
    /// probabilities are rounded to FP16 inside the matrix instructions of a
    /// chunked prefill; decode, the KV cache and all other ops stay FP32.
    /// This changes results, so it is never the default. A bridge that
    /// cannot do it (Metal, the CPU test bridge, older builds) is an error.
    pub fn set_attention_f16(&mut self, on: bool) -> Result<()> {
        self.set_option(c"attention_f16", on)?;
        self.attention_f16 = on;
        Ok(())
    }
    /// Opt-in relaxed-precision prefill GEMMs (ADR 0016, Metal 4 tensor ops):
    /// the GPU matrix hardware may trade accuracy for speed in the projections
    /// of a chunked prefill (>= 8 tokens, > 16 rows). Decode, the KV cache,
    /// attention and every other op are unaffected. Changes results, so never
    /// the default; unavailable tensor ops or a bridge without the option are
    /// an error.
    pub fn set_matmul_relaxed(&mut self, on: bool) -> Result<()> {
        self.metal_only("--matmul-relaxed")?;
        self.set_option(c"matmul_relaxed", on)?;
        self.matmul_relaxed = on;
        Ok(())
    }
    /// Metal: highest kernel level the bridge may use (spec/metal.md, "Kernel
    /// levels"): 1 reference kernels that mirror the CPU code, 2 SIMD-group
    /// kernels, 3 Metal 4 tensor ops (default; each op falls back to the
    /// highest level it has). All levels compute the same values up to
    /// summation order; lower levels are slower and easier to read.
    pub fn set_kernel_level(&mut self, level: KernelLevel) -> Result<()> {
        self.metal_only("--metal-kernels")?;
        self.set_option_value(c"kernel_level", level as i32)?;
        self.kernel_level = level;
        Ok(())
    }
    pub fn kernel_level(&self) -> KernelLevel {
        self.kernel_level
    }
    /// Kernel level for reports: Metal only (other bridges have no levels).
    pub fn kernel_level_name(&self) -> Option<&'static str> {
        (self.api.kind == Kind::Metal).then(|| self.kernel_level.name())
    }
    /// Metal-only options fail with the option's name on other bridges,
    /// instead of the HIP bridge's generic "unknown bridge option".
    fn metal_only(&self, option: &str) -> Result<()> {
        if self.api.kind == Kind::Metal {
            return Ok(());
        }
        Err(err(format!(
            "{option} is Metal only (spec/metal.md); not available on {}",
            self.api.kind.name()
        )))
    }
    /// "f32" or "relaxed_prefill" (see set_matmul_relaxed).
    pub fn matmul_precision(&self) -> &'static str {
        if self.matmul_relaxed {
            "relaxed_prefill"
        } else {
            "f32"
        }
    }
    fn set_option(&self, name: &std::ffi::CStr, on: bool) -> Result<()> {
        self.set_option_value(name, on as i32)
    }
    fn set_option_value(&self, name: &std::ffi::CStr, value: i32) -> Result<()> {
        let api = &self.api;
        let set = api.set_option.ok_or_else(|| {
            err(format!(
                "{} bridge has no engine options ({} unavailable)",
                api.kind.name(),
                name.to_string_lossy()
            ))
        })?;
        // SAFETY: valid context and NUL-terminated name; the bridge copies nothing.
        api.check(unsafe { set(api.context, name.as_ptr(), value) })
    }
    /// "f32" or "f16_prefill" (see set_attention_f16).
    pub fn attention_precision(&self) -> &'static str {
        if self.attention_f16 {
            "f16_prefill"
        } else {
            "f32"
        }
    }
    /// Device bytes of one run's state and scratch for `capacity` tokens with
    /// the default prefill chunk. Attention KV/score rows grow with capacity;
    /// activations grow with the chunk; DeltaNet state has a fixed size.
    pub fn run_bytes(&self, capacity: usize) -> usize {
        self.run_bytes_chunked(capacity, capacity.min(PREFILL_CHUNK))
    }
    /// Bytes of one copy of every DeltaNet layer's state (conv history + recurrent).
    fn delta_state_bytes(&self) -> usize {
        self.blocks
            .iter()
            .map(|b| match &b.mixer {
                DeviceMixer::Delta { c, qkv, .. } => {
                    4 * ((qkv.rows * (c.kernel - 1)).max(1)
                        + c.value_heads * c.key_dim * c.value_dim)
                }
                _ => 0,
            })
            .sum()
    }
    pub fn run_bytes_chunked(&self, capacity: usize, chunk: usize) -> usize {
        let mut floats = 0usize;
        for b in &self.blocks {
            floats = floats.saturating_add(match &b.mixer {
                DeviceMixer::Attention { c, .. } => {
                    let q = c.heads * c.head_dim;
                    let kv = c.kv_heads * c.head_dim;
                    kv_rows(capacity)
                        .saturating_mul(2 * kv)
                        .saturating_add(score_row(capacity, c.head_dim).saturating_mul(c.heads))
                        .saturating_add(chunk.saturating_mul(q * 4 + kv * 2))
                        .saturating_add(chunk.next_multiple_of(8).saturating_mul(q))
                }
                DeviceMixer::Delta { c, qkv, .. } => {
                    let q = c.value_heads * c.key_dim;
                    let v = c.value_heads * c.value_dim;
                    (qkv.rows * (c.kernel - 1).max(1) + c.value_heads * c.key_dim * c.value_dim)
                        .saturating_add(
                            chunk.saturating_mul(qkv.rows * 2 + q * 2 + c.value_heads * 4 + v * 4),
                        )
                }
            });
        }
        let h = self.config.hidden_size;
        let i = self.config.intermediate_size;
        floats
            .saturating_add(chunk.saturating_mul(h * 5 + i * 3))
            .saturating_add(
                self.config
                    .vocab_size
                    .saturating_mul(VERIFY_ROWS.min(chunk)),
            )
            .saturating_mul(4)
    }
    /// KV reserves `capacity` tokens per run, explicitly chosen by the caller.
    /// The limit is the model context and a fixed per-run memory budget, not
    /// a short-context constant: KV costs ~24 KiB/token for Qwen3.5-0.8B.
    pub fn new_run(&self, capacity: usize) -> Result<DeviceRun<'_>> {
        self.new_run_chunked(capacity, PREFILL_CHUNK)
    }
    /// `chunk` (clamped to capacity) bounds tokens per prefill forward pass.
    pub fn new_run_chunked(&self, capacity: usize, chunk: usize) -> Result<DeviceRun<'_>> {
        self.new_run_with(capacity, chunk, 0)
    }
    /// A run whose verify passes can be rolled back (`verify_rollback` /
    /// `accept`, ADR 0026): room for `snapshots` per-token copies of every
    /// DeltaNet state (~19 MiB each for Qwen3.5-0.8B). Needs `qr_snapshots`.
    pub fn new_run_rollback(&self, capacity: usize, snapshots: usize) -> Result<DeviceRun<'_>> {
        require(
            self.api.snapshots && (1..VERIFY_ROWS).contains(&snapshots),
            "bridge has no state snapshots (qr_snapshots), or snapshots not 1..=7",
        )?;
        self.new_run_with(capacity, PREFILL_CHUNK, snapshots)
    }
    fn new_run_with(
        &self,
        capacity: usize,
        chunk: usize,
        snapshots: usize,
    ) -> Result<DeviceRun<'_>> {
        require(
            capacity > 0 && capacity <= self.config.max_position_embeddings,
            "device capacity must be 1..=model context limit",
        )?;
        require(chunk > 0, "device prefill chunk must be positive")?;
        let chunk = chunk.min(capacity);
        let bytes = self
            .run_bytes_chunked(capacity, chunk)
            .saturating_add(snapshots.saturating_mul(self.delta_state_bytes()));
        require(
            bytes <= MAX_RUN_BYTES,
            &format!(
                "device run for capacity {capacity} needs {bytes} bytes > budget {MAX_RUN_BYTES}"
            ),
        )?;
        let alloc = |n| self.api.alloc(n);
        let mut mixers = Vec::new();
        for b in &self.blocks {
            mixers.push(match &b.mixer {
                DeviceMixer::Attention { c, .. } => {
                    let q = c.heads * c.head_dim;
                    let kv = c.kv_heads * c.head_dim;
                    MixerRun::Attention {
                        mixed: alloc(chunk * q * 2)?,
                        key: alloc(chunk * kv)?,
                        value: alloc(chunk * kv)?,
                        // Padded to 8 rows: Metal attention_mma loads 8-row Q tiles.
                        query: alloc(chunk.next_multiple_of(8) * q)?,
                        gate: alloc(chunk * q)?,
                        context: alloc(chunk * q)?,
                        keys: alloc(kv_rows(capacity) * kv)?,
                        values: alloc(kv_rows(capacity) * kv)?,
                        scores: alloc(score_row(capacity, c.head_dim) * c.heads)?,
                    }
                }
                DeviceMixer::Delta { c, qkv, .. } => {
                    let q = c.value_heads * c.key_dim;
                    let v = c.value_heads * c.value_dim;
                    MixerRun::Delta {
                        projected: alloc(chunk * qkv.rows)?,
                        convolved: alloc(chunk * qkv.rows)?,
                        // Live state, then `snapshots` copies (ADR 0026).
                        conv_state: alloc((qkv.rows * (c.kernel - 1)).max(1) * (1 + snapshots))?,
                        recurrent: alloc(
                            c.value_heads * c.key_dim * c.value_dim * (1 + snapshots),
                        )?,
                        query: alloc(chunk * q)?,
                        key: alloc(chunk * q)?,
                        a: alloc(chunk * c.value_heads)?,
                        b: alloc(chunk * c.value_heads)?,
                        z: alloc(chunk * v)?,
                        decay: alloc(chunk * c.value_heads)?,
                        beta: alloc(chunk * c.value_heads)?,
                        core: alloc(chunk * v)?,
                        normalized: alloc(chunk * v)?,
                        gated: alloc(chunk * v)?,
                    }
                }
            });
        }
        let h = self.config.hidden_size;
        let i = self.config.intermediate_size;
        Ok(DeviceRun {
            model: self,
            edit: None,
            capacity,
            chunk,
            snapshots,
            snapshot_rows: 0,
            pending: None,
            position: 0,
            failed: false,
            mixers,
            x: alloc(chunk * h)?,
            norm: alloc(chunk * h)?,
            mixed: alloc(chunk * h)?,
            residual: alloc(chunk * h)?,
            gate: alloc(chunk * i)?,
            up: alloc(chunk * i)?,
            product: alloc(chunk * i)?,
            ffn: alloc(chunk * h)?,
            logits: alloc(self.config.vocab_size * VERIFY_ROWS.min(chunk))?,
        })
    }
}
enum MixerRun {
    Attention {
        mixed: Buffer,
        key: Buffer,
        value: Buffer,
        query: Buffer,
        gate: Buffer,
        context: Buffer,
        keys: Buffer,
        values: Buffer,
        scores: Buffer,
    },
    Delta {
        projected: Buffer,
        convolved: Buffer,
        conv_state: Buffer,
        recurrent: Buffer,
        query: Buffer,
        key: Buffer,
        a: Buffer,
        b: Buffer,
        z: Buffer,
        decay: Buffer,
        beta: Buffer,
        core: Buffer,
        normalized: Buffer,
        gated: Buffer,
    },
}
/// A validated edit manifest plus host FP32 copies of the down matrices of
/// the edited layers only. The overlay itself is `EditManifest::apply`, the
/// same function the CPU engine calls; device weights are never modified.
struct DeviceEdit {
    manifest: EditManifest,
    down: Vec<Option<Matrix>>,
}
/// Mutable state + temporary activations; never changes immutable uploaded weights.
pub struct DeviceRun<'a> {
    model: &'a DeviceModel,
    edit: Option<DeviceEdit>,
    capacity: usize,
    /// Rows of every activation buffer = max tokens per forward pass.
    chunk: usize,
    /// Per-token DeltaNet state copies allocated (0 = no rollback, ADR 0026).
    snapshots: usize,
    /// Snapshots the next forward stores (set by `verify_rollback` only).
    snapshot_rows: usize,
    /// (start position, rows) of a verify pass awaiting `accept`.
    pending: Option<(usize, usize)>,
    position: usize,
    failed: bool,
    mixers: Vec<MixerRun>,
    x: Buffer,
    norm: Buffer,
    mixed: Buffer,
    residual: Buffer,
    gate: Buffer,
    up: Buffer,
    product: Buffer,
    ffn: Buffer,
    logits: Buffer,
}
#[derive(Debug)]
pub struct DeviceStep {
    pub operations: Vec<(String, Vec<f32>)>,
    pub layers: Vec<Vec<f32>>,
    pub normalized: Vec<f32>,
    pub logits: Vec<f32>,
}
impl DeviceRun<'_> {
    pub fn position(&self) -> usize {
        self.position
    }
    /// Rebuild state/scratch, retaining the device weights. Allocation failure
    /// leaves the existing (possibly poisoned) run intact rather than half reset.
    pub fn reset(&mut self) -> Result<()> {
        let mut fresh = self
            .model
            .new_run_with(self.capacity, self.chunk, self.snapshots)?;
        fresh.edit = self.edit.take();
        *self = fresh;
        Ok(())
    }
    /// Apply `edit` to every following forward pass (None removes it). Same
    /// validation as the CPU engine; `model` must be the one this device was
    /// loaded from (its revision and down matrices are used).
    pub fn set_edit(&mut self, model: &Model, edit: Option<&EditManifest>) -> Result<()> {
        let Some(manifest) = edit else {
            self.edit = None;
            return Ok(());
        };
        let c = model.config();
        require(
            model.blocks().len() == self.model.blocks.len()
                && c.hidden_size == self.model.config.hidden_size
                && c.intermediate_size == self.model.config.intermediate_size,
            "edit model does not match the device model",
        )?;
        manifest.validate(
            model.revision(),
            model.blocks().len(),
            c.hidden_size,
            c.intermediate_size,
        )?;
        let down = model
            .blocks()
            .iter()
            .enumerate()
            .map(|(l, b)| {
                manifest
                    .edits
                    .iter()
                    .any(|e| e.layer == l)
                    .then(|| b.ffn.matrices()[2].clone())
            })
            .collect();
        self.edit = Some(DeviceEdit {
            manifest: manifest.clone(),
            down,
        });
        Ok(())
    }
    pub fn step(&mut self, id: u32, capture: bool) -> Result<DeviceStep> {
        require(!self.failed, "device run failed; reset required")?;
        require(
            (id as usize) < self.model.config.vocab_size && self.position < self.capacity,
            "device token/capacity out of bounds",
        )?;
        let result = self.forward(&[id], capture, 1);
        if result.is_err() {
            self.failed = true;
        }
        result
    }
    /// Empty and split prefill retain the same absolute position/state.
    /// Only the LAST position computes vocabulary logits: earlier positions
    /// skip the tied-embedding projection (the largest single matrix) and the
    /// full-logit readback, since their predictions are discarded anyway.
    pub fn prefill(&mut self, ids: &[u32]) -> Result<Option<Vec<f32>>> {
        require(
            !self.failed
                && ids.len() <= self.capacity - self.position
                && ids
                    .iter()
                    .all(|&v| (v as usize) < self.model.config.vocab_size),
            "device prefill token/capacity/failed",
        )?;
        let mut last = None;
        let mut unsynced = 0;
        let chunks = ids.len().div_ceil(self.chunk);
        for (i, part) in ids.chunks(self.chunk).enumerate() {
            let final_chunk = i + 1 == chunks;
            unsynced += part.len();
            let result = self
                .forward(part, false, usize::from(final_chunk))
                .and_then(|step| {
                    // Without a readback nothing waits for the queue; flush periodically.
                    if !final_chunk && unsynced >= PREFILL_FLUSH_INTERVAL {
                        unsynced = 0;
                        self.model.api.sync()?;
                    }
                    Ok(step)
                });
            match result {
                Ok(step) => last = final_chunk.then_some(step.logits),
                Err(e) => {
                    self.failed = true;
                    return Err(e);
                }
            }
        }
        Ok(last)
    }
    /// Verify pass of speculative decoding (ADR 0023): one forward over
    /// `ids` (at most `VERIFY_ROWS`) with vocabulary logits for every row,
    /// so each drafted token can be checked. Weights are read once for all
    /// rows, as in a prefill chunk.
    pub fn verify(&mut self, ids: &[u32]) -> Result<Vec<Vec<f32>>> {
        require(
            !self.failed
                && !ids.is_empty()
                && ids.len() <= VERIFY_ROWS.min(self.chunk)
                && ids.len() <= self.capacity - self.position
                && ids
                    .iter()
                    .all(|&v| (v as usize) < self.model.config.vocab_size),
            "device verify rows/capacity/failed",
        )?;
        let result = self.forward(ids, false, ids.len());
        if result.is_err() {
            self.failed = true;
        }
        let logits = result?.logits;
        let v = self.model.config.vocab_size;
        Ok(logits.chunks_exact(v).map(<[f32]>::to_vec).collect())
    }
    /// `verify` whose rows can be taken back (ADR 0026): the DeltaNet
    /// layers also store their state after each row but the last, so
    /// `accept(rows)` can keep any prefix. Needs a `new_run_rollback` run
    /// with at least `ids.len() - 1` snapshots. Attention KV needs no copy:
    /// rows past the kept prefix are simply overwritten later.
    pub fn verify_rollback(&mut self, ids: &[u32]) -> Result<Vec<Vec<f32>>> {
        require(
            !ids.is_empty() && ids.len() <= self.snapshots + 1,
            "verify_rollback: more rows than snapshots + 1",
        )?;
        let start = self.position;
        self.snapshot_rows = ids.len() - 1;
        let out = self.verify(ids);
        self.snapshot_rows = 0;
        let out = out?;
        self.pending = Some((start, ids.len()));
        Ok(out)
    }
    /// Keep the first `rows` (1..=t) tokens of the last `verify_rollback`
    /// pass: the state becomes exactly that after those tokens. A device
    /// copy of each DeltaNet layer's snapshot `rows - 1` (~19 MiB in all).
    pub fn accept(&mut self, rows: usize) -> Result<()> {
        let (start, t) = self
            .pending
            .take()
            .ok_or_else(|| err("accept needs a verify_rollback pass"))?;
        require((1..=t).contains(&rows), "accept rows out of range")?;
        if rows == t {
            return Ok(());
        }
        let api = &self.model.api;
        for (b, s) in self.model.blocks.iter().zip(&self.mixers) {
            if let (
                DeviceMixer::Delta { c, qkv, .. },
                MixerRun::Delta {
                    conv_state,
                    recurrent,
                    ..
                },
            ) = (&b.mixer, s)
            {
                let state = c.value_heads * c.key_dim * c.value_dim;
                let conv = (qkv.rows * (c.kernel - 1)).max(1);
                let conv_part = (c.kernel > 1).then_some((conv_state, conv));
                for (buffer, n) in std::iter::once((recurrent, state)).chain(conv_part) {
                    let src = buffer.at(rows * n)?;
                    // SAFETY: snapshot `rows - 1` and the live state are disjoint
                    // ranges of the same allocation (sized 1 + snapshots copies).
                    api.check(unsafe { (api.copy)(api.context, buffer.pointer, src, n) })?;
                }
            }
        }
        self.position = start + rows;
        Ok(())
    }
    pub fn chunk(&self) -> usize {
        self.chunk
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    /// One forward pass over `ids.len() <= chunk` consecutive positions.
    /// Every op is dispatched once for all rows (projections as GEMM: each
    /// weight read once per chunk), including the causal attention read
    /// (row r sees position + r + 1 keys; ADR 0007). Time-recurrent ops (conv history, the
    /// DeltaNet rule) loop the tokens in order inside each native thread, so
    /// state evolves exactly as in token-by-token decode.
    /// `capture` (validation) is only available for a single token.
    /// `logit_rows`: 0 = none (prefill chunk), 1 = last row, `t` = every row (verify).
    fn forward(&mut self, ids: &[u32], capture: bool, logit_rows: usize) -> Result<DeviceStep> {
        self.pending = None; // moving on invalidates an unaccepted verify pass
        let t = ids.len();
        require(
            t > 0
                && t <= self.chunk
                && (!capture || t == 1)
                && (logit_rows <= 1 || logit_rows == t),
            "device forward chunk/capture",
        )?;
        let model = self.model;
        let api = &model.api;
        let h = model.config.hidden_size;
        for (r, &id) in ids.iter().enumerate() {
            let x = self.x.at(r * h)?;
            if model.embedding.bf16 {
                // Widen one BF16 row; the native side indexes [id*h + t] in uint16.
                api.run(
                    Op::EmbedBf16,
                    h,
                    &[x, model.embedding.buffer.pointer],
                    &[id as usize, h],
                    &[],
                )?;
            } else {
                // SAFETY: ID checked before enqueue; embedding row is in the allocation.
                let row = model.embedding.buffer.pointer.wrapping_add(id as usize * h);
                api.check(unsafe { (api.copy)(api.context, x, row, h) })?;
            }
        }
        let mut layers = Vec::new();
        let mut operations = Vec::new();
        // Captures exist only for t == 1, i.e. row 0 of each buffer.
        let mut record = |layer: usize, name: &str, buffer: &Buffer, n: usize| -> Result<()> {
            if capture {
                operations.push((format!("layer.{layer}.{name}"), buffer.read_prefix(n)?));
            }
            Ok(())
        };
        let norm = |x: *mut f32, w: &Buffer, out: *mut f32, d, groups, eps, offset| {
            api.run(
                Op::RmsNorm,
                groups,
                &[out, x, w.pointer],
                &[d],
                &[eps, offset],
            )
        };
        let i_size = model.config.intermediate_size;
        // Add followed by RmsNorm of the sum runs as one op when the bridge has it.
        let fused_add_norm = api.runs(Op::AddRmsNorm);
        let add_norm = |sum: *mut f32, u: *mut f32, v: *mut f32, w: &Buffer, out: *mut f32, eps| {
            if fused_add_norm {
                api.run(
                    Op::AddRmsNorm,
                    t,
                    &[sum, u, v, out, w.pointer],
                    &[h],
                    &[eps, 1.],
                )
            } else {
                api.run(Op::Add, h * t, &[sum, u, v], &[], &[])?;
                norm(sum, w, out, h, t, eps, 1.)
            }
        };
        // Decode: a GEMV whose output feeds Add + RmsNorm runs both in one
        // dispatch (the last block does the norm). Not while capturing, whose
        // records read the buffers in between.
        let tail = t == 1 && !capture && fused_add_norm;
        let project_norm = |m: &DeviceMatrix,
                            x: &Buffer,
                            y: &Buffer,
                            sum: &Buffer,
                            u: &Buffer,
                            v: &Buffer,
                            w: &Buffer,
                            eps: f32|
         -> Result<bool> {
            let fused = api.gemm_bf16_add_norm.filter(|_| {
                tail && m.bf16 && m.rows == h && x.len >= m.cols && y.len >= h && sum.len >= h
            });
            let Some(fused) = fused else {
                m.project(x, y, t)?;
                return Ok(false);
            };
            let args = AddNormArgs {
                sum: sum.pointer,
                u: u.pointer,
                v: v.pointer,
                weight: w.pointer,
                norm: self.norm.pointer,
                eps,
                offset: 1.,
            };
            // SAFETY: as in `project`; sum/u/v/norm are live chunk buffers of
            // at least h floats, and y is u or v (read after the GEMV).
            api.check(unsafe {
                fused(
                    api.context,
                    y.pointer,
                    m.buffer.pointer,
                    x.pointer,
                    int(m.rows)?,
                    int(m.cols)?,
                    &args,
                )
            })?;
            Ok(true)
        };
        let mut final_normed = false;
        // The previous layer's last add already normalized x for this layer.
        let mut normed = false;
        for (layer, (b, s)) in model.blocks.iter().zip(&self.mixers).enumerate() {
            if !normed {
                norm(self.x.pointer, &b.input, self.norm.pointer, h, t, b.eps, 1.)?;
            }
            record(layer, "input_norm", &self.norm, h)?;
            // The DeltaNet arm binds its own `b` (a projection).
            let (post, post_eps) = (&b.post, b.eps);
            let post_normed = match (&b.mixer, s) {
                (
                    DeviceMixer::Attention {
                        c,
                        q,
                        k,
                        v,
                        o,
                        qn,
                        kn,
                        qkn,
                    },
                    MixerRun::Attention {
                        mixed,
                        key,
                        value,
                        query,
                        gate,
                        context,
                        keys,
                        values,
                        scores,
                    },
                ) => {
                    project_all(&self.norm, &[(q, mixed), (k, key), (v, value)], t)?;
                    // Rows of mixed/key/value are contiguous; t = token*heads + head.
                    // AttentionKv appends all t keys/values (positions
                    // position..position+t). One dispatch when the bridge fuses them.
                    if api.runs(Op::AttentionQueryKv) {
                        api.run(
                            Op::AttentionQueryKv,
                            (c.heads + c.kv_heads) * t,
                            &[
                                query.pointer,
                                mixed.pointer,
                                qkn.pointer,
                                gate.pointer,
                                keys.pointer,
                                key.pointer,
                                value.pointer,
                                values.pointer,
                            ],
                            &[
                                c.head_dim,
                                c.rotary_dim,
                                self.position,
                                c.heads,
                                c.kv_heads,
                                t,
                            ],
                            &[c.eps, c.theta],
                        )?;
                    } else {
                        api.run(
                            Op::AttentionQuery,
                            c.heads * t,
                            &[query.pointer, mixed.pointer, qn.pointer, gate.pointer],
                            &[c.head_dim, c.rotary_dim, self.position, c.heads],
                            &[c.eps, c.theta],
                        )?;
                        api.run(
                            Op::AttentionKv,
                            c.kv_heads * t,
                            &[
                                keys.pointer,
                                key.pointer,
                                kn.pointer,
                                value.pointer,
                                values.pointer,
                            ],
                            &[c.head_dim, c.rotary_dim, self.position, c.kv_heads],
                            &[c.eps, c.theta],
                        )?;
                    }
                    // All t query rows in one dispatch. Causal: row r sees positions
                    // 0..=position+r (later KV rows of this chunk are already written
                    // but outside its length). Metal: online softmax for t > 1.
                    api.run(
                        Op::AttentionRead,
                        c.heads,
                        &[
                            context.pointer,
                            query.pointer,
                            keys.pointer,
                            values.pointer,
                            gate.pointer,
                            scores.pointer,
                        ],
                        &[
                            c.head_dim,
                            c.heads,
                            c.kv_heads,
                            self.position + 1,
                            self.capacity,
                            t,
                            score_row(self.capacity, c.head_dim),
                        ],
                        &[],
                    )?;
                    project_norm(
                        o,
                        context,
                        &self.mixed,
                        &self.residual,
                        &self.x,
                        &self.mixed,
                        post,
                        post_eps,
                    )?
                }
                (
                    DeviceMixer::Delta {
                        c,
                        qkv,
                        z,
                        a,
                        b,
                        conv,
                        dt,
                        alog,
                        gates,
                        norm: weight,
                        out,
                    },
                    MixerRun::Delta {
                        projected,
                        convolved,
                        conv_state,
                        recurrent,
                        query,
                        key,
                        a: aa,
                        b: bb,
                        z: zz,
                        decay,
                        beta,
                        core,
                        normalized,
                        gated,
                    },
                ) => {
                    project_all(
                        &self.norm,
                        &[(qkv, projected), (z, zz), (a, aa), (b, bb)],
                        t,
                    )?;
                    let channels = qkv.rows;
                    let vd = c.value_heads * c.value_dim;
                    let vh = c.value_heads;
                    let fused = api
                        .delta_mixer
                        .filter(|_| api.runs(Op::DeltaRuleGated) && api.runs(Op::RmsNormGate));
                    if let Some(mixer) = fused.filter(|_| self.snapshot_rows == 0) {
                        // The same four ops (ConvSilu, DeltaQk, DeltaRuleGated,
                        // RmsNormGate) in one bridge call; decode runs them as
                        // one block per value head.
                        let d = DeltaArgs {
                            convolved: convolved.pointer,
                            projected: projected.pointer,
                            conv_w: conv.pointer,
                            conv_state: conv_state.pointer,
                            query: query.pointer,
                            key: key.pointer,
                            core: core.pointer,
                            recurrent: recurrent.pointer,
                            a: aa.pointer,
                            b: bb.pointer,
                            gates: gates.pointer,
                            gated: gated.pointer,
                            norm_w: weight.pointer,
                            z: zz.pointer,
                            key_heads: int(c.key_heads)?,
                            value_heads: int(c.value_heads)?,
                            key_dim: int(c.key_dim)?,
                            value_dim: int(c.value_dim)?,
                            kernel: int(c.kernel)?,
                            tokens: int(t)?,
                            channels: int(channels)?,
                            eps: c.eps,
                        };
                        // SAFETY: the same live buffers and shapes the four ops use.
                        api.check(unsafe { mixer(api.context, &d) })?;
                    } else {
                        // One dispatch per op for the whole chunk. Conv history and the
                        // recurrent state advance token by token inside each thread.
                        api.run(
                            Op::ConvSilu,
                            channels,
                            &[
                                convolved.pointer,
                                projected.pointer,
                                conv.pointer,
                                conv_state.pointer,
                            ],
                            &[c.kernel, t, channels, self.snapshot_rows],
                            &[],
                        )?;
                        api.run(
                            Op::DeltaQk,
                            vh * t,
                            &[query.pointer, convolved.pointer, key.pointer],
                            &[c.key_heads, c.value_heads, c.key_dim, channels],
                            &[],
                        )?;
                        let rule = if api.runs(Op::DeltaRuleGated) {
                            // The rule computes the gates from a, b, dt and A_log.
                            (Op::DeltaRuleGated, [aa.pointer, bb.pointer, gates.pointer])
                        } else {
                            api.run(
                                Op::DeltaGates,
                                vh * t,
                                &[
                                    decay.pointer,
                                    aa.pointer,
                                    bb.pointer,
                                    dt.pointer,
                                    alog.pointer,
                                    beta.pointer,
                                ],
                                &[vh],
                                &[],
                            )?;
                            (
                                Op::DeltaRule,
                                [decay.pointer, beta.pointer, ptr::null_mut()],
                            )
                        };
                        api.run(
                            rule.0,
                            vd,
                            &[
                                core.pointer,
                                query.pointer,
                                key.pointer,
                                convolved.pointer,
                                recurrent.pointer,
                                rule.1[0],
                                rule.1[1],
                                rule.1[2],
                            ],
                            &[
                                c.key_dim,
                                c.value_dim,
                                c.key_heads,
                                c.value_heads,
                                t,
                                channels,
                                self.snapshot_rows,
                            ],
                            &[],
                        )?;
                        if api.runs(Op::RmsNormGate) {
                            api.run(
                                Op::RmsNormGate,
                                vh * t,
                                &[gated.pointer, core.pointer, weight.pointer, zz.pointer],
                                &[c.value_dim],
                                &[c.eps, 0.],
                            )?;
                        } else {
                            norm(
                                core.pointer,
                                weight,
                                normalized.pointer,
                                c.value_dim,
                                vh * t,
                                c.eps,
                                0.,
                            )?;
                            api.run(
                                Op::SiluProduct,
                                vd * t,
                                &[gated.pointer, zz.pointer, normalized.pointer],
                                &[],
                                &[],
                            )?;
                        }
                    }
                    project_norm(
                        out,
                        gated,
                        &self.mixed,
                        &self.residual,
                        &self.x,
                        &self.mixed,
                        post,
                        post_eps,
                    )?
                }
                _ => return Err(err("device mixer/state mismatch")),
            };
            record(layer, "mixed", &self.mixed, h)?;
            if !post_normed {
                add_norm(
                    self.residual.pointer,
                    self.x.pointer,
                    self.mixed.pointer,
                    &b.post,
                    self.norm.pointer,
                    b.eps,
                )?;
            }
            record(layer, "after_mixer", &self.residual, h)?;
            record(layer, "post_norm", &self.norm, h)?;
            let glu = api.gemm_bf16_glu.filter(|_| {
                b.gate.bf16 && b.up.bf16 && b.gate.rows == b.up.rows && b.gate.cols == b.up.cols
            });
            if let Some(glu) = glu {
                require(
                    self.norm.len >= t * h
                        && b.gate.cols == h
                        && self.product.len >= t * i_size
                        && self.gate.len >= t * i_size
                        && self.up.len >= t * i_size
                        && b.gate.rows == i_size,
                    "device GLU shape",
                )?;
                // SAFETY: as in `project`; product, gate and up are distinct
                // chunk buffers of t * intermediate floats on this context.
                api.check(unsafe {
                    glu(
                        api.context,
                        self.product.pointer,
                        self.gate.pointer,
                        self.up.pointer,
                        b.gate.buffer.pointer,
                        b.up.buffer.pointer,
                        self.norm.pointer,
                        int(i_size)?,
                        int(h)?,
                        int(t)?,
                    )
                })?;
            } else {
                project_all(&self.norm, &[(&b.gate, &self.gate), (&b.up, &self.up)], t)?;
                api.run(
                    Op::SiluProduct,
                    i_size * t,
                    &[self.product.pointer, self.gate.pointer, self.up.pointer],
                    &[],
                    &[],
                )?;
            }
            record(layer, "ffn_product", &self.product, i_size)?;
            // Edit overlay (opt-in): read the product back, apply the CPU
            // engine's EditManifest::apply row by row, write it back, project,
            // then add the direction-removal correction. Syncs edited layers only.
            let edit = self
                .edit
                .as_ref()
                .and_then(|e| e.down[layer].as_ref().map(|d| (&e.manifest, d)));
            let correction = if let Some((manifest, down)) = edit {
                let mut product = self.product.read_prefix(t * i_size)?;
                let mut correction = Vec::with_capacity(t * h);
                for row in product.chunks_mut(i_size) {
                    correction.extend(manifest.apply(layer, row, down)?);
                }
                self.product.write_prefix(&product)?;
                record(layer, "ffn_intervened_product", &self.product, i_size)?;
                Some(correction)
            } else {
                None
            };
            // x = residual + ffn, normalized for the next layer (or by the
            // final norm after the last one), in the same dispatch when possible.
            let next = model.blocks.get(layer + 1).filter(|_| fused_add_norm);
            let (weight, eps) = match model.blocks.get(layer + 1) {
                Some(n) => (&n.input, n.eps),
                None => (&model.norm, model.config.rms_norm_eps),
            };
            let down_normed =
                if correction.is_none() && (next.is_some() || layer + 1 == model.blocks.len()) {
                    project_norm(
                        &b.down,
                        &self.product,
                        &self.ffn,
                        &self.x,
                        &self.residual,
                        &self.ffn,
                        weight,
                        eps,
                    )?
                } else {
                    b.down.project(&self.product, &self.ffn, t)?;
                    false
                };
            if let Some(correction) = correction {
                // `mixed` is dead after the residual add above: reuse as scratch.
                self.mixed.write_prefix(&correction)?;
                api.run(
                    Op::Add,
                    h * t,
                    &[self.ffn.pointer, self.ffn.pointer, self.mixed.pointer],
                    &[],
                    &[],
                )?;
            }
            record(layer, "ffn", &self.ffn, h)?;
            // The layer output goes straight to x (x is not an input here),
            // fused with the next layer's input norm when there is one.
            normed = next.is_some();
            if down_normed {
                final_normed = next.is_none();
            } else if let Some(next) = next {
                add_norm(
                    self.x.pointer,
                    self.residual.pointer,
                    self.ffn.pointer,
                    &next.input,
                    self.norm.pointer,
                    next.eps,
                )?;
            } else {
                api.run(
                    Op::Add,
                    h * t,
                    &[self.x.pointer, self.residual.pointer, self.ffn.pointer],
                    &[],
                    &[],
                )?;
            }
            record(layer, "output", &self.x, h)?;
            if capture {
                layers.push(self.x.read_prefix(h)?);
            }
        }
        // Only the last row continues to the final norm and vocabulary.
        if !final_normed {
            norm(
                self.x.at((t - 1) * h)?,
                &model.norm,
                self.norm.pointer,
                h,
                1,
                model.config.rms_norm_eps,
                1.,
            )?;
        }
        // D2H synchronizes the stream, so async kernel faults cannot count as success.
        // A prefill chunk without logits still advances state; its pending
        // work is synchronized by the next readback or explicit flush.
        let logits = match logit_rows {
            0 => Vec::new(),
            1 => {
                model.embedding.project(&self.norm, &self.logits, 1)?;
                self.logits.read_prefix(model.config.vocab_size)?
            }
            _ => {
                // Every row: norm all of x (the last row again, same values) and
                // project them together; the weights are read once.
                norm(
                    self.x.pointer,
                    &model.norm,
                    self.norm.pointer,
                    h,
                    t,
                    model.config.rms_norm_eps,
                    1.,
                )?;
                model.embedding.project(&self.norm, &self.logits, t)?;
                self.logits.read_prefix(t * model.config.vocab_size)?
            }
        };
        let normalized = if capture {
            self.norm.read_prefix(h)?
        } else {
            Vec::new()
        };
        self.position += t;
        Ok(DeviceStep {
            operations,
            layers,
            normalized,
            logits,
        })
    }
    /// Explicit opt-in inspection only; regular inference keeps all state on GPU.
    pub fn inspect_state(&self) -> Result<ModelState> {
        require(!self.failed, "cannot inspect poisoned device state")?;
        let mut layers = Vec::new();
        for (b, s) in self.model.blocks.iter().zip(&self.mixers) {
            layers.push(match (&b.mixer, s) {
                (DeviceMixer::Attention { c, .. }, MixerRun::Attention { keys, values, .. }) => {
                    let d = c.kv_heads * c.head_dim;
                    let read = |b: &Buffer| -> Result<Vec<Vec<f32>>> {
                        if self.position == 0 {
                            Ok(Vec::new())
                        } else {
                            Ok(b.read_prefix(self.position * d)?
                                .chunks_exact(d)
                                .map(<[f32]>::to_vec)
                                .collect())
                        }
                    };
                    LayerState::Attention(KvCache {
                        keys: read(keys)?,
                        values: read(values)?,
                    })
                }
                (
                    DeviceMixer::Delta { c, qkv, .. },
                    MixerRun::Delta {
                        conv_state,
                        recurrent,
                        ..
                    },
                ) => LayerState::Delta(DeltaState {
                    convolution: if c.kernel == 1 {
                        Vec::new()
                    } else {
                        conv_state.read_prefix(qkv.rows * (c.kernel - 1))?
                    },
                    recurrent: recurrent.read_prefix(c.value_heads * c.key_dim * c.value_dim)?,
                }),
                _ => return Err(err("device state type mismatch")),
            });
        }
        Ok(ModelState {
            position: self.position,
            layers,
        })
    }
    /// Host copy of the fixed-size DeltaNet state (conv history + recurrent
    /// matrix) at the current position. Attention KV is not copied: rows
    /// before `position` are never rewritten while the run only moves
    /// forward from here, so `restore` just rewinds the position over them.
    pub fn checkpoint(&self) -> Result<DeviceCheckpoint> {
        require(!self.failed, "cannot checkpoint poisoned device state")?;
        let mut delta = Vec::new();
        for (b, s) in self.model.blocks.iter().zip(&self.mixers) {
            delta.push(match (&b.mixer, s) {
                (
                    DeviceMixer::Delta { c, qkv, .. },
                    MixerRun::Delta {
                        conv_state,
                        recurrent,
                        ..
                    },
                ) => Some(DeltaState {
                    convolution: if c.kernel == 1 {
                        Vec::new()
                    } else {
                        conv_state.read_prefix(qkv.rows * (c.kernel - 1))?
                    },
                    recurrent: recurrent.read_prefix(c.value_heads * c.key_dim * c.value_dim)?,
                }),
                _ => None,
            });
        }
        Ok(DeviceCheckpoint {
            position: self.position,
            delta,
        })
    }
    /// Rewind to `checkpoint`. The caller guarantees the run has only moved
    /// forward since the checkpoint was taken (so KV rows below its position
    /// are still the ones it saw); `position >= checkpoint.position` is checked.
    /// Position 0 with zero states is a fresh run without reallocation.
    pub fn restore(&mut self, checkpoint: &DeviceCheckpoint) -> Result<()> {
        require(
            !self.failed
                && checkpoint.position <= self.position
                && checkpoint.delta.len() == self.mixers.len(),
            "device restore needs an earlier checkpoint of this run",
        )?;
        self.write_delta(&checkpoint.delta)?;
        self.position = checkpoint.position;
        Ok(())
    }
    /// A checkpoint of the initial (all-zero) state, for rewinding to position 0.
    pub fn initial_checkpoint(&self) -> DeviceCheckpoint {
        DeviceCheckpoint {
            position: 0,
            delta: self
                .model
                .blocks
                .iter()
                .map(|b| match &b.mixer {
                    DeviceMixer::Delta { c, qkv, .. } => Some(DeltaState {
                        convolution: vec![0.; qkv.rows * (c.kernel - 1)],
                        recurrent: vec![0.; c.value_heads * c.key_dim * c.value_dim],
                    }),
                    _ => None,
                })
                .collect(),
        }
    }
    /// Upload a complete host state (e.g. a CPU snapshot) into this run.
    /// Shapes must match the model; `state.position` must fit the capacity.
    pub fn load_state(&mut self, state: &ModelState) -> Result<()> {
        require(
            !self.failed
                && state.position <= self.capacity
                && state.layers.len() == self.mixers.len(),
            "device state does not fit this run (capacity/layers)",
        )?;
        let mut delta = Vec::new();
        for ((b, s), l) in self
            .model
            .blocks
            .iter()
            .zip(&self.mixers)
            .zip(&state.layers)
        {
            match (&b.mixer, s, l) {
                (
                    DeviceMixer::Attention { c, .. },
                    MixerRun::Attention { keys, values, .. },
                    LayerState::Attention(kv),
                ) => {
                    let d = c.kv_heads * c.head_dim;
                    require(
                        kv.keys.len() == state.position
                            && kv.values.len() == state.position
                            && kv.keys.iter().chain(&kv.values).all(|r| r.len() == d),
                        "device state KV shape",
                    )?;
                    if state.position > 0 {
                        keys.write_prefix(&kv.keys.concat())?;
                        values.write_prefix(&kv.values.concat())?;
                    }
                    delta.push(None);
                }
                (DeviceMixer::Delta { .. }, MixerRun::Delta { .. }, LayerState::Delta(d)) => {
                    delta.push(Some(d.clone()))
                }
                _ => return Err(err("device state layer type mismatch")),
            }
        }
        self.write_delta(&delta)?;
        self.position = state.position;
        Ok(())
    }
    fn write_delta(&self, delta: &[Option<DeltaState>]) -> Result<()> {
        for ((b, s), d) in self.model.blocks.iter().zip(&self.mixers).zip(delta) {
            match (&b.mixer, s, d) {
                (DeviceMixer::Attention { .. }, MixerRun::Attention { .. }, None) => {}
                (
                    DeviceMixer::Delta { c, qkv, .. },
                    MixerRun::Delta {
                        conv_state,
                        recurrent,
                        ..
                    },
                    Some(d),
                ) => {
                    require(
                        d.convolution.len() == qkv.rows * (c.kernel - 1)
                            && d.recurrent.len() == c.value_heads * c.key_dim * c.value_dim,
                        "device Delta state shape",
                    )?;
                    if !d.convolution.is_empty() {
                        conv_state.write_prefix(&d.convolution)?;
                    }
                    recurrent.write_prefix(&d.recurrent)?;
                }
                _ => return Err(err("device checkpoint layer mismatch")),
            }
        }
        Ok(())
    }
}
/// Host copy of the DeltaNet layers' state at one position (None for
/// attention layers, whose KV rows stay on the device). See `checkpoint`.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceCheckpoint {
    position: usize,
    delta: Vec<Option<DeltaState>>,
}
impl DeviceCheckpoint {
    pub fn position(&self) -> usize {
        self.position
    }
    pub(crate) fn delta(&self) -> &[Option<DeltaState>] {
        &self.delta
    }
}

#[cfg(test)]
mod tests;

mod chat;
mod cli;
pub use chat::DeviceChat;
pub use cli::{
    long_checkpoints, run_cli, validate_device, validate_long, ATTENTION_F16_TOLERANCE,
    MATMUL_RELAXED_TOLERANCE,
};
