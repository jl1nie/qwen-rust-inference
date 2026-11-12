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
    model::Model,
    state::{DeltaState, KvCache, LayerState, ModelState},
    tensor::{finite, require, InferenceError, Matrix, Result},
    weights::{err, TextConfig},
};
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
type Transfer = unsafe extern "C" fn(*mut c_void, *mut f32, *const f32, usize) -> i32;
type Run = unsafe extern "C" fn(*mut c_void, i32, i32, *const Params) -> i32;
type Gemv = unsafe extern "C" fn(*mut c_void, *mut f32, *const f32, *const f32, i32, i32) -> i32;
type Gemm =
    unsafe extern "C" fn(*mut c_void, *mut f32, *const f32, *const f32, i32, i32, i32) -> i32;
/// Native ABI version. v2 added BF16 matrices (`qr_gemv_bf16`, op EmbedBf16)
/// and batched projection `qr_gemm`/`qr_gemm_bf16` for chunked prefill (ADR 0006).
const ABI: i32 = 2;
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
        let words: Vec<u32> = values
            .chunks(2)
            .map(|p| (p[0].to_bits() >> 16) | (p.get(1).map_or(0, |v| v.to_bits()) & 0xffff_0000))
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
    fn read(&self) -> Result<Vec<f32>> {
        self.read_prefix(self.len)
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
}
impl DeviceMatrix {
    fn upload(api: &Rc<Api>, w: &Matrix, format: WeightFormat) -> Result<Self> {
        int(w.rows())?;
        int(w.cols())?;
        let bf16 = format == WeightFormat::Bf16Exact && w.values().iter().all(|&v| bf16_exact(v));
        Ok(Self {
            buffer: if bf16 {
                api.upload_bf16(w.values())?
            } else {
                api.upload(w.values())?
            },
            rows: w.rows(),
            cols: w.cols(),
            bf16,
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
/// Immutable device weights. Run state cannot be shared accidentally across runs.
pub struct DeviceModel {
    api: Rc<Api>,
    config: TextConfig,
    embedding: DeviceMatrix,
    norm: Buffer,
    blocks: Vec<DeviceBlock>,
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
                },
                MixerWeights::Delta(w) => DeviceMixer::Delta {
                    c: w.config.clone(),
                    qkv: matrix(&w.qkv)?,
                    z: matrix(&w.z)?,
                    a: matrix(&w.a)?,
                    b: matrix(&w.b)?,
                    conv: vec(w.conv.values())?,
                    dt: vec(&w.dt)?,
                    alog: vec(&w.a_log)?,
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
        })
    }
    /// Device bytes of one run's state and scratch for `capacity` tokens with
    /// the default prefill chunk. Attention KV/score rows grow with capacity;
    /// activations grow with the chunk; DeltaNet state has a fixed size.
    pub fn run_bytes(&self, capacity: usize) -> usize {
        self.run_bytes_chunked(capacity, capacity.min(PREFILL_CHUNK))
    }
    pub fn run_bytes_chunked(&self, capacity: usize, chunk: usize) -> usize {
        let mut floats = 0usize;
        for b in &self.blocks {
            floats = floats.saturating_add(match &b.mixer {
                DeviceMixer::Attention { c, .. } => {
                    let q = c.heads * c.head_dim;
                    let kv = c.kv_heads * c.head_dim;
                    capacity
                        .saturating_mul(2 * kv + c.heads)
                        .saturating_add(chunk.saturating_mul(q * 5 + kv * 2))
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
            .saturating_add(chunk.saturating_mul(h * 6 + i * 3))
            .saturating_add(self.config.vocab_size)
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
        require(
            capacity > 0 && capacity <= self.config.max_position_embeddings,
            "device capacity must be 1..=model context limit",
        )?;
        require(chunk > 0, "device prefill chunk must be positive")?;
        let chunk = chunk.min(capacity);
        let bytes = self.run_bytes_chunked(capacity, chunk);
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
                        query: alloc(chunk * q)?,
                        gate: alloc(chunk * q)?,
                        context: alloc(chunk * q)?,
                        keys: alloc(capacity * kv)?,
                        values: alloc(capacity * kv)?,
                        scores: alloc(capacity * c.heads)?,
                    }
                }
                DeviceMixer::Delta { c, qkv, .. } => {
                    let q = c.value_heads * c.key_dim;
                    let v = c.value_heads * c.value_dim;
                    MixerRun::Delta {
                        projected: alloc(chunk * qkv.rows)?,
                        convolved: alloc(chunk * qkv.rows)?,
                        conv_state: alloc((qkv.rows * (c.kernel - 1)).max(1))?,
                        recurrent: alloc(c.value_heads * c.key_dim * c.value_dim)?,
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
            capacity,
            chunk,
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
            output: alloc(chunk * h)?,
            logits: alloc(self.config.vocab_size)?,
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
/// Mutable state + temporary activations; never changes immutable uploaded weights.
pub struct DeviceRun<'a> {
    model: &'a DeviceModel,
    capacity: usize,
    /// Rows of every activation buffer = max tokens per forward pass.
    chunk: usize,
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
    output: Buffer,
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
        let fresh = self.model.new_run_chunked(self.capacity, self.chunk)?;
        *self = fresh;
        Ok(())
    }
    pub fn step(&mut self, id: u32, capture: bool) -> Result<DeviceStep> {
        require(!self.failed, "device run failed; reset required")?;
        require(
            (id as usize) < self.model.config.vocab_size && self.position < self.capacity,
            "device token/capacity out of bounds",
        )?;
        let result = self.forward(&[id], capture, true);
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
            let result = self.forward(part, false, final_chunk).and_then(|step| {
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
    pub fn chunk(&self) -> usize {
        self.chunk
    }
    /// One forward pass over `ids.len() <= chunk` consecutive positions.
    /// Every op is dispatched once for all rows (projections as GEMM: each
    /// weight read once per chunk) except the attention read, which runs per
    /// token with its causal length. Time-recurrent ops (conv history, the
    /// DeltaNet rule) loop the tokens in order inside each native thread, so
    /// state evolves exactly as in token-by-token decode.
    /// `capture` (validation) is only available for a single token.
    fn forward(&mut self, ids: &[u32], capture: bool, want_logits: bool) -> Result<DeviceStep> {
        let t = ids.len();
        require(
            t > 0 && t <= self.chunk && (!capture || t == 1),
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
        for (layer, (b, s)) in model.blocks.iter().zip(&self.mixers).enumerate() {
            norm(self.x.pointer, &b.input, self.norm.pointer, h, t, b.eps, 1.)?;
            record(layer, "input_norm", &self.norm, h)?;
            match (&b.mixer, s) {
                (
                    DeviceMixer::Attention {
                        c,
                        q,
                        k,
                        v,
                        o,
                        qn,
                        kn,
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
                    q.project(&self.norm, mixed, t)?;
                    k.project(&self.norm, key, t)?;
                    v.project(&self.norm, value, t)?;
                    let qd = c.heads * c.head_dim;
                    // Rows of mixed/key/value are contiguous; t = token*heads + head.
                    api.run(
                        Op::AttentionQuery,
                        c.heads * t,
                        &[query.pointer, mixed.pointer, qn.pointer, gate.pointer],
                        &[c.head_dim, c.rotary_dim, self.position, c.heads],
                        &[c.eps, c.theta],
                    )?;
                    // Appends all t keys/values (positions position..position+t).
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
                    for r in 0..t {
                        // Causal: row r sees positions 0..=position+r (later KV rows of
                        // this chunk are already written but outside its length).
                        api.run(
                            Op::AttentionRead,
                            c.heads,
                            &[
                                context.at(r * qd)?,
                                query.at(r * qd)?,
                                keys.pointer,
                                values.pointer,
                                gate.at(r * qd)?,
                                scores.pointer,
                            ],
                            &[
                                c.head_dim,
                                c.heads,
                                c.kv_heads,
                                self.position + r + 1,
                                self.capacity,
                            ],
                            &[],
                        )?;
                    }
                    o.project(context, &self.mixed, t)?;
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
                    qkv.project(&self.norm, projected, t)?;
                    z.project(&self.norm, zz, t)?;
                    a.project(&self.norm, aa, t)?;
                    b.project(&self.norm, bb, t)?;
                    let channels = qkv.rows;
                    let vd = c.value_heads * c.value_dim;
                    let vh = c.value_heads;
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
                        &[c.kernel, t, channels],
                        &[],
                    )?;
                    api.run(
                        Op::DeltaQk,
                        vh * t,
                        &[query.pointer, convolved.pointer, key.pointer],
                        &[c.key_heads, c.value_heads, c.key_dim, channels],
                        &[],
                    )?;
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
                    api.run(
                        Op::DeltaRule,
                        vd,
                        &[
                            core.pointer,
                            query.pointer,
                            key.pointer,
                            convolved.pointer,
                            recurrent.pointer,
                            decay.pointer,
                            beta.pointer,
                        ],
                        &[
                            c.key_dim,
                            c.value_dim,
                            c.key_heads,
                            c.value_heads,
                            t,
                            channels,
                        ],
                        &[],
                    )?;
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
                    out.project(gated, &self.mixed, t)?;
                }
                _ => return Err(err("device mixer/state mismatch")),
            }
            record(layer, "mixed", &self.mixed, h)?;
            api.run(
                Op::Add,
                h * t,
                &[self.residual.pointer, self.x.pointer, self.mixed.pointer],
                &[],
                &[],
            )?;
            record(layer, "after_mixer", &self.residual, h)?;
            norm(
                self.residual.pointer,
                &b.post,
                self.norm.pointer,
                h,
                t,
                b.eps,
                1.,
            )?;
            record(layer, "post_norm", &self.norm, h)?;
            b.gate.project(&self.norm, &self.gate, t)?;
            b.up.project(&self.norm, &self.up, t)?;
            api.run(
                Op::SiluProduct,
                i_size * t,
                &[self.product.pointer, self.gate.pointer, self.up.pointer],
                &[],
                &[],
            )?;
            record(layer, "ffn_product", &self.product, i_size)?;
            b.down.project(&self.product, &self.ffn, t)?;
            record(layer, "ffn", &self.ffn, h)?;
            api.run(
                Op::Add,
                h * t,
                &[self.output.pointer, self.residual.pointer, self.ffn.pointer],
                &[],
                &[],
            )?;
            // SAFETY: same context, distinct allocations, t*h within both chunks.
            require(
                self.x.len >= t * h && self.output.len >= t * h,
                "device copy",
            )?;
            api.check(unsafe {
                (api.copy)(api.context, self.x.pointer, self.output.pointer, t * h)
            })?;
            record(layer, "output", &self.x, h)?;
            if capture {
                layers.push(self.x.read_prefix(h)?);
            }
        }
        // Only the last row continues to the final norm and vocabulary.
        norm(
            self.x.at((t - 1) * h)?,
            &model.norm,
            self.norm.pointer,
            h,
            1,
            model.config.rms_norm_eps,
            1.,
        )?;
        // D2H synchronizes the stream, so async kernel faults cannot count as success.
        // A prefill chunk without logits still advances state; its pending
        // work is synchronized by the next readback or explicit flush.
        let logits = if want_logits {
            model.embedding.project(&self.norm, &self.logits, 1)?;
            self.logits.read()?
        } else {
            Vec::new()
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
                    recurrent: recurrent.read()?,
                }),
                _ => return Err(err("device state type mismatch")),
            });
        }
        Ok(ModelState {
            position: self.position,
            layers,
        })
    }
}

#[cfg(test)]
mod tests;

mod cli;
pub use cli::{long_checkpoints, run_cli, validate_device, validate_long};
