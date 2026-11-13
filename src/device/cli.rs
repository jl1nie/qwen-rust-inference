//! Separate commands make the experimental device backend explicit.
//! Edits and traces reuse the CPU engine's EditManifest and Trace (ADR 0010);
//! `<engine>-chat` reuses the CPU chat logic and snapshot files (ADR 0011).
use super::chat::Observer;
use super::*;
use crate::{
    edit::repetition,
    generate::{decode_with, Sampling, StopReason},
    tensor::Backend,
    text::{user_chat, TextTokenizer},
    trace::{Trace, TraceContext},
    validation::{compare, top},
    weights::load_model,
};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};
const HELP: &str = "Experimental device engines, FP32 compute (CPU fallback never automatic):
  rocm-check / rocm-infer    Linux/WSL2, --features rocm, HIP bridge (tools/build-rocm.sh)
  metal-check / metal-infer  macOS Apple GPU, --features metal (tools/build-metal.sh)
  <engine>-check --library PATH [--model-dir DIR (--ids 16,10,16,28 | --text-file PATH --tokens N)]
                 [--f32-weights] [--attention-f16] [--matmul-relaxed] [--metal-kernels LEVEL] [--report PATH]
  <engine>-infer --library PATH --model-dir DIR (--ids 16,10,16,28 | --prompt TEXT | --prompt-file PATH)
                 [--max-new-tokens 16] [--capacity N] [--thinking]
                 [--prefill-chunk N] [--f32-weights] [--attention-f16] [--matmul-relaxed] [--metal-kernels LEVEL] [--report PATH]
                 [--edit manifest.json] [--compare-edit]
                 [--trace PATH --trace-position N [--trace-layer N] [--trace-max-bytes N] [--trace-full]]
  <engine>-verify --library PATH --model-dir DIR --text-file PATH --tokens N [--rows 8] [--runs 20]
                 [--matmul-relaxed] [--metal-kernels LEVEL]
                 Times a speculative-decoding verify pass (1..rows tokens, logits for
                 every row) after N raw tokens, against one decode step; also the
                 DeltaNet checkpoint/restore and agreement with sequential steps (ADR 0023).
Without --model-dir, check compares the independent tiny hybrid fixture.
--ids check copies intermediate activations and ALL state at every position.
--text-file/--tokens is the long-context check: the first N raw tokens of the
file, prefilled in chunks; final-position logits and ALL state are compared
with the CPU at chunk boundaries 1,31,32,63,64,65,...,2^k-1,2^k,2^k+1,...,N.
--capacity defaults to prompt + max-new-tokens (model limit and run budget apply).
Matrices are kept as BF16 on the device when that is lossless (the released
checkpoint is BF16); --f32-weights forces the FP32 copy for A/B comparison.
--attention-f16 (ROCm only, opt-in, changes results; ADR 0014): chunked
prefill attention rounds Q, K, V and the softmax weights to FP16 inside RDNA3
matrix instructions (FP32 accumulate); decode and the KV cache stay FP32.
--metal-kernels reference|simdgroup|tensor (Metal; default tensor) caps the
kernel level (spec/metal.md): reference kernels mirror the CPU code,
simdgroup kernels use SIMD groups and 8x8 matrices, tensor uses Metal 4 tensor
ops (GPU matrix hardware) where available; device_info shows
kernel_level_max. All levels give the same values up to summation order.
--matmul-relaxed (Metal only, opt-in, changes results; ADR 0016) lets the
level-3 prefill projections trade accuracy for speed.
Prefill runs in chunks of --prefill-chunk tokens (default 128): projections
read each weight once per chunk; 1 reproduces token-by-token prefill.
--edit applies the CPU engine's manifest (same validation and EditManifest::apply)
to read-back FFN products of the edited layers; device weights never change.
--compare-edit also runs an unedited baseline and reports both generations.
--trace writes the CPU trace format (NDJSON, engine tagged) for ONE position:
prompt up to it is prefilled in chunks, that position runs alone with captures.
Device traces hold the per-layer checkpoints (input, input_norm, mixer,
after_mixer, post_norm, ffn.product, ffn.intervened_product, ffn.output,
layer) plus final_norm and logits; mixer internals (Q/K/V, attention
probabilities, DeltaNet internals) are CPU-trace only.
  <engine>-chat --library PATH --model-dir DIR [--prompt TEXT | --continue]
                [--load-state PATH] [--save-state PATH] [--system TEXT | --system-file PATH]
                [--max-new-tokens 64] [--capacity 8192] [--prefill-chunk N] [--thinking]
                [--seed N --temperature 0.8 --top-k 40] [--stats] [--json] [--no-cache]
                [--state-max-bytes N] [--f32-weights] [--attention-f16] [--matmul-relaxed] [--metal-kernels LEVEL] [--edit manifest.json]
                [--trace PATH [--trace-position N,M,...] [--trace-layer N] [--trace-max-bytes N] [--trace-full]]
chat is the CPU `chat` on the device: same commands (/continue /reset /save
/load /quit), same prefix reuse and the SAME snapshot files, so a chat saved
here resumes on the CPU and vice versa. A system message is prefilled once and
kept as a rewind point: save it without a prompt (stdin </dev/null) and load it
for every question to reuse a long system document (the token cache).
In chat, /edit PATH|off changes the edit and recomputes the same history under
it (snapshots record the edit); --trace-position takes absolute positions and
/trace N or /trace +K adds more: each is traced whenever it is computed.
Reports are opt-in local plaintext and may contain prompt IDs/generated text.
No automatic downloads or CPU fallback.";
struct Options(BTreeMap<String, String>);
impl Options {
    fn parse(args: &[String]) -> Result<Self> {
        let mut out = BTreeMap::new();
        let mut i = 0;
        while i < args.len() {
            let key = &args[i];
            require(key.starts_with("--"), "expected --option")?;
            let flag = [
                "--thinking",
                "--f32-weights",
                "--attention-f16",
                "--matmul-relaxed",
                "--compare-edit",
                "--trace-full",
                "--continue",
                "--stats",
                "--json",
                "--no-cache",
            ];
            let value = if flag.contains(&key.as_str()) {
                "true".into()
            } else {
                i += 1;
                args.get(i)
                    .ok_or_else(|| err("missing option value"))?
                    .clone()
            };
            require(
                out.insert(key.clone(), value).is_none(),
                "duplicate device option",
            )?;
            i += 1;
        }
        Ok(Self(out))
    }
    fn take(&mut self, key: &str) -> Option<String> {
        self.0.remove(key)
    }
    fn number(&mut self, key: &str, default: usize) -> Result<usize> {
        self.take(key)
            .map(|v| v.parse().map_err(err))
            .transpose()
            .map(|v| v.unwrap_or(default))
    }
    fn finish(self) -> Result<()> {
        require(
            self.0.is_empty(),
            &format!("unknown/unused device options: {:?}", self.0.keys()),
        )
    }
}
fn ids(value: &str) -> Result<Vec<u32>> {
    value
        .split(',')
        .map(|v| v.trim().parse().map_err(err))
        .collect()
}
fn emit(report: &Value, path: Option<PathBuf>) -> Result<()> {
    let text = serde_json::to_string_pretty(report).map_err(err)?;
    if let Some(path) = path {
        // create_new prevents accidentally replacing a previous measurement.
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
        let mut file = options.open(path).map_err(err)?;
        file.write_all(text.as_bytes()).map_err(err)?;
        file.write_all(b"\n").map_err(err)?;
    }
    println!("{text}");
    Ok(())
}
/// Device checkpoint name -> CPU trace operation name (src/trace.rs).
fn trace_operation(name: &str) -> Result<&'static str> {
    Ok(match name {
        "input_norm" => "input_norm",
        "mixed" => "mixer",
        "after_mixer" => "after_mixer",
        "post_norm" => "post_norm",
        "ffn_product" => "ffn.product",
        "ffn_intervened_product" => "ffn.intervened_product",
        "ffn" => "ffn.output",
        "output" => "layer",
        _ => return Err(err("unknown device capture")),
    })
}
/// One captured device step in the CPU trace layout. `embedding` is the host
/// row (device BF16 widening is exact). Without an edit the intervened product
/// equals the product, as on the CPU, so the same record is repeated.
fn trace_device_step(
    trace: &mut Trace<File>,
    run_id: &str,
    position: usize,
    embedding: &[f32],
    step: &DeviceStep,
    layer: Option<usize>,
) -> Result<()> {
    let ctx = |l| TraceContext {
        run_id,
        token_position: position,
        layer: l,
    };
    if layer.is_none_or(|l| l == 0) {
        trace.record(&ctx(0), "embedding", embedding, None)?;
    }
    let mut input = embedding;
    for (l, output) in step.layers.iter().enumerate() {
        if layer.is_none_or(|x| x == l) {
            trace.record(&ctx(l), "input", input, None)?;
            let prefix = format!("layer.{l}.");
            let ops: Vec<_> = step
                .operations
                .iter()
                .filter_map(|(k, v)| k.strip_prefix(&prefix).map(|n| (n, v)))
                .collect();
            let edited = ops.iter().any(|(n, _)| *n == "ffn_intervened_product");
            for (name, values) in ops {
                trace.record(&ctx(l), trace_operation(name)?, values, None)?;
                if name == "ffn_product" && !edited {
                    trace.record(&ctx(l), "ffn.intervened_product", values, None)?;
                }
            }
        }
        input = output;
    }
    let last = step.layers.len() - 1;
    trace.record(&ctx(last), "final_norm", &step.normalized, None)?;
    trace.record(&ctx(last), "logits", &step.logits, None)
}
/// Validate this bridge against Rust CPU and optionally the pinned tiny fixture.
/// Numeric thresholds are fixed in ADR 0004, not command-line tuning knobs.
pub fn validate_device(
    model: &Model,
    device: &DeviceModel,
    ids: &[u32],
    tiny: bool,
) -> Result<Value> {
    require(!ids.is_empty(), "empty validation IDs")?;
    let (atol, rtol) = if tiny {
        (2e-5, 2e-4)
    } else {
        real_tolerance(device)
    };
    let mut gpu = device.new_run(ids.len())?;
    let mut cpu = model.new_state();
    let mut records = Vec::new();
    let mut check = |p: usize, name: &str, actual: &[f32], expected: &[f32]| -> Result<()> {
        let m = compare(
            actual,
            expected,
            atol,
            rtol,
            &format!("position.{p}.{name}"),
        )?;
        records.push(json!({"position":p,"operation":name,"error":m}));
        Ok(())
    };
    for (p, &id) in ids.iter().enumerate() {
        let a = gpu.step(id, true)?;
        let b = model.step(id, &mut cpu, Backend::Reference, None)?;
        let mut operations = a.operations.iter();
        for (l, block) in b.blocks.iter().enumerate() {
            for (name, expected) in [
                ("input_norm", &block.input_norm),
                ("mixed", &block.mixed),
                ("after_mixer", &block.after_mixer),
                ("post_norm", &block.post_norm),
                ("ffn_product", &block.ffn_product),
                ("ffn", &block.ffn),
                ("output", &block.output),
            ] {
                let (actual_name, actual) = operations
                    .next()
                    .ok_or_else(|| err("missing device capture"))?;
                require(
                    *actual_name == format!("layer.{l}.{name}"),
                    "capture order mismatch",
                )?;
                check(p, actual_name, actual, expected)?;
            }
        }
        check(p, "final_norm", &a.normalized, &b.normalized)?;
        check(p, "logits", &a.logits, &b.logits)?;
        let state = gpu.inspect_state()?;
        for (l, (a, b)) in state.layers.iter().zip(&cpu.layers).enumerate() {
            match (a, b) {
                (LayerState::Delta(a), LayerState::Delta(b)) => {
                    if !a.convolution.is_empty() {
                        check(
                            p,
                            &format!("layer.{l}.conv_state"),
                            &a.convolution,
                            &b.convolution,
                        )?;
                    }
                    check(
                        p,
                        &format!("layer.{l}.recurrent_state"),
                        &a.recurrent,
                        &b.recurrent,
                    )?;
                }
                (LayerState::Attention(a), LayerState::Attention(b)) => {
                    for (t, (a, b)) in a.keys.iter().zip(&b.keys).enumerate() {
                        check(p, &format!("layer.{l}.key.{t}"), a, b)?;
                    }
                    for (t, (a, b)) in a.values.iter().zip(&b.values).enumerate() {
                        check(p, &format!("layer.{l}.value.{t}"), a, b)?;
                    }
                }
                _ => return Err(err("state type mismatch")),
            }
        }
        if tiny {
            let f: Value = serde_json::from_str(include_str!("../../tests/fixtures/hybrid.json"))
                .map_err(err)?;
            for (l, values) in a.layers.iter().enumerate() {
                let expected: Vec<f32> =
                    serde_json::from_value(f["layers"][p][l].clone()).map_err(err)?;
                check(p, &format!("fixture.layer.{l}"), values, &expected)?;
            }
            let expected: Vec<f32> = serde_json::from_value(f["logits"][p].clone()).map_err(err)?;
            check(p, "fixture.logits", &a.logits, &expected)?;
        }
    }
    // Test reset and all split points with no intermediate copies. This also
    // checks that capture itself does not change the arithmetic/state.
    let expected = gpu.inspect_state()?;
    for split in 0..=ids.len() {
        gpu.reset()?;
        gpu.prefill(&ids[..split])?;
        gpu.prefill(&ids[split..])?;
        let actual = gpu.inspect_state()?;
        for (l, (a, b)) in actual.layers.iter().zip(&expected.layers).enumerate() {
            match (a, b) {
                (LayerState::Delta(a), LayerState::Delta(b)) => {
                    if !a.convolution.is_empty() {
                        check(
                            split,
                            &format!("split.layer.{l}.conv"),
                            &a.convolution,
                            &b.convolution,
                        )?;
                    }
                    check(
                        split,
                        &format!("split.layer.{l}.recurrent"),
                        &a.recurrent,
                        &b.recurrent,
                    )?;
                }
                (LayerState::Attention(a), LayerState::Attention(b)) => {
                    for (a, b) in a
                        .keys
                        .iter()
                        .chain(&a.values)
                        .zip(b.keys.iter().chain(&b.values))
                    {
                        check(split, &format!("split.layer.{l}.kv"), a, b)?;
                    }
                }
                _ => return Err(err("split state type")),
            }
        }
    }
    Ok(
        json!({"status":"passed","device_info":device.device_info(),"tiny_fixture":tiny,"model_revision":model.revision(),"model_identity":model.identity(),"ids":ids,"atol":atol,"rtol":rtol,"records":records}),
    )
}
/// Boundaries straddling powers of two catch chunk/offset mistakes; N is last.
pub fn long_checkpoints(n: usize) -> Vec<usize> {
    let mut points = vec![1];
    let mut k = 32;
    while k <= n {
        points.extend([k - 1, k, k + 1]);
        k *= 2;
    }
    points.push(n);
    points.retain(|&p| p >= 1 && p <= n);
    points.sort_unstable();
    points.dedup();
    points
}
/// Long-context validation without per-position capture (O(N) not O(N^2)).
/// Device prefill skips intermediate logits, so the comparison at each chunk
/// boundary also proves that skipping them leaves the state untouched.
/// CPU oracle: `Backend::Vector`, itself validated against Reference over 711
/// positions and the official chunk prefill up to 512 (spec/validation-*.json).
/// Real-weight tolerance (ADR 0004), or ADR 0014's for the opt-in 16-bit
/// prefill attention, which rounds Q, K, V and P to FP16.
fn real_tolerance(device: &DeviceModel) -> (f64, f64) {
    match (device.attention_precision(), device.matmul_precision()) {
        ("f32", "f32") => (3e-3, 3e-3),
        ("f32", _) => MATMUL_RELAXED_TOLERANCE,
        _ => ATTENTION_F16_TOLERANCE,
    }
}
/// ADR 0014: (atol, rtol), about 2.3x the worst error measured on the RX 7900
/// XTX (8.86e-3 in the --ids check, 7.96e-3 in the 2048-token long check;
/// FP32 attention 1.4e-4 / 2.6e-4). The greedy token must still match.
pub const ATTENTION_F16_TOLERANCE: (f64, f64) = (2e-2, 2e-2);
/// ADR 0016: (atol, rtol) for the opt-in relaxed prefill GEMMs (Metal), about
/// 2.3x the worst error measured on M5 (5.06e-2 in the 2048-token long check;
/// FP32 3.2e-4). The greedy token must still match.
pub const MATMUL_RELAXED_TOLERANCE: (f64, f64) = (1.2e-1, 1.2e-1);
pub fn validate_long(model: &Model, device: &DeviceModel, ids: &[u32]) -> Result<Value> {
    require(!ids.is_empty(), "empty validation IDs")?;
    let (atol, rtol) = real_tolerance(device);
    let mut gpu = device.new_run(ids.len())?;
    let mut cpu = model.new_state();
    let mut records = Vec::new();
    let mut done = 0;
    let (mut gpu_seconds, mut cpu_seconds) = (0., 0.);
    for point in long_checkpoints(ids.len()) {
        let start = Instant::now();
        let actual = gpu.prefill(&ids[done..point])?.unwrap();
        gpu_seconds += start.elapsed().as_secs_f64();
        let start = Instant::now();
        let mut expected = Vec::new();
        for &id in &ids[done..point] {
            expected = model.step_compact(id, &mut cpu, Backend::Vector, None)?;
        }
        cpu_seconds += start.elapsed().as_secs_f64();
        done = point;
        let mut worst = compare(
            &actual,
            &expected,
            atol,
            rtol,
            &format!("position.{point}.logits"),
        )?;
        let same_top1 = top(&actual, 1)[0].0 == top(&expected, 1)[0].0;
        require(
            same_top1,
            &format!("position.{point}: greedy token differs"),
        )?;
        let state = gpu.inspect_state()?;
        require(
            state.position == cpu.position,
            "long check position mismatch",
        )?;
        let mut state_elements = 0;
        for (l, (a, b)) in state.layers.iter().zip(&cpu.layers).enumerate() {
            let pairs: Vec<(&[f32], &[f32], String)> = match (a, b) {
                (LayerState::Delta(a), LayerState::Delta(b)) => vec![
                    (
                        &a.convolution[..],
                        &b.convolution[..],
                        format!("layer.{l}.conv"),
                    ),
                    (
                        &a.recurrent[..],
                        &b.recurrent[..],
                        format!("layer.{l}.recurrent"),
                    ),
                ],
                (LayerState::Attention(a), LayerState::Attention(b)) => {
                    require(a.keys.len() == b.keys.len(), "long check KV length")?;
                    // One flattened comparison per tensor keeps the report small.
                    let flat = |v: &[Vec<f32>]| v.concat();
                    let (ak, bk, av, bv) = (
                        flat(&a.keys),
                        flat(&b.keys),
                        flat(&a.values),
                        flat(&b.values),
                    );
                    let mk = compare(
                        &ak,
                        &bk,
                        atol,
                        rtol,
                        &format!("position.{point}.layer.{l}.keys"),
                    )?;
                    let mv = compare(
                        &av,
                        &bv,
                        atol,
                        rtol,
                        &format!("position.{point}.layer.{l}.values"),
                    )?;
                    state_elements += mk.elements + mv.elements;
                    for m in [mk, mv] {
                        if m.max_absolute > worst.max_absolute {
                            worst = m;
                        }
                    }
                    Vec::new()
                }
                _ => return Err(err("long check state type")),
            };
            for (a, b, name) in pairs {
                if a.is_empty() && b.is_empty() {
                    continue;
                }
                let m = compare(a, b, atol, rtol, &format!("position.{point}.{name}"))?;
                state_elements += m.elements;
                if m.max_absolute > worst.max_absolute {
                    worst = m;
                }
            }
        }
        eprintln!("long check: {point}/{} positions passed", ids.len());
        records.push(json!({"position":point,"logits_top1":top(&actual,1)[0].0,"state_elements":state_elements,"worst":worst}));
    }
    Ok(
        json!({"status":"passed","mode":"long","device_info":device.device_info(),"model_revision":model.revision(),
        "model_identity":model.identity(),"tokens":ids.len(),"atol":atol,"rtol":rtol,"cpu_backend":"vector",
        "device_prefill_seconds":gpu_seconds,"cpu_prefill_seconds":cpu_seconds,"records":records}),
    )
}
pub fn run_cli(args: &[String]) -> Result<()> {
    if args.is_empty() || args.len() == 1 || args.get(1).is_some_and(|v| v == "--help" || v == "-h")
    {
        println!("{HELP}");
        return Ok(());
    }
    let (kind, command) = match args[0].as_str() {
        #[cfg(all(feature = "rocm", target_os = "linux"))]
        "rocm-check" => (Kind::Rocm, "check"),
        #[cfg(all(feature = "rocm", target_os = "linux"))]
        "rocm-infer" => (Kind::Rocm, "infer"),
        #[cfg(all(feature = "metal", target_os = "macos"))]
        "metal-check" => (Kind::Metal, "check"),
        #[cfg(all(feature = "metal", target_os = "macos"))]
        "metal-infer" => (Kind::Metal, "infer"),
        #[cfg(all(feature = "rocm", target_os = "linux"))]
        "rocm-chat" => return run_chat(Kind::Rocm, &args[1..]),
        #[cfg(all(feature = "metal", target_os = "macos"))]
        "metal-chat" => return run_chat(Kind::Metal, &args[1..]),
        #[cfg(all(feature = "rocm", target_os = "linux"))]
        "rocm-verify" => return run_verify(Kind::Rocm, &args[1..]),
        #[cfg(all(feature = "metal", target_os = "macos"))]
        "metal-verify" => return run_verify(Kind::Metal, &args[1..]),
        _ => return Err(err("device command not available in this build")),
    };
    let mut o = Options::parse(&args[1..])?;
    let library = PathBuf::from(
        o.take("--library")
            .ok_or_else(|| err("--library PATH required"))?,
    );
    let directory = o.take("--model-dir").map(PathBuf::from);
    let input = o.take("--ids").map(|v| ids(&v)).transpose()?;
    let report = o.take("--report").map(PathBuf::from);
    let (text_file, tokens) = if command == "check" {
        (o.take("--text-file").map(PathBuf::from), o.take("--tokens"))
    } else {
        (None, None)
    };
    let prompt_file = if command == "infer" {
        o.take("--prompt-file")
    } else {
        None
    };
    let prompt = if command == "infer" {
        let inline = o.take("--prompt");
        require(
            !(inline.is_some() && prompt_file.is_some()),
            "use one of --prompt/--prompt-file",
        )?;
        match prompt_file {
            Some(path) => Some(std::fs::read_to_string(path).map_err(err)?),
            None => inline,
        }
    } else {
        None
    };
    let thinking = if command == "infer" {
        o.take("--thinking").is_some()
    } else {
        false
    };
    let max = if command == "infer" {
        o.number("--max-new-tokens", 16)?
    } else {
        0
    };
    let capacity = if command == "infer" {
        o.take("--capacity")
            .map(|v| v.parse::<usize>().map_err(err))
            .transpose()?
    } else {
        None
    };
    let chunk = if command == "infer" {
        o.number("--prefill-chunk", PREFILL_CHUNK)?
    } else {
        PREFILL_CHUNK
    };
    let format = if o.take("--f32-weights").is_some() {
        WeightFormat::F32
    } else {
        WeightFormat::Bf16Exact
    };
    let attention_f16 = o.take("--attention-f16").is_some();
    let matmul_relaxed = o.take("--matmul-relaxed").is_some();
    let kernel_level = o
        .take("--metal-kernels")
        .map(|v| KernelLevel::parse(&v))
        .transpose()?;
    // Edit and trace options mirror the CPU `infer` command (src/cli.rs).
    let edit_path = if command == "infer" {
        o.take("--edit").map(PathBuf::from)
    } else {
        None
    };
    let edit: Option<crate::edit::EditManifest> = edit_path
        .as_ref()
        .map(|p| {
            let bytes = std::fs::read(p).map_err(err)?;
            serde_json::from_slice(&bytes).map_err(err)
        })
        .transpose()?;
    let compare_edit = command == "infer" && o.take("--compare-edit").is_some();
    require(
        !compare_edit || edit.is_some(),
        "--compare-edit requires --edit",
    )?;
    let (trace_path, trace_position, trace_layer, trace_cap, trace_full) = if command == "infer" {
        let parse = |v: Option<String>| v.map(|v| v.parse::<usize>().map_err(err)).transpose();
        (
            o.take("--trace").map(PathBuf::from),
            parse(o.take("--trace-position"))?,
            parse(o.take("--trace-layer"))?,
            parse(o.take("--trace-max-bytes"))?,
            o.take("--trace-full").is_some(),
        )
    } else {
        (None, None, None, None, false)
    };
    require(
        trace_path.is_some()
            || (trace_position.is_none()
                && trace_layer.is_none()
                && trace_cap.is_none()
                && !trace_full),
        "trace options require --trace",
    )?;
    require(
        trace_path.is_none() || trace_position.is_some(),
        "device --trace requires --trace-position (one position)",
    )?;
    require(
        !trace_full || (trace_layer.is_some() && trace_cap.is_some()),
        "--trace-full requires --trace-layer and --trace-max-bytes",
    )?;
    o.finish()?;
    if command == "infer" {
        require(
            directory.is_some() && prompt.is_some() != input.is_some(),
            "infer needs --model-dir and exactly one of --ids/--prompt",
        )?;
        require(!thinking || prompt.is_some(), "--thinking needs --prompt")?;
    }
    let long = text_file.is_some() || tokens.is_some();
    if long {
        require(
            text_file.is_some() && tokens.is_some() && directory.is_some() && input.is_none(),
            "long check needs --model-dir, --text-file and --tokens (not --ids)",
        )?;
    }
    if command == "check" && !long {
        require(
            directory.is_some() == input.is_some(),
            "real check requires BOTH --model-dir and --ids; tiny check accepts neither",
        )?;
    }
    let tiny = directory.is_none();
    let load_start = Instant::now();
    let model = if let Some(dir) = &directory {
        load_model(dir)?
    } else {
        Model::from_fixture(
            &serde_json::from_str(include_str!("../../tests/fixtures/hybrid.json")).map_err(err)?,
        )?
    };
    let load_seconds = load_start.elapsed().as_secs_f64();
    let tokenizer = if prompt.is_some() || long {
        Some(TextTokenizer::load(directory.as_ref().unwrap())?)
    } else {
        None
    };
    let input = if let Some(input) = input {
        input
    } else if let (Some(path), Some(n)) = (&text_file, &tokens) {
        // Raw text without a chat template: the check is about positions, not chat.
        let n: usize = n.parse().map_err(err)?;
        let text = std::fs::read_to_string(path).map_err(err)?;
        let mut all = tokenizer.as_ref().unwrap().encode(&text)?;
        require(
            all.len() >= n,
            &format!("text file has only {} tokens, --tokens {n}", all.len()),
        )?;
        all.truncate(n);
        all
    } else if let Some(prompt) = prompt {
        tokenizer
            .as_ref()
            .unwrap()
            .encode(&user_chat(&prompt, thinking))?
    } else {
        let f: Value =
            serde_json::from_str(include_str!("../../tests/fixtures/hybrid.json")).map_err(err)?;
        serde_json::from_value(f["ids"].clone()).map_err(err)?
    };
    require(!input.is_empty(), "empty input IDs")?;
    let upload_start = Instant::now();
    let mut device = DeviceModel::load_with(&model, &library, kind, format)?;
    if let Some(level) = kernel_level {
        device.set_kernel_level(level)?;
    }
    if matmul_relaxed {
        device.set_matmul_relaxed(true)?;
    }
    if attention_f16 {
        device.set_attention_f16(true)?;
    }
    let (bf16_matrices, matrices, weight_bytes) = device.weight_summary();
    let upload_seconds = upload_start.elapsed().as_secs_f64();
    if command == "check" {
        let mut result = if long {
            validate_long(&model, &device, &input)?
        } else {
            validate_device(&model, &device, &input, tiny)?
        };
        result["library"] = json!(library);
        result["engine"] = json!(kind.name());
        result["weights"] = json!({"format":device.weight_format(),"bf16_matrices":bf16_matrices,"matrices":matrices,"matrix_bytes":weight_bytes});
        result["attention"] = json!(device.attention_precision());
        result["matmul"] = json!(device.matmul_precision());
        result["kernels"] = json!(device.kernel_level_name());
        return emit(&result, report);
    }
    let capacity = capacity
        .unwrap_or_else(|| (input.len() + max.max(1)).min(model.config().max_position_embeddings));
    require(
        trace_layer.is_none_or(|l| l < model.blocks().len()),
        "trace layer out of range",
    )?;
    let mut trace = trace_path
        .as_ref()
        .map(|p| OpenOptions::new().write(true).create_new(true).open(p))
        .transpose()
        .map_err(err)?
        .map(|file| {
            Trace::new(file, trace_cap.unwrap_or(1_048_576), trace_full)
                .with_engine(kind.trace_name())
        });
    let run_id = format!(
        "{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(err)?
            .as_nanos()
    );
    let mut traced = 0usize;
    let mut run = device.new_run_chunked(capacity, chunk)?;
    run.set_edit(&model, edit.as_ref())?;
    let start = Instant::now();
    // Chunked prefill, except the traced prompt position, which runs alone.
    let last = match (trace.as_mut(), trace_position) {
        (Some(t), Some(p)) if p < input.len() => {
            run.prefill(&input[..p])?;
            let step = run.step(input[p], true)?;
            let row = model.embedding.row(input[p] as usize)?;
            trace_device_step(t, &run_id, p, &row, &step, trace_layer)?;
            traced += 1;
            match run.prefill(&input[p + 1..])? {
                Some(logits) => logits,
                None => step.logits,
            }
        }
        _ => run.prefill(&input)?.unwrap(),
    };
    let prefill_seconds = start.elapsed().as_secs_f64();
    let last_top = top(&last, 5);
    let last_logits = last.clone();
    let mut decode_seconds = 0.;
    let mut decode_steps = 0;
    let start = Instant::now();
    let generated = decode_with(
        &input,
        last,
        max,
        &[248044, 248046],
        Sampling::Greedy,
        |token| {
            if run.position() >= capacity {
                return Ok(None);
            }
            let position = run.position();
            let capture = trace.is_some() && trace_position == Some(position);
            let start = Instant::now();
            let step = run.step(token, capture)?;
            decode_seconds += start.elapsed().as_secs_f64();
            decode_steps += 1;
            if capture {
                let row = model.embedding.row(token as usize)?;
                trace_device_step(
                    trace.as_mut().unwrap(),
                    &run_id,
                    position,
                    &row,
                    &step,
                    trace_layer,
                )?;
                traced += 1;
            }
            Ok(Some(step.logits))
        },
    )?;
    let generation_seconds = start.elapsed().as_secs_f64();
    require(
        trace.is_none() || traced == 1,
        "trace position was not reached (prompt + generated tokens)",
    )?;
    // Unedited baseline on a fresh run of the same device (not timed).
    let edit_report = if let Some(manifest) = &edit {
        let mut value = json!({"manifest":edit_path,"edits":manifest.edits.len(),
            "edited_repetition":repetition(&generated.history.generated)});
        if compare_edit {
            let mut base = device.new_run_chunked(capacity, chunk)?;
            let base_last = base.prefill(&input)?.unwrap();
            let delta = base_last
                .iter()
                .zip(&last_logits)
                .map(|(a, b)| f64::from(a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            let baseline = decode_with(
                &input,
                base_last,
                max,
                &[248044, 248046],
                Sampling::Greedy,
                |token| {
                    if base.position() >= capacity {
                        return Ok(None);
                    }
                    Ok(Some(base.step(token, false)?.logits))
                },
            )?;
            value["compare"] = json!({"prefill_logits_delta_l2":delta,
                "baseline_generated_ids":baseline.history.generated,
                "baseline_repetition":repetition(&baseline.history.generated),
                "baseline_text":tokenizer.as_ref().map(|t| t.decode(&baseline.history.generated)).transpose()?});
        }
        Some(value)
    } else {
        None
    };
    let text = tokenizer
        .as_ref()
        .map(|t| t.decode(&generated.history.generated))
        .transpose()?;
    emit(
        &json!({"backend":match kind { Kind::Metal => "experimental_metal", _ => "experimental_rocm" },
        "weights":{"format":device.weight_format(),"bf16_matrices":bf16_matrices,"matrices":matrices,"matrix_bytes":weight_bytes},"device_info":device.device_info(),"attention":device.attention_precision(),"matmul":device.matmul_precision(),"kernels":device.kernel_level_name(),"model_revision":model.revision(),"model_identity":model.identity(),"library":library,
        "prompt_ids":input,"generated_ids":generated.history.generated,"text":text,"stop":generated.stop,"prefill_top5":last_top,
        "load_seconds":load_seconds,"prefill_chunk":run.chunk(),"bridge_and_upload_seconds":upload_seconds,"capacity":capacity,"prefill_seconds":prefill_seconds,
        "decode_forward_seconds":decode_seconds,"decode_forward_steps":decode_steps,
        "decode_forward_tokens_per_second":if decode_seconds>0. { Some(decode_steps as f64/decode_seconds) } else { None },
        "generation_seconds_including_selection":generation_seconds,"release_build":!cfg!(debug_assertions),
        "edit":edit_report,"trace":trace_path.as_ref().map(|p| json!({"path":p,"position":trace_position,"layer":trace_layer,"full":trace_full})),
        "timing_note":"single run, no warmup; full-logit readback and queue completion included; validation captures disabled (a traced position and edited layers synchronize)"}),
        report,
    )
}

fn read_edit(path: &Path) -> Result<crate::edit::EditManifest> {
    serde_json::from_slice(&std::fs::read(path).map_err(err)?).map_err(err)
}
/// `<engine>-chat`: the CPU `chat` command (src/cli.rs) on a device run.
fn run_chat(kind: Kind, args: &[String]) -> Result<()> {
    use crate::snapshot::DEFAULT_LIMIT;
    use std::io::{self, BufRead, IsTerminal};
    let mut o = Options::parse(args)?;
    let library = PathBuf::from(
        o.take("--library")
            .ok_or_else(|| err("--library PATH required"))?,
    );
    let dir = PathBuf::from(
        o.take("--model-dir")
            .ok_or_else(|| err("--model-dir DIR required"))?,
    );
    let load = o.take("--load-state");
    let save = o.take("--save-state");
    let prompt = o.take("--prompt");
    let continuation = o.take("--continue").is_some();
    require(
        !(prompt.is_some() && continuation),
        "use --prompt or --continue, not both",
    )?;
    let max = o.number("--max-new-tokens", 64)?;
    require(max > 0, "chat max-new-tokens must be positive")?;
    let capacity = o.number("--capacity", 8192)?;
    let chunk = o.number("--prefill-chunk", PREFILL_CHUNK)?;
    let limit = o
        .take("--state-max-bytes")
        .map(|v| v.parse::<u64>().map_err(err))
        .transpose()?
        .unwrap_or(DEFAULT_LIMIT);
    let stats = o.take("--stats").is_some();
    let json = o.take("--json").is_some();
    let no_cache = o.take("--no-cache").is_some();
    let thinking = o.take("--thinking").is_some();
    let seed = o.take("--seed");
    let temp = o.take("--temperature");
    let top_k = o.take("--top-k");
    let system = match (o.take("--system"), o.take("--system-file")) {
        (Some(_), Some(_)) => return Err(err("use --system or --system-file, not both")),
        (Some(s), None) => Some(s),
        (None, Some(path)) => Some(std::fs::read_to_string(path).map_err(err)?),
        (None, None) => None,
    };
    let format = if o.take("--f32-weights").is_some() {
        WeightFormat::F32
    } else {
        WeightFormat::Bf16Exact
    };
    let attention_f16 = o.take("--attention-f16").is_some();
    let matmul_relaxed = o.take("--matmul-relaxed").is_some();
    let kernel_level = o
        .take("--metal-kernels")
        .map(|v| KernelLevel::parse(&v))
        .transpose()?;
    let edit = o
        .take("--edit")
        .map(|p| read_edit(Path::new(&p)))
        .transpose()?;
    let trace_path = o.take("--trace").map(PathBuf::from);
    let trace_positions: BTreeSet<usize> = o
        .take("--trace-position")
        .map(|v| {
            // Empty items are skipped, so `$(seq -s, 0 39)` (trailing comma) works.
            v.split(',')
                .map(str::trim)
                .filter(|p| !p.is_empty())
                .map(|p| p.parse().map_err(err))
                .collect()
        })
        .transpose()?
        .unwrap_or_default();
    let trace_layer = o
        .take("--trace-layer")
        .map(|v| v.parse::<usize>().map_err(err))
        .transpose()?;
    let trace_cap = o
        .take("--trace-max-bytes")
        .map(|v| v.parse::<usize>().map_err(err))
        .transpose()?;
    let trace_full = o.take("--trace-full").is_some();
    require(
        trace_path.is_some()
            || (trace_positions.is_empty()
                && trace_layer.is_none()
                && trace_cap.is_none()
                && !trace_full),
        "trace options require --trace",
    )?;
    require(
        !trace_full || (trace_layer.is_some() && trace_cap.is_some()),
        "--trace-full requires --trace-layer and --trace-max-bytes",
    )?;
    require(
        load.is_none()
            || !(seed.is_some()
                || temp.is_some()
                || top_k.is_some()
                || thinking
                || system.is_some()),
        "loaded state carries sampling/system/thinking settings; do not override them",
    )?;
    require(
        seed.is_some() || (temp.is_none() && top_k.is_none()),
        "temperature/top-k require --seed",
    )?;
    o.finish()?;
    let tokenizer = TextTokenizer::load(&dir)?;
    let model = load_model(&dir)?;
    require(
        trace_layer.is_none_or(|l| l < model.blocks().len()),
        "trace layer out of range",
    )?;
    let mut device = DeviceModel::load_with(&model, &library, kind, format)?;
    if let Some(level) = kernel_level {
        device.set_kernel_level(level)?;
    }
    if matmul_relaxed {
        device.set_matmul_relaxed(true)?;
    }
    if attention_f16 {
        device.set_attention_f16(true)?;
    }
    let capacity = capacity.min(model.config().max_position_embeddings);
    // One trace file for the whole chat. Records of a position computed again
    // (after a rewind or an edit change) carry the next pass number in run_id.
    let pass = std::rc::Rc::new(std::cell::Cell::new(0usize));
    let trace = trace_path
        .as_ref()
        .map(|p| OpenOptions::new().write(true).create_new(true).open(p))
        .transpose()
        .map_err(err)?
        .map(|file| {
            std::rc::Rc::new(std::cell::RefCell::new(
                Trace::new(file, trace_cap.unwrap_or(1_048_576), trace_full)
                    .with_engine(kind.trace_name()),
            ))
        });
    let run_base = format!(
        "{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(err)?
            .as_nanos()
    );
    let observer = || -> Option<Observer<'_>> {
        let trace = trace.clone()?;
        let (pass, model, run_base) = (pass.clone(), &model, run_base.clone());
        let mut failed = false;
        Some(Box::new(move |position, id, step| {
            if failed {
                return;
            }
            let run_id = format!("{run_base}-pass{}", pass.get());
            let result = model.embedding.row(id as usize).and_then(|row| {
                trace_device_step(
                    &mut trace.borrow_mut(),
                    &run_id,
                    position,
                    &row,
                    step,
                    trace_layer,
                )
            });
            match result {
                Ok(()) => eprintln!("[trace position={position} pass={}]", pass.get()),
                Err(e) => {
                    failed = true;
                    eprintln!("trace stopped: {e}");
                }
            }
        }))
    };
    let new_run = || device.new_run_chunked(capacity, chunk);
    let mut chat = if let Some(path) = &load {
        let start = Instant::now();
        let chat = DeviceChat::load(&model, new_run()?, &tokenizer, Path::new(path), limit)?;
        if stats {
            eprintln!(
                "loaded positions={} elapsed_ms={:.2}",
                chat.position(),
                start.elapsed().as_secs_f64() * 1000.
            );
        }
        chat
    } else {
        let sampling = match seed {
            Some(seed) => Sampling::Seeded {
                seed: seed.parse().map_err(err)?,
                temperature: temp.unwrap_or_else(|| "0.8".into()).parse().map_err(err)?,
                top_k: top_k.unwrap_or_else(|| "40".into()).parse().map_err(err)?,
            },
            None => Sampling::Greedy,
        };
        DeviceChat::new(&model, new_run()?, sampling, thinking, system)?
    };
    chat.set_observer(trace_positions, observer());
    if let Some(e) = &edit {
        let n = chat.set_edit(&model, Some(e))?;
        if n > 0 {
            eprintln!("[edit applied; recomputed {n} positions]");
        }
    }
    let prime = |chat: &mut DeviceChat| -> Result<()> {
        if no_cache {
            return Ok(());
        }
        let start = Instant::now();
        let n = chat.prime(&tokenizer)?;
        if stats && n > 0 {
            eprintln!(
                "system prefill={n} elapsed_ms={:.2}",
                start.elapsed().as_secs_f64() * 1000.
            );
        }
        Ok(())
    };
    prime(&mut chat)?;
    let display = |turn: &crate::chat::Turn, elapsed: f64| -> Result<()> {
        if json {
            println!("{}", serde_json::to_string(turn).map_err(err)?);
        } else {
            println!("{}", turn.text);
        }
        if stats {
            eprintln!(
                "stop={:?} generated={} reused={} prefill={} elapsed_ms={elapsed:.2}",
                turn.stop,
                turn.generated.len(),
                turn.reused_tokens,
                turn.prefill_tokens
            );
        } else if turn.stop == StopReason::MaxTokens {
            eprintln!("[maximum reached; /continue resumes this answer]");
        }
        Ok(())
    };
    if prompt.is_some() || continuation {
        let start = Instant::now();
        pass.set(pass.get() + 1);
        let turn = match &prompt {
            Some(p) => chat.user_turn(&tokenizer, p, max, !no_cache)?,
            None => chat.continue_turn(&tokenizer, max)?,
        };
        display(&turn, start.elapsed().as_secs_f64() * 1000.)?;
    } else {
        let stdin = io::stdin();
        let interactive = stdin.is_terminal();
        let mut lines = stdin.lock().lines();
        loop {
            if interactive {
                print!("you> ");
                io::stdout().flush().map_err(err)?;
            }
            let Some(line) = lines.next() else { break };
            let line = line.map_err(err)?;
            let text = line.trim();
            if text.is_empty() {
                continue;
            }
            if text == "/quit" {
                break;
            }
            if text == "/reset" {
                if let Err(e) = chat.reset().and_then(|_| prime(&mut chat)) {
                    eprintln!("error: {e}");
                }
                continue;
            }
            if let Some(path) = text.strip_prefix("/save ") {
                match chat.save(&model, Path::new(path.trim()), limit) {
                    Ok(()) => eprintln!("saved: {}", path.trim()),
                    Err(e) => eprintln!("error: {e}"),
                }
                continue;
            }
            if let Some(path) = text.strip_prefix("/load ") {
                // Validate the file on the host first so a bad file keeps the chat.
                match crate::snapshot::load_chat(&model, &tokenizer, Path::new(path.trim()), limit)
                {
                    Ok(c) => {
                        pass.set(pass.get() + 1);
                        if let Err(e) = chat.replace(&model, c) {
                            eprintln!("error: {e} (started a new chat)");
                        }
                    }
                    Err(e) => eprintln!("error: {e}"),
                }
                continue;
            }
            if let Some(arg) = text.strip_prefix("/edit") {
                let arg = arg.trim();
                let result = match arg {
                    "" => {
                        eprintln!("edit: {}", if chat.edit().is_some() { "on" } else { "off" });
                        continue;
                    }
                    "off" => Ok(None),
                    path => read_edit(Path::new(path)).map(Some),
                };
                pass.set(pass.get() + 1);
                match result.and_then(|e| chat.set_edit(&model, e.as_ref())) {
                    Ok(n) => eprintln!(
                        "[edit {}; recomputed {n} positions]",
                        if arg == "off" { "removed" } else { "applied" }
                    ),
                    Err(e) => eprintln!("error: {e}"),
                }
                continue;
            }
            if let Some(arg) = text.strip_prefix("/trace") {
                let arg = arg.trim();
                if trace.is_none() {
                    eprintln!("error: start with --trace PATH to record traces");
                } else if arg.is_empty() {
                    eprintln!(
                        "position={} observed={:?}",
                        chat.position(),
                        chat.observed()
                    );
                } else {
                    let p = match arg.strip_prefix('+') {
                        Some(k) => k.parse::<usize>().map(|k| chat.position() + k),
                        None => arg.parse::<usize>(),
                    };
                    match p {
                        Ok(p) => {
                            chat.observe(p);
                            eprintln!("[will trace position {p}]");
                        }
                        Err(e) => eprintln!("error: {e}"),
                    }
                }
                continue;
            }
            let start = Instant::now();
            let result = if text == "/continue" {
                chat.continue_turn(&tokenizer, max)
            } else if text.starts_with('/') {
                eprintln!("commands: /continue /reset /save PATH /load PATH /edit [PATH|off] /trace [N|+K] /quit");
                continue;
            } else {
                pass.set(pass.get() + 1);
                chat.user_turn(&tokenizer, &line, max, !no_cache)
            };
            match result {
                Ok(turn) => display(&turn, start.elapsed().as_secs_f64() * 1000.)?,
                Err(e) => eprintln!("error: {e} (a device error needs /reset)"),
            }
        }
    }
    if let Some(path) = save {
        chat.save(&model, Path::new(&path), limit)?;
    }
    Ok(())
}

/// `<engine>-verify` (ADR 0023): how long a verify pass of speculative
/// decoding takes. After N raw tokens of a text, from one checkpoint, it
/// times (median of `--runs`) a decode step and `DeviceRun::verify` over the
/// next 1..=rows tokens, the DeltaNet checkpoint and restore, and checks the
/// verify logits against the same tokens decoded one at a time.
fn run_verify(kind: Kind, args: &[String]) -> Result<()> {
    let mut o = Options::parse(args)?;
    let library = PathBuf::from(
        o.take("--library")
            .ok_or_else(|| err("--library PATH required"))?,
    );
    let dir = PathBuf::from(
        o.take("--model-dir")
            .ok_or_else(|| err("--model-dir DIR required"))?,
    );
    let text = o
        .take("--text-file")
        .ok_or_else(|| err("--text-file PATH required"))?;
    let n = o.number("--tokens", 2048)?;
    let rows = o.number("--rows", VERIFY_ROWS)?;
    let runs = o.number("--runs", 20)?.max(1);
    let relaxed = o.take("--matmul-relaxed").is_some();
    let level = o
        .take("--metal-kernels")
        .map(|v| KernelLevel::parse(&v))
        .transpose()?;
    o.finish()?;
    require(
        n > 0 && (1..=VERIFY_ROWS).contains(&rows),
        "--tokens must be positive and --rows 1..=8",
    )?;
    let tokenizer = TextTokenizer::load(&dir)?;
    let mut ids = tokenizer.encode(&std::fs::read_to_string(&text).map_err(err)?)?;
    require(
        ids.len() >= n + rows,
        &format!("text has {} tokens, need --tokens + --rows", ids.len()),
    )?;
    ids.truncate(n + rows);
    let model = load_model(&dir)?;
    let mut device = DeviceModel::load_with(&model, &library, kind, WeightFormat::Bf16Exact)?;
    if relaxed {
        device.set_matmul_relaxed(true)?;
    }
    if let Some(level) = level {
        device.set_kernel_level(level)?;
    }
    // With snapshots when the bridge has them, so the rollback is timed too.
    let mut run = match device.new_run_rollback(n + rows, rows.clamp(2, VERIFY_ROWS) - 1) {
        Ok(run) => run,
        Err(_) => device.new_run(n + rows)?,
    };
    let rollback = run.verify_rollback(&ids[..1]).is_ok();
    run.reset()?;
    let t = Instant::now();
    run.prefill(&ids[..n])?;
    let prefill_s = t.elapsed().as_secs_f64();
    let median = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let cp = run.checkpoint()?;
    let next = &ids[n..n + rows];
    // Sequential reference: the same tokens, one decode step each.
    let mut sequential = Vec::new();
    for &id in next {
        sequential.push(run.step(id, false)?.logits);
    }
    let argmax = |l: &[f32]| {
        let mut b = 0;
        for i in 1..l.len() {
            if l[i] > l[b] {
                b = i;
            }
        }
        b
    };
    let mut decode = Vec::new();
    let (mut save, mut load) = (Vec::new(), Vec::new());
    for _ in 0..runs {
        run.restore(&cp)?;
        let t = Instant::now();
        run.step(next[0], false)?;
        decode.push(t.elapsed().as_secs_f64() * 1e3);
        let t = Instant::now();
        let c = run.checkpoint()?;
        save.push(t.elapsed().as_secs_f64() * 1e3);
        let t = Instant::now();
        run.restore(&c)?;
        load.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let decode_ms = median(decode);
    let mut table = Vec::new();
    for r in 1..=rows {
        let mut times = Vec::new();
        let mut out = Vec::new();
        for _ in 0..runs {
            run.restore(&cp)?;
            let t = Instant::now();
            out = run.verify(&next[..r])?;
            times.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let ms = median(times);
        // verify_rollback stores r - 1 snapshots of every DeltaNet state.
        // (accept's device copy, ~19 MiB, was below the run-to-run noise.)
        let mut kept = Vec::new();
        if rollback {
            for _ in 0..runs {
                run.restore(&cp)?;
                let t = Instant::now();
                run.verify_rollback(&next[..r])?;
                kept.push(t.elapsed().as_secs_f64() * 1e3);
            }
        }
        let mut max_diff = 0f32;
        let mut same = 0;
        for (a, b) in out.iter().zip(&sequential) {
            same += usize::from(argmax(a) == argmax(b));
            for (x, y) in a.iter().zip(b) {
                max_diff = max_diff.max((x - y).abs());
            }
        }
        // Bit pattern of every row, so kernel variants can be compared exactly.
        let mut hash = <sha2::Sha256 as sha2::Digest>::new();
        for row in &out {
            for v in row {
                sha2::Digest::update(&mut hash, v.to_bits().to_le_bytes());
            }
        }
        table.push(json!({
            "rows": r,
            "logits_sha256": format!("{:x}", sha2::Digest::finalize(hash)),
            "ms": ms,
            "relative_to_decode": ms / decode_ms,
            "verify_rollback_ms": (!kept.is_empty()).then(|| median(kept.clone())),
            "argmax_equal_rows": same,
            "max_abs_logit_diff": max_diff,
        }));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "engine": kind.name(),
            "device": device.device_info(),
            "kernel_level": device.kernel_level_name(),
            "matmul": device.matmul_precision(),
            "context_tokens": n,
            "prefill_s": prefill_s,
            "runs": runs,
            "decode_step_ms": decode_ms,
            "delta_checkpoint_ms": median(save),
            "delta_restore_ms": median(load),
            "verify": table,
        }))
        .map_err(err)?
    );
    Ok(())
}
