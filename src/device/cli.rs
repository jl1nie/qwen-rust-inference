//! Separate commands make the experimental device backend explicit; CPU chat
//! snapshots are intentionally not silently accepted on a different engine.
//! Edits and traces reuse the CPU engine's EditManifest and Trace (ADR 0010).
use super::*;
use crate::{
    edit::repetition,
    generate::{decode_with, Sampling},
    tensor::Backend,
    text::{user_chat, TextTokenizer},
    trace::{Trace, TraceContext},
    validation::{compare, top},
    weights::load_model,
};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs::{File, OpenOptions},
    io::Write,
    path::PathBuf,
    time::{Instant, SystemTime, UNIX_EPOCH},
};
const HELP: &str = "Experimental device engines, FP32 compute (CPU fallback never automatic):
  rocm-check / rocm-infer    Linux/WSL2, --features rocm, HIP bridge (tools/build-rocm.sh)
  metal-check / metal-infer  macOS Apple GPU, --features metal (tools/build-metal.sh)
  <engine>-check --library PATH [--model-dir DIR (--ids 16,10,16,28 | --text-file PATH --tokens N)]
                 [--f32-weights] [--report PATH]
  <engine>-infer --library PATH --model-dir DIR (--ids 16,10,16,28 | --prompt TEXT | --prompt-file PATH)
                 [--max-new-tokens 16] [--capacity N] [--thinking]
                 [--prefill-chunk N] [--f32-weights] [--report PATH]
                 [--edit manifest.json] [--compare-edit]
                 [--trace PATH --trace-position N [--trace-layer N] [--trace-max-bytes N] [--trace-full]]
Without --model-dir, check compares the independent tiny hybrid fixture.
--ids check copies intermediate activations and ALL state at every position.
--text-file/--tokens is the long-context check: the first N raw tokens of the
file, prefilled in chunks; final-position logits and ALL state are compared
with the CPU at chunk boundaries 1,31,32,63,64,65,...,2^k-1,2^k,2^k+1,...,N.
--capacity defaults to prompt + max-new-tokens (model limit and run budget apply).
Matrices are kept as BF16 on the device when that is lossless (the released
checkpoint is BF16); --f32-weights forces the FP32 copy for A/B comparison.
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
Reports are opt-in local plaintext and may contain prompt IDs/generated text.
No automatic downloads, CPU fallback, sampling, or chat snapshots.";
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
                "--compare-edit",
                "--trace-full",
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
    let (atol, rtol) = if tiny { (2e-5, 2e-4) } else { (3e-3, 3e-3) };
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
pub fn validate_long(model: &Model, device: &DeviceModel, ids: &[u32]) -> Result<Value> {
    require(!ids.is_empty(), "empty validation IDs")?;
    let (atol, rtol) = (3e-3, 3e-3);
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
    let device = DeviceModel::load_with(&model, &library, kind, format)?;
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
            Trace::new(file, trace_cap.unwrap_or(1_048_576), trace_full).with_engine(match kind {
                Kind::Metal => "metal",
                _ => "rocm",
            })
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
            trace_device_step(t, &run_id, p, row, &step, trace_layer)?;
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
                    row,
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
        "weights":{"format":device.weight_format(),"bf16_matrices":bf16_matrices,"matrices":matrices,"matrix_bytes":weight_bytes},"device_info":device.device_info(),"model_revision":model.revision(),"model_identity":model.identity(),"library":library,
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
