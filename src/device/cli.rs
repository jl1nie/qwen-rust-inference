//! Separate commands make the experimental device backend explicit; CPU chat
//! snapshots/edits are intentionally not silently accepted on a different engine.
use super::*;
use crate::{
    generate::{decode_with, Sampling},
    tensor::Backend,
    text::{user_chat, TextTokenizer},
    validation::{compare, top},
    weights::load_model,
};
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs::OpenOptions, io::Write, path::PathBuf, time::Instant};
const HELP: &str = "Experimental FP32 device engines (CPU fallback never automatic):
  rocm-check / rocm-infer    Linux/WSL2, --features rocm, HIP bridge (tools/build-rocm.sh)
  metal-check / metal-infer  macOS Apple GPU, --features metal (tools/build-metal.sh)
  <engine>-check --library PATH [--model-dir DIR --ids 16,10,16,28] [--report PATH]
  <engine>-infer --library PATH --model-dir DIR (--ids 16,10,16,28 | --prompt TEXT)
                 [--max-new-tokens 16] [--capacity 1024] [--thinking] [--report PATH]
Without --model-dir, check compares the independent tiny hybrid fixture.
Check copies intermediate activations and ALL state; infer does not.
Reports are opt-in local plaintext and may contain prompt IDs/generated text.
No automatic downloads, CPU fallback, sampling, edits, or chat snapshots.";
struct Options(BTreeMap<String, String>);
impl Options {
    fn parse(args: &[String]) -> Result<Self> {
        let mut out = BTreeMap::new();
        let mut i = 0;
        while i < args.len() {
            let key = &args[i];
            require(key.starts_with("--"), "expected --option")?;
            let value = if key == "--thinking" {
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
    let prompt = if command == "infer" {
        o.take("--prompt")
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
        o.number("--capacity", 1024)?
    } else {
        0
    };
    o.finish()?;
    if command == "infer" {
        require(
            directory.is_some() && prompt.is_some() != input.is_some(),
            "infer needs --model-dir and exactly one of --ids/--prompt",
        )?;
        require(!thinking || prompt.is_some(), "--thinking needs --prompt")?;
    }
    if command == "check" {
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
    let tokenizer = if prompt.is_some() {
        Some(TextTokenizer::load(directory.as_ref().unwrap())?)
    } else {
        None
    };
    let input = if let Some(input) = input {
        input
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
    let device = DeviceModel::load(&model, &library, kind)?;
    let upload_seconds = upload_start.elapsed().as_secs_f64();
    if command == "check" {
        let mut result = validate_device(&model, &device, &input, tiny)?;
        result["library"] = json!(library);
        result["engine"] = json!(kind.name());
        return emit(&result, report);
    }
    let mut run = device.new_run(capacity)?;
    let start = Instant::now();
    let last = run.prefill(&input)?.unwrap();
    let prefill_seconds = start.elapsed().as_secs_f64();
    let last_top = top(&last, 5);
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
            let start = Instant::now();
            let logits = run.step(token, false)?.logits;
            decode_seconds += start.elapsed().as_secs_f64();
            decode_steps += 1;
            Ok(Some(logits))
        },
    )?;
    let generation_seconds = start.elapsed().as_secs_f64();
    let text = tokenizer
        .as_ref()
        .map(|t| t.decode(&generated.history.generated))
        .transpose()?;
    emit(
        &json!({"backend":match kind { Kind::Metal => "experimental_metal_fp32", _ => "experimental_rocm_fp32" },"device_info":device.device_info(),"model_revision":model.revision(),"model_identity":model.identity(),"library":library,
        "prompt_ids":input,"generated_ids":generated.history.generated,"text":text,"stop":generated.stop,"prefill_top5":last_top,
        "load_seconds":load_seconds,"bridge_and_upload_seconds":upload_seconds,"capacity":capacity,"prefill_seconds":prefill_seconds,
        "decode_forward_seconds":decode_seconds,"decode_forward_steps":decode_steps,
        "decode_forward_tokens_per_second":if decode_seconds>0. { Some(decode_steps as f64/decode_seconds) } else { None },
        "generation_seconds_including_selection":generation_seconds,"release_build":!cfg!(debug_assertions),
        "timing_note":"single run, no warmup; full-logit readback and queue completion included; validation captures disabled"}),
        report,
    )
}
