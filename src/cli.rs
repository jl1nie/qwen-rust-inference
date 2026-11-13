use qwen_rust_inference::{
    block::MixerWeights,
    edit::{repetition, EditManifest},
    generate::{generate, generate_observed, Sampling},
    model::Model,
    state::KvCache,
    tensor::Backend,
    text::{user_chat, TextTokenizer},
    trace::Trace,
    weights::{load_config, load_model, load_mtp},
};
use std::{
    collections::BTreeMap,
    error::Error,
    fs::{self, OpenOptions},
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};
const HELP:&str="qwen-rust-inference\n  demo ffn [--unit N] [--trace PATH --trace-full]\n  demo attention [--cache]\n  demo deltanet\n  demo lesson --trace PATH\n  demo hybrid [--max-new-tokens N]\n  chat --model-dir DIR [--prompt TEXT | --continue] [--load-state PATH] [--save-state PATH]\n       [--system TEXT | --system-file PATH] [--max-new-tokens 64] [--stats] [--json] [--no-cache]\n       [--edit manifest.json]   (/edit PATH|off recomputes the history under the new edit)\n       [--trace PATH [--trace-position N,M,...] [--trace-layer N] [--trace-max-bytes N] [--trace-full]]   (/trace [N|+K])\n  bench --model-dir DIR [--backend scalar|gemm|vector] [--capture full|compact]\n        [--ids 16,10,16,28] [--decode-steps 8] [--warmup 1] [--runs 3]\n  inspect-config --model-dir DIR\n  mtp-accept --model-dir DIR --prompt-file PATH (one prompt per line) [--max-new-tokens 128] [--depth 3] [--thinking]\n  infer --model-dir DIR (--ids 16,10,16,28 | --prompt TEXT) [--raw] [--thinking]\n        [--max-new-tokens 16] [--seed N --temperature 0.8 --top-k 40]\n        [--edit manifest.json] [--compare-edit]\n        [--trace PATH --trace-layer N --trace-position N --trace-max-bytes N --trace-full]\n  rocm-check / rocm-infer / rocm-chat --help (Linux/WSL2, requires --features rocm)\n  metal-check / metal-infer / metal-chat --help (macOS, requires --features metal)\nNo automatic downloads. Text-only; CPU single-batch by default.";
struct Options {
    values: BTreeMap<String, String>,
}
impl Options {
    fn parse(args: &[String]) -> Result<Self, Box<dyn Error>> {
        let mut values = BTreeMap::new();
        let mut i = 0;
        while i < args.len() {
            let key = &args[i];
            if !key.starts_with("--") {
                return Err(format!("unexpected argument: {key}").into());
            }
            let flag = matches!(
                key.as_str(),
                "--continue"
                    | "--stats"
                    | "--json"
                    | "--no-cache"
                    | "--cache"
                    | "--raw"
                    | "--thinking"
                    | "--trace-full"
                    | "--compare-edit"
            );
            let value = if flag {
                "true".into()
            } else {
                i += 1;
                args.get(i).ok_or("missing option value")?.clone()
            };
            if values.insert(key.clone(), value).is_some() {
                return Err(format!("duplicate option: {key}").into());
            }
            i += 1;
        }
        Ok(Self { values })
    }
    fn take(&mut self, key: &str) -> Option<String> {
        self.values.remove(key)
    }
    fn number(&mut self, key: &str, default: usize) -> Result<usize, Box<dyn Error>> {
        Ok(self
            .take(key)
            .map(|v| v.parse())
            .transpose()?
            .unwrap_or(default))
    }
    fn finish(self) -> Result<(), Box<dyn Error>> {
        if self.values.is_empty() {
            Ok(())
        } else {
            Err(format!("unknown/unused options: {:?}", self.values.keys()).into())
        }
    }
}
fn demo_model() -> Result<Model, Box<dyn Error>> {
    Ok(Model::from_fixture(&serde_json::from_str(include_str!(
        "../tests/fixtures/hybrid.json"
    ))?)?)
}
pub fn run(args: &[String]) -> Result<(), Box<dyn Error>> {
    if args.is_empty() || args == ["--help"] || args == ["-h"] {
        println!("{HELP}");
        return Ok(());
    }
    if matches!(args.first().map(String::as_str), Some("chat" | "bench"))
        && args.get(1).is_some_and(|s| s == "--help" || s == "-h")
    {
        println!("{HELP}");
        return Ok(());
    }
    if args[0] == "demo" {
        let which = args.get(1).ok_or("missing demo name")?;
        let mut o = Options::parse(&args[2..])?;
        let m = demo_model()?;
        match which.as_str() {
            "lesson" => {
                let path = o
                    .take("--trace")
                    .ok_or("demo lesson requires --trace PATH")?;
                o.finish()?;
                let file = OpenOptions::new().write(true).create_new(true).open(path)?;
                let mut trace = Trace::new(file, 1_048_576, true);
                let lesson = qwen_rust_inference::lesson::write_lesson(&mut trace)?;
                println!("{}", serde_json::to_string_pretty(&lesson)?);
            }
            "attention" => {
                let cached = o.take("--cache").is_some();
                o.finish()?;
                let MixerWeights::Attention(w) = &m.blocks()[3].mixer else {
                    unreachable!()
                };
                let xs = (0..3)
                    .map(|i| m.embedding().row(i).map(|r| r.into_owned()))
                    .collect::<Result<Vec<_>, _>>()?;
                let full = w.sequence(&xs, Backend::Reference)?;
                let mut state = KvCache::default();
                for (p, x) in xs.iter().enumerate() {
                    let a = if cached {
                        w.step(x, p, &mut state, Backend::Reference)?
                    } else {
                        full[p].clone()
                    };
                    println!(
                        "position={p} q={:?} k={:?} v={:?} output={:?} KV_positions={}",
                        a.query,
                        a.key,
                        a.value,
                        a.output,
                        state.keys.len()
                    );
                }
            }
            "deltanet" => {
                o.finish()?;
                let MixerWeights::Delta(w) = &m.blocks()[0].mixer else {
                    unreachable!()
                };
                let mut s = w.new_state();
                for p in 0..3 {
                    let a = w.step(&m.embedding().row(p)?, &mut s, Backend::Reference)?;
                    println!("position={p} q={:?} k={:?} beta={:?} decay_log={:?}\nconvolution={:?}\nrecurrent={:?}\noutput={:?}",a.query,a.key,a.beta,a.decay,s.convolution,s.recurrent,a.output);
                }
            }
            "hybrid" => {
                let max = o.number("--max-new-tokens", 8)?;
                o.finish()?;
                let result = generate(&m, &[1, 4, 2], max, &[], Sampling::Greedy, None)?;
                println!(
                    "synthetic model: {:?}; stop={:?}",
                    result.history, result.stop
                );
            }
            _ => return Err(format!("unsupported demo: {which}").into()),
        }
        return Ok(());
    }
    let mut o = Options::parse(&args[1..])?;
    let dir = PathBuf::from(o.take("--model-dir").ok_or("--model-dir is required")?);
    if args[0] == "chat" {
        return run_chat(&dir, o);
    }
    if args[0] == "mtp-accept" {
        return run_mtp_accept(&dir, o);
    }
    if args[0] == "inspect-config" {
        o.finish()?;
        println!("{}", serde_json::to_string_pretty(&load_config(&dir)?)?);
        return Ok(());
    }
    if args[0] == "bench" {
        let ids: Vec<u32> = o
            .take("--ids")
            .unwrap_or_else(|| "16,10,16,28".into())
            .split(',')
            .map(str::parse)
            .collect::<Result<_, _>>()?;
        let steps = o.number("--decode-steps", 8)?;
        let warmup = o.number("--warmup", 1)?;
        let runs = o.number("--runs", 3)?;
        let backend = qwen_rust_inference::benchmark::parse_backend(
            &o.take("--backend").unwrap_or_else(|| "vector".into()),
        )?;
        let capture = o.take("--capture").unwrap_or_else(|| "compact".into());
        if capture != "full" && capture != "compact" {
            return Err("--capture must be full or compact".into());
        }
        o.finish()?;
        let start = std::time::Instant::now();
        let model = load_model(&dir)?;
        let load_ms = start.elapsed().as_secs_f64() * 1000.;
        let result = qwen_rust_inference::benchmark::benchmark(
            &model,
            &ids,
            steps,
            warmup,
            runs,
            backend,
            capture == "full",
        )?;
        println!(
            "{}",
            serde_json::to_string_pretty(
                &serde_json::json!({"load_ms":load_ms,"benchmark":result})
            )?
        );
        return Ok(());
    }
    if args[0] != "infer" {
        return Err(HELP.into());
    }
    let prompt = o.take("--prompt");
    let ids = o.take("--ids");
    if prompt.is_some() == ids.is_some() {
        return Err("specify exactly one of --prompt or --ids".into());
    }
    let raw = o.take("--raw").is_some();
    let thinking = o.take("--thinking").is_some();
    if ids.is_some() && (raw || thinking) {
        return Err("--raw/--thinking require --prompt".into());
    }
    let max = o.number("--max-new-tokens", 16)?;
    let sampling = if let Some(seed) = o.take("--seed") {
        Sampling::Seeded {
            seed: seed.parse()?,
            temperature: o
                .take("--temperature")
                .unwrap_or_else(|| "0.8".into())
                .parse()?,
            top_k: o.number("--top-k", 40)?,
        }
    } else {
        Sampling::Greedy
    };
    let edit: Option<EditManifest> = o
        .take("--edit")
        .map(|p| fs::read(p).map(|v| serde_json::from_slice(&v)))
        .transpose()?
        .transpose()?;
    let compare_edit = o.take("--compare-edit").is_some();
    if compare_edit && edit.is_none() {
        return Err("--compare-edit requires --edit".into());
    }
    let trace_path = o.take("--trace");
    let trace_full = o.take("--trace-full").is_some();
    let trace_layer = o
        .take("--trace-layer")
        .map(|v| v.parse::<usize>())
        .transpose()?;
    let trace_position = o
        .take("--trace-position")
        .map(|v| v.parse::<usize>())
        .transpose()?;
    let explicit_cap = o.take("--trace-max-bytes");
    let cap = explicit_cap
        .as_ref()
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(1_048_576);
    if trace_path.is_none()
        && (trace_full
            || trace_layer.is_some()
            || trace_position.is_some()
            || explicit_cap.is_some())
    {
        return Err("trace options require --trace".into());
    }
    if trace_full && (trace_layer.is_none() || trace_position.is_none() || explicit_cap.is_none()) {
        return Err(
            "real-model --trace-full requires --trace-layer, --trace-position, --trace-max-bytes"
                .into(),
        );
    }
    o.finish()?;
    let tokenizer = if prompt.is_some() {
        Some(TextTokenizer::load(&dir)?)
    } else {
        None
    };
    let ids: Vec<u32> = if let Some(ids) = ids {
        ids.split(',').map(str::parse).collect::<Result<_, _>>()?
    } else {
        let prompt = prompt.unwrap();
        let text = if raw {
            prompt
        } else {
            user_chat(&prompt, thinking)
        };
        tokenizer.as_ref().unwrap().encode(&text)?
    };
    let model = load_model(&dir)?;
    if trace_layer.is_some_and(|l| l >= model.blocks().len()) {
        return Err("trace layer out of range".into());
    }
    let eos = vec![248044, 248046];
    let mut trace = trace_path
        .map(|p| OpenOptions::new().write(true).create_new(true).open(p))
        .transpose()?
        .map(|file| Trace::new(file, cap, trace_full));
    let run_id = format!(
        "{}-{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    );
    if compare_edit {
        let mut base = model.new_state();
        let mut changed = model.new_state();
        for (p, &id) in ids.iter().enumerate() {
            let before = model.step(id, &mut base, Backend::Optimized, None)?;
            let after = model.step(id, &mut changed, Backend::Optimized, edit.as_ref())?;
            let delta: Vec<_> = after
                .logits
                .iter()
                .zip(&before.logits)
                .map(|(a, b)| a - b)
                .collect();
            // An intervention changes the expected answer, so report differences
            // without treating nonzero values as a failed equivalence test.
            eprintln!(
                "edit position={p} logits_delta_l2={}",
                delta
                    .iter()
                    .map(|v| f64::from(*v).powi(2))
                    .sum::<f64>()
                    .sqrt()
            );
            for (l, (a, b)) in before.blocks.iter().zip(&after.blocks).enumerate() {
                eprintln!(
                    "layer={l} ffn_delta_max={} residual_delta_max={}",
                    a.ffn
                        .iter()
                        .zip(&b.ffn)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0_f32, f32::max),
                    a.output
                        .iter()
                        .zip(&b.output)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0_f32, f32::max)
                );
            }
        }
        let baseline = generate(&model, &ids, max, &eos, sampling, None)?;
        eprintln!(
            "baseline_ids={:?} repetition={}",
            baseline.history.generated,
            serde_json::to_string(&repetition(&baseline.history.generated))?
        );
    }
    let result = if trace.is_none() {
        generate(&model, &ids, max, &eos, sampling, edit.as_ref())?
    } else {
        generate_observed(
            &model,
            &ids,
            max,
            &eos,
            sampling,
            edit.as_ref(),
            |p, step| {
                if trace_position.is_none_or(|position| position == p) {
                    if let Some(t) = &mut trace {
                        t.step(&run_id, p, step, trace_layer)?;
                    }
                }
                Ok(())
            },
        )?
    };
    if let Some(t) = tokenizer {
        // 応答終了の制御トークンだけを表示から外す。思考区切りは残す。
        let mut visible = result.history.generated.as_slice();
        if visible.last().is_some_and(|id| eos.contains(id)) {
            visible = &visible[..visible.len() - 1];
        }
        println!("{}", t.decode(visible)?);
    } else {
        println!("{}", serde_json::to_string(&result.history.generated)?);
    }
    eprintln!(
        "stop={:?} repetition={}",
        result.stop,
        serde_json::to_string(&repetition(&result.history.generated))?
    );
    Ok(())
}

/// Chat trace: CPU trace records (all mixer internals) for chosen absolute
/// positions, each time they are computed; `-passN` counts the chat calls.
struct ChatTrace {
    trace: Trace<fs::File>,
    positions: std::collections::BTreeSet<usize>,
    layer: Option<usize>,
    run_id: String,
    pass: usize,
    stopped: bool,
}
impl qwen_rust_inference::session::Watch for ChatTrace {
    fn wants(&self, position: usize) -> bool {
        !self.stopped && self.positions.contains(&position)
    }
    fn observe(
        &mut self,
        position: usize,
        a: &qwen_rust_inference::model::StepActivations,
    ) -> qwen_rust_inference::tensor::Result<()> {
        let run_id = format!("{}-pass{}", self.run_id, self.pass);
        match self.trace.step(&run_id, position, a, self.layer) {
            Ok(()) => eprintln!("[trace position={position} pass={}]", self.pass),
            // The chat goes on without traces rather than failing the turn.
            Err(e) => {
                self.stopped = true;
                eprintln!("trace stopped: {e}");
            }
        }
        Ok(())
    }
}
fn run_chat(dir: &std::path::Path, mut o: Options) -> Result<(), Box<dyn Error>> {
    use qwen_rust_inference::{
        chat::ChatSession,
        snapshot::{load_chat, save_chat, DEFAULT_LIMIT},
    };
    use std::io::{self, BufRead, IsTerminal, Write};
    let load = o.take("--load-state");
    let save = o.take("--save-state");
    let prompt = o.take("--prompt");
    let continuation = o.take("--continue").is_some();
    if prompt.is_some() && continuation {
        return Err("use --prompt or --continue, not both".into());
    }
    let max = o.number("--max-new-tokens", 64)?;
    if max == 0 {
        return Err("chat max-new-tokens must be positive".into());
    }
    let limit = o
        .take("--state-max-bytes")
        .map(|v| v.parse::<u64>())
        .transpose()?
        .unwrap_or(DEFAULT_LIMIT);
    let stats = o.take("--stats").is_some();
    let json = o.take("--json").is_some();
    let backend = o.take("--backend");
    let seed = o.take("--seed");
    let temp = o.take("--temperature");
    let top_k = o.take("--top-k");
    let thinking = o.take("--thinking").is_some();
    let system = match (o.take("--system"), o.take("--system-file")) {
        (Some(_), Some(_)) => return Err("use --system or --system-file, not both".into()),
        (Some(s), None) => Some(s),
        (None, Some(path)) => Some(std::fs::read_to_string(path)?),
        (None, None) => None,
    };
    let no_cache = o.take("--no-cache").is_some();
    let read_edit = |p: &str| -> Result<EditManifest, Box<dyn Error>> {
        Ok(serde_json::from_slice(&std::fs::read(p)?)?)
    };
    let edit = o.take("--edit").map(|p| read_edit(&p)).transpose()?;
    let trace_path = o.take("--trace");
    let positions: std::collections::BTreeSet<usize> = o
        .take("--trace-position")
        // Empty items are skipped, so `$(seq -s, 0 39)` (trailing comma) works.
        .map(|v| {
            v.split(',')
                .map(str::trim)
                .filter(|p| !p.is_empty())
                .map(str::parse)
                .collect()
        })
        .transpose()?
        .unwrap_or_default();
    let trace_layer = o
        .take("--trace-layer")
        .map(|v| v.parse::<usize>())
        .transpose()?;
    let trace_cap = o
        .take("--trace-max-bytes")
        .map(|v| v.parse::<usize>())
        .transpose()?;
    let trace_full = o.take("--trace-full").is_some();
    if trace_path.is_none()
        && (!positions.is_empty() || trace_layer.is_some() || trace_cap.is_some() || trace_full)
    {
        return Err("trace options require --trace".into());
    }
    if trace_full && (trace_layer.is_none() || trace_cap.is_none()) {
        return Err("--trace-full requires --trace-layer and --trace-max-bytes".into());
    }
    if load.is_some()
        && (backend.is_some()
            || seed.is_some()
            || temp.is_some()
            || top_k.is_some()
            || thinking
            || system.is_some())
    {
        return Err(
            "loaded state carries backend/sampling/system/thinking settings; do not override them"
                .into(),
        );
    }
    if seed.is_none() && (temp.is_some() || top_k.is_some()) {
        return Err("temperature/top-k require --seed".into());
    }
    o.finish()?;
    let tokenizer = TextTokenizer::load(dir)?;
    let model = load_model(dir)?;
    if trace_layer.is_some_and(|l| l >= model.blocks().len()) {
        return Err("trace layer out of range".into());
    }
    let mut trace = trace_path
        .map(|p| OpenOptions::new().write(true).create_new(true).open(p))
        .transpose()?
        .map(|file| -> Result<ChatTrace, Box<dyn Error>> {
            Ok(ChatTrace {
                trace: Trace::new(file, trace_cap.unwrap_or(1_048_576), trace_full),
                positions,
                layer: trace_layer,
                run_id: format!(
                    "{}-{}",
                    std::process::id(),
                    SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
                ),
                pass: 0,
                stopped: false,
            })
        })
        .transpose()?;
    // A watch borrowed afresh for each call. The pass number (run_id suffix)
    // advances on user turns, edit changes and loads, as in device chat.
    macro_rules! watch {
        () => {
            trace
                .as_mut()
                .map(|t| t as &mut dyn qwen_rust_inference::session::Watch)
        };
    }
    macro_rules! next_pass {
        () => {
            if let Some(t) = trace.as_mut() {
                t.pass += 1;
            }
        };
    }
    let mut chat = if let Some(path) = load {
        load_chat(&model, &tokenizer, std::path::Path::new(&path), limit)?
    } else {
        let b = backend
            .as_ref()
            .map(|b| qwen_rust_inference::benchmark::parse_backend(b))
            .transpose()?
            .unwrap_or_else(Backend::inference);
        let sampling = if let Some(seed) = seed {
            Sampling::Seeded {
                seed: seed.parse()?,
                temperature: temp.unwrap_or_else(|| "0.8".into()).parse()?,
                top_k: top_k.unwrap_or_else(|| "40".into()).parse()?,
            }
        } else {
            Sampling::Greedy
        };
        let mut chat = ChatSession::new(&model, b, sampling, thinking, system)?;
        if !no_cache {
            let start = std::time::Instant::now();
            let n = chat.prime_watched(&model, &tokenizer, watch!())?;
            if stats && n > 0 {
                eprintln!(
                    "system prefill={n} elapsed_ms={:.2}",
                    start.elapsed().as_secs_f64() * 1000.
                );
            }
        }
        chat
    };
    if let Some(e) = edit {
        chat.set_edit_watched(&model, Some(e), watch!())?;
    }
    let display =
        |turn: &qwen_rust_inference::chat::Turn, elapsed: f64| -> Result<(), Box<dyn Error>> {
            if json {
                println!("{}", serde_json::to_string(turn)?);
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
            } else if turn.stop == qwen_rust_inference::generate::StopReason::MaxTokens {
                eprintln!("[maximum reached; /continue resumes this answer]");
            }
            Ok(())
        };
    if prompt.is_some() || continuation {
        let start = std::time::Instant::now();
        let result = if let Some(prompt) = prompt {
            next_pass!();
            chat.user_turn_watched(&model, &tokenizer, &prompt, max, !no_cache, watch!())?
        } else {
            chat.continue_turn_watched(&model, &tokenizer, max, watch!())?
        };
        display(&result, start.elapsed().as_secs_f64() * 1000.)?;
    } else {
        let stdin = io::stdin();
        let interactive = stdin.is_terminal();
        let mut lines = stdin.lock().lines();
        loop {
            if interactive {
                print!("you> ");
                io::stdout().flush()?;
            }
            let Some(line) = lines.next() else { break };
            let line = line?;
            let text = line.trim();
            if text.is_empty() {
                continue;
            }
            if text == "/quit" {
                break;
            }
            if text == "/reset" {
                chat.reset(&model)?;
                if !no_cache {
                    chat.prime_watched(&model, &tokenizer, watch!())?;
                }
                continue;
            }
            if let Some(path) = text.strip_prefix("/save ") {
                match save_chat(&model, &chat, std::path::Path::new(path.trim()), limit) {
                    Ok(()) => eprintln!("saved: {}", path.trim()),
                    Err(e) => eprintln!("error: {e}"),
                }
                continue;
            }
            if let Some(path) = text.strip_prefix("/load ") {
                match load_chat(&model, &tokenizer, std::path::Path::new(path.trim()), limit) {
                    Ok(loaded) => {
                        next_pass!();
                        chat = loaded;
                    }
                    Err(e) => eprintln!("error: {e}"),
                }
                continue;
            }
            if let Some(arg) = text.strip_prefix("/edit") {
                let arg = arg.trim();
                let result = match arg {
                    "" => {
                        let on = chat.engine().edit().is_some();
                        eprintln!("edit: {}", if on { "on" } else { "off" });
                        continue;
                    }
                    "off" => Ok(None),
                    path => read_edit(path).map(Some),
                };
                next_pass!();
                match result.and_then(|e| Ok(chat.set_edit_watched(&model, e, watch!())?)) {
                    Ok(()) => eprintln!(
                        "[edit {}; history recomputed]",
                        if arg == "off" { "removed" } else { "applied" }
                    ),
                    Err(e) => eprintln!("error: {e}"),
                }
                continue;
            }
            if let Some(arg) = text.strip_prefix("/trace") {
                let position = chat.engine().state().position;
                match (trace.as_mut(), arg.trim()) {
                    (None, _) => eprintln!("error: start with --trace PATH to record traces"),
                    (Some(t), "") => eprintln!("position={position} observed={:?}", t.positions),
                    (Some(t), arg) => {
                        let p = match arg.strip_prefix('+') {
                            Some(k) => k.parse::<usize>().map(|k| position + k),
                            None => arg.parse::<usize>(),
                        };
                        match p {
                            Ok(p) => {
                                t.positions.insert(p);
                                eprintln!("[will trace position {p}]");
                            }
                            Err(e) => eprintln!("error: {e}"),
                        }
                    }
                }
                continue;
            }
            let start = std::time::Instant::now();
            let result = if text == "/continue" {
                chat.continue_turn_watched(&model, &tokenizer, max, watch!())
            } else if text.starts_with('/') {
                eprintln!(
                    "commands: /continue /reset /save PATH /load PATH /edit [PATH|off] /trace [N|+K] /quit"
                );
                continue;
            } else {
                next_pass!();
                chat.user_turn_watched(&model, &tokenizer, &line, max, !no_cache, watch!())
            };
            match result {
                Ok(turn) => display(&turn, start.elapsed().as_secs_f64() * 1000.)?,
                Err(e) => eprintln!("error: {e}"),
            }
        }
    }
    if let Some(path) = save {
        save_chat(&model, &chat, std::path::Path::new(&path), limit)?;
    }
    Ok(())
}

/// Greedy acceptance of the MTP head's drafts (spec/mtp-acceptance.md).
/// Each prompt is generated greedily by the main model. At every generated
/// position i the head drafts `depth` tokens from (h_i, s_{i+1}), chaining
/// its own output; draft j is accepted if drafts 1..=j all equal the greedy
/// tokens s_{i+2..=i+1+j}. That is exactly what greedy speculative decoding
/// would accept, so the generated text itself is never changed.
fn run_mtp_accept(dir: &std::path::Path, mut o: Options) -> Result<(), Box<dyn Error>> {
    use qwen_rust_inference::{generate::select, mtp::MtpState};
    let file = o.take("--prompt-file").ok_or("--prompt-file is required")?;
    let max = o.number("--max-new-tokens", 128)?;
    let depth = o.number("--depth", 3)?.max(1);
    let thinking = o.take("--thinking").is_some();
    o.finish()?;
    let prompts: Vec<String> = fs::read_to_string(file)?
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_owned)
        .collect();
    let tokenizer = TextTokenizer::load(dir)?;
    let model = load_model(dir)?;
    let mtp = load_mtp(dir)?;
    let eos = [248044u32, 248046];
    let backend = Backend::inference();
    let greedy = |l: &[f32]| select(l, Sampling::Greedy, &mut 0);
    let (mut main_s, mut mtp_s) = (0f64, 0f64);
    let (mut main_n, mut mtp_n) = (0usize, 0usize);
    let mut total_accept = vec![0usize; depth];
    let mut total_avail = vec![0usize; depth];
    let (mut prompt_hit, mut prompt_n) = (0usize, 0usize);
    let mut runs = Vec::new();
    for text in &prompts {
        let ids = tokenizer.encode(&user_chat(text, thinking))?;
        let n = ids.len();
        let mut seq = ids.clone();
        let mut state = model.new_state();
        let mut cache = MtpState::default();
        let mut drafts: Vec<(usize, Vec<u32>)> = Vec::new();
        let mut i = 0;
        while i < seq.len() {
            if i >= n && eos.contains(&seq[i]) {
                break;
            }
            let t = std::time::Instant::now();
            let (h, logits) = model.step_hidden(seq[i], &mut state, backend, None)?;
            main_s += t.elapsed().as_secs_f64();
            main_n += 1;
            if i + 1 == seq.len() {
                if seq.len() - n >= max {
                    break;
                }
                seq.push(greedy(&logits)?);
            }
            // The true pair (h_i, s_{i+1}) stays in the head's KV cache.
            let t = std::time::Instant::now();
            let first = mtp.step(&model, &h, seq[i + 1], &mut cache, backend)?;
            mtp_s += t.elapsed().as_secs_f64();
            mtp_n += 1;
            let mut d = vec![greedy(&first.logits)?];
            if i + 1 >= n {
                // Chain the head on its own output, then drop those positions.
                let keep = cache.len();
                let mut g = first.hidden;
                for _ in 1..depth {
                    let t = std::time::Instant::now();
                    let next = mtp.step(&model, &g, *d.last().unwrap(), &mut cache, backend)?;
                    mtp_s += t.elapsed().as_secs_f64();
                    mtp_n += 1;
                    d.push(greedy(&next.logits)?);
                    g = next.hidden;
                }
                cache.truncate(keep);
            }
            drafts.push((i, d));
            i += 1;
        }
        let mut accept = vec![0usize; depth];
        let mut avail = vec![0usize; depth];
        let (mut hit, mut seen) = (0usize, 0usize);
        for (i, d) in &drafts {
            if i + 1 < n {
                // Prompt text (teacher forcing): does the head predict s_{i+2}?
                if i + 2 < n {
                    seen += 1;
                    hit += usize::from(d[0] == seq[i + 2]);
                }
                continue;
            }
            let mut all = true;
            for (j, &dj) in d.iter().enumerate() {
                let target = i + 2 + j;
                if target >= seq.len() {
                    break;
                }
                avail[j] += 1;
                all &= dj == seq[target];
                accept[j] += usize::from(all);
            }
        }
        for j in 0..depth {
            total_accept[j] += accept[j];
            total_avail[j] += avail[j];
        }
        prompt_hit += hit;
        prompt_n += seen;
        let rates: Vec<f64> = (0..depth)
            .map(|j| accept[j] as f64 / avail[j].max(1) as f64)
            .collect();
        runs.push(serde_json::json!({
            "prompt": text,
            "prompt_tokens": n,
            "generated_tokens": seq.len() - n,
            "output": tokenizer.decode(&seq[n..])?,
            "accepted": accept,
            "available": avail,
            "prefix_acceptance": rates,
            "prompt_next_next_accuracy": hit as f64 / seen.max(1) as f64,
        }));
    }
    let rates: Vec<f64> = (0..depth)
        .map(|j| total_accept[j] as f64 / total_avail[j].max(1) as f64)
        .collect();
    // With k drafts, one verify pass yields 1 + sum_{j<=k} P(drafts 1..j accepted) tokens.
    let expected: Vec<f64> = (1..=depth)
        .map(|k| 1. + rates[..k].iter().sum::<f64>())
        .collect();
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "depth": depth,
            "max_new_tokens": max,
            "thinking": thinking,
            "prefix_acceptance": rates,
            "conditional_acceptance": (0..depth).map(|j| if j == 0 { rates[0] } else { rates[j] / rates[j - 1].max(f64::MIN_POSITIVE) }).collect::<Vec<_>>(),
            "tokens_per_verify_pass": expected,
            "accepted": total_accept,
            "available": total_avail,
            "prompt_next_next_accuracy": prompt_hit as f64 / prompt_n.max(1) as f64,
            "ms_per_main_step": 1e3 * main_s / main_n.max(1) as f64,
            "ms_per_mtp_step": 1e3 * mtp_s / mtp_n.max(1) as f64,
            "runs": runs,
        }))?
    );
    Ok(())
}
