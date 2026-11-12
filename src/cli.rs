use qwen_rust_inference::{
    block::MixerWeights,
    edit::{repetition, EditManifest},
    generate::{generate, generate_observed, Sampling},
    model::Model,
    state::KvCache,
    tensor::Backend,
    text::{user_chat, TextTokenizer},
    trace::Trace,
    weights::{load_config, load_model},
};
use std::{
    collections::BTreeMap,
    error::Error,
    fs::{self, OpenOptions},
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};
const HELP:&str="qwen-rust-inference\n  demo ffn [--unit N] [--trace PATH --trace-full]\n  demo attention [--cache]\n  demo deltanet\n  demo lesson --trace PATH\n  demo hybrid [--max-new-tokens N]\n  chat --model-dir DIR [--prompt TEXT | --continue] [--load-state PATH] [--save-state PATH]\n       [--max-new-tokens 64] [--stats] [--json] [--no-cache]\n  bench --model-dir DIR [--backend scalar|gemm|vector] [--capture full|compact]\n        [--ids 16,10,16,28] [--decode-steps 8] [--warmup 1] [--runs 3]\n  inspect-config --model-dir DIR\n  infer --model-dir DIR (--ids 16,10,16,28 | --prompt TEXT) [--raw] [--thinking]\n        [--max-new-tokens 16] [--seed N --temperature 0.8 --top-k 40]\n        [--edit manifest.json] [--compare-edit]\n        [--trace PATH --trace-layer N --trace-position N --trace-max-bytes N --trace-full]\n  rocm-check / rocm-infer --help (Linux/WSL2, requires --features rocm)\n  metal-check / metal-infer --help (macOS, requires --features metal)\nNo automatic downloads. Text-only; CPU single-batch by default.";
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
                    .map(|i| m.embedding().row(i).map(<[f32]>::to_vec))
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
                    let a = w.step(m.embedding().row(p)?, &mut s, Backend::Reference)?;
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
    let system = o.take("--system");
    let no_cache = o.take("--no-cache").is_some();
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
        ChatSession::new(&model, b, sampling, thinking, system)?
    };
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
            chat.user_turn_with_cache(&model, &tokenizer, &prompt, max, !no_cache)?
        } else {
            chat.continue_turn(&model, &tokenizer, max)?
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
                    Ok(loaded) => chat = loaded,
                    Err(e) => eprintln!("error: {e}"),
                }
                continue;
            }
            let start = std::time::Instant::now();
            let result = if text == "/continue" {
                chat.continue_turn(&model, &tokenizer, max)
            } else if text.starts_with('/') {
                eprintln!("commands: /continue /reset /save PATH /load PATH /quit");
                continue;
            } else {
                chat.user_turn_with_cache(&model, &tokenizer, &line, max, !no_cache)
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
