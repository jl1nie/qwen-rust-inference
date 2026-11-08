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
const HELP:&str="qwen-rust-inference\n  demo ffn [--unit N] [--trace PATH --trace-full]\n  demo attention [--cache]\n  demo deltanet\n  demo lesson --trace PATH\n  demo hybrid [--max-new-tokens N]\n  inspect-config --model-dir DIR\n  infer --model-dir DIR (--ids 16,10,16,28 | --prompt TEXT) [--raw] [--thinking]\n        [--max-new-tokens 16] [--seed N --temperature 0.8 --top-k 40]\n        [--edit manifest.json] [--compare-edit]\n        [--trace PATH --trace-layer N --trace-position N --trace-max-bytes N --trace-full]\nNo automatic downloads. Text-only, one user turn; CPU single-batch.";
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
                "--cache" | "--raw" | "--thinking" | "--trace-full" | "--compare-edit"
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
                let MixerWeights::Attention(w) = &m.blocks[3].mixer else {
                    unreachable!()
                };
                let xs = (0..3)
                    .map(|i| m.embedding.row(i).map(<[f32]>::to_vec))
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
                let MixerWeights::Delta(w) = &m.blocks[0].mixer else {
                    unreachable!()
                };
                let mut s = w.new_state();
                for p in 0..3 {
                    let a = w.step(m.embedding.row(p)?, &mut s, Backend::Reference)?;
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
    if args[0] == "inspect-config" {
        o.finish()?;
        println!("{}", serde_json::to_string_pretty(&load_config(&dir)?)?);
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
    if trace_layer.is_some_and(|l| l >= model.blocks.len()) {
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
    let result = generate_observed(
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
    )?;
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
