//! Training commands for book 3 (`--features train`).
//!
//! - `train-capture`: the last layer's FFN inputs over a text, saved once.
//! - `train-fact`: teach one question its answer, by training the last FFN.
//! - `train-eval`: compare the trained FFN with the original on a cache
//!   (perplexity, KL, top-1 agreement).
//! - `train-teacher`: Strata's top-K output → teacher data.
//! - `train-distill`: train toward the teacher data (KL).
//! - `train-delta`: what training changed in the weights.
//! - `train-text`: learn prompt/response pairs (or a text) by cross-entropy.
//! - `train-probe`: linear probes on each layer's vectors (no training of the model).
use crate::cli::Options;
use qwen_rust_inference::{
    model::Model,
    tensor::{Backend, InferenceError},
    text::TextTokenizer,
    train::{
        cache::LayerCache,
        checkpoint::FfnCheckpoint,
        corpus::{parse_pairs, score, Corpus},
        ffn::TopFfn,
        ffn::TopInputs,
        fit::{compare_models, fit, mean_loss_and_grad, Example, Fact},
        lora::PROJECTIONS,
        optim::{Optimizer, OptimizerConfig},
        teacher::{parse_logpos, TeacherData, COMMON_VOCAB},
    },
    weights::load_model,
};
use std::{error::Error, fs, io::Write, path::PathBuf};

const HELP: &str = "book 3 training (CPU; the last layer's FFN only)
  train-capture --model-dir DIR --text-file PATH --out CACHE [--max-tokens 512] [--ids-out IDS.json]
      Reads the text (raw, no chat template) and saves the last layer's FFN inputs.
  train-fact --model-dir DIR --prompt TEXT --answer TEXT --out CHECKPOINT
             [--steps 30] [--optimizer adam|sgd] [--lr 1e-4] [--batch 16]
             [--resume CHECKPOINT] [--log PATH.ndjson]
             [--lora RANK [--lora-alpha RANK] [--lora-targets gate,up,down] [--lora-seed 0]]
      Teaches user_chat(PROMPT) to answer ANSWER, then the end of the turn.
      With --lora, the FFN is frozen and only rank-RANK adapters are trained
      (scale alpha/rank); the checkpoint then holds only the adapters.
  train-eval --model-dir DIR --checkpoint CHECKPOINT --text-cache CACHE [--batch 16]
      Original vs trained FFN on the cached text: perplexity, KL, top-1 agreement (JSON).
  train-teacher --model-dir DIR --logpos STRATA_LOGPOS.tsv (--text-file PATH [--max-tokens 512] | --ids-file IDS.json)
                --teacher NAME --out TEACHER.json
      Strata's teacher-forced top-K (STRATA_LOGPOS + STRATA_LOGPOS_TOPK) for the same IDs,
      renormalized over the 248,070 IDs both models share.
  train-distill --model-dir DIR --teacher TEACHER.json --out CHECKPOINT [--text-cache CACHE]
                [the training options of train-fact]
      Trains the last FFN (or adapters) toward the teacher: KL over the top K + other.
  train-text --model-dir DIR (--pairs PAIRS.jsonl [--holdout 0] | --text-file PATH [--max-tokens 512])
             --out CHECKPOINT [the training options of train-fact]
      Learns text by cross-entropy (distillation from the teacher's generated text).
      PAIRS.jsonl: one {\"prompt\": ..., \"response\": ...} per line; each is user_chat(prompt) +
      response + <|im_end|>, and the response is scored; with \"score\": PART, only PART's
      tokens (it must occur once and fall on token boundaries); with \"teacher\": the teacher's
      top K at those positions (tools/teacher_topk.py), learned by KL instead of the one ID.
      Loss and top-1 are always of the text's own next IDs. The last --holdout pairs are not
      trained on, only scored before and after. --text-file: every position of a raw text.
  train-delta --model-dir DIR --checkpoint CHECKPOINT [--edit manifest.json] [--rank 16]
              [--directions 3] [--ids 8]
      What training changed in the FFN (JSON): size and singular values of ΔW per matrix,
      the units holding the change, the main directions of ΔW_down read through the LM head,
      and, with --edit on the same layer, the hand edit as a ΔW and its cosine with training's.
  train-probe --model-dir DIR --items A.jsonl,B.jsonl,... [--steps 200] [--lr 0.01] [--l2 1e-3]
      Linear probes (JSON): reads each item's last position (lens-batch items: user_chat(prompt) +
      prefix) at the embedding, every layer's exit and the last FFN's input, and asks a logistic
      regression for the item's id. Each file is one wording; each is scored by a probe trained
      on the other files only (leave one wording out). The model is not changed.
Then: infer --model-dir DIR --prompt TEXT --ffn-checkpoint CHECKPOINT [--compare-edit]
      lens --model-dir DIR --prompt TEXT --position N --ffn-checkpoint CHECKPOINT";

/// End of an assistant turn (`<|im_end|>`).
const END_OF_TURN: u32 = 248046;

pub fn run(args: &[String]) -> Result<(), Box<dyn Error>> {
    if args.get(1).is_some_and(|a| a == "--help" || a == "-h") {
        println!("{HELP}");
        return Ok(());
    }
    let mut o = Options::parse(&args[1..])?;
    let dir = PathBuf::from(o.take("--model-dir").ok_or("--model-dir is required")?);
    match args[0].as_str() {
        "train-capture" => capture(&dir, o),
        "train-fact" => teach(&dir, o),
        "train-eval" => evaluate(&dir, o),
        "train-teacher" => import_teacher(&dir, o),
        "train-distill" => distill(&dir, o),
        "train-delta" => delta(&dir, o),
        "train-text" => train_text(&dir, o),
        "train-probe" => probe(&dir, o),
        other => Err(format!("unknown command {other}\n{HELP}").into()),
    }
}

fn capture(dir: &std::path::Path, mut o: Options) -> Result<(), Box<dyn Error>> {
    let text = fs::read_to_string(o.take("--text-file").ok_or("--text-file is required")?)?;
    let out = PathBuf::from(o.take("--out").ok_or("--out is required")?);
    let max = o.number("--max-tokens", 512)?;
    let ids_out = o.take("--ids-out");
    o.finish()?;
    let tokenizer = TextTokenizer::load(dir)?;
    let mut ids = tokenizer.encode(&text)?;
    ids.truncate(max);
    if let Some(p) = ids_out {
        // The same IDs, for the teacher to read (train-teacher --ids-file).
        fs::write(p, serde_json::to_vec(&ids)?)?;
    }
    let model = load_model(dir)?;
    let start = std::time::Instant::now();
    let cache = LayerCache::capture(&model, &ids, Backend::inference())?;
    cache.write(&out)?;
    eprintln!(
        "captured {} positions of layer {} in {:.1} s -> {}",
        ids.len(),
        cache.header.layer,
        start.elapsed().as_secs_f64(),
        out.display()
    );
    Ok(())
}

fn optimizer_config(o: &mut Options) -> Result<OptimizerConfig, Box<dyn Error>> {
    let lr: f32 = o
        .take("--lr")
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(1e-4);
    Ok(match o.take("--optimizer").as_deref().unwrap_or("adam") {
        "adam" => OptimizerConfig::adam(lr),
        "sgd" => OptimizerConfig::Sgd { lr },
        other => return Err(format!("unknown optimizer {other}").into()),
    })
}

/// The training options shared by `train-fact` and `train-distill`.
struct Training {
    out: PathBuf,
    steps: usize,
    batch: usize,
    resume: Option<String>,
    config: OptimizerConfig,
    log_path: Option<String>,
    lora: Option<(usize, f32, Vec<usize>, u64)>,
}
impl Training {
    fn parse(o: &mut Options) -> Result<Self, Box<dyn Error>> {
        let out = PathBuf::from(o.take("--out").ok_or("--out is required")?);
        let steps = o.number("--steps", 30)?;
        let batch = o.number("--batch", 16)?;
        let resume = o.take("--resume");
        let config = optimizer_config(o)?;
        let log_path = o.take("--log");
        let rank = o.take("--lora").map(|r| r.parse::<usize>()).transpose()?;
        let alpha: Option<f32> = o.take("--lora-alpha").map(|v| v.parse()).transpose()?;
        let targets = o.take("--lora-targets");
        let seed = o.number("--lora-seed", 0)? as u64;
        if rank.is_none() && (alpha.is_some() || targets.is_some()) {
            return Err("--lora-alpha/--lora-targets require --lora".into());
        }
        if rank.is_some() && resume.is_some() {
            return Err("--resume takes the adapters from the checkpoint; drop --lora".into());
        }
        let targets: Vec<usize> = targets
            .as_deref()
            .unwrap_or("gate,up,down")
            .split(',')
            .map(|t| {
                PROJECTIONS
                    .iter()
                    .position(|p| *p == t.trim())
                    .ok_or_else(|| format!("unknown LoRA target {t}"))
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            out,
            steps,
            batch,
            resume,
            config,
            log_path,
            lora: rank.map(|r| (r, alpha.unwrap_or(r as f32), targets, seed)),
        })
    }
    /// The trainable top layer (with adapters, or resumed) and its optimizer.
    fn prepare<'a>(&self, model: &'a Model) -> Result<(TopFfn<'a>, Optimizer), Box<dyn Error>> {
        let mut top = TopFfn::new(model)?;
        if let Some((rank, alpha, targets, seed)) = &self.lora {
            top.ffn = top.ffn.with_lora(*rank, *alpha, targets, *seed)?;
        }
        let mut optimizer = Optimizer::new(self.config, &top.ffn.sizes());
        if let Some(path) = &self.resume {
            // Refuses a checkpoint from another model.
            let c = FfnCheckpoint::read(path.as_ref(), model)?;
            if c.layer != top.layer {
                return Err("--resume checkpoint is for another layer".into());
            }
            top.ffn = c.ffn;
            optimizer = c
                .optimizer
                .ok_or("--resume checkpoint has no optimizer state")?;
        }
        Ok((top, optimizer))
    }
    /// Trains, saves the checkpoint, and returns the seconds spent training
    /// and a JSON summary of the setup.
    fn run(
        &self,
        model: &Model,
        top: &mut TopFfn,
        optimizer: Optimizer,
        inputs: &TopInputs,
        examples: &[Example],
    ) -> Result<serde_json::Value, Box<dyn Error>> {
        let mut optimizer = optimizer;
        let mut log = self.log_path.as_ref().map(fs::File::create).transpose()?;
        let start = std::time::Instant::now();
        fit(
            top,
            inputs,
            examples,
            &mut optimizer,
            self.steps,
            self.batch,
            |s| {
                eprintln!(
                    "step {} loss {:.5} grad_norm {:.4e}",
                    s.step, s.loss, s.grad_norm
                );
                if let Some(f) = &mut log {
                    let err =
                        |e: &dyn std::fmt::Display| InferenceError(format!("training log: {e}"));
                    let line = serde_json::to_string(s).map_err(|e| err(&e))?;
                    writeln!(f, "{line}").map_err(|e| err(&e))?;
                }
                Ok(())
            },
        )?;
        let seconds = start.elapsed().as_secs_f64();
        // The resumed optimizer's settings, which may differ from the flags.
        let used = optimizer.config;
        FfnCheckpoint {
            base_identity: model.identity().into(),
            layer: top.layer,
            ffn: top.ffn.clone(),
            optimizer: Some(optimizer),
        }
        .write(&self.out)?;
        Ok(serde_json::json!({
            "steps": self.steps,
            "optimizer": used,
            "lora": top.ffn.lora.as_ref().map(|l| serde_json::json!({
                "rank": l.rank,
                "alpha": l.alpha,
                "targets": PROJECTIONS.iter().zip(&l.adapters).filter(|(_, a)| a.is_some()).map(|(p, _)| *p).collect::<Vec<_>>(),
            })),
            "trained_parameters": top.ffn.trained_parameters(),
            "seconds": seconds,
            "checkpoint": self.out,
        }))
    }
}

fn teach(dir: &std::path::Path, mut o: Options) -> Result<(), Box<dyn Error>> {
    let prompt = o.take("--prompt").ok_or("--prompt is required")?;
    let answer = o.take("--answer").ok_or("--answer is required")?;
    let training = Training::parse(&mut o)?;
    o.finish()?;

    let tokenizer = TextTokenizer::load(dir)?;
    let model = load_model(dir)?;
    let fact = Fact::new(&tokenizer, &prompt, &answer, END_OF_TURN)?;
    let cache = LayerCache::capture(&model, &fact.ids, Backend::inference())?;
    let (mut top, optimizer) = training.prepare(&model)?;
    let before = fact.report(&top, &cache.inputs)?;
    let mut summary = training.run(&model, &mut top, optimizer, &cache.inputs, &fact.examples())?;
    let after = fact.report(&top, &cache.inputs)?;
    let answer_ids: Vec<u32> = before.iter().map(|r| r.id).collect();
    let s = summary.as_object_mut().unwrap();
    s.insert("prompt".into(), prompt.into());
    s.insert("answer".into(), answer.into());
    s.insert(
        "answer_tokens".into(),
        serde_json::to_value(
            answer_ids
                .iter()
                .map(|&id| tokenizer.decode(&[id]))
                .collect::<Result<Vec<_>, _>>()?,
        )?,
    );
    s.insert("before".into(), serde_json::to_value(before)?);
    s.insert("after".into(), serde_json::to_value(after)?);
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

/// `STRATA_LOGPOS` output → teacher data for the IDs of a text.
fn import_teacher(dir: &std::path::Path, mut o: Options) -> Result<(), Box<dyn Error>> {
    let logpos = fs::read_to_string(o.take("--logpos").ok_or("--logpos is required")?)?;
    let ids = read_ids(dir, &mut o)?;
    let teacher = o
        .take("--teacher")
        .ok_or("--teacher NAME is required (teacher, quantization, engine version)")?;
    let out = PathBuf::from(o.take("--out").ok_or("--out is required")?);
    o.finish()?;
    let lines = parse_logpos(&logpos)?;
    let data = TeacherData::from_logpos(&lines, &ids, COMMON_VOCAB, &teacher)?;
    data.write(&out)?;
    let k: Vec<usize> = data.positions.iter().map(|p| p.ids.len()).collect();
    eprintln!(
        "{} teacher positions over {} IDs (entries per position {}..{}) -> {}",
        data.positions.len(),
        ids.len(),
        k.iter().min().unwrap(),
        k.iter().max().unwrap(),
        out.display()
    );
    Ok(())
}

/// The IDs of `--text-file` (raw, cut to `--max-tokens`), or of `--ids-file`
/// (a JSON array).
fn read_ids(dir: &std::path::Path, o: &mut Options) -> Result<Vec<u32>, Box<dyn Error>> {
    match (o.take("--text-file"), o.take("--ids-file")) {
        (Some(path), None) => {
            let max = o.number("--max-tokens", 512)?;
            let mut ids = TextTokenizer::load(dir)?.encode(&fs::read_to_string(path)?)?;
            ids.truncate(max);
            Ok(ids)
        }
        (None, Some(path)) => Ok(serde_json::from_slice(&fs::read(path)?)?),
        _ => Err("give exactly one of --text-file or --ids-file".into()),
    }
}

fn distill(dir: &std::path::Path, mut o: Options) -> Result<(), Box<dyn Error>> {
    let teacher = TeacherData::read(o.take("--teacher").ok_or("--teacher is required")?.as_ref())?;
    let cache_path = o.take("--text-cache");
    let training = Training::parse(&mut o)?;
    o.finish()?;
    let model = load_model(dir)?;
    let cache = match cache_path {
        Some(p) => LayerCache::read(p.as_ref(), &model)?,
        None => LayerCache::capture(&model, &teacher.ids, Backend::inference())?,
    };
    teacher.check_ids(&cache.header.ids)?;
    let examples = teacher.examples()?;
    let (mut top, optimizer) = training.prepare(&model)?;
    let kl = |top: &TopFfn| {
        mean_loss_and_grad(top, &cache.inputs, &examples, training.batch).map(|(l, _)| l)
    };
    let before = kl(&top)?;
    let mut summary = training.run(&model, &mut top, optimizer, &cache.inputs, &examples)?;
    let after = kl(&top)?;
    let s = summary.as_object_mut().unwrap();
    s.insert("teacher".into(), teacher.teacher.clone().into());
    s.insert("positions".into(), examples.len().into());
    s.insert("kl_to_teacher_before".into(), before.into());
    s.insert("kl_to_teacher_after".into(), after.into());
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn evaluate(dir: &std::path::Path, mut o: Options) -> Result<(), Box<dyn Error>> {
    let checkpoint = o.take("--checkpoint").ok_or("--checkpoint is required")?;
    let cache = o.take("--text-cache").ok_or("--text-cache is required")?;
    let batch = o.number("--batch", 16)?;
    o.finish()?;
    let model = load_model(dir)?;
    // Refuses a checkpoint from another model before computing anything.
    let c = FfnCheckpoint::read(checkpoint.as_ref(), &model)?;
    let cache = LayerCache::read(cache.as_ref(), &model)?;
    let reference = TopFfn::new(&model)?;
    let mut trained = TopFfn::new(&model)?;
    trained.ffn = c.ffn;
    let comparison = compare_models(
        &reference,
        &trained,
        &cache.inputs,
        &cache.header.ids,
        batch,
    )?;
    println!("{}", serde_json::to_string_pretty(&comparison)?);
    Ok(())
}

fn delta(dir: &std::path::Path, mut o: Options) -> Result<(), Box<dyn Error>> {
    let checkpoint = o.take("--checkpoint").ok_or("--checkpoint is required")?;
    let edit: Option<qwen_rust_inference::edit::EditManifest> = o
        .take("--edit")
        .map(|p| fs::read(p).map(|v| serde_json::from_slice(&v)))
        .transpose()?
        .transpose()?;
    let rank = o.number("--rank", 16)?;
    let directions = o.number("--directions", 3)?;
    let ids = o.number("--ids", 8)?;
    o.finish()?;
    let tokenizer = TextTokenizer::load(dir)?;
    let model = load_model(dir)?;
    let c = FfnCheckpoint::read(checkpoint.as_ref(), &model)?;
    let start = std::time::Instant::now();
    let report = qwen_rust_inference::train::delta::analyze(
        &model,
        c.layer,
        &c.ffn,
        edit.as_ref(),
        rank,
        (directions, ids),
    )?;
    eprintln!("analyzed in {:.1} s", start.elapsed().as_secs_f64());
    let mut value = serde_json::to_value(&report)?;
    // The IDs of each direction as text too.
    for d in value["directions"].as_array_mut().unwrap() {
        for end in ["plus", "minus"] {
            let tokens: Vec<String> = d[end]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| tokenizer.decode(&[p[0].as_u64().unwrap() as u32]))
                .collect::<Result<_, _>>()?;
            d[format!("{end}_tokens")] = serde_json::to_value(tokens)?;
        }
    }
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(())
}

fn train_text(dir: &std::path::Path, mut o: Options) -> Result<(), Box<dyn Error>> {
    let pairs_path = o.take("--pairs");
    let text_path = o.take("--text-file");
    let holdout = o.number("--holdout", 0)?;
    let max = o.number("--max-tokens", 512)?;
    let training = Training::parse(&mut o)?;
    o.finish()?;
    let tokenizer = TextTokenizer::load(dir)?;
    let model = load_model(dir)?;
    let backend = Backend::inference();
    let start = std::time::Instant::now();
    let (train, held, source) = match (pairs_path, text_path) {
        (Some(p), None) => {
            let pairs = parse_pairs(&fs::read_to_string(&p)?)?;
            if holdout >= pairs.len() {
                return Err(format!(
                    "--holdout {holdout} leaves no pair of {} to train on",
                    pairs.len()
                )
                .into());
            }
            let (a, b) = pairs.split_at(pairs.len() - holdout);
            let train = Corpus::from_pairs(&model, &tokenizer, a, END_OF_TURN, backend)?;
            let held = if b.is_empty() {
                None
            } else {
                Some(Corpus::from_pairs(
                    &model,
                    &tokenizer,
                    b,
                    END_OF_TURN,
                    backend,
                )?)
            };
            (
                train,
                held,
                serde_json::json!({"pairs": p, "trained": a.len(), "held_out": b.len()}),
            )
        }
        (None, Some(p)) => {
            if holdout > 0 {
                return Err("--holdout takes --pairs".into());
            }
            let mut ids = tokenizer.encode(&fs::read_to_string(&p)?)?;
            ids.truncate(max);
            (
                Corpus::from_text(&model, &ids, backend)?,
                None,
                serde_json::json!({"text": p}),
            )
        }
        _ => return Err("give exactly one of --pairs or --text-file".into()),
    };
    eprintln!(
        "read {} tokens, {} scored positions in {:.1} s",
        train.tokens + held.as_ref().map_or(0, |h| h.tokens),
        train.examples.len() + held.as_ref().map_or(0, |h| h.examples.len()),
        start.elapsed().as_secs_f64()
    );
    let (mut top, optimizer) = training.prepare(&model)?;
    let scores = |top: &TopFfn| -> Result<serde_json::Value, Box<dyn Error>> {
        let t = score(top, &train.inputs, &train.labels, training.batch)?;
        let h = held
            .as_ref()
            .map(|h| score(top, &h.inputs, &h.labels, training.batch))
            .transpose()?;
        Ok(serde_json::json!({"trained": t, "held_out": h}))
    };
    let before = scores(&top)?;
    let mut summary = training.run(&model, &mut top, optimizer, &train.inputs, &train.examples)?;
    let after = scores(&top)?;
    let s = summary.as_object_mut().unwrap();
    s.insert("source".into(), source);
    s.insert("scored_positions".into(), train.examples.len().into());
    s.insert("before".into(), before);
    s.insert("after".into(), after);
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn probe(dir: &std::path::Path, mut o: Options) -> Result<(), Box<dyn Error>> {
    use qwen_rust_inference::lens::LensItem;
    use qwen_rust_inference::train::probe::{
        capture_sites, held_out_groups, site_names, ProbeConfig,
    };
    let files: Vec<String> = o
        .take("--items")
        .ok_or("--items A.jsonl,B.jsonl,... is required")?
        .split(',')
        .map(String::from)
        .collect();
    let float = |o: &mut Options, key: &str, default: f64| -> Result<f64, Box<dyn Error>> {
        Ok(o.take(key)
            .map(|v| v.parse())
            .transpose()?
            .unwrap_or(default))
    };
    let d = ProbeConfig::default();
    let config = ProbeConfig {
        steps: o.number("--steps", d.steps)?,
        lr: float(&mut o, "--lr", d.lr)?,
        l2: float(&mut o, "--l2", d.l2)?,
    };
    o.finish()?;
    let tokenizer = TextTokenizer::load(dir)?;
    let model = load_model(dir)?;
    let backend = Backend::inference();
    let start = std::time::Instant::now();
    let mut ids: Vec<String> = Vec::new();
    let (mut labels, mut groups) = (Vec::new(), Vec::new());
    let mut sites: Vec<Vec<f32>> = Vec::new();
    for (g, f) in files.iter().enumerate() {
        for (n, l) in fs::read_to_string(f)?.lines().enumerate() {
            if l.trim().is_empty() {
                continue;
            }
            let item: LensItem =
                serde_json::from_str(l).map_err(|e| format!("{f} line {}: {e}", n + 1))?;
            let class = match ids.iter().position(|i| *i == item.id) {
                Some(c) => c,
                None => {
                    ids.push(item.id.clone());
                    ids.len() - 1
                }
            };
            let v = capture_sites(&model, &tokenizer, &item.prompt, &item.prefix, backend)?;
            if sites.is_empty() {
                sites = vec![Vec::new(); v.len()];
            }
            sites.iter_mut().zip(v).for_each(|(s, x)| s.extend(x));
            labels.push(class);
            groups.push(g);
        }
        eprintln!("read {f} ({:.0} s)", start.elapsed().as_secs_f64());
    }
    let dim = model.config().hidden_size;
    let names = site_names(model.blocks().len());
    let mut rows = Vec::new();
    for (name, x) in names.iter().zip(&sites) {
        let per = held_out_groups(x, dim, &labels, &groups, ids.len(), config)?;
        let (c, n) = per.iter().fold((0, 0), |(a, b), &(c, n)| (a + c, b + n));
        eprintln!("{name}: {c}/{n}");
        rows.push(serde_json::json!({
            "site": name,
            "correct": c,
            "count": n,
            "accuracy": c as f64 / n as f64,
            "per_file": per.iter().map(|&(c, n)| serde_json::json!({"correct": c, "count": n})).collect::<Vec<_>>(),
        }));
    }
    let out = serde_json::json!({
        "files": files,
        "classes": ids.len(),
        "examples": labels.len(),
        "chance": 1.0 / ids.len() as f64,
        "config": {"steps": config.steps, "lr": config.lr, "l2": config.l2, "optimizer": "adam", "standardized": true},
        "seconds": start.elapsed().as_secs_f64(),
        "sites": rows,
    });
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}
