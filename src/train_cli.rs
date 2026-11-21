//! Training commands for book 3 (`--features train`).
//!
//! - `train-capture`: the last layer's FFN inputs over a text, saved once.
//! - `train-fact`: teach one question its answer, by training the last FFN.
//! - `train-eval`: compare the trained FFN with the original on a cache
//!   (perplexity, KL, top-1 agreement).
use crate::cli::Options;
use qwen_rust_inference::{
    tensor::{Backend, InferenceError},
    text::TextTokenizer,
    train::{
        cache::LayerCache,
        checkpoint::FfnCheckpoint,
        ffn::TopFfn,
        fit::{compare_models, fit, Fact},
        optim::{Optimizer, OptimizerConfig},
    },
    weights::load_model,
};
use std::{error::Error, fs, io::Write, path::PathBuf};

const HELP: &str = "book 3 training (CPU; the last layer's FFN only)
  train-capture --model-dir DIR --text-file PATH --out CACHE [--max-tokens 512]
      Reads the text (raw, no chat template) and saves the last layer's FFN inputs.
  train-fact --model-dir DIR --prompt TEXT --answer TEXT --out CHECKPOINT
             [--steps 30] [--optimizer adam|sgd] [--lr 1e-4] [--batch 16]
             [--resume CHECKPOINT] [--log PATH.ndjson]
      Teaches user_chat(PROMPT) to answer ANSWER, then the end of the turn.
  train-eval --model-dir DIR --checkpoint CHECKPOINT --text-cache CACHE [--batch 16]
      Original vs trained FFN on the cached text: perplexity, KL, top-1 agreement (JSON).
Then: infer --model-dir DIR --prompt TEXT --ffn-checkpoint CHECKPOINT";

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
        other => Err(format!("unknown command {other}\n{HELP}").into()),
    }
}

fn capture(dir: &std::path::Path, mut o: Options) -> Result<(), Box<dyn Error>> {
    let text = fs::read_to_string(o.take("--text-file").ok_or("--text-file is required")?)?;
    let out = PathBuf::from(o.take("--out").ok_or("--out is required")?);
    let max = o.number("--max-tokens", 512)?;
    o.finish()?;
    let tokenizer = TextTokenizer::load(dir)?;
    let mut ids = tokenizer.encode(&text)?;
    ids.truncate(max);
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

fn teach(dir: &std::path::Path, mut o: Options) -> Result<(), Box<dyn Error>> {
    let prompt = o.take("--prompt").ok_or("--prompt is required")?;
    let answer = o.take("--answer").ok_or("--answer is required")?;
    let out = PathBuf::from(o.take("--out").ok_or("--out is required")?);
    let steps = o.number("--steps", 30)?;
    let batch = o.number("--batch", 16)?;
    let resume = o.take("--resume");
    let config = optimizer_config(&mut o)?;
    let log_path = o.take("--log");
    o.finish()?;

    let tokenizer = TextTokenizer::load(dir)?;
    let model = load_model(dir)?;
    let fact = Fact::new(&tokenizer, &prompt, &answer, END_OF_TURN)?;
    let cache = LayerCache::capture(&model, &fact.ids, Backend::inference())?;
    let mut top = TopFfn::new(&model)?;
    let mut optimizer = Optimizer::new(config, &top.ffn.sizes());
    if let Some(path) = resume {
        let c = FfnCheckpoint::read(path.as_ref())?;
        if c.base_identity != model.identity() || c.layer != top.layer {
            return Err("--resume checkpoint was trained from a different model".into());
        }
        top.ffn = c.ffn;
        optimizer = c
            .optimizer
            .ok_or("--resume checkpoint has no optimizer state")?;
    }
    let examples = fact.examples();
    let before = fact.report(&top, &cache.inputs)?;
    let mut log = log_path.map(fs::File::create).transpose()?;
    let start = std::time::Instant::now();
    fit(
        &mut top,
        &cache.inputs,
        &examples,
        &mut optimizer,
        steps,
        batch,
        |s| {
            eprintln!(
                "step {} loss {:.5} grad_norm {:.4e}",
                s.step, s.loss, s.grad_norm
            );
            if let Some(f) = &mut log {
                let err = |e: &dyn std::fmt::Display| InferenceError(format!("training log: {e}"));
                let line = serde_json::to_string(s).map_err(|e| err(&e))?;
                writeln!(f, "{line}").map_err(|e| err(&e))?;
            }
            Ok(())
        },
    )?;
    let seconds = start.elapsed().as_secs_f64();
    let after = fact.report(&top, &cache.inputs)?;
    FfnCheckpoint {
        base_identity: model.identity().into(),
        layer: top.layer,
        ffn: top.ffn.clone(),
        optimizer: Some(optimizer),
    }
    .write(&out)?;
    let answer_ids: Vec<u32> = before.iter().map(|r| r.id).collect();
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "prompt": prompt,
            "answer": answer,
            "answer_tokens": answer_ids.iter().map(|&id| tokenizer.decode(&[id])).collect::<Result<Vec<_>, _>>()?,
            "steps": steps,
            "optimizer": config,
            "seconds": seconds,
            "before": before,
            "after": after,
            "checkpoint": out,
        }))?
    );
    Ok(())
}

fn evaluate(dir: &std::path::Path, mut o: Options) -> Result<(), Box<dyn Error>> {
    let checkpoint = o.take("--checkpoint").ok_or("--checkpoint is required")?;
    let cache = o.take("--text-cache").ok_or("--text-cache is required")?;
    let batch = o.number("--batch", 16)?;
    o.finish()?;
    let model = load_model(dir)?;
    let c = FfnCheckpoint::read(checkpoint.as_ref())?;
    // Refuses a checkpoint from another model before computing anything.
    c.apply(&model)?;
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
