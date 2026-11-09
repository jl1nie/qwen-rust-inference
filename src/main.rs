use qwen_rust_inference::{
    ffn::educational_demo,
    trace::{Trace, TraceContext},
};
use std::{
    error::Error,
    fs::OpenOptions,
    time::{SystemTime, UNIX_EPOCH},
};
const USAGE:&str="Usage: qwen-rust-inference demo ffn [--unit N] [--trace PATH [--trace-full] [--trace-max-bytes N]]\nSee --help for the other demos and real-model inference.";
fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
fn demo_ffn() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.as_slice() == ["--help"] || args.as_slice() == ["-h"] {
        println!("{USAGE}");
        return Ok(());
    }
    if args.first().map(String::as_str) != Some("demo")
        || args.get(1).map(String::as_str) != Some("ffn")
    {
        return Err(USAGE.into());
    }
    let mut unit = 3;
    let mut path = None;
    let mut full = false;
    let mut cap = 65536;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--unit" | "--trace" | "--trace-max-bytes" => {
                let value = args.get(i + 1).ok_or("missing option value")?;
                match args[i].as_str() {
                    "--unit" => unit = value.parse()?,
                    "--trace" => path = Some(value),
                    _ => cap = value.parse()?,
                }
                i += 2;
            }
            "--trace-full" => {
                full = true;
                i += 1;
            }
            _ => return Err(format!("unknown option: {}", args[i]).into()),
        }
    }
    if path.is_none() && (full || cap != 65536) {
        return Err("trace options require --trace PATH".into());
    }
    let (weights, x) = educational_demo()?;
    let baseline = weights.forward(&x, None)?;
    let edited = weights.forward(&x, Some(unit))?;
    let column = weights.writing_vector(unit)?;
    println!("Educational f32 FFN: hidden=8 intermediate=16 seed=42 (synthetic weights)");
    println!(
        "input={x:?}\ngate={:?}\nup={:?}\nsilu_gate={:?}\nproduct={:?}",
        baseline.gate, baseline.up, baseline.activated_gate, baseline.product
    );
    println!(
        "unit={unit} activation={} writing_vector={column:?}",
        baseline.product[unit]
    );
    let observed: Vec<_> = baseline
        .output
        .iter()
        .zip(&edited.output)
        .map(|(a, b)| a - b)
        .collect();
    let expected: Vec<_> = column.iter().map(|w| w * baseline.product[unit]).collect();
    println!(
        "baseline={:?}\nzeroed={:?}\ndelta={observed:?}\nactivation_times_column={expected:?}",
        baseline.output, edited.output
    );
    if let Some(path) = path {
        let file = OpenOptions::new().write(true).create_new(true).open(path)?;
        let mut trace = Trace::new(file, cap, full);
        let run_id = format!(
            "{}-{}",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
        );
        let context = TraceContext {
            run_id: &run_id,
            token_position: 0,
            layer: 0,
        };
        for (name, values, selection) in [
            ("input", &x, None),
            ("ffn.gate", &baseline.gate, Some(unit)),
            ("ffn.up", &baseline.up, Some(unit)),
            ("ffn.silu", &baseline.activated_gate, Some(unit)),
            ("ffn.product", &baseline.product, Some(unit)),
            (
                "ffn.intervened_product",
                &edited.intervened_product,
                Some(unit),
            ),
            ("ffn.output", &baseline.output, None),
            ("ffn.edited_output", &edited.output, None),
            ("ffn.delta", &observed, None),
        ] {
            trace.record(&context, name, values, selection)?;
        }
    }
    Ok(())
}
mod cli;
fn run() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if matches!(
        args.first().map(String::as_str),
        Some("rocm-check" | "rocm-infer")
    ) {
        #[cfg(all(feature = "rocm", target_os = "linux"))]
        return Ok(qwen_rust_inference::rocm::run_cli(&args)?);
        #[cfg(not(all(feature = "rocm", target_os = "linux")))]
        return Err("ROCm commands need Linux/WSL2 and --features rocm".into());
    }
    if args.first().map(String::as_str) == Some("demo")
        && args.get(1).map(String::as_str) == Some("ffn")
    {
        demo_ffn()
    } else {
        cli::run(&args)
    }
}
