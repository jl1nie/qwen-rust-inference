use qwen_rust_inference::{
    text::{user_chat, TextTokenizer},
    validation::{compare, top},
    weights::load_model,
};
use serde_json::{json, Value};
use std::{fs, path::PathBuf};
fn numbers(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}
#[test]
#[ignore = "explicit opt-in: run tools/download-model.sh and generate_hybrid.py --model-dir first"]
fn public_bf16_weights_in_f32_match_pinned_source() {
    use sha2::{Digest, Sha256};
    let dir =
        PathBuf::from(std::env::var("QWEN_MODEL_DIR").unwrap_or_else(|_| "models/qwen".into()));
    let reference_bytes = fs::read(dir.join("reference.json")).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&reference_bytes)),
        fs::read_to_string(dir.join("reference.sha256"))
            .unwrap()
            .trim()
    );
    let reference: Value = serde_json::from_slice(&reference_bytes).unwrap();
    drop(reference_bytes);
    assert_eq!(reference["metadata"]["atol"], 3e-3);
    assert_eq!(reference["metadata"]["rtol"], 3e-3);
    assert_eq!(reference["metadata"]["dtype"], "float32");
    assert_eq!(reference["metadata"]["weights_dtype"], "bfloat16");
    assert_eq!(
        reference["metadata"]["source_revision"],
        "a6f5c8676b158c367dc3389f68dde7a97fb619d6"
    );
    assert_eq!(
        format!(
            "{:x}",
            Sha256::digest(include_bytes!("../tools/generate_hybrid.py"))
        ),
        reference["metadata"]["generator_sha256"]
    );
    let model = load_model(&dir).unwrap();
    let backend = qwen_rust_inference::benchmark::parse_backend(
        &std::env::var("QWEN_TEST_BACKEND").unwrap_or_else(|_| "vector".into()),
    )
    .unwrap();
    assert_eq!(
        model.revision(),
        reference["model_revision"].as_str().unwrap()
    );
    let mut records = Vec::new();
    for (case, v) in reference["cases"].as_array().unwrap().iter().enumerate() {
        let mut state = model.new_state();
        let mut final_logits = Vec::new();
        for (p, id) in v["ids"].as_array().unwrap().iter().enumerate() {
            let step = model
                .step(id.as_u64().unwrap() as u32, &mut state, backend, None)
                .unwrap();
            for (l, b) in step.blocks.iter().enumerate() {
                for (op, actual) in [
                    ("input_norm", &b.input_norm),
                    ("mixer", &b.mixed),
                    ("after_mixer", &b.after_mixer),
                    ("post_norm", &b.post_norm),
                    ("ffn_product", &b.ffn_product),
                    ("ffn", &b.ffn),
                    ("layer", &b.output),
                ] {
                    let expected = if op == "layer" {
                        numbers(&v["layers"][p][l])
                    } else {
                        numbers(&v["operations"][p][format!("{l}.{op}")])
                    };
                    let location = format!("case={case} token={p} layer={l} {op}");
                    let error = compare(actual, &expected, 3e-3, 3e-3, &location).unwrap();
                    records.push(json!({"where":location,"error":error}));
                }
            }
            let expected = numbers(&v["logits"][p]);
            let error = compare(&step.logits, &expected, 3e-3, 3e-3, "logits").unwrap();
            compare(
                &step.logits,
                &numbers(&v["full_logits"][p]),
                3e-3,
                3e-3,
                "source chunk prefill logits",
            )
            .unwrap();
            records.push(json!({"case":case,"token":p,"logits":error,"rust_top5":top(&step.logits,5),"reference_top5":top(&expected,5)}));
            final_logits = step.logits;
        }
        let ids: Vec<_> = v["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|i| i.as_u64().unwrap() as u32)
            .collect();
        // Every split is also exercised on the small model; here verify a real
        // continuation boundary without retaining all vocabulary activations.
        let mut split = model.new_state();
        let mid = ids.len() / 2;
        model
            .prefill(&ids[..mid], &mut split, backend, None)
            .unwrap();
        let rest = model
            .prefill(&ids[mid..], &mut split, backend, None)
            .unwrap();
        assert_eq!(split, state);
        compare(
            &rest.last().unwrap().logits,
            &final_logits,
            0.,
            0.,
            "real split prefill",
        )
        .unwrap();
    }
    let tokenizer = TextTokenizer::load(&dir).unwrap();
    assert_eq!(tokenizer.eos, 248046);
    assert_ne!(tokenizer.eos, 248069); // </think> is not end-of-answer.
    for case in reference["tokenizer_cases"].as_array().unwrap() {
        let rendered = user_chat(case["prompt"].as_str().unwrap(), false);
        assert_eq!(rendered, case["rendered"]);
        let ids = tokenizer.encode(&rendered).unwrap();
        assert_eq!(serde_json::to_value(&ids).unwrap(), case["ids"]);
        assert_eq!(tokenizer.decode(&ids).unwrap(), rendered);
    }
    let report = json!({"model_revision":model.revision(),"reference":reference["metadata"],"ignored_tensors":model.unused(),"comparisons":records});
    fs::write(
        dir.join("validation-report.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    eprintln!(
        "real validation passed; report={}",
        dir.join("validation-report.json").display()
    );
}
