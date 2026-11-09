//! Opt-in validation against full-position binary fixtures. No network/Python here.
use qwen_rust_inference::{
    generate::{generate_observed, Sampling, StopReason},
    model::{Model, StepActivations},
    state::{LayerState, ModelState},
    tensor::Backend,
    validation::{compare, top, ErrorMetrics},
    weights::load_model,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{self, BufReader, Read},
    path::{Path, PathBuf},
};

#[derive(Serialize, Deserialize)]
struct FileInfo {
    bytes: u64,
    sha256: String,
}
#[derive(Debug, Serialize)]
struct Aggregate {
    comparisons: usize,
    elements: usize,
    max_absolute: f64,
    max_relative: f64,
    min_cosine: f64,
}
impl Default for Aggregate {
    fn default() -> Self {
        Self {
            comparisons: 0,
            elements: 0,
            max_absolute: 0.,
            max_relative: 0.,
            min_cosine: 1.,
        }
    }
}
impl Aggregate {
    fn add(&mut self, m: &ErrorMetrics) {
        self.comparisons += 1;
        self.elements += m.elements;
        self.max_absolute = self.max_absolute.max(m.max_absolute);
        self.max_relative = self.max_relative.max(m.max_relative);
        self.min_cosine = self.min_cosine.min(m.cosine);
    }
}
fn read_values(reader: &mut impl Read, n: usize) -> io::Result<Vec<f32>> {
    let size = n
        .checked_mul(4)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "record overflow"))?;
    let mut bytes = vec![0; size];
    reader.read_exact(&mut bytes)?;
    let values: Vec<_> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    if values.iter().any(|v| !v.is_finite()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "non-finite fixture",
        ));
    }
    Ok(values)
}
fn end(reader: &mut impl Read) {
    let mut extra = [0];
    assert_eq!(
        reader.read(&mut extra).unwrap(),
        0,
        "trailing fixture bytes"
    );
}
fn verify_file(root: &Path, name: &str, info: &FileInfo) {
    assert_eq!(
        Path::new(name).file_name().and_then(|s| s.to_str()),
        Some(name),
        "non-local fixture path"
    );
    let mut file = File::open(root.join(name)).unwrap();
    assert_eq!(
        file.metadata().unwrap().len(),
        info.bytes,
        "file length: {name}"
    );
    let mut hasher = Sha256::new();
    let mut bytes = vec![0; 1024 * 1024];
    loop {
        let n = file.read(&mut bytes).unwrap();
        if n == 0 {
            break;
        }
        hasher.update(&bytes[..n]);
    }
    assert_eq!(
        format!("{:x}", hasher.finalize()),
        info.sha256,
        "hash: {name}"
    );
}
fn ids(v: &Value) -> Vec<u32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
        .collect()
}
fn winner(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |best, (i, &v)| {
            if v > best.1 {
                (i, v)
            } else {
                best
            }
        })
        .0
}
fn check_record(
    reader: &mut impl Read,
    a: &StepActivations,
    location: &str,
    layers: &mut [Aggregate],
    logits: &mut Aggregate,
) -> Vec<f32> {
    for (l, (actual, metric)) in a.layers.iter().zip(layers).enumerate() {
        let expected = read_values(reader, actual.len()).unwrap();
        metric.add(
            &compare(
                actual,
                &expected,
                3e-3,
                3e-3,
                &format!("{location} layer={l}"),
            )
            .unwrap(),
        );
    }
    let expected = read_values(reader, a.logits.len()).unwrap();
    logits.add(
        &compare(
            &a.logits,
            &expected,
            3e-3,
            3e-3,
            &format!("{location} logits"),
        )
        .unwrap(),
    );
    expected
}
fn check_state(root: &Path, name: &str, state: &ModelState) -> Aggregate {
    let mut reader = BufReader::new(File::open(root.join(name)).unwrap());
    let mut metrics = Aggregate::default();
    for (l, layer) in state.layers.iter().enumerate() {
        let mut check = |actual: &[f32], op: &str| {
            let expected = read_values(&mut reader, actual.len()).unwrap();
            metrics.add(
                &compare(
                    actual,
                    &expected,
                    3e-3,
                    3e-3,
                    &format!("{name} layer={l} {op}"),
                )
                .unwrap(),
            );
        };
        match layer {
            LayerState::Delta(s) => {
                check(&s.convolution, "conv");
                check(&s.recurrent, "recurrent");
            }
            LayerState::Attention(s) => {
                for k in &s.keys {
                    check(k, "key");
                }
                for v in &s.values {
                    check(v, "value");
                }
            }
        }
    }
    end(&mut reader);
    metrics
}
fn dimensions(state: &ModelState) -> Vec<(usize, usize)> {
    state
        .layers
        .iter()
        .filter_map(|s| match s {
            LayerState::Delta(s) => Some((s.convolution.len(), s.recurrent.len())),
            _ => None,
        })
        .collect()
}
fn check_kv_positions(state: &ModelState) {
    for s in &state.layers {
        if let LayerState::Attention(s) = s {
            assert_eq!(s.keys.len(), state.position);
            assert_eq!(s.values.len(), state.position);
        }
    }
}
fn validate_context(model: &Model, root: &Path, case: &Value, backend: Backend) -> Value {
    let name = case["name"].as_str().unwrap();
    let prompt = ids(&case["ids"]);
    let continuation = ids(&case["continuation"]);
    assert_eq!(
        prompt.len(),
        case["prefill_length"].as_u64().unwrap() as usize
    );
    assert_eq!(continuation.len(), 8);
    assert_eq!(
        case["records"].as_u64().unwrap() as usize,
        prompt.len() + continuation.len()
    );
    let mut reader =
        BufReader::new(File::open(root.join(case["values"].as_str().unwrap())).unwrap());
    let mut state = model.new_state();
    let fixed = dimensions(&state);
    let mut layer_metrics: Vec<_> = (0..model.blocks().len())
        .map(|_| Aggregate::default())
        .collect();
    let mut logits = Aggregate::default();
    let mut checkpoints = Vec::new();
    let mut top1_matches = 0;
    let mut positions = 0;
    let mut boundaries: Vec<_> = [1, 31, 32, 63, 64, 65, 127, 128, 129, 255, 256, 511, 512]
        .into_iter()
        .filter(|&n| n < prompt.len())
        .collect();
    boundaries.push(prompt.len());
    let mut start = 0;
    for &stop in &boundaries {
        assert!(model
            .prefill_observed(&[], &mut state, backend, None, |_, _| panic!())
            .unwrap()
            .is_none());
        model.prefill_observed(&prompt[start..stop],&mut state,backend,None,|p,a| {
            let expected=check_record(&mut reader,a,&format!("{name} position={p}"),&mut layer_metrics,&mut logits);
            positions+=1;top1_matches+=usize::from(winner(&a.logits)==winner(&expected));
            if [0,30,31,62,63,64,126,127,128,254,255,510,511].contains(&p) {checkpoints.push(json!({"position":p,"rust_top5":top(&a.logits,5),"reference_top5":top(&expected,5)}));}
            if (p+1)%32==0 {eprintln!("{name}: compared {} positions",p+1);}Ok(())
        }).unwrap();
        assert_eq!(dimensions(&state), fixed);
        check_kv_positions(&state);
        start = stop;
    }
    let prefill_state = check_state(root, case["prefill_state"].as_str().unwrap(), &state);
    for &token in &continuation {
        let p = state.position;
        let a = model.step(token, &mut state, backend, None).unwrap();
        let expected = check_record(
            &mut reader,
            &a,
            &format!("{name} continuation position={p}"),
            &mut layer_metrics,
            &mut logits,
        );
        positions += 1;
        top1_matches += usize::from(winner(&a.logits) == winner(&expected));
    }
    end(&mut reader);
    assert_eq!(dimensions(&state), fixed);
    check_kv_positions(&state);
    let decode_state = check_state(root, case["decode_state"].as_str().unwrap(), &state);
    json!({"name":name,"prefill_length":prompt.len(),"continuation_length":continuation.len(),"positions_compared":positions,"split_boundaries":boundaries,"layers":layer_metrics,"logits":logits,"prefill_state":prefill_state,"decode_state":decode_state,"top1_matches":top1_matches,"checkpoints":checkpoints})
}
#[test]
#[ignore = "opt-in: generate_extended.py --model-dir models/qwen, then run this test alone in release"]
fn long_context_continuation_states_and_natural_eos() {
    let model_dir =
        PathBuf::from(std::env::var("QWEN_MODEL_DIR").unwrap_or_else(|_| "models/qwen".into()));
    let root = model_dir.join("extended");
    let bytes = fs::read(root.join("manifest.json")).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        fs::read_to_string(root.join("manifest.sha256"))
            .unwrap()
            .trim()
    );
    let manifest: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(manifest["schema_version"], 1);
    let meta = &manifest["metadata"];
    for (key, value) in [
        (
            "generator_sha256",
            format!(
                "{:x}",
                Sha256::digest(include_bytes!("../tools/generate_extended.py"))
            ),
        ),
        (
            "helper_sha256",
            format!(
                "{:x}",
                Sha256::digest(include_bytes!("../tools/generate_hybrid.py"))
            ),
        ),
        (
            "source_sha256",
            format!(
                "{:x}",
                Sha256::digest(include_bytes!("../spec/upstream/modeling_qwen3_5.py"))
            ),
        ),
    ] {
        assert_eq!(meta[key], value);
    }
    assert_eq!(
        meta["source_revision"],
        "a6f5c8676b158c367dc3389f68dde7a97fb619d6"
    );
    assert_eq!(meta["dtype"], "float32");
    assert_eq!(meta["weights_dtype"], "bfloat16");
    assert_eq!(meta["torch"], "2.6.0+cpu");
    assert_eq!(meta["atol"], 3e-3);
    assert_eq!(meta["rtol"], 3e-3);
    for (name, info) in manifest["files"].as_object().unwrap() {
        verify_file(&root, name, &serde_json::from_value(info.clone()).unwrap());
    }
    let cases = manifest["cases"].as_array().unwrap();
    assert_eq!(
        cases
            .iter()
            .map(|c| c["prefill_length"].as_u64().unwrap())
            .collect::<Vec<_>>(),
        [32, 128, 512]
    );
    let model = load_model(&model_dir).unwrap();
    let backend = qwen_rust_inference::benchmark::parse_backend(
        &std::env::var("QWEN_TEST_BACKEND").unwrap_or_else(|_| "vector".into()),
    )
    .unwrap();
    assert_eq!(model.revision(), manifest["model_revision"]);
    let contexts: Vec<_> = cases
        .iter()
        .map(|c| validate_context(&model, &root, c, backend))
        .collect();
    let mut generations = Vec::new();
    assert_eq!(manifest["generation"].as_array().unwrap().len(), 2);
    for case in manifest["generation"].as_array().unwrap() {
        let prompt = ids(&case["ids"]);
        let expected_ids = ids(&case["generated"]);
        let eos = ids(&case["eos"]);
        assert_eq!(eos, [248044, 248046]);
        assert!(!eos.contains(&248069));
        assert!(eos.contains(expected_ids.last().unwrap()));
        let mut reader =
            BufReader::new(File::open(root.join(case["values"].as_str().unwrap())).unwrap());
        let mut layers: Vec<_> = (0..model.blocks().len())
            .map(|_| Aggregate::default())
            .collect();
        let mut logits = Aggregate::default();
        let mut count = 0;
        let actual =
            generate_observed(&model, &prompt, 64, &eos, Sampling::Greedy, None, |p, a| {
                if p + 1 >= prompt.len() {
                    check_record(
                        &mut reader,
                        a,
                        &format!("natural EOS step={count}"),
                        &mut layers,
                        &mut logits,
                    );
                    count += 1;
                }
                Ok(())
            })
            .unwrap();
        end(&mut reader);
        assert_eq!(actual.stop, StopReason::Eos);
        assert_eq!(actual.history.generated, expected_ids);
        assert_eq!(count, expected_ids.len());
        eprintln!("natural EOS passed: {:?}", actual.history.generated);
        generations.push(json!({"prompt":case["prompt"],"generated":actual.history.generated,"text":case["text"],"stop":"Eos","prediction_steps_compared":count,"layers":layers,"logits":logits}));
    }
    let report = json!({"model_revision":model.revision(),"backend":format!("{backend:?}"),"generation_backend":format!("{:?}",Backend::inference()),"metadata":meta,"contexts":contexts,"generation":generations});
    fs::write(
        root.join("validation-report.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    eprintln!("extended validation passed");
}
#[test]
fn binary_reference_rejects_truncation_and_nonfinite() {
    let mut valid = io::Cursor::new([1.25_f32.to_le_bytes(), (-2_f32).to_le_bytes()].concat());
    assert_eq!(read_values(&mut valid, 2).unwrap(), [1.25, -2.]);
    end(&mut valid);
    assert!(read_values(&mut io::Cursor::new(vec![0; 3]), 1).is_err());
    assert!(read_values(&mut io::Cursor::new(f32::NAN.to_le_bytes()), 1).is_err());
    assert!(read_values(&mut io::Cursor::new(Vec::<u8>::new()), usize::MAX).is_err());
}
#[test]
#[ignore = "opt-in: requires the pinned local tokenizer and extended manifest, no model allocation"]
fn extended_text_inputs_match_rust_tokenizer() {
    use qwen_rust_inference::text::{user_chat, TextTokenizer};
    let dir =
        PathBuf::from(std::env::var("QWEN_MODEL_DIR").unwrap_or_else(|_| "models/qwen".into()));
    let bytes = fs::read(dir.join("extended/manifest.json")).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        fs::read_to_string(dir.join("extended/manifest.sha256"))
            .unwrap()
            .trim()
    );
    let manifest: Value = serde_json::from_slice(&bytes).unwrap();
    let tokenizer = TextTokenizer::load(&dir).unwrap();
    for case in manifest["generation"].as_array().unwrap() {
        let rendered = user_chat(case["prompt"].as_str().unwrap(), false);
        assert_eq!(rendered, case["rendered"]);
        assert_eq!(tokenizer.encode(&rendered).unwrap(), ids(&case["ids"]));
    }
    let tail = tokenizer.encode(" したがって、答えを確認します。").unwrap();
    for case in manifest["cases"].as_array().unwrap() {
        let basis = tokenizer.encode(case["text"].as_str().unwrap()).unwrap();
        let length = case["prefill_length"].as_u64().unwrap() as usize;
        let repeated: Vec<_> = basis.into_iter().cycle().take(length).collect();
        assert_eq!(repeated, ids(&case["ids"]));
        assert_eq!(&tail[..8], ids(&case["continuation"]));
    }
}
#[test]
fn recorded_extended_manifest_provenance_and_record_sizes() {
    let bytes = include_bytes!("../spec/extended-reference-manifest.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        include_str!("../spec/extended-reference-manifest.sha256").trim()
    );
    let v: Value = serde_json::from_slice(bytes).unwrap();
    let meta = &v["metadata"];
    assert_eq!(
        meta["generator_sha256"],
        format!(
            "{:x}",
            Sha256::digest(include_bytes!("../tools/generate_extended.py"))
        )
    );
    assert_eq!(
        meta["helper_sha256"],
        format!(
            "{:x}",
            Sha256::digest(include_bytes!("../tools/generate_hybrid.py"))
        )
    );
    assert_eq!(
        meta["source_sha256"],
        format!(
            "{:x}",
            Sha256::digest(include_bytes!("../spec/upstream/modeling_qwen3_5.py"))
        )
    );
    assert_eq!(
        v["model_revision"],
        qwen_rust_inference::weights::MODEL_REVISION
    );
    assert_eq!(meta["atol"], 3e-3);
    assert_eq!(meta["rtol"], 3e-3);
    let per_record = (24 * 1024 + 248320) * 4;
    for case in v["cases"]
        .as_array()
        .unwrap()
        .iter()
        .chain(v["generation"].as_array().unwrap())
    {
        let name = case["values"].as_str().unwrap();
        assert_eq!(
            v["files"][name]["bytes"].as_u64().unwrap(),
            case["records"].as_u64().unwrap() * per_record
        );
        assert_eq!(v["files"][name]["sha256"].as_str().unwrap().len(), 64);
    }
}
