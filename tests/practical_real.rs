use qwen_rust_inference::{
    chat::ChatSession,
    generate::{Sampling, StopReason},
    model::Model,
    snapshot::{load_chat, save_chat, DEFAULT_LIMIT},
    tensor::Backend,
    text::{render_chat, Message, TextTokenizer},
    weights::load_model,
};
use serde_json::{json, Value};
use std::{
    fs,
    path::PathBuf,
    time::{Instant, SystemTime, UNIX_EPOCH},
};
#[test]
#[ignore = "opt-in: pinned real weights; run alone after vector conformance/benchmark"]
fn real_compact_chat_cache_and_midturn_snapshot() {
    let dir =
        PathBuf::from(std::env::var("QWEN_MODEL_DIR").unwrap_or_else(|_| "models/qwen".into()));
    let t = TextTokenizer::load(&dir).unwrap();
    let m: Model = load_model(&dir).unwrap();
    let fixture: Value = serde_json::from_str(include_str!("fixtures/chat-template.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let messages: Vec<Message> = serde_json::from_value(case["messages"].clone()).unwrap();
        let rendered = render_chat(
            &messages,
            case["thinking"].as_bool().unwrap(),
            case["add_generation_prompt"].as_bool().unwrap(),
        )
        .unwrap();
        assert_eq!(
            serde_json::to_value(t.encode(&rendered).unwrap()).unwrap(),
            case["ids"]
        );
    }
    let mut full = m.new_state();
    let mut compact = m.new_state();
    for id in [16, 10, 16, 28] {
        let a = m.step(id, &mut full, Backend::Vector, None).unwrap();
        assert_eq!(
            a.logits,
            m.step_compact(id, &mut compact, Backend::Vector, None)
                .unwrap()
        );
        assert_eq!(full, compact);
    }
    drop(full);
    drop(compact);
    let mut chat = ChatSession::new(&m, Backend::Vector, Sampling::Greedy, false, None).unwrap();
    let first = chat
        .user_turn(&m, &t, "数字だけで答えてください。1+1は？", 1)
        .unwrap();
    assert_eq!(first.stop, StopReason::MaxTokens);
    let root = dir.join(format!(
        "practical-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir(&root).unwrap();
    let state = root.join("midturn.qrs");
    save_chat(&m, &chat, &state, DEFAULT_LIMIT).unwrap();
    let snapshot_bytes = fs::metadata(&state).unwrap().len();
    let mut resumed = load_chat(&m, &t, &state, DEFAULT_LIMIT).unwrap();
    assert_eq!(chat, resumed);
    let continued = chat.continue_turn(&m, &t, 16).unwrap();
    let restored = resumed.continue_turn(&m, &t, 16).unwrap();
    assert_eq!(continued.generated, restored.generated);
    assert_eq!(chat, resumed);
    assert_eq!(continued.stop, StopReason::Eos);
    drop(chat);
    let mut replay = resumed.clone();
    let start = Instant::now();
    let cached = resumed
        .user_turn(&m, &t, "その数に1を足すと？数字だけ答えてください。", 16)
        .unwrap();
    let cached_ms = start.elapsed().as_secs_f64() * 1000.;
    let start = Instant::now();
    let fresh = replay
        .user_turn_with_cache(
            &m,
            &t,
            "その数に1を足すと？数字だけ答えてください。",
            16,
            false,
        )
        .unwrap();
    let replay_ms = start.elapsed().as_secs_f64() * 1000.;
    assert_eq!(cached.generated, fresh.generated);
    assert_eq!(resumed.engine().state(), replay.engine().state());
    assert!(cached.reused_tokens > 0);
    assert_eq!(fresh.reused_tokens, 0);
    let report = json!({"model_revision":m.revision(),"model_identity":m.identity(),"backend":"vector","compact_equals_full":true,"midturn_snapshot_bytes":snapshot_bytes,"resume_equals_uninterrupted":true,"first":first,"continued":continued,"turn2_cached":cached,"turn2_replay":fresh,"cache_and_replay_equal":true,"cached_ms":cached_ms,"replay_ms":replay_ms,"memory":qwen_rust_inference::benchmark::memory()});
    fs::write(
        dir.join("practical-validation.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    fs::remove_dir_all(root).unwrap();
    eprintln!("real chat/resume conformance passed");
}
