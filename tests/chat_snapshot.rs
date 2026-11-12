use qwen_rust_inference::{
    chat::ChatSession,
    generate::Sampling,
    model::Model,
    session::InferenceSession,
    snapshot::{load_chat, load_session, save_chat, save_session, DEFAULT_LIMIT},
    tensor::{Backend, Result},
    text::{render_chat, Message, TextCodec},
};
use std::{
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};
struct TestCodec;
impl TextCodec for TestCodec {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(text.chars().map(|c| c as u32 % 16).collect())
    }
    fn decode(&self, ids: &[u32]) -> Result<String> {
        Ok(ids
            .iter()
            .map(|&i| (b'a' + (i % 19) as u8) as char)
            .collect())
    }
}
fn model() -> Model {
    let mut v: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/hybrid.json")).unwrap();
    v["config"]["max_position_embeddings"] = 4096.into();
    Model::from_fixture(&v).unwrap()
}
fn dir(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "qwen-{name}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir(&path).unwrap();
    path
}
#[test]
fn template_matches_pinned_jinja() {
    use sha2::{Digest, Sha256};
    let bytes = include_bytes!("fixtures/chat-template.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        include_str!("fixtures/chat-template.sha256").trim()
    );
    let v: serde_json::Value = serde_json::from_slice(bytes).unwrap();
    assert_eq!(
        v["metadata"]["generator_sha256"],
        format!(
            "{:x}",
            Sha256::digest(include_bytes!("../tools/generate_chat.py"))
        )
    );
    for case in v["cases"].as_array().unwrap() {
        let messages: Vec<Message> = serde_json::from_value(case["messages"].clone()).unwrap();
        assert_eq!(
            render_chat(
                &messages,
                case["thinking"].as_bool().unwrap(),
                case["add_generation_prompt"].as_bool().unwrap()
            )
            .unwrap(),
            case["rendered"]
        );
    }
}
#[test]
fn seeded_pending_snapshot_continues_exactly_and_rejects_corruption() {
    let m = model();
    let path = dir("raw");
    let file = path.join("state.qrs");
    let mut s = InferenceSession::new(
        &m,
        Backend::Vector,
        Sampling::Seeded {
            temperature: 0.8,
            top_k: 8,
            seed: 42,
        },
    )
    .unwrap();
    s.prefill(&m, &[1, 4, 2]).unwrap();
    s.generate(&m, 3, &[]).unwrap();
    save_session(&m, &s, &file, DEFAULT_LIMIT).unwrap();
    assert!(save_session(&m, &s, &file, DEFAULT_LIMIT).is_err());
    let mut restored = load_session(&m, &file, DEFAULT_LIMIT).unwrap();
    assert_eq!(s, restored);
    let a = s.generate(&m, 12, &[]).unwrap();
    let b = restored.generate(&m, 12, &[]).unwrap();
    assert_eq!(a.history, b.history);
    assert_eq!(s, restored);
    assert!(load_session(&m, &file, 32).is_err());
    let mut bad = fs::read(&file).unwrap();
    let n = bad.len();
    bad[n - 40] ^= 1;
    fs::write(path.join("bad"), &bad).unwrap();
    assert!(load_session(&m, &path.join("bad"), DEFAULT_LIMIT).is_err());
    fs::write(path.join("short"), &bad[..n - 10]).unwrap();
    assert!(load_session(&m, &path.join("short"), DEFAULT_LIMIT).is_err());
    let mut different: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/hybrid.json")).unwrap();
    different["config"]["max_position_embeddings"] = 4095.into();
    let other = Model::from_fixture(&different).unwrap();
    assert!(load_session(&other, &file, DEFAULT_LIMIT).is_err());
    different["config"]["max_position_embeddings"] = 4096.into();
    let value = different["tensors"]["model.language_model.embed_tokens.weight"]["values"][0]
        .as_f64()
        .unwrap();
    different["tensors"]["model.language_model.embed_tokens.weight"]["values"][0] =
        (value + 0.125).into();
    let changed_weights = Model::from_fixture(&different).unwrap();
    assert_ne!(m.identity(), changed_weights.identity());
    assert!(load_session(&changed_weights, &file, DEFAULT_LIMIT).is_err());
    fs::remove_dir_all(path).unwrap();
}
#[test]
fn cached_multiturn_equals_full_replay_and_chat_snapshot() {
    let m = model();
    let t = TestCodec;
    let mut cached = ChatSession::new(
        &m,
        Backend::Vector,
        Sampling::Greedy,
        false,
        Some("system".into()),
    )
    .unwrap();
    let mut replay = cached.clone();
    for prompt in ["first", "second", "third"] {
        let a = cached.user_turn(&m, &t, prompt, 4).unwrap();
        let b = replay
            .user_turn_with_cache(&m, &t, prompt, 4, false)
            .unwrap();
        assert_eq!(a.generated, b.generated);
        assert_eq!(cached.engine().state(), replay.engine().state());
        if prompt != "first" {
            assert!(a.reused_tokens > 0);
            assert_eq!(b.reused_tokens, 0);
        }
    }
    let path = dir("chat");
    let file = path.join("chat.qrs");
    save_chat(&m, &cached, &file, DEFAULT_LIMIT).unwrap();
    let mut resumed = load_chat(&m, &t, &file, DEFAULT_LIMIT).unwrap();
    assert_eq!(cached, resumed);
    assert_eq!(
        cached.continue_turn(&m, &t, 3).unwrap().generated,
        resumed.continue_turn(&m, &t, 3).unwrap().generated
    );
    assert_eq!(
        cached.user_turn(&m, &t, "fourth", 3).unwrap().generated,
        resumed.user_turn(&m, &t, "fourth", 3).unwrap().generated
    );
    let before = cached.clone();
    assert!(cached.user_turn(&m, &t, &"x".repeat(5000), 3).is_err());
    assert_eq!(cached, before);
    fs::remove_dir_all(path).unwrap();
}
#[test]
fn empty_and_nonpending_states_eos_and_semantic_header_errors() {
    use sha2::{Digest, Sha256};
    let m = model();
    let root = dir("edge");
    let path = root.join("empty");
    let mut s = InferenceSession::new(&m, Backend::Vector, Sampling::Greedy).unwrap();
    save_session(&m, &s, &path, DEFAULT_LIMIT).unwrap();
    assert_eq!(s, load_session(&m, &path, DEFAULT_LIMIT).unwrap());
    s.prefill(&m, &[1, 4]).unwrap();
    let nonpending = root.join("prefill");
    save_session(&m, &s, &nonpending, DEFAULT_LIMIT).unwrap();
    assert_eq!(s, load_session(&m, &nonpending, DEFAULT_LIMIT).unwrap());
    let next = s
        .logits()
        .unwrap()
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap()
        .0 as u32;
    assert_eq!(
        s.generate(&m, 10, &[next]).unwrap().stop,
        qwen_rust_inference::generate::StopReason::Eos
    );
    let eos = root.join("eos");
    save_session(&m, &s, &eos, DEFAULT_LIMIT).unwrap();
    let mut restored = load_session(&m, &eos, DEFAULT_LIMIT).unwrap();
    s.prefill(&m, &[2]).unwrap();
    restored.prefill(&m, &[2]).unwrap();
    assert_eq!(s, restored);
    // Recompute the checksum so version/pending validation, not just corruption
    // detection, must reject these syntactically valid but inconsistent files.
    let valid = fs::read(&nonpending).unwrap();
    let n = u64::from_le_bytes(valid[8..16].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&valid[16..16 + n]).unwrap();
    let mut nonfinite = valid.clone();
    let payload = 16 + n + 8;
    nonfinite[payload..payload + 4].copy_from_slice(&f32::NAN.to_le_bytes());
    let digest = Sha256::digest(&nonfinite[..nonfinite.len() - 32]);
    let end = nonfinite.len();
    nonfinite[end - 32..].copy_from_slice(&digest);
    let bad_float = root.join("nonfinite");
    fs::write(&bad_float, nonfinite).unwrap();
    assert!(load_session(&m, &bad_float, DEFAULT_LIMIT)
        .unwrap_err()
        .to_string()
        .contains("NaN/Inf"));
    for key in ["version", "pending"] {
        let mut h = header.clone();
        if key == "version" {
            h["version"] = 2.into();
        } else {
            h["live"]["pending"] = Some(1_u32).into();
        }
        let h = serde_json::to_vec(&h).unwrap();
        let mut bytes = valid[..8].to_vec();
        bytes.extend((h.len() as u64).to_le_bytes());
        bytes.extend(h);
        bytes.extend(&valid[16 + n..valid.len() - 32]);
        let checksum = Sha256::digest(&bytes);
        bytes.extend(checksum);
        let bad = root.join(key);
        fs::write(&bad, bytes).unwrap();
        assert!(load_session(&m, &bad, DEFAULT_LIMIT).is_err());
    }
    let unused_chat = ChatSession::new(&m, Backend::Vector, Sampling::Greedy, false, None).unwrap();
    let chat = root.join("freshchat");
    save_chat(&m, &unused_chat, &chat, DEFAULT_LIMIT).unwrap();
    assert_eq!(
        unused_chat,
        load_chat(&m, &TestCodec, &chat, DEFAULT_LIMIT).unwrap()
    );
    fs::remove_dir_all(root).unwrap();
}
#[test]
fn primed_system_prefix_is_reused_and_saved() {
    let m = model();
    let t = TestCodec;
    let new = || {
        ChatSession::new(
            &m,
            Backend::Vector,
            Sampling::Greedy,
            false,
            Some("long system document".into()),
        )
        .unwrap()
    };
    let mut primed = new();
    let n = primed.prime(&m, &t).unwrap();
    assert!(n > 0);
    assert_eq!(primed.prime(&m, &t).unwrap(), 0);
    let root = dir("primed");
    let file = root.join("primed.qrs");
    save_chat(&m, &primed, &file, DEFAULT_LIMIT).unwrap();
    let mut loaded = load_chat(&m, &t, &file, DEFAULT_LIMIT).unwrap();
    assert_eq!(primed, loaded);
    let mut plain = new();
    let a = loaded.user_turn(&m, &t, "question", 4).unwrap();
    let b = plain.user_turn(&m, &t, "question", 4).unwrap();
    assert_eq!(a.reused_tokens, n);
    assert_eq!(b.reused_tokens, 0);
    assert_eq!(a.generated, b.generated);
    assert_eq!(loaded.engine().state(), plain.engine().state());
    fs::remove_dir_all(root).unwrap();
}
#[test]
fn chat_edit_change_recomputes_history_and_is_saved() {
    use qwen_rust_inference::edit::{Edit, EditManifest};
    let m = model();
    let t = TestCodec;
    let manifest = EditManifest {
        model_revision: m.revision().to_string(),
        edits: vec![Edit {
            layer: 1,
            tensor_name: "model.language_model.layers.1.mlp.down_proj.weight".into(),
            columns: vec![0, 1],
            direction: None,
            coefficient: 1.,
        }],
    };
    let mut plain = ChatSession::new(
        &m,
        Backend::Vector,
        Sampling::Greedy,
        false,
        Some("sys".into()),
    )
    .unwrap();
    plain.prime(&m, &t).unwrap();
    plain.user_turn(&m, &t, "first", 4).unwrap();
    let mut edited = plain.clone();
    edited.set_edit(&m, Some(manifest.clone())).unwrap();
    assert_eq!(edited.engine().tokens(), plain.engine().tokens());
    assert_ne!(edited.engine().state(), plain.engine().state());
    let mut back = edited.clone();
    back.set_edit(&m, None).unwrap();
    assert_eq!(back, plain);
    let root = dir("edit");
    save_chat(&m, &edited, &root.join("e.qrs"), DEFAULT_LIMIT).unwrap();
    let loaded = load_chat(&m, &t, &root.join("e.qrs"), DEFAULT_LIMIT).unwrap();
    assert_eq!(loaded, edited);
    assert_eq!(loaded.engine().edit(), Some(&manifest));
    let mut wrong = manifest.clone();
    wrong.model_revision = "other".into();
    assert!(edited.clone().set_edit(&m, Some(wrong)).is_err());
    fs::remove_dir_all(root).unwrap();
}
#[test]
fn watched_chat_sees_chosen_positions_without_changing_it() {
    use qwen_rust_inference::{model::StepActivations, session::Watch};
    use std::collections::BTreeSet;
    struct At {
        wanted: BTreeSet<usize>,
        seen: Vec<(usize, usize)>,
    }
    impl Watch for At {
        fn wants(&self, p: usize) -> bool {
            self.wanted.contains(&p)
        }
        fn observe(&mut self, p: usize, a: &StepActivations) -> Result<()> {
            self.seen.push((p, a.blocks.len()));
            Ok(())
        }
    }
    let m = model();
    let t = TestCodec;
    let new = || {
        ChatSession::new(
            &m,
            Backend::Vector,
            Sampling::Greedy,
            false,
            Some("sys".into()),
        )
        .unwrap()
    };
    let (mut plain, mut watched) = (new(), new());
    let mut w = At {
        wanted: BTreeSet::from([2, 9, 30]),
        seen: vec![],
    };
    plain.prime(&m, &t).unwrap();
    watched.prime_watched(&m, &t, Some(&mut w)).unwrap();
    for p in ["first", "second"] {
        let a = plain.user_turn(&m, &t, p, 4).unwrap();
        let b = watched
            .user_turn_watched(&m, &t, p, 4, true, Some(&mut w))
            .unwrap();
        assert_eq!(a.generated, b.generated);
    }
    let a = plain.continue_turn(&m, &t, 2).unwrap();
    let b = watched
        .continue_turn_watched(&m, &t, 2, Some(&mut w))
        .unwrap();
    assert_eq!(a.generated, b.generated);
    assert_eq!(plain, watched);
    let positions: BTreeSet<usize> = w.seen.iter().map(|&(p, _)| p).collect();
    assert_eq!(positions, BTreeSet::from([2, 9, 30]));
    assert!(w.seen.iter().all(|&(_, n)| n == m.blocks().len()));
    // Recomputation after an edit change is watched too; a closure watches all.
    let mut all = Vec::new();
    let mut closure = |p: usize, _: &StepActivations| -> Result<()> {
        all.push(p);
        Ok(())
    };
    watched
        .set_edit_watched(&m, None, Some(&mut closure))
        .unwrap();
    let fed = watched.engine().state().position;
    assert_eq!(all, (0..fed).collect::<Vec<_>>());
    assert_eq!(plain, watched);
}
