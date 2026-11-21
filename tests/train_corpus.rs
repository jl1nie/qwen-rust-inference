//! Learning text by cross-entropy (`train::corpus`), on the fixture.
#![cfg(feature = "train")]
use qwen_rust_inference::tensor::{Backend, InferenceError, Result};
use qwen_rust_inference::text::{user_chat, TextCodec};
use qwen_rust_inference::train::cache::LayerCache;
use qwen_rust_inference::train::corpus::{parse_pairs, score, Corpus, Pair};
use qwen_rust_inference::train::ffn::TopFfn;
use qwen_rust_inference::train::fit::{fit, Target};
use qwen_rust_inference::train::grad::log_softmax;
use qwen_rust_inference::train::optim::{Optimizer, OptimizerConfig};
mod train_common;
use train_common::fixture;

/// One ID per character, folded into the fixture's 19 IDs (18 = end).
struct Chars;
impl TextCodec for Chars {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(text.chars().map(|c| c as u32 % 17).collect())
    }
    fn decode(&self, _: &[u32]) -> Result<String> {
        Err(InferenceError("unused".into()))
    }
}
const END: u32 = 18;

fn pairs() -> Vec<Pair> {
    ["ab", "cd", "ef"]
        .iter()
        .zip(["xyz", "pq", "rst"])
        .map(|(p, r)| Pair {
            prompt: p.to_string(),
            response: r.to_string(),
            score: None,
        })
        .collect()
}

#[test]
fn pairs_parse_from_json_lines() {
    let text =
        "{\"prompt\": \"a\", \"response\": \"b\"}\n\n{\"prompt\": \"c\", \"response\": \"d\"}\n";
    let p = parse_pairs(text).unwrap();
    assert_eq!(p.len(), 2);
    assert_eq!(p[1].response, "d");
    let e = parse_pairs("{\"prompt\": \"a\", \"response\": \"b\"}\n{\"prompt\": 1}").unwrap_err();
    assert!(e.0.contains("line 2"), "{e}");
    assert!(parse_pairs("{\"prompt\": \"a\", \"response\": \"b\", \"x\": 1}").is_err());
    let s = parse_pairs("{\"prompt\": \"a\", \"response\": \"bcd\", \"score\": \"c\"}").unwrap();
    assert_eq!(s[0].score.as_deref(), Some("c"));
}

fn scored(response: &str, part: &str) -> Pair {
    Pair {
        prompt: "ab".into(),
        response: response.into(),
        score: Some(part.into()),
    }
}

#[test]
fn a_score_span_scores_only_the_tokens_that_cover_it() {
    let (model, _) = fixture();
    // "ab" -> "xyzw", scoring "yz": the positions that predict y and z only
    // (not x, w, or the end of the turn).
    let c = Corpus::from_pairs(
        &model,
        &Chars,
        &[scored("xyzw", "yz")],
        END,
        Backend::Vector,
    )
    .unwrap();
    let chat = user_chat("ab", false);
    let prompt_len = Chars.encode(&chat).unwrap().len();
    let ids = Chars.encode(&format!("{chat}xyzw")).unwrap();
    let targets: Vec<Target> = c.examples.iter().map(|e| e.target.clone()).collect();
    assert_eq!(
        targets,
        [
            Target::Id(ids[prompt_len + 1]),
            Target::Id(ids[prompt_len + 2])
        ]
    );
    let alone = LayerCache::capture(&model, &ids, Backend::Vector).unwrap();
    let d = model.config().hidden_size;
    // The first scored position is the one after x (prompt_len), predicting y.
    assert_eq!(
        c.inputs.post_norm[..d],
        alone.inputs.post_norm[prompt_len * d..(prompt_len + 1) * d]
    );
    // A span at the very start of the response is predicted from the
    // prompt's last position.
    let first =
        Corpus::from_pairs(&model, &Chars, &[scored("xyzw", "x")], END, Backend::Vector).unwrap();
    assert_eq!(first.examples.len(), 1);
    assert_eq!(
        first.inputs.post_norm[..d],
        alone.inputs.post_norm[(prompt_len - 1) * d..prompt_len * d]
    );
    // Pairs with and without a span mix.
    let mut mixed = pairs();
    mixed.push(scored("xyzw", "zw"));
    let m = Corpus::from_pairs(&model, &Chars, &mixed, END, Backend::Vector).unwrap();
    let counts: Vec<usize> = m.sequences.iter().map(|s| s.1).collect();
    assert_eq!(counts, [4, 3, 4, 2]);
}

/// Like `Chars`, but "yz" is one token.
struct MergesYz;
impl TextCodec for MergesYz {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        let mut it = text.chars().peekable();
        while let Some(c) = it.next() {
            if c == 'y' && it.peek() == Some(&'z') {
                it.next();
                ids.push(17);
            } else {
                ids.push(c as u32 % 17);
            }
        }
        Ok(ids)
    }
    fn decode(&self, _: &[u32]) -> Result<String> {
        Err(InferenceError("unused".into()))
    }
}

#[test]
fn a_score_span_must_occur_once_and_fall_on_token_boundaries() {
    let (model, _) = fixture();
    let err = |codec: &dyn TextCodec, p: Pair| {
        Corpus::from_pairs(&model, codec, &[p], END, Backend::Vector)
            .unwrap_err()
            .0
    };
    assert!(err(&Chars, scored("xyzw", "q")).contains("pair 1"));
    assert!(err(&Chars, scored("xyzw", "")).contains("pair 1"));
    assert!(err(&Chars, scored("xyxy", "xy")).contains("once"));
    // "yz" is one token, so neither "y" nor "z" can be scored alone...
    assert!(err(&MergesYz, scored("xyzw", "y")).contains("token"));
    assert!(err(&MergesYz, scored("xyzw", "zw")).contains("token"));
    // ...but the token as a whole can.
    let c = Corpus::from_pairs(
        &model,
        &MergesYz,
        &[scored("xyzw", "xyz")],
        END,
        Backend::Vector,
    )
    .unwrap();
    assert_eq!(c.examples.len(), 2);
}

#[test]
fn corpus_keeps_the_scored_positions_of_each_pair() {
    let (model, _) = fixture();
    let pairs = pairs();
    let c = Corpus::from_pairs(&model, &Chars, &pairs, END, Backend::Vector).unwrap();
    // Each pair scores its response and the end of the turn.
    let counts: Vec<usize> = c.sequences.iter().map(|s| s.1).collect();
    assert_eq!(counts, [4, 3, 4]);
    assert_eq!(c.examples.len(), 11);
    assert_eq!(c.inputs.n, 11);
    let d = model.config().hidden_size;
    let mut total = 0;
    for (s, p) in c.sequences.iter().zip(&pairs) {
        let chat = user_chat(&p.prompt, false);
        let mut ids = Chars.encode(&format!("{chat}{}", p.response)).unwrap();
        ids.push(END);
        let prompt_len = Chars.encode(&chat).unwrap().len();
        total += ids.len();
        // The same inputs as reading this pair alone, at its scored positions.
        let alone = LayerCache::capture(&model, &ids, Backend::Vector).unwrap();
        for k in 0..s.1 {
            let pos = prompt_len - 1 + k;
            let e = &c.examples[s.0 + k];
            assert_eq!(e.position, s.0 + k);
            assert_eq!(e.target, Target::Id(ids[pos + 1]));
            let row = |v: &[f32], i: usize| v[i * d..(i + 1) * d].to_vec();
            assert_eq!(
                row(&c.inputs.post_norm, s.0 + k),
                row(&alone.inputs.post_norm, pos)
            );
            assert_eq!(
                row(&c.inputs.after_mixer, s.0 + k),
                row(&alone.inputs.after_mixer, pos)
            );
        }
    }
    assert_eq!(c.tokens, total);
    let text = Corpus::from_text(&model, &[1, 4, 2, 8, 3], Backend::Vector).unwrap();
    assert_eq!(text.examples.len(), 4);
    assert_eq!(text.examples[3].target, Target::Id(3));
}

#[test]
fn score_is_the_mean_cross_entropy_and_top1() {
    let (model, _) = fixture();
    let c = Corpus::from_pairs(&model, &Chars, &pairs(), END, Backend::Vector).unwrap();
    let top = TopFfn::new(&model).unwrap();
    let s = score(&top, &c.inputs, &c.examples, 3).unwrap();
    let f = top.forward(&c.inputs).unwrap();
    let vocab = model.config().vocab_size;
    let mut loss = 0.;
    let mut hits = 0;
    for (e, logits) in c.examples.iter().zip(f.head.logits.chunks_exact(vocab)) {
        let Target::Id(id) = e.target else { panic!() };
        loss -= log_softmax(logits).unwrap()[id as usize];
        let best = (0..vocab)
            .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
            .unwrap();
        hits += usize::from(best == id as usize);
    }
    assert!((s.loss - loss / 11.).abs() < 1e-9);
    assert_eq!(s.top1, hits as f64 / 11.);
}

#[test]
fn training_on_pairs_lowers_their_loss_and_holdout_is_scored_apart() {
    let (model, _) = fixture();
    let all = pairs();
    let train = Corpus::from_pairs(&model, &Chars, &all[..2], END, Backend::Vector).unwrap();
    let held = Corpus::from_pairs(&model, &Chars, &all[2..], END, Backend::Vector).unwrap();
    let mut top = TopFfn::new(&model).unwrap();
    let before = (
        score(&top, &train.inputs, &train.examples, 4).unwrap(),
        score(&top, &held.inputs, &held.examples, 4).unwrap(),
    );
    let mut o = Optimizer::new(OptimizerConfig::adam(0.02), &top.ffn.sizes());
    fit(
        &mut top,
        &train.inputs,
        &train.examples,
        &mut o,
        40,
        4,
        |_| Ok(()),
    )
    .unwrap();
    let after_train = score(&top, &train.inputs, &train.examples, 4).unwrap();
    let after_held = score(&top, &held.inputs, &held.examples, 4).unwrap();
    // The 8-wide fixture's final RMSNorm bounds how low the loss can go
    // (as in train_fit.rs), so a clear drop is all that is asked.
    assert!(
        after_train.loss < 0.8 * before.0.loss,
        "{before:?} -> {after_train:?}"
    );
    assert!(after_train.top1 > before.0.top1);
    // The held-out pair was not trained on; it is only measured.
    assert_eq!(after_held.positions, 4);
    assert!(after_held.loss.is_finite() && after_held.loss != before.1.loss);
}
