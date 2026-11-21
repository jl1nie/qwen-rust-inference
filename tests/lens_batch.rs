//! `lens::logit_lens_many` and `lens::lens_item` (the `lens-batch` command),
//! on the fixture with a one-ID-per-character codec.
use qwen_rust_inference::lens::{lens_item, logit_lens, logit_lens_many, LensItem};
use qwen_rust_inference::model::Model;
use qwen_rust_inference::tensor::{Backend, Result};
use qwen_rust_inference::text::{user_chat, TextCodec};

fn fixture() -> Model {
    let v: serde_json::Value = serde_json::from_str(include_str!("fixtures/hybrid.json")).unwrap();
    Model::from_fixture(&v).unwrap()
}

/// One ID per character, folded into the fixture's 19 IDs.
struct Chars;
impl TextCodec for Chars {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(text.chars().map(|c| c as u32 % 17).collect())
    }
    fn decode(&self, ids: &[u32]) -> Result<String> {
        Ok(ids
            .iter()
            .map(|i| format!("<{i}>"))
            .collect::<Vec<_>>()
            .concat())
    }
}

#[test]
fn many_targets_equal_one_at_a_time() {
    let model = fixture();
    let mut state = model.new_state();
    let mut a = None;
    for id in [1, 4, 2, 8] {
        a = Some(
            model
                .step(id, &mut state, Backend::Reference, None)
                .unwrap(),
        );
    }
    let a = a.unwrap();
    let many = logit_lens_many(&model, &a, &[3, 11, 0], Backend::Reference).unwrap();
    for (rows, t) in many.iter().zip([3, 11, 0]) {
        assert_eq!(
            rows,
            &logit_lens(&model, &a, t, Backend::Reference).unwrap()
        );
    }
    assert!(logit_lens_many(&model, &a, &[19], Backend::Reference).is_err());
}

#[test]
fn lens_item_reads_the_position_after_the_prefix() {
    let model = fixture();
    let item = LensItem {
        id: "q".into(),
        prompt: "ab".into(),
        prefix: "xy".into(),
        targets: [
            ("first".to_string(), "c".to_string()),
            ("second".to_string(), "de".to_string()),
        ]
        .into_iter()
        .collect(),
    };
    let r = lens_item(&model, &Chars, &item, None, 3, &[], Backend::Reference).unwrap();
    // The same position by hand: the chat prompt, then the prefix.
    let ids = Chars
        .encode(&format!("{}xy", user_chat("ab", false)))
        .unwrap();
    assert_eq!(r.position, ids.len() - 1);
    let mut state = model.new_state();
    let mut a = None;
    for &id in &ids {
        a = Some(
            model
                .step(id, &mut state, Backend::Reference, None)
                .unwrap(),
        );
    }
    let a = a.unwrap();
    let chosen = qwen_rust_inference::validation::top(&a.logits, 1)[0].0 as u32;
    assert_eq!(r.chosen.0, chosen);
    // Targets (sorted by name) and "chosen" last; the first token of each text.
    let names: Vec<&str> = r.targets.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(names, ["first", "second", "chosen"]);
    assert_eq!(r.targets[0].id, 'c' as u32 % 17);
    assert_eq!(r.targets[1].id, 'd' as u32 % 17);
    for t in &r.targets {
        let rows = logit_lens(&model, &a, t.id, Backend::Reference).unwrap();
        assert_eq!(t.ranks, rows.iter().map(|x| x.rank).collect::<Vec<_>>());
        assert_eq!(
            t.probabilities,
            rows.iter().map(|x| x.probability).collect::<Vec<_>>()
        );
    }
    assert_eq!(*r.targets[2].ranks.last().unwrap(), 0);
    assert_eq!(r.tops.len(), model.blocks().len());
    assert_eq!(r.generated.as_ref().unwrap().matches('<').count(), 3);

    let none = LensItem {
        targets: Default::default(),
        ..item
    };
    assert!(lens_item(&model, &Chars, &none, None, 0, &[], Backend::Reference).is_err());
}
