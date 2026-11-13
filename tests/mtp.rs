//! MTP head mechanics on the synthetic hybrid fixture (its attention block
//! stands in for the head's layer): chaining drafts and truncating the head's
//! KV cache must leave the next true step exactly as if no draft had run, and
//! `step_hidden` must return the same logits as `step`.
use qwen_rust_inference::{
    block::MixerWeights,
    model::Model,
    mtp::{MtpState, MtpWeights},
    tensor::{Backend, Matrix},
};
#[test]
fn mtp_draft_chain_truncates_back() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/hybrid.json")).unwrap();
    let model = Model::from_fixture(&fixture).unwrap();
    let h = model.config().hidden_size;
    let block = model
        .blocks()
        .iter()
        .find(|b| matches!(b.mixer, MixerWeights::Attention(_)))
        .unwrap()
        .clone();
    let fc: Vec<f32> = (0..h * 2 * h)
        .map(|i| ((i * 7919 % 97) as f32 - 48.) / 400.)
        .collect();
    let mtp = MtpWeights {
        embedding_norm: vec![0.; h],
        hidden_norm: vec![0.1; h],
        fc: Matrix::new(h, 2 * h, fc).unwrap(),
        block,
        norm: model.norm().to_vec(),
        eps: model.config().rms_norm_eps,
    };
    let ids = [1u32, 3, 2, 5, 4];
    let b = Backend::Reference;
    let (mut s1, mut s2) = (model.new_state(), model.new_state());
    let (mut plain, mut chained) = (MtpState::default(), MtpState::default());
    for (p, w) in ids.windows(2).enumerate() {
        let full = model.step(w[0], &mut s1, b, None).unwrap();
        let (hidden, logits) = model.step_hidden(w[0], &mut s2, b, None).unwrap();
        assert_eq!(full.logits, logits);
        assert_eq!(full.normalized, hidden);
        let a = mtp.step(&model, &hidden, w[1], &mut plain, b).unwrap();
        let c = mtp.step(&model, &hidden, w[1], &mut chained, b).unwrap();
        assert_eq!(a.logits, c.logits, "position {p}");
        let keep = chained.len();
        let mut g = c.hidden;
        for t in [0u32, 2] {
            g = mtp.step(&model, &g, t, &mut chained, b).unwrap().hidden;
        }
        assert_eq!(chained.len(), keep + 2);
        chained.truncate(keep);
        assert_eq!(chained.len(), plain.len());
    }
}
/// A draft over a vocabulary subset picks the argmax of the full logits
/// restricted to those IDs (ties to the smaller ID, as greedy does).
#[test]
fn draft_vocab_argmax_is_restricted_full_argmax() {
    use qwen_rust_inference::mtp::DraftVocab;
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/hybrid.json")).unwrap();
    let model = Model::from_fixture(&fixture).unwrap();
    let h = model.config().hidden_size;
    let vocab = model.config().vocab_size as u32;
    let hidden: Vec<f32> = (0..h).map(|i| (i as f32 * 0.37).sin()).collect();
    let full = model
        .embedding()
        .project(&hidden, Backend::Reference)
        .unwrap();
    for ids in [
        vec![vocab - 1, 0, 2],
        (0..vocab).step_by(2).collect(),
        (0..vocab).collect(),
    ] {
        let d = DraftVocab::new(&model, &ids).unwrap();
        let best = ids
            .iter()
            .copied()
            .fold(None::<u32>, |b, i| match b {
                Some(b)
                    if full[b as usize] > full[i as usize]
                        || (full[b as usize] == full[i as usize] && b < i) =>
                {
                    Some(b)
                }
                _ => Some(i),
            })
            .unwrap();
        assert_eq!(d.argmax(&hidden, Backend::Reference).unwrap(), best);
        assert!(ids.iter().all(|&i| d.contains(i)));
    }
}
