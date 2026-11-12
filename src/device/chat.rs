//! Multi-turn chat on a device run with the CPU engine's prefix reuse
//! (src/chat.rs) and its snapshot format (src/snapshot.rs).
//!
//! Prefix reuse ("token cache") on a hybrid model: attention KV rows can be
//! kept for any prefix, but a DeltaNet layer's state is one fixed-size matrix
//! that has already absorbed every later token, so it cannot be cut back to
//! an arbitrary earlier position. We therefore keep exact checkpoints only at
//! the places a later turn returns to: the end of the system message (prime)
//! and the end of the last user message (anchor). Rewinding = restore the
//! DeltaNet copy and move the position back; the KV rows below it are still
//! the ones computed for that prefix.
//!
//! Edits and traces (ADR 0010) work during a chat too: the edit is the run's
//! overlay and is stored in snapshots; an observer sees chosen absolute
//! positions, each computed alone with captures when it is (re)computed.
use super::{DeviceCheckpoint, DeviceRun, DeviceStep};
use crate::{
    chat::{ChatSession, Turn},
    edit::EditManifest,
    generate::{select, Sampling, StopReason},
    model::Model,
    session::InferenceSession,
    snapshot::{load_chat, save_chat},
    state::{KvCache, LayerState, ModelState},
    tensor::{require, Backend, Result},
    text::{render_chat, render_system, Message, Role, TextCodec},
    weights::err,
};
use std::{collections::BTreeSet, path::Path};
/// Called with (position, token ID, captured step) for every observed position.
pub type Observer<'a> = Box<dyn FnMut(usize, u32, &DeviceStep) + 'a>;
const EOS: [u32; 2] = [248044, 248046];
/// A rewind point: the token IDs it covers, their final logits and the
/// DeltaNet state there. Always a prefix of the live tokens.
#[derive(Clone)]
struct Anchor {
    tokens: Vec<u32>,
    logits: Vec<f32>,
    rng: u64,
    checkpoint: DeviceCheckpoint,
}
pub struct DeviceChat<'a> {
    run: DeviceRun<'a>,
    model_id: String,
    revision: String,
    /// CPU backend recorded in snapshots (a CPU resume uses it).
    backend: Backend,
    sampling: Sampling,
    rng: u64,
    tokens: Vec<u32>,
    /// Last sampled ID, not yet fed to the model (same as InferenceSession).
    pending: Option<u32>,
    logits: Option<Vec<f32>>,
    anchor: Option<Anchor>,
    /// End of the primed system message; survives later turns.
    system: Option<Anchor>,
    messages: Vec<Message>,
    thinking: bool,
    response_ids: Vec<u32>,
    last_stop: Option<StopReason>,
    observer: Option<Observer<'a>>,
    observed: BTreeSet<usize>,
}
impl<'a> DeviceChat<'a> {
    pub fn new(
        model: &Model,
        run: DeviceRun<'a>,
        sampling: Sampling,
        thinking: bool,
        system: Option<String>,
    ) -> Result<Self> {
        // Reuse the CPU constructor's validation of sampling parameters.
        let cpu = ChatSession::new(model, Backend::inference(), sampling, thinking, system)?;
        Self::from_session(model, run, cpu)
    }
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }
    pub fn position(&self) -> usize {
        self.run.position()
    }
    pub fn capacity(&self) -> usize {
        self.run.capacity()
    }
    pub fn edit(&self) -> Option<&EditManifest> {
        self.run.edit.as_ref().map(|e| &e.manifest)
    }
    /// Observe these absolute positions whenever they are computed (prefill,
    /// generation, or recomputation after a rewind or an edit change).
    pub fn set_observer(&mut self, positions: BTreeSet<usize>, observer: Option<Observer<'a>>) {
        self.observed = positions;
        self.observer = observer;
    }
    pub fn observe(&mut self, position: usize) {
        self.observed.insert(position);
    }
    pub fn observed(&self) -> &BTreeSet<usize> {
        &self.observed
    }
    /// Change the edit mid-conversation (see `ChatSession::set_edit`): the
    /// same token history is recomputed under the new edit, re-taking the
    /// system and anchor checkpoints on the way; messages, pending token and
    /// RNG stay. Returns the number of recomputed positions.
    pub fn set_edit(&mut self, model: &Model, edit: Option<&EditManifest>) -> Result<usize> {
        self.run.set_edit(model, edit)?;
        let tokens = std::mem::take(&mut self.tokens);
        let pending = self.pending;
        let fed = tokens.len() - usize::from(pending.is_some());
        let rng = self.rng;
        let mut points: Vec<(usize, bool, u64)> = Vec::new();
        if let Some(s) = &self.system {
            points.push((s.tokens.len(), true, s.rng));
        }
        if let Some(a) = &self.anchor {
            points.push((a.tokens.len(), false, a.rng));
        }
        points.sort_unstable();
        self.rewind(None)?;
        for (end, is_system, point_rng) in points {
            let done = self.tokens.len();
            self.feed(&tokens[done..end])?;
            self.rng = point_rng;
            let here = Some(self.here()?);
            if is_system {
                self.system = here;
            } else {
                self.anchor = here;
            }
        }
        let done = self.tokens.len();
        self.feed(&tokens[done..fed])?;
        if let Some(id) = pending {
            self.tokens.push(id);
            self.pending = Some(id);
        }
        self.rng = rng;
        Ok(fed)
    }
    /// Prefill the system message alone (see `ChatSession::prime`).
    pub fn prime(&mut self, t: &impl TextCodec) -> Result<usize> {
        if self.last_stop.is_some() || !self.tokens.is_empty() {
            return Ok(0);
        }
        let [m] = self.messages.as_slice() else {
            return Ok(0);
        };
        require(m.role == Role::System, "prime needs only a system message")?;
        let ids = t.encode(&render_system(&m.content))?;
        require(
            ids.len() <= self.capacity(),
            "system message exceeds device capacity",
        )?;
        self.feed(&ids)?;
        self.system = Some(self.here()?);
        Ok(ids.len())
    }
    fn here(&self) -> Result<Anchor> {
        Ok(Anchor {
            tokens: self.tokens.clone(),
            logits: self
                .logits
                .clone()
                .ok_or_else(|| err("anchor without logits"))?,
            rng: self.rng,
            checkpoint: self.run.checkpoint()?,
        })
    }
    /// Feed pending (if any) then `ids` to the device; logits of the last one.
    fn feed(&mut self, ids: &[u32]) -> Result<()> {
        let mut all: Vec<u32> = self.pending.take().into_iter().collect();
        let pending = all.len();
        all.extend_from_slice(ids);
        if all.is_empty() {
            return Ok(());
        }
        let logits = self.advance(&all)?;
        self.logits = Some(logits);
        self.tokens.extend_from_slice(&all[pending..]);
        Ok(())
    }
    /// Chunked prefill of `ids`, except observed positions, which run alone
    /// with captures and go to the observer. Logits of the last ID.
    fn advance(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        let base = self.run.position();
        let mut logits = None;
        let mut start = 0;
        for (i, &id) in ids.iter().enumerate() {
            if self.observer.is_some() && self.observed.contains(&(base + i)) {
                if start < i {
                    self.run.prefill(&ids[start..i])?;
                }
                let step = self.run.step(id, true)?;
                if let Some(f) = self.observer.as_mut() {
                    f(base + i, id, &step);
                }
                logits = Some(step.logits);
                start = i + 1;
            }
        }
        if start < ids.len() {
            logits = self.run.prefill(&ids[start..])?;
        }
        logits.ok_or_else(|| err("advance without tokens"))
    }
    fn rewind(&mut self, to: Option<Anchor>) -> Result<()> {
        match to {
            Some(a) => {
                self.run.restore(&a.checkpoint)?;
                self.tokens = a.tokens;
                self.logits = Some(a.logits);
            }
            None => {
                let zero = self.run.initial_checkpoint();
                self.run.restore(&zero)?;
                self.tokens.clear();
                self.logits = None;
            }
        }
        self.pending = None;
        Ok(())
    }
    pub fn user_turn(
        &mut self,
        t: &impl TextCodec,
        text: &str,
        max_new: usize,
        reuse: bool,
    ) -> Result<Turn> {
        require(max_new > 0, "chat max-new-tokens must be positive")?;
        let mut messages = self.messages.clone();
        messages.push(Message::new(Role::User, text));
        let base = t.encode(&render_chat(&messages, self.thinking, false)?)?;
        let target = t.encode(&render_chat(&messages, self.thinking, true)?)?;
        // Checked before anything changes so a too-long turn keeps the chat.
        require(
            target.len() <= self.capacity(),
            "chat turn exceeds device capacity",
        )?;
        let anchor_point = if target.starts_with(&base) {
            base.len()
        } else {
            0
        };
        let prefix = if anchor_point > 0 {
            &base[..]
        } else {
            &target[..]
        };
        // Longest kept state whose tokens are a prefix of the new prompt:
        // live tokens, the last user anchor, the system prefix, or nothing.
        let fits = |a: &Option<Anchor>| {
            a.as_ref()
                .filter(|a| reuse && prefix.starts_with(&a.tokens))
                .cloned()
        };
        if !(reuse && prefix.starts_with(&self.tokens)) {
            let to = fits(&self.anchor).or_else(|| fits(&self.system));
            self.rewind(to)?;
        }
        let reused = self.tokens.len();
        self.feed(&prefix[reused..])?;
        self.anchor = if anchor_point > 0 {
            Some(self.here()?)
        } else {
            None
        };
        if anchor_point > 0 {
            self.feed(&target[anchor_point..])?;
        }
        self.messages = messages;
        let (generated, stop) = self.generate(max_new)?;
        self.response_ids = generated;
        self.last_stop = Some(stop.clone());
        let answer = self.answer(t)?;
        self.messages
            .push(Message::new(Role::Assistant, answer.clone()));
        Ok(Turn {
            text: answer,
            generated: self.response_ids.clone(),
            stop,
            reused_tokens: reused,
            prefill_tokens: target.len() - reused,
        })
    }
    pub fn continue_turn(&mut self, t: &impl TextCodec, max_new: usize) -> Result<Turn> {
        require(
            self.last_stop == Some(StopReason::MaxTokens) && max_new > 0,
            "only a length-limited turn can continue",
        )?;
        let (generated, stop) = self.generate(max_new)?;
        self.response_ids.extend(&generated);
        self.last_stop = Some(stop.clone());
        let answer = self.answer(t)?;
        self.messages.last_mut().unwrap().content = answer.clone();
        Ok(Turn {
            text: answer,
            generated,
            stop,
            reused_tokens: self.tokens.len(),
            prefill_tokens: 0,
        })
    }
    /// Same loop as `InferenceSession::generate`, bounded by device capacity.
    fn generate(&mut self, max_new: usize) -> Result<(Vec<u32>, StopReason)> {
        require(self.logits.is_some(), "chat has no prompt")?;
        let mut out = Vec::new();
        let mut stop = StopReason::MaxTokens;
        for _ in 0..max_new {
            if let Some(id) = self.pending {
                if self.run.position() >= self.capacity() {
                    stop = StopReason::ContextLimit;
                    break;
                }
                self.logits = Some(self.advance(&[id])?);
                self.pending = None;
            }
            let token = select(self.logits.as_ref().unwrap(), self.sampling, &mut self.rng)?;
            self.tokens.push(token);
            self.pending = Some(token);
            out.push(token);
            if EOS.contains(&token) {
                stop = StopReason::Eos;
                break;
            }
        }
        Ok((out, stop))
    }
    fn answer(&self, t: &impl TextCodec) -> Result<String> {
        let mut ids = self.response_ids.as_slice();
        if ids.last().is_some_and(|v| EOS.contains(v)) {
            ids = &ids[..ids.len() - 1];
        }
        t.decode(ids)
    }
    /// Forget the conversation, keep settings and the system message (and its
    /// primed state). A failed device run is reallocated.
    pub fn reset(&mut self) -> Result<()> {
        if self.run.failed {
            self.run.reset()?;
            self.system = None;
        }
        self.messages.truncate(usize::from(
            self.messages
                .first()
                .is_some_and(|m| m.role == Role::System),
        ));
        self.anchor = None;
        self.response_ids.clear();
        self.last_stop = None;
        self.rng = match self.sampling {
            Sampling::Seeded { seed, .. } => seed,
            _ => 0,
        };
        let system = self.system.clone();
        self.rewind(system)
    }
    /// Host copy in the CPU engine's types (reads all state from the device).
    pub fn to_session(&self) -> Result<ChatSession> {
        let live = self.run.inspect_state()?;
        let session = |state, tokens: &[u32], pending, logits, rng| InferenceSession {
            failed: false,
            model_id: self.model_id.clone(),
            revision: self.revision.clone(),
            backend: self.backend,
            sampling: self.sampling,
            rng,
            state,
            tokens: tokens.to_vec(),
            pending,
            logits,
            edit: self.edit().cloned(),
        };
        let anchor = self.anchor.as_ref().map(|a| {
            session(
                prefix_state(&live, &a.checkpoint),
                &a.tokens,
                None,
                Some(a.logits.clone()),
                a.rng,
            )
        });
        Ok(ChatSession {
            engine: session(
                live,
                &self.tokens,
                self.pending,
                self.logits.clone(),
                self.rng,
            ),
            anchor,
            messages: self.messages.clone(),
            thinking: self.thinking,
            response_ids: self.response_ids.clone(),
            last_stop: self.last_stop.clone(),
        })
    }
    /// Upload a CPU-format chat (from `load_chat` or a CPU `ChatSession`).
    pub fn from_session(model: &Model, mut run: DeviceRun<'a>, c: ChatSession) -> Result<Self> {
        let e = &c.engine;
        require(
            e.model_id == model.identity() && e.revision == model.revision(),
            "chat model mismatch",
        )?;
        require(
            e.tokens.len() < run.capacity(),
            "saved chat does not fit --capacity",
        )?;
        if run.position() > 0 || run.failed {
            run.reset()?;
        }
        run.set_edit(model, e.edit.as_ref())?;
        run.load_state(&e.state)?;
        let checkpoint = |s: &ModelState| DeviceCheckpoint {
            position: s.position,
            delta: s
                .layers
                .iter()
                .map(|l| match l {
                    LayerState::Delta(d) => Some(d.clone()),
                    LayerState::Attention(_) => None,
                })
                .collect(),
        };
        let anchor = match &c.anchor {
            Some(a) => {
                // The anchor reuses the live KV rows, so they must be the same.
                require(
                    e.tokens.starts_with(&a.tokens)
                        && prefix_state(&e.state, &checkpoint(&a.state)) == a.state,
                    "saved anchor is not a prefix of the live state",
                )?;
                Some(Anchor {
                    tokens: a.tokens.clone(),
                    logits: a
                        .logits
                        .clone()
                        .ok_or_else(|| err("anchor without logits"))?,
                    rng: a.rng,
                    checkpoint: checkpoint(&a.state),
                })
            }
            None => None,
        };
        // A primed-only snapshot: the live state is the system prefix.
        let system = (c.last_stop.is_none() && !e.tokens.is_empty() && e.pending.is_none())
            .then(|| -> Result<Anchor> {
                Ok(Anchor {
                    tokens: e.tokens.clone(),
                    logits: e
                        .logits
                        .clone()
                        .ok_or_else(|| err("primed chat without logits"))?,
                    rng: e.rng,
                    checkpoint: checkpoint(&e.state),
                })
            })
            .transpose()?;
        Ok(Self {
            run,
            model_id: e.model_id.clone(),
            revision: e.revision.clone(),
            backend: e.backend,
            sampling: e.sampling,
            rng: e.rng,
            tokens: e.tokens.clone(),
            pending: e.pending,
            logits: e.logits.clone(),
            anchor,
            system,
            messages: c.messages,
            thinking: c.thinking,
            response_ids: c.response_ids,
            last_stop: c.last_stop,
            observer: None,
            observed: BTreeSet::new(),
        })
    }
    pub fn save(&self, model: &Model, path: &Path, limit: u64) -> Result<()> {
        save_chat(model, &self.to_session()?, path, limit)
    }
    pub fn load(
        model: &Model,
        run: DeviceRun<'a>,
        t: &impl TextCodec,
        path: &Path,
        limit: u64,
    ) -> Result<Self> {
        Self::from_session(model, run, load_chat(model, t, path, limit)?)
    }
    /// Give the run back (e.g. to load another snapshot into it).
    pub fn into_run(self) -> DeviceRun<'a> {
        self.run
    }
    /// Load `c` into this chat's run, keeping the observer and its positions.
    /// Checked before the run is touched; a device error during the upload
    /// leaves a fresh, empty chat with the same settings.
    pub fn replace(&mut self, model: &Model, c: ChatSession) -> Result<()> {
        require(
            c.engine.tokens.len() < self.capacity(),
            "saved chat does not fit --capacity",
        )?;
        let device = self.run.model;
        let (capacity, chunk) = (self.run.capacity, self.run.chunk);
        let placeholder = device.new_run_chunked(1, 1)?;
        let run = std::mem::replace(&mut self.run, placeholder);
        match Self::from_session(model, run, c) {
            Ok(mut next) => {
                next.observer = self.observer.take();
                next.observed = std::mem::take(&mut self.observed);
                *self = next;
                Ok(())
            }
            Err(e) => {
                self.run = device.new_run_chunked(capacity, chunk)?;
                self.run.set_edit(model, None)?;
                self.system = None;
                self.tokens.clear();
                self.reset()?;
                Err(e)
            }
        }
    }
}
/// `live` cut back to `at`: KV rows below its position, its DeltaNet copy.
fn prefix_state(live: &ModelState, at: &DeviceCheckpoint) -> ModelState {
    ModelState {
        position: at.position,
        layers: live
            .layers
            .iter()
            .zip(at.delta())
            .map(|(l, d)| match (l, d) {
                (_, Some(d)) => LayerState::Delta(d.clone()),
                (LayerState::Attention(kv), None) => LayerState::Attention(KvCache {
                    keys: kv.keys[..at.position].to_vec(),
                    values: kv.values[..at.position].to_vec(),
                }),
                (l, None) => l.clone(),
            })
            .collect(),
    }
}
