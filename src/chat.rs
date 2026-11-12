//! 会話テンプレートとKVの境界を明示する。前回の思考prefixをそのまま
//! 追記すると公式テンプレートと別のID列になるので、user末尾のanchorを使う。
use crate::{
    edit::EditManifest,
    generate::{Sampling, StopReason},
    model::Model,
    session::InferenceSession,
    tensor::{require, Backend, Result},
    text::{render_chat, render_system, Message, Role, TextCodec},
};
use serde::Serialize;
#[derive(Debug, Clone, PartialEq)]
pub struct ChatSession {
    pub(crate) engine: InferenceSession,
    pub(crate) anchor: Option<InferenceSession>,
    pub(crate) messages: Vec<Message>,
    pub(crate) thinking: bool,
    pub(crate) response_ids: Vec<u32>,
    pub(crate) last_stop: Option<StopReason>,
}
#[derive(Debug, Serialize)]
pub struct Turn {
    pub text: String,
    pub generated: Vec<u32>,
    pub stop: StopReason,
    pub reused_tokens: usize,
    pub prefill_tokens: usize,
}
impl ChatSession {
    pub fn new(
        model: &Model,
        backend: Backend,
        sampling: Sampling,
        thinking: bool,
        system: Option<String>,
    ) -> Result<Self> {
        Ok(Self {
            engine: InferenceSession::new(model, backend, sampling)?,
            anchor: None,
            messages: system
                .into_iter()
                .map(|s| Message::new(Role::System, s))
                .collect(),
            thinking,
            response_ids: vec![],
            last_stop: None,
        })
    }
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }
    pub fn engine(&self) -> &InferenceSession {
        &self.engine
    }
    /// Prefill the system message alone so the next turns (and a saved
    /// snapshot) reuse it: a long system document is computed once. Returns
    /// the number of prefilled tokens (0 without a system message or when
    /// the conversation already started).
    pub fn prime(&mut self, model: &Model, tokenizer: &impl TextCodec) -> Result<usize> {
        let Some(ids) = self.system_prefix(tokenizer)? else {
            return Ok(0);
        };
        let mut proposed = self.engine.clone();
        proposed.prefill(model, &ids)?;
        self.engine = proposed;
        Ok(ids.len())
    }
    /// System-prefix IDs when nothing else has been computed yet.
    pub(crate) fn system_prefix(&self, tokenizer: &impl TextCodec) -> Result<Option<Vec<u32>>> {
        if self.last_stop.is_some() || !self.engine.tokens().is_empty() {
            return Ok(None);
        }
        match self.messages.as_slice() {
            [m] if m.role == Role::System => {
                Ok(Some(tokenizer.encode(&render_system(&m.content))?))
            }
            _ => Ok(None),
        }
    }
    pub fn user_turn(
        &mut self,
        model: &Model,
        tokenizer: &impl TextCodec,
        text: &str,
        max_new: usize,
    ) -> Result<Turn> {
        self.user_turn_with_cache(model, tokenizer, text, max_new, true)
    }
    pub fn user_turn_with_cache(
        &mut self,
        model: &Model,
        tokenizer: &impl TextCodec,
        text: &str,
        max_new: usize,
        reuse: bool,
    ) -> Result<Turn> {
        // 途中でcontext超過/encode/演算が失敗しても、元の会話を残す。
        let mut proposed = self.clone();
        let result = proposed.user_inner(model, tokenizer, text, max_new, reuse)?;
        *self = proposed;
        Ok(result)
    }
    fn user_inner(
        &mut self,
        model: &Model,
        tokenizer: &impl TextCodec,
        text: &str,
        max_new: usize,
        reuse: bool,
    ) -> Result<Turn> {
        require(max_new > 0, "chat max-new-tokens must be positive")?;
        self.messages.push(Message::new(Role::User, text));
        let base = tokenizer.encode(&render_chat(&self.messages, self.thinking, false)?)?;
        let target = tokenizer.encode(&render_chat(&self.messages, self.thinking, true)?)?;
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
        let mut candidate = if reuse && prefix.starts_with(self.engine.tokens()) {
            self.engine.clone()
        } else if let Some(a) = self
            .anchor
            .as_ref()
            .filter(|a| reuse && prefix.starts_with(a.tokens()))
        {
            a.clone()
        } else {
            let mut fresh =
                InferenceSession::new(model, self.engine.backend, self.engine.sampling)?;
            fresh.edit = self.engine.edit.clone();
            fresh
        };
        candidate.inherit_rng(&self.engine);
        let reused = candidate.tokens().len();
        candidate.prefill(model, &prefix[reused..])?;
        candidate.consume_pending(model)?;
        self.anchor = if anchor_point > 0 {
            Some(candidate.clone())
        } else {
            None
        };
        if anchor_point > 0 {
            candidate.prefill(model, &target[anchor_point..])?;
        }
        self.engine = candidate;
        let generated = self.engine.generate(model, max_new, &[248044, 248046])?;
        self.response_ids = generated.history.generated;
        self.last_stop = Some(generated.stop.clone());
        let answer = self.answer(tokenizer)?;
        self.messages
            .push(Message::new(Role::Assistant, answer.clone()));
        Ok(Turn {
            text: answer,
            generated: self.response_ids.clone(),
            stop: generated.stop,
            reused_tokens: reused,
            prefill_tokens: target.len() - reused,
        })
    }
    fn answer(&self, t: &impl TextCodec) -> Result<String> {
        let mut ids = self.response_ids.as_slice();
        if ids.last().is_some_and(|v| [248044, 248046].contains(v)) {
            ids = &ids[..ids.len() - 1];
        }
        t.decode(ids)
    }
    pub fn continue_turn(
        &mut self,
        model: &Model,
        t: &impl TextCodec,
        max_new: usize,
    ) -> Result<Turn> {
        require(
            self.last_stop == Some(StopReason::MaxTokens) && max_new > 0,
            "only a length-limited turn can continue",
        )?;
        let mut proposed = self.clone();
        let generated = proposed
            .engine
            .generate(model, max_new, &[248044, 248046])?;
        proposed.response_ids.extend(&generated.history.generated);
        proposed.last_stop = Some(generated.stop.clone());
        let answer = proposed.answer(t)?;
        proposed.messages.last_mut().unwrap().content = answer.clone();
        let result = Turn {
            text: answer,
            generated: generated.history.generated,
            stop: generated.stop,
            reused_tokens: proposed.engine.tokens().len(),
            prefill_tokens: 0,
        };
        *self = proposed;
        Ok(result)
    }
    /// Change the weight edit mid-conversation. Every kept state (live and
    /// anchor) was computed under the old edit, so the same token history is
    /// recomputed under the new one: messages, pending token, RNG and the
    /// reuse points stay, only the numbers change.
    pub fn set_edit(&mut self, model: &Model, edit: Option<EditManifest>) -> Result<()> {
        let mut fresh = InferenceSession::new(model, self.engine.backend, self.engine.sampling)?;
        fresh.set_edit(model, edit)?;
        fresh.rng = self.engine.rng;
        let fed = self.engine.tokens.len() - usize::from(self.engine.pending.is_some());
        let mut anchor = None;
        if let Some(a) = &self.anchor {
            fresh.prefill(model, &a.tokens)?;
            let mut copy = fresh.clone();
            copy.rng = a.rng;
            anchor = Some(copy);
        }
        let done = fresh.tokens.len();
        fresh.prefill(model, &self.engine.tokens[done..fed])?;
        if let Some(id) = self.engine.pending {
            fresh.tokens.push(id);
            fresh.pending = Some(id);
        }
        self.engine = fresh;
        self.anchor = anchor;
        Ok(())
    }
    pub fn reset(&mut self, model: &Model) -> Result<()> {
        let edit = self.engine.edit.clone();
        let system = self
            .messages
            .first()
            .filter(|m| m.role == Role::System)
            .map(|m| m.content.clone());
        *self = Self::new(
            model,
            self.engine.backend,
            self.engine.sampling,
            self.thinking,
            system,
        )?;
        self.engine.edit = edit;
        Ok(())
    }
}
