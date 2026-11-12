//! 生成済みと入力済みを区別する連続推論。最後の選択IDはpendingのまま
//! 保存でき、再開時に一度だけモデルへ渡す。RNGもrunの状態の一部。
use crate::{
    edit::EditManifest,
    generate::{select, Generated, Sampling, StopReason},
    model::{Model, StepActivations},
    state::{ModelState, TokenHistory},
    tensor::{require, Backend, Result},
};
type Observer<'a> = Option<&'a mut dyn FnMut(usize, &StepActivations) -> Result<()>>;
#[derive(Debug, Clone, PartialEq)]
pub struct InferenceSession {
    pub(crate) failed: bool,
    pub(crate) model_id: String,
    pub(crate) revision: String,
    pub(crate) backend: Backend,
    pub(crate) sampling: Sampling,
    pub(crate) rng: u64,
    pub(crate) state: ModelState,
    pub(crate) tokens: Vec<u32>,
    pub(crate) pending: Option<u32>,
    pub(crate) logits: Option<Vec<f32>>,
    /// Weight-edit overlay applied to every step (validated in `set_edit`).
    pub(crate) edit: Option<EditManifest>,
}
impl InferenceSession {
    pub fn new(model: &Model, backend: Backend, sampling: Sampling) -> Result<Self> {
        if let Sampling::Seeded {
            temperature, top_k, ..
        } = sampling
        {
            require(
                temperature.is_finite() && temperature > 0. && top_k > 0,
                "invalid sampling parameters",
            )?;
        }
        Ok(Self {
            failed: false,
            model_id: model.identity.clone(),
            revision: model.revision.clone(),
            backend,
            sampling,
            rng: match sampling {
                Sampling::Seeded { seed, .. } => seed,
                _ => 0,
            },
            state: model.new_state(),
            tokens: vec![],
            pending: None,
            logits: None,
            edit: None,
        })
    }
    /// Apply `edit` to every following step (None removes it). Already
    /// computed state is kept; `ChatSession::set_edit` recomputes history.
    pub fn set_edit(&mut self, model: &Model, edit: Option<EditManifest>) -> Result<()> {
        if let Some(e) = &edit {
            let c = &model.config;
            e.validate(
                &model.revision,
                model.blocks.len(),
                c.hidden_size,
                c.intermediate_size,
            )?;
        }
        self.edit = edit;
        Ok(())
    }
    pub fn edit(&self) -> Option<&EditManifest> {
        self.edit.as_ref()
    }
    pub fn state(&self) -> &ModelState {
        &self.state
    }
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }
    pub fn pending(&self) -> Option<u32> {
        self.pending
    }
    pub fn logits(&self) -> Option<&[f32]> {
        self.logits.as_deref()
    }
    pub fn backend(&self) -> Backend {
        self.backend
    }
    fn check_model(&self, model: &Model) -> Result<()> {
        require(
            !self.failed && self.model_id == model.identity && self.revision == model.revision,
            "session failed or model mismatch; reset the session",
        )
    }
    fn ingest(
        &mut self,
        model: &Model,
        id: u32,
        is_pending: bool,
        observer: &mut Observer<'_>,
    ) -> Result<()> {
        let p = self.state.position;
        // 演算が途中で失敗すると一部の層だけ更新された可能性がある。
        // その状態を保存/継続しない。observerの失敗は完成したstepを保持する。
        self.failed = true;
        let (logits, observed) = if let Some(f) = observer {
            let a = model.step(id, &mut self.state, self.backend, self.edit.as_ref())?;
            let result = f(p, &a);
            (a.logits, result)
        } else {
            (
                model.step_compact(id, &mut self.state, self.backend, self.edit.as_ref())?,
                Ok(()),
            )
        };
        self.failed = false;
        self.logits = Some(logits);
        if is_pending {
            self.pending = None;
        } else {
            self.tokens.push(id);
        }
        observed
    }
    pub fn reset(&mut self, model: &Model) -> Result<()> {
        let edit = self.edit.take();
        *self = Self::new(model, self.backend, self.sampling)?;
        self.edit = edit;
        Ok(())
    }
    fn flush(&mut self, model: &Model, observer: &mut Observer<'_>) -> Result<()> {
        if let Some(id) = self.pending {
            self.ingest(model, id, true, observer)?;
        }
        Ok(())
    }
    pub(crate) fn consume_pending(&mut self, model: &Model) -> Result<()> {
        self.check_model(model)?;
        self.flush(model, &mut None)
    }
    pub fn prefill_observed(
        &mut self,
        model: &Model,
        ids: &[u32],
        mut observe: impl FnMut(usize, &StepActivations) -> Result<()>,
    ) -> Result<()> {
        self.prefill_inner(model, ids, &mut Some(&mut observe))
    }
    pub fn prefill(&mut self, model: &Model, ids: &[u32]) -> Result<()> {
        self.prefill_inner(model, ids, &mut None)
    }
    fn prefill_inner(
        &mut self,
        model: &Model,
        ids: &[u32],
        observer: &mut Observer<'_>,
    ) -> Result<()> {
        self.check_model(model)?;
        if ids.is_empty() {
            return Ok(());
        }
        let needed = ids.len().checked_add(usize::from(self.pending.is_some()));
        require(
            ids.iter()
                .all(|&id| (id as usize) < model.config.vocab_size)
                && needed.is_some_and(|n| {
                    n <= model
                        .config
                        .max_position_embeddings
                        .saturating_sub(self.state.position)
                }),
            "session context/ID limit",
        )?;
        self.flush(model, observer)?;
        for &id in ids {
            self.ingest(model, id, false, observer)?;
        }
        Ok(())
    }
    pub fn generate(&mut self, model: &Model, max_new: usize, eos: &[u32]) -> Result<Generated> {
        self.generate_inner(model, max_new, eos, &mut None)
    }
    pub fn generate_observed(
        &mut self,
        model: &Model,
        max_new: usize,
        eos: &[u32],
        mut observer: impl FnMut(usize, &StepActivations) -> Result<()>,
    ) -> Result<Generated> {
        self.generate_inner(model, max_new, eos, &mut Some(&mut observer))
    }
    fn generate_inner(
        &mut self,
        model: &Model,
        max_new: usize,
        eos: &[u32],
        observer: &mut Observer<'_>,
    ) -> Result<Generated> {
        self.check_model(model)?;
        require(self.logits.is_some(), "session has no prompt")?;
        let mut history = TokenHistory {
            prompt: self.tokens.clone(),
            generated: vec![],
        };
        let mut stop = StopReason::MaxTokens;
        for _ in 0..max_new {
            if self.pending.is_some() && self.state.position >= model.config.max_position_embeddings
            {
                stop = StopReason::ContextLimit;
                break;
            }
            self.flush(model, observer)?;
            let token = select(self.logits.as_ref().unwrap(), self.sampling, &mut self.rng)?;
            self.tokens.push(token);
            self.pending = Some(token);
            history.generated.push(token);
            if eos.contains(&token) {
                stop = StopReason::Eos;
                break;
            }
        }
        Ok(Generated { history, stop })
    }
    /// anchorへ戻るときもsamplingの乱数列は現在の続きから使う。
    pub(crate) fn inherit_rng(&mut self, other: &Self) {
        self.rng = other.rng;
    }
}
