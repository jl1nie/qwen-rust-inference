//! Deliberately narrow chat surface: a single text user message, no tool or
//! multimodal routing. The resulting string is fixture-compared to pinned Jinja.
use crate::{
    tensor::{require, Result},
    weights::err,
};
use sha2::{Digest, Sha256};
use std::path::Path;
pub fn user_chat(prompt: &str, thinking: bool) -> String {
    let tail = if thinking {
        "<think>\n"
    } else {
        "<think>\n\n</think>\n\n"
    };
    format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n{tail}",
        prompt.trim()
    )
}
pub struct TextTokenizer {
    tokenizer: tokenizers::Tokenizer,
    pub eos: u32,
}
impl TextTokenizer {
    pub fn load(dir: &Path) -> Result<Self> {
        let config = std::fs::read(dir.join("tokenizer_config.json")).map_err(err)?;
        let template = std::fs::read(dir.join("chat_template.jinja")).map_err(err)?;
        require(
            format!("{:x}", Sha256::digest(config))
                == "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c",
            "tokenizer config changed",
        )?;
        require(
            format!("{:x}", Sha256::digest(template))
                == "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
            "chat template changed",
        )?;
        let bytes = std::fs::read(dir.join("tokenizer.json")).map_err(err)?;
        require(
            format!("{:x}", Sha256::digest(&bytes))
                == "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
            "tokenizer changed",
        )?;
        let tokenizer = tokenizers::Tokenizer::from_bytes(&bytes).map_err(err)?;
        let eos = tokenizer
            .token_to_id("<|im_end|>")
            .ok_or_else(|| err("missing EOS"))?;
        require(eos == 248046, "unexpected EOS ID")?;
        Ok(Self { tokenizer, eos })
    }
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(text, false)
            .map_err(err)?
            .get_ids()
            .to_vec())
    }
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer.decode(ids, false).map_err(err)
    }
}
