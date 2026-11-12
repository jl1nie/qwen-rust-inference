//! Text-only codec and fixed single/multi-turn chat templates. No tool or
//! multimodal routing. Rendered strings are compared to pinned Jinja fixtures.
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
/// The system message exactly as `render_chat` begins every later turn. It
/// does not depend on later messages, so its state can be computed once and
/// reused as a cached prefix (`ChatSession::prime`).
pub fn render_system(content: &str) -> String {
    format!("<|im_start|>system\n{}<|im_end|>\n", content.trim())
}
pub struct TextTokenizer {
    tokenizer: tokenizers::Tokenizer,
    pub eos: u32,
}
/// 会話と状態のテストを巨大なtokenizerファイルから独立させる境界。
/// 本番では常に固定tokenizer.jsonを解釈するTextTokenizerを使う。
pub trait TextCodec {
    fn encode(&self, text: &str) -> Result<Vec<u32>>;
    fn decode(&self, ids: &[u32]) -> Result<String>;
}
impl TextCodec for TextTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        TextTokenizer::encode(self, text)
    }
    fn decode(&self, ids: &[u32]) -> Result<String> {
        TextTokenizer::decode(self, ids)
    }
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

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Message {
    pub role: Role,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}
impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            reasoning_content: None,
        }
    }
}
/// 固定Jinjaのtext-only会話分岐。過去assistantの推論部分を省くため、
/// 次ターンでは前回の入力ID列がそのまま接頭辞になるとは限らない。
pub fn render_chat(
    messages: &[Message],
    thinking: bool,
    add_generation_prompt: bool,
) -> Result<String> {
    require(!messages.is_empty(), "empty chat")?;
    let first = usize::from(messages[0].role == Role::System);
    require(first < messages.len(), "chat needs a user query")?;
    for (i, m) in messages.iter().enumerate().skip(first) {
        require(
            m.role
                == if (i - first) % 2 == 0 {
                    Role::User
                } else {
                    Role::Assistant
                },
            "chat roles must alternate user/assistant",
        )?;
        require(
            !(m.role == Role::User
                && m.content.trim().starts_with("<tool_response>")
                && m.content.trim().ends_with("</tool_response>")),
            "tool-response messages are unsupported",
        )?;
    }
    if add_generation_prompt {
        require(
            messages.last().unwrap().role == Role::User,
            "generation prompt needs a final user message",
        )?;
    }
    let last_query = messages.iter().rposition(|m| m.role == Role::User).unwrap();
    let mut out = String::new();
    for (i, m) in messages.iter().enumerate() {
        let content = m.content.trim();
        match m.role {
            Role::System => {
                require(i == 0, "system must be first")?;
                out.push_str(&render_system(content));
            }
            Role::User => out.push_str(&format!("<|im_start|>user\n{content}<|im_end|>\n")),
            Role::Assistant => {
                let (reason, answer) = if let Some(reason) = &m.reasoning_content {
                    (reason.trim(), content)
                } else if let Some((before, _)) = content.split_once("</think>") {
                    (
                        before.rsplit("<think>").next().unwrap().trim(),
                        content
                            .rsplit("</think>")
                            .next()
                            .unwrap()
                            .trim_start_matches('\n'),
                    )
                } else {
                    ("", content)
                };
                out.push_str("<|im_start|>assistant\n");
                if i > last_query {
                    out.push_str(&format!("<think>\n{reason}\n</think>\n\n"));
                }
                out.push_str(answer);
                out.push_str("<|im_end|>\n");
            }
        }
    }
    if add_generation_prompt {
        out.push_str("<|im_start|>assistant\n");
        out.push_str(if thinking {
            "<think>\n"
        } else {
            "<think>\n\n</think>\n\n"
        });
    }
    Ok(out)
}
