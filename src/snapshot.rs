//! version + 制限付きJSON header + little-endian f32 payload + SHA-256。
//! checksumは破損検出であり暗号化/認証ではない。保存は明示操作に限定し、
//! 会話履歴・状態を外部へ送信しない。既存ファイルはatomic publishでも上書きしない。
use crate::{
    benchmark::state_bytes,
    chat::ChatSession,
    generate::{Sampling, StopReason},
    model::Model,
    session::InferenceSession,
    state::LayerState,
    tensor::{finite, require, Backend, Result},
    text::{render_chat, Message, Role, TextCodec},
    weights::err,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};
const MAGIC: &[u8; 8] = b"QWENRS01";
pub const DEFAULT_LIMIT: u64 = 256 * 1024 * 1024;
const HEADER_LIMIT: usize = 8 * 1024 * 1024;
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SessionHeader {
    position: usize,
    tokens: Vec<u32>,
    pending: Option<u32>,
    logits: bool,
    rng: u64,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatHeader {
    messages: Vec<Message>,
    thinking: bool,
    response_ids: Vec<u32>,
    last_stop: Option<StopReason>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Header {
    version: u32,
    model_identity: String,
    revision: String,
    backend: Backend,
    sampling: Sampling,
    live: SessionHeader,
    anchor: Option<SessionHeader>,
    chat: Option<ChatHeader>,
}
fn info(s: &InferenceSession) -> SessionHeader {
    SessionHeader {
        position: s.state.position,
        tokens: s.tokens.clone(),
        pending: s.pending,
        logits: s.logits.is_some(),
        rng: s.rng,
    }
}
fn header(s: &InferenceSession) -> Header {
    Header {
        version: 1,
        model_identity: s.model_id.clone(),
        revision: s.revision.clone(),
        backend: s.backend,
        sampling: s.sampling,
        live: info(s),
        anchor: None,
        chat: None,
    }
}
fn expected_bytes(m: &Model, h: &SessionHeader) -> Result<usize> {
    require(
        h.position <= m.config.max_position_embeddings
            && h.position.checked_add(usize::from(h.pending.is_some())) == Some(h.tokens.len())
            && h.tokens.iter().all(|&i| (i as usize) < m.config.vocab_size),
        "snapshot token/position mismatch",
    )?;
    require(
        h.pending
            .is_none_or(|id| h.position > 0 && h.tokens.last() == Some(&id))
            && h.logits == (h.position > 0),
        "snapshot pending/logits mismatch",
    )?;
    let mut n = 0usize;
    for b in &m.blocks {
        let floats = match &b.mixer {
            crate::block::MixerWeights::Delta(w) => {
                let c = &w.config;
                w.qkv.rows().checked_mul(c.kernel - 1).and_then(|a| {
                    c.value_heads
                        .checked_mul(c.key_dim)
                        .and_then(|b| b.checked_mul(c.value_dim))
                        .and_then(|b| a.checked_add(b))
                })
            }
            crate::block::MixerWeights::Attention(w) => h
                .position
                .checked_mul(w.config.kv_heads)
                .and_then(|n| n.checked_mul(w.config.head_dim))
                .and_then(|n| n.checked_mul(2)),
        }
        .ok_or_else(|| err("snapshot size overflow"))?;
        n = n
            .checked_add(floats)
            .ok_or_else(|| err("snapshot size overflow"))?;
    }
    if h.logits {
        n = n
            .checked_add(m.config.vocab_size)
            .ok_or_else(|| err("snapshot size overflow"))?;
    }
    n.checked_mul(4)
        .ok_or_else(|| err("snapshot size overflow"))
}
fn validate_session(m: &Model, s: &InferenceSession) -> Result<()> {
    require(
        !s.failed
            && s.model_id == m.identity
            && s.revision == m.revision
            && s.state.layers.len() == m.blocks.len(),
        "snapshot model mismatch",
    )?;
    let expected = expected_bytes(m, &info(s))?;
    require(
        state_bytes(&s.state) + s.logits.as_ref().map_or(0, |v| v.len() * 4) == expected,
        "snapshot state shape mismatch",
    )?;
    let initial = m.new_state();
    for (a, b) in s.state.layers.iter().zip(initial.layers) {
        match (a, b) {
            (LayerState::Delta(a), LayerState::Delta(b)) => {
                require(
                    a.convolution.len() == b.convolution.len()
                        && a.recurrent.len() == b.recurrent.len(),
                    "snapshot Delta shape",
                )?;
                if !a.convolution.is_empty() {
                    finite(&a.convolution)?;
                }
                finite(&a.recurrent)?;
            }
            (LayerState::Attention(a), LayerState::Attention(_)) => {
                require(
                    a.keys.len() == s.state.position && a.values.len() == s.state.position,
                    "snapshot KV positions",
                )?;
            }
            _ => return Err(err("snapshot layer type mismatch")),
        }
    }
    if let Some(v) = &s.logits {
        require(v.len() == m.config.vocab_size, "snapshot logits shape")?;
        finite(v)?;
    }
    // Attention row dimensions are checked separately to reject same-total-size
    // rearrangements of keys/values, not merely a matching total byte count.
    for (b, s) in m.blocks.iter().zip(&s.state.layers) {
        if let (crate::block::MixerWeights::Attention(w), LayerState::Attention(s)) = (&b.mixer, s)
        {
            for v in s.keys.iter().chain(&s.values) {
                require(
                    v.len() == w.config.kv_heads * w.config.head_dim,
                    "snapshot KV row shape",
                )?;
                finite(v)?;
            }
        }
    }
    Ok(())
}
fn write_part(w: &mut impl Write, hash: &mut Sha256, bytes: &[u8]) -> Result<()> {
    w.write_all(bytes).map_err(err)?;
    hash.update(bytes);
    Ok(())
}
fn write_values(w: &mut impl Write, hash: &mut Sha256, v: &[f32]) -> Result<()> {
    for chunk in v.chunks(4096) {
        let bytes: Vec<_> = chunk
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect();
        write_part(w, hash, &bytes)?;
    }
    Ok(())
}
fn write_state(w: &mut impl Write, hash: &mut Sha256, s: &InferenceSession) -> Result<()> {
    for layer in &s.state.layers {
        match layer {
            LayerState::Delta(s) => {
                write_values(w, hash, &s.convolution)?;
                write_values(w, hash, &s.recurrent)?;
            }
            LayerState::Attention(s) => {
                for v in s.keys.iter().chain(&s.values) {
                    write_values(w, hash, v)?;
                }
            }
        }
    }
    if let Some(v) = &s.logits {
        write_values(w, hash, v)?;
    }
    Ok(())
}
fn save(
    m: &Model,
    s: &InferenceSession,
    anchor: Option<&InferenceSession>,
    h: Header,
    path: &Path,
    limit: u64,
) -> Result<()> {
    validate_session(m, s)?;
    if let Some(a) = anchor {
        validate_session(m, a)?;
        require(
            a.backend == s.backend && a.sampling == s.sampling,
            "anchor settings mismatch",
        )?;
    }
    let bytes = serde_json::to_vec(&h).map_err(err)?;
    require(bytes.len() <= HEADER_LIMIT, "snapshot header limit")?;
    let payload = expected_bytes(m, &h.live)?
        .checked_add(
            h.anchor
                .as_ref()
                .map(|a| expected_bytes(m, a))
                .transpose()?
                .unwrap_or(0),
        )
        .ok_or_else(|| err("snapshot size overflow"))?;
    require(
        (payload as u64)
            .checked_add(bytes.len() as u64 + 56)
            .is_some_and(|n| n <= limit),
        "snapshot byte limit",
    )?;
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let name = path
        .file_name()
        .ok_or_else(|| err("invalid snapshot path"))?
        .to_string_lossy();
    let temp: PathBuf = parent.join(format!(
        ".{name}.{}.{}.tmp",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(err)?
            .as_nanos()
    ));
    let result = (|| {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let file = options.open(&temp).map_err(err)?;
        let mut writer = BufWriter::new(file);
        let mut hash = Sha256::new();
        write_part(&mut writer, &mut hash, MAGIC)?;
        write_part(&mut writer, &mut hash, &(bytes.len() as u64).to_le_bytes())?;
        write_part(&mut writer, &mut hash, &bytes)?;
        write_part(&mut writer, &mut hash, &(payload as u64).to_le_bytes())?;
        write_state(&mut writer, &mut hash, s)?;
        if let Some(a) = anchor {
            write_state(&mut writer, &mut hash, a)?;
        }
        writer.write_all(&hash.finalize()).map_err(err)?;
        writer.flush().map_err(err)?;
        writer.get_ref().sync_all().map_err(err)?;
        drop(writer);
        // hard_link publishes only if destination does not exist, without rename's
        // overwrite race. The temporary file is on the same filesystem.
        fs::hard_link(&temp, path).map_err(err)?;
        Ok(())
    })();
    let _ = fs::remove_file(&temp);
    result
}
pub fn save_session(m: &Model, s: &InferenceSession, path: &Path, limit: u64) -> Result<()> {
    save(m, s, None, header(s), path, limit)
}
pub fn save_chat(m: &Model, c: &ChatSession, path: &Path, limit: u64) -> Result<()> {
    let mut h = header(&c.engine);
    h.anchor = c.anchor.as_ref().map(info);
    h.chat = Some(ChatHeader {
        messages: c.messages.clone(),
        thinking: c.thinking,
        response_ids: c.response_ids.clone(),
        last_stop: c.last_stop.clone(),
    });
    save(m, &c.engine, c.anchor.as_ref(), h, path, limit)
}
struct Reader {
    file: BufReader<File>,
    hash: Sha256,
}
impl Reader {
    fn bytes(&mut self, n: usize) -> Result<Vec<u8>> {
        let mut v = vec![0; n];
        self.file.read_exact(&mut v).map_err(err)?;
        self.hash.update(&v);
        Ok(v)
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
    }
    fn values(&mut self, v: &mut [f32]) -> Result<()> {
        for chunk in v.chunks_mut(4096) {
            let bytes = self.bytes(chunk.len() * 4)?;
            for (a, b) in chunk.iter_mut().zip(bytes.chunks_exact(4)) {
                *a = f32::from_le_bytes(b.try_into().unwrap());
            }
        }
        if !v.is_empty() {
            finite(v)?;
        }
        Ok(())
    }
}
fn read_state(
    r: &mut Reader,
    m: &Model,
    h: &SessionHeader,
    settings: &Header,
) -> Result<InferenceSession> {
    let mut session = InferenceSession::new(m, settings.backend, settings.sampling)?;
    session.state.position = h.position;
    for (layer, w) in session.state.layers.iter_mut().zip(&m.blocks) {
        match (layer, &w.mixer) {
            (LayerState::Delta(s), _) => {
                r.values(&mut s.convolution)?;
                r.values(&mut s.recurrent)?;
            }
            (LayerState::Attention(s), crate::block::MixerWeights::Attention(w)) => {
                let dim = w.config.kv_heads * w.config.head_dim;
                for _ in 0..h.position {
                    let mut row = vec![0.; dim];
                    r.values(&mut row)?;
                    s.keys.push(row);
                }
                for _ in 0..h.position {
                    let mut row = vec![0.; dim];
                    r.values(&mut row)?;
                    s.values.push(row);
                }
            }
            _ => return Err(err("snapshot model layout mismatch")),
        }
    }
    session.logits = if h.logits {
        let mut v = vec![0.; m.config.vocab_size];
        r.values(&mut v)?;
        Some(v)
    } else {
        None
    };
    session.tokens = h.tokens.clone();
    session.pending = h.pending;
    session.rng = h.rng;
    validate_session(m, &session)?;
    Ok(session)
}
fn load(
    m: &Model,
    path: &Path,
    limit: u64,
) -> Result<(Header, InferenceSession, Option<InferenceSession>)> {
    let file = File::open(path).map_err(err)?;
    let length = file.metadata().map_err(err)?.len();
    require(length <= limit && length >= 56, "snapshot file size limit")?;
    let mut reader = Reader {
        file: BufReader::new(file),
        hash: Sha256::new(),
    };
    require(reader.bytes(8)? == MAGIC, "snapshot magic/version mismatch")?;
    let n = usize::try_from(reader.u64()?).map_err(err)?;
    require(
        n <= HEADER_LIMIT && (n as u64) <= length - 56,
        "snapshot header length",
    )?;
    let h: Header = serde_json::from_slice(&reader.bytes(n)?).map_err(err)?;
    require(
        h.version == 1 && h.model_identity == m.identity && h.revision == m.revision,
        "snapshot model/version mismatch",
    )?;
    let expected = expected_bytes(m, &h.live)?
        .checked_add(
            h.anchor
                .as_ref()
                .map(|a| expected_bytes(m, a))
                .transpose()?
                .unwrap_or(0),
        )
        .ok_or_else(|| err("snapshot size overflow"))?;
    require(
        reader.u64()? == expected as u64 && expected as u64 + n as u64 + 56 == length,
        "snapshot payload length mismatch",
    )?;
    let live = read_state(&mut reader, m, &h.live, &h)?;
    let anchor = h
        .anchor
        .as_ref()
        .map(|a| read_state(&mut reader, m, a, &h))
        .transpose()?;
    let expected_hash = reader.hash.finalize();
    let mut actual_hash = [0; 32];
    reader.file.read_exact(&mut actual_hash).map_err(err)?;
    require(
        expected_hash.as_slice() == actual_hash,
        "snapshot checksum mismatch",
    )?;
    let mut extra = [0];
    require(
        reader.file.read(&mut extra).map_err(err)? == 0,
        "snapshot trailing bytes",
    )?;
    Ok((h, live, anchor))
}
pub fn load_session(m: &Model, path: &Path, limit: u64) -> Result<InferenceSession> {
    let (h, s, a) = load(m, path, limit)?;
    require(
        h.chat.is_none() && a.is_none(),
        "expected raw session snapshot",
    )?;
    Ok(s)
}
pub fn load_chat(m: &Model, t: &impl TextCodec, path: &Path, limit: u64) -> Result<ChatSession> {
    let (h, engine, anchor) = load(m, path, limit)?;
    let c = h.chat.ok_or_else(|| err("expected chat snapshot"))?;
    if c.last_stop.is_some() {
        require(
            c.messages.last().is_some_and(|m| m.role == Role::Assistant)
                && !c.response_ids.is_empty(),
            "snapshot chat history mismatch",
        )?;
        let prefix = &c.messages[..c.messages.len() - 1];
        let mut target = t.encode(&render_chat(prefix, c.thinking, true)?)?;
        target.extend(&c.response_ids);
        require(
            target == engine.tokens,
            "snapshot chat token history mismatch",
        )?;
        let mut visible = c.response_ids.as_slice();
        if visible
            .last()
            .is_some_and(|id| [248044, 248046].contains(id))
        {
            visible = &visible[..visible.len() - 1];
        }
        require(
            t.decode(visible)? == c.messages.last().unwrap().content,
            "snapshot response text mismatch",
        )?;
        if let Some(a) = &anchor {
            require(
                a.pending.is_none()
                    && a.tokens == t.encode(&render_chat(prefix, c.thinking, false)?)?,
                "snapshot anchor history mismatch",
            )?;
        }
        require(
            c.last_stop != Some(StopReason::Eos)
                || c.response_ids
                    .last()
                    .is_some_and(|id| [248044, 248046].contains(id)),
            "snapshot EOS mismatch",
        )?;
    } else {
        require(
            c.messages.is_empty() || (c.messages.len() == 1 && c.messages[0].role == Role::System),
            "snapshot empty chat mismatch",
        )?;
        require(
            engine.tokens.is_empty() && anchor.is_none() && c.response_ids.is_empty(),
            "snapshot unused chat state",
        )?;
    }
    Ok(ChatSession {
        engine,
        anchor,
        messages: c.messages,
        thinking: c.thinking,
        response_ids: c.response_ids,
        last_stop: c.last_stop,
    })
}
