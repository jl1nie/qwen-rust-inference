//! 教材用のバイト単位BPE。本書第1章の手順をそのままコードにしたもの。
//!
//! 本番の符号化は `text::TextTokenizer`（tokenizersクレート）が行う。この実装は
//! 同じ `tokenizer.json` を読み、同じIDを出すことをテストで確かめる。
//! 対象は特殊トークン（`<|im_start|>` など）を含まない平文に限る。
//! `tokenizer.json` の正規化（NFC）は省くので、入力はNFCであること。
//!
//! 手順は三つ。
//! 1. 前処理の正規表現で、文字列を区間に分ける。
//! 2. 区間をUTF-8のバイト列にし、1バイトを1断片とする。
//! 3. 隣り合う断片の組のうち、結合規則の順位が最も小さい組を結合する。
//!    結合できる組がなくなったら、各断片を語彙表のIDに置き換える。
use crate::{tensor::Result, weights::err};
use serde_json::Value;
use std::{collections::HashMap, path::Path};

/// 結合を一回行った後の区間の状態。
pub struct MergeStep {
    /// 適用した規則の順位（`merges` の0始まりの添字）。
    pub rank: usize,
    /// 結合後の断片。
    pub pieces: Vec<Vec<u8>>,
}

/// 一つの区間を符号化した記録。
pub struct SegmentTrace {
    pub text: String,
    pub steps: Vec<MergeStep>,
    pub ids: Vec<u32>,
}

pub struct TeachingBpe {
    /// 断片 → ID。断片は1バイトを1文字に写した文字列で表す（`byte_char`）。
    vocab: HashMap<String, u32>,
    /// `"左 右"` → 順位。`tokenizer.json` の `merges` と同じ書き方。
    ranks: HashMap<String, usize>,
    pattern: onig::Regex,
    byte_char: [char; 256],
}

/// GPT-2以来のバイト単位BPEは、各バイトを表示できる1文字に写して語彙表に載せる。
/// 印字できるバイトはそのままの文字、それ以外（空白や制御文字、0x80以上の一部）は
/// U+0100以降の文字へ順に割り当てる。たとえば空白0x20は「Ġ」になる。
fn byte_chars() -> [char; 256] {
    let printable = |b: u32| (0x21..=0x7e).contains(&b) || (0xa1..=0xac).contains(&b) || b >= 0xae;
    let mut table = ['\0'; 256];
    let mut next = 0x100;
    for b in 0..256u32 {
        table[b as usize] = if printable(b) {
            char::from_u32(b).unwrap()
        } else {
            next += 1;
            char::from_u32(next - 1).unwrap()
        };
    }
    table
}

impl TeachingBpe {
    pub fn load(dir: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(dir.join("tokenizer.json")).map_err(err)?;
        Self::from_json(&text)
    }

    pub fn from_json(text: &str) -> Result<Self> {
        let json: Value = serde_json::from_str(text).map_err(err)?;
        let model = &json["model"];
        if model["type"] != "BPE" {
            return Err(err("not a BPE tokenizer"));
        }
        let vocab = model["vocab"]
            .as_object()
            .ok_or_else(|| err("missing vocab"))?
            .iter()
            .map(|(piece, id)| {
                Ok((
                    piece.clone(),
                    id.as_u64().ok_or_else(|| err("bad id"))? as u32,
                ))
            })
            .collect::<Result<_>>()?;
        let ranks = model["merges"]
            .as_array()
            .ok_or_else(|| err("missing merges"))?
            .iter()
            .enumerate()
            .map(|(rank, rule)| {
                Ok((
                    rule.as_str().ok_or_else(|| err("bad merge"))?.to_owned(),
                    rank,
                ))
            })
            .collect::<Result<_>>()?;
        // 前処理は Split（正規表現）と ByteLevel の並び。正規表現だけを取り出す。
        let regex = json["pre_tokenizer"]["pretokenizers"]
            .as_array()
            .and_then(|steps| steps.iter().find(|s| s["type"] == "Split"))
            .and_then(|split| split["pattern"]["Regex"].as_str())
            .ok_or_else(|| err("missing pre-tokenizer regex"))?;
        let pattern = onig::Regex::new(regex).map_err(err)?;
        Ok(Self {
            vocab,
            ranks,
            pattern,
            byte_char: byte_chars(),
        })
    }

    /// 前処理：正規表現に一致した部分と、一致しなかった間の部分を、それぞれ区間にする。
    pub fn segments<'a>(&self, text: &'a str) -> Vec<&'a str> {
        let mut out = Vec::new();
        let mut last = 0;
        for (start, end) in self.pattern.find_iter(text) {
            if start > last {
                out.push(&text[last..start]);
            }
            if end > start {
                out.push(&text[start..end]);
            }
            last = end;
        }
        if last < text.len() {
            out.push(&text[last..]);
        }
        out
    }

    fn piece(&self, bytes: &[u8]) -> String {
        bytes.iter().map(|&b| self.byte_char[b as usize]).collect()
    }

    /// 一つの区間を結合し、結合の各段を記録する。
    pub fn trace_segment(&self, segment: &str) -> Result<SegmentTrace> {
        let mut pieces: Vec<Vec<u8>> = segment.bytes().map(|b| vec![b]).collect();
        let mut steps = Vec::new();
        loop {
            // 隣り合う組すべての順位を調べ、最も小さい順位の組を選ぶ（同順位なら左）。
            let best = (0..pieces.len().saturating_sub(1))
                .filter_map(|i| {
                    let rule = format!("{} {}", self.piece(&pieces[i]), self.piece(&pieces[i + 1]));
                    self.ranks.get(&rule).map(|&rank| (rank, i))
                })
                .min();
            let Some((rank, i)) = best else { break };
            let right = pieces.remove(i + 1);
            pieces[i].extend(right);
            steps.push(MergeStep {
                rank,
                pieces: pieces.clone(),
            });
        }
        let ids = pieces
            .iter()
            .map(|p| {
                let piece = self.piece(p);
                self.vocab
                    .get(&piece)
                    .copied()
                    .ok_or_else(|| err(format!("piece not in vocab: {piece}")))
            })
            .collect::<Result<_>>()?;
        Ok(SegmentTrace {
            text: segment.to_owned(),
            steps,
            ids,
        })
    }

    pub fn trace(&self, text: &str) -> Result<Vec<SegmentTrace>> {
        self.segments(text)
            .into_iter()
            .map(|s| self.trace_segment(s))
            .collect()
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.trace(text)?.into_iter().flat_map(|s| s.ids).collect())
    }
}

/// 断片を表示用の文字列にする。UTF-8として読めなければ16進のバイト列にする。
pub fn show_piece(bytes: &[u8]) -> String {
    match std::str::from_utf8(bytes) {
        Ok(s) => s.to_owned(),
        Err(_) => bytes
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<Vec<_>>()
            .join(" "),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 本書1.3節の小さな例：規則は「b c」「a b」「a bc」の順。
    const TOY: &str = r#"{
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": "\\S+"}, "behavior": "Isolated", "invert": false},
            {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}
        ]},
        "model": {"type": "BPE",
            "vocab": {"a": 0, "b": 1, "c": 2, "bc": 3, "ab": 4, "abc": 5},
            "merges": ["b c", "a b", "a bc"]}
    }"#;

    #[test]
    fn toy_example_merges_by_rank() {
        let bpe = TeachingBpe::from_json(TOY).unwrap();
        let t = bpe.trace_segment("ababc").unwrap();
        let shown: Vec<Vec<String>> = t
            .steps
            .iter()
            .map(|s| s.pieces.iter().map(|p| show_piece(p)).collect())
            .collect();
        assert_eq!(
            t.steps.iter().map(|s| s.rank).collect::<Vec<_>>(),
            [0, 1, 2]
        );
        assert_eq!(
            shown,
            [
                vec!["a", "b", "a", "bc"],
                vec!["ab", "a", "bc"],
                vec!["ab", "abc"]
            ]
        );
        assert_eq!(t.ids, [4, 5]);
        assert_eq!(bpe.encode("abc").unwrap(), [5]);
    }

    #[test]
    fn segments_keep_unmatched_gaps() {
        let bpe = TeachingBpe::from_json(TOY).unwrap();
        assert_eq!(bpe.segments("ab  abc"), ["ab", "  ", "abc"]);
    }

    #[test]
    fn byte_table_is_a_bijection_with_known_points() {
        let t = byte_chars();
        assert_eq!(t[b'a' as usize], 'a');
        assert_eq!(t[b' ' as usize], 'Ġ');
        assert_eq!(t[b'\n' as usize], 'Ċ');
        let mut seen = std::collections::HashSet::new();
        assert!(t.iter().all(|c| seen.insert(*c)));
    }

    #[test]
    fn missing_piece_is_an_error() {
        let bpe = TeachingBpe::from_json(TOY).unwrap();
        assert!(bpe.encode("d").is_err());
    }
}
