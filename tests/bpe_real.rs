//! 教材用BPE（src/bpe.rs）が、本番のTextTokenizer（tokenizersクレート）と同じIDを出すことを、
//! 実モデルの tokenizer.json で確かめる。対象は特殊トークンを含まないNFCの平文。
use qwen_rust_inference::{bpe::TeachingBpe, text::TextTokenizer};
use std::path::PathBuf;

const SAMPLES: &[&str] = &[
    "日本の首都はどこですか？一言で。",
    "東京です。",
    "user\n日本の首都はどこですか？一言で。",
    "Hello, world!",
    "It's a test. They'll see what we've done, isn't it?",
    "The quick brown fox jumps over the lazy dog.",
    "  leading spaces and trailing spaces   ",
    "line one\nline two\n\n\nline five\r\nwindows line",
    "tab\tseparated\tvalues",
    "a  b   c    d",
    "2026年11月20日、気温は12.5度。",
    "1234567890 3.14159 -42 1e-6",
    "fn main() { println!(\"hi {}\", 1 + 2); }",
    "let x: Vec<u32> = (0..10).map(|i| i * i).collect();",
    "GPUのVRAMは8GB、帯域は272 GB/s。",
    "ＡＢＣ　全角の英字と空白",
    "ｶﾀｶﾅ半角とカタカナ全角",
    "絵文字🙂と記号★☆→←",
    "!!!??? ... --- ***",
    "中文和日本語と한국어",
    "Привет, мир! Ελληνικά.",
    "naïve café résumé",
    "\"quoted\" and 'single' and `backtick`",
    "https://github.com/jl1nie/qwen-rust-inference/blob/main/README.md",
    "    indented code block\n        deeper",
    "",
];

#[test]
#[ignore = "explicit opt-in: needs the real tokenizer under QWEN_MODEL_DIR (default models/qwen)"]
fn teaching_bpe_matches_the_production_tokenizer() {
    let dir =
        PathBuf::from(std::env::var("QWEN_MODEL_DIR").unwrap_or_else(|_| "models/qwen".into()));
    let production = TextTokenizer::load(&dir).unwrap();
    let teaching = TeachingBpe::load(&dir).unwrap();
    for text in SAMPLES {
        assert_eq!(
            teaching.encode(text).unwrap(),
            production.encode(text).unwrap(),
            "mismatch for {text:?}"
        );
    }
    // 本書第1章の表：質問の8個のID。
    assert_eq!(
        teaching.encode("日本の首都はどこですか？一言で。").unwrap(),
        [161607, 110161, 236576, 173399, 10992, 118333, 15685, 1710]
    );
}
