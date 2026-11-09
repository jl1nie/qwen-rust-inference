//! # Qwenの1トークンを下から読む
//!
//! 学習の入口は `tensor` → `ops` → `ffn`。次に `attention` と
//! `deltanet` を対比し、`block` の残差接続、`model` のlogits、
//! `generate` の自己回帰へ進む。実重みの解釈は `weights` に隔離する。
//! `state` は過去を保持し、返り値のactivationsは今回の計算だけを表す。
//! 数式・形状・許容誤差はspec/、読み方と小さな数値例はspec/learning.md。
//!
//! 教育用スカラーループは `Backend::Reference`、実モデル用の行列積は
//! `Backend::Optimized`。高水準モデル呼出しやPython推論には依存しない。
pub mod attention;
pub mod benchmark;
pub mod block;
pub mod chat;
pub mod deltanet;
pub mod edit;
pub mod ffn;
pub mod generate;
pub mod lesson;
pub mod model;
pub mod ops;
pub mod session;
pub mod snapshot;
pub mod state;
pub mod tensor;
pub mod text;
pub mod trace;
pub mod validation;
pub mod weights;
