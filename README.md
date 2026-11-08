# Qwenを理解するRust推論器

1トークンが生成されるまでを自分のコードで説明・検証するための、CPU推論器の段階的実装です。
現在の実装範囲は **Phase 0・1**。実モデルの文章生成はまだできません。

## 実行

Rust 1.85.1（rustfmt・clippyを含む）とCリンカが必要です。
依存ソースを `vendor/` に同梱しているため、ツールチェーン準備後のビルド・テストはネットワークもモデル重みも不要です。

```sh
cargo run --offline --locked -- demo ffn
cargo run --offline --locked -- demo ffn --unit 3
cargo run --offline --locked -- demo ffn --trace /tmp/ffn.ndjson --trace-full --trace-max-bytes 65536
cargo fmt --check
cargo clippy --offline --locked --all-targets -- -D warnings
cargo test --offline --locked
cargo test --offline --locked --test fixtures -- --nocapture
```

デモは幅8・中間幅16、seed 42の固定LCGで作る教育用の合成重みです。
gate、up、SiLU、積、downの出力を表示します。選んだ中間ユニットの反応値とdown行列の列、
無効化前後の差分と「反応値×列ベクトル」を比較できます。元重みは変更しません。
ランダム重みに文章としての意味はありません。

トレースはデフォルト無効、外部送信なし。指定時のみローカルNDJSONを書きます。
既存ファイルは上書きしません。全配列は `--trace-full` が必要で、改行を含むバイト上限を超える前にエラーになります。
現段階の層番号・位置は0、logits候補は未実装のためnullです。

## 参照とfixture

モデル・重み・tokenizer・chat templateはモデルcommit
`2fc06364715b967f1860aea9cf38778875588b17` に固定。
Transformersソースは `a6f5c8676b158c367dc3389f68dde7a97fb619d6` に固定。
[参照lock](spec/references.lock.json) にファイル一覧、Git/LFS識別子、手元の参照資料のSHA-256があります。
実重みのダウンロードや読込みは実装していません。

fixtureは、同梱した公式ソースから `Qwen3_5MLP` と `Qwen3_5RMSNorm` のクラスをASTで抽出し、
CPU PyTorch 2.6.0のf32で生成します。任意のカーネル差し替えデコレータだけを無効化し、クラスの演算本文はそのまま実行します。
Transformers全体や高速カーネル、BF16での一致を主張するものではありません。
独立した手計算ケースもRustテストにあります。全中間値、入力重み、版、生成条件、許容誤差、ハッシュを保存しています。

再生成は明示コマンドのみ（Pythonは比較用fixture生成に限定、Rustの実行経路からは呼びません）。

```sh
python3.12 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu -r tools/requirements-reference.lock
.venv/bin/python tools/generate_fixtures.py
(cd tests/fixtures && sha256sum -c SHA256SUMS)
# 再生成後に確認
cargo test --offline --locked
```

厳密なファイルハッシュ再現にはlock記載のPython 3.12.14・Linux x86_64も揃えてください。
異なる環境ではmetadata差を確認し、数値誤差を測定してください。許容誤差を失敗回避のために拡大しないでください。
通常CIは同梱fixtureを読み、Pythonをインストール・起動しません。

## 仕様と今後

- [演算仕様・形状・誤差基準](spec/phase-01.md)
- [参照対応表](spec/tensors.md)
- [進捗と検証結果](spec/progress.md)
- [依存とライセンス](spec/dependencies.md)
- [設計判断](spec/adr-0001.md)

Phase 2は因果Attentionと層別KVキャッシュ、Phase 3は逐次Gated DeltaNet、
Phase 4は小規模ハイブリッドモデルと生成、Phase 5は公開重み・tokenizer、
Phase 6は方向除去overlayによる重み編集です。フェーズごとに仕様→失敗するテスト→実装→参照照合を進めます。
`attention`、`deltanet`、`inspect-config`、`infer` は未対応としてCLIで拒否します。
GPU・量子化・バッチ・画像・MTP高速化・学習・Web UIは初期対象外です。

ライセンスはApache-2.0。同梱参照資料の著作権・ライセンスは [NOTICE](NOTICE) を参照してください。
