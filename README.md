# Qwenを理解するRust推論器

1トークンが生成されるまでを、自分のコードで説明・検証するためのCPU推論器です。
**Phase 0〜6を実装**し、固定revisionのQwen3.5-0.8B公開重みで、テキスト経路を検証しました。
Rustだけで推論します。Pythonは公式ソースからの比較fixture生成に限定しています。

まず [コードの読み方と手計算例](spec/learning.md) を参照してください。
AttentionのGQA・位置回転、DeltaNetの状態更新、FFNの書込み列には日本語の学習用コメントがあります。

## 教育用デモとオフラインテスト

Rust 1.85.1（rustfmt・clippy）とC/C++ツールチェーンが必要です。
依存はCargo.lock・vendorに固定。ツールチェーン準備後はネットワーク・実重み・Pythonなしで動きます。
デモの小型モデルは合成重みで、文章の意味を学習したものではありません。

```sh
cargo run --offline --locked -- demo ffn --unit 3
cargo run --offline --locked -- demo attention --cache
cargo run --offline --locked -- demo deltanet
cargo run --offline --locked -- demo hybrid --max-new-tokens 8
cargo fmt --check
cargo clippy --offline --locked --all-targets -- -D warnings
cargo test --offline --locked
```

FFNは幅8・中間幅16。ハイブリッドデモは幅8、DeltaNet×3＋Attention×1です。
`Backend::Reference` の明示ループを残し、実モデルは `matrixmultiply` を使います。
[バックエンドの選定理由](spec/adr-0002.md) と [依存ライセンス](spec/dependencies.md) を記録しています。

## 公開重みでテキスト推論

通常テストではダウンロードしません。以下の**明示コマンド**だけが外部取得します。
重み約1.75GB、読み込んだ言語部f32重みと一時バッファに数GBのRAMが必要です。
短い文脈・少数生成トークンを対象にしています。

```sh
tools/download-model.sh ./models/qwen
cargo run --release --offline --locked -- inspect-config --model-dir ./models/qwen
cargo run --release --offline --locked -- infer --model-dir ./models/qwen --ids 16,10,16,28 --max-new-tokens 8
cargo run --release --offline --locked -- infer --model-dir ./models/qwen --prompt '1+1は？' --max-new-tokens 16
cargo run --release --offline --locked -- infer --model-dir ./models/qwen --prompt 'Hello' --max-new-tokens 8 --seed 42 --temperature 0.8 --top-k 40
```

`--prompt` は単一userメッセージのtext-only chat templateを使います。
`--raw` はテンプレートなし、`--thinking` は思考開始付きです。
会話履歴・systemメッセージ・ツール・画像入力はCLIの対応範囲に含めません。
終了は `<|im_end|>` とendoftext、または生成数/文脈上限。`</think>`では終了しません。
seed固定samplingの乱数はこのRust実装のLCGであり、PyTorchのsampling列との一致は保証しません。

確認例：`1+1は？`、生成上限8で `1+1 は **2** です` を出力しました。
文の自然さではなく、[実モデルの演算別検証結果](spec/validation-real.json) を正しさの根拠にしています。
3入力・計9位置、全24層を比較し、logits最大絶対誤差は約3.53e-5、全比較最大は約6.29e-5でした。
BF16重みを両実装ともf32へ変換した比較であり、BF16ネイティブ計算との比較ではありません。

## トレースと介入

トレースはデフォルト無効で、外部送信しません。指定された新規ファイルへNDJSONを書き、既存ファイルを上書きしません。
Q/K/V・スコア・確率・ゲート・FFN中間値・DeltaNet状態・残差・logits上位候補を観測できます。
配列はフラットなf32ビューとしてshapeを記録します。多軸の格納規約は[対応表](spec/tensors.md)と各型を参照してください。

```sh
cargo run --offline --locked -- demo ffn --trace /tmp/ffn.ndjson --trace-full
cargo run --release --offline --locked -- infer --model-dir ./models/qwen --ids 16,10,16,28 --max-new-tokens 1 --trace /tmp/qwen.ndjson --trace-layer 3 --trace-position 0 --trace-max-bytes 65536
```

実モデルで `--trace-full` を使う場合、対象層・対象位置・バイト上限をすべて明示してください。
最終logits全配列も記録するため、十分な上限が必要です。上限超過は次のレコードを書かずエラーになります。
出力済みレコードはファイルに残ります。トレースは入力内容を推測できる情報を含むため、保存先を管理してください。
Rust APIの `Model::prefill` は観測用に全stepの活性値を返します。大量位置には使わず、
通常生成は `generate` で各stepの活性値を解放してください。

```sh
cargo run --release --offline --locked -- infer --model-dir ./models/qwen --ids 16,10,16,28 --max-new-tokens 4 --edit examples/zero-unit.json --compare-edit
cargo run --release --offline --locked -- infer --model-dir ./models/qwen --ids 16,10,16,28 --max-new-tokens 4 --edit examples/remove-coordinate-direction.json --compare-edit
```

編集manifestはモデルrevision・ゼロ始まりの層・完全tensor名・列・方向・係数を保存します。
元重みは不変で、ユニット無効化と方向成分除去は活性値/出力差分のoverlayです。
比較時はFFN・残差・logitsの差と、生成列の反復トークン/二連続組の再出現数を表示します。
例の方向は第0座標で、意味的な反復方向を発見したものではありません。
研究記事のループ抑制や性能向上の再現は未実施です。

## 固定参照とfixture再生成

モデル・重み・tokenizer・テンプレートは `2fc06364715b967f1860aea9cf38778875588b17`、
Transformersは `a6f5c8676b158c367dc3389f68dde7a97fb619d6`。
[参照lock](spec/references.lock.json) にGit/LFS識別子と同梱資料のSHA-256があります。

公式クラス/関数本文をASTで抽出し、型注釈と任意カーネルラッパーを外してCPU PyTorchで実行します。
クラスの数式本文は変更しません。動的キャッシュの格納だけを小さな比較用アダプタで提供します。
Rustには公式chunk式を移植せず、参照側のchunk prefillと逐次decodeの両方を比較します。
実モデルfixtureではメモリを抑えるため再帰状態全量を省き、小型fixtureでは全stepを保存します。

```sh
python3.12 -m venv .venv
.venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu -r tools/requirements-reference.lock
.venv/bin/python tools/generate_fixtures.py
.venv/bin/python tools/generate_hybrid.py
(cd tests/fixtures && sha256sum -c SHA256SUMS)
cargo test --offline --locked
# 実モデル統合検証はopt-in（通常CIではignore）
.venv/bin/python tools/generate_hybrid.py --model-dir models/qwen
cargo test --release --offline --locked --test real_model -- --ignored --nocapture
```

ファイルハッシュ再現環境はPython 3.12.14 / Linux x86_64 / CPU PyTorch 2.6.0。
入力・重み・版・生成条件・誤差基準・生成器hashをfixtureに保存します。
実モデルの大きいfixtureと重みはGitに入れません。通常CIは同梱した小型fixtureを読みます。

## 仕様・制限

- [Phase 0・1の演算仕様](spec/phase-01.md) / [Phase 2〜6の仕様と許容誤差](spec/phase-02-06.md)
- [形状・tensor名・参照対応表](spec/tensors.md)
- [進捗・実行した検証・残る制限](spec/progress.md)

対応はCPU・単一バッチ・短いテキスト。最大文脈長での実行、GPU、量子化、画像・動画、
MTP、学習、分散、投機的decode、Web UI、研究性能の再現は対象外です。
実モデルの別revision・別アーキテクチャ・別dtype・別RoPEは受け入れません。
演算エラーで中断したModelStateは再利用せずresetしてください。

Apache-2.0。参照資料・依存の著作権とライセンスは [NOTICE](NOTICE) とvendor内に保持しています。
