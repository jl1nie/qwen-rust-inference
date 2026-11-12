# Qwenを理解するRust推論器

1トークンが生成されるまでを、自分のコードで説明・検証するためのCPU推論器です。
**Phase 0〜6を実装**し、固定revisionのQwen3.5-0.8B公開重みで、テキスト経路を検証しました。
Rustだけで推論します。Pythonは公式ソースからの比較fixture生成に限定しています。

まず [コードの読み方と手計算例](spec/learning.md) を参照してください。
[保存トレースを読む8問の演習](spec/exercises.md) と [解答](spec/exercise-solutions.md) も用意しています。
AttentionのGQA・位置回転、DeltaNetの状態更新、FFNの書込み列には日本語の学習用コメントがあります。

## 教育用デモとオフラインテスト

Rust 1.85.1（rustfmt・clippy）とC/C++ツールチェーンが必要です。
依存はCargo.lock・vendorに固定。ツールチェーン準備後はネットワーク・実重み・Pythonなしで動きます。
デモの小型モデルは合成重みで、文章の意味を学習したものではありません。

```sh
cargo run --offline --locked -- demo ffn --unit 3
cargo run --offline --locked -- demo attention --cache
cargo run --offline --locked -- demo deltanet
cargo run --offline --locked -- demo lesson --trace /tmp/lesson.ndjson
cargo run --offline --locked -- demo hybrid --max-new-tokens 8
cargo fmt --check
cargo clippy --offline --locked --all-targets -- -D warnings
cargo test --offline --locked
```

FFNは幅8・中間幅16。ハイブリッドデモは幅8、DeltaNet×3＋Attention×1です。
`Backend::Reference` の明示ループとsgemmを残し、通常推論は検証済みの単一ベクトル用 `Vector` を使います。
[初期の選定理由](spec/adr-0002.md)・[実用性改善のADR](spec/adr-0003.md) と [依存ライセンス](spec/dependencies.md) を記録しています。

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

`infer --prompt` は単一userメッセージのtext-only chat templateを使います。
`--raw` はテンプレートなし、`--thinking` は思考開始付きです。
複数ターンやsystemメッセージには後述の `chat` を使います。ツール・画像入力は未対応です。
終了は `<|im_end|>` とendoftext、または生成数/文脈上限。`</think>`では終了しません。
seed固定samplingの乱数はこのRust実装のLCGであり、PyTorchのsampling列との一致は保証しません。

確認例：`1+1は？`、生成上限8で `1+1 は **2** です` を出力しました。
文の自然さではなく、[実モデルの演算別検証結果](spec/validation-real.json) を正しさの根拠にしています。
初期の演算別検証では3入力・計9位置、全24層を比較し、logits最大絶対誤差は約3.53e-5、全比較最大は約6.29e-5でした。
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

## 32・128・512トークンへの検証拡大

[検証仕様と実行手順](spec/extended-validation.md) に、全位置の24層出力と全語彙logits、
各prefill後の8位置の継続decode、全層の状態、自然なEOS終了を照合するopt-in検証を追加しました。
参照値はストリーム読みのf32バイナリで、通常CIのネットワーク・モデル不要という条件を保ちます。

```sh
.venv/bin/python tools/generate_extended.py --model-dir models/qwen
cargo test --release --offline --locked --test extended_real -- --ignored --nocapture --test-threads 1
```

長い入力は `Model::prefill_observed` で処理し、全stepの活性値を保持しません。
保存した [教材トレース](examples/lesson-trace.ndjson) と [数値解答](examples/lesson-answers.json) は
小型の合成モデルによるもので、Rustだけで再生成できます。

検証は合格し、[測定結果](spec/validation-extended.json) を保存しました。
継続decodeを含む696位置と自然なEOS生成の15予測位置、計711位置を比較し、
logitsの最大絶対誤差は約6.48e-5でした。詳細な範囲と相対誤差の読み方は検証仕様を参照してください。

## 複数ターン・保存と再開・速度計測

```sh
cargo run --release --offline --locked -- chat --model-dir models/qwen
# /continue /reset /save /tmp/chat.qrs /load /tmp/chat.qrs /quit

cargo run --release --offline --locked -- chat --model-dir models/qwen --prompt '数字だけで答えてください。1+1は？' --max-new-tokens 1 --save-state /tmp/partial.qrs
cargo run --release --offline --locked -- chat --model-dir models/qwen --load-state /tmp/partial.qrs --continue --max-new-tokens 16 --save-state /tmp/complete.qrs
cargo run --release --offline --locked -- chat --model-dir models/qwen --load-state /tmp/complete.qrs --prompt 'その数に1を足すと？数字だけ答えてください。' --max-new-tokens 16

cargo run --release --offline --locked -- bench --model-dir models/qwen --backend vector --decode-steps 8 --warmup 1 --runs 2 > /tmp/bench.json
```

実測例は、保存前「1」→再開「1+1=2」→次ターン「2+1=3」。
[実用性の仕様・全オプション](spec/practical.md) と [実モデル検証](spec/measurements/chat-resume.json) を参照してください。
`chat` はモデルを一度ロードして会話を続けます。`--system`/`--thinking`/seed固定samplingに対応し、
再開時には保存した設定を使います。過去assistantの思考部分が公式テンプレートで省かれるため、
一致するuser末尾のanchorへ戻り、必要な部分を再計算します。常に全cacheを盲目的に追記しません。

この環境の[計測結果](spec/measurements/cpu-baseline-vector.json)では平均decodeが約1.23→2.64 token/s、
ロードを含むピークRSSが約5.78→4.76GBでした。warmup 1回・測定2回の短い固定入力による観測です。
単一ベクトル向けCPU演算を追加し、巨大配列の共有時コピーと観測用コピーを省きました。
[711位置の参照検証](spec/validation-vector.json)も既存の誤差基準で合格しています。

snapshotは重みを含まず、KV/再帰/畳み込み状態、logits、pendingトークン、RNG、会話とanchorを保存します。
同じmodel-dirが必要で、別プロセスではモデルのロード時間が再び掛かります。
保存は明示opt-in、ローカルのみ、既存ファイルの上書きなし、デフォルト上限256MiB。
会話を含む平文/数値データで暗号化はしません。モデル・形状・version・checksumの不一致を拒否します。
weightsは読み取り専用のModel APIで扱い、編集は別overlayを使います。

## 実験用ROCm（RX 7900 XTX / WSL2）

任意機能 `--features rocm` とHIP/rocBLASブリッジで、FP32の重み・KV・DeltaNet状態を
GPUに保持する経路を追加した。通常のCPUビルドにはROCmは不要。
**GPUでのコンパイル・実行・性能は未検証**。CPUテストブリッジでの算術検証と区別する。

[WSL設定・ビルド・実機比較・生成の手順](spec/rocm.md)に従い、
`rocm-check`で小規模→実重みの順に検証してから`rocm-infer`を実行する。
GPU版はgreedy・単一run。BF16重みとまとめてprefill（ADR 0006、0007）はコードを共有するがROCm実機では未検証。会話保存、重み介入は未対応。

## 実験用Metal（Apple Silicon / macOS 15+）

任意機能 `--features metal` とObjective-C++のMetalブリッジで、ROCm版と同じRust側エンジン
（`src/device.rs`）をApple GPUで動かします。Command Line Toolsだけでビルドでき、Xcodeは不要です。
Apple M5で小型fixture・実重みの全層比較に合格し、生成結果がCPU版のgreedy出力と一致しました。

```sh
bash tools/build-metal.sh
cargo build --release --offline --locked --features metal
./target/release/qwen-rust-inference metal-check --library target/metal/libqwen_metal.dylib
./target/release/qwen-rust-inference metal-infer --library target/metal/libqwen_metal.dylib \
  --model-dir models/qwen --prompt '1+1は？' --max-new-tokens 32
```

行列は配布形式のBF16のまま（値は変えずに）GPUに置き、prefillは最大128トークンずつまとめて計算します。
Attentionの読出しもチャンク単位で、Apple GPUの行列演算命令を使って計算します。
M5での1回測定では、454トークンの入力でprefill約730 token/s・decode約50 token/s、
約1万トークンの入力でもprefill 18秒・decode約40 token/sでした（[記録](spec/measurements/metal-m5-attention.json)、
[decodeの記録](spec/measurements/metal-m5-decode-split.json)）。
長文脈では8192トークンまでCPU版と全状態を照合しています
（[長文脈の記録](spec/measurements/metal-m5-long.json)、`metal-check --text-file … --tokens N`）。
対応範囲はROCm版と同じです（greedy・単一run）。手順と設計は[spec/metal.md](spec/metal.md)、
[ADR 0005](spec/adr-0005.md)、[ADR 0006](spec/adr-0006.md)、[ADR 0007](spec/adr-0007.md)、[ADR 0008](spec/adr-0008.md)を参照してください。
