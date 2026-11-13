# Qwenを理解するRust推論器

![層3のAttention確率。生成した「東京」が質問の「首都」を強く参照している](spec/images/attention-layer3.svg)

実重みのQwen3.5-0.8Bが「日本の首都はどこですか？一言で。」に「東京です。」と答えたときの、層3（最初のAttention層）の8ヘッド分のAttention確率です。
行が読む側の位置、列が読まれる側の位置です。答えの「東京」を出す位置が、質問の「首都」を0.60の重みで読んでいます（オレンジ枠）。
この図はトレースから描いています（[描き方](#トレースと介入)）。

Qwen3.5-0.8Bの公開重みで、1トークンが生成されるまでを自分のコードで説明・検証するためのRust推論器です。
次の4つを、この順に目的にしています。

1. **Qwenの構造をコードで理解する**：transformerが中で何をしているかを、式・コード・トレースで追えるようにする。
2. **同じ計算をハードウェアごとに速くするとどうなるか**：Apple GPU（Metal）とAMD GPU（ROCm）で、段階を追って速くした事例。
3. **まだ速くできていない部分**：残った差を数値で示し、次に試すアプローチを整理する（[今後の課題](spec/roadmap.md)）。
4. **他の実装はどう速くしているか**：llama.cpp・MLX・Strataのアプローチとの比較（[他の実装との比較](spec/other-engines.md)）。

## 1. Qwenの構造をコードで理解する

Qwen3.5-0.8Bは24層で、3層のGated DeltaNet（線形Attention）と1層のAttentionを6回繰り返す混成構造です
（幅1024、FFNの中間幅3584、語彙248,320）。
Attention層は過去の全位置のK/Vを読み、DeltaNet層は過去を固定サイズの状態行列に畳み込んで持ちます。
上の図は、その最初のAttention層（層3）が何を読んでいるかをトレースから描いたものです。

**CPU版が基準**（教材であり正解）です。固定revisionの重みで、テキスト経路を公式実装と最大711位置で照合しました。
CPU版はRustだけで書いてあり、AttentionのGQA・位置回転、DeltaNetの状態更新、FFNの書込み列には日本語の学習用コメントがあります。

- **中を見る**：Q/K/V・Attentionのスコアと確率・DeltaNetの状態・FFNの中間値・残差・logitsをNDJSONで記録できます。会話中も位置を指定して記録できます。
- **介入する**：FFNのユニットを消す、ある方向の成分を除く、といった重みへの介入を、生成や会話の途中で付け外しできます（元の重みは不変）。
- **会話する**：複数ターンの会話、保存と再開（CPU版とGPU版で同じファイル）、長いsystem文書を一度だけ計算して再利用するキャッシュ。
- **学ぶ**：[コードの読み方と手計算例](spec/learning.md)、[保存トレースを読む8問の演習](spec/exercises.md)と[解答](spec/exercise-solutions.md)、
  合成重みの小型モデルで動く[デモ](#教育用デモとオフラインテスト)。

## 2. ハードウェアごとに速くした事例

GPU版（Metal・ROCm）は同じRustのエンジン（`src/device.rs`）を共有し、自作のGPUカーネル（Metal Shading Language／HIP C++）を
薄いブリッジ（Objective-C++／C++）で呼びます。外部の推論ライブラリや機械学習フレームワークは使いません。
GPU版はすべてCPU版と照合し（全層・全状態、長文脈は8192トークンまで）、既定の設定は値を変えません（足す順序が変わるだけ）。
値が変わる高速化はオプションに分けています（Metalの `--matmul-relaxed`、ROCmの `--attention-f16`）。

- **Metal（Apple M5）**：カーネルを、CPU版の式をそのまま写した段階1、SIMD groupと8×8行列命令の段階2、
  行列演算器（Metal 4テンソル演算）を使う段階3に分けてあり、`--metal-kernels` で切り替えて同じ入力で比べられます
  （2644トークンのprefillで309 → 1,736 → 2,155 token/s、生成トークンはどの段階でも同じ）。
  decodeは約73〜77 token/sで、同じ条件のMLX（BF16）の約95%です。MLXは途中もBF16で計算しますが、こちらは途中もFP32のままです。
  約1万トークンのprefillは5.3秒（精度を緩めるオプションで3.2秒）。
- **ROCm（RX 7900 XTX・WSL2）**：decodeは発行回数を459回から129回に減らすなどして約170から約330 token/sに、
  約1万トークンのprefillは4.4秒から2.5秒になりました。重みの読み出しはメモリ帯域の約半分（約500GB/s）です。
- **2つのGPUで知見を共有**：ROCmで見つけたdecodeの改善をMetalに取り入れています（[ADR 0019](spec/adr-0019.md) → [ADR 0020](spec/adr-0020.md)）。
  各高速化の理由・数値・採用しなかった案は[ADR](spec/)と[測定記録](spec/measurements/)に残しています。

| | CPU | Metal（Apple GPU） | ROCm（AMD GPU） |
|---|---|---|---|
| 推論・生成 | ✓（基準） | ✓（M5で検証） | ✓（RX 7900 XTX・WSL2で検証） |
| 会話・保存と再開・systemキャッシュ | ✓ | ✓（ファイル共通） | ✓（ファイル共通、CPU版と一致） |
| 重み介入 | ✓ | ✓ | ✓（CPU版と一致） |
| トレース | 全演算 | 層ごとの出力（mixer内部はCPU版のみ） | 同左 |
| sampling | greedy・seed固定 | `infer` はgreedy、`chat` はseed固定も可 | 同左 |
| decode速度（参考） | 約40 token/s（M5・既定8スレッド）、約27 token/s（Ryzen 9 9900X・8スレッド） | 約73〜77 token/s（M5） | 約330 token/s（RX 7900 XTX） |

正確に言うと、次の限界があります。

- 「pure Rust」なのはCPU版です。GPU版のカーネルはMetal Shading LanguageとHIP C++で書いています。
- Metalの段階3（prefillの行列積とAttention）は、AppleのMetal 4テンソル演算（`matmul2d`）の中で計算されます。
  正しさはCPU版との照合で確かめていますが、計算の中身まで自分のコードで追えるのは段階2までです。
- MLXに迫っているのはdecodeです。prefillはMLXの約1/3（`--matmul-relaxed` で約半分）です
  （[比較](spec/measurements/mlx-comparison-m5.json)、[測定](spec/measurements/metal-m5-tensor-matmul.json)）。
- 比較は0.8Bの1つのモデルでのものです。MLXは量子化や大きなモデルにも対応する汎用のフレームワークで、こちらはこのモデル専用です。
  ROCm版は他の推論エンジンと同じ条件で測っていません。

## 3. まだ速くできていない部分

[今後の課題](spec/roadmap.md)に、残った差の内訳と次の候補をまとめています。要点は次のとおりです。

- **Metalのprefill**：差の大部分は、入力を丸めない行列積（prefillの約77%）です。活性値をBF16で持つ（値が変わる）か、
  2つのBF16に分けて掛ける（ほぼ値を変えない）案があります。DeltaNetの状態更新をチャンク単位の行列形式にする案もあります。
- **decode**：重みの読み出しで頭打ちです。手元の重みにはMTP（次の次のトークンを予測する層）が1層あり、
  これで下書きして本体でまとめて検証すれば、greedyの出力を変えずに1回の読み出しで複数トークンを確定できる可能性があります。
  CPU版で測った採用率は下書き1で68%、3つ下書きして1回の検証で平均2.4トークンです（[ADR 0022](spec/adr-0022.md)）。
  DeltaNetの状態を巻き戻せないことと、小さなモデルでは下書きの語彙射影が重いことが課題です。
- **ROCm**：発行の隙間を減らす層ごとの常駐カーネル、行数の少ない行列×ベクトル積の帯域、prefillの内訳の計測。

## 4. 他の実装はどう速くしているか

[他の実装との比較](spec/other-engines.md)で、汎用のllama.cpp・MLXと、特定モデル専用のStrataのアプローチを、
このリポジトリの設計と並べています。

Pythonは公式ソースからの比較fixture生成と、図を描く補助スクリプトだけです。

## すぐ試す

```sh
tools/download-model.sh ./models/qwen          # 公開重み約1.75GB（明示的に実行したときだけ取得）
cargo run --release --offline --locked -- chat --model-dir ./models/qwen

# Apple Silicon Macなら GPU 版（Xcode不要）
bash tools/build-metal.sh
cargo build --release --offline --locked --features metal
./target/release/qwen-rust-inference metal-chat --library target/metal/libqwen_metal.dylib --model-dir ./models/qwen
```

コードを読むなら、まず [コードの読み方と手計算例](spec/learning.md) を参照してください。

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
重みファイルは約1.75GBです。行列は配布形式のBF16のまま持つので、読み込み後のRAMは約1.6GBです（[ADR 0012](spec/adr-0012.md)）。
CPU版は短い文脈・少数生成トークン向けです（M5でdecode約40 token/s、1スレッドなら約16）。長い文脈はGPU版を使ってください。

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
`infer` は1位置、`chat` は `--trace-position N,M,...` や `/trace N`・`/trace +K` で指定した絶対位置を、計算されるたびに記録します。
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

冒頭の図は、`chat` で全位置をトレースし、Python標準ライブラリだけのスクリプト（`tools/plot_attention.py`）でSVGにしたものです。

```sh
cargo run --release --offline --locked -- chat --model-dir ./models/qwen \
  --prompt '日本の首都はどこですか？一言で。' --max-new-tokens 6 \
  --trace /tmp/l3.ndjson --trace-position $(seq -s, 0 39) --trace-layer 3 \
  --trace-max-bytes 400000000 --trace-full --save-state /tmp/l3.qrs
python3 tools/plot_attention.py /tmp/l3.ndjson --layer 3 --snapshot /tmp/l3.qrs \
  --tokenizer models/qwen/tokenizer.json --mark 0:20:4 --mark 3:20:4 -o /tmp/attention.svg \
  --title 'Qwen3.5-0.8B, layer 3 attention: 日本の首都はどこですか？一言で。→ 東京です。'
```

軸のトークン名はsnapshotのトークン列と `tokenizer.json` から復元します。色は確率の平方根で、小さな重みも見えるようにしています。
位置ごとにlogits全体（約2.5MB）も記録されるので、トレースは約70MBになります。

```sh
cargo run --release --offline --locked -- infer --model-dir ./models/qwen --ids 16,10,16,28 --max-new-tokens 4 --edit examples/zero-unit.json --compare-edit
cargo run --release --offline --locked -- infer --model-dir ./models/qwen --ids 16,10,16,28 --max-new-tokens 4 --edit examples/remove-coordinate-direction.json --compare-edit
```

編集manifestはモデルrevision・ゼロ始まりの層・完全tensor名・列・方向・係数を保存します。
元重みは不変で、ユニット無効化と方向成分除去は活性値/出力差分のoverlayです。
比較時はFFN・残差・logitsの差と、生成列の反復トークン/二連続組の再出現数を表示します。
例の方向は第0座標で、意味的な反復方向を発見したものではありません。
研究記事のループ抑制や性能向上の再現は未実施です。
`chat` では `--edit` で開始し、`/edit PATH`・`/edit off` で会話の途中に付け外しできます。
変えると同じトークン列を新しい介入で計算し直すので、キャッシュに2種類の介入が混ざりません。介入は保存ファイルにも記録されます。

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

対応は単一バッチのテキストです。CPU版は短い文脈向けで、長い文脈はGPU版（Metal・ROCmとも検証済み）を使います。
量子化（値が変わる形式）、画像・動画、MTPを使った生成、学習、分散、投機的decode、Web UI、研究性能の再現は対象外です（MTPによる投機的decodeと量子化は[今後の課題](spec/roadmap.md)で候補にしています。MTPの採用率は `mtp-accept` で測れます、[ADR 0022](spec/adr-0022.md)）。
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
systemメッセージ（`--system` または `--system-file`）は最初に単独でprefillし、以後のターンと `/reset` で再利用します。
その直後に保存したファイルは長い文書のキャッシュになります。会話中の重み介入（`/edit`）とトレース（`/trace`）は[トレースと介入](#トレースと介入)を参照してください。
保存ファイルはGPU版（`metal-chat`・`rocm-chat`）と共通です。

この環境の[計測結果](spec/measurements/cpu-baseline-vector.json)では平均decodeが約1.23→2.64 token/s、
ロードを含むピークRSSが約5.78→4.76GBでした。warmup 1回・測定2回の短い固定入力による観測です。
その後、重みをf32へ広げずBF16のまま持つようにして、ピークRSSは約1.5GBになりました（計算結果はビット単位で同一、[記録](spec/measurements/host-bf16.json)）。
Metal版はその同じメモリをGPUがコピーせずに読むので、GPU版でもピークRSSは約2.0GBです。
行列の射影は出力行を複数スレッドで分担します（[ADR 0013](spec/adr-0013.md)、`src/parallel.rs`）。各行は1スレッドが同じ順序で計算するので、結果はスレッド数によらずビット単位で同一です。
decodeは重みの読み出し帯域で頭打ちになり、Ryzen 9 9900X（WSL2）では1スレッド約21 → 2〜8スレッド約27 token/sでした。
Apple M5（性能コア4＋高効率コア6）はメモリ帯域が広いので伸び続け、1スレッド16.2 → 2スレッド27.7 → 4スレッド34.1 → 6〜10スレッド約40 token/sでした（[記録](spec/measurements/cpu-parallel-m5.json)）。`QWEN_THREADS=1` で直列に戻せます（既定は最大8）。
単一ベクトル向けCPU演算を追加し、巨大配列の共有時コピーと観測用コピーを省きました。
[711位置の参照検証](spec/validation-vector.json)も既存の誤差基準で合格しています。

snapshotは重みを含まず、KV/再帰/畳み込み状態、logits、pendingトークン、RNG、会話とanchorを保存します。
同じmodel-dirが必要で、別プロセスではモデルのロード時間が再び掛かります。
保存は明示opt-in、ローカルのみ、既存ファイルの上書きなし、デフォルト上限256MiB。
会話を含む平文/数値データで暗号化はしません。モデル・形状・version・checksumの不一致を拒否します。
weightsは読み取り専用のModel APIで扱い、編集は別overlayを使います。

## MTPの採用率を測る

重みに含まれるMTP層（次の次のトークンを予測する1層）を読み込み、本体のgreedy生成に対して下書きがどれだけ当たるかを測ります。
生成そのものには使いません。本体とidentityは変わりません（[ADR 0022](spec/adr-0022.md)、`src/mtp.rs`）。

```sh
cargo run --release --offline --locked -- mtp-accept --model-dir models/qwen \
  --prompt-file spec/measurements/mtp-prompts.txt --max-new-tokens 128 --depth 3
```

`--prompt-file` は1行1プロンプトです。M5で8プロンプトを測ると、下書き1〜3の採用率（そこまで全部当たる率）は68%・43%・27%、
3つ下書きしたときに1回の検証で確定するトークンは平均2.38でした（[記録](spec/measurements/mtp-acceptance-cpu.json)）。
投機的decodeにしたときの見込みと課題は[今後の課題](spec/roadmap.md)の3.にあります。

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
M5での測定では、2644トークンの入力でprefill約2,200 token/s（1.2秒）・decode約74 token/s、
約1万トークンの入力でもprefill 5.2秒・decode約65 token/sでした（[prefillの記録](spec/measurements/metal-m5-tensor-matmul.json)、
[decodeの記録](spec/measurements/metal-m5-decode-small-kernels.json)）。decodeは同じ条件のMLX（BF16）の約95%で、
残りの大部分は上限近くまで出ている重みの読み出しです（ROCm版のADR 0019の考え方を取り入れたADR 0020・0021）。prefillの行列積とAttentionはMetal 4のテンソル演算（M5のニューラルアクセラレータ）で、
入力を丸めずに計算します（ADR 0015・0016で合計約3〜3.4倍）。`--matmul-relaxed` を付けると行列積の精度を緩めて
さらに速くなります（2644トークンで0.65秒前後、値は少し変わります。既定では無効）。

GPUのカーネルは、CPU版の式をそのまま写した段階1（reference）、SIMD groupと8×8行列命令で速くした段階2（simdgroup）、
Metal 4のテンソル演算を使う段階3（tensor）に分けてあり、`--metal-kernels reference|simdgroup|tensor` で切り替えられます
（既定はtensor、生成トークンはどの段階でも同じ）。読む順と、演算ごとにどのカーネルが動くかの表は
[spec/metal.md](spec/metal.md) の「カーネルの段階」にあります（[ADR 0018](spec/adr-0018.md)）。
長文脈では8192トークンまでCPU版と全状態を照合しています
（[長文脈の記録](spec/measurements/metal-m5-long.json)、`metal-check --text-file … --tokens N`）。
CPU版と同じ編集manifest（`--edit`、`--compare-edit`）と、1位置分の層ごとのトレース（`--trace … --trace-position N`）も使えます。
AttentionやDeltaNetの内部はCPU版のトレースだけにあります。

`metal-chat` はCPU版 `chat` と同じ会話・コマンド・snapshotファイルをGPUで扱います。
GPUで保存した会話をCPUで再開することも、その逆もできます。
systemメッセージは最初に1回だけprefillして巻き戻し点として残すので、長い文書をsystemに入れて保存すれば
「トークンキャッシュ」になります（2637トークンのprefill 3.6秒 → 読込0.24秒、[記録](spec/measurements/metal-m5-chat.json)）。

```sh
./target/release/qwen-rust-inference metal-chat --library target/metal/libqwen_metal.dylib \
  --model-dir models/qwen --system-file 長い文書.txt --save-state /tmp/doc.qrs </dev/null
./target/release/qwen-rust-inference metal-chat --library target/metal/libqwen_metal.dylib \
  --model-dir models/qwen --load-state /tmp/doc.qrs --prompt 'この文書を要約して' --stats
```

DeltaNetの状態は全トークンを吸収した固定サイズの行列なので、KVのように任意の位置で切り戻せません。
そこで巻き戻す場所（system末尾と直前のuser末尾）だけでDeltaNet状態（約19MiB）を控えます（[ADR 0011](spec/adr-0011.md)）。
会話中も `--edit`・`/edit PATH|off`（履歴を新しい介入で計算し直す）と、絶対位置のトレース（`--trace-position N,M`、`/trace +K`）が使えます。
どちらもCPU版 `chat` に同じ形であり、CPU版のトレースにはAttention・DeltaNetの内部まで入ります。
`infer` はgreedy・単一run、`chat` はseed固定samplingも使えます。手順と設計は[spec/metal.md](spec/metal.md)、
[ADR 0005](spec/adr-0005.md)、[ADR 0006](spec/adr-0006.md)、[ADR 0007](spec/adr-0007.md)、[ADR 0008](spec/adr-0008.md)、[ADR 0009](spec/adr-0009.md)、
[ADR 0010](spec/adr-0010.md)、[ADR 0011](spec/adr-0011.md)、[ADR 0015](spec/adr-0015.md)、[ADR 0016](spec/adr-0016.md)、[ADR 0017](spec/adr-0017.md)、[ADR 0018](spec/adr-0018.md)、[ADR 0020](spec/adr-0020.md)、[ADR 0021](spec/adr-0021.md)を参照してください。

## 実験用ROCm（RX 7900 XTX / WSL2）

任意機能 `--features rocm` とHIP/rocBLASブリッジで、Metal版と同じRust側エンジンをAMD GPUで動かす経路です。
通常のCPUビルドにはROCmは不要です。
RX 7900 XTX（gfx1100）・WSL2・ROCm 7.2.4で、小規模fixture・実重み・8192トークンまでの長文脈をCPU版と照合して合格しています。
`rocm-chat` の会話・保存と再開（CPU版との相互）・systemキャッシュ・重み介入・トレースもCPU版と一致しました。
WSL2では `HSA_ENABLE_DXG_DETECTION=1` がないとGPUが見えません（ブリッジは自分で設定します。`rocminfo` には付けてください）。

[WSL設定・ビルド・実機比較・生成の手順](spec/rocm.md)に従い、
`rocm-check`で小規模→実重みの順に検証してから`rocm-infer`・`rocm-chat`を実行してください。
HIPのカーネルはCPUテストブリッジ用の素朴な本体と分け、GPU向けに分割し直しています（[ADR 0013](spec/adr-0013.md)）。
decodeは発行回数を約460回から約130回に減らし、射影の読み込みを並べ、カーネルの命令数を減らして、約170から約330 token/sになりました。約1万トークンのprefillも4.4秒から2.5秒です（値は和の順序だけ変わり、生成トークンは同一。[ADR 0019](spec/adr-0019.md)）。
`--attention-f16` で、まとめてprefillのAttentionだけをFP16の行列命令で計算できます（opt-in、結果が少し変わります。約1万トークンでprefill 1.39倍、[ADR 0014](spec/adr-0014.md)）。
重み介入とトレース（ADR 0010）、会話と保存/再開（ADR 0011）はMetal版とコードを共有し、ROCm実機でもCPU版と一致しました。

### 速度の推移（[ADR 0019](spec/adr-0019.md)）

RX 7900 XTX・BF16・チャンク128・64トークン生成、3回の中央値（実行ごとに約±5%ぶれる）。
生成トークンは全段階で同一です。

decode（token/s）：

| 入力 | 前 | 1回目 | 2回目 | 3回目 | 4回目 | 現在 | 倍率 |
|---|---|---|---|---|---|---|---|
| 4トークン | 175 | 272 | 289 | 316 | 329 | 約330 | 1.9倍 |
| 331トークン | 173 | 267 | 290 | 317 | 324 | 330 | 1.9倍 |
| 1267トークン | 172 | 261 | 284 | 294 | 312 | 315 | 1.8倍 |
| 2060トークン | 166 | 255 | 284 | 307 | 314 | 320 | 1.9倍 |
| 9872トークン | 149 | 228 | 260 | 280 | 288 | 290 | 1.9倍 |

prefill：331トークン 0.118 → 0.092秒、1267トークン 0.398 → 0.293秒、2060トークン 0.667 → 0.47秒、
9872トークン 4.45 → 2.47秒（1.8倍）。AttentionとDeltaNetのカーネルを軽くした効果です。

| 1トークン（4トークン入力のdecode） | 前 | 現在 |
|---|---|---|
| 時間 | 5.7ms | 3.0ms |
| GPUへの発行回数 | 459回 | 129回 |
| 実効帯域（重み1.5GBを毎トークン読む換算） | 約260GB/s（ピーク960の27%） | 約500GB/s（約52%） |
| 射影の読み出し帯域 | 320〜600GB/s | 約700GB/s（語彙射影は約880） |

| 段階 | 主な変更 | decode（331トークン） |
|---|---|---|
| 1回目 | 1スレッド/ヘッドだった3演算を1 wave/ヘッドに、同じ入力の射影をまとめる、隣り合う演算の融合、重みの先読み | 173 → 267（+54%） |
| 2回目 | カーネルの命令数削減（範囲チェック・レジスタ配列の実行時添字の除去）、融合を2つ追加 | → 290（+9%） |
| 3回目 | DeltaNetの4演算を1回の発行に、残差加算＋正規化を直前の射影の末尾へ | → 317（+9%） |
| 4回目 | 1トークン専用の行列×ベクトル積（共有関数がインライン化されていなかった） | → 324（+2%） |
| 5回目 | DeltaNetのヘッドを4ブロックに分割、decodeのAttentionでLDSを経由しない | → 330（+2%） |

systemキャッシュ付きの質問1回は174ms → 111ms、2ターン会話の1ターンは約80ms → 約68msです。
帯域から見た上限は約600 token/sですが、そこへ近づくには層ごとに常駐するカーネルのような
別の構造が要ります（ADR 0019の「Remaining」）。値は[記録](spec/measurements/rocm-7900xtx-decode.json)。
