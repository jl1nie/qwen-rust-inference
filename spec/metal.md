# Apple Silicon（Metal）で動かす実験用GPUバックエンド

**状態：Apple M5 / macOS 27.3で小型fixture・実重み比較・生成を確認済み。**
記録は [metal-m5.json](measurements/metal-m5.json)・[metal-m5-bf16.json](measurements/metal-m5-bf16.json)・
[metal-m5-attention.json](measurements/metal-m5-attention.json)・
[metal-m5-decode-split.json](measurements/metal-m5-decode-split.json)・
[metal-m5-norm.json](measurements/metal-m5-norm.json)、
設計は [ADR 0005](adr-0005.md)・[ADR 0006](adr-0006.md)（BF16重みとまとめてprefill）・
[ADR 0007](adr-0007.md)（まとめてAttention読出し）・[ADR 0008](adr-0008.md)（decodeの位置分割）・[ADR 0009](adr-0009.md)（RMSNormの並列化）・
[ADR 0010](adr-0010.md)（重み介入とトレース）・[ADR 0011](adr-0011.md)（会話と保存/再開）・
[ADR 0015](adr-0015.md)（prefillの高速化）。
ROCm版（[rocm.md](rocm.md)）と同じRust側エンジン `src/device.rs` を共有する。

## 1. 必要なもの

- Apple Silicon Mac、macOS 15以降（カーネルを安全な数学モードで実行時コンパイルするため）
- Rust 1.85.1、Command Line Tools（clang＋macOS SDK）
- Xcodeの `metal` シェーダコンパイラは不要。Rust依存の追加もない

## 2. ビルドと小規模比較

```sh
bash tools/build-metal.sh
cargo build --release --offline --locked --features metal
./target/release/qwen-rust-inference metal-check --library target/metal/libqwen_metal.dylib
```

`build-metal.sh` は `native/metal/kernels.metal` を文字列として埋め込み、
`target/metal/libqwen_metal.dylib` を作る。dlopenはそのファイル内のコードを実行するため、
信頼できるローカルビルドだけを指定する。ABI版とブリッジ種別（Metal=2）を検証し、
ROCm用やCPUテスト用のライブラリは拒否する。CPUへの自動切替はしない。

小規模比較は既存hybrid fixture（固定Transformers由来）とCPU参照の両方に対し、
全層7演算・最終norm・logits・各位置のKV/畳み込み/再帰状態、reset、全prefill分割を検証する。
許容誤差はADR 0004の `atol=2e-5, rtol=2e-4`（M5での最大絶対誤差は約1.3e-6）。

## 3. 実重みの比較と生成

```sh
./target/release/qwen-rust-inference metal-check --library target/metal/libqwen_metal.dylib \
  --model-dir models/qwen --ids 16,10,16,28
./target/release/qwen-rust-inference metal-infer --library target/metal/libqwen_metal.dylib \
  --model-dir models/qwen --prompt '1+1は？' --max-new-tokens 32

# 無視指定のGPUテスト（実重みは任意）
QWEN_DEVICE_LIBRARY=$PWD/target/metal/libqwen_metal.dylib QWEN_MODEL_DIR=$PWD/models/qwen \
  cargo test --release --offline --locked --features metal --test device_gpu -- --ignored
```

実重み比較の閾値は `atol=rtol=3e-3`。M5では2入力とも合格し、最大絶対誤差は約2.0e-4。
`1+1は？` の32トークン生成はCPU版のgreedy出力と同一だった。
生成はgreedy・単一run。出力JSONとレポートの扱いはROCm版と同じ。

### 長文脈の比較

```sh
./target/release/qwen-rust-inference metal-check --library target/metal/libqwen_metal.dylib \
  --model-dir models/qwen --text-file 任意の長文.txt --tokens 8192
./target/release/qwen-rust-inference metal-infer --library target/metal/libqwen_metal.dylib \
  --model-dir models/qwen --prompt-file 長い質問.txt --max-new-tokens 64
```

テキストをテンプレートなしでトークン化し、先頭Nトークンを1,31,32,33,63,64,65,…,2^k±1,Nで区切って
prefillする。各区切りで最終位置のlogits・greedy上位1件・全DeltaNet状態・全KVをCPU版
（`Backend::Vector`）と比較する。位置ごとの中間値は保存しないので、所要時間は長さに比例する。
CPU版自体は公式chunk prefillとの比較が512トークンまで、Reference版との比較が711位置まで済んでいる。
それより長い区間はRust CPU版との一致であり、公式実装との直接照合ではない。

## 4. 実装の読み方

| ファイル | 内容 |
|---|---|
| `src/device.rs` | Rust所有権と結線（ROCm/Metal共通） |
| `native/metal/bridge.mm` | MTLBuffer確保、ポインタ→(buffer, offset)解決、エンコードと同期 |
| `native/metal/kernels.metal` | 演算本体。`native/rocm/bridge.cpp` の `body()` と1対1 |

Apple GPUはCPUとメモリを共有するので、全バッファを `MTLStorageModeShared` で確保し、
その `contents` アドレスをRustに渡す。重み行列だけは逆に、Rustがページ境界に揃えて持つBF16の領域を
`newBufferWithBytesNoCopy` でそのままGPUバッファにする（コピーしない）。Rust側はROCm版と同じくポインタを直接読まない。
全カーネルを1つのcompute encoderへ直列に積み、logits読出し時だけcommitして完了を待つ。
行列は値が変わらない場合だけBF16で持ち（公開重みは全行列が該当）、カーネル内でf32へ戻して
f32で積算する。GEMVは1行を32レーンのSIMD groupで分担して `simd_sum` で足す。重みの転置コピーはしない。
prefillは最大128トークンずつまとめ、射影を行列×行列にして重みを1チャンクで1回だけ読む。
8トークン以上は64×64の出力タイルを4つのSIMD group（各32×32）で `simdgroup_float8x8` を使って計算する（ADR 0015）。
内積の足し方（列を32ずつ、8列ずつ順に）は変えていないので、タイルの大きさを変えても値は同じ。
畳み込み履歴とDeltaNet状態は1スレッド内でトークン順に更新するので、状態の進み方はdecodeと同じ。
DeltaNet状態の列はチャンクの間レジスタに置き、最初に1回読んで最後に1回書く（ADR 0015）。
decodeのAttention読出しは位置を32個ずつのブロックに分け、(ブロック,KVヘッド) ごとのthreadgroupで
部分的なsoftmax（最大値・分母・未正規化の加重和）を求め、2段目で全ブロックを合成する（ADR 0008）。
以前の3段（スコア → ヘッドごとのsoftmax → 加重和）は、head_dimが512を超える場合の予備として残す。
prefillではチャンク全行を1回で読む（ADR 0007）。8行×（KVヘッドを共有するQヘッド）を1つのthreadgroupにまとめ、
各Qヘッドを1つのSIMD groupが受け持つ。K/Vは32位置ずつ、8×8のタイルをGPUメモリから直接読み（同じKVヘッドの
Qヘッド同士はキャッシュで共有）、`QK^T` と `PV` を `simdgroup_float8x8` で計算する（ADR 0015）。
KVのバッファは32位置単位に切り上げて確保し、因果マスクの外側は確率0にする。
softmaxは最大値と分母を更新しながら進めるオンライン方式なので、スコア列を保存しない。
head_dimが64/128/256以外（小型fixture）は同じ方式のスカラー版を使う。prefillは最終位置だけ語彙射影する。

## 5. 測定と未対応

M5での測定（ウォームアップなし、BF16重み、チャンク128、2回測定、ADR 0015）：

| 入力 | prefill | decode |
|---|---|---|
| 454トークン | 1405 token/s（0.32秒） | 67.4 token/s |
| 2644トークン | 1567 token/s（1.69秒） | 63.5 token/s |
| 10141トークン | 1292 token/s（7.85秒） | 48.7 token/s |

ADR 0015の前（DeltaNet状態をメモリに毎トークン読み書き、Attentionは8位置ずつ、行列積は32×32タイル）は
454トークンで743、2644トークンで727 token/s（3.64秒）、10141トークンで556 token/s（18.25秒）だった。
同じプロンプトでMLX（BF16）はprefill 5,300〜6,200 token/s、decode 74〜80 token/s
（[比較](measurements/mlx-comparison-m5.json)）。残る差の大半は行列積で、MLXは16bitの行列演算を使う。

decodeは位置分割の前（ADR 0007）は2644トークンで43.2、10141トークンで29.3 token/s、
RMSNormを1スレッドで計算していた前（ADR 0008）は454トークンで50.1 token/sだった。
FFNのgate/up/SiLUを1カーネルに融合する案は試したが、速度が変わらなかったので採用していない（ADR 0009）。

Attention読出しをトークンごとに発行していたADR 0006の版は、2644トークンで344 token/s、
10141トークンで88秒だった。生成トークンは同一。FP32重み・1トークンずつだった最初の版は
2644トークンでprefill 39.7 token/s・decode 28.8 token/s、10141トークンでprefill 306秒だった。
`--f32-weights` と `--prefill-chunk 1` で前と同じ経路を選べる。
長文脈でもdecodeが少し遅くなるのは、K/Vの読出し量が文脈長に比例するため。
decodeはlogits全体の読出しとGPU完了待ちを含む。CPU版はM5でdecode約17 token/s（短い入力）。
重みはBF16のまま約1.5GBをホストに1つだけ持ち、GPUはそのメモリをコピーせずに読む（ADR 0012、`qr_wrap`）。
以前はホストのFP32（約3GB）とGPU用BF16の2つを持っていた。`metal-infer` のピークRSSは約5.5GB→2.0GB、重みの転送は1.6秒→0.03秒。

GEMVは単体測定でメモリ読出しの実効値（約132GB/s）近くまで出ており、1トークンのGPU時間14.6msのうち11.5msを占める。
BF16のままならdecodeの上限は約80 token/s。
未対応：GEMV以外の約400 dispatch（約3ms）の削減、量子化（値が変わる形式）、GPU sampling。

## 6. 重み介入とトレース（ADR 0010）

CPU版と同じ編集manifestとNDJSON形式を使える。編集は `edit.rs` の `EditManifest::apply` そのもので、
編集対象の層だけFFNのproductをホストへ読み戻して適用し、方向成分の差分を `Add` で足す。
GPU上の重みは変えない。トレースは1位置だけで、そこまでをまとめてprefillし、その位置だけ単独で取得する。

```sh
./target/release/qwen-rust-inference metal-infer --library target/metal/libqwen_metal.dylib \
  --model-dir models/qwen --ids 16,10,16,28 --max-new-tokens 12 --edit examples/zero-unit.json --compare-edit
./target/release/qwen-rust-inference metal-infer --library target/metal/libqwen_metal.dylib \
  --model-dir models/qwen --prompt-file long.txt --max-new-tokens 1 \
  --trace /tmp/metal.ndjson --trace-position 8000 --trace-layer 3 --trace-max-bytes 4000000 --trace-full
```

GPU版トレースに入るのは層ごとのinput・input_norm・mixer・after_mixer・post_norm・ffn.product・
ffn.intervened_product・ffn.output・layerと、最終のfinal_norm・logits。各レコードに `"engine":"metal"` が付く。
Q/K/V、Attentionのスコアと確率、DeltaNet内部、ffn.gate/up/siluはCPU版トレースだけにある
（行列命令版と位置分割版のAttentionは確率を作らない）。同じ位置のCPU版トレースと比べると、
どの演算でどれだけ誤差が入るかを確かめられる。実重みでは全レコードが最大3.1e-5以内で一致した。

## 7. 会話・保存と再開・systemのキャッシュ（ADR 0011）

```sh
./target/release/qwen-rust-inference metal-chat --library target/metal/libqwen_metal.dylib --model-dir models/qwen
# /continue /reset /save PATH /load PATH /quit
./target/release/qwen-rust-inference metal-chat --library target/metal/libqwen_metal.dylib \
  --model-dir models/qwen --system-file 長い文書.txt --save-state /tmp/doc.qrs </dev/null
./target/release/qwen-rust-inference metal-chat --library target/metal/libqwen_metal.dylib \
  --model-dir models/qwen --load-state /tmp/doc.qrs --prompt '質問' --stats
./target/release/qwen-rust-inference chat --model-dir models/qwen --load-state /tmp/doc.qrs --prompt '質問'
```

会話の進め方（テンプレート、anchor、pending、`--stats` の reused/prefill）はCPU版 `chat` と同じコードの写しで、
snapshotファイルも同じ形式・同じ検査を使う。GPUで保存した会話をCPU版で再開でき、その逆もできる。
保存時は全状態をホストへ読み出し、読込時はKVとDeltaNet状態をGPUへ書き込む。
`--capacity`（既定8192）を超えるターンは、会話を変えずに拒否する。GPUのエラー後は `/reset` が必要。

巻き戻しの仕組み：AttentionのKVは位置ごとの行なので、前の位置へ戻っても下の行はそのまま使える。
DeltaNetの状態は全トークンを吸収した固定サイズの行列（1層1MiB）で、途中の位置へは切り戻せない。
そこで戻る可能性がある場所だけでDeltaNet状態（18層で約19MiB）をホストへ控える。
戻る場所はsystemメッセージの末尾と直前のuserメッセージの末尾の2つで、戻すときは控えを書き戻して位置を下げるだけ。

systemメッセージは最初に単独でprefillする（`--no-cache` を除く）。その直後に保存したsnapshotは
長い文書のキャッシュになり、何度読み込んでも質問の分だけをprefillする。
M5では2637トークンの文書のprefillが3.6秒、保存ファイル86MB、読込0.24秒。
質問21トークン＋生成48トークンは、キャッシュありで0.92秒、なしで4.46秒、出力は同一だった
（[記録](measurements/metal-m5-chat.json)）。ファイルの大半はKV（FP32、1トークン24KiB）。

### 会話中の重み介入とトレース

```sh
./target/release/qwen-rust-inference metal-chat --library target/metal/libqwen_metal.dylib \
  --model-dir models/qwen --edit examples/zero-unit.json \
  --trace /tmp/chat.ndjson --trace-position 19 --trace-layer 3 --trace-max-bytes 8000000 --trace-full
# /edit PATH  /edit off  /edit（状態表示）  /trace（現在位置と対象）  /trace N  /trace +K
```

介入は計算済みの状態の一部なので、snapshotに保存し、CPU版chatでも同じ介入で再開する（CPU版chatも `--edit` と `/edit` に対応）。
会話の途中で介入を変えると、同じトークン列を位置0から新しい介入で計算し直し、system末尾とanchorの控えも取り直す。
メッセージ・pendingトークン・乱数は変えない。キャッシュに2種類の介入が混ざることはなく、
`/edit off` で介入なしの数値にそのまま戻る。

トレースの位置は会話全体の絶対位置で、その位置が計算されるたびに記録する（prefill・生成、巻き戻しや介入変更による再計算も含む）。
同じ位置を再計算した記録は `run_id` 末尾の `-passN`（ターン・読込・介入変更ごとに増える）で区別する。
バイト上限に達するとトレースだけを止め、会話は続ける。
実重みでは、層3・位置19の全レコードがCPU版 `infer` のトレースと4e-5以内で一致した。
CPU版 `chat` も同じオプションと `/trace` を持ち、指定位置だけを観測用のstepで計算する。
同じ会話をCPU版とGPU版でトレースすると、位置とpassが一致し、共通レコードの差は最大1.1e-4だった。
CPU版の記録にはAttention内部（Q/K/V、スコア、確率）・DeltaNet内部・FFNのgate/upも入るので、
GPUで会話しながら位置を決め、保存した会話をCPU版で読み込んでその位置の内部を見る、という使い方ができる。
