# Apple Silicon（Metal）で動かす実験用GPUバックエンド

**状態：Apple M5 / macOS 27.3で小型fixture・実重み比較・生成を確認済み。**
記録は [metal-m5.json](measurements/metal-m5.json)・[metal-m5-bf16.json](measurements/metal-m5-bf16.json)、
設計は [ADR 0005](adr-0005.md)・[ADR 0006](adr-0006.md)（BF16重みとまとめてprefill）。
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
その `contents` アドレスをRustに渡す。Rust側はROCm版と同じくポインタを直接読まない。
全カーネルを1つのcompute encoderへ直列に積み、logits読出し時だけcommitして完了を待つ。
行列は値が変わらない場合だけBF16で持ち（公開重みは全行列が該当）、カーネル内でf32へ戻して
f32で積算する。GEMVは1行を32レーンのSIMD groupで分担して `simd_sum` で足す。重みの転置コピーはしない。
prefillは最大128トークンずつまとめ、射影を行列×行列にして重みを1チャンクで1回だけ読む。
8トークン以上は32×32タイルを `simdgroup_float8x8` で計算する。畳み込み履歴とDeltaNet状態は
1スレッド内でトークン順に更新するので、状態の進み方はdecodeと同じ。
Attentionの読出しは (位置,ヘッド) ごとのスコア → ヘッドごとのthreadgroup softmax →
(次元,ヘッド) ごとの加重和の3段に分けた。1スレッドの直列処理は「文脈長×head_dim×2」から
「head_dim」（スコア）または「文脈長」（加重和）に減ったが、加重和は文脈長に比例して残る。
prefillは最終位置だけ語彙射影する。Attentionの読出しだけはまだトークンごとに発行している。

## 5. 測定と未対応

M5での1回測定（ウォームアップなし、BF16重み、チャンク128）：

| 入力 | prefill | decode |
|---|---|---|
| 454トークン | 615 token/s | 49.9 token/s |
| 2644トークン | 344 token/s（7.7秒） | 43.1 token/s |
| 10141トークン | 115 token/s（88秒） | 28.8 token/s |

FP32重み・1トークンずつだった前版は、2644トークンでprefill 39.7 token/s・decode 28.8 token/s、
10141トークンでprefill 306秒だった。`--f32-weights` と `--prefill-chunk 1` で前と同じ経路を選べる。
長くなるとprefillが遅くなるのは、トークンごとのAttention読出しが文脈長に比例するため。
decodeはlogits全体の読出しとGPU完了待ちを含む。CPU版の記録（約2.6 token/s）は別条件の測定。
GPU上の行列は約1.5GBだが、CPU版のFP32重み（約3GB）もホストに残る。

未対応：Attention読出しのまとめ処理、融合カーネル、量子化（値が変わる形式）、GPU sampling、
重み介入、NDJSONトレース、複数ターン会話、状態保存/復元。
