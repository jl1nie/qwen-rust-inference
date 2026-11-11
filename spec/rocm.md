# RX 7900 XTX / WSL2 で動かす実験用ROCmバックエンド

**状態：実装済み、GPUでのコンパイル・実行・速度測定は未検証。**
このクラウド環境にはHIPコンパイラもAMD GPUもない。CPUテストブリッジは
同じカーネル本体の算術を検証するが、HIPのコンパイル、rocBLASの配置、
非同期動作やWSLドライバの検証の代用にはならない。GPUの合格記録はまだない。

## 1. WSLの準備

[AMDのWSL対応表（版固定の例：7.2）](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/compatibility/compatibilityrad/wsl/wsl_compatibility.html)
にRX 7900 XTXが掲載されている。同表はUbuntu 22.04/24.04.2と、
対応するWindows用WSL2ドライバを指定する。これは最新推奨版を意味しない。
導入時には採用する**一つのROCm版**の対応表と、そこからリンクされる
「Install Radeon software for WSL with ROCm」に従って組合せを揃える。
WindowsネイティブHIP SDKとは別の、WSL内Linux向けROCm開発環境を使う。
WSLに通常のLinux用カーネルドライバを追加しない。

必要な開発ツールはRust 1.85.1、C++17対応hipcc、HIPランタイム、rocBLASの
ヘッダとライブラリ。PyTorch、Python、Dockerは推論・ビルドに不要。
以下の情報を実機の測定記録に保存する（GPU検証レポートとは別）。

```powershell
# Windows側PowerShell
wsl --version
wsl -l -v
Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion
```

```bash
# WSL内。hipccが別の場所ならROCM_PATHを設定する。
cat /etc/os-release
uname -r
/opt/rocm/bin/hipcc --version
/opt/rocm/bin/rocminfo
```

`rocminfo`でRX 7900 XTX / gfx1100が見えることを先に確認する。
Windows側とWSL側のバージョンが不明な状態でインストールコマンドを
一律に実行しない。初回検証はGPU 0だけを使用する。

## 2. ビルドと小規模比較

リポジトリをWSLのLinuxファイルシステムに置き、ルートで実行する。
通常のCargoビルドにはROCmは不要。任意機能でRust側のローダーを有効にし、
HIP側の共有ライブラリは明示コマンドで作る。新しいRust依存は追加していない。

```bash
bash tools/build-rocm.sh
cargo build --release --offline --locked --features rocm
export QWEN_ROCM_LIBRARY="$PWD/target/rocm/libqwen_rocm.so"
export QWEN_DEVICE_LIBRARY="$QWEN_ROCM_LIBRARY"   # tests/device_gpu.rs
cargo test --release --offline --locked --features rocm \
  --test device_gpu device_tiny -- --ignored --exact
mkdir -p traces
cargo run --release --offline --locked --features rocm -- \
  rocm-check --library "$QWEN_ROCM_LIBRARY" --report traces/rocm-tiny.json
```

`ROCM_PATH=/opt/rocm-X.Y.Z bash tools/build-rocm.sh`でインストール先を指定できる。
ビルド対象はgfx1100、FP32、FMAの自動縮約なし。共有ライブラリは信頼できる
ローカルビルドだけを指定する。dlopenはそのファイル内のコードを実行する。
ABIバージョンとGPU/CPUブリッジ種別を検証し、失敗時にCPUへ自動切替しない。

小規模比較は固定Transformers由来の既存hybrid fixtureとも照合する。
全層の7演算、最終norm、logits、各位置のKV/畳み込み/再帰状態をCPUと比較し、
最初の不一致の位置・層・演算・要素番号で停止する。resetと全prefill分割も検証。
許容誤差は既存の小規模基準 `atol=2e-5, rtol=2e-4`。

## 3. 実重みの比較と生成

既にダウンロード済みの固定Qwen3.5-0.8B重みを使う。自動ダウンロードはしない。
モデルのrevision・ハッシュ・tensor検査はCPU版と共通。まず文字列を介さず比較する。

```bash
export QWEN_MODEL_DIR="$PWD/models/qwen"
cargo test --release --offline --locked --features rocm \
  --test device_gpu device_real -- --ignored --exact
cargo run --release --offline --locked --features rocm -- \
  rocm-check --library "$QWEN_ROCM_LIBRARY" --model-dir "$QWEN_MODEL_DIR" \
  --ids 16,10,16,28 --report traces/rocm-real.json
cargo run --release --offline --locked --features rocm -- \
  rocm-infer --library "$QWEN_ROCM_LIBRARY" --model-dir "$QWEN_MODEL_DIR" \
  --prompt '1+1は？' --max-new-tokens 16 --capacity 1024 \
  --report traces/rocm-infer.json
```

実重み比較は全層7演算とlogits・状態をCPUのFP32参照版と照合する。
閾値は既存の実モデル基準 `atol=rtol=3e-3`。これだけで公式fixtureとの直接照合を
置き換えたことにはならない。CPU版は別途固定fixtureとの照合が済んでいる。
長文脈32/128/512のGPU比較やBF16計算は未検証。まず短い複数入力を通す。
比較処理は全分割を再実行するため、長いID列を渡すと検証時間が大きく増える。

生成はgreedy、終了は `<|im_end|>` / `<|endoftext|>` / 最大生成数 / 文脈容量。
`</think>`では終了しない。`--ids`でも生成可能で、その場合textはnull、ID列が返る。
結果はJSONを標準出力に表示する。ファイル保存は`--report`指定時だけで、
既存ファイルは上書きしない。入力IDや応答を含み得るためGitに入れない。

## 4. GPU上に何を保持するか

読み順は `src/device.rs`（Rust所有権と結線、Metal版と共通）→ `native/rocm/bridge.cpp`
（演算本体）→ `src/device/cli.rs`（比較/生成）。CPU版の数式と対比して読む。

| 種類 | 所有者 | 形状・寿命 |
|---|---|---|
| 不変重み | DeviceModel | 全層を一度だけFP32でGPUに転送。埋め込みと語彙射影は共有 |
| KV | DeviceRunのAttention状態 | 層ごとに[capacity,kv_heads,head_dim]、読める範囲はpositionまで |
| 畳み込み状態 | DeviceRunのDelta状態 | [channels,kernel-1]、古い→新しい順 |
| 再帰状態 | DeviceRunのDelta状態 | [value_heads,key_dim,value_dim]、トークン数に依存しない |
| 活性値 | DeviceRunのscratch | 事前確保して次のstepで再利用。状態とは別のBuffer |
| 履歴ID | 生成ループ | CPU上で保持。内部のKV/再帰状態ではない |

Rustのrow-major重み[out,in]は、rocBLASからはcolumn-major[in,out]と見える。
転置GEMVを呼ぶことで、重みの転置コピーなしに同じ射影を計算する。
単一ストリームで射影→正規化→状態更新→残差を順序付ける。
通常推論では層の途中でCPUへ戻さず、各トークンの全logitsだけを読み出す。
prefillも現段階は逐次stepであり、各位置でlogitsを読み出す。

GPU状態を一部更新した後に失敗した場合、runはresetするまで再使用できない。
不正ID・容量超過は実行前に拒否する。resetは重みを再転送せず状態を再確保する。
容量上限は短文脈用に4096、デフォルト1024（モデル上限も適用）。

## 5. 測定と未対応

rocm-inferのJSONはロード、ブリッジ初期化+upload、prefill、decode forward、
選択を含む生成時間を分離する。logitsのD2Hとストリーム同期を含む時間なので、
未完了のカーネルを速いと誤計測しない。初期状態の確保はprefill時間に含めない。
最初の生成トークンはprefillのlogitsから選ぶため、N個生成しても通常decode
forwardはN-1回。EOS/容量制限ならさらに少ない。1回測定・ウォームアップなしの
参考値であり、CPUとの正式な比較は同じID・生成条件で複数回測定して行う。
CPU/GPU併存のためホストRAMにもFP32重みが残る。VRAM使用量はまだ測っていない。

未対応：BF16/FP16、量子化、GEMM/chunk prefill、融合カーネル、GPU sampling、
GPU版の重み介入・NDJSON全中間値・複数ターン会話・状態保存/復元。
既存CPU版のこれらの機能はそのまま使える。GPU版に渡して黙って無視はしない。
現在の正規化とAttentionはヘッド単位の単純ループで、速度の最終形ではない。
実機で一致を確認後、削減演算の並列化、rocBLASの選択、起動回数の削減へ進む。

通常CIでは `cargo test --features rocm` がC++のCPUテストブリッジをビルドする。
非正方GEMV、独立fixture、全状態、分割、reset、独立run、部分失敗の拒否を検証する。
本物のGPUテストはignored。**CPUブリッジの成功をGPUの合格として報告しない。**
