# RX 7900 XTX / WSL2 で動かす実験用ROCmバックエンド

**状態：RX 7900 XTX（gfx1100）・WSL2で検証済み**（[ADR 0013](adr-0013.md)、[記録](measurements/rocm-7900xtx.json)）。
小規模fixture・実重み（全層7演算と全状態）・2048/8192トークンの長文脈チェックがCPU版との比較で合格。
会話・保存と再開・systemキャッシュ・重み介入・トレースもCPU版と一致（[記録](measurements/rocm-7900xtx-chat.json)）。
Metal版のADR 0015〜0018の後も値・速度とも変化なし（[記録](measurements/rocm-7900xtx-regression.json)）。
decode約170 token/s、約1万トークンのprefillが4.4秒（3回測定の中央値）。
Rust側のエンジン（`src/device.rs`）はMetal版と共通。HIP側は `native/rocm/` の3ファイル。

## 1. WSLの準備

[AMDのWSL対応表（版固定の例：7.2）](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/compatibility/compatibilityrad/wsl/wsl_compatibility.html)
にRX 7900 XTXが掲載されている。同表はUbuntu 22.04/24.04.2と、
対応するWindows用WSL2ドライバを指定する。これは最新推奨版を意味しない。
導入時には採用する**一つのROCm版**の対応表と、そこからリンクされる
「Install Radeon software for WSL with ROCm」に従って組合せを揃える。
WindowsネイティブHIP SDKとは別の、WSL内Linux向けROCm開発環境を使う。
WSLに通常のLinux用カーネルドライバを追加しない。

検証した組合せ：Windows 11（10.0.26300）、WSL 2.6.3（カーネル6.6.87.2）、Ubuntu 24.04.2、
ROCm 7.2.4（hipcc 7.2.53211、rocBLAS 5.2、`rocdxg-roct` 1.2.0）、AMDドライバ 32.0.32015.7029。

必要な開発ツールはRust 1.85.1、C++17対応hipcc、HIPランタイム、rocBLASの
ヘッダとライブラリ。PyTorch、Python、Dockerは推論・ビルドに不要。
以下の情報を実機の測定記録に保存する。

```powershell
# Windows側PowerShell
wsl --version
Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion
```

```bash
# WSL内。hipccが別の場所ならROCM_PATHを設定する。
cat /etc/os-release
uname -r
/opt/rocm/bin/hipcc --version
HSA_ENABLE_DXG_DETECTION=1 /opt/rocm/bin/rocminfo
```

**WSL2ではGPUは `/dev/dxg`（librocdxg）経由でしか見えず、HSAランタイムは
`HSA_ENABLE_DXG_DETECTION=1` のときだけそこを探す。** 未設定だと `rocminfo` は
`hsa_init Failed, possibly no supported GPU devices` で終わる（`/dev/kfd` はWSLにはない）。
このリポジトリのブリッジは `/dev/dxg` があり変数が未設定なら自分で設定するので、
`rocm-*` コマンドには不要。`rocminfo` など他のツールには付けるか、`~/.bashrc` に
`export HSA_ENABLE_DXG_DETECTION=1` を書く。
`rocminfo`でRX 7900 XTX / gfx1100が見えることを先に確認する。初回検証はGPU 0だけを使用する。
`rocprofv3` はこの環境では異常終了した（カーネル時間は測れていない）。

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

`--features rocm` なしで `cargo build`/`cargo test --release` すると同じ
`target/release/qwen-rust-inference` がROCmコマンドなしで作り直される
（`device commands need --features rocm`）。測定前に `--features rocm` で作り直す。

`ROCM_PATH=/opt/rocm-X.Y.Z bash tools/build-rocm.sh`でインストール先を指定できる。
ビルド対象はgfx1100、FMAの自動縮約なし（新しいカーネルは `fmaf` を明示）。
計算はFP32で、行列はBF16のまま持ちカーネル内でf32へ戻して積算する。共有ライブラリは信頼できる
ローカルビルドだけを指定する。dlopenはそのファイル内のコードを実行する。
ABIバージョンとGPU/CPUブリッジ種別を検証し、失敗時にCPUへ自動切替しない。

小規模比較は固定Transformers由来の既存hybrid fixtureとも照合する。
全層の7演算、最終norm、logits、各位置のKV/畳み込み/再帰状態をCPUと比較し、
最初の不一致の位置・層・演算・要素番号で停止する。resetと全prefill分割も検証。
許容誤差は既存の小規模基準 `atol=2e-5, rtol=2e-4`。
fixtureは小さい次元（head_dim 4など）なので、ガード付きの同じGPUカーネルを通る。
タイルGEMM（16行以上のチャンク）だけは実重みの `device_real` で検証する。

## 3. 実重みの比較と生成

既にダウンロード済みの固定Qwen3.5-0.8B重みを使う（`tools/download-model.sh ./models/qwen`）。
推論中の自動ダウンロードはしない。モデルのrevision・ハッシュ・tensor検査はCPU版と共通。

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

実重み比較は全層7演算とlogits・状態をCPU版と照合する（実測の最大誤差1.43e-4）。
閾値は既存の実モデル基準 `atol=rtol=3e-3`。これだけで公式fixtureとの直接照合を
置き換えたことにはならない。CPU版は別途固定fixtureとの照合が済んでいる。
`device_real` は300トークンをチャンク7/64/128で流し、チャンク1と比べる（GEMV・タイルGEMM・
まとめAttentionと1行Attentionの両方を通る）。
`--ids` の比較は全分割を再実行するため、長いID列には向かない。長文脈は次の比較を使う。

```bash
cargo run --release --offline --locked --features rocm -- \
  rocm-check --library "$QWEN_ROCM_LIBRARY" --model-dir "$QWEN_MODEL_DIR" \
  --text-file 任意の長文.txt --tokens 2048 --report traces/rocm-long.json
```

テキストの先頭Nトークンを1,31,32,33,63,64,65,…,2^k±1,Nで区切ってprefillし、各区切りで
最終位置のlogits・全DeltaNet状態・全KVをCPU版と比較する。2048トークンで合格（最大誤差2.64e-4、
GPU側0.83秒、CPU側105秒）、8192トークンでも合格（最大誤差1.04e-3、GPU側3.5秒、CPU側873秒。
Metal版は1.83e-3、[metal.md](metal.md)）。

会話・保存と再開・systemのキャッシュ、重み介入とトレースもMetal版と同じコマンドで使える（ADR 0010、0011）。

```bash
cargo run --release --offline --locked --features rocm -- \
  rocm-chat --library "$QWEN_ROCM_LIBRARY" --model-dir "$QWEN_MODEL_DIR" --stats
# /continue /reset /save PATH /load PATH /edit PATH|off /trace N /quit
```

保存ファイルはCPU版・Metal版と共通で、ROCmで保存した会話をCPU版で再開でき、その逆もできる。
RX 7900 XTXでMetal版と同じ手順を確かめた（[記録](measurements/rocm-7900xtx-chat.json)）。

- 2ターンの会話：応答と `--stats` の reused/prefill（0/23、16/36）がCPU版と同じ。
- ROCmで保存→CPU版で再開、CPU版で保存→ROCmで再開：どちらも同じ続きとreused/prefill。
- systemキャッシュ：2659トークンの文書をprefillして保存（0.86秒、87MB）。読み込み0.18秒で、
  質問は22トークンだけprefillし0.17秒（`--no-cache` では2681トークン・1.02秒）。出力は同一。
- `--edit` と `/edit PATH|off`：会話の途中で付け外ししてもCPU版と同じ出力。外すと介入なしの応答に戻る。
  介入付きのsnapshotはどちらのエンジンで保存・再開しても同じ続きになる。
- `--trace-position`：CPU版chatのトレースと位置・passが一致し、共通レコードの差は最大1.37e-4
  （`rocm-infer` と CPU版 `infer` では6.9e-5）。

`rocm-infer` の生成はgreedy（`rocm-chat` はseed固定samplingも可）、終了は `<|im_end|>` / `<|endoftext|>` / 最大生成数 / 文脈容量。
`</think>`では終了しない。`--ids`でも生成可能で、その場合textはnull、ID列が返る。
結果はJSONを標準出力に表示する。ファイル保存は`--report`指定時だけで、
既存ファイルは上書きしない。入力IDや応答を含み得るためGitに入れない。

## 4. GPU上に何を保持し、どう分割するか

読み順は `src/device.rs`（Rust所有権と結線、Metal版と共通）→ `native/rocm/reference.h`
（演算の式、CPU版と1対1）→ `native/rocm/kernels.h`（GPUでの分割）→ `native/rocm/bridge.cpp`
（C ABIと振り分け）→ `src/device/cli.rs`（比較/生成）。

| 種類 | 所有者 | 形状・寿命 |
|---|---|---|
| 不変重み | DeviceModel | 全層を一度だけGPUに転送。行列はBF16（ADR 0006、`--f32-weights` でFP32）、他はFP32。埋め込みと語彙射影は共有。ホストにもBF16のまま残る（ADR 0012） |
| KV | DeviceRunのAttention状態 | 層ごとに[capacity,kv_heads,head_dim]、読める範囲はpositionまで |
| 畳み込み状態 | DeviceRunのDelta状態 | [channels,kernel-1]、古い→新しい順 |
| 再帰状態 | DeviceRunのDelta状態 | [value_heads,key_dim,value_dim]、トークン数に依存しない |
| 活性値 | DeviceRunのscratch | 事前確保して次のstepで再利用。状態とは別のBuffer |
| 履歴ID | 生成ループ | CPU上で保持。内部のKV/再帰状態ではない |

`reference.h` の本体は「1要素・1ヘッド・1列を1スレッド」の素朴な形で、CPUテストブリッジは
これだけを実行する。HIPでは軽い要素ごとの演算だけがこれを使い、重い演算は別のカーネルで分割する。
最初の実機実行ではすべてをこの本体で動かしていたため、AttentionはGPU全体で8スレッドしか動かず、
2060トークンのprefillに431秒かかった（ADR 0013の分析）。

| 演算 | HIPでの分割（gfx1100、wave32） |
|---|---|
| BF16射影（1〜15行） | 重み1行を1 wave。16バイト単位で読み、最大8トークンで同じ重みを共有 |
| BF16射影（16行以上） | 64行×32トークンのタイル。重みとXをLDSに置き、チャンクあたり重みを4回だけ読む |
| FP32射影（`--f32-weights`） | rocBLAS `sgemv`/`sgemm`（転置指定で重みのコピーなし） |
| RMSNorm | 1グループを1ブロック、ブロック内で和 |
| DeltaRule | 1列を4レーン（各32行）で持ち、状態はチャンクの間レジスタに置く |
| Attention（prefill） | KVヘッドごとに、そのKVを共有するQヘッド×行のwaveを1ブロックに。16位置ずつK/VをLDSに置き、butterflyで内積を集め、online softmax |
| Attention（decode） | 同じカーネルで位置を約64分割（32の倍数）し、`attention_combine` で合成 |
| その他 | `reference.h` の本体（要素・ヘッドごと） |

RDNA3の行列命令（WMMA）はFP16/BF16入力なので使っていない。活性値を丸めると値が変わるため
（ADR 0006の「値を変えない形式だけ」）。和の順序はCPUと異なり、ADR 0004の許容誤差で比較する。

単一ストリームで射影→正規化→状態更新→残差を順序付ける。
通常推論では層の途中でCPUへ戻さず、各トークンの全logitsだけを読み出す。
prefillは最大128トークンずつ（`--prefill-chunk`）まとめて各演算を1回ずつ発行し、
語彙射影とlogits読出しは最終位置だけで行う。途中は一定トークンごとにキューを同期し、GPU障害をその時点で検出する。

GPU状態を一部更新した後に失敗した場合、runはresetするまで再使用できない。
不正ID・容量超過は実行前に拒否する。resetは重みを再転送せず状態を再確保する。
容量上限はモデルの文脈長と、1 runあたりの状態予算4GiB（`MAX_RUN_BYTES`、重みは含まない）。
Qwen3.5-0.8BのKVは約24KiB/トークン。`--capacity` 省略時は入力長＋最大生成数。

## 5. 測定と未対応

rocm-inferのJSONはロード、ブリッジ初期化+upload、prefill、decode forward、
選択を含む生成時間を分離する。logitsのD2Hとストリーム同期を含む時間なので、
未完了のカーネルを速いと誤計測しない。初期状態の確保はprefill時間に含めない。
最初の生成トークンはprefillのlogitsから選ぶため、N個生成しても通常decode
forwardはN-1回。

RX 7900 XTX、BF16、チャンク128、64トークン生成、3回の中央値（ロード約2.6秒、upload約1.2秒）：

| 入力 | prefill | decode |
|---|---|---|
| 4トークン | 0.015秒 | 173 token/s |
| 331トークン | 0.115秒（2870 token/s） | 176 token/s |
| 1267トークン | 0.388秒（3270 token/s） | 175 token/s |
| 2060トークン | 0.652秒（3160 token/s） | 169 token/s |
| 9758トークン | 4.42秒（2210 token/s） | 151 token/s |

ADR 0013の前は、同じ入力で2060トークンのprefill 431秒・decode 2.2 token/sだった。生成トークンは前後で同一。

**測り方**：長い入力の前に、短い入力→数百→数千と長さを伸ばして所要時間の伸び方を確かめ、
`timeout` を付けて流す。2乗で伸びる箇所はそこで分かる。Windowsのタスクマネージャでは
GPUの「Compute」グラフで使用率を見られる。

decodeは1トークンあたり約5.8ms。重み約1.5GBを毎トークン読むので約260GB/s（7900 XTXは960GB/s）で、
カーネル発行や小さな演算の比率が大きい。CPU/GPU併存のためホストRAMにも重みが残る（約1.5GB）。
HIPのブリッジは `qr_wrap` を持たないので、Metalと違いGPUへコピーする。VRAM使用量はまだ測っていない。

### 16bit Attention（opt-in、[ADR 0014](adr-0014.md)）

`rocm-check`・`rocm-infer`・`rocm-chat` に `--attention-f16` を付けると、まとめてprefillのAttentionだけが
RDNA3の行列命令（WMMA、FP16入力・FP32積算）で計算される。Q・K・V・softmaxの重みをFP16に丸めるので
**計算結果が変わる**。decode・KVキャッシュ・保存ファイルはFP32のまま。既定では使わない。

| | FP32（既定） | 16bit |
|---|---|---|
| 実重み `--ids` の最大誤差 | 1.43e-4 | 8.86e-3 |
| 2048トークン長文脈の最大誤差 | 2.64e-4 | 7.96e-3（全区切りでgreedyトークン一致） |
| prefill 2060トークン | 0.652秒 | 0.605秒（1.08倍） |
| prefill 9758トークン | 4.42秒 | 3.18秒（1.39倍） |

64トークンの生成結果はすべての入力でFP32と同じだった（[記録](measurements/rocm-7900xtx-attention-f16.json)）。
照合の許容誤差は専用の `atol=rtol=2e-2`（`ATTENTION_F16_TOLERANCE`、実測最大の約2.3倍）で、
greedyトークンの一致も要求する。既存の `3e-3` では位置31で不一致になる。
head_dimが16の倍数でない（小規模fixtureなど）、Metal、CPUテストブリッジでは明示的なエラーになり、
黙ってFP32で計算することはない。1000トークン未満では速くならない。

Metal専用の `--matmul-relaxed`（ADR 0016）と `--metal-kernels` は、ROCmでは
`--matmul-relaxed is Metal only (spec/metal.md); not available on ROCm` のように名前付きのエラーで拒否する。
報告JSONの `kernels` はMetalだけの項目で、ROCmでは `null`。

未対応：FP16の重みや16bit KVキャッシュ、量子化、融合カーネル、HIP graph、GPU sampling、
GPU版のmixer内部のトレース（CPU版のトレースを使う）。
GPU版に渡して黙って無視はしない。

通常CIでは `cargo test --features rocm` がC++のCPUテストブリッジをビルドする。
非正方GEMV、独立fixture、全状態、分割、reset、独立run、部分失敗の拒否を検証する。
CPUブリッジは `reference.h` だけを実行し、`kernels.h` の分割は検証しない。
本物のGPUテストはignored。**CPUブリッジの成功をGPUの合格として報告しない。**
