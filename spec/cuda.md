# CUDA（RTX 4060 / WSL2）：GEMVの単体測定と推論器のバックエンド

**状態：推論器のCUDAバックエンドをRTX 4060（sm_89）・WSL2・CUDA 13.3で検証済み**（[ADR 0033](adr-0033.md)、[記録](measurements/cuda-rtx4060.json)、7章）。
小規模fixture・実重み・8192トークンまでの長文脈がCPU版との比較で合格。会話の保存と再開もCPU版と相互に一致する。
ROCm版のHIPソース（`native/rocm`）をそのままnvccでビルドし、Rust側のエンジン（`src/device.rs`）はMetal・ROCm版と共通。

1〜6章はその前に行った、BF16 GEMV単体の照合と測定（イシュー #46）。
コードは [`tools/bench_cuda_gemv/`](../tools/bench_cuda_gemv/README.md)（Rustの推論器とは独立したCUDA C++の1ファイル）。
記録は [3584×1024](measurements/cuda-rtx4060-gemv.json) と [1024×3584](measurements/cuda-rtx4060-gemv-1024x3584.json)。

## 1. WSLの準備

WSL2ではGPUドライバはWindows側のNVIDIAドライバが提供し、`/usr/lib/wsl/lib/libcuda.so.1` として見える。
WSL内にLinux用のドライバ（`cuda-drivers`、`nvidia-driver-*`）を入れてはいけない。
NVIDIAの `wsl-ubuntu` リポジトリからドライバ抜きのToolkitだけを入れる。
Toolkitの版は、`nvidia-smi` の「CUDA UMD Version」（ドライバが対応するCUDAの上限）以下にする。

```bash
nvidia-smi                          # Windowsドライバが見えることを確認
sudo tools/setup-cuda-wsl.sh        # cuda-keyring + cuda-toolkit-13-3（CUDA_VERSION=13-2 などで変更可）
```

検証した組合せ：Windows 11（10.0.26300.9818）、WSL 2.7.3.0（カーネル6.6.114.1）、Ubuntu 22.04.5、
NVIDIAドライバ 610.88（CUDA UMD 13.3）、CUDA Toolkit 13.3（nvcc V13.3.73）。
GPUはGeForce RTX 4060（AD107、sm_89、24 SM、L2 24 MiB、128ビットGDDR6、8 GB、公称272 GB/s）。

## 2. ビルドと実行

```bash
tools/bench_cuda_gemv/run.sh spec/measurements/cuda-rtx4060-gemv.json
BENCH_ARGS="--rows 1024 --cols 3584" tools/bench_cuda_gemv/run.sh spec/measurements/cuda-rtx4060-gemv-1024x3584.json
```

ビルドは `nvcc -O3 -std=c++17 -arch=native --fmad=false -lineinfo -Xcompiler -ffp-contract=off`。
`--fmad=false` と `-ffp-contract=off` によって、明示した `fmaf` だけが融合積和になる。
これでCPUでの加算順の再現がビット単位で一致する。照合が一つでも失敗すると、計時はせずに終了コード3で終わる。

## 3. カーネルと照合

1 warpが1出力行を受け持つ。各laneはBF16をFP32へ広げて `fmaf` で部分和を作り、
`__shfl_down_sync` で距離16、8、4、2、1と集約し、lane 0が `y[row]` を書く。1ブロックは256スレッド（8行）。

| 名前 | 変更点 |
|---|---|
| `scalar` | 基準。lane `l` は成分 `l, l+32, …` を2バイトずつ読む |
| `vec8` | まとめ読み。16バイト（8成分）ずつ。lane `l` は塊 `l, l+32, …` |
| `pf2` / `pf4` | 先行ロード。`vec8` の16バイトロードを2個／4個まとめて発行してから積和する。加算順は `vec8` と同じ |

16バイトのロードには、行の先頭が16バイト境界に揃っている必要がある（`cols % 8 == 0` かつ基底アドレスが揃っている）。
揃わない行は `scalar` の経路に落ちる。

照合入力は決定的（SplitMix64、BF16は最近接偶数丸め）で、CPUとGPUは同じビット列を使う。

- 全要素1（3584×1024、全行がちょうど1024）
- 非一様な符号付き値（3584×1024、1024×3584）
- 32の倍数でない幅：45、100（8で割ると余り4）、1000（`pf*` の端数処理）、8、1
- 基底を2バイト、8バイトずらしたもの
- `y` の後ろの番兵が書き換わっていないこと

判定は二重にした。

1. 各カーネルの加算順と集約の木をCPUで再現した値と、**ビット単位で一致**すること
2. 倍精度の参照値に対して `|gpu − ref| ≤ 1e-6 + 1e-5 × Σ|w x|` を満たすこと
   - 1 laneの加算は ⌈cols/32⌉ 回、集約が5段なので、FP32の丸め誤差は約 `(cols/32 + 5) × 2^-24 × Σ|w x|` で抑えられる
   - cols ≤ 3584 で7e-6以下、`atol` は打ち消し合って0に近い行のため

結果は全40件（10通りの入力×4カーネル）とも合格し、ビット単位でも一致した。
最大絶対差は3584×1024で5.4e-6、1024×3584で1.7e-5、許容誤差に対する比は最大0.005。

## 4. 計時の方法

| 項目 | 方法 |
|---|---|
| カーネル時間 | 200回の起動を一度CUDA Graphに取り込み、その再生をcudaEventの対で挟む。ウォームアップ後に15回測り、1回あたりの中央値と範囲を出す |
| hot | 同じ重み7,340,032バイトを毎回読む（L2 24 MiBに収まる） |
| cold | 重みの複製を、max(8×L2, 256 MiB) 以上になるまで循環させる。3584×1024では37個、271,581,184バイト |
| 発行込み | 通常のストリーム起動を200回連続、ホストでの発行時間、1回の起動＋`cudaStreamSynchronize` を別に記録 |
| 素の読み出し | 演算なしで `uint4` を読むだけのカーネル。cold作業集合全体を1回の起動で読む |
| 換算帯域 | `2 × rows × cols / 1回の秒数`。重みのバイト数のみで、プロファイラの実転送量ではない |

WSL2では、空カーネルの通常起動がデバイス上でも1回約9.5 µsかかる。CUDA Graphの1ノードは約0.8 µsで済む。
hotのGEMVは7 µs前後なので、通常起動のまま測ると、測っているのは発行の費用になる。そのため、カーネル時間はGraphで測り、発行込みの値は別の項目にした。

Nsight Compute（`ncu`）はWindows側でGPUカウンタへのアクセスを許可しないと `ERR_NVGPUCTRPERM` になる。
許可はNVIDIAコントロールパネル →「開発者」→「GPUパフォーマンスカウンターの管理」で行う。
今回は許可しておらず、実転送量は載せていない。

## 5. 結果（RTX 4060）

3584×1024（15回の中央値、括弧内は換算帯域GB/s）：

| カーネル | hot（Graph） | cold（Graph） | hot（通常起動） | cold（通常起動） |
|---|---|---|---|---|
| `scalar` | 10.11 µs（726） | 30.73 µs（239） | 14.79 µs（496） | 31.49 µs（233） |
| `vec8` | 6.89 µs（1,065） | 29.86 µs（246） | 12.57 µs（584） | 31.33 µs（234） |
| `pf2` | 7.00 µs（1,049） | 29.91 µs（245） | 13.17 µs（557） | 32.69 µs（225） |
| `pf4` | 6.99 µs（1,050） | 29.79 µs（246） | 12.75 µs（576） | 31.36 µs（234） |

素の読み出しは252 GB/s（中央値）。空カーネルはGraphで0.75 µs、通常起動で9.54 µs。
ホストでの発行は1回約6 µs、起動＋同期の往復は約38 µs。

1024×3584（Graph）：`scalar` 10.61 / 30.77 µs、`vec8` 7.38 / 30.03 µs、`pf2` 8.68 / 30.19 µs、`pf4` 9.33 / 30.41 µs（hot / cold）。

読み取れること：

- **coldは帯域で決まる。** 3584×1024の `vec8` は246 GB/s。素の読み出しの約98%、公称272 GB/sの約90%にあたる。
  基準の `scalar` でも239 GB/sで、まとめ読みの効果は約3%しかない。
- **先行ロードはRTX 4060では効かない。** coldの差は測定のばらつき（±1%程度）の中にある。
  24 SMに448ブロックが乗るので、ロードを重ねなくても、帯域を埋めるだけの要求がすでに出ている。
  1024×3584のhotでは、`pf2`・`pf4` のほうが遅かった（7.4 → 8.7、9.3 µs）。
  1行が3584成分と長く、レジスタの増加で同時に動けるwarpが減った可能性がある（未確認）。
- **hotはL2から読む。** `vec8` で約1,065 GB/s。ここでは2バイトずつ読む `scalar` との差がはっきり出る（10.1 → 6.9 µs）。
- **WSL2の発行費用。** 通常起動ではhotの時間の約半分が発行の費用になる。
  推論器に組み込むときは、CUDA Graphにまとめるか、起動回数を減らす必要がある。
  ROCm（[ADR 0019](adr-0019.md)）では、HIP Graphにしても1ノードあたりの費用が2.3 µsから下がらなかった。CUDAではGraphで約1/12になる。

## 6. Metal・ROCmとの比較（3584×1024、1位置、重みはBF16、積和はFP32）

| | CUDA（このページ） | Metal（[ADR 0009](adr-0009.md)、[記録](measurements/metal-m5-norm.json)） | ROCm（[ADR 0019](adr-0019.md)、[記録](measurements/rocm-7900xtx-decode.json)） |
|---|---|---|---|
| 機種 | RTX 4060 | Apple M5 | RX 7900 XTX |
| 公称帯域 | 272 GB/s | ― | 960 GB/s |
| 素の読み出し | 252 GB/s | 132 GB/s | ― |
| cold 1回の時間 | 29.9 µs（`vec8`） | 56.0 µs | 約12.1 µs（4ロード） |
| cold 換算帯域 | 246 GB/s（素の読み出しの98%） | 131 GB/s（同99%） | 1ロード321、4ロード605 GB/s（公称の63%） |
| 先行ロードの効果 | なし（245〜246 GB/s） | ― | 1.9倍 |
| hot | 6.89 µs（L2 24 MiB） | ― | ―（Infinity Cache 96 MBの値はdecode全体の分析のみ） |

比較条件の違い：

- **Metal**：リポジトリの `gemv_bf16` を、bridgeと同じ起動形で、一つのcommand bufferに多数のdispatchを入れて測った。
  64個の複製（512 MiB以上）を循環させ、ウォームアップ後の4回の最良値をとっている。CUDAは15回の中央値。
  Metalのdispatchは費用が小さく（空で0.48 µs）、CUDAのGraph再生（0.75 µs）と同程度の条件になる。
- **ROCm**：リポジトリにない単体HIPプログラムで、約1 GBの異なる重みを読んだ。
  発行の形と統計の取り方は記録に残っていない。カーネルはCUDAの `pf4` と同じ形
  （wave幅32、1 waveが1行、1ブロック256スレッドで8行、16バイトロードを4個先行、wave内集約）。
  時間は換算帯域605 GB/sからの逆算。
- **公称帯域に対する比**は、CUDA約90%、ROCm約63%。RTX 4060では、基準形のロードの数ですでに帯域が埋まっている。
  ROCmで先行ロードが効いたのは、帯域が3.5倍あるため、同じ形では発行されるロードの数が足りなかったからだと考えられる（推測、未検証）。

## 7. 推論器のCUDAバックエンド（ADR 0033）

ROCm版のブリッジ（`native/rocm/bridge.cpp`・`kernels.h`・`reference.h`）を、nvccでそのままビルドする。
`native/cuda/` の2つのヘッダが、HIPとrocBLASの名前をCUDAとcuBLASへ対応させる。
gfx1100のwaveは32 laneで、CUDAのwarpと同じ幅なので、カーネルの分割は変えていない。
差は `#ifdef QR_CUDA` の数か所だけ。

- **shuffleのマスク**：HIPの `__shfl_xor` はマスクを取らない。CUDAの `__shfl_xor_sync` は、マスクに含めたlaneが全員その命令に来ることを要求する。
  kernels.hのshuffleは、ほぼすべてwarp全体がそろって呼ぶ。そうでないのはDeltaRuleの列（P laneの組）だけで、
  組ごとに抜けるので、自分の組のlaneをマスクに指定する（`qr_shfl_xor_group`）。
- **ブロック間の読み出し**：最後のブロックが他ブロックの結果を読む箇所（`gemv_bf16_add_norm`、`delta_mixer_split`）は、
  HIPのデバイス範囲の原子的読み出しの代わりに `__ldcg`（L1を通らずL2から読む）を使う。
- **未対応**：`--attention-f16`（RDNA3のWMMA命令、ADR 0014）。CUDA版はこのオプションを拒否する。
- cuBLAS（`--f32-weights` のときだけ使う）はTF32を使わない設定にしている。

```bash
bash tools/build-cuda.sh                      # target/cuda/libqwen_cuda.so（CUDA_ARCH=sm_89 などで指定可）
cargo build --release --offline --locked --features cuda
export QWEN_CUDA_LIBRARY="$PWD/target/cuda/libqwen_cuda.so"
export QWEN_DEVICE_LIBRARY="$QWEN_CUDA_LIBRARY" QWEN_MODEL_DIR="$PWD/models/qwen"
cargo test --release --offline --locked --features cuda --test device_gpu -- --ignored --test-threads 1
cargo run --release --offline --locked --features cuda -- \
  cuda-check --library "$QWEN_CUDA_LIBRARY" --model-dir "$QWEN_MODEL_DIR" --ids 16,10,16,28
cargo run --release --offline --locked --features cuda -- \
  cuda-infer --library "$QWEN_CUDA_LIBRARY" --model-dir "$QWEN_MODEL_DIR" --prompt '1+1は？'
```

コマンドとオプションは `rocm-*` と同じ（`cuda-check`・`cuda-infer`・`cuda-chat`・`cuda-verify`・`cuda-speculative`）。
プロファイラとA/B切り替えの環境変数は `QWEN_CUDA_PROFILE`・`QWEN_CUDA_FIXED_GEMV`（ROCm版の `QWEN_ROCM_*` に対応）。

照合（ROCm版と同じコマンドと許容誤差）：

| 項目 | CUDA（RTX 4060） | ROCm（RX 7900 XTX） |
|---|---|---|
| `device_gpu`（小規模＋実重み4種） | 合格 | 合格 |
| `--ids 16,10,16,28`（1,364件） | 1.45e-4 | 1.52e-4 |
| 長文脈2048トークン | 1.54e-4 | 2.63e-4 |
| 長文脈8192トークン | 2.47e-4（GPU側5.9秒） | 1.04e-3 |
| 会話：CPU版との生成ID、保存と再開の相互 | 一致 | 一致 |

速度（BF16、チャンク128、64トークン生成、3回の中央値。入力とトークン数はROCm版の記録と同じ）：

| 入力 | prefill | decode | ROCmのdecode |
|---|---|---|---|
| 4トークン | 0.011秒 | 130 token/s | 329 |
| 331トークン | 0.121秒 | 130 | 330 |
| 1,267トークン | 0.444秒 | 128 | 315 |
| 2,060トークン | 0.750秒 | 127 | 320 |
| 9,872トークン | 4.99秒 | 113 | 290 |

decodeは1トークン約7.7ms。重み1.5GBを毎トークン読む換算で約195GB/s。
公称272GB/sの約72%、素の読み出し（252GB/s）の約77%にあたる（ROCmは公称の約52%）。
`QWEN_CUDA_PROFILE=host` では、時間の85%がGPUの完了待ちで、1トークンあたり130回の発行だった。
WSL2の通常起動は1回約9.5µsかかる（5章）。全部の発行がそれを払えば約1.2msになるが、今回の測定では内訳を分けていない。

**その後の調査**（ADR 0033の「Follow-ups」、Nsight Systems）：

- **CUDA Graph（不採用）**：通常の起動でも、カーネルの間の空きは1ステップ226µs（中央値1.4µs）しかなかった。
  発行の費用は、前のカーネルの実行と重なって隠れている。
  1ステップ分をGraphに記録してその場で更新すると、空きは23µsに減った。
  しかしホストで記録と更新を終えるまでGPUが待つので、読み出しから最初のカーネルまでが99µsから498µsに延びた。
  1ステップは7.50→7.58msで、速くならない。カーネルが7.1msを占め、Graphで取り戻せるのは最大でも約4%。
- **argmaxの2段化（採用）**：greedyの選択（演算16）が1トークン240µsかかっていた。1ブロックで248,320個を読んでいたため。
  4,096個ずつの区間を別のブロックで調べ、2つ目のカーネルで区間の結果をまとめるようにした。
  大小と同値の規則は全順序なので、まとめ方を変えても選ぶ位置は同じになる（ビット単位で一致）。
  240→8µs、1ステップ7.50→7.20msで、decodeは4トークン入力131→136、1,267トークン127→133、9,872トークン112→115 token/s。
  生成IDは同一。同値・NaN・無限大・区間の境目を含む入力で、CPUの規則と照合するテストを追加した（GPUとCPUテストブリッジの両方）。
  ROCm版も同じカーネルを使うが、RX 7900 XTXでは測っていない。
