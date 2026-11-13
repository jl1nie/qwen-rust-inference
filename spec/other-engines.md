# 他の実装はどう速くしているか

READMEの構成の(4)です。同じ系統のモデル（Gated DeltaNetとAttentionの混成、Qwen3.5／Qwen3-Next系）を、
ほかの実装がどう速くしているかを、ソースコードから確かめて並べます。
比べる相手は、汎用のllama.cpp、Apple向けで汎用のMLX（mlx-lm）、特定モデル専用のStrataの3つです。

調べた版（2026-11-13時点）：

- llama.cpp：master `5676dfbe`
- MLX 0.32.4（`36969c09`）とmlx-lm 0.32.2（`93e6d4a6`）。M5での比較測定に使ったのと同じ版です
- Strata：main `13eedf55`（2026-11-12）

出典は各リポジトリのファイルと行です。速度の数値は、明記したもの以外はそれぞれの作者の自己申告で、こちらでは再現していません。
内容は調べた時点のもので、各プロジェクトは変わり続けています。


## 一覧

| | このリポジトリ | llama.cpp | MLX（mlx-lm） | Strata |
|---|---|---|---|---|
| 位置づけ | 1モデル専用・教材 | 汎用（多数のモデル・HW） | 汎用（Apple Silicon） | 1モデル専用（Qwen3.8-Flash-Next、MoE 125B） |
| 重み | BF16のまま | 量子化（Q8_0、Q4_K、IQ系など）が主 | BF16、量子化も可 | 量子化パック（Q2〜IQ3） |
| 途中の計算 | FP32 | 型による（F16・BF16・F32、活性値もq8_1に量子化して掛ける経路あり） | BF16（DeltaNetの状態などはFP32） | 混在（DeltaNetの状態はFP32、出力はFP16） |
| DeltaNet（prefill） | 1トークンずつ（FP32状態） | CUDA・一部のHIPでチャンク形式、Metal・CPUは1トークンずつ | 1トークンずつ（FP32状態） | 既定は1トークンずつ、チャンク形式はopt-in |
| 行列積の演算器 | Metal 4テンソル演算（FP32入力）、HIPは自作 | Metalはsimdgroup行列またはテンソル演算、CUDA/HIPはMMQなど | BF16でM5のNAX（テンソル演算） | 自作CUDA/HIP＋ggmlのMMQ |
| MTP・投機的decode | 採用率の計測だけ（下書き1で68%、[ADR 0022](adr-0022.md)）、生成には未使用（[候補](roadmap.md#3-decodemtpによる投機的デコードmetalrocm共通)） | あり（MTP、別モデル、n-gramなど） | このモデルでは不可 | あり（MTPで最大3トークン） |
| 正しさの確かめ方 | CPU版と全層・全状態を照合 | モデルごとの比較とテスト | — | DeltaNetカーネルをFP64の参照と照合 |

## llama.cpp

**モデルの組み立て**：Qwen3.5のdenseモデルは `src/models/qwen35.cpp` で、DeltaNet層とAttention層を交互に組みます
（[src/models/qwen35.cpp#L176-L179](https://github.com/ggml-org/llama.cpp/blob/5676dfbe01e7ebe0387913680eee0ed39bf4a363/src/models/qwen35.cpp#L176-L179)）。DeltaNet層は、`ggml_ssm_conv` とSiLU、Q/KのL2正規化、
状態更新、`z` によるゲート付きRMSNorm、出力射影の順で、このリポジトリの `src/deltanet.rs` と同じ流れです
（[src/models/qwen35.cpp#L425-L489](https://github.com/ggml-org/llama.cpp/blob/5676dfbe01e7ebe0387913680eee0ed39bf4a363/src/models/qwen35.cpp#L425-L489)）。

**状態更新は1つの演算 `ggml_gated_delta_net`**：delta ruleを1つのggml演算にまとめ、各バックエンドが専用カーネルを持ちます
（[ggml/include/ggml.h#L2424](https://github.com/ggml-org/llama.cpp/blob/5676dfbe01e7ebe0387913680eee0ed39bf4a363/ggml/include/ggml.h#L2424)）。この演算は投機的decodeの巻き戻し用に、状態の控えを複数持てます。
以前は汎用の演算を組み合わせてチャンク形式をグラフで書いていましたが、削除されました（[PR #27877](https://github.com/ggml-org/llama.cpp/pull/27877)で無効化、[#32078](https://github.com/ggml-org/llama.cpp/pull/32078)で削除）。
いまチャンク形式はバックエンドのカーネルの中にだけあります。

- **CUDA**：1トークンずつのカーネル（[ggml/src/ggml-cuda/gated_delta_net.cu#L75](https://github.com/ggml-org/llama.cpp/blob/5676dfbe01e7ebe0387913680eee0ed39bf4a363/ggml/src/ggml-cuda/gated_delta_net.cu#L75)）と、
  チャンク（16トークン）単位で行列演算器を使うカーネルがあります（[ggml/src/ggml-cuda/gated-delta-net-mma.cu](https://github.com/ggml-org/llama.cpp/blob/5676dfbe01e7ebe0387913680eee0ed39bf4a363/ggml/src/ggml-cuda/gated-delta-net-mma.cu)、[PR #29353](https://github.com/ggml-org/llama.cpp/pull/29353)）。
  後者は状態をFP32で持ち、行列演算器への入力を**BF16の上位と下位（hi/lo）の2つに分けて**精度を保ちます
  （同ファイルL12、L30-L39）。使うのはヘッドの次元128・キーのヘッド16・64トークン以上・特定のGPUのときだけです（L386-L418）。
  Qwen3.5-0.8Bは次元128・キーのヘッド16なので、条件に合います。
- **HIP（ROCm）**：同じチャンク形式のカーネルを、RDNA3.5かつ2048トークン以上のときだけ使います（同ファイルL394）。
  RX 7900 XTX（RDNA3）では1トークンずつのカーネルです。
- **Metal**：1トークンずつのループで、内積は `simd_sum` です（[ggml/src/ggml-metal/kernels/gated_delta_net.metal#L75](https://github.com/ggml-org/llama.cpp/blob/5676dfbe01e7ebe0387913680eee0ed39bf4a363/ggml/src/ggml-metal/kernels/gated_delta_net.metal#L75)）。
  状態キャッシュへのコピーと融合しています（[ggml/src/ggml-metal/ggml-metal-fusion.cpp#L833](https://github.com/ggml-org/llama.cpp/blob/5676dfbe01e7ebe0387913680eee0ed39bf4a363/ggml/src/ggml-metal/ggml-metal-fusion.cpp#L833)）。

**行列積**：

- CUDA/HIPでは、量子化した重みを、q8_1に量子化した活性値と内積します。重みをメモリ上に展開し直さない
  （[ggml/src/ggml-cuda/mmvq.cu#L34](https://github.com/ggml-org/llama.cpp/blob/5676dfbe01e7ebe0387913680eee0ed39bf4a363/ggml/src/ggml-cuda/mmvq.cu#L34)）。バッチの大きさでMMVQ・MMQなどを選び（[ggml/src/ggml-cuda/ggml-cuda.cu#L2486-L2507](https://github.com/ggml-org/llama.cpp/blob/5676dfbe01e7ebe0387913680eee0ed39bf4a363/ggml/src/ggml-cuda/ggml-cuda.cu#L2486-L2507)）、
  タイルの形はGPUの世代ごとに設定しています（`mmq-config-{ampere,rdna3,cdna,...}.cuh`）。
- Metalでは、量子化ブロックをthreadgroupメモリに展開してから `simdgroup_multiply_accumulate`、
  テンソル演算のあるGPUでは `mpp::tensor_ops::matmul2d` で掛けます（[ggml/src/ggml-metal/kernels/mul_mm.metal#L70](https://github.com/ggml-org/llama.cpp/blob/5676dfbe01e7ebe0387913680eee0ed39bf4a363/ggml/src/ggml-metal/kernels/mul_mm.metal#L70)、L106、L314）。
  このリポジトリの段階2・段階3と同じ道具です。

**そのほか**：

- Flash Attention（`ggml_flash_attn_ext`、[src/llama-graph.cpp#L3122](https://github.com/ggml-org/llama.cpp/blob/5676dfbe01e7ebe0387913680eee0ed39bf4a363/src/llama-graph.cpp#L3122)）。
- KVキャッシュの量子化（`-ctk/-ctv`、Q8_0・Q4_0など、[common/arg.cpp#L281-L292](https://github.com/ggml-org/llama.cpp/blob/5676dfbe01e7ebe0387913680eee0ed39bf4a363/common/arg.cpp#L281-L292)）。
- グラフを一度組んでバックエンドに割り振る（`ggml_backend_sched`）、CUDA graphs（[ggml/src/ggml-cuda/ggml-cuda.cu#L3200](https://github.com/ggml-org/llama.cpp/blob/5676dfbe01e7ebe0387913680eee0ed39bf4a363/ggml/src/ggml-cuda/ggml-cuda.cu#L3200)）。
- 演算の融合：RMSNorm＋乗算（＋加算）、RMSNorm＋乗算＋RoPE、conv＋SiLU、gate/upとGLUなど
  （[ggml/src/ggml-cuda/ggml-cuda.cu#L5112-L5137](https://github.com/ggml-org/llama.cpp/blob/5676dfbe01e7ebe0387913680eee0ed39bf4a363/ggml/src/ggml-cuda/ggml-cuda.cu#L5112-L5137)、[ggml/src/ggml-metal/ggml-metal-fusion.cpp#L817-L835](https://github.com/ggml-org/llama.cpp/blob/5676dfbe01e7ebe0387913680eee0ed39bf4a363/ggml/src/ggml-metal/ggml-metal-fusion.cpp#L817-L835)）。
  このリポジトリのADR 0019・0021の融合と同じ種類です。

**MTPによる投機的decode**：Qwen3.5とQwen3-NextのMTP層を読み込み、専用のグラフで下書きします
（[src/models/qwen35.cpp#L104-L139](https://github.com/ggml-org/llama.cpp/blob/5676dfbe01e7ebe0387913680eee0ed39bf4a363/src/models/qwen35.cpp#L104-L139)、L510-L633、`--spec-type draft-mtp`、[PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673)）。
PR #22673では、Qwen3.6で3トークンの下書きの採用率が約75%、decodeが2倍以上と報告されています（prompt処理は遅くなる）。
ほかに別モデル・EAGLE3・n-gramによる下書きもあります（[common/common.h#L177-L187](https://github.com/ggml-org/llama.cpp/blob/5676dfbe01e7ebe0387913680eee0ed39bf4a363/common/common.h#L177-L187)）。

## MLX（mlx-lm）

M5で比較に使った版です。decodeはこちらの約105%、prefillは約3倍の速さでした（[比較](measurements/mlx-comparison-m5.json)）。

**DeltaNet**：prefillとdecodeとも、手書きのMetalカーネル（`mx.fast.metal_kernel`）で、トークンを1つずつ順に処理します
（[mlx_lm/models/gated_delta.py#L96-L128](https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/models/gated_delta.py#L96-L128)、L309-L360、L639-L647）。
既定の「packed」版は、1つのSIMD groupに状態の8行を置き、1行を4レーンで分担します（同L247-L266）。
状態はFP32、q・k・v・出力はBF16です（同L482-L493、L637）。
**このリポジトリのDeltaNetと同じ方式**（FP32の状態、1トークンずつ、1列を4レーン、[ADR 0016](adr-0016.md)）です。
なお、MLX本体にはチャンク形式でテンソル演算を使う `mx.fast.gated_delta_update` がありますが、mlx-lm 0.32.2のqwen3_5はまだ使っていません
（[mlx/backend/metal/kernels/gated_delta_update_nax.h](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/mlx/backend/metal/kernels/gated_delta_update_nax.h)）。

**行列積**：decodeは行列×ベクトル積（gemv）、prefillはsteel GEMMです。
M5の行列演算器（NAX）を使うGEMMは、FP32では（TF32を有効にしない限り）使われず、BF16では使われます
（[mlx/backend/metal/matmul.cpp](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/mlx/backend/metal/matmul.cpp)）。mlx-lmのモデルは活性値もBF16なので、prefillの行列積はNAXのBF16 GEMMで動きます。

**Attention**：`mx.fast.scaled_dot_product_attention`（[mlx_lm/models/base.py#L132](https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/models/base.py#L132)）。
decodeは2パスのベクトル版、prefillはNAXのflash attentionのカーネルがライブラリに入っています（prefillでの選択はコードを追って確かめていない推定）。

**グラフ**：遅延評価で、prefillは2048トークンごとに全体を1つのグラフにしてから評価します。
decodeは次のstepを `mx.async_eval` で先に積みます（[mlx_lm/generate.py#L323](https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/generate.py#L323)、L437-L472）。
モデル全体の `mx.compile` はしておらず、SwiGLUなど小さな要素ごとの演算だけを融合しています。

**MTP・投機的decode**：MTPの重みは読み込み時に捨てています（[mlx_lm/models/qwen3_5.py#L354](https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/models/qwen3_5.py#L354)）。
投機的decodeは全キャッシュの巻き戻しが必要で（[mlx_lm/generate.py#L542-L545](https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/generate.py#L542-L545)）、
DeltaNetのキャッシュは巻き戻せない（`is_trimmable()` がFalse、[mlx_lm/models/cache.py#L164-L165](https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/models/cache.py#L164-L165)）ので、
このモデルでは使えません。DeltaNetの状態を巻き戻せないという、[今後の課題](roadmap.md)の3.と同じ壁です。

**量子化**：affine・mxfp4・nvfp4・mxfp8、混合ビット、AWQ・GPTQなど（[mlx_lm/convert.py#L200-L224](https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/convert.py#L200-L224)）。

**prefillが約3倍速い理由（推定）**：上のとおり、DeltaNetは同じ方式なので、差の大部分は行列積です。
MLXは活性値がBF16なのでNAXのBF16 GEMMを使えますが、こちらは活性値をFP32のまま渡すので、
入力を丸めない遅い経路になります（[ADR 0016](adr-0016.md)：同じ行列で約2.6 TFLOPS、丸めると6.2〜6.4 TFLOPS）。
この差は「精度を保つ」という方針の代償で、`--matmul-relaxed` を付けると差が約3倍から1.5〜1.9倍に縮むことと合っています。

## Strata

1つのMoEモデル（Qwen3.8-Flash-Next：48層×512エキスパート、本体125B、トークンごとに約6B）を、
12GBのGPUで動かすための専用エンジンです。MITライセンスです。
エンジンはC++と自作のCUDA/HIP/SYCL/Metalカーネルで、CPU側のエキスパート計算と一部の量子化カーネルにはllama.cppのggmlを固定版で使っています
（[third_party/strata-ggml/UPSTREAM.md](https://github.com/Niko1221/Strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/third_party/strata-ggml/UPSTREAM.md)）。モデルのグラフはllama.cppのものではありません。

**メモリの階層**（[docs/HOW_IT_WORKS.md](https://github.com/Niko1221/Strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/docs/HOW_IT_WORKS.md)）：

- よく使われるエキスパートをGPUに常駐させ、使われ方に合わせて入れ替えます（`src/core/expert_cache.cpp`）。
- 全エキスパートはRAMに置き、GPUにないものはCPU（AVX-512/AVX2）がその場で計算して、GPUの処理と重ねます。
- 28.8GBのN-gram埋め込み表はSSDに置き、既定ではOSのキャッシュを通さずに直接読みます（mmapは選択肢の1つ、[docs/DETAILS.md](https://github.com/Niko1221/Strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/docs/DETAILS.md)）。

これらは大きなMoEを小さなVRAMに収めるための工夫で、0.8Bのdenseモデルを丸ごとGPUに置くこのリポジトリには当てはまりません。

**DeltaNet**（[docs/GDN_CONTRACT.md](https://github.com/Niko1221/Strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/docs/GDN_CONTRACT.md)）：式はこのリポジトリと同じです（4タップのconv＋SiLU、Q/KのL2正規化、
softplusのゲートとsigmoidのβ、状態更新、`z` によるゲート付きRMSNorm）。状態はFP32です。
decodeは状態の1列を1スレッドで計算します（`src/kernels/cuda/gdn.cu`）。prefillは厳密なカーネルが既定で、
FP32のチャンク形式とFP16/FP32のテンソルコア版はopt-inです。どれもFP64の参照と照合しています。
**既定は厳密、速くて値の変わるものはopt-in**という分け方は、このリポジトリの方針と同じです。

**MTPによる投機的decode**（`src/core/mtp.cpp`、`src/core/verify.cpp`、`include/strata/core/mtp.hpp`）：

- モデル自身のMTP層で最大3トークンを下書きし、本体48層の1回の検証でまとめて確かめます（既定 `--spec 4 --spec-min-p 0.5`）。
- 1回の検証で2.4〜3.2トークン確定し、1.6〜1.8倍と報告されています。オフラインのgreedyの採用率は1〜3トークン目で0.89・0.86・0.85です（`tools/mtp_probe.py`）。
- 出力するトークンはすべて本体の検証で決まるので、下書き側の近似は下書きの質にしか影響しない、と説明しています。
  ただし、samplingでの棄却サンプリングはopt-inで、IQ系の量子化パックでは1トークンと複数トークンの計算で丸めが変わり、
  温度0でも下書きの有無で答えが変わることがある、とも書かれています（環境変数で揃えられる）。
  このリポジトリで投機的decodeを入れるときに確かめるべき点と同じです。

**速度**（[docs/DETAILS.md](https://github.com/Niko1221/Strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/docs/DETAILS.md)「Speed (measured)」、作者の測定）：RTX 5070 12GB・Ryzen 5 7600・DDR5 64GB・Windows、MTPあり、
256トークン生成で、decodeはQ2_0で87〜93 token/s（1K〜4Kトークン）、IQ3_Sで52〜53 token/s、prefillは約430〜2,200 token/sです。
文章の種類と採用率で変わるとされています。

## このリポジトリへの示唆

- **prefillの行列積**：MLXとの差は、活性値をBF16で持つかどうかでほぼ説明できます。
  llama.cppのCUDAのチャンク形式DeltaNetが使う「BF16の上位・下位に分けて行列演算器に入れる」方法は、
  [今後の課題](roadmap.md)の1.で挙げた「活性値を2つのBF16に分けて掛ける」案と同じ考え方で、実際に使われている例です。
- **DeltaNetのチャンク形式**：llama.cpp（CUDA、一部のHIP）、Strata（opt-in）、MLX本体（mlx-lmはまだ不使用）が持っています。
  ただしMetalではllama.cppもmlx-lmも1トークンずつで、こちらと同じです。
- **MTPによる投機的decode**：llama.cppとStrataは実装済みで、採用率75〜89%、decode 1.6〜2倍以上の報告があります。
  MLXはDeltaNetの状態を巻き戻せないので使えていません。llama.cppは状態の控えを複数持つことで、この壁を越えています。
  このリポジトリで次に試す価値が最も大きいのはこれだと考えています（[今後の課題](roadmap.md)の3.）。
  CPU版で測ったこのモデル（0.8B）の採用率は、下書き1〜3で68%・43%・27%（そこまで全部当たる率）で、上の大きなモデルの報告より低めです。
  また小さなモデルでは、下書きの語彙射影（509MB）が本体1トークン分の約1/3を占めるのが特有の問題です（[ADR 0022](adr-0022.md)）。
- **量子化**：3つとも速さの大きな柱です。このリポジトリでは「値が変わる」ので、入れるなら量子化の影響をトレースで観察する教材として、オプションで入れます。
