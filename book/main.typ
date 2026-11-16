#import "style.typ": book-title, book-subtitle, manuscript, source

#show: manuscript

#align(left)[
  #text(size: 24pt, weight: "bold")[#book-title]
  #v(4pt)
  #text(size: 12pt)[#book-subtitle]
]

Qwen3.5-0.8Bを題材に、次の1トークンの計算をRustでたどる。
その計算をApple SiliconのGPUに載せ、プロンプトを読むとき、
1トークンずつ生成するとき、数トークンをまとめて検証するときの実行方法を比べる。

#outline(title: [目次], depth: 1)

#pagebreak()
= Qwenが「東京」を出すまで
// 6頁。実モデルの出力 → トークンIDと埋め込み → Rustの推論ループ。
== 出力を観察する
== 次のトークンを選ぶ
== 位置と層を区別する

#pagebreak()
= Transformerのブロックを手で計算する
// 10頁。正規化と残差、FFN、Attention、DeltaNet、24層の構成。
== 一つのブロックで何が起こるか
== FFNは各位置に何を加えるか
// 初稿プレビューでは4–5頁。完成稿では16–17頁を予定。
#pagebreak()
#include "chapters/state-spread.typ"

#pagebreak()
= Rustの参照実装を、GPU実装のものさしにする
// 6頁。手計算 → 合成重み → 公式実装との照合 → Metalブリッジ。
== 重み・状態・活性値をRustの型で読む
== 何を、どの精度で照合するか
== 同じ重みを1位置と複数位置で読む

#pagebreak()
= 重みを使い回す ― Metalのprefill
// 14頁。同じFFN射影を参照カーネル、SIMD group、matmul2dの順で追う。
== プロンプトを処理する時間を測る
== 行列積をタイルに分ける
== SIMD groupからMetal 4のテンソル演算へ
== タイルをまたいでAttentionを計算する
== DeltaNetの更新と精度のトレードオフ

#pagebreak()
= 重みを読み切る ― Metalのdecode
// 12頁。帯域の見積もり、BF16 GEMV、小カーネルと融合、MLX比較。
== 1トークンで読むバイト数を数える
== BF16 GEMVをlaneで分担する
== 発行回数と演算融合を測る
== 文脈長とKVキャッシュの読み出し
== MLXとの比較から次の仮説を立てる

#pagebreak()
= 1回の検証で複数トークンを進める ― MTP
// 12頁。下書き、少数行の検証、状態復元、GPU上の選択と同期。
== 下書きと本体の予測を突き合わせる
== 数トークンをまとめて検証する
// 初稿プレビューでは10–11頁。完成稿では56–57頁を予定。
#pagebreak()
#include "chapters/rollback-spread.typ"

#pagebreak()
== argmaxと次の埋め込み取得をGPU上でつなぐ
== 確定したトークン数と所要時間を測る

#pagebreak()
#heading(numbering: none)[付録]
== ROCmの実装と実測
== CUDAでBF16 GEMVを検証する設計
== コードと測定の再現
#source("README.md")[参照コミットのREADME]
