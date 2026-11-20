#import "style.typ": book-title, manuscript

#show: manuscript

#v(24mm)
#text(size: 23pt, weight: "bold")[#book-title　上巻]
#v(7pt)
#text(size: 13pt)[Qwenの1トークンをCPUで追う]

#v(12mm)
Rustの配列とループから出発し、LLMが次のトークンを選ぶまでを読む。Qwen3.5-0.8Bを題材に、文字列、ID、埋め込み、各層で変わる値、過去から持ち越す状態を一つずつ結びつける。全巻で使う数式と添字は、この巻の小さな例で導入する。

きっかけは、QwenのFFNの重みを直接編集した #link("https://rikiyat.github.io/blog/weight-space-programming/")[Weight-Space Programming] だった。重みの一列を変えると、どの値が変わり、次のトークンの選択にどう届くのか。この巻では、その問いを読むのに必要な計算を一つずつ確かめる。介入そのものは、照合の方法を整えた後、下巻の最終章（第11章）で試す。

#pagebreak()
#outline(title: [目次], depth: 1)

LLMや線形代数を学んだことがなくても、第1章から順に読める。式の添字は、対応するRustの配列と照らし合わせる。CPUの小さな例から始め、実モデルのトレースを見る。

#table(
  columns: (1.45fr, 1.25fr),
  table.header([上巻で確かめること], [下巻で使う場面]),
  [位置とトークンID、次のIDを選ぶ位置], [prefill・decodeの境界、MTPの検証],
  [FFNの射影と、トークン間に残る状態], [行列積での重みの再利用、KVの読出しと巻き戻し],
  [行列とベクトルの積、BF16とFP32の区別], [第4章の照合と速度の見積もり、各GPUの比較],
)

#pagebreak()
#include "chapters/01-token.typ"
#pagebreak()
#include "chapters/02-tools.typ"
#pagebreak()
#include "chapters/03-block.typ"
