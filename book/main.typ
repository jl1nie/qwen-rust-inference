#import "style.typ": book-title, book-subtitle, manuscript

#show: manuscript

#v(27mm)
#text(size: 24pt, weight: "bold")[#book-title]
#v(6pt)
#text(size: 12pt)[#book-subtitle]
#v(16mm)

Qwen3.5-0.8Bに、次の1トークンを選ばせる。そのとき各層はどの数字を読み、何を次の位置へ残すのか。まずCPU上のRustで計算をたどる。同じ計算をApple SiliconのGPUに移し、入力をまとめて読むときと、答えを1トークンずつ書くときで、実装をどう変えるかを見る。

最後に、答えを先読みしてまとめて検証する。先読みが外れたとき、CPU編で見た二種類の状態の違いが、GPU上で必要な操作を決める。

#pagebreak()
#outline(title: [目次], depth: 1)

配列とループが読めれば、第1章から順に読める。前半の小さな計算には、GPUも公開重みも要らない。後半は図と実測値で読み進め、Apple SiliconのMacがあれば手元でも試せる。実行手順と記録の所在は付録にまとめた。

#pagebreak()
#include "chapters/01-token.typ"
#pagebreak()
#include "chapters/02-block.typ"
#pagebreak()
#include "chapters/03-reference.typ"
#pagebreak()
#include "chapters/04-prefill.typ"
#pagebreak()
#include "chapters/05-decode.typ"
#pagebreak()
#include "chapters/06-mtp.typ"
#pagebreak()
#include "chapters/appendix.typ"
