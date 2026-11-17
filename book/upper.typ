#import "style.typ": book-title, manuscript

#show: manuscript

#v(24mm)
#text(size: 23pt, weight: "bold")[#book-title　上巻]
#v(7pt)
#text(size: 13pt)[Qwenの1トークンをCPUで追う]

#v(12mm)
Rustの配列とループから出発し、LLMが次のトークンを選ぶまでを読む。Qwen3.5-0.8Bを題材に、文字列、ID、埋め込み、各層で変わる値、過去から持ち越す状態を、同じ推論の中で結びつける。

#pagebreak()
#outline(title: [目次], depth: 1)

LLMや線形代数を学んだことがなくても、第1章から順に読める。式の添字は、対応するRustの配列と照らし合わせる。CPUの小さな例から始め、実モデルのトレースを見る。GPUで同じ計算をどう実行するかは下巻で扱う。

#pagebreak()
#include "chapters/01-token.typ"
#pagebreak()
#include "chapters/02-block.typ"
#pagebreak()
#include "chapters/03-reference.typ"
