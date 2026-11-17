#import "style.typ": book-title, manuscript

#show: manuscript

#v(24mm)
#text(size: 23pt, weight: "bold")[#book-title　下巻]
#v(7pt)
#text(size: 13pt)[QwenをMetalで速くする]

#v(12mm)
上巻で読んだFFN射影とAttentionの状態を、同じQwenのGPU実装で追う。プロンプトをまとめて読むとき、1トークンずつ生成するとき、下書きを検証するときに、計算とデータの置き方をどう変えるかを見る。

#pagebreak()
#outline(title: [目次], depth: 1)

上巻では、第2章でFFNとモデルの状態、第3章で行列の形と参照実装を扱う。下巻ではGPUの用語を必要なところで導入する。Apple Siliconの実機があれば測定を再現でき、保存済みのログからも結果を追える。章番号は上巻から続く。

#pagebreak()
#counter(heading).update(3)
#include "chapters/04-prefill.typ"
#pagebreak()
#include "chapters/05-decode.typ"
#pagebreak()
#include "chapters/06-mtp.typ"
#pagebreak()
#include "chapters/appendix.typ"
