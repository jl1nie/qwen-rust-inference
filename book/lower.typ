#import "style.typ": book-title, manuscript

#show: manuscript

#v(24mm)
#text(size: 23pt, weight: "bold")[#book-title　下巻]
#v(7pt)
#text(size: 13pt)[QwenをGPUで速くする]

#v(12mm)
上巻でCPUに追った二つのループを、GPUで速くする。まず、計算が合っていることの確かめ方と、速さの上限の見積もり方をCPUで決める。次にApple SiliconのMetalで、上巻のFFN射影やAttentionの状態を、GPUがどう読み書きするかを追う。続いてROCmで同じ推論の実測を読み、CUDAでは同じ射影を組んでCPUで照合して単体で測り、ROCmのカーネルをそのまま使って推論器全体も動かす。最後に、重みを8ビット・4ビットに減らすと何がどれだけ変わるかを測る。Metal・ROCmの結果と照らし合わせるときは、射影の形と精度に加え、発行方法や計時範囲も確認する。

#pagebreak()
#outline(title: [目次], depth: 1)

上巻では、第1章で全体像、第2章で文字列とID、第3章でprefillとdecode、第4章で数の扱い、第5章で1 layerの計算、第6章でパラメータへの介入を扱う。下巻ではGPUの用語を必要なところで導入する。Apple Siliconの実機があれば測定を再現でき、保存済みのログからも結果を追える。章番号は上巻から続く。

#pagebreak()
#counter(heading).update(6)
#include "chapters/07-baseline.typ"
#pagebreak()
#include "chapters/08-prefill.typ"
#pagebreak()
#include "chapters/09-decode.typ"
#pagebreak()
#include "chapters/10-mtp.typ"
#pagebreak()
#include "chapters/11-rocm.typ"
#pagebreak()
#include "chapters/12-cuda.typ"
#pagebreak()
#include "chapters/13-quantization.typ"
#pagebreak()
#include "chapters/appendix.typ"
