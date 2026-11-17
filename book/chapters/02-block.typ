#import "../style.typ": exercise, source

= Transformerのブロックを手で計算する

一つの層に入ってくるのは、前の層から渡された同じ位置のベクトル $x$ だ。まず、過去や今回の位置の情報を扱う演算を行い、その結果を元のベクトルへ足す。次に、その位置だけで行うFFNを通し、その出力も足す。この加算経路を *残差* と呼ぶ。

$ x_1 = x + "mixer"("norm"(x)) $
$ x_2 = x_1 + "FFN"("norm"(x_1)) $

#figure(
  table(
    columns: (1fr, 1.6fr, 1.1fr),
    table.header([段階], [変換への入力], [加算先]),
    [位置の情報を読む], [正規化した $x$], [$x$],
    [FFN], [正規化した $x_1$], [$x_1$],
  ),
  caption: [元のベクトルを加算経路に残し、変換の結果を足す。],
)

RMSNormでは、成分の二乗の平均に小さな値を足し、平方根を取る。その値で入力を割り、学習済みの成分別係数を掛ける。係数を1、小さな値を無視できるとすれば、$(3,4)$ は $sqrt((9+16)/2)$ で割る。残差の足し算では、変換前のベクトルを足し算の片側に残す。

`mixer` は、この本でAttentionとGated DeltaNetをまとめて指す呼び方だ。どちらが入る層でも、その後ろにはFFNがある。以下では、まず形の小さなFFNから読む。

#source("src/block.rs")[ブロック内の演算順]

#pagebreak()
== FFNでベクトルを広げる

FFNは、同じ位置の成分を組み替えて、残差へ加える値を作る。Qwenでは1,024成分の入力を二つの射影でそれぞれ3,584成分に広げる。一方を `gate`、もう一方を `up` と呼ぶ。

$ g = W_"gate" x, quad u = W_"up" x $
$ a = "SiLU"(g) ⊙ u, quad y = W_"down" a $

$⊙$ は成分ごとの積を表す。SiLUは入力 $z$ を $z / (1 + exp(-z))$ に変える。最後の射影で $a$ を元の幅へ戻せば、残差に足せる。行列の形は $W_"gate", W_"up"$ がともに $[3584,1024]$、$W_"down"$ が $[1024,3584]$。gateの値が0の成分は、upの値が大きくても要素積では0になる。

小さな例で、二つの射影が $g=(0,1)$、$u=(1,2)$ を返したとしよう。$s = "SiLU"(1)$ とすると、積は $(0,2s)$。次の頁では、この二つの射影を小さな行列とRustの添字で確認する。

#pagebreak()
== 二つの射影をRustの添字にする

入力を $(1,2)$ とする。gateの行列を $mat(2,-1;-1,1)$、upを恒等行列にすれば、前頁の $g=(0,1)$ と $u=(1,2)$ が実際に得られる。gateとupは同じ入力から別々のベクトルを作る。

```rust
let x = [1.0_f32, 2.0];
let gate = [[2.0, -1.0], [-1.0, 1.0]];
let g = gate.map(|r| r[0] * x[0] + r[1] * x[1]);
assert_eq!(g, [0.0, 1.0]);
let u = x; // この例の up は恒等行列。
```

実装の `self.gate.project(x, backend)` と `self.up.project(x, backend)` に当たる部分だ。たとえばgateの1行目は $2 dot 1 - 1 dot 2 = 0$。この行が決めるのは、中間ベクトルの先頭成分。最後の射影でその成分がどの出力へ影響するかは、downの列を見る。

#pagebreak()
== 一つの中間成分を消してみる

最後の重みを $W_"down" = mat(1, 3; 2, 4)$ とすると、出力は次の式になる。

$ y = 0 dot mat(1; 2) + 2s dot mat(3; 4) = mat(6s; 8s) $

添字1の中間成分 $2s$ を0にすると、列 $(3,4)$ の寄与が消える。行列の行は出力成分一つを作り、列は入力成分一つが各出力へ書き込む方向を表す。この見方を持っておくと、FFNの幅を増やす意味と、後でGPUに割り振る行を区別できる。

```rust
// 添字 j の中間成分が出力 i に与える寄与。
let contribution = down[i][j] * activation[j];
```

CPU版では、試したい中間ユニットを指定すると、その寄与を打ち消したときの差を観測できる。小型の例は `hand_ffn_intervention` の手計算と対応する。

#source("src/ffn.rs")[FFNの実装] ／ #source("spec/learning.md")[手計算と演習]

#pagebreak()
== 同じ重みを二つの位置で使う

二つ目の入力を $(2,1)$ に取り替えてみる。gateは $(3,-1)$、upは $(2,1)$。先頭の中間成分は $2 "SiLU"(3)$ になる。重みは同じなので、同じ演算を位置ごとに並べて計算できる。

#table(
 columns: (1.05fr, 1.3fr, 1.3fr),
 table.header([位置], [gateの出力], [upの出力]),
 [位置0], [$(0,1)$], [$(1,2)$],
 [位置1], [$(3,-1)$], [$(2,1)$],
)

実装のFFNは、二つの射影、SiLU、要素積、downの射影の順に進む。各位置は自分の入力から計算できる。この独立性が、下巻で行列の重みを複数位置の計算に使い回す前提になる。

これに対して次のAttentionでは、位置1の入力だけでなく、位置0で作ったKとVも計算に入る。同じ射影を並べられる部分と、他の位置の値を読む部分を見分けよう。

#source("src/ffn.rs")[射影、SiLU、要素積の順序]

#pagebreak()
#include "attention-walkthrough.typ"

#pagebreak()
#include "deltanet-walkthrough.typ"

#pagebreak()
#include "state-spread.typ"

#pagebreak()
== 24層へ戻る

合成重みの小型モデルでは、幅8のベクトルがDeltaNet 3層とAttention 1層を進む。実モデルではこの並びが6組あり、幅は1,024、合計24層になる。どの層にもFFNがある。各層が持つ過去の状態は、その層の重みが計算した値だ。層が変われば、同じ位置でも状態の中身は違う。

```text
層 0–3   DeltaNet  DeltaNet  DeltaNet  Attention
層 4–7   DeltaNet  DeltaNet  DeltaNet  Attention
         …
層20–23   DeltaNet  DeltaNet  DeltaNet  Attention
```

#exercise[幅8のモデルをCPUで1ステップ進め、トレースからFFN出力と状態を一つずつ探そう。FFN出力は同じ位置の残差へ加わる。AttentionのK/VやDeltaNetの再帰行列は次の位置に残る。]

#source("spec/exercises.md")[保存トレースの演習] ／ #source("spec/tensors.md")[24層の形状]
