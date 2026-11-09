# コードを読んで「次の1トークン」を説明する

この推論器は文章生成の速さより、計算を追えることを優先しています。
まず `cargo run --offline -- demo ffn` を動かし、表示された小さい配列を
`src/ffn.rs` の式と対応させてください。以降は `attention.rs`、
`deltanet.rs`、`block.rs`、`model.rs`、`generate.rs` の順に読むと、
単一演算から自己回帰生成へつながります。数式の直前には学習用コメントがあります。

## 1. 行列の「行」と「列」は何を表すか

`Matrix` は常にrow-majorの `[out,in]`。`linear(x)` の出力iは
`sum_j W[i,j]*x[j]` です。行iは「出力iを作る読み方」、列jは
「入力成分jが全出力にどこへ書くか」を表します。
`Backend::Reference` は添字順のループ、`Optimized` はsgemm、`Vector` は単一ベクトル向けの8累積器です。
両者は `tests/components.rs` とモデル全体の同値テストで比較します。

## 2. FFNで一つのユニットを消す

`gate = W_gate x`、`up = W_up x`、`product = SiLU(gate) * up`、
`output = W_down product`。FFNはこの位置の特徴を変換し、他位置の情報を
直接読みません。SiLUは `x*sigmoid(x)` なので負の入力にも小さい反応があります。

手計算fixtureでは x=[1,2]、gate=[0,1]、up=[1,2]。
s=SiLU(1)=0.7310586 とおくと product=[0,2s]。
down=[[1,3],[2,4]] なので出力は [6s,8s] です。
ユニット1の書込み列は [3,4]。このユニットを消すと、その反応2sと列の積
[6s,8s] だけが消えます。`hand_ffn_intervention` テストで独立に照合しています。

実モデルでも `writing_vector(j)` が同じ列を返します。行番号jを消すと
出力ニューロンを変えることになり、中間ユニットの除去とは異なります。

## 3. Attentionは過去のどこを読むか

各位置の隠れベクトルからQ（問い合わせ）、K（対応を測るキー）、V（読む値）を
作ります。今回のQと過去位置ごとのKの内積を `sqrt(head_dim)` で割り、
softmaxで和1の確率にします。その確率でVを足し合わせます。

1次元でscores=[0,ln 3]なら確率=[1/4,3/4]。
V=[2,6]の読出しは5です。`attention_and_rotary_hand_examples` テストで確認できます。
因果性は「現在より未来のK/Vを確率計算へ渡さない」ことで保ちます。
全列版は全QKVを先に作り、各位置で先頭から現在までを読む独立の経路です。

Qwenの注意点は、Q/Kのzero-centered RMSNorm、部分RoPE、GQA、出力ゲートです。
Q=8ヘッド・KV=2ヘッドならQの0..3はKV0、Qの4..7はKV1を共有します。
RoPEの組は隣接要素ではなく回転範囲の前半/後半。
Qと出力ゲートはヘッドごとに交互に格納されています。
これらを順番に `AttentionWeights::project` で確認してください。

KVキャッシュは「過去の位置回転済みKとV」。入力トークン列でも残差ベクトルでも
ありません。位置offsetを変えて過去Kを再回転すると誤ります。
層ごとに別のキャッシュが必要です。`demo attention --cache` はその成長も表示します。

## 4. DeltaNetは固定サイズの記憶へどう書くか

`demo deltanet` では、短い畳み込み履歴と再帰行列Sを別々に表示します。
畳み込みは各チャンネルで直近kernel個を混ぜる処理で、未来を読みません。
Q/Kは平方和によるL2正規化です。RMSNormの平方平均と混同しないでください。

1ヘッドのSの形状は `[key_dim,value_dim]`。

1. `S *= exp(g)` で古い記憶を減衰する。gは0以下。
2. `memory = k^T S` で現在のキーについて予測している値を読む。
3. `delta = beta * (v-memory)` で予測誤差を書く量を決める。
4. `S += k delta^T` でrank-1の修正を加える。
5. `y = q^T S` で更新後の記憶を問い合わせる。

例えばSがゼロ、k=q=[1,0]、v=[2,3,4]、beta=1、g=0なら、
S=[[2,3,4],[0,0,0]]、y=[2,3,4]になります。
次にbeta=0なら新しいvは書かれません。gを大きな負数にすると過去Sはほぼ消えます。
`delta_rule_gate_boundaries_and_hand_calculation` と非正方行列のテストがあります。
低水準delta_ruleへ渡すqはhead_dimによるスケールを適用済みです。

AttentionのKVは文脈長に比例して増えますが、Sは同じ大きさのままです。
そのため「全過去のVを保持」するAttentionと、「過去を行列へ要約」するDeltaNetは
異なる記憶の仕組みです。Rustは公式のchunk式を使わず、stepごとに上の式を実行します。
出力は直接係数RMSNorm→SiLU(z)ゲート→out射影で隠れ幅に戻します。

## 5. 残差とlogitsから生成へ

`block.rs` の残差経路をたどると、
`x1 = x + mixer(norm(x))`、`x2 = x1 + FFN(norm(x1))` です。
元のxと正規化後の値を別変数にしてあり、どこへ加算されるか確認できます。
3 DeltaNet + 1 Attentionを繰り返し、最終正規化後に語彙へ射影します。

入力は埋め込み行のlookup、出力は同じ埋め込み行列との内積です（tied embeddings）。
logitsはまだ確率ではありません。greedyは最大logitのID、samplingは
温度とtop-kを適用した確率から固定seedで選びます。
選んだIDを次の入力として同じrunの状態へ渡すのが自己回帰です。
生成履歴、キャッシュ、重み、今回の活性値は別々の型で保持します。

`</think>`（248069）は推論区切りで、応答全体の終わりではありません。
この固定モデルでは `<|im_end|>`（248046）とconfigのendoftext（248044）を
終了に使い、最大トークン数でも停止します。制御したlogitsで終了処理を試験しており、
ランダム重みが自然に終了トークンを出すことには依存しません。

## 6. 重み編集を出力差分で理解する

down列w_jから方向d成分を除去する式は
`w'_j = w_j - alpha*d*(d^T w_j)/(d^T d)`。
したがってFFN出力へ加わる修正は
`-alpha*d*(d^T w_j)/(d^T d)*product[j]` です。
`edit.rs` は行列を変更せず、この差分だけを計算します。
方向の長さを変えても同じ成分除去になります（ゼロ方向は拒否）。

`examples/remove-coordinate-direction.json` は第0座標方向という算術例です。
意味のある「反復方向」を発見した例ではありません。複数の方向編集は元列への
加算overlay、ユニット無効化はその前にproductへ適用する規約です。
無編集・空overlay・編集後に無編集へ戻した結果を比較してください。

## 7. 参照一致をどう読むか

小型fixtureは公式ソースの本文から生成し、手計算例と相互に検証しています。
実モデルでは重みをBF16からf32に変換し、参照側もf32で計算します。
ネイティブBF16計算と同じトークン列になるという主張ではありません。
絶対誤差・相対誤差・cosineを層/演算ごとに見て、最初の差異から調べます。
生成文字列が自然に見えるだけでは正しさの証拠になりません。

`cargo test --offline --test components -- --nocapture`、`--test hybrid`、
`--test generation_edits`を順に実行し、テストの入力を少し変えて予測してから
結果を見ると、配列規約と状態の意味を確認できます。

## 8. 保存トレースを使う演習

[8問の演習](exercises.md) は、同梱した幅8・4層モデルの完全トレースを読む教材です。
先に手で予測し、jqで必要な位置・層だけを抽出してから、
[解答と理由](exercise-solutions.md) を確認してください。
FFNの除去差分、Attentionの確率、DeltaNetの一つの状態要素、
最後のlogitまで、同じ3位置の計算を続けて追います。

```sh
lesson_dir=$(mktemp -d)
cargo run --offline --locked -- demo lesson --trace "$lesson_dir/trace.ndjson" > "$lesson_dir/answers.json"
cargo test --offline --locked --test lesson
```

長い入力では `Model::prefill_observed` が各stepをobserverに渡して解放し、
最後のlogitsだけを返します。全stepをVecに保持する観測用prefillとの違いも
[長い実モデル検証](extended-validation.md) で確認できます。

## 9. 会話を再開するときの「まだ読んでいない1トークン」

`session.rs` では、生成IDを選んだ直後と、そのIDを次の入力にする時刻を分けます。
最後のIDはpendingとして出力済みですが、state.positionにはまだ含まれません。
従って `tokens.len = state.position + pendingの有無`。
保存時にこの1個とlogits/RNGを落とすと、再開でトークンが重複したり乱数列が戻ったりします。

`chat.rs` は公式テンプレートが過去assistantのreasoningを省くことを前提に、
前回user末尾のanchorを保持します。新しいID列とprefixが一致することを確認してから使い、
assistantの正規化済み内容と次userを再入力します。KVをただ伸ばせばよいとは限りません。

[snapshotの形式とCLI演習](practical.md)で、途中保存→/continue→次turnを試してください。
Full stepは中間値の観察用、compact stepは同じ演算を層ごとに解放する実行用です。
Vectorの8本の和は加算順序が違うので、sgemmや素朴なループとの数値比較を残しています。
