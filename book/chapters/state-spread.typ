#import "../style.typ": exercise, source
#import "state-example.typ": state-history

== Attentionは位置ごとにKとVを残す

文章を1トークンずつ読むとき、直前までの計算をどんな形で残せば、次のトークンを処理できるだろう。Qwenには二つの答えがある。ここでは、同じ位置の入力から得られた値を $v = 6$ とし、AttentionとGated DeltaNetで何が残るかを比べる。

=== 過去の値を重み付きで読む

小さなAttentionを考える。位置0の計算が終わり、キー $k_0$ と値 $v_0 = 2$ が保存されている。次に位置1を処理し、キー $k_1$ と値 $v_1 = 6$ を作る。今回のクエリ $q_1$ は、この二つのキーを使って今回読む値の配分を決める。

内積から得たスコアを $(0, ln(3))$ とする。指数を取ると $(1, 3)$。和の $4$ で割ると、位置0と位置1を読む重みが得られる。

$ y = 1/4 dot 2 + 3/4 dot 6 = 5 $

#table(
  columns: (1fr, 1fr),
  table.header([位置1を読む前], [位置1を読んだ後]),
  [$(k_0, 2)$], [$(k_0, 2), (k_1, 6)$],
)

位置2には、位置2の入力から作る別のクエリがある。そのクエリでも、位置0と位置1のK/Vを読み直せるよう、計算が終わった位置のK/Vを残す。これが *KVキャッシュ* である。処理が1位置進むたび、有効な位置が一つ増える。

#source("src/attention.rs")[参照コード　Attention]

#pagebreak()
== DeltaNetは状態行列を書き換える

DeltaNetでは、過去を同じ形の状態行列に蓄える。更新だけを追うため、ここでは畳み込みを省き、減衰係数を1、更新率を $1/2$ とする。キーとクエリは2成分、値は1成分なので、状態 $S$ は2行1列になる。

$ S = mat(2; 0), quad k = q = mat(1; 0), quad v = 6 $

今のキーで状態を読むと $k^T S = 2$。実際の値との差は $6 - 2 = 4$ で、その半分をキーの方向に足す。更新後の状態は $mat(4; 0)$。クエリでこれを読むと出力は $4$ になる。

```rust
let mut state = [2.0, 0.0];
let key = [1.0, 0.0];
let memory = key[0] * state[0] + key[1] * state[1];
let delta = 0.5 * (6.0 - memory);
for (cell, key_component) in state.iter_mut().zip(key) {
    *cell += key_component * delta;
}
assert_eq!(state, [4.0, 0.0]);
```

#state-history()

#exercise[位置1を取り消すとしたら、AttentionとDeltaNetのどの値を戻せばよいだろうか。この表を第7章でも使う。]

#source("src/deltanet.rs")[参照コード　Gated DeltaNet]
