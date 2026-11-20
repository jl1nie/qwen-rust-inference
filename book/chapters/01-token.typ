#import "../style.typ": exercise, source

= 「東京」が出るまで：二つのループ

質問は「日本の首都はどこですか？一言で。」。Qwen3.5-0.8Bの公開重みをRustの推論器で動かすと、「東京です。」と返った。この章では、最初の「東京」が出るまでの道のりを、細部に入る前に全体として見る。

== 二つのループで次のトークンを選ぶ

推論器は文字列を直接は扱わない。文字列を整数の列に変え、整数一つ分ずつ計算を進め、次の整数を選んで、最後に文字列へ戻す。この整数を *トークンID*、IDが指す文字列の断片を *トークン* と呼ぶ。

計算は二重のループになっている。外側のループは、トークンを一つずつ選んでは入力に戻す。内側のループは、外側のループの1回の中で、一つのトークンを24個の *層* に順に通す。

#figure(
  block(width: 100%, inset: 8pt, stroke: 0.6pt + luma(140), radius: 3pt, {
    set text(size: 8pt)
    set par(first-line-indent: 0em, justify: false)
    let node(body, fill: none) = box(stroke: 0.5pt, inset: 4pt, radius: 2pt, fill: fill, body)
    [*外側のループ*：1回で1トークン進む]
    v(6pt)
    align(center, grid(
      columns: 7, column-gutter: 3pt, align: horizon,
      node[ID], [→],
      node(fill: luma(235))[*内側のループ*\ 埋め込み → 層0 → … → 層23\ → 最終正規化 → 語彙射影],
      [→], node[logits], [→], node[次のID],
    ))
    v(4pt)
    align(center)[↑ 選んだIDを、次の回の入力に戻す]
  }),
  caption: [推論の二重ループ。内側のループは毎回同じ24層を通る。],
)

#table(
  columns: (1.15fr, 2.3fr),
  table.header([処理], [渡すもの]),
  [会話テンプレート], [質問に発話者や区切りを添えた文字列],
  [トークナイザ], [文字列の断片を表すトークンIDの列],
  [内側のループ], [1個のIDから、次のIDの候補すべてに付けたスコア],
  [外側のループ], [スコアから選んだID。次の回の入力になる],
  [トークナイザの復号], [選ばれたIDの列から復元した応答の文字列],
)

層の計算に使う数の表を *重み* と呼ぶ。Qwen3.5-0.8Bの重みは約8億個の数で、学習によって決まっている。本書は学習をしない。公開された重みを読み込み、その数を使って計算する手順を追う。

== トークナイザは文字列をIDの列にする

まず質問だけを、Qwen3.5-0.8Bに付属する固定版のトークナイザに渡す。区切りが入っていない日本語の文でも、次の8個のIDになる。表の断片は読みやすいように文字へ復号したものだ。

#table(
  columns: (1fr, 1fr, 1fr, 1fr),
  align: center,
  table.header([日本の], [首都], [はどこ], [ですか]),
  [161607], [110161], [236576], [173399],
  [？], [一言], [で], [。],
  [10992], [118333], [15685], [1710],
)

一文字ずつ数える方法なら「首」と「都」は別々だが、この語彙では二文字の「首都」が一つのトークンになる。「はどこ」のように、文法上の語の境界をまたぐ断片もある。

トークナイザの手順は三つある。まず *前処理* が、正規表現で文字列を区間に分ける。文字の種類で区切るので、かなと漢字が続く「日本の首都はどこですか」は一つの区間になり、残りは「？一言で」と「。」の二区間だ。次に、各区間をUTF-8のバイト列にし、*BPE*（Byte Pair Encoding）で隣り合う断片を結合していく。最後に、結合し終えた断片を語彙表のIDに置き換える。表の8個のIDは、3区間それぞれでこの手順を進めた結果だ。

語彙表と結合規則は、学習済みの重みと一緒に配布される `tokenizer.json` に入っている。本番の推論器（`TextTokenizer`）は、このファイルを読むtokenizersクレートに処理を任せる。本書では、同じ手順を数十行で書いた教材用の `TeachingBpe` を読む。二つが同じIDを出すことはテストで確かめてある。

#exercise[「首都」のIDと「はどこ」のIDの差から、二つの意味の近さを計算できるだろうか。できない。IDは語彙表の項目番号にすぎない。モデルが計算に使う値は、この後で見るように、IDで引いた埋め込みの行にある。]

#source("src/bpe.rs")[教材用BPE] ／ #source("src/text.rs")[本番の符号化と復号] ／ #source("tests/bpe_real.rs")[二つの一致を確かめるテスト]

#include "bpe-walkthrough.typ"

== 会話テンプレートは入力の文字列を作る

チャットでは質問そのものに加え、誰の発話か、どこから応答を始めるかも入力に含める。`user_chat` 関数が質問を次の文字列に包み、その全体をトークナイザに渡す。`<|im_start|>` などは文字列の区切りを表す *特殊トークン* で、それぞれが一つのIDになる。`<think>` と `</think>` の間は、モデルが答える前に考えを書く欄だ。この設定ではその欄を空にして、すぐ答えさせている。

```text
<|im_start|>user
日本の首都はどこですか？一言で。<|im_end|>
<|im_start|>assistant
<think>

</think>

```

この実行で、本体が応答を選び始める前の入力は20トークンになる。代表的な位置を抜き出した。*位置* はその列内の順番、*ID* は語彙表の項目番号だ。

#table(
  columns: (1fr, 1.6fr, 1.2fr),
  table.header([位置（0始まり）], [断片], [ID]),
  [0・13], [`<|im_start|>`], [248045],
  [3], [日本の], [161607],
  [4], [首都], [110161],
  [11], [`<|im_end|>`], [248046],
  [19], [改行2つ], [271],
)

位置0と13のように、同じIDは列の別の場所にも現れる。

#source("src/text.rs")[テンプレートを組む `user_chat`]

== 外側のループ：選んだIDを入力に戻す

20個のIDを、位置0から順に一つずつモデルに通す。1個通すたびに、モデルは語彙の248,320個のIDそれぞれにスコアを付ける。このスコアを *logits* と呼ぶ。最後の位置19を通した後のlogitsで最もスコアの高いIDが、最初の応答のID 115197（「東京」）になる。最大のスコアを選ぶこの方法を *greedy* と呼ぶ。

```rust
// select（src/generate.rs）のgreedy部分。同点なら小さいIDを残す。
let mut best = 0;
for i in 1..logits.len() {
    if logits[i] > logits[best] {
        best = i;
    }
}
```

選んだIDは、次の回の入力として位置20に置く。位置20を通した後のlogitsから「です」を選び、それを位置21に置く。このように、出力を次の入力に戻して1トークンずつ進めることを *自己回帰* と呼ぶ。応答のID列 `[115197, 36298, 1710]` を復号すると「東京です。」になる。終わりを表すID（`<|im_end|>` の248046など）を選ぶか、生成数の上限に達したらループを止める。

外側のループには二つの段階がある。プロンプトの20トークンを順に入れる段階を *prefill*、応答として選んだトークンを1個ずつ入れる段階を *decode* と呼ぶ。CPU版ではどちらも1トークンずつ同じ計算を通す。違うのは、prefillでは次に入れるIDが最初から決まっているのに対し、decodeでは直前のlogitsから選ぶまで決まらないことだ。下巻でGPUを使うとき、この違いが速くする方法を分ける。

```rust
// InferenceSession（src/session.rs）の抜粋。
// prefill：プロンプトのIDを順に1ステップずつ通す
for &id in ids {
    self.ingest(model, id, false, observer)?;
}
// decode：直前のlogitsから選び、次の回にそのIDを通す
for _ in 0..max_new {
    self.flush(model, observer)?; // 前の回に選んだIDを1ステップ通す
    let token = select(self.logits.as_ref().unwrap(), self.sampling, &mut self.rng)?;
    self.pending = Some(token);
    if eos.contains(&token) {
        break;
    }
}
```

#source("src/session.rs")[prefillとdecodeのループ] ／ #source("src/generate.rs")[次のIDの選択 `select`]

== 内側のループ：1トークンが24層を通る

外側のループの1回、つまり1ステップの中身が内側のループだ。`Model::step_hidden` から、計算の部分だけを抜き出す。

```rust
let mut x = self.embedding.row(id as usize)?.to_vec();
for (i, (b, s)) in self.blocks.iter().zip(&mut state.layers).enumerate() {
    x = b.step_captured(&x, s, state.position, backend, i, edit, false)?.output;
}
let norm = rms_norm(&x, &self.norm, self.config.rms_norm_eps,
                    Scale::QwenZeroCentered)?;
let logits = self.embedding.project(&norm, backend)?;
state.position += 1;
```

1行目は、IDを行番号にして *埋め込み表* から1行を取り出す。埋め込み表は248,320行×1,024列の重みで、各行が一つのトークンを表す1,024個の数の並び、つまり *ベクトル* だ。「首都」なら110161行目を取る。ここで初めて、文字列の断片が計算に使える数になる。

続く `for` が24層の繰り返しだ。各層 `b` は1,024個の数を受け取り、同じ長さの数を返す。`s` はその層が持つ *状態* で、後の節で扱う。24層を出たベクトルは、*最終正規化* `rms_norm` で大きさをそろえる。最後の `project` は、そのベクトルと埋め込み表の各行との近さを測り、248,320個のスコアにする。Qwenは、入力を数にするときの埋め込み表を、出口でスコアを付けるときにも使う。この計算は第2章で詳しく見る。

24層の中身はすべて同じ形をしている。各層は、ベクトルを少し書き換えてから次の層へ渡す。書き換えは二回ある。一回目は *mixer* が、他の位置の情報を読んで書き足す。二回目は *FFN* が、その位置のベクトルだけから書き足す。どちらも元のベクトルを捨てずに足し込むので、この加算の経路を *残差* と呼ぶ。式とRustは第3章で追う。

mixerには2種類ある。過去の位置を一つずつ読み直す *Attention* と、過去を固定サイズの行列にまとめて持つ *DeltaNet* だ。Qwen3.5-0.8BはDeltaNetを3層続けた後にAttentionを1層置き、この4層の組を6回繰り返す。

#figure(
  table(
    columns: (auto, 1fr, 1fr, 1fr, 1fr),
    align: center,
    inset: 4pt,
    table.header([組], [層 4k], [層 4k+1], [層 4k+2], [層 4k+3]),
    ..range(6).map(k => (
      [#k],
      [DeltaNet], [DeltaNet], [DeltaNet],
      table.cell(fill: luma(225))[Attention],
    )).flatten(),
  ),
  caption: [24層の並び。Attentionは層3, 7, 11, 15, 19, 23の6層。どの層にもFFNがある。],
)

本書で使う寸法は、この表に集める。値はモデルに付属する設定ファイル `config.json` のものだ。

#table(
  columns: (1.6fr, 1fr),
  table.header([項目], [値]),
  [層の数], [24（DeltaNet 18、Attention 6）],
  [ベクトルの幅], [1,024],
  [語彙の数（埋め込み表の行数）], [248,320],
  [FFNの中間の幅], [3,584],
  [Attentionのヘッド], [Q 8個、K・V 2個、各256次元],
  [DeltaNetのヘッド], [16個、各128次元],
  [DeltaNetの畳み込みの幅], [4],
)

#source("src/model.rs")[内側のループ `Model::step_hidden`] ／ #source("spec/upstream/config.json")[モデルの設定] ／ #source("spec/tensors.md")[重みの形]

== 二つのループが交わる所：過去を持ち越す状態

内側のループは、一つの位置のベクトルだけを24層に通す。しかし「東京」の後に「です」を選ぶには、質問に「首都」があったことを知らなければならない。位置20の計算をしている時点で、位置4の「首都」は外側のループで16回も前に通り過ぎている。

実際の計算でも、後の位置は前の位置を読んでいる。最初のAttention層（層3）を、位置20を計算している時点で観測した。Attentionは、読む位置ごとに割合（合計すると1）を付けて、その割合で前の位置の情報を混ぜる。8個ある *ヘッド* （並行して読む読み手）のうち3個から、位置4「首都」に置いた割合を抜き出した。

#figure(
  table(
    columns: (1.6fr, 1fr, 1fr, 1fr),
    align: center,
    inset: 5pt,
    table.header([読む位置 ＼ ヘッド], [0], [1], [3]),
    [4 首都 → 4 首都], [0.637], [0.390], [0.538],
    [19 改行 → 4 首都], [0.012], [0.001], [0.002],
    [20 東京 → 4 首都],
    table.cell(fill: luma(225))[*0.598*],
    table.cell(fill: luma(225))[0.460],
    table.cell(fill: luma(225))[0.566],
  ),
  caption: [層3で「首都」に置いた割合（実測値の抜粋）。各位置が読めるのは、自分と自分より前の位置だけだ。],
)

位置20で「東京」を入力として処理するとき、3個のヘッドがいずれも半分前後の割合を位置4「首都」に置いている。位置20のlogitsが選ぶのは「です」だ。一方、「東京」を選んだ位置19は、この層ではほとんど「首都」を読んでいない。位置19が「首都」の情報をどこから得たかは、この層だけでは分からない。前の3層のDeltaNetや、後のAttention層も関わるからだ。

どちらにしても、後の位置の計算には前の位置の何かが要る。そこで各層は、外側のループの回をまたいで値を残す。これが *状態* で、Qwenには二つの持ち方がある。

#table(
  columns: (1fr, 1.6fr, 1.4fr),
  table.header([層], [残すもの], [大きさ]),
  [Attention（6層）], [全位置のK・Vという2種類のベクトル（*KVキャッシュ*）], [1位置・1層あたり4 KiB。位置が増えるほど伸びる],
  [DeltaNet（18層）], [過去をまとめた行列と、直近3位置の値], [1層あたり約1 MiB。位置が増えても変わらない],
)

Attention層は、位置20でも位置4のK・Vをそのまま読み直せる。代わりに、文脈が長くなるほど残す量が増える。DeltaNet層は、新しい位置の情報を行列に書き込み、古い情報を少しずつ薄める。残す量は一定だが、個々の位置を読み直すことはできない。KとVが何か、行列をどう書き換えるかは第3章で扱う。

```rust
// src/state.rs
pub struct KvCache { pub keys: Vec<Vec<f32>>, pub values: Vec<Vec<f32>> }
pub struct DeltaState { pub convolution: Vec<f32>, pub recurrent: Vec<f32> }
pub enum LayerState { Attention(KvCache), Delta(DeltaState) }
pub struct ModelState { pub position: usize, pub layers: Vec<LayerState> }
```

`ModelState` は、次に入れる位置の番号と、24層それぞれの状態を持つ。内側のループで `zip(&mut state.layers)` としていたのは、層ごとに自分の状態を書き換えながら進むためだ。新しい会話を始めるときは、この状態を空に戻す。

#source("src/state.rs")[状態の型] ／ #source("spec/images/attention-layer3.svg")[層3・8ヘッドのヒートマップ]

== コードの地図

ここまでに出た処理と、それを担うRustの関数を並べる。第2章と第3章では、この表の下半分を一つずつ開いていく。

#table(
  columns: (1.25fr, 1fr, 1.45fr),
  table.header([処理], [ファイル], [主な関数・型]),
  [前処理とBPE（教材用）], [`src/bpe.rs`], [`TeachingBpe::encode`],
  [符号化と復号（本番）], [`src/text.rs`], [`TextTokenizer`],
  [会話テンプレート], [`src/text.rs`], [`user_chat`],
  [外側のループ], [`src/session.rs`], [`InferenceSession`],
  [次のIDの選択], [`src/generate.rs`], [`select`],
  [内側のループ], [`src/model.rs`], [`Model::step_hidden`],
  [1層（残差・mixer・FFN）], [`src/block.rs`], [`BlockWeights::step`],
  [Attention], [`src/attention.rs`], [`AttentionWeights::step`],
  [DeltaNet], [`src/deltanet.rs`], [`DeltaWeights::step`],
  [FFN], [`src/ffn.rs`], [`FfnWeights::forward`],
  [正規化・softmaxなど], [`src/ops.rs`], [`rms_norm`、`softmax`],
  [状態], [`src/state.rs`], [`ModelState`],
  [重みの読み込み], [`src/weights.rs`], [`load_model`],
)

#exercise[会話入力の位置13にある `<|im_start|>` について、次の四つを区別しよう。(1) トークンID、(2) 位置、(3) 内側のループで最初に取り出す値、(4) 層3がこの位置の計算の後に残すもの。答えは、(1) 248045、(2) 13、(3) 埋め込み表の248045行目（位置0と同じ1,024個の数）、(4) 位置13のK・V。]
