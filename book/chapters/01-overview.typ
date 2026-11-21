#import "../style.typ": exercise, source, sidebar

= 全体像：Qwenが「東京」を出すまで

質問は「日本の首都はどこですか？一言で。」。Qwen3.5-0.8Bの公開パラメータをRustの推論器で動かすと、「東京です。」と返る。この章では、式を使わずに、この答えが出るまでの仕組みを全部ひととおり見る。細部は後の章で一つずつ開くので、ここでは全体の地図を頭に入れてほしい。速くするための工夫（下巻）は、この章では扱わない。

== LLMは次のトークンを選ぶ

推論器は文字列を直接は扱わない。文字列を整数の列に変え、計算を進め、次の整数を一つ選び、最後に文字列へ戻す。この整数を *トークンID*、IDが指す文字列の断片を *トークン* と呼ぶ。

LLMがすることは、突き詰めると一つだけだ。それまでのIDの列を受け取り、語彙にある248,320個のIDそれぞれに「次に来そうな度合い」のスコアを付ける。このスコアの並びを *logits* と呼ぶ。質問を入れた直後のlogitsから、上位を抜き出した。

#figure(
  table(
    columns: (0.6fr, 1fr, 0.8fr, 0.8fr),
    align: (center, left, right, right),
    table.header([順位], [候補のトークン], [logits], [確率]),
    [1], [東京], [17.24], [0.603],
    [2], [東京都], [15.88], [0.154],
    [3], [`**`（太字の始まり）], [14.56], [0.041],
    [4], [大阪], [14.07], [0.025],
    [5], [日本], [13.90], [0.021],
  ),
  caption: [最初の応答を選ぶときの上位5候補（実測）。確率は全候補で合計すると1になる。],
)

logitsは大きいほど有力という目安で、そのままでは確率ではない。全部の候補について合計が1になるように変換したものが、右の列の確率だ。この変換を *softmax* と呼ぶ（第4章）。「東京」が6割、「東京都」が1割半で、「大阪」にもわずかに確率が残っている。

次のIDの選び方は二通りある。

- *greedy*：最も確率の高い候補を選ぶ。同じ入力からは、いつも同じ答えが出る。本書の実例はすべてgreedyで、ここでは「東京」が選ばれる。
- *sampling*：確率に従ってくじを引く。推論器の `Sampling::Seeded` では、*温度* で確率の差を強めたり弱めたりし（温度を0.8にすると「東京」は0.77になる）、*top-k* で上位k個の候補だけを残してから引く。

手元で動かすには、リポジトリの `tools/download-model.sh` で公開パラメータ（約1.75 GB）を取得し、`infer` を実行する。上巻の小さな例（`demo ffn` など）は、モデルがなくても動く。

```text
cargo run --release -- infer --model-dir models/qwen \
    --prompt "日本の首都はどこですか？一言で。"
```

#source("src/generate.rs")[次のIDを選ぶ `select` と `Sampling`] ／ #source("README.md")[モデルの取得と実行]

== 文字列からIDの列へ

質問は、そのままモデルに入るわけではない。まず *会話テンプレート* で、誰の発話か、どこから応答を始めるかを示す特殊トークンを添える。テンプレートには、モデルが答える前に考えを書く `<think>` の欄もある。本書の実例では、この欄を空にしてすぐ答えさせている。こうしてできた文字列を *トークナイザ* がIDの列に変える。質問の部分は8個、テンプレート全体では20個のIDになる。

トークナイザは、文字列をUTF-8のバイト列にし、よく一緒に現れるバイトの組を学習済みの規則に従って順に結合していく（BPE）。「首都」は6バイトから一つのトークンになる。生成したIDの列は、逆に語彙表を引いて文字列に戻す。この操作を *detokenize* と呼ぶ。IDの列 `[115197, 36298, 1710]` は「東京です。」に戻る。手順の詳細とRustの実装は第2章で追う。

== 二つのループ

計算は二重のループになっている。

- *外側のループ*（*自己回帰生成*）：次のIDを一つ選んでは、それを入力の最後に足す。選んだ「東京」を足して次を選ぶと「です」、それを足すと「。」になる。出力を次の入力に戻して1トークンずつ進めるので、自己回帰と呼ぶ。
- *内側のループ*（*forward pass*）：外側のループの1回の中で、一つのIDを24個の *layer* に下から順に通し、最後にlogitsを出す。

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
      node(fill: luma(235))[*内側のループ（forward pass）*\ 埋め込み → layer 0 → … → layer 23\ → 正規化 → LM head],
      [→], node[logits], [→], node[次のID],
    ))
    v(4pt)
    align(center)[↑ 選んだIDを、入力の最後に足す]
  }),
  caption: [推論の二重ループ。内側のループは毎回同じ24個のlayerを通る。],
)

外側のループは、次のどれかで止まる。終わりを表す *停止ID*（`<|im_end|>` の248046か、`<|endoftext|>` の248044）を選んだとき、生成数の上限に達したとき、扱えるIDの数の上限（後述の262,144）に達したときだ。「東京です。」の後には停止IDが選ばれて止まる。

コードでは、外側のループが `InferenceSession::generate_inner`、内側のループが `Model::step_hidden` にある。内側のループの計算部分は、ほぼ次の5行だ。

```rust
// Model::step_hidden（src/model.rs）から計算の部分だけを抜き出した。
let mut x = self.embedding.row(id as usize)?.to_vec();      // 埋め込み
for (i, (b, s)) in self.blocks.iter().zip(&mut state.layers).enumerate() {
    x = b.step_captured(&x, s, state.position, backend, i, edit, false)?.output;
}                                                            // 24個のlayer
let norm = rms_norm(&x, &self.norm, self.config.rms_norm_eps, Scale::QwenZeroCentered)?;
let logits = self.embedding.project(&norm, backend)?;       // LM head
```

#source("src/session.rs")[外側のループ] ／ #source("src/model.rs")[内側のループ `Model::step_hidden`]

== layer × positionの図

二つのループを一枚にすると、縦がlayer、横が *position*（IDの列の中の順番）の図になる。

#figure(
  {
    set text(size: 6.8pt)
    set par(first-line-indent: 0em, justify: false)
    let cell(body, fill: none) = box(width: 100%, inset: (x: 1pt, y: 3pt), stroke: 0.4pt + luma(130), fill: fill, align(center, body))
    let lab(body) = align(right + horizon, body)
    let used = luma(200)
    let resp = luma(238)
    let cols = 7
    let row(label, items) = (lab(label),) + items
    grid(
      columns: (auto, 1.5fr) + (1fr,) * (cols - 1),
      gutter: 2pt,
      [], ..([0], [1], [2], […], [19], [20], [21]).map(c => align(center, c)),
      ..row([logits（LM head）], (cell[○], cell[○], cell[○], cell[…], cell(fill: used)[◎], cell(fill: used)[◎], cell(fill: used)[◎])),
      ..row([layer 23], (cell[□], cell[□], cell[□], cell[…], cell[□], cell(fill: resp)[□], cell(fill: resp)[□])),
      ..row([⋮], (cell[⋮], cell[⋮], cell[⋮], cell[⋮], cell[⋮], cell(fill: resp)[⋮], cell(fill: resp)[⋮])),
      ..row([layer 0], (cell[□], cell[□], cell[□], cell[…], cell[□], cell(fill: resp)[□], cell(fill: resp)[□])),
      ..row([埋め込み], (cell[●], cell[●], cell[●], cell[…], cell[●], cell(fill: resp)[●], cell(fill: resp)[●])),
      ..row([入力のID], (cell(text(size: 5.5pt, "<|im_start|>")), cell[user], cell[改行], cell[…], cell[改行2つ], cell(fill: resp)[東京], cell(fill: resp)[です])),
    )
  },
  kind: image,
  caption: [layer × positionの図。position 0〜19はプロンプト、20以降は応答（網掛け）。◎は次のIDを選ぶのに使うlogits。],
)

- *1列* は、一つのIDを入れて、下から上へforward passを1回通すことに当たる。一番下で埋め込み、途中で24個のlayer、一番上でlogitsを出す。
- *列は1本ずつ右へ伸びる。* 図は最初から全部あるのではない。始めにあるのはプロンプトの20列（position 0〜19）だけだ。position 19の頂上のlogitsから「東京」を選んで初めて、position 20の列ができる。position 20の頂上から「です」を選ぶと、position 21の列ができる。
- *各列の□は、真下と、同じ行の左の列に頼る。* 真下からは同じpositionの値を受け取る（内側のループ）。同じ行の左からは、そのlayerが過去のpositionについて残した値を読む。右の列には頼らない。左の列が残す値と、その読み方が、この章の後半の主題だ。
- *使うlogitsは、頂上の◎だけ。* プロンプトの列のうち、次のIDを選ぶのに使うのは最後のposition 19だけだ。position 0〜18の頂上は計算しても使わない。次のIDはプロンプトとして決まっているからだ。それでもposition 0〜18を計算するのは、各layerがそこで残す値を、右の列が読むためだ。

横に伸ばせる長さには上限がある。プロンプトと応答を合わせて262,144 positionまでだ（設定ファイルの `max_position_embeddings`）。これは最初から用意された幅ではなく、右へ伸ばしてよい上限で、推論器の `prefill_inner` が確かめている。

一番下の *埋め込み* は、IDを行番号にして、248,320行×1,024列の *埋め込み表* から1行を取り出す。各行は1,024個の数の並び（ベクトル）で、ここで初めてトークンが計算に使える数になる。一番上の *LM head* は、最後のベクトルを埋め込み表の各行と比べ、248,320個のスコア（logits）を出す。Qwen3.5-0.8BのLM headは、入口の埋め込み表をそのまま使い回す（*tied embedding*）。入口では「ID → ベクトル」、出口では「ベクトル → 各IDのスコア」と、同じ表を逆向きに使う。

== パラメータは全positionで共有する

layerの計算に使う数の表を *パラメータ* と呼ぶ。パラメータは学習で決まった値で、推論中は変わらない。図の横方向にどれだけ列が伸びても、layer 3の□はどの列でもlayer 3の同じパラメータを使う。つまり、パラメータは縦一列ぶんの一組だけだ。

#table(
  columns: (1.7fr, 1fr),
  align: (left, right),
  table.header([置き場所], [パラメータの数]),
  [埋め込み表（LM headと共用）], [2.54億],
  [DeltaNetのlayer 18個], [1.90億],
  [Attentionのlayer 6個], [0.44億],
  [FFN（24個のlayerすべて）], [2.64億],
  [正規化の係数], [ごくわずか],
  [合計（言語モデルの部分）], [約7.5億],
)

パラメータは1個2バイトの *BF16* という形式で保存し、計算は1個4バイトの *f32* で行う（第4章）。言語モデルの部分だけで約1.5 GBになる。配布されるファイルには、このほかに画像を読むための部分（約1億個）と、MTP（下巻第10章）のための部分（約2,000万個）も入っている。本書は画像を扱わない。

本書は学習をしない。学習では、正解の文章を入れ、全部の列の頂上で「次のIDを当てられたか」を同時に採点し、その結果からパラメータを少しずつ直していく。どの列の採点も同じ一組のパラメータに集められて、何十億もの文章のどこでも働く値が残った。「首都」と聞かれたら「東京」と答えるような知識も、このパラメータのどこかにある。第6章では、その一列を実際に変えて、答えがどう変わるかを確かめる。

#source("spec/tensors.md")[パラメータの形]

== 1 layerの中

24個のlayerは、どれも同じ形をしている。1 layerは、真下から受け取ったベクトルを、2回書き足してから真上へ渡す。

+ ベクトルの大きさをそろえる（*正規化*）。
+ *token mixer* が、自分と左のpositionを読んで、書き足す値を作る。
+ 元のベクトルに足す（*残差接続*）。
+ もう一度、大きさをそろえる。
+ *FFN* が、自分のpositionだけから、書き足す値を作る。
+ 元のベクトルに足す。

どちらの書き足しも、元のベクトルを捨てずに足し込む。そのため、各列を下から上へ1本のベクトルが流れ、各layerがそれを読んでは少しずつ書き足していく、と見ることができる。この流れを *residual stream* と呼ぶ。最初は「首都」という語そのものを表していたベクトルが、上へ行くにつれて、周りの語や次に来そうな語の情報を含むようになる。

正規化は、ベクトルの成分全体の大きさを一定にそろえる段だ。24回も足し込むと、大きさが位置やlayerによってばらつく。各部品に入る前にそろえておけば、部品は大きさではなく向き（成分の比）に集中できる。24個のlayerを出た後、LM headの前にも1回入る。計算方法（RMSNorm）は第4章で扱う。

#source("src/block.rs")[1 layerの順序]

== FFNとtoken mixerの役割分担

1 layerの中の部品は、どこから情報を取ってくるかで役割が分かれる。

#table(
  columns: (1fr, 1.5fr, 2fr),
  table.header([部品], [どこを読むか], [何をするか]),
  [FFN], [自分のpositionだけ], [手元のベクトルを、パラメータに従って組み替える。過去は見ない],
  [Attention], [自分と左の全positionを、正確に], [左のpositionから、必要な情報を探して持ってくる],
  [DeltaNet], [左のpositionをまとめた行列], [同じく持ってくるが、相手は1枚にまとめた要約],
)

*FFN* は、1,024個の成分を3,584個に広げ、また1,024個に戻す。広げるとき、同じ入力から二通りの値を作り、一方（gate）で他方を「どれだけ通すか」を成分ごとに絞る。この形をSwiGLUと呼ぶ（第5章）。

AttentionとDeltaNetは、どちらもtoken mixerだ。どちらも各positionで、自分のベクトルから3本のベクトルを作る。

- *q*（query）：このpositionが「何を探しているか」。
- *k*（key）：このpositionが「何についての情報か」という見出し。
- *v*（value）：見出しが合ったときに渡す中身。

違いは、kとvの残し方だ。

- *Attention* は、全positionのkとvを並べて残す。これを *KV cache* と呼ぶ。今のpositionのqを、残してある全部のkと比べ、よく合うpositionのvを多めに受け取る。position 4の「首都」のk・vも、position 20からそのまま読み直せる。その代わり、列が伸びるたびにKV cacheも伸びる。本書の実装（f32で持つ）では、1 position・1 layerあたり4 KiBだ。読み出した結果には、受け取る側のpositionが「どれだけ取り込むか」を決める *出力ゲート* がかかる。
- *DeltaNet*（正確にはGated DeltaNet）は、kとvを1枚の行列（*state*）に重ねて書き込む。qでその行列を引くと、合うvを混ぜたものが返る。行列の大きさは一定で、列が伸びても増えない。その代わり、古い内容は薄れ、混ざる。position 4だけを取り出して読み直すことはできない。書き込みの前後には、Attentionにない仕組みがある。
  - q・k・vを作る前に、直前3 positionと自分の値を混ぜる短い畳み込みを通す。そのための直近の値も、stateと一緒に残す。
  - positionごとに、行列の古い内容を *忘れる強さ* と、新しい内容を *書き込む強さ* を決める。どちらもパラメータから計算され、入力によって変わる。
  - 読み出した結果に出力ゲートがかかる。

Attentionの比較は、一度に一つの見方しかできない。「直前の語」と「話題の語」を同時に見たいこともある。そこで、q・k・vを何組かに分け、独立に探す。この1組を *ヘッド* と呼ぶ。

- Attentionのlayerには、qのヘッドが8個（各256成分）、k・vのヘッドが2個ある。qのヘッド4個が、k・vのヘッド1個を共有して読む。この共有をGQA（Grouped-Query Attention）と呼ぶ。KV cacheは、共有しない場合の4分の1で済む。
- DeltaNetのlayerは16個のヘッドを持ち、ヘッドごとに128×128の行列を1枚ずつ書き換える。16枚 × 128 × 128 × 4バイトで、1 layerあたり1 MiBになる。

24個のlayerには、この2種類のtoken mixerが3対1で並ぶ。DeltaNetを3個続けた後にAttentionを1個置き、この4個の組を6回繰り返す。どのlayerにもFFNがある。

#figure(
  table(
    columns: (auto, 1fr, 1fr, 1fr, 1fr),
    align: center,
    inset: 4pt,
    table.header([組], [layer 4k], [layer 4k+1], [layer 4k+2], [layer 4k+3]),
    ..range(6).map(k => (
      [#k],
      [DeltaNet], [DeltaNet], [DeltaNet],
      table.cell(fill: luma(225))[Attention],
    )).flatten(),
  ),
  caption: [24個のlayerの並び。Attentionはlayer 3, 7, 11, 15, 19, 23の6個。],
)

DeltaNetは安く、大まかな流れを拾う。Attentionは高くつくが、遠くの特定の語を正確に拾い直せる。大半を安い方にして、4 layerに1回だけ正確に読み返す、という組み合わせだ。Qwenの開発元は、この混ぜ方がどちらか一方だけの構成より良かったと説明している。比率の詳しい根拠は公開されていない（第3章）。

本書で使う寸法は、次の表にまとめる。値はモデルに付属する設定ファイル `config.json` のものだ。

#table(
  columns: (1.6fr, 1fr),
  table.header([項目], [値]),
  [layerの数], [24（DeltaNet 18、Attention 6）],
  [ベクトルの幅], [1,024],
  [語彙の数（埋め込み表の行数）], [248,320],
  [FFNの中間の幅], [3,584],
  [Attentionのヘッド], [q 8個、k・v 2個、各256成分],
  [DeltaNetのヘッド], [16個、各128成分],
  [DeltaNetの畳み込みの幅], [4（自分と直前3 position）],
  [扱えるpositionの上限], [262,144],
)

#source("spec/upstream/config.json")[モデルの設定] ／ #source("src/state.rs")[残す値の型（`KvCache`、`DeltaState`）]

== どこを見るかと語順

Attentionで、左のどのpositionをどれだけ読むかは、qとkの比べ方で決まる。よく合うほど大きなスコアになり、スコアをsoftmaxで合計1の割り振りにしたものを *attention weight* と呼ぶ。qとkを作るパラメータは学習で決まっているので、何を探し、何を見出しにするかも学習で決まっている。

自分より右のpositionは、まだ存在しない列だ。そのため、比べる相手は自分と左だけに限る。この制限を *因果マスク* と呼ぶ。

実際に、最初のAttention（layer 3）で、position 4の「首都」に置いたattention weightを、8個のヘッドのうち3個について抜き出した。

#figure(
  table(
    columns: (1.6fr, 1fr, 1fr, 1fr),
    align: center,
    inset: 5pt,
    table.header([読む側のposition ＼ ヘッド], [0], [1], [3]),
    [4 首都], [0.637], [0.390], [0.538],
    [19 改行2つ], [0.012], [0.001], [0.002],
    [20 東京],
    table.cell(fill: luma(225))[*0.598*],
    table.cell(fill: luma(225))[0.460],
    table.cell(fill: luma(225))[0.566],
  ),
  caption: [layer 3で、position 4「首都」に置いたattention weight（実測値の抜粋）。],
)

position 20で「東京」を処理するとき、3個のヘッドがいずれも半分前後を「首都」に置いている。ヘッドごとに値が違い、見ている所が少しずつ違うことも分かる。一方、「東京」を選んだposition 19は、このlayerではほとんど「首都」を読んでいない。position 19が「首都」の情報をどこから得たかは、このlayerだけでは分からない。手前の3個のDeltaNetや、後のAttentionも関わるからだ（第3章）。

ここで一つ問題がある。qとkの比較は、ベクトルの中身だけを見ている。このままでは「日本の首都」と「首都の日本」で、どの語がどの語をどれだけ読むかが同じになる。語の並び順の情報が、どこにも入っていないからだ。

Qwenは、qとkを比べる前に、positionの番号に比例した角度だけ回す。この方法を *RoPE*（Rotary Position Embedding）と呼ぶ。時計の針で考えるとよい。position 3のkは3目盛り、position 20のqは20目盛り回す。二つを比べると、見えるのは針どうしの角度の差、つまり17目盛りだ。二つの針を同じだけ回しても差は変わらないので、比較の結果は「何番目か」ではなく「どれだけ離れているか」で決まる。こうして比較に、中身が合うかに加えて、距離が入る。

RoPEは、どこを見るかを直接決めるわけではない。決めるのは、学習で決まったqとkの作り方だ。RoPEはそこに距離の物差しを渡す。その物差しを使って、学習の結果「直前の語を見る」「近くの語を見る」「離れていても特定の語を見る」といった見方ができるようになる。

細部を二つ添える。

- Qwen3.5-0.8Bでは、qとkの256成分のうち64成分だけを回し、残りは回さない。比較の一部は中身だけを見て、一部は中身と距離を見る。
- 比べる前に、qとkの大きさもヘッドごとにそろえる（正規化）。

DeltaNetはRoPEを使わない。左のpositionから順に行列へ書き込んでいくので、並び順は処理の順番に最初から入っているからだ。並び順の情報が別に要るのは、全positionを一度に並べて比べるAttentionだけだ。回転の計算は第5章で追う。

#source("src/attention.rs")[Attention] ／ #source("spec/images/attention-layer3.svg")[layer 3・8ヘッドのヒートマップ]

== prefillとdecode

外側のループには二つの段階がある。

- *prefill*：プロンプトのIDを順に入れて、position 0〜19の列を埋める段階。IDは最初から全部分かっている。
- *decode*：一つ左の列の頂上で選んだIDを入れて、応答の列を1本ずつ足す段階。

どちらの段階でも、1列の計算は同じだ。推論器でも、prefillとdecodeは同じ関数で1列を計算する。違うのは、入れるIDがどこから来るかだけだ。

```rust
// InferenceSession（src/session.rs）の抜粋。ingest が1列の計算。
for &id in ids {                   // prefill：プロンプトのIDを順に入れる
    self.ingest(model, id, false, observer)?;
}
for _ in 0..max_new {              // decode
    self.flush(model, observer)?;  // 前の回に選んだIDを入れる（中で ingest）
    let token = select(self.logits.as_ref().unwrap(), self.sampling, &mut self.rng)?;
    self.pending = Some(token);
    if eos.contains(&token) {
        break;
    }
}
```

この違いは、計算の順番に効いてくる。prefillでは、一番下の行（入力のID）が最初から全部そろっている。そのため、layer 0を全部の列でまとめて計算し、次にlayer 1を全部の列で、と行ごとに進めることもできる。decodeでは、次のIDが一つ左の列の頂上を見るまで決まらないので、1列ずつ下から上へ進むしかない。パラメータは行（layer）ごとに一組なので、行ごとにまとめれば、同じパラメータを何列分にも使い回せる。実際の推論で、prefillが1トークンあたりdecodeよりずっと速いのはこのためだ。第3章でこの違いを1 positionずつ追い、下巻でGPUを使って速くする。

#source("src/session.rs")[prefillとdecodeのループ]

== 本書の道筋とコードの地図

上巻は、ここで見た計算をCPUで正しく理解し、最後にパラメータに介入する。下巻は、同じ計算をGPUで速くする。

#table(
  columns: (0.7fr, 2.3fr),
  table.header([章], [この章の地図のどこを開くか]),
  [第2章], [文字列とIDの間（会話テンプレート、BPE、detokenize）],
  [第3章], [layer × positionの図を、prefillとdecodeの実際の値で辿る],
  [第4章], [計算に使う数の扱い（BF16、softmax、活性化関数、正規化）],
  [第5章], [1 layerを手で計算する（FFN、Attention、DeltaNet）],
  [第6章], [FFNのパラメータを一列変えて、答えがどう変わるかを見る],
  [下巻], [同じ計算をGPUで速くする（第7〜13章）],
)

ここまでに出た処理と、それを担うRustの関数を並べる。

#table(
  columns: (1.25fr, 1fr, 1.45fr),
  table.header([処理], [ファイル], [主な関数・型]),
  [会話テンプレート], [`src/text.rs`], [`user_chat`],
  [符号化とdetokenize（本番）], [`src/text.rs`], [`TextTokenizer`],
  [前処理とBPE（教材用）], [`src/bpe.rs`], [`TeachingBpe::encode`],
  [外側のループ], [`src/session.rs`], [`InferenceSession`],
  [次のIDの選択], [`src/generate.rs`], [`select`],
  [内側のループ], [`src/model.rs`], [`Model::step_hidden`],
  [1 layer（正規化・残差接続）], [`src/block.rs`], [`BlockWeights::step`],
  [Attention], [`src/attention.rs`], [`AttentionWeights::step`],
  [DeltaNet], [`src/deltanet.rs`], [`DeltaWeights::step`],
  [FFN], [`src/ffn.rs`], [`FfnWeights::forward`],
  [正規化・softmaxなど], [`src/ops.rs`], [`rms_norm`、`softmax`],
  [残す値（KV cache、state）], [`src/state.rs`], [`ModelState`],
  [パラメータの読み込み], [`src/weights.rs`], [`load_model`],
)

#exercise[会話テンプレートのposition 13にある `<|im_start|>` について、次の四つを区別しよう。(1) トークンID、(2) position、(3) 埋め込みで取り出す値、(4) layer 3がこの列の計算の後に残すもの。答えは、(1) 248045、(2) 13、(3) 埋め込み表の248045行目（position 0と同じ1,024個の数）、(4) position 13のk・v（KV cacheに1組加わる）。]

#exercise[layer 0がposition 13の計算の後に残すものは何か。DeltaNetのlayerなので、kとvを書き込んだ後の行列（16ヘッド分）と、畳み込みのための直近の値だ。position 12の後と大きさは変わらない。]
