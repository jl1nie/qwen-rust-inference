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

== FFN：自分のpositionだけで組み替える

1 layerの中の部品は、どこから情報を取ってくるかで役割が分かれる。

#table(
  columns: (1fr, 1.5fr, 2fr),
  table.header([部品], [どこを読むか], [何をするか]),
  [FFN], [自分のpositionだけ], [手元のベクトルを、パラメータに従って組み替える。過去は見ない],
  [Attention], [自分と左の全positionを、1つずつ], [左のpositionから、必要な情報を探して持ってくる],
  [DeltaNet], [左のpositionを書き込んだ行列], [同じく持ってくるが、相手は1枚に重ねて書いた要約],
)

*FFN* は、1,024個の成分を3,584個に広げ、また1,024個に戻す。広げるとき、同じ入力から二通りの値を作り、一方（gate）で他方を「どれだけ通すか」を成分ごとに絞る。この形をSwiGLUと呼ぶ（第5章）。FFNは何も残さないので、position同士のつながりには関わらない。token mixerが左から持ってきた情報を、そのpositionで組み替えて次のlayerに渡す役目だ。

残る2つのtoken mixerは、どちらも *q・k・v* という3本のベクトルを使う。まず、その作り方から見る。

== q・k・vはどう作られるか

layer 3（Attention）で、position 20の列を計算しているところを考える。真下から来たベクトル x は1,024個の数で、「東京」を入れた列のこの時点の値だ。これに、layer 3専用の3枚の表を掛ける。

```text
x（1,024個。入力ごとに変わる）
  ├─ W_Q（4,096行×1,024列）を掛ける → 4,096個 → q とゲート
  ├─ W_K（  512行×1,024列）を掛ける →   512個 → k
  └─ W_V（  512行×1,024列）を掛ける →   512個 → v
```

「表を掛ける」とは、表の各行について、x の1,024個の成分にその行の1,024個の値をそれぞれ掛けて、足し合わせることだ。1行から1個の数が出る。x が4成分、表が2行の小さな例で確かめる。

#figure(
  table(
    columns: (1.2fr, 2fr, 0.6fr),
    align: (center, left, center),
    table.header([W_Kの行], [x = (1, 0, 2, 1) との計算], [kの成分]),
    [(1, 0, 0, 1)], [1×1 + 0×0 + 0×2 + 1×1], [2],
    [(0, 1, 1, 0)], [0×1 + 1×0 + 1×2 + 0×1], [2],
  ),
  caption: [表を掛ける小さな例。行の値は「xのどの成分をどれだけ拾うか」を表す。],
)

表の行の値は学習で決まっていて、推論中は変わらない。position 4（「首都」）の列でも、layer 3なら同じ表を使う。変わるのは x の方で、position 4では x が違うので、同じ表から違うkができる。W_Kは「xのどんな特徴を見出しにするか」、W_Qは「何を探すか」、W_Vは「何を渡すか」を、学習で決めた表だと言える。

- *q*（query）：このpositionが「何を探しているか」。
- *k*（key）：このpositionが「何についての情報か」という見出し。
- *v*（value）：見出しが合ったときに渡す中身。

ここで出てきた値は、決まる時と寿命で3種類に分けられる。

#table(
  columns: (1.1fr, 1.3fr, 1.3fr, 1.3fr),
  table.header([種類], [例], [いつ決まるか], [positionごとに]),
  [学習したパラメータ], [W_Q、W_K、W_V、出力の表], [学習時。推論中は変わらない], [同じ（全positionで共有）],
  [その場で計算する値], [q、k、v、ゲート、attention weight], [その列を計算する時], [違う。qとゲートは使い捨て],
  [残す値], [KV cacheのk・v、DeltaNetのstate], [その列を計算した後], [会話が続く間、右の列が読む],
)

*ヘッド* は、作った数の並びを決まった長さで区切った一片だ。別の何かがあるわけではない。

```text
k（512個）    = [ kヘッド0：256個 | kヘッド1：256個 ]
W_Qの出力（4,096個）
  = [ qヘッド0：256 | ゲート0：256 | qヘッド1：256 | ゲート1：256 | … | qヘッド7 | ゲート7 ]
```

qヘッド0の256個は、W_Qの決まった256行から作られる。行がヘッドごとに分かれているので、ヘッドごとに別の見方を学習している。推論器でも、作った並びを `chunks_exact` で256個ずつ区切っているだけだ。

DeltaNetも作り方は同じだ。1枚の表（6,144行×1,024列）から6,144個の数を作り、q・k・vを2,048個ずつ取り、それぞれを128個ずつ16ヘッドに区切る。違うのは、作ったk・vをどう使うかだ。

#source("src/attention.rs")[Attentionの射影とヘッドの区切り] ／ #source("spec/tensors.md")[表の形]

== Attention：並べて残し、1つずつ比べる

layer 3のposition 20では、次の7段階が進む。

+ *作る*：x にW_Q・W_K・W_Vを掛け、q（8ヘッド×256）、ゲート（8×256）、k（2×256）、v（2×256）を作る。
+ *そろえる*：qとkの大きさを、ヘッドごとに正規化する。
+ *回す*：各ヘッドのqとkの先頭64成分を、position 20の角度だけ回す（後述のRoPE）。
+ *残す*：kとvを、position 20の分として、layer 3の *KV cache* の末尾に足す。KV cacheにはposition 0〜20の21組がそろう。本書の実装はf32で持つので、1 position・1 layerあたり4 KiBだ。
+ *ヘッドごとに読む*：qヘッド0〜7が、それぞれ独立に読む。
  - qヘッド0〜3はk・vヘッド0を、qヘッド4〜7はk・vヘッド1を相手にする。qヘッド4個がk・vヘッド1個を共有するこの方式をGQA（Grouped-Query Attention）と呼び、KV cacheは共有しない場合の4分の1で済む。
  - layer 3のKV cacheにある相手の21個のkとqを比べてスコアを出す。よく合うほど大きい。
  - スコアをsoftmaxで合計1の割り振りにする。これが *attention weight* だ。
  - attention weightの割合で21個のvを混ぜ、256個の結果を出す。
+ *絞る*：各ヘッドの結果に、そのヘッドのゲートをかける。受け取る側のpositionが「どれだけ取り込むか」を決める *出力ゲート* だ。
+ *まとめる*：8ヘッドの結果（8×256 = 2,048個）をつなぎ、出力の表で1,024個に戻して、residual streamに足す。

比べる相手は自分と左だけだ。右のpositionはまだ存在しない列なので、比べる相手から外す。この制限を *因果マスク* と呼ぶ。

最初のAttention（layer 3）で、position 4の「首都」に置いたattention weightを、8個のqヘッドのうち3個について抜き出した。

#figure(
  table(
    columns: (1.6fr, 1fr, 1fr, 1fr),
    align: center,
    inset: 5pt,
    table.header([読む側のposition ＼ qヘッド], [0], [1], [3]),
    [4 首都], [0.637], [0.390], [0.538],
    [19 改行2つ], [0.012], [0.001], [0.002],
    [20 東京],
    table.cell(fill: luma(225))[*0.598*],
    table.cell(fill: luma(225))[0.460],
    table.cell(fill: luma(225))[0.566],
  ),
  caption: [layer 3で、position 4「首都」に置いたattention weight（実測値の抜粋）。],
)

position 20で「東京」を処理するとき、3個のヘッドがいずれも半分前後を「首都」に置いている。値はヘッドごとに違い、見ている所が少しずつ違う。一方、「東京」を選んだposition 19は、このlayerではほとんど「首都」を読んでいない。position 19が「首都」の情報をどこから得たかは、このlayerだけでは分からない。手前の3個のDeltaNetや、後のAttentionも関わるからだ（第3章）。

ここで一つ問題がある。qとkの比較は、ベクトルの中身だけを見ている。このままでは「日本の首都」と「首都の日本」で、どの語がどの語をどれだけ読むかが同じになる。語の並び順の情報が、どこにも入っていないからだ。

そこで、比べる前にqとkを、positionの番号に比例した角度だけ回す。この方法を *RoPE*（Rotary Position Embedding）と呼ぶ。時計の針で考えるとよい。position 3のkは3目盛り、position 20のqは20目盛り回す。二つを比べると、見えるのは針どうしの角度の差、つまり17目盛りだ。両方を同じだけ回しても差は変わらないので、比較の結果は「何番目か」ではなく「どれだけ離れているか」で決まる。

各ヘッドの256成分のうち、回すのは64成分だ。64成分を2個ずつ32組にし、組ごとに回る速さを変える。残りの192成分は回さない。

#table(
  columns: (1fr, 1.4fr, 1.4fr),
  align: (center, right, right),
  table.header([組], [1 positionで回る角度], [一周する距離]),
  [0], [1ラジアン], [約6 position],
  [4], [0.13ラジアン], [約47 position],
  [8], [0.018ラジアン], [約350 position],
  [16], [0.0003ラジアン], [約2万 position],
  [31], [ごくわずか], [約3,800万 position],
  [回さない192成分], [0], [—],
)

秒針から、何年もかかって一周する針まで、32本の針と動かない部分があるようなものだ。回る速さは設定ファイルの値（`rope_theta`）から式で決まっていて、学習した値ではない。8個のヘッドはどれも同じ針の組を持つ。

ヘッドごとに違うのは、どの針に情報を載せるかだ。qとkの比較は全成分の掛け算を足し合わせたものなので、大きな値が載った成分ほど比較を左右する。

- 速く回る組に大きな値を載せたヘッドは、少し離れるだけで角度の差が変わる。近くのpositionを見分けるのに向く（たとえば「直前の語」を見る）。
- 遅い組や回さない成分に載せたヘッドは、距離でほとんど変わらず、中身だけで比べる。遠くの特定の語を見るのに向く（たとえば、何position離れていても「首都」を見る）。

どの成分にどれだけ載せるかは、そのヘッドを担当するW_Q・W_Kの行が決め、それは学習で決まっている。周期は固定で、その使い方を学習している、ということだ。ただし、ヘッドごとに見る場所が違う一番の理由は、距離より中身にある。ヘッドごとにW_Qの担当の行が違い、「何を探すか」そのものが違うからだ。回転の計算は第5章で追う。

#source("src/attention.rs")[回転 `rotate` と読み出し `attend`] ／ #source("spec/images/attention-layer3.svg")[layer 3・8ヘッドのヒートマップ]

== DeltaNet：1枚に書き込み、まとめて引く

DeltaNetは、kとvを並べて残さない。ヘッドごとに128×128の行列（*state*）を1枚持ち、kとvをそこへ書き込む。kが宛先、vが中身の書き込み帳だと考えるとよい。kを2成分、vを1成分にした小さな例で確かめる。

- k = (1, 0) の宛先に、中身 v = 5 を書く。次に、k = (0, 1) の宛先に v = 3 を書く。
- q = (1, 0) で引くと、5が返る。宛先がぴったり合うので、その中身が出てくる。q = (0, 1) なら3だ。
- q = (0.7, 0.7) で引くと、両方を混ぜた5.6（0.7×5 + 0.7×3）が返る。

Attentionなら5と3を別々に残し、qと比べてから混ぜ方を決める。DeltaNetは最初から1枚に重ねて書く。そのため、行列の大きさは列が伸びても変わらないが、宛先のよく似た2つの中身は混ざって区別できなくなる。position 4の「首都」だけを取り出して読み直すこともできない。

layer 0のある position では、各ヘッドで次の5段階が独立に進む。

+ *作る*：x に表を掛けてq・k・vを作る。直前3 positionの値と混ぜる短い畳み込みを通し、qとkを長さ1にそろえる。同じ x から、*忘れる強さ*、*書き込む強さ*、出力ゲートも計算する。どれもpositionごとに変わる。
+ *忘れる*：このlayer（layer 0）の行列全体に、忘れる強さから決まる0〜1の係数を掛けて薄める。
+ *差分を書く*：まず今のkでlayer 0の行列を引き、「この宛先に今書いてある中身」を読み出す。新しいvとの差に書き込む強さを掛け、その分だけを書き足す。同じ宛先に同じ中身が何度来ても、書き込みは積み上がらない。差分（delta）だけを書くこの規則を *デルタ則* と呼び、DeltaNetの名前の由来になっている。
+ *読む*：書き込んだ後のlayer 0の行列をqで引き、128個の答えを出す。比べて割り振る段階がないので、softmaxもattention weightもない。
+ *まとめる*：16ヘッドの答えをそれぞれ正規化して出力ゲートをかけ、つないで（2,048個）、出力の表で1,024個に戻し、residual streamに足す。

行列には左から順に書いた分しか入っていないので、因果マスクは要らない。RoPEも使わない。古い内容ほど何度も薄められ、直前3 positionは畳み込みで混ざる。並び順は、左から順に書いていく処理そのものに入っている。16ヘッド × 128 × 128 × 4バイトで、行列は1 layerあたり1 MiBだ。直前3 positionの値も、畳み込みのために一緒に残す。

#source("src/deltanet.rs")[デルタ則 `delta_rule`]

== 2つのtoken mixerを比べる

#table(
  columns: (0.9fr, 1.6fr, 1.6fr),
  table.header([], [Attention], [DeltaNet]),
  [q・k・v], [学習済みの表をxに掛ける], [同じ（1枚の表から3本）],
  [作った後], [正規化、RoPEで回す], [畳み込み、正規化。RoPEなし],
  [ヘッド], [q 8個、k・v 2個（各256）], [q・k・v 16個（各128）],
  [他の値], [出力ゲート], [出力ゲート、忘れる・書き込む強さ],
  [残すもの], [k・v（KV cache）。列ごとに伸びる], [行列（state）。大きさ一定],
  [読み方], [qを全部のkと1つずつ比べる], [行列にqを掛ける],
  [因果マスク], [要る], [要らない],
  [得意なこと], [遠くの特定の語を正確に], [安く、大まかな流れを],
)

24個のlayerには、この2種類が3対1で並ぶ。DeltaNetを3個続けた後にAttentionを1個置き、この4個の組を6回繰り返す。どのlayerにもFFNがある。

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

大半を安いDeltaNetにして、4 layerに1回だけ、正確に読み直せるAttentionを入れる組み合わせだ。Qwenの開発元は、この混ぜ方がどちらか一方だけの構成より良かったと説明している。比率の詳しい根拠は公開されていない（第3章）。

本書で使う寸法は、次の表にまとめる。値はモデルに付属する設定ファイル `config.json` のものだ。

#table(
  columns: (1.6fr, 1fr),
  table.header([項目], [値]),
  [layerの数], [24（DeltaNet 18、Attention 6）],
  [ベクトルの幅], [1,024],
  [語彙の数（埋め込み表の行数）], [248,320],
  [FFNの中間の幅], [3,584],
  [Attentionのヘッド], [q 8個、k・v 2個、各256成分],
  [RoPEで回す成分], [各ヘッド256成分のうち64成分],
  [DeltaNetのヘッド], [q・k・vとも16個、各128成分],
  [DeltaNetの畳み込みの幅], [4（自分と直前3 position）],
  [扱えるpositionの上限], [262,144],
)

#source("spec/upstream/config.json")[モデルの設定] ／ #source("src/state.rs")[残す値の型（`KvCache`、`DeltaState`）]

== 残す値はいつ書かれ、いつ読まれるか

KV cacheとstateは、layerごとに別々にある。6個のAttention layerがそれぞれ自分のKV cacheを、18個のDeltaNet layerがそれぞれ自分のstateを持つ。あるlayerが書いたものを読むのは、同じlayerだけだ。推論器の `ModelState` は、この置き場所を24個持っている。

```rust
// src/state.rs。layers は24個、layer i の置き場所が layers[i]。
pub struct ModelState { pub position: usize, pub layers: Vec<LayerState> }
pub enum LayerState { Attention(KvCache), Delta(DeltaState) }
// KV cache：position ごとに k・v を1組ずつ末尾に足す。
pub struct KvCache { pub keys: Vec<Vec<f32>>, pub values: Vec<Vec<f32>> }
// state：畳み込み用の直近の値（convolution）と行列（recurrent）。
pub struct DeltaState { pub convolution: Vec<f32>, pub recurrent: Vec<f32> }
```

内側のループの `zip(&mut state.layers)` は、layer iの計算に、layer iの置き場所だけを渡すためのものだ。

#table(
  columns: (1.3fr, 1.6fr, 1.4fr),
  table.header([layer], [置き場所の中身], [大きさ（f32）]),
  [0, 1, 2, 4, 5, 6, …（DeltaNetの18個）], [自分専用のstate：16ヘッド分の行列と、畳み込み用の直前3 positionの値], [行列1 MiB＋直近の値72 KiB。positionが増えても一定],
  [3, 7, 11, 15, 19, 23（Attentionの6個）], [自分専用のKV cache：positionごとのk・v（2ヘッド分）], [1 positionあたり4 KiB。positionごとに伸びる],
)

position 20の列では、次の順に書いて読む。

```text
position 20の列（下から上へ）
  layer 0  DeltaNet ：layer 0の行列を薄める → 差分を書く → qで読む
  layer 1  DeltaNet ：layer 1の行列で同じこと
  layer 2  DeltaNet ：layer 2の行列で同じこと
  layer 3  Attention：k・vをlayer 3のKV cacheの末尾に足す → qでposition 0〜20を読む
  layer 4  DeltaNet ：layer 4の行列で同じこと
  …
  layer 23 Attention：layer 23のKV cacheに足して読む
→ logits →「です」を選ぶ → position 21の列へ
```

規則は三つだけだ。

+ *書くのは、その列がそのlayerに来た時に1回。* Attention layerはk・vを1組足し、DeltaNet layerは行列を1回書き換える。
+ *読むのは、書いた直後の同じ列と、それより右の全部の列。* ただし、同じlayerの置き場所しか読まない。position 20がlayer 7のKV cacheを読むのは、position 20の計算がlayer 7に来た時だ。
+ *書き換えられ方が違う。* KV cacheは左から順に末尾へ足していくだけで、一度足したk・vは書き換えない。position 4の「首都」のk・vは、position 20からもそのまま読める。DeltaNetの行列は毎列書き換わり、position 20が読むのはposition 0〜20の書き込みが重なった後の1枚だ。

layer × positionの図で「同じ行の左の列が残した値」と呼んだものの正体がこれだ。Attentionの行では、左の列が1組ずつ置いたk・vが行に沿って並ぶ。DeltaNetの行では、左の列から順に書き換えられてきた1枚が、右の列へ受け渡されていく。

prefillも、この置き場所を更新する。prefillでposition 0〜19を計算し終えると、6個のKV cacheにはそれぞれ20組がたまり、18個のstateはposition 0〜19を順に書き込んだ1枚になる。prefillの成果は、この置き場所の中身と、最初の応答を選ぶposition 19のlogitsの二つだ。position 0〜18のlogitsを使わないのに計算するのは、この置き場所を作るためだ。decodeは、その続きから1列ずつ、KV cacheに1組足し、stateを1回書き換える。

量の目安を出しておく。position 20の時点で、KV cacheは6 layer × 21 position × 4 KiBで約0.5 MiB、stateは18 layer × (1 MiB＋72 KiB)で約19 MiBだ。短い会話ではstateの方がずっと大きいが、positionが数千を超えると、伸び続けるKV cacheの方が大きくなる。

#source("src/state.rs")[置き場所の型]

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
