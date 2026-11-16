#import "../style.typ": exercise, source

#heading(numbering: none)[付録　ROCmの実測とCUDAでの検証設計]

#heading(level: 2, numbering: none)[A　同じモデルをAMD GPUで実行する]

MetalとROCmは、重み、モデルの演算順、状態の採用規則を扱うRust側の実行器を共有する。AMD側のカーネルはHIPで書き、ブリッジがGPUへ演算を発行する。ここで読む実測はRX 7900 XTX、WSL2、ROCm 7.2.4の記録だ。

4トークンの入力から始めるdecodeでは、改善前は約175 token/s。1.5 GBを毎ステップ読む換算で約260 GB/sだった。ピーク帯域に対して余地があったため、個別の射影がどれだけ読み出せるかと、小カーネルの発行にかかる時間を分けて調べた。

WSL2上の短い演算は、計時イベントの費用や、同じカーネルを繰り返したときのキャッシュの影響を受ける。実装では測定方式を記録して、単体測定と通常decodeを併記した。

#source("spec/rocm.md")[ROCm版の構成と実行環境]

#pagebreak()
#heading(level: 2, numbering: none)[A.1　ヘッドと重みの読み出しを割り振る]

Q/Kの正規化などは、1ヘッドを1 threadで計算していたものを、1 waveで成分を分担するように変えた。大きな射影では、各laneが次に使う重みを先に読み込み、積和の入力を待つ時間を減らした。小さな射影と隣り合う演算は、条件の合う範囲で一括発行や融合を行った。

#table(
  columns: (1.4fr, 1fr, 1fr),
  table.header([4トークン入力], [変更前], [変更後]),
  [1ステップの発行], [459回], [129回],
  [decode], [約175 token/s], [約330 token/s],
  [重み量で換算した帯域], [約260 GB/s], [約500 GB/s],
)

値は一連の変更を含む。長い入力ではdecodeの時間とK/Vの量が違うため、入力長ごとの結果を参照する。各段階で測定入力の生成IDを確認している。

#source("spec/adr-0019.md")[計測方法と変更の段階]

#pagebreak()
#heading(level: 2, numbering: none)[A.2　投機的デコードの採算は装置で変わる]

途中状態の巻き戻しをROCmへ移植した段階では、通常greedyと同じID列を生成できた。下書き2個・8,192語の条件では速度は約0.97倍。検証できることと、その検証に掛かる費用を回収できることは、別々に測れる。

その後、GPU上のargmaxとデバイス上での下書きの接続を使った測定では、下書き2個で1.28–1.30倍だった。通常greedyもデバイスargmaxの場合の比較だ。M5では同じ下書き数で1.63倍だった。装置間の倍率を読む際は、通常decodeの速さを基準に、少数行の検証にどれだけ時間が掛かるかも見る。

#source("spec/adr-0029.md")[巻き戻しの移植] ／
#source("spec/adr-0032.md")[デバイス上の選択と通し測定]

#pagebreak()
#heading(level: 2, numbering: none)[B　CUDAで同じ問いを検証する]

CUDAでは、第5章のBF16 GEMVを一つ選び、128位置のGEMMに進む前に、1位置の重み読み出しを調べる。対象は3,584行×1,024列。重みはBF16のまま読み、CPUでFP32を使って照合できる問題を作る。以下は移植実験の設計だ。

```text
1 warp に出力行を一つ割り当てる
各laneで連続する重みの担当分をロードする
BF16の重みを使ってFP32の部分和を作る
warp内で部分和を集約し、出力行へ書く
```

単純な割り当てを基準にして、後からまとめ読みや先行ロードを試す。GPUの発売世代や機種で結果が変わるので、機種名、実行環境、行列の形、精度を測定ログに残す。

#pagebreak()
#heading(level: 2, numbering: none)[B.1　キャッシュと発行を分けて測る]

まず同じ重みを繰り返し読む試行を行う。次に、より広い作業集合を循環させ、デバイスのキャッシュへ収まる場合と収まらない場合で時間を比べる。1回の発行に掛かる費用を見る小さな測定と、内積そのものの測定も別に行う。

#table(
  columns: (1fr, 1.6fr),
  table.header([変えるもの], [記録するもの]),
  [作業集合の大きさ], [有効バイト数、時間、換算帯域],
  [ロード方法], [同じ形での速度、CPU参照との差],
  [測定の粒度], [個別の発行、複数回の連続実行],
)

#exercise[キャッシュ内の試行で速くなり、キャッシュ外では変わらなかった。モデル全体で効くと言うために、何を追加で調べるか。実際の重みを順に読む経路と、decode全体の時間を測る。]

#pagebreak()
#heading(level: 2, numbering: none)[C　コードと測定を再現する]

実装と重みは、#source("spec/references.lock.json")[参照のロックファイル]で版を確認する。保存済みの数値と比較するときは、各ログに記載された実装版も見る。CPUだけなら、実モデルを使わずに合成重みのデモから始められる。

```sh
cargo run --offline --locked -- demo ffn --unit 3
cargo run --offline --locked -- demo attention --cache
cargo run --offline --locked -- demo deltanet
cargo test --offline --locked
```

Apple Siliconでは `bash tools/build-metal.sh` でブリッジをビルドし、`cargo build --release --offline --locked --features metal` で実行器を用意する。公開重み、ライブラリの指定、段階別測定、投機的デコードのコマンドは実装の案内に合わせる。

- #source("README.md")[公開重みの取得、実行コマンド、再現の入口]
- #source("spec/learning.md")[CPUの手計算とトレース]
- #source("spec/metal.md")[Metalの測定方法と切り替え]
- #source("spec/rocm.md")[ROCmの実行環境と検証]
- #source("spec/measurements/metal-m5-greedy-fused.json")[MetalのMTP測定ログ]
