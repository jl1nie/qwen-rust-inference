#import "../style.typ": source

#heading(numbering: none)[付録　コードと測定を再現する]

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
- #source("spec/cuda.md")[CUDAの準備とGEMVの照合・測定]
- #source("spec/measurements/metal-m5-greedy-fused.json")[MetalのMTP測定ログ]
