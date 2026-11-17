# Rustで学ぶLLM推論

Rustのコードを読めて、LLMの内部を知りたい人に向けた、技術書典用のTypst原稿です。

- 上巻「Qwenの1トークンをCPUで追う」は第1〜3章。文字列からID、埋め込み、各層の状態、次のIDまでを扱います。
- 下巻「QwenをMetalで速くする」は第4〜6章と付録。prefill、decode、投機的デコードの実装と測定を追います。

## 組版

Typst CLIとNoto Serif CJK JP、Noto Sans CJK JPを用意します。コードにはDejaVu Sans MonoとNoto Sans Mono CJK JPを指定しています。リポジトリのルートから実行します。

```sh
typst compile book/upper.typ /tmp/qwen-upper.pdf
typst compile book/lower.typ /tmp/qwen-lower.pdf
```

既存の `main.typ` でも、全章を一つに束ねた通読用PDFを組めます。判型は仮のA5・左綴じです。

## 現在の原稿

二分冊の組版試作は上巻28頁、下巻48頁です。刊行時の頁数は、ほかの節にも実例とコードを入れてから決めます。

- 上巻12–17頁で、2位置のAttentionを射影、RoPE、KV、softmax、Rustの順に追います。
- 下巻4–11頁で、RustのループからMetalのタイルの配置、ロード、同期、書き戻しまでを追います。
- 上巻20–21頁と下巻36–37頁は、同じ状態表でAttentionとDeltaNetの更新・巻き戻しを説明します。
- [執筆計画](PLAN.md) に、読者の前提、各章の到達点、今後の具体化をまとめています。
