# Rustで学ぶLLM推論

Rustのコードを読めて、LLMの内部を知りたい人に向けた、技術書典用のTypst原稿です。

- 上巻「Qwenの1トークンをCPUで追う」は全巻の基礎編（第1〜3章）です。トークン、埋め込み、状態、演算量と読み出し量を、小さな例から扱います。
- 下巻「QwenをMetalで速くする」は第4〜8章と付録です。AppleのMetalとGPUの両方を題材に、prefill、decode、投機的デコードを追い、ROCmの実測と、CUDAで組んだ同じ射影の照合・実測へ進みます。

## 組版

Typst CLIとNoto Serif CJK JP、Noto Sans CJK JPを用意します。コードにはDejaVu Sans MonoとNoto Sans Mono CJK JPを指定しています。リポジトリのルートから実行します。

```sh
typst compile book/upper.typ /tmp/qwen-upper.pdf
typst compile book/lower.typ /tmp/qwen-lower.pdf
```

既存の `main.typ` でも、全章を一つに束ねた通読用PDFを組めます。判型は仮のA5・左綴じです。

## 現在の原稿

二分冊の組版試作は上巻38頁、下巻56頁です。刊行時は上巻を総68頁・本文64頁で固定します。下巻も総68頁・本文64頁を第一目標とし、原稿量が足りなければ総64頁・本文60頁とします。総頁数との差4頁は表紙まわりとして計上し、本文には扉・目次・付録等を含めます。

- 上巻16–21頁で、2位置のAttentionを射影、RoPE、KV、softmax、Rustの順に追います。
- 上巻ではBPE、FFN、DeltaNetを小さな数値で計算し、35–36頁で演算量と読み出し量から速度を見積もります。
- 下巻4–11頁で、RustのループからMetalのタイルの配置、ロード、同期、書き戻しまでを追います。
- 上巻28–29頁と下巻36–37頁は、同じ状態表でAttentionとDeltaNetの更新・巻き戻しを説明します。
- [執筆計画](PLAN.md) に、読者の前提、各章の到達点、今後の具体化をまとめています。
