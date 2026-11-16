# Rustで学ぶLLM推論

副題は「Qwenの1トークンを、数式からGPUカーネルまで」。
技術書典向けのTypst原稿の雛形です。

## 組版

Typst CLIと日本語フォントの Noto Serif CJK JP、Noto Sans CJK JP を用意します。コードには DejaVu Sans Mono と Noto Sans Mono CJK JP を指定しています。

リポジトリのルートから実行します。

```sh
typst compile book/main.typ /tmp/qwen-book.pdf
typst watch book/main.typ /tmp/qwen-book.pdf
```

判型は仮のA5、左綴じです。本文・前付け・付録で68頁、表紙別を想定しています。

| 構成 | 予定頁 | 頁数 |
|---|---:|---:|
| 前付け | 1–2 | 2 |
| 第1章　Qwenが「東京」を出すまで | 3–8 | 6 |
| 第2章　Transformerのブロックを手で計算する | 9–18 | 10 |
| 第3章　Rustの参照実装を、GPU実装のものさしにする | 19–24 | 6 |
| 第4章　重みを使い回す ― Metalのprefill | 25–38 | 14 |
| 第5章　重みを読み切る ― Metalのdecode | 39–50 | 12 |
| 第6章　1回の検証で複数トークンを進める ― MTP | 51–62 | 12 |
| 付録　ROCmの実測・CUDAでの検証設計・再現手順 | 63–68 | 6 |

雛形PDFは章構成と試作を確認するための短い版です。第2章の試作は雛形の4–5頁、完成稿では16–17頁を予定します。第6章の巻き戻しでも同じ状態の図を使います。

- `main.typ` に全体の章構成と執筆メモがあります。
- `style.typ` で判型、書体、演習、実装の参照コミットを管理します。掲載コードや測定の参照先はこのコミットに揃えます。
- `chapters/state-spread.typ` にAttentionとDeltaNetの状態を比較する試作があります。
