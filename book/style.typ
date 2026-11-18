#let book-title = "Rustで学ぶLLM推論"
#let book-subtitle = "Qwenの1トークンを、数式からGPUカーネルまで"
#let source-revision = "8e187217eeaed0280b69e244f9a9bb6db76874f1"
#let source(path, body) = link("https://github.com/jl1nie/qwen-rust-inference/blob/" + source-revision + "/" + path, body)

#let accent = rgb("#164b61")

#let manuscript(body) = {
  set document(title: book-title + " ― " + book-subtitle)
  set page(paper: "a5", margin: (inside: 17mm, outside: 14mm, top: 16mm, bottom: 17mm), numbering: "1", number-align: center, binding: left)
  set text(font: "Noto Serif CJK JP", lang: "ja", size: 11pt)
  set par(leading: 0.8em, justify: true)
  set heading(numbering: "1.1")
  show heading: set text(font: "Noto Sans CJK JP", fill: accent)
  show raw: set text(font: ("DejaVu Sans Mono", "Noto Sans Mono CJK JP"), size: 8.5pt)
  show raw.where(block: true): it => block(
    width: 100%, fill: rgb("#f1f5f6"), inset: 8pt, radius: 2pt, it,
  )
  body
}

#let exercise(body) = block(
  width: 100%, inset: 9pt, fill: rgb("#eef5f6"), radius: 2pt,
  [#strong[演習] #body],
)

// 囲み記事：本文の流れから外した話題（結論と表だけを置き、詳細は出典へ）。
#let sidebar(title, body) = block(
  width: 100%, inset: 10pt, fill: rgb("#f6f4ee"), radius: 2pt,
  stroke: (left: 2.5pt + accent),
  [#text(font: "Noto Sans CJK JP", weight: "bold", fill: accent)[#title] #v(4pt) #body],
)
