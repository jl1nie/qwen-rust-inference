#let book-title = "Rustで学ぶLLM推論"
#let book-subtitle = "Qwenの1トークンを、数式からGPUカーネルまで"
#let source-revision = "0a9cf1e1f0e14d0dffc28f8e0cebbf9e2d7f45b5"
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
