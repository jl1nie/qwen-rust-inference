#let book-title = "Rustで学ぶLLM推論"
#let book-subtitle = "Qwenの1トークンを、数式からGPUカーネルまで"
#let source-revision = "8e187217eeaed0280b69e244f9a9bb6db76874f1"
#let source(path, body) = link("https://github.com/jl1nie/qwen-rust-inference/blob/" + source-revision + "/" + path, body)

#let accent = black

// 紙面は「桝形山通信 Vol.1」の技術同人誌用テンプレート（ver 2.12）に合わせる。
// 余白・柱・見出し罫・段落字下げをテンプレートから移し、文字の大きさは
// テンプレートのPDFと同じ本文9pt、コード7.2pt（0.8em）にする。
// 書体は印刷用にSIL OFLのものを使う。テンプレートのMeiryo UIは画面用なので、
// 本文はBIZ UDPゴシック、見出しはNoto Sans CJK JP（源ノ角ゴシック）、
// コードは半角と全角が1:2のUDEV Gothicにした。
// 柱はhydraパッケージの代わりにqueryで組み、オフラインで組版できるようにした。
#let running-head = context {
  let page = here().page()
  let chapters = query(heading.where(level: 1, outlined: true))
  if chapters.any(c => c.location().page() == page) { return }
  let before = chapters.filter(c => c.location().page() < page)
  if before.len() == 0 { return }
  let chapter = before.last()
  let number = counter(heading).at(chapter.location()).first()
  let label = text(size: 0.8em, underline[第#number;章　#chapter.body])
  if calc.odd(page) { align(right, label) } else { align(left, label) }
}

#let manuscript(body) = {
  set document(title: book-title + " ― " + book-subtitle)
  set page(
    paper: "a5",
    binding: left,
    margin: (top: 15mm, bottom: 15mm, inside: 20mm, outside: 12mm),
    numbering: "1",
    number-align: center,
    header: running-head,
  )
  set text(font: "BIZ UDPGothic", lang: "ja", size: 9pt)
  set par(leading: 0.8em, justify: true, first-line-indent: (amount: 1em, all: true))
  set heading(numbering: "1.1")
  show heading: set text(font: "Noto Sans CJK JP", fill: accent)
  show heading.where(level: 1): it => {
    set par(first-line-indent: 0em)
    let rule = (thickness: 6pt, paint: luma(100), cap: "butt")
    block(width: 100%, inset: (x: 0pt, y: 14pt), above: 0pt, below: 18pt,
      stroke: (top: rule, bottom: rule),
      align(center)[
        #if it.numbering != none [
          #text(size: 10pt, fill: luma(80))[第#counter(heading).display("1")章]
          #v(2pt)
        ]
        #text(size: 18pt, it.body)
      ])
  }
  show heading.where(level: 2): block.with(
    width: 100%,
    inset: 5pt,
    stroke: (
      left: (thickness: 6pt, paint: luma(140), cap: "butt"),
      bottom: (thickness: 0.8pt, paint: luma(170), cap: "butt"),
    ),
  )
  show heading.where(level: 3): block.with(
    width: 100%,
    inset: 4pt,
    stroke: (bottom: (thickness: 0.8pt, paint: luma(120), cap: "butt")),
  )
  show outline.entry.where(level: 1): it => {
    v(14pt, weak: true)
    strong(it)
  }
  // 数式中の日本語（"1回の秒数" など）も本文と同じ書体で組む。
  show math.equation: set text(font: ("New Computer Modern Math", "BIZ UDPGothic"))
  show raw: set text(font: "UDEV Gothic", size: 7.2pt)
  // 節ごとに改ページせず紙面に沿って流すので、コード・表・演習は頁をまたいで割らない。
  show raw.where(block: true): it => block(
    width: 100%, fill: luma(244), inset: 8pt, radius: 2pt, breakable: false, it,
  )
  show table: set par(first-line-indent: 0em, justify: false)
  show table: it => block(breakable: false, it)
  show figure.caption: set par(first-line-indent: 0em)
  body
}

#let exercise(body) = block(
  width: 100%, inset: 9pt, fill: luma(238), radius: 2pt, breakable: false,
  {
    set par(first-line-indent: 0em)
    [#strong[演習] #body]
  },
)

// 囲み記事：本文の流れから外した話題（結論と表だけを置き、詳細は出典へ）。
#let sidebar(title, body) = block(
  width: 100%, inset: 10pt, fill: luma(246), radius: 2pt, breakable: false,
  stroke: (left: (thickness: 2.5pt, paint: luma(120))),
  {
    set par(first-line-indent: 0em)
    [#text(font: "Noto Sans CJK JP", weight: "bold")[#title] #v(4pt) #body]
  },
)
