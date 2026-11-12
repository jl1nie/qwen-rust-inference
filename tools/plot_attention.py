#!/usr/bin/env python3
"""Attention probabilities from a trace as an SVG heatmap (stdlib only).

Input: an NDJSON trace with `attention.probabilities` records for every
position of one attention layer (`--trace-full`), e.g. from the CPU chat:

  qwen-rust-inference chat --model-dir models/qwen --prompt '...' \
    --trace t.ndjson --trace-position 0,1,...,N --trace-layer 3 \
    --trace-max-bytes 400000000 --trace-full --save-state t.qrs
  python3 tools/plot_attention.py t.ndjson --layer 3 --snapshot t.qrs \
    --tokenizer models/qwen/tokenizer.json -o attention.svg

Each panel is one query head: row = position doing the reading (query),
column = position read (key). Rows are causal, so the upper triangle is
empty. Token labels come from the snapshot header and tokenizer.json
(byte-level BPE decoded per token; a character split across tokens shows
as its pieces). Colour is sqrt(probability) so small weights stay visible.
"""
import argparse
import json
import struct
from html import escape


def byte_decoder():
    # GPT-2 byte-level mapping: printable bytes map to themselves, the rest
    # to code points from 256 upwards, in order.
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def labels(tokenizer_path, ids):
    t = json.load(open(tokenizer_path, encoding="utf-8"))
    vocab = {i: s for s, i in t["model"]["vocab"].items()}
    special = {a["id"]: a["content"] for a in t.get("added_tokens", [])}
    dec = byte_decoder()
    out = []
    for i in ids:
        if i in special:
            s = special[i]
            out.append(s.strip("<|>") if s.startswith("<|") else s)
            continue
        raw = bytes(dec[c] for c in vocab[i] if c in dec)
        s = raw.decode("utf-8", errors="replace").replace("�", "·")
        s = s.replace("\n", "↵").replace(" ", "␣")
        out.append(s or "·")
    return out


def snapshot_tokens(path):
    with open(path, "rb") as f:
        head = f.read(16)
        assert head[:8] == b"QWENRS01", "not a qwen-rust-inference snapshot"
        (n,) = struct.unpack("<Q", head[8:16])
        return json.loads(f.read(n))["live"]["tokens"]


def colour(p):
    # white -> deep blue, sqrt scale
    v = min(1.0, max(0.0, p)) ** 0.5
    r = round(255 + (24 - 255) * v)
    g = round(255 + (66 - 255) * v)
    b = round(255 + (160 - 255) * v)
    return f"#{r:02x}{g:02x}{b:02x}"


def main():
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a.add_argument("trace")
    a.add_argument("--layer", type=int, required=True)
    a.add_argument("--snapshot", help="snapshot whose tokens label the axes")
    a.add_argument("--tokenizer", default="models/qwen/tokenizer.json")
    a.add_argument("--heads", type=int, default=8, help="query heads")
    a.add_argument("--columns", type=int, default=4)
    a.add_argument("--cell", type=int, default=13)
    a.add_argument("--title", default=None)
    a.add_argument("--subtitle", default=None)
    a.add_argument(
        "--mark",
        action="append",
        default=[],
        metavar="HEAD:QUERY:KEY",
        help="outline one cell (repeatable)",
    )
    a.add_argument("-o", "--output", required=True)
    o = a.parse_args()

    rows = {}
    for line in open(o.trace, encoding="utf-8"):
        r = json.loads(line)
        if r["operation"] == "attention.probabilities" and r["layer"] == o.layer:
            # Keep the first computation of each position (pass1).
            rows.setdefault(r["token_position"], r["values"])
    assert rows, "no attention.probabilities records (use --trace-full)"
    n = max(rows) + 1
    assert sorted(rows) == list(range(n)), "trace every position 0..N-1"
    h = o.heads
    for p, v in rows.items():
        assert v is not None and len(v) == h * (p + 1), f"position {p}: shape"

    names = labels(o.tokenizer, snapshot_tokens(o.snapshot)[:n]) if o.snapshot else [str(i) for i in range(n)]
    cell, cols = o.cell, o.columns
    panel = n * cell
    left, top, gap, bottom = 120, 56, 22, 110
    width = left + cols * panel + (cols - 1) * gap + 20
    grid_rows = -(-h // cols)
    height = top + grid_rows * (panel + 24) + (grid_rows - 1) * gap + bottom
    s = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Helvetica, Arial, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
    ]
    title = o.title or f"Layer {o.layer} attention probabilities"
    s.append(f'<text x="{left}" y="28" font-size="18" font-weight="bold" fill="#111">{escape(title)}</text>')
    subtitle = o.subtitle or (
        "row = query position (reads), column = key position (read); "
        "one panel per head; colour = sqrt(probability)"
    )
    s.append(f'<text x="{left}" y="46" font-size="12" fill="#555">{escape(subtitle)}</text>')
    marks = [tuple(map(int, m.split(":"))) for m in o.mark]
    for head in range(h):
        gx = left + (head % cols) * (panel + gap)
        gy = top + (head // cols) * (panel + 24 + gap) + 18
        s.append(f'<text x="{gx}" y="{gy - 5}" font-size="12" fill="#333">head {head}</text>')
        s.append(f'<rect x="{gx}" y="{gy}" width="{panel}" height="{panel}" fill="#f6f6f6"/>')
        for q in range(n):
            v = rows[q]
            for k in range(q + 1):
                p = v[head * (q + 1) + k]
                s.append(
                    f'<rect x="{gx + k * cell}" y="{gy + q * cell}" width="{cell}" '
                    f'height="{cell}" fill="{colour(p)}"><title>{escape(names[q])} → '
                    f"{escape(names[k])}: {p:.3f}</title></rect>"
                )
        for mh, mq, mk in marks:
            if mh == head:
                v = rows[mq][head * (mq + 1) + mk]
                s.append(
                    f'<rect x="{gx + mk * cell - 1}" y="{gy + mq * cell - 1}" width="{cell + 2}" '
                    f'height="{cell + 2}" fill="none" stroke="#e8590c" stroke-width="2.5"/>'
                )
                s.append(
                    f'<text x="{gx + mk * cell + cell + 5}" y="{gy + mq * cell - 4}" font-size="11" '
                    f'fill="#e8590c" font-weight="bold">{escape(names[mq])} → {escape(names[mk])} '
                    f"{v:.2f}</text>"
                )
        if head % cols == 0:
            for q in range(n):
                s.append(
                    f'<text x="{gx - 6}" y="{gy + q * cell + cell - 3}" font-size="10" '
                    f'text-anchor="end" fill="#222">{escape(names[q])}</text>'
                )
        if head // cols == grid_rows - 1:
            for k in range(n):
                x = gx + k * cell + cell - 3
                y = gy + panel + 6
                s.append(
                    f'<text x="{x}" y="{y}" font-size="10" text-anchor="end" fill="#222" '
                    f'transform="rotate(-60 {x} {y})">{escape(names[k])}</text>'
                )
    s.append("</svg>")
    open(o.output, "w", encoding="utf-8").write("\n".join(s) + "\n")


if __name__ == "__main__":
    main()
