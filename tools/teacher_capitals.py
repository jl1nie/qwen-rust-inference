#!/usr/bin/env python3
"""Ask a teacher model for the prefecture capitals and keep its right answers (book 3).

Sends each item's prompt (examples/prefecture-capitals*.jsonl) to an
OpenAI-compatible chat server (llama.cpp's llama-server), greedy and with
thinking off, and checks the reply against the item's correct capital with
the scoring of tools/summarize_capitals.py. Right replies become train-text
pairs {"prompt", "response", "score"}, where "score" is the capital as the
reply writes it (with 市 if it occurs once, else without). Wrong replies are
dropped and listed in the report.

Writes the pairs (JSON lines) to --out and the report (every reply, kept or
not) as JSON to stdout. Standard library only; never used by Rust inference.
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_capitals import correct_answer  # noqa: E402


def ask(url, model, prompt, max_tokens):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        reply = json.load(r)
    choice = reply["choices"][0]
    return choice["message"]["content"], choice.get("finish_reason")


def span(prefecture, capital, response):
    """The part of the response to score, or None."""
    if prefecture == "東京都":
        candidates = ["新宿区", "新宿"]
    else:
        candidates = [capital, capital[:-1]]
    for c in candidates:
        if response.count(c) == 1:
            return c
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("items", type=Path)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="teacher")
    ap.add_argument("--ids", help="comma-separated item IDs to ask (default: all)")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--out", type=Path, required=True, help="pairs JSONL to write")
    a = ap.parse_args()
    items = [json.loads(l) for l in a.items.read_text(encoding="utf-8").splitlines() if l.strip()]
    if a.ids:
        wanted = a.ids.split(",")
        by_id = {i["id"]: i for i in items}
        items = [by_id[w] for w in wanted]
    pairs, report = [], []
    for item in items:
        p, capital = item["id"], item["targets"]["correct"]
        response, finish = ask(a.url, a.model, item["prompt"], a.max_tokens)
        right = correct_answer(p, capital, response)
        part = span(p, capital, response) if right else None
        kept = right and part is not None and finish == "stop"
        report.append({
            "id": p,
            "prompt": item["prompt"],
            "response": response,
            "finish_reason": finish,
            "correct": right,
            "score": part,
            "kept": kept,
        })
        if kept:
            pairs.append({"prompt": item["prompt"], "response": response, "score": part})
        print(f"{p}: {'kept' if kept else 'dropped'}: {response!r}", file=sys.stderr)
    a.out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in pairs), encoding="utf-8")
    json.dump({"asked": len(report), "kept": len(pairs), "replies": report}, sys.stdout, ensure_ascii=False, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
