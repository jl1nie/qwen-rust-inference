#!/usr/bin/env python3
"""Summarize `lens-batch` over examples/prefecture-capitals.jsonl (book 3).

Reads the JSON lines that `lens-batch --generate N` prints, scores the free
greedy answers, groups the prefectures, and writes a measurement-v1 JSON.

Scoring of the free answer (the reply to the plain prompt):
- the capital's name differs from the prefecture's: the answer must contain
  the capital without 市 (宇都宮), except 東京都, whose office is in 新宿区;
- the names are the same: the answer must contain the capital with 市
  (青森市), so that 青森県 does not count.

Groups, at the forced position (the token after "…所在地は**"):
- same-name: the correct and the shortcut (prefecture + 市) first tokens
  are the same, so knowing and the shortcut cannot be told apart;
- different, right / different, wrong: the model's next token is / is not
  the correct capital's first token.

Standard library only; never used by Rust inference.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path


def correct_answer(prefecture, capital, answer):
    if prefecture == "東京都":
        return "新宿" in answer
    stem = prefecture if prefecture == "北海道" else prefecture[:-1]
    if capital[:-1] == stem:
        return capital in answer
    return capital[:-1] in answer


def top_from(ranks, k=1):
    """First layer from which rank < k holds through the last layer, or None."""
    first = None
    for layer in range(len(ranks) - 1, -1, -1):
        if ranks[layer] < k:
            first = layer
        else:
            break
    return first


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results", type=Path, help="lens-batch output (JSON lines)")
    ap.add_argument("--model-revision", required=True)
    ap.add_argument("--device-info", required=True)
    ap.add_argument("--recorded", required=True, help="UTC date, YYYY-MM-DD")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    rows = [json.loads(l) for l in a.results.read_text(encoding="utf-8").splitlines() if l.strip()]
    items = []
    for r in rows:
        t = {x["name"]: x for x in r["targets"]}
        c, s, ch = t["correct"], t["shortcut"], t["chosen"]
        capital = c["text"]
        group = "same-name" if c["id"] == s["id"] else ("different, right" if ch["id"] == c["id"] else "different, wrong")
        items.append({
            "prefecture": r["id"],
            "capital": capital,
            "free_answer": r["generated"],
            "free_correct": None if r["generated"] is None else correct_answer(r["id"], capital, r["generated"]),
            "group": group,
            "chosen_token": r["chosen"][1],
            "correct_token": c["token"],
            "shortcut_token": s["token"],
            "correct_rank_last": c["ranks"][-1],
            "correct_top1_from": top_from(c["ranks"]),
            "correct_top10_from": top_from(c["ranks"], 10),
            "shortcut_rank_last": s["ranks"][-1],
            "shortcut_top1_from": top_from(s["ranks"]),
            "correct_ranks": c["ranks"],
            "shortcut_ranks": s["ranks"],
        })
    groups = {}
    for g in ["same-name", "different, right", "different, wrong"]:
        members = [i for i in items if i["group"] == g]
        if not members:
            continue
        layers = len(members[0]["correct_ranks"])
        med = lambda key: [statistics.median(i[key][l] for i in members) for l in range(layers)]
        groups[g] = {
            "count": len(members),
            "prefectures": [i["prefecture"] for i in members],
            "median_correct_rank_by_layer": med("correct_ranks"),
            "median_shortcut_rank_by_layer": med("shortcut_ranks"),
        }
    scored = [i for i in items if i["free_correct"] is not None]
    out = {
        "schema": "qwen-rust-inference/measurement-v1",
        "status": "passed",
        "recorded_utc": a.recorded,
        "device_info": a.device_info,
        "model_revision": a.model_revision,
        "note": a.note,
        "free_answers": {
            "correct": sum(i["free_correct"] for i in scored),
            "total": len(scored),
        },
        "groups": groups,
        "items": items,
    }
    json.dump(out, sys.stdout, ensure_ascii=False, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
