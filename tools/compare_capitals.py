#!/usr/bin/env python3
"""Compare `lens-batch` over the prefecture capitals before and after training.

Inputs: two lens-batch outputs (JSON lines, with --generate) for the same
items, and a split file {"groups": {group: [prefecture, ...]}}. For every
group: free answers correct before -> after, and at the forced position the
rank at the last layer of the correct capital's and the shortcut's first
token. Prints JSON; scoring as tools/summarize_capitals.py.

Standard library only; never used by Rust inference.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_capitals import correct_answer  # noqa: E402


def load(path):
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = r
    return out


def view(r):
    t = {x["name"]: x for x in r["targets"]}
    return {
        "free_answer": r["generated"],
        "free_correct": correct_answer(r["id"], t["correct"]["text"], r["generated"] or ""),
        "chosen_token": r["chosen"][1],
        "correct_rank_last": t["correct"]["ranks"][-1],
        "shortcut_rank_last": t["shortcut"]["ranks"][-1],
        "forced_correct": r["chosen"][0] == t["correct"]["id"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("before", type=Path)
    ap.add_argument("after", type=Path)
    ap.add_argument("split", type=Path)
    a = ap.parse_args()
    before, after = load(a.before), load(a.after)
    split = json.loads(a.split.read_text(encoding="utf-8"))
    groups = {}
    for g, members in split["groups"].items():
        rows = []
        for p in members:
            if p not in before or p not in after:
                continue
            b, f = view(before[p]), view(after[p])
            rows.append({"prefecture": p, "before": b, "after": f})
        if not rows:
            continue
        n = len(rows)
        count = lambda side, key: sum(r[side][key] for r in rows)
        med = lambda side, key: statistics.median(r[side][key] for r in rows)
        groups[g] = {
            "count": n,
            "free_correct": [count("before", "free_correct"), count("after", "free_correct")],
            "forced_correct": [count("before", "forced_correct"), count("after", "forced_correct")],
            "median_correct_rank_last": [med("before", "correct_rank_last"), med("after", "correct_rank_last")],
            "median_shortcut_rank_last": [med("before", "shortcut_rank_last"), med("after", "shortcut_rank_last")],
            "prefectures": rows,
        }
    json.dump({"groups": groups}, sys.stdout, ensure_ascii=False, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
