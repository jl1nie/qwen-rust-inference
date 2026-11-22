#!/usr/bin/env python3
"""Turn a `thinking_survey.py --keep-text` run into training pairs.

Keeps the replies that closed <think> (the teacher finished its thinking and
answered), and writes one pair per line for `train-text --pairs`:
{"prompt": ..., "response": thinking + </think> + reply, "thinking": true}.

Standard library only; never used by Rust inference.
"""
import argparse
import json


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("survey", help="thinking_survey.py --keep-text output")
    p.add_argument("--max-think-tokens", type=int, default=None,
                   help="drop replies whose thinking is longer than this")
    args = p.parse_args()
    d = json.load(open(args.survey, encoding="utf-8"))
    for r in d["per_prompt"]:
        if not r["closed_think"]:
            continue
        if args.max_think_tokens and r["think_tokens"] > args.max_think_tokens:
            continue
        print(json.dumps({"prompt": r["prompt"], "response": r["text"], "thinking": True},
                         ensure_ascii=False))


if __name__ == "__main__":
    main()
