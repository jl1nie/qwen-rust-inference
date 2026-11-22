#!/usr/bin/env python3
"""Count thinking loops: does the model close <think>, and how often does it say "Wait"?

Runs `<engine>-chat --thinking --json --no-cache` once per prompt (greedy),
for one condition (the base model, an --edit manifest, or an
--ffn-checkpoint), and prints one JSON object: per prompt, whether the model
chose </think> (ID 248069) within the limit, the thinking length in tokens,
the stop reason, the number of "Wait" in the thinking, and, for a reply cut
at the limit, whether it ends in a short repeating loop (a phrase repeated
with a period of at most 64 tokens over the last 200); and the totals.
Volume 1 ch. 5 counted the same way (spec/measurements/cpu-wsp-edit-ch5.json).

Standard library only; never used by Rust inference.
"""
import argparse
import json
import subprocess
import sys

END_THINK = 248069


def tail_period(ids, window=200, longest=64, share=0.95):
    """The shortest period p <= longest such that, over the last `window`
    IDs, at least `share` of them repeat the ID p places before; None when
    the end of the reply is not a short repeating loop."""
    if len(ids) < window + longest:
        return None
    tail = ids[-window - longest:]
    for p in range(1, longest + 1):
        same = sum(tail[i] == tail[i - p] for i in range(longest, len(tail)))
        if same >= share * window:
            return p
    return None


def survey(args, prompt):
    cmd = [args.binary, f"{args.engine}-chat" if args.engine != "cpu" else "chat"]
    if args.engine != "cpu":
        cmd += ["--library", args.library]
    cmd += ["--model-dir", args.model_dir, "--prompt", prompt, "--thinking",
            "--max-new-tokens", str(args.max_new_tokens), "--json", "--no-cache"]
    if args.edit:
        cmd += ["--edit", args.edit]
    if args.ffn_checkpoint:
        cmd += ["--ffn-checkpoint", args.ffn_checkpoint]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    r = json.loads(out.splitlines()[0])
    ids = r["generated"]
    closed = END_THINK in ids
    think = r["text"].split("</think>")[0]
    return {
        "prompt": prompt,
        "closed_think": closed,
        "think_tokens": ids.index(END_THINK) if closed else None,
        "stop": r["stop"],
        "count_Wait": think.count("Wait"),
        # Stuck in a short loop at the limit (a repeated phrase, not "Wait").
        "loop_period": None if closed else tail_period(ids),
        "answer": r["text"].split("</think>", 1)[1].strip() if closed else None,
        **({"text": r["text"], "generated": ids} if args.keep_text else {}),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--prompts", required=True, help="one prompt per line")
    p.add_argument("--binary", default="target/release/qwen-rust-inference")
    p.add_argument("--engine", default="rocm", choices=["cpu", "rocm", "cuda", "metal"])
    p.add_argument("--library", default="target/rocm/libqwen_rocm.so")
    p.add_argument("--model-dir", default="models/qwen")
    p.add_argument("--max-new-tokens", type=int, default=2000)
    p.add_argument("--edit")
    p.add_argument("--ffn-checkpoint")
    p.add_argument("--label", default="")
    p.add_argument("--keep-text", action="store_true",
                   help="also keep each reply's text and generated IDs (training data)")
    args = p.parse_args()
    prompts = [l.strip() for l in open(args.prompts, encoding="utf-8") if l.strip()]
    rows = []
    for q in prompts:
        rows.append(survey(args, q))
        print(json.dumps(rows[-1], ensure_ascii=False), file=sys.stderr)
    print(json.dumps({
        "label": args.label,
        "engine": args.engine,
        "edit": args.edit,
        "ffn_checkpoint": args.ffn_checkpoint,
        "max_new_tokens": args.max_new_tokens,
        "closed": sum(r["closed_think"] for r in rows),
        "count_Wait": sum(r["count_Wait"] for r in rows),
        "short_loops": sum(r["loop_period"] is not None for r in rows),
        "per_prompt": rows,
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
