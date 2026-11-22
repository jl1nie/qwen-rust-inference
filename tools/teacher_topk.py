#!/usr/bin/env python3
"""Add a teacher's top-K distributions to train-text pairs (book 3, distillation).

For each pair {"prompt", "response", "score"?} of a pairs JSONL, the token
sequence the student reads is user_chat(prompt) + response + <|im_end|>
(src/text.rs, thinking off). The teacher shares the student's tokenizer, so
its llama-server tokenizes the same text to the same IDs. At each scored
position (the tokens of "score", or the whole response and the end of the
turn, as train-text scores them) the teacher reads the IDs up to that
position and returns its top K next IDs with their log-probabilities: a plain
softmax of the logits (temperature < 0), one request per position.

Writes the pairs with a "teacher" field {"name", "ids", "positions": [{"position",
"ids", "log_p"}]}; train-text checks that the IDs and positions are its own.
Prints per pair how much of the teacher's mass the top K covers and whether
the text's own next ID is in it. Standard library only.
"""
import argparse
import json
import math
import sys
import urllib.request
from pathlib import Path

END_OF_TURN = 248046


def user_chat(prompt):
    return f"<|im_start|>user\n{prompt.strip()}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def post(url, path, body):
    req = urllib.request.Request(
        url.rstrip("/") + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def tokenize(url, text):
    return post(url, "/tokenize", {"content": text, "add_special": False, "parse_special": True})["tokens"]


def scored_positions(url, pair, ids):
    """Positions whose next ID is scored, as train-text chooses them."""
    chat = user_chat(pair["prompt"])
    prompt_len = len(tokenize(url, chat))
    part = pair.get("score")
    if part is None:
        return list(range(prompt_len - 1, len(ids) - 1))
    response = pair["response"]
    if response.count(part) != 1:
        raise ValueError(f"{part!r} must occur once in {response!r}")
    start = response.index(part)
    a = tokenize(url, chat + response[:start])
    b = tokenize(url, chat + response[: start + len(part)])
    if ids[: len(a)] != a or ids[: len(b)] != b:
        raise ValueError(f"{part!r} does not fall on token boundaries")
    return list(range(len(a) - 1, len(b) - 1))


def top_k(url, prefix, k):
    r = post(url, "/completion", {
        "prompt": prefix,
        "n_predict": 1,
        "n_probs": k,
        "temperature": -1,
        "cache_prompt": True,
        "post_sampling_probs": False,
    })
    top = r["completion_probabilities"][0]["top_logprobs"]
    return [t["id"] for t in top], normalized([t["logprob"] for t in top])


def normalized(log_p):
    """The teacher's f32 softmax can sum slightly above 1 over the top K
    (by about 1e-6); then it is divided back to 1, leaving "other" empty."""
    mass = sum(math.exp(x) for x in log_p)
    if mass <= 1:
        return log_p
    shift = math.log(mass)
    return [min(x - shift, 0.0) for x in log_p]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pairs", type=Path)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--name", required=True, help="the teacher, its quantization and the engine version")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    pairs = [json.loads(l) for l in a.pairs.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = []
    for n, pair in enumerate(pairs, 1):
        ids = tokenize(a.url, user_chat(pair["prompt"]) + pair["response"]) + [END_OF_TURN]
        positions = []
        for p in scored_positions(a.url, pair, ids):
            top_ids, log_p = top_k(a.url, ids[: p + 1], a.top_k)
            positions.append({"position": p, "ids": top_ids, "log_p": log_p})
            mass = sum(math.exp(x) for x in log_p)
            hit = ids[p + 1] in top_ids
            print(f"pair {n} position {p}: top-{len(top_ids)} mass {mass:.4f}, next ID "
                  f"{'rank ' + str(top_ids.index(ids[p + 1])) if hit else 'not in the top K'}",
                  file=sys.stderr)
        out.append({**pair, "teacher": {"name": a.name, "ids": ids, "positions": positions}})
    a.out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out), encoding="utf-8")


if __name__ == "__main__":
    main()
