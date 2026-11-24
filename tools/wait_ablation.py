#!/usr/bin/env python3
"""Ablate parts of a trained layers-15/19 FFN change, for the Wait experiment.

The hand edit (examples/weight-space-programming.json) subtracts, in layers
15 and 19, one 1,024-number vector d from 64 columns of mlp.down_proj. A
train-stack run changes the whole FFN; its change of down_proj is
ΔW = W_trained − W_base. This tool writes new FFN checkpoints in which ΔW
is altered, so tools/thinking_survey.py can measure what each part does:

  B  ΔW without its component along d:            W_base + (I − d̂d̂ᵀ)ΔW
  C  only ΔW's component along d (gate, up = base): W_base + d̂d̂ᵀΔW
  D  ΔW without its largest-energy direction orthogonal to d
     (the top left singular vector of (I − d̂d̂ᵀ)ΔW)
  R  ΔW without a random unit direction u ⟂ d whose share of ‖ΔW‖² is
     within 0.005 of d's: a Gaussian mix of the top K left singular vectors
     of (I − d̂d̂ᵀ)ΔW, K and the mix redrawn until the share matches
     (numpy generator seeded with --seed)

gate and up keep their trained values except in C. Each output checkpoint is
the input file with only the replaced tensors' bytes rewritten (same header,
same metadata). Prints the shares as JSON.

  python3 tools/wait_ablation.py --run traces/wait-runs/kl-stack-1519-s1 \\
      --out traces/wait-runs/ablate/kl-stack-1519-s1 --conditions BCD
  python3 tools/wait_ablation.py --run ... --out .../kl-stack-1519-s1-R1 --conditions R --seed 1001

Outputs are <out>-<condition>-layer<L>.ckpt (for R: <out>-layer<L>.ckpt).
"""
import argparse
import json
import struct

import numpy as np


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def read(path, name):
    h, off = header(path)
    t = h[name]
    a, b = t["data_offsets"]
    with open(path, "rb") as f:
        f.seek(off + a)
        raw = f.read(b - a)
    if t["dtype"] == "F32":
        v = np.frombuffer(raw, np.float32)
    elif t["dtype"] == "BF16":
        v = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
    else:
        raise SystemExit(f"{name}: unsupported dtype {t['dtype']}")
    return v.reshape(t["shape"]).astype(np.float64)


def write(src, dst, replace):
    data = bytearray(open(src, "rb").read())
    h, off = header(src)
    for name, arr in replace.items():
        a, b = h[name]["data_offsets"]
        buf = np.ascontiguousarray(arr, dtype=np.float32).tobytes()
        if len(buf) != b - a:
            raise SystemExit(f"{name}: shape mismatch")
        data[off + a : off + b] = buf
    open(dst, "wb").write(data)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", required=True, help="checkpoint prefix: <run>-layer15.ckpt, <run>-layer19.ckpt")
    p.add_argument("--out", required=True, help="output prefix")
    p.add_argument("--conditions", default="BCD", help="any of B, C, D, or R alone")
    p.add_argument("--seed", type=int, default=0, help="for R")
    p.add_argument("--edit", default="examples/weight-space-programming.json")
    p.add_argument("--base", default="models/qwen/model.safetensors-00001-of-00001.safetensors")
    a = p.parse_args()
    if "R" in a.conditions and a.conditions != "R":
        raise SystemExit("R is written alone (its own --out and --seed)")
    edit = json.load(open(a.edit))
    rng = np.random.default_rng(a.seed)
    shares = {}
    for e in edit["edits"]:
        layer = e["layer"]
        src = f"{a.run}-layer{layer}.ckpt"
        pre = f"model.language_model.layers.{layer}.mlp."
        w_base = read(a.base, pre + "down_proj.weight")
        trained = read(src, "down")
        delta = trained - w_base
        total = (delta**2).sum()
        d = np.array(e["direction"], dtype=np.float64)
        dh = d / np.linalg.norm(d)
        along = np.outer(dh, dh @ delta)
        rest = delta - along
        u_svd, _, _ = np.linalg.svd(rest, full_matrices=False)
        s = {"share_along_d": float((along**2).sum() / total)}
        for c in a.conditions:
            if c == "B":
                write(src, f"{a.out}-B-layer{layer}.ckpt", {"down": w_base + rest})
            elif c == "C":
                write(src, f"{a.out}-C-layer{layer}.ckpt", {
                    "down": w_base + along,
                    "gate": read(a.base, pre + "gate_proj.weight"),
                    "up": read(a.base, pre + "up_proj.weight"),
                })
            elif c == "D":
                top = u_svd[:, 0]
                s["share_top_orthogonal"] = float(np.linalg.norm(top @ delta) ** 2 / total)
                write(src, f"{a.out}-D-layer{layer}.ckpt", {"down": trained - np.outer(top, top @ delta)})
            elif c == "R":
                target = s["share_along_d"]
                for tries in range(1_000_000):
                    k = int(rng.integers(2, 64))
                    u = u_svd[:, :k] @ rng.standard_normal(k)
                    u /= np.linalg.norm(u)
                    share = float(np.linalg.norm(u @ delta) ** 2 / total)
                    if abs(share - target) < 0.005:
                        break
                else:
                    raise SystemExit("no matching random direction")
                s.update({"share_random": share, "k": k, "cos_with_d": float(u @ dh)})
                write(src, f"{a.out}-layer{layer}.ckpt", {"down": trained - np.outer(u, u @ delta)})
            else:
                raise SystemExit(f"unknown condition {c}")
        shares[str(layer)] = s
    print(json.dumps(shares, indent=1))


if __name__ == "__main__":
    main()
