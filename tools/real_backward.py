#!/usr/bin/env python3
"""Gradients on the real weights with PyTorch autograd (ADR 0037, step 4); never used by Rust training.

The pinned upstream classes run the real model (BF16 weights widened to f32,
on the CPU). For a short tokenized text, the residual stream entering layer
FROM is computed without gradients; from there to the head, autograd gives
the gradient of the next-ID cross-entropy (summed over every position but
the last) toward that residual and toward the FFNs of the TRAIN layers.

The full FFN gradients are large (3.7M floats per matrix), so the output
keeps each matrix's Frobenius norm and a few whole rows. The output is a
local file (default models/qwen/backward-reference.json), compared by the
ignored test `train_real_stack_matches_pytorch` (tests/train_real.rs).
"""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_backward import block, build  # noqa: E402
from generate_hybrid import ROOT, SOURCE, reference  # noqa: E402

TEXT = ('日本の首都は東京です。京都はかつての都で、多くの寺や神社があります。'
        'The capital of France is Paris, and the capital of Italy is Rome. '
        'Wait, let me check the numbers again: 12 × 13 = 156.')
FROM, TRAIN = 15, (15, 19)
ROWS = {'gate': [0, 100, 1000, 2000, 3583], 'up': [0, 100, 1000, 2000, 3583], 'down': [0, 7, 100, 511, 1023]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', type=Path, default=ROOT / 'models/qwen')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    assert torch.__version__ == '2.6.0+cpu'
    torch.use_deterministic_algorithms(True)
    env, pins = reference()
    from safetensors.torch import load_file
    from tokenizers import Tokenizer
    config = json.loads((a.model_dir / 'config.json').read_text())['text_config']
    tensors = {k: v.float() for k, v in load_file(str(a.model_dir / 'model.safetensors-00001-of-00001.safetensors')).items()
               if k.startswith('model.language_model.') and '.mtp' not in k}
    cfg, layers, final, rotary = build(env, config, tensors)
    ids = Tokenizer.from_file(str(a.model_dir / 'tokenizer.json')).encode(TEXT, add_special_tokens=False).ids
    embed = tensors['model.language_model.embed_tokens.weight']
    with torch.no_grad():
        h = F.embedding(torch.tensor([ids]), embed)
        for layer in layers[:FROM]:
            h = block(layer, h, rotary)
    x = h.detach().clone().requires_grad_(True)
    h = x
    for layer in layers[FROM:]:
        h = block(layer, h, rotary)
    logits = F.linear(final(h), embed)
    loss = F.cross_entropy(logits[0, :-1], torch.tensor(ids[1:]), reduction='sum')
    weights = [w for i in TRAIN for w in (layers[i][2].gate_proj.weight, layers[i][2].up_proj.weight, layers[i][2].down_proj.weight)]
    g = torch.autograd.grad(loss, [x, *weights])
    ffn = []
    for j, i in enumerate(TRAIN):
        entry = dict(layer=i)
        for k, name in enumerate(['gate', 'up', 'down']):
            m = g[1 + 3 * j + k]
            entry[name] = dict(norm=m.norm().item(), rows={str(r): m[r].tolist() for r in ROWS[name]})
        ffn.append(entry)
    result = dict(text=TEXT, ids=ids, from_layer=FROM, loss=loss.item(), x=x[0].flatten().tolist(),
                  dx=g[0][0].flatten().tolist(), ffn=ffn,
                  metadata=dict(torch=torch.__version__, python=platform.python_version(),
                                model_revision=pins['model']['revision'],
                                source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                                generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                dtype='float32', weights_dtype='bfloat16', device='cpu'))
    out = a.output or a.model_dir / 'backward-reference.json'
    out.write_text(json.dumps(result, allow_nan=False) + '\n')
    print(out, len(ids), 'tokens, loss', loss.item())


if __name__ == '__main__':
    main()
