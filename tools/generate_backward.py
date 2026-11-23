#!/usr/bin/env python3
"""Gradient fixtures for the training backward (ADR 0037, step 3); never used by Rust training.

Runs the pinned upstream classes (as tools/generate_hybrid.py does) on the
synthetic hybrid model of tests/fixtures/hybrid.json, and writes what
PyTorch autograd gives for fixed losses:

- per layer: the mixer alone, and the whole block, against an input x and a
  weighting c (loss = sum(c * output)): the gradient toward x, and toward the
  block's FFN weights;
- the stack: ids -> 4 layers -> final norm -> tied head, cross-entropy of the
  next ID at every position: the gradient toward the embedded input and
  toward the FFN weights of every layer.

The upstream DeltaNet runs its chunked rule (no cache), a different
algorithm from the token-by-token recurrence of the Rust backward.
Torch only produces expected numbers; no Rust arithmetic is used.
"""
import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_hybrid import ROOT, SOURCE, reference  # noqa: E402

FIXTURE = ROOT / 'tests/fixtures/hybrid.json'


def lcg(seed, n):
    """Deterministic values in [-1, 1), written into the fixture as is."""
    out = []
    for _ in range(n):
        seed = (1664525 * seed + 1013904223) & 0xffffffff
        out.append((seed >> 8) / (1 << 23) - 1.)
    return out


def build(env, config, tensors):
    cfg = SimpleNamespace(**config, hidden_act='silu', attention_bias=False, attention_dropout=0.)
    cfg._attn_implementation = 'eager'
    layers = []
    for i, kind in enumerate(cfg.layer_types):
        mixer = env['Qwen3_5Attention' if kind == 'full_attention' else 'Qwen3_5GatedDeltaNet'](cfg, i)
        mlp = env['Qwen3_5MLP'](cfg, cfg.intermediate_size)
        n1 = env['Qwen3_5RMSNorm'](cfg.hidden_size, cfg.rms_norm_eps)
        n2 = env['Qwen3_5RMSNorm'](cfg.hidden_size, cfg.rms_norm_eps)
        prefix = f'model.language_model.layers.{i}.'
        name = 'self_attn' if kind == 'full_attention' else 'linear_attn'
        for part, module in [(name, mixer), ('mlp', mlp), ('input_layernorm', n1), ('post_attention_layernorm', n2)]:
            p = prefix + part + '.'
            module.load_state_dict({k[len(p):]: v.clone() for k, v in tensors.items() if k.startswith(p)}, strict=True, assign=True)
            module.eval()
            for w in module.parameters():
                w.requires_grad_(part == 'mlp')
        layers.append((kind, mixer, mlp, n1, n2))
    final = env['Qwen3_5RMSNorm'](cfg.hidden_size, cfg.rms_norm_eps)
    final.weight.data.copy_(tensors['model.language_model.norm.weight'])
    final.weight.requires_grad_(False)
    rotary = env['Qwen3_5TextRotaryEmbedding'](cfg)
    return cfg, layers, final, rotary


def mix(kind, mixer, h, rotary):
    n = h.shape[1]
    if kind == 'full_attention':
        pos = torch.arange(n)
        mask = torch.where(pos[None, :] <= pos[:, None], 0., float('-inf'))[None, None]
        return mixer(h, rotary(h, pos.reshape(1, -1)), mask)[0]
    return mixer(h)


def block(layer, x, rotary):
    kind, mixer, mlp, n1, n2 = layer
    after = x + mix(kind, mixer, n1(x), rotary)
    return after + mlp(n2(after))


def grads(loss, tensors):
    return [g[0].flatten().tolist() if g.dim() == 3 else g.flatten().tolist()
            for g in torch.autograd.grad(loss, tensors)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=ROOT / 'tests/fixtures/hybrid-backward.json')
    a = p.parse_args()
    assert torch.__version__ == '2.6.0+cpu'
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    env, pins = reference()
    fixture = json.loads(FIXTURE.read_text())
    config = fixture['config']
    tensors = {k: torch.tensor(v['values'], dtype=torch.float32).reshape(v['shape']) for k, v in fixture['tensors'].items()}
    cfg, layers, final, rotary = build(env, config, tensors)
    hidden, n = cfg.hidden_size, 7
    x_values, c_values = lcg(5, n * hidden), lcg(6, n * hidden)
    result = dict(n=n, x=x_values, c=c_values, layers=[])
    for i, layer in enumerate(layers):
        kind, mixer, mlp, n1, n2 = layer
        c = torch.tensor(c_values).reshape(1, n, hidden)
        x = torch.tensor(x_values).reshape(1, n, hidden).requires_grad_(True)
        (mixer_dx,) = grads((c * mix(kind, mixer, x, rotary)).sum(), [x])
        x = torch.tensor(x_values).reshape(1, n, hidden).requires_grad_(True)
        ffn = [mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight]
        block_dx, dgate, dup, ddown = grads((c * block(layer, x, rotary)).sum(), [x, *ffn])
        result['layers'].append(dict(layer=i, kind=kind, mixer_dx=mixer_dx, block_dx=block_dx,
                                     dgate=dgate, dup=dup, ddown=ddown))
    ids = fixture['ids'] + [7, 0, 5, 11]
    embed = tensors['model.language_model.embed_tokens.weight']
    x0 = F.embedding(torch.tensor([ids]), embed).requires_grad_(True)
    h = x0
    for layer in layers:
        h = block(layer, h, rotary)
    logits = F.linear(final(h), embed)
    loss = F.cross_entropy(logits[0, :-1], torch.tensor(ids[1:]), reduction='sum')
    ffn = [w for _, _, mlp, _, _ in layers for w in (mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight)]
    g = grads(loss, [x0, *ffn])
    result['stack'] = dict(ids=ids, loss=loss.item(), dembedded=g[0],
                           ffn=[dict(layer=i, dgate=g[1 + 3 * i], dup=g[2 + 3 * i], ddown=g[3 + 3 * i]) for i in range(len(layers))])
    result['metadata'] = dict(torch=torch.__version__, python=platform.python_version(),
                              source_revision=pins['transformers']['revision'],
                              source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                              generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                              weights_sha256=hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
                              dtype='float32', device='cpu', threads=1)
    a.output.write_text(json.dumps(result, sort_keys=True, indent=1, allow_nan=False) + '\n')
    a.output.with_suffix('.sha256').write_text(hashlib.sha256(a.output.read_bytes()).hexdigest() + '\n')
    print(a.output)


if __name__ == '__main__':
    main()
