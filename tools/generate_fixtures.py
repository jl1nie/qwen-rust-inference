#!/usr/bin/env python3
"""Offline comparison only. Execute the two pinned upstream classes, not Rust code."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import platform
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    assert torch.__version__ == '2.6.0+cpu', torch.__version__
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    pins = json.loads((ROOT / 'spec/references.lock.json').read_text())
    source = ROOT / 'spec/upstream/modeling_qwen3_5.py'
    assert digest(source) == pins['local_sha256']['spec/upstream/modeling_qwen3_5.py']
    tree = ast.parse(source.read_text())
    classes = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in ('Qwen3_5MLP', 'Qwen3_5RMSNorm'):
            node.decorator_list = []  # Disable optional fused kernel dispatch; run source math.
            classes.append(node)
    assert len(classes) == 2
    env = dict(torch=torch, nn=nn, Qwen3_5Config=object, ACT2FN={'silu': nn.functional.silu})
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(source), 'exec'), env)
    state = 42
    def values(n):
        nonlocal state
        result = []
        for _ in range(n):
            state = (state * 1664525 + 1013904223) & 0xffffffff
            result.append(((state >> 24) - 128) / 256)
        return torch.tensor(result, dtype=torch.float32)
    gate, up, down = values(128).reshape(16, 8), values(128).reshape(16, 8), values(128).reshape(8, 16)
    x = values(8)
    model = env['Qwen3_5MLP'](SimpleNamespace(hidden_size=8, hidden_act='silu'), 16)
    with torch.no_grad():
        model.gate_proj.weight.copy_(gate)
        model.up_proj.weight.copy_(up)
        model.down_proj.weight.copy_(down)
        g, u = model.gate_proj(x), model.up_proj(x)
        activated = model.act_fn(g)
        product = activated * u
        edited = product.clone()
        edited[3] = 0
        norm = env['Qwen3_5RMSNorm'](8, 1e-6)
        norm.weight.copy_(torch.tensor([0., .1, -.2, .3, -.4, .5, -.6, .7]))
        data = dict(
            x=x.tolist(), gate_weights=gate.flatten().tolist(), up_weights=up.flatten().tolist(),
            down_weights=down.flatten().tolist(), gate=g.tolist(), up=u.tolist(),
            activated_gate=activated.tolist(), product=product.tolist(),
            output=model(x).tolist(), edited_output=model.down_proj(edited).tolist(),
            norm_weight=norm.weight.tolist(), norm=norm(x).tolist(),
            direct_norm=(x * torch.rsqrt(x.square().mean()+1e-6) * norm.weight).tolist(),
            softmax=torch.softmax(x, dim=-1).tolist(), residual=(x+model(x)).tolist(),
            matmul=(gate @ up.T).flatten().tolist(),
        )
    result = dict(metadata=dict(
        schema_version=1, generator='tools/generate_fixtures.py', generator_sha256=digest(Path(__file__)),
        python=platform.python_version(), torch=torch.__version__, device='cpu', dtype='float32',
        seed=42, rng='LCG32: a=1664525,c=1013904223; ((state>>24)-128)/256',
        source_revision=pins['transformers']['revision'], source_sha256=digest(source),
        conditions='single thread, no_grad, deterministic algorithms, unfused source classes via AST',
        atol=2e-6, rtol=2e-5,
    ), data=data)
    target = ROOT / 'tests/fixtures/phase1.json'
    target.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    (ROOT / 'tests/fixtures/SHA256SUMS').write_text(f'{digest(target)}  phase1.json\n')
    print(target)

if __name__ == '__main__':
    main()
