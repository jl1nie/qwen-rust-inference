#!/usr/bin/env python3
"""Opt-in long-context reference fixtures, using pinned upstream arithmetic only.

Binary records avoid millions of Python float objects / multi-GB JSON. ctypes
copies bytes from a live contiguous CPU tensor; Python is never an inference
runtime of the Rust program. No downloads happen in this script.
"""
import argparse
import ctypes
import hashlib
import json
import platform
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import load_file
from tokenizers import Tokenizer
import jinja2
from generate_hybrid import reference, Cache, ROOT, SOURCE


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_tensor(file, tensor):
    assert tensor.device.type == 'cpu' and tensor.dtype == torch.float32
    tensor = tensor.contiguous()
    # data_ptr points at the view's first element, unlike untyped_storage which
    # can include unrelated positions of its parent tensor. Keep tensor alive.
    file.write(ctypes.string_at(tensor.data_ptr(), tensor.numel() * 4))


class Session:
    """State adapter + module wiring; every neural operation is upstream code."""
    def __init__(self, env, config, tensors):
        self.env = env
        self.config = config
        self.tensors = tensors
        cfg = SimpleNamespace(**config)
        cfg._attn_implementation = 'eager'
        self.modules = []
        for i, kind in enumerate(cfg.layer_types):
            mixer = env['Qwen3_5Attention' if kind == 'full_attention' else 'Qwen3_5GatedDeltaNet'](cfg, i)
            mlp = env['Qwen3_5MLP'](cfg, cfg.intermediate_size)
            n1 = env['Qwen3_5RMSNorm'](cfg.hidden_size, cfg.rms_norm_eps)
            n2 = env['Qwen3_5RMSNorm'](cfg.hidden_size, cfg.rms_norm_eps)
            prefix = f'model.language_model.layers.{i}.'
            for name, module in [('self_attn' if kind == 'full_attention' else 'linear_attn', mixer), ('mlp', mlp), ('input_layernorm', n1), ('post_attention_layernorm', n2)]:
                key = prefix + name + '.'
                module.load_state_dict({k[len(key):]: v for k, v in tensors.items() if k.startswith(key)}, strict=True, assign=True)
                module.eval()
            self.modules.append((mixer, mlp, n1, n2))
        self.norm = env['Qwen3_5RMSNorm'](cfg.hidden_size, cfg.rms_norm_eps)
        self.norm.weight.data.copy_(tensors['model.language_model.norm.weight'])
        self.rotary = env['Qwen3_5TextRotaryEmbedding'](cfg)
        self.reset()

    def reset(self):
        self.cache = Cache()
        self.position = 0

    def forward(self, ids):
        x = F.embedding(torch.tensor([ids]), self.tensors['model.language_model.embed_tokens.weight'])
        positions = torch.arange(self.position, self.position + len(ids)).reshape(1, -1)
        rotary = self.rotary(x, positions)
        keys = torch.arange(self.position + len(ids))
        mask = torch.where(keys[None, :] <= positions[0, :, None], 0., float('-inf'))[None, None]
        layers = []
        for i, (mixer, mlp, n1, n2) in enumerate(self.modules):
            h = n1(x)
            if self.config['layer_types'][i] == 'full_attention':
                mixed = mixer(h, rotary, mask, past_key_values=self.cache)[0]
            else:
                mixed = mixer(h, cache_params=self.cache)
            x = x + mixed
            x = x + mlp(n2(x))
            layers.append(x)
        logits = F.linear(self.norm(x), self.tensors['model.language_model.embed_tokens.weight'])
        self.position += len(ids)
        return layers, logits

    def snapshot(self, path):
        with path.open('wb') as f:
            for i, kind in enumerate(self.config['layer_types']):
                if kind == 'linear_attention':
                    n = self.config['linear_conv_kernel_dim'] - 1
                    write_tensor(f, self.cache.conv[i][0, :, -n:])
                    write_tensor(f, self.cache.recurrent[i][0])
                else:
                    k, v = self.cache.kv[i]
                    write_tensor(f, k.transpose(1, 2)[0])
                    write_tensor(f, v.transpose(1, 2)[0])


def write_records(file, layers, logits):
    for t in range(logits.shape[1]):
        for layer in layers:
            write_tensor(file, layer[0, t])
        write_tensor(file, logits[0, t])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, required=True)
    args = parser.parse_args()
    assert torch.__version__ == '2.6.0+cpu'
    assert sys.byteorder == 'little'
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    env, pins = reference()
    model_dir = args.model_dir
    assert (model_dir / 'REVISION').read_text().strip() == pins['model']['revision']
    weight_path = model_dir / 'model.safetensors-00001-of-00001.safetensors'
    weight_sha = sha(weight_path)
    assert weight_sha == '04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696'
    config_path = model_dir / 'config.json'
    assert sha(config_path) == pins['local_sha256']['spec/upstream/config.json']
    config = json.loads(config_path.read_text())['text_config']
    tensors = {k: v.float() for k, v in load_file(str(weight_path)).items() if k.startswith('model.language_model.')}
    session = Session(env, config, tensors)
    tokenizer = Tokenizer.from_file(str(model_dir / 'tokenizer.json'))
    template = jinja2.Environment().from_string((model_dir / 'chat_template.jinja').read_text())
    out = model_dir / 'extended'
    out.mkdir(exist_ok=True)
    manifest = dict(schema_version=1, config=config, model_revision=pins['model']['revision'], cases=[], generation=[], files={},
                    metadata=dict(torch=torch.__version__, python=platform.python_version(), device='cpu', dtype='float32', weights_dtype='bfloat16', threads=1,
                                  source_revision=pins['transformers']['revision'], source_sha256=sha(SOURCE), generator_sha256=sha(Path(__file__)), helper_sha256=sha(ROOT / 'tools/generate_hybrid.py'),
                                  checkpoint_sha256=weight_sha, atol=3e-3, rtol=3e-3,
                                  format='little-endian f32; per position: layer 0..23 outputs, then all vocabulary logits',
                                  conditions='upstream chunk prefill, cached recurrent decode; no_grad, deterministic; no logits interventions'))
    def register(path):
        manifest['files'][path.name] = dict(bytes=path.stat().st_size, sha256=sha(path))
        return path.name

    texts = [
        '数学の確認です。1+1は2、2+2は4です。理由を短く説明します。\n',
        'Learning attention: a query reads past keys and values. The residual carries the original input.\n',
        '第1節: Rustで数値を確認する。 Chapter 2: fixed-size recurrent memory.\n行列、正規化、位置、状態を順番に観察します。\n',
    ]
    continuation = tokenizer.encode(' したがって、答えを確認します。', add_special_tokens=False).ids[:8]
    assert len(continuation) == 8
    for length, text in zip([32, 128, 512], texts):
        source_ids = tokenizer.encode(text, add_special_tokens=False).ids
        ids = (source_ids * ((length + len(source_ids) - 1) // len(source_ids)))[:length]
        session.reset()
        path = out / f'context-{length}.f32'
        print(f'reference prefill length={length}', flush=True)
        with path.open('wb') as f:
            layers, logits = session.forward(ids)
            write_records(f, layers, logits)
            del layers, logits
            state_prefill = out / f'context-{length}-prefill-state.f32'
            session.snapshot(state_prefill)
            for token in continuation:
                layers, logits = session.forward([token])
                write_records(f, layers, logits)
                del layers, logits
        state_decode = out / f'context-{length}-decode-state.f32'
        session.snapshot(state_decode)
        manifest['cases'].append(dict(name=f'context-{length}', text=text, ids=ids, continuation=continuation,
                                      prefill_length=length, records=length + len(continuation), values=register(path),
                                      prefill_state=register(state_prefill), decode_state=register(state_decode)))
        print(f'reference saved length={length}', flush=True)

    eos = [248044, 248046]
    for index, prompt in enumerate(['数字だけで答えてください。1+1は？', '英語で一語だけ答えてください。猫は英語で？']):
        session.reset()
        rendered = template.render(messages=[dict(role='user', content=prompt)], add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.encode(rendered, add_special_tokens=False).ids
        layers, logits = session.forward(ids)
        generated = []
        path = out / f'greedy-{index}.f32'
        with path.open('wb') as f:
            for _ in range(64):
                # Only the last prompt position predicts the first generated ID.
                write_records(f, [v[:, -1:] for v in layers], logits[:, -1:])
                token = logits[0, -1].argmax().item()
                generated.append(token)
                del layers, logits
                if token in eos:
                    break
                layers, logits = session.forward([token])
        assert generated[-1] in eos, f'no natural EOS: {prompt!r}: {generated}'
        manifest['generation'].append(dict(prompt=prompt, rendered=rendered, ids=ids, generated=generated, text=tokenizer.decode(generated, skip_special_tokens=False),
                                           records=len(generated), max_new_tokens=64, eos=eos, stop='Eos', values=register(path)))
        print(f'natural EOS prompt={index} generated={len(generated)} ids={generated}', flush=True)
    target = out / 'manifest.json'
    target.write_text(json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + '\n')
    target.with_suffix('.sha256').write_text(sha(target) + '\n')
    print(target, flush=True)


if __name__ == '__main__':
    with torch.no_grad():
        main()
