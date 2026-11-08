#!/usr/bin/env python3
"""Comparison fixture generator; never used by Rust inference.

Execute the pinned upstream class/function bodies after stripping optional kernel
wrappers and type annotations. No Rust arithmetic is used to make expected values.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import platform
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'spec/upstream/modeling_qwen3_5.py'
NAMES = {'Qwen3_5MLP', 'Qwen3_5RMSNorm', 'Qwen3_5RMSNormGated',
         'Qwen3_5Attention', 'Qwen3_5GatedDeltaNet', 'Qwen3_5TextRotaryEmbedding',
         'apply_mask_to_padding_states', 'causal_conv1d_fn', 'causal_conv1d_update',
         'l2norm', 'torch_recurrent_gated_delta_rule', 'torch_chunk_gated_delta_rule',
         'rotate_half', 'apply_rotary_pos_emb', 'repeat_kv', 'eager_attention_forward'}
class Clean(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        node.decorator_list = [d for d in node.decorator_list if isinstance(d, ast.Name) and d.id == 'staticmethod']
        node.returns = None
        for a in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
            a.annotation = None
        if node.args.vararg: node.args.vararg.annotation = None
        if node.args.kwarg: node.args.kwarg.annotation = None
        return self.generic_visit(node)
    def visit_ClassDef(self, node):
        node.decorator_list = []
        return self.generic_visit(node)

def reference():
    pins = json.loads((ROOT/'spec/references.lock.json').read_text())
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == pins['local_sha256']['spec/upstream/modeling_qwen3_5.py']
    tree = ast.parse(SOURCE.read_text())
    nodes = [Clean().visit(n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in NAMES]
    assert len(nodes) == len(NAMES)
    env = dict(torch=torch, nn=nn, F=F, math=math, ACT2FN={'silu':F.silu},
               is_torchdynamo_exporting=lambda:False, ALL_ATTENTION_FUNCTIONS=SimpleNamespace(get_interface=lambda name, fallback:fallback))
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(SOURCE), 'exec'), env)
    return env, pins

class Cache:
    """Only state storage, not replacement arithmetic: upstream updates the tensors."""
    def __init__(self): self.conv = {}; self.recurrent = {}; self.kv = {}
    def get_conv_state(self, layer, state_idx=0): return self.conv.get(layer)
    def get_recurrent_state(self, layer, state_idx=0): return self.recurrent.get(layer)
    def is_recording_past(self, layer): return False
    def update_conv_state(self, x, layer, conv_kernel_size):
        old = self.conv.get(layer)
        if old is not None: x = torch.cat([old[:, :, -(conv_kernel_size-1):],x],dim=-1)
        self.conv[layer] = F.pad(x,(max(0,conv_kernel_size-x.shape[-1]),0))[:, :, -conv_kernel_size:].clone()
        return x
    def update_recurrent_state(self, state, layer): self.recurrent[layer] = state.clone()
    def update(self, k, v, layer):
        if layer in self.kv:
            old_k, old_v = self.kv[layer]; k=torch.cat([old_k,k],dim=2); v=torch.cat([old_v,v],dim=2)
        self.kv[layer]=(k,v)
        return k,v

def run(env, config, tensors, ids, sequential=True, capture_states=True):
    cfg = SimpleNamespace(**config)
    cfg._attn_implementation = 'eager'
    cfg.hidden_act = 'silu'; cfg.attention_bias=False;cfg.attention_dropout=0.
    modules=[]
    for i,kind in enumerate(cfg.layer_types):
        mixer = env['Qwen3_5Attention' if kind=='full_attention' else 'Qwen3_5GatedDeltaNet'](cfg,i)
        mlp=env['Qwen3_5MLP'](cfg,cfg.intermediate_size)
        input_norm=env['Qwen3_5RMSNorm'](cfg.hidden_size,cfg.rms_norm_eps)
        post_norm=env['Qwen3_5RMSNorm'](cfg.hidden_size,cfg.rms_norm_eps)
        prefix=f'model.language_model.layers.{i}.'
        for name,module in [('self_attn' if kind=='full_attention' else 'linear_attn',mixer),('mlp',mlp),('input_layernorm',input_norm),('post_attention_layernorm',post_norm)]:
            module.load_state_dict({k[len(prefix+name+'.'):]:v for k,v in tensors.items() if k.startswith(prefix+name+'.')},strict=True,assign=True)
            module.eval()
        modules.append((mixer,mlp,input_norm,post_norm))
    final_norm=env['Qwen3_5RMSNorm'](cfg.hidden_size,cfg.rms_norm_eps)
    final_norm.weight.data.copy_(tensors['model.language_model.norm.weight'])
    rotary=env['Qwen3_5TextRotaryEmbedding'](cfg)
    cache=Cache(); result=dict(layers=[],logits=[],operations=[],states=[])
    groups = [[i] for i in ids] if sequential else [ids]
    position=0
    for group in groups:
        x=F.embedding(torch.tensor([group]),tensors['model.language_model.embed_tokens.weight'])
        embeddings=rotary(x,torch.arange(position,position+len(group)).reshape(1,-1))
        per_layer=[]; operations={}
        for i,(mixer,mlp,n1,n2) in enumerate(modules):
            h=n1(x); operations[f'{i}.input_norm']=h[0].tolist()
            if cfg.layer_types[i]=='full_attention':
                # Explicit causal additive mask for the source eager implementation.
                positions=torch.arange(position+len(group))
                mask=torch.where(positions[None,:]<=torch.arange(position,position+len(group))[:,None],0.,float('-inf'))[None,None]
                mixed=mixer(h,embeddings,mask,past_key_values=cache if sequential else None)[0]
            else: mixed=mixer(h,cache_params=cache if sequential else None)
            operations[f'{i}.mixer']=mixed[0].tolist()
            x=x+mixed; operations[f'{i}.after_mixer']=x[0].tolist()
            h=n2(x); operations[f'{i}.post_norm']=h[0].tolist()
            gate=mlp.gate_proj(h); up=mlp.up_proj(h); product=mlp.act_fn(gate)*up
            operations[f'{i}.ffn_product']=product[0].tolist()
            ffn=mlp(h); operations[f'{i}.ffn']=ffn[0].tolist()
            x=x+ffn; per_layer.append(x[0].tolist())
        logits=F.linear(final_norm(x),tensors['model.language_model.embed_tokens.weight'])
        for t in range(len(group)):
            result['layers'].append([layer[t] for layer in per_layer]);result['logits'].append(logits[0,t].tolist())
            result['operations'].append({k:v[t] for k,v in operations.items()})
            if capture_states:
                result['states'].append({str(i):{'conv':cache.conv[i][0,:,-(cfg.linear_conv_kernel_dim-1):].flatten().tolist(),'recurrent':s.flatten().tolist()} for i,s in cache.recurrent.items()})
        position+=len(group)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--model-dir',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    assert torch.__version__ == '2.6.0+cpu'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    env,pins=reference()
    if a.model_dir:
        from safetensors.torch import load_file
        config=json.loads((a.model_dir/'config.json').read_text())['text_config']
        tensors={k:v.float() for k,v in load_file(str(a.model_dir/'model.safetensors-00001-of-00001.safetensors')).items() if k.startswith('model.language_model.')}
        cases=[[16,10,16,28],[1,2,3],[100,200]]
        target=a.output or a.model_dir/'reference.json'
        result=dict(config=config,cases=[])
        for ids in cases:
            value=run(env,config,tensors,ids,capture_states=False)
            full=run(env,config,tensors,ids,False,capture_states=False)
            # Source chunk prefill and recurrent decode are independent paths.
            value['full_logits']=full['logits'];value['ids']=ids;result['cases'].append(value)
        from tokenizers import Tokenizer
        import jinja2
        tokenizer=Tokenizer.from_file(str(a.model_dir/'tokenizer.json'))
        template=jinja2.Environment().from_string((a.model_dir/'chat_template.jinja').read_text())
        result['tokenizer_cases']=[]
        for prompt in ['1+1は？','Hello, world!','  空白と\n改行🙂  ']:
            rendered=template.render(messages=[dict(role='user',content=prompt)],add_generation_prompt=True,enable_thinking=False)
            ids=tokenizer.encode(rendered,add_special_tokens=False).ids
            result['tokenizer_cases'].append(dict(prompt=prompt,rendered=rendered,ids=ids))
        result['model_revision']=pins['model']['revision']
    else:
        config=dict(hidden_size=8,intermediate_size=16,num_hidden_layers=4,vocab_size=19,
                    layer_types=['linear_attention']*3+['full_attention'],num_attention_heads=2,num_key_value_heads=1,head_dim=4,
                    linear_num_key_heads=2,linear_num_value_heads=2,linear_key_head_dim=4,linear_value_head_dim=4,
                    linear_conv_kernel_dim=3,rms_norm_eps=1e-6,max_position_embeddings=128,
                    rope_parameters=dict(rope_type='default',rope_theta=10000.,partial_rotary_factor=0.5,mrope_section=[1,0,0]),tie_word_embeddings=True)
        cfg=SimpleNamespace(**config,hidden_act='silu',attention_bias=False,attention_dropout=0.)
        tensors={};state=71
        def fill(shape,center=0.):
            nonlocal state
            n=math.prod(shape);v=[]
            for _ in range(n):
                state=(1664525*state+1013904223)&0xffffffff;v.append(((state>>24)-128)/512+center)
            return torch.tensor(v,dtype=torch.float32).reshape(shape)
        tensors['model.language_model.embed_tokens.weight']=fill([19,8])
        tensors['model.language_model.norm.weight']=fill([8])
        for i,kind in enumerate(config['layer_types']):
            modules=[('linear_attn' if i<3 else 'self_attn',env['Qwen3_5GatedDeltaNet' if i<3 else 'Qwen3_5Attention'](cfg,i)),('mlp',env['Qwen3_5MLP'](cfg,16)),('input_layernorm',env['Qwen3_5RMSNorm'](8)),('post_attention_layernorm',env['Qwen3_5RMSNorm'](8))]
            for name,module in modules:
                for key,value in module.state_dict().items():
                    tensors[f'model.language_model.layers.{i}.{name}.{key}']=fill(list(value.shape),1. if name=='linear_attn' and key=='norm.weight' else 0.)
        ids=[1,4,2,8,3]
        result=dict(config=config,tensors={k:dict(shape=list(v.shape),values=v.flatten().tolist()) for k,v in tensors.items()},ids=ids)
        result.update(run(env,config,tensors,ids));result['full_logits']=run(env,config,tensors,ids,False)['logits']
        target=a.output or ROOT/'tests/fixtures/hybrid.json'
    result['metadata']=dict(torch=torch.__version__,python=platform.python_version(),source_revision=pins['transformers']['revision'],source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),dtype='float32',weights_dtype='bfloat16' if a.model_dir else 'float32',device='cpu',threads=1,atol=3e-3 if a.model_dir else 2e-5,rtol=3e-3 if a.model_dir else 2e-4)
    target.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n');print(target)
    target.with_suffix('.sha256').write_text(hashlib.sha256(target.read_bytes()).hexdigest()+'\n')

if __name__=='__main__':
    with torch.no_grad(): main()
