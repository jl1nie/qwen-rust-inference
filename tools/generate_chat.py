#!/usr/bin/env python3
"""Offline fixture generation for the pinned text-only multi-turn Jinja branch."""
import hashlib
import json
from pathlib import Path
import jinja2
from tokenizers import Tokenizer
import tokenizers
ROOT=Path(__file__).resolve().parents[1]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    template_path=ROOT/'spec/upstream/chat_template.jinja'
    pins=json.loads((ROOT/'spec/references.lock.json').read_text())
    assert sha(template_path)==pins['local_sha256']['spec/upstream/chat_template.jinja']
    tokenizer=Tokenizer.from_file(str(ROOT/'models/qwen/tokenizer.json'))
    template=jinja2.Environment().from_string(template_path.read_text())
    cases=[
        ([dict(role='user',content='1+1は？')],False,True),
        ([dict(role='system',content=' 簡潔に答える。 '),dict(role='user',content='1+1は？'),dict(role='assistant',content='1+1=2'),dict(role='user',content='その数に1を足すと？')],False,True),
        ([dict(role='user',content='A'),dict(role='assistant',content='<think>\nreason\n</think>\n\n answer'),dict(role='user',content='B'),dict(role='assistant',content='done'),dict(role='user',content='C')],True,True),
        ([dict(role='user',content='A'),dict(role='assistant',content='answer',reasoning_content='private reasoning'),dict(role='user',content='B')],False,True),
        ([dict(role='user',content='A'),dict(role='assistant',content='<think>\nreason\n</think>\n  answer')],False,False),
        ([dict(role='system',content=''),dict(role='user',content='  空白\n改行🙂  '),dict(role='assistant',content='\nYes.\n'),dict(role='user',content='次')],False,False),
    ]
    out=dict(metadata=dict(model_revision=pins['model']['revision'],template_sha256=sha(template_path),generator_sha256=sha(Path(__file__)),tokenizers=tokenizers.__version__,jinja2=jinja2.__version__),cases=[])
    for messages,thinking,add_generation in cases:
        rendered=template.render(messages=messages,enable_thinking=thinking,add_generation_prompt=add_generation)
        out['cases'].append(dict(messages=messages,thinking=thinking,add_generation_prompt=add_generation,rendered=rendered,ids=tokenizer.encode(rendered,add_special_tokens=False).ids))
    target=ROOT/'tests/fixtures/chat-template.json';target.write_text(json.dumps(out,sort_keys=True,indent=2,ensure_ascii=False)+'\n')
    target.with_suffix('.sha256').write_text(sha(target)+'\n');print(target)
if __name__=='__main__': main()
