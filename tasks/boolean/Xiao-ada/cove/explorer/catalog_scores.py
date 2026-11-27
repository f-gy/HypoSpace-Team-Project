"""Complete-answer LM probabilities for a fixed, train-derived class catalog.

These are teacher-forced candidate evaluations, not free generation attempts.
All candidate and prefix token costs are published separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np

from .canonical_decode import completion_tokens
from .coverage_lora import LORA,dataset,model_load
from .posterior_bank import bank
from .prefix_cache import build_prefix,fresh_prefix
from .prompt_archive import recorded_prompt
from .tasks import write_json
from .upstream import ROOT

OUT=ROOT/'results/all80'


def score_prompt(model,tok,prompt,items,batch_size=8):
    import mlx.core as mx
    import mlx.nn as nn
    ids=tok.apply_chat_template([{'role':'user','content':prompt}],add_generation_prompt=True,tokenize=True)
    aliases=[{'answer':a,'key':h['key']} for h in items for a in h['aliases']]
    encoded=completion_tokens(tok,aliases)
    start=time.monotonic();prefix=build_prefix(model,ids);logprobs=[]
    for begin in range(0,len(encoded),batch_size):
        group=encoded[begin:begin+batch_size];b=len(group);length=max(len(h['tokens']) for h in group)
        inputs=np.zeros((b,length),dtype=np.int32);targets=np.zeros_like(inputs);mask=np.zeros_like(inputs,dtype=np.float32)
        for i,h in enumerate(group):
            n=len(h['tokens']);inputs[i,:n]=[ids[-1]]+h['tokens'][:-1]
            targets[i,:n]=h['tokens'];mask[i,:n]=1
        cache=fresh_prefix(prefix)
        for c in cache:
            keys,values,offset=c.state
            c.state=(mx.repeat(keys,b,axis=0),mx.repeat(values,b,axis=0),offset)
        logits=model(mx.array(inputs),cache=cache)
        ce=nn.losses.cross_entropy(logits,mx.array(targets),reduction='none').astype(mx.float32)*mx.array(mask)
        values=-ce.sum(axis=1);mx.eval(values)
        logprobs.extend(values.tolist())
    class_logq=[float(np.logaddexp(logprobs[i],logprobs[i+1])) for i in range(0,len(encoded),2)]
    return {'prompt':prompt,'class_keys':[h['key'] for h in items],'alias_log_probabilities':logprobs,
            'class_log_mass':class_logq,'seconds':time.monotonic()-start,
            'cost':{'candidate_sequences':len(encoded),'candidate_forward_batches':(len(encoded)+batch_size-1)//batch_size,
                    'unique_prefix_token_positions':len(ids)-1,'candidate_token_positions':sum(len(h['tokens']) for h in encoded),
                    'logical_conditioned_tokens':sum(len(ids)+len(h['tokens']) for h in encoded)},
            'interpretation':'Finite represented-alias class mass. No ground-truth consistency test or reference set enters scoring.'}


def run(split):
    import mlx.core as mx
    from mlx_lm.tuner.utils import linear_to_lora_layers
    d=dataset();items=bank();model,tok=model_load();model.freeze();linear_to_lora_layers(model,8,LORA)
    path=ROOT/'artifacts/preference_adapters/coverage_continuation/adapters.safetensors'
    model.load_weights(list(mx.load(str(path)).items()),strict=False);model.eval()
    for tid in d['splits'][split]:
        dest=OUT/'lm_scores'/f'{split}_{tid}.json'
        if dest.exists():continue
        record=score_prompt(model,tok,recorded_prompt(split,tid,101 if split=='validation' else 17),items)
        record.update(task_id=tid,split=split,adapter_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        write_json(dest,record);print(f'Scored {split} {tid}: {record["cost"]["candidate_sequences"]} candidates',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('split',choices=['validation','test']);run(parser.parse_args().split)
