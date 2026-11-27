"""Generate only IID steps beyond the immutable 16-step public prefixes."""
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.utils import linear_to_lora_layers
from explorer.all80_experiment import source_iid
from explorer.coverage_lora import LORA,dataset,model_load
from explorer.experiment import assess
from explorer.prefix_cache import build_prefix,fresh_prefix
from explorer.prompt_archive import recorded_prompt
from explorer.tasks import public_task,write_json
from explorer.upstream import evaluator

d=dataset();tasks={t['id']:t for t in d['tasks']};ev=evaluator(d['metadata'])
model,tok=model_load();model.freeze();linear_to_lora_layers(model,8,LORA);model.eval()
zero={n:mx.zeros_like(v) for n,v in tree_flatten(model.trainable_parameters())}
adapt=mx.load(str(ROOT/'artifacts/preference_adapters/coverage_continuation/adapters.safetensors'))
original=source_iid('test');out=ROOT/'results/all80';path=out/'iid_extras.jsonl'
new_calls=0
for method,weights in [('base_iid',zero),('frozen_iid',adapt)]:
 model.load_weights(list(weights.items()),strict=False);mx.eval(model.parameters())
 for tid in d['splits']['test']:
  budget=len(tasks[tid]['reference_keys'])
  if budget<=16:continue
  task=public_task(tasks[tid])
  for seed in [17,29,43]:
   prompt=recorded_prompt('test',tid,seed)
   ids=tok.apply_chat_template([{'role':'user','content':prompt}],add_generation_prompt=True,tokenize=True)
   prefix=build_prefix(model,ids)
   history=[original[(method,tid,seed,t)]['outcome'] for t in range(16)]
   for step in range(16,budget):
    key=(method,tid,seed,step)
    if key in original:history.append(original[key]['outcome']);continue
    request_seed=random.Random(int.from_bytes(hashlib.sha256(f'{seed}:{step}:{tid}'.encode()).digest()[:8],'big')).getrandbits(31)
    mx.random.seed(request_seed);start=time.monotonic();parts=[]
    for last in stream_generate(model,tok,[ids[-1]],max_tokens=128,
                                sampler=make_sampler(temp=1.,top_p=1.,top_k=0,min_p=0.),
                                prompt_cache=fresh_prefix(prefix)):
     parts.append(last.text)
    text=''.join(parts);h=assess(ev,d['metadata'],task,text,history)
    row={'method':method,'task_id':tid,'seed':seed,'step':step,'raw_response':text,'outcome':h,
         'prompt':prompt,'request_seed':request_seed,'seconds':time.monotonic()-start,'error':None,
         'usage':{'prompt_tokens':len(ids),'completion_tokens':last.generation_tokens,'total_tokens':len(ids)+last.generation_tokens}}
    with path.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    original[key]=row;history.append(h);new_calls+=1
 print(method,'additional calls',new_calls,flush=True)
write_json(out/'iid_extra_generation_audit.json',{'new_calls_this_invocation':new_calls,'expected_total_extra_calls':42,
                                               'original_16_step_outputs_replayed_without_repair':True,
                                               'all_new_outputs_retained':True})
