"""Numerically compare cached batched class scoring with full teacher forcing."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.tuner.utils import linear_to_lora_layers
from explorer.canonical_decode import completion_tokens
from explorer.coverage_lora import LORA,dataset,model_load,batch
from explorer.posterior_bank import bank
from explorer.tasks import write_json

d=dataset();items=bank();model,tok=model_load();model.freeze();linear_to_lora_layers(model,8,LORA)
model.load_weights(list(mx.load(str(ROOT/'artifacts/preference_adapters/coverage_continuation/adapters.safetensors')).items()),strict=False);model.eval()
tid=d['splits']['validation'][0];record=json.loads((ROOT/'results/all80/lm_scores'/f'validation_{tid}.json').read_text())
prefix=tok.apply_chat_template([{'role':'user','content':record['prompt']}],add_generation_prompt=True,tokenize=True)
answers=[{'answer':a,'key':h['key']} for h in items for a in h['aliases']];encoded=completion_tokens(tok,answers)
checks=[]
for index in [0,31,65]:
 full=prefix+encoded[index]['tokens'];g={'aliases':[{'tokens':full,'prompt_length':len(prefix)}],'class_ratio':1.}
 ids,mask,_=batch(g);logits=model(ids[:,:-1]);ce=nn.losses.cross_entropy(logits,ids[:,1:],reduction='none').astype(mx.float32)*mask
 direct=-float(ce.sum().item());cached=record['alias_log_probabilities'][index]
 checks.append({'alias_index':index,'full_logp':direct,'cached_batched_logp':cached,'absolute_difference':abs(direct-cached)})
assert max(c['absolute_difference'] for c in checks)<.05
result={'pass':True,'checks':checks,'note':'Small quantized GEMM shape differences are allowed; full probabilities include EOS and exclude prompt tokens.'}
write_json(ROOT/'results/all80/catalog_score_verification.json',result);print(json.dumps(result,indent=2))
