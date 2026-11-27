import json,tempfile,os,hashlib,collections
from pathlib import Path
import sgve_core as c
from boolean_dataset import BooleanObservation
from hypospace_validator import generated_universe
root=Path('work/lora_balanced');meta={'metadata':{'variables':['x','y','z'],'operators':['AND','OR','NOT'],'max_depth':2,'mechanistic_opts':c.b.mechanistic_opts},'datasets_by_n_observations':{}}
with tempfile.NamedTemporaryFile(mode='w',delete=False) as f:json.dump(meta,f);tmp=f.name
try:
 b=c.BooleanBenchmarkRefined(tmp);v=c.install(b,policy='enumerated',universe=generated_universe(b));train=list(map(json.loads,(root/'train.jsonl').read_text().splitlines()));dev=list(map(json.loads,(root/'dev.jsonl').read_text().splitlines()));errors=[];n=0
 for r in train:
  obs=[BooleanObservation(o['inputs'],o['output']) for o in r['observations']]
  for t in r['targets']:
   n+=1;ok,_=v.validate_expression(t['expression'],obs)
   if not ok or repr(v.get_expression_mechanistic_key(t['expression']))!=t['mechanistic_key']:errors.append([r['id'],t['expression']])
 assert len({r['id'] for r in train})==len(train) and len({r['id'] for r in dev})==len(dev) and all('targets' not in r for r in dev)
 result={'checked_training_target_pairs':n,'errors':errors,'error_count':len(errors),'train_sha256':hashlib.sha256((root/'train.jsonl').read_bytes()).hexdigest(),'dev_sha256':hashlib.sha256((root/'dev.jsonl').read_bytes()).hexdigest(),'protocol_sha256':hashlib.sha256((root/'PROTOCOL.md').read_bytes()).hexdigest()};(root/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));assert not errors
finally:os.unlink(tmp)
