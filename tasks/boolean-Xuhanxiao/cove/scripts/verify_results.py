"""Re-score public evidence and verify aggregate results without a GPU."""
import gzip,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from explorer.experiment import assess
from explorer.upstream import evaluator
from explorer.tasks import prepare
if not (ROOT/'.cache/tasks.json').exists():prepare(json.loads((ROOT/'config.json').read_text()))
d=json.loads((ROOT/'.cache/tasks.json').read_text());by_id={t['id']:t for t in d['tasks']};ev=evaluator(d['metadata'])
manifest=json.loads((ROOT/'results/coverage_plan/split_manifest.json').read_text())['splits']
for split in ['validation','test']:
 out=ROOT/'results/final'/split;summary=json.loads((out/'summary.json').read_text());groups=defaultdict(list)
 with gzip.open(out/'evidence.jsonl.gz','rt') as f:
  for line in f:
   r=json.loads(line);c=r['contract'];assert c['split']==split and c['task_id'] in manifest[split]
   assert 'reference_keys' not in r['task'] and r['error'] is None
   groups[(c['method'],c['task_id'],c['seed'])].append(r)
 stats=defaultdict(list);count=0
 for (method,tid,seed),rows in groups.items():
  rows.sort(key=lambda r:r['step']);assert len(rows)==16
  history=[];seen=set();valid=set();nv=0
  for i,r in enumerate(rows):
   assert r['step']==i
   a=assess(ev,d['metadata'],r['task'],r['raw_response'],history)
   for key in ['expression','key','in_space','valid','new']:assert a[key]==r['outcome'][key],(method,tid,i,key)
   nv+=int(a['valid'])
   if a['in_space']:seen.add(a['key'])
   if a['valid']:valid.add(a['key'])
   history.append(r['outcome']);count+=1
  ref=set(by_id[tid]['reference_keys']);stats[method].append([nv/16,len(seen)/16,len(valid&ref)/len(ref)])
 for method,values in stats.items():
  means=np.mean(values,axis=0)
  for k,v in zip(['VR','NR','RR'],means):assert abs(v-summary['metrics'][method][k])<1e-10
 assert count==summary['total_calls']
 print(f'PASS {split}: {count} raw outputs rescored; VR/NR/RR match published summary')
print('Published evidence verified. No GPU, model inference or credentials used.')
