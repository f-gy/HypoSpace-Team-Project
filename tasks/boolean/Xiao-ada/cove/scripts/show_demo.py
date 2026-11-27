"""Show the first fixed validation task from saved real outputs, without inference."""
import gzip,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
tid=json.loads((root/'results/coverage_plan/split_manifest.json').read_text())['splits']['validation'][0]
with gzip.open(root/'results/final/validation/evidence.jsonl.gz','rt') as f:rows=[json.loads(s) for s in f]
selected=[r for r in rows if r['task']['id']==tid and r['step']<3]
print('Fixed validation task:',tid)
print('Observations:',json.dumps(selected[0]['task']['observations']))
for method in ['base_iid','balanced_sft_iid','canonical_coverage_iid']:
 print('\n'+method)
 for r in selected:
  if r['contract']['method']==method:
   print({'attempt':r['step']+1,'response':r['raw_response'],'validator_pass':r['outcome']['valid'],'new_valid_key':r['outcome']['new']})
print('\nSaved real evidence; this display does not generate or select a best test example.')
