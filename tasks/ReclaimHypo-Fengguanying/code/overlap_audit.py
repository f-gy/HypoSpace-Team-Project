import json,itertools,statistics
from pathlib import Path
from boolean_dataset import BooleanDiscoveryGame
p=Path('work/lora_balanced');audit=json.load(open(p/'audit.json'));order=list(map(json.loads,Path('work/lora_7b/training_order.jsonl').read_text().splitlines()));opt={'apply_commutativity':True,'apply_idempotence_and_or':True,'flatten_associativity':True};es=BooleanDiscoveryGame.generate_all_expressions(['x','y','z'],{'AND','OR','NOT'},2,opt);pts=list(itertools.product([0,1],repeat=3));lookup={repr(e.mechanistic_key(**opt)):''.join(str(e.truth_table[t]) for t in pts) for e in es};old={lookup[x['key']] for x in order};used={x['id'] for x in order};rows=list(map(json.loads,(p/'dev.jsonl').read_text().splitlines()));q=[{'id':r['id'],'old_exact_prompt':r['id'] in used,'old_origin_behavior_overlap':bool(set(audit['task_origin_truths'][r['id']])&old)} for r in rows];(p/'old_overlap_per_item.json').write_text(json.dumps(q,indent=2));olddev=list(map(json.loads,Path('work/lora_revised/dev.jsonl').read_text().splitlines()));obs=lambda r:frozenset((tuple(o['inputs'][v] for v in ('x','y','z')),o['output']) for o in r['observations']);report={'prior_development_exact_new_dev':len({r['id'] for r in olddev}&{r['id'] for r in rows}),'prior_development_subset_pairs_new_dev':sum(obs(x)<=obs(y) or obs(y)<=obs(x) for x in olddev for y in rows),'old_used_origin_overlap_count':sum(x['old_origin_behavior_overlap'] for x in q),'old_used_exact_count':sum(x['old_exact_prompt'] for x in q)}
if (p/'dev_scored.jsonl').exists():
 scored=list(map(json.loads,(p/'dev_scored.jsonl').read_text().splitlines()));res={}
 for subset,ids in [('clean',{x['id'] for x in q if not x['old_origin_behavior_overlap']}),('old_exact',{x['id'] for x in q if x['old_exact_prompt']})]:
  res[subset]={'n_items':len(ids)}
  for m in ['base','oldplain','plain_1337','weighted_1337','plain_31415','weighted_31415']:
   r=[x for x in scored if x['mode']==m and x['id'] in ids];res[subset][m]={k:statistics.mean(x[k] for x in r) for k in ['validity','recovery','behavior_coverage']};res[subset][m]['valid_candidates']=sum(x['valid_count'] for x in r)
 (p/'old_overlap_results.json').write_text(json.dumps(res,indent=2))
(p/'overlap_summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
