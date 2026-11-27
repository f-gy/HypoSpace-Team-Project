import json,itertools,collections,hashlib,math
from pathlib import Path
from boolean_dataset import BooleanDiscoveryGame
N=['x','y','z'];PTS=list(itertools.product([0,1],repeat=3));OPT={'apply_commutativity':True,'apply_idempotence_and_or':True,'flatten_associativity':True};ROOT=Path('work/lora_balanced')
def sha(x):return hashlib.sha256(x.encode()).hexdigest()
def serialize(obs):return ';'.join(''.join(map(str,k))+'='+str(v) for k,v in sorted(obs))
def partition(truths):
 out={}
 for anchor in (0,1):
  items=sorted([t for t in truths if t[0]==anchor],key=lambda t:sha('balanced-truth-split-v1:'+''.join(map(str,t))));n=math.ceil(len(items)*.7)
  for i,t in enumerate(items):out[t]='train' if i<n else 'dev'
 return out
def main():
 es=BooleanDiscoveryGame.generate_all_expressions(N,{'AND','OR','NOT'},2,OPT);truths={tuple(e.truth_table[p] for p in PTS) for e in es};assign=partition(truths);pool={};orig=collections.defaultdict(set)
 for t in sorted(truths):
  for n in (4,5):
   for extra in itertools.combinations(range(1,8),n-1):
    ix=(0,)+extra;obs=tuple((PTS[i],t[i]) for i in ix);key=serialize(obs);pool[key]=obs;orig[key].add((assign[t],t))
 ambiguous={k for k,v in orig.items() if len({s for s,_ in v})>1}; candidates={k:v for k,v in pool.items() if k not in ambiguous};by_obs={frozenset(obs):k for k,obs in candidates.items()};remove=set()
 for k,obs in candidates.items():
  if len(obs)!=5:continue
  s=next(iter(orig[k]))[0]
  for drop in range(1,5):
   short=obs[:drop]+obs[drop+1:];other=by_obs.get(frozenset(short))
   if other and next(iter(orig[other]))[0]!=s:remove.update((k,other))
 strata=collections.defaultdict(list)
 for k,obs in candidates.items():
  if k in remove:continue
  split=next(iter(orig[k]))[0];strata[(split,obs[0][1],len(obs))].append(k)
 selected={};audit={'all_mechanisms':len(es),'unique_truth_tables':len(truths),'truth_group_counts':{s:sum(x==s for x in assign.values()) for s in ('train','dev')},'truth_anchor_by_split':{s:{str(a):sum(assign[t]==s and t[0]==a for t in truths) for a in (0,1)} for s in ('train','dev')},'candidate_observation_keys':len(pool),'ambiguous_exact_excluded':len(ambiguous),'cross_subset_excluded':len(remove),'strata':{},'task_origin_truths':{}}
 for (split,a,n),keys in sorted(strata.items()):
  keys.sort(key=lambda k:sha('balanced-task-select-v1:'+k));limit=32 if split=='train' else 10;chosen=keys[:limit];audit['strata'][f'{split}_{a}_{n}']={'available':len(keys),'selected':len(chosen),'limit':limit}
  for k in chosen:selected[k]=split
 rows={'train':[],'dev':[]};answer_sets={};behavior_sets={}
 for k,s in sorted(selected.items()):
  obs=pool[k];matches=[e for e in es if all(e.truth_table[x]==y for x,y in obs)];row={'id':k,'observations':[{'inputs':dict(zip(N,x)),'output':y} for x,y in obs]}
  keys={repr(e.mechanistic_key(**OPT)) for e in matches};beh={tuple(e.truth_table[p] for p in PTS) for e in matches};answer_sets[k]=keys;behavior_sets[k]=beh;audit['task_origin_truths'][k]=[''.join(map(str,t)) for split,t in sorted(orig[k])];
  if s=='train':row['targets']=[{'expression':e.formula,'mechanistic_key':repr(e.mechanistic_key(**OPT))} for e in matches]
  rows[s].append(row)
 for s,rs in rows.items():(ROOT/(s+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rs))
 ta={r['id'] for r in rows['train']};da={r['id'] for r in rows['dev']};aobs=[frozenset(pool[k]) for k in ta];dobs=[frozenset(pool[k]) for k in da]; shared_struct={k for i in ta for k in answer_sets[i]} & {k for i in da for k in answer_sets[i]};shared_beh={k for i in ta for k in behavior_sets[i]} & {k for i in da for k in behavior_sets[i]};audit['excluded_difficulty']={'ambiguous_count':len(ambiguous),'ambiguous_mean_solutions':sum(sum(all(e.truth_table[x]==y for x,y in pool[k]) for e in es) for k in ambiguous)/len(ambiguous),'selected_train_mean_solutions':sum(len(answer_sets[k]) for k in ta)/len(ta),'selected_dev_mean_solutions':sum(len(answer_sets[k]) for k in da)/len(da)};audit['sizes']={s:len(rs) for s,rs in rows.items()};audit['train_targets']=sum(len(r['targets']) for r in rows['train']);audit['train_dev_exact_overlap']=len(ta&da);audit['train_dev_subset_pairs']=sum(a<=b or b<=a for a in aobs for b in dobs);audit['shared_possible_structure_count']=len(shared_struct);audit['shared_possible_behavior_count']=len(shared_beh);audit['target_count_distributions']={s:dict(collections.Counter(str(len(answer_sets[r['id']])) for r in rs)) for s,rs in rows.items()};audit['train_target_min']=min(len(r['targets']) for r in rows['train']); audit['dev_targets_persisted']=False
 assert not ta&da and audit['train_dev_subset_pairs']==0
 oldrows={r['id']:r for r in map(json.loads,Path('work/lora_revised/train.jsonl').read_text().splitlines())};order=list(map(json.loads,Path('work/lora_7b/training_order.jsonl').read_text().splitlines()));used={r['id'] for r in order};oldsets={k:frozenset((tuple(o['inputs'][n] for n in N),o['output']) for o in v['observations']) for k,v in oldrows.items()};audit['old_used_prompt_count']=len(used);audit['old_used_exact_new_dev']=len(used&da);audit['old_used_vs_new_dev_subset_pairs']=sum(oldsets[k]<=b or b<=oldsets[k] for k in used for b in dobs);audit['old_train_exact_new_dev']=len(set(oldrows)&da);audit['old_training_target_valid_in_some_new_dev']=sum(any(all(tuple(o['inputs'][n] for n in N) in next(e for e in es if repr(e.mechanistic_key(**OPT))==r['key']).truth_table and next(e for e in es if repr(e.mechanistic_key(**OPT))==r['key']).truth_table[tuple(o['inputs'][n] for n in N)]==o['output'] for o in d['observations']) for d in rows['dev']) for r in order)
 (ROOT/'audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps({k:v for k,v in audit.items() if k!='task_origin_truths'},indent=2))
if __name__=='__main__':main()
