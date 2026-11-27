"""Paired adaptive/fixed-budget study of learned execution and LM catalog scores."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math

import numpy as np

from .coverage_lora import dataset
from .experiment import assess
from .posterior_bank import bank,numpy_predictions,observation_log_likelihood,exact_tree_values
from .tasks import public_task,write_json
from .upstream import ROOT,evaluator

OUT=ROOT/'results/all80'
METHODS=['base_iid','frozen_iid','llm_catalog_wor','cove_llm_4306','cove_llm_4307','cove_llm_4308',
         'cove_only_4306','uniform_wor','symbolic_wor']


def source_iid(split):
    result={}
    for folder,source_name,target_name in [('final','base_iid','base_iid'),
                                          ('canonical_preference_v5','coverage_continuation_iid','frozen_iid')]:
        with gzip.open(ROOT/'results'/folder/split/'evidence.jsonl.gz','rt') as f:
            for line in f:
                r=json.loads(line);c=r['contract']
                if c['method']==source_name:result[(target_name,c['task_id'],c['seed'],r['step'])]=r
    extra=OUT/'iid_extras.jsonl'
    if extra.exists():
        for line in extra.read_text().splitlines():
            r=json.loads(line);result[(r['method'],r['task_id'],r['seed'],r['step'])]=r
    return result


def weights_for(method,score,neural_likelihood):
    """Learned proposal weights: no exact execution or solution labels accepted."""
    if method=='llm_catalog_wor':return np.asarray(score['class_log_mass'])
    if method.startswith('cove_llm_'):
        seed=int(method.rsplit('_',1)[1])
        return np.asarray(score['class_log_mass'])+10*neural_likelihood[seed]
    if method=='cove_only_4306':return 10*neural_likelihood[4306]
    if method=='uniform_wor':return np.zeros(len(score['class_keys']))
    raise ValueError(method)


def ranked_sample(logweights,seed):
    rng=np.random.default_rng(seed)
    return np.argsort(-(np.asarray(logweights)+rng.gumbel(size=len(logweights))),kind='stable').tolist()


def evaluate(split):
    d=dataset();items=bank();keys=[h['key'] for h in items]
    ev=evaluator(d['metadata']);tasks={t['id']:t for t in d['tasks']}
    seeds=[101] if split=='validation' else [17,29,43]
    models={seed:json.loads((OUT/'executors'/str(seed)/'weights.json').read_text()) for seed in [4306,4307,4308]}
    pred={seed:numpy_predictions(w,items) for seed,w in models.items()}
    # Evaluation-only exact executor. Passed solely to symbolic_wor; COVE uses
    # predictions from learned weights and observed data, not these values.
    bits=np.array([[0,0],[0,1],[1,0],[1,1]])
    exact=np.stack([exact_tree_values(h['tree'],bits) for h in items])
    original=source_iid(split)
    settings={'methods':METHODS,'protocols':['adaptive_1x','fixed16'],'seeds':seeds,
              'primary_method':'cove_llm_4306','primary_protocol':'adaptive_1x',
              'primary_target':'Macro VR, NR and RR all > .8',
              'learned_executor_seeds':'4306 primary; 4307/4308 fixed replication, no test selection',
              'posterior_tau':1.,'posterior_beta':10.,'class_sampling':'Gumbel top-N, no replacement',
              'catalog_source':'Original training split aliases plus grammar-only structural representatives',
              'benchmark_budget':'adaptive_1x uses N=reference K, matching official README query-multiplier=1.0; K is supplied only as budget, not a solution-membership label',
              'fairness':'All methods receive same task, observations and budget in a protocol; both protocols reported separately',
              'proposal_type':'LLM-scored finite catalog selection, not unrestricted autoregressive generation',
              'exact_constraints':'Only symbolic_wor uses exact execution. Proposed COVE likelihood uses learned 179-parameter modules.',
              'test_status':'Exploratory known-v1 observation follow-up; auxiliary unseen-depth executor probe is separate',
              'source_sha256':{n:hashlib.sha256((ROOT/'explorer'/n).read_bytes()).hexdigest()
                               for n in ['all80_experiment.py','catalog_scores.py','posterior_bank.py','learned_executor.py']}}
    write_json(OUT/f'evaluation_protocol_{split}.json',settings)
    records=[];episodes=[]
    for tid in d['splits'][split]:
        task=public_task(tasks[tid]);refs=set(tasks[tid]['reference_keys'])
        score=json.loads((OUT/'lm_scores'/f'{split}_{tid}.json').read_text());assert score['class_keys']==keys
        likelihood={s:observation_log_likelihood(p,task['observations']) for s,p in pred.items()}
        exact_valid=np.ones(len(items),dtype=bool)
        for o in task['observations']:
            exact_valid &= exact[:,2*o['inputs']['x']+o['inputs']['y']]==o['output']
        for seed in seeds:
            sampling_seed=int.from_bytes(hashlib.sha256(f'{seed}:{tid}:catalog'.encode()).digest()[:8],'big')
            rankings={m:ranked_sample(1000*np.asarray(exact_valid,dtype=float)
                                      if m=='symbolic_wor' else weights_for(m,score,likelihood),sampling_seed)
                      for m in METHODS if not m.endswith('_iid')}
            for protocol in ['adaptive_1x','fixed16']:
                n=len(refs) if protocol=='adaptive_1x' else 16
                for method in METHODS:
                    history=[];seen=set();valid_seen=set();valid_count=0
                    for step in range(n):
                        if method.endswith('_iid'):
                            source=original[(method,tid,seed,step)]
                            text=source['raw_response'];proposal={'kind':'unchanged IID output replay','original_step':step}
                        else:
                            index=rankings[method][step];item=items[index]
                            text=item['aliases'][0]
                            proposal={'kind':'fixed-catalog selection','class_index':index,
                                      'llm_log_class_mass':score['class_log_mass'][index] if method.startswith(('llm_','cove_llm_')) else None,
                                      'learned_log_likelihood':float(likelihood[int(method.rsplit('_',1)[1])][index]) if method.startswith(('cove_llm_','cove_only_')) else None,
                                      'uses_exact_constraint_solver':method=='symbolic_wor'}
                        outcome=assess(ev,d['metadata'],task,text,history)
                        if outcome['in_space']:seen.add(outcome['key'])
                        if outcome['valid']:valid_count+=1;valid_seen.add(outcome['key'])
                        row={'split':split,'protocol':protocol,'method':method,'task_id':tid,'seed':seed,'step':step,
                             'budget':n,'task':task,'raw_response':text,'outcome':outcome,'proposal':proposal,
                             'sampling_seed':sampling_seed}
                        records.append(row);history.append(outcome)
                    episodes.append({'protocol':protocol,'method':method,'task':tid,'seed':seed,'budget':n,
                                     'VR':valid_count/n,'NR':len(seen)/n,'RR':len(valid_seen&refs)/len(refs),
                                     'reference_size':len(refs),'recovered':len(valid_seen&refs),
                                     'outside_reference':len(valid_seen-refs)})
    out=OUT/split;out.mkdir(parents=True,exist_ok=True)
    with gzip.GzipFile(filename=str(out/'evidence.jsonl.gz'),mode='wb',mtime=0) as f:
        for r in records:f.write((json.dumps(r,ensure_ascii=False)+'\n').encode())
    write_json(out/'episodes.json',episodes)
    metrics={protocol:{method:{k:float(np.mean([r[k] for r in episodes if r['protocol']==protocol and r['method']==method]))
                              for k in ['VR','NR','RR']} for method in METHODS} for protocol in ['adaptive_1x','fixed16']}
    summary={'split':split,'tasks':len(d['splits'][split]),'seeds':seeds,'metrics':metrics,
             'outputs':len(records),'test_reference_labels_in_proposed_sampler':False,
             'note':'Candidate scoring cost is separate from final proposal budget; symbolic and executor-only controls explicitly included.'}
    write_json(out/'summary.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('split',choices=['validation','test']);evaluate(parser.parse_args().split)
