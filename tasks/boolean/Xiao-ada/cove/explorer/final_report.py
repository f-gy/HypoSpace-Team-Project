"""Aggregate preregistered final runs, export evidence and paired uncertainty."""
from __future__ import annotations
import ast,csv,gzip,hashlib,json,math,statistics
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from .upstream import ROOT
from .tasks import write_json
from .experiment import read_rows

METHODS=['base_iid','balanced_sft_iid','canonical_coverage_iid','string_coverage_iid','base_memory']
RUN=ROOT/'runs/final_v1';OUT=ROOT/'results/final'


def aggregate(split):
    data=json.loads((ROOT/'.cache/tasks.json').read_text())
    ids=json.loads((ROOT/'results/coverage_plan/split_manifest.json').read_text())['splits'][split]
    refs={t['id']:set(t['reference_keys']) for t in data['tasks']}
    seeds=[101] if split=='validation' else [17,29,43]
    records=[];curves=[];evidence=[];hashes={};outliers=[];strata=[]
    def complexity(key):
        node=ast.literal_eval(key)
        def walk(v):
            if not isinstance(v,tuple):return 0
            return int(bool(v) and isinstance(v[0],str) and v[0] in {"AND","OR","NOT"})+sum(walk(x) for x in v[1:] if isinstance(x,tuple)) if v and isinstance(v[0],str) else sum(walk(x) for x in v)
        return walk(node)
    for method in METHODS:
        for tid in ids:
            for seed in seeds:
                p=RUN/f'{split}_{method}_{tid}_{seed}.jsonl'
                rows=read_rows(p)
                if len(rows)!=16:raise ValueError(f'Incomplete episode: {p.name} has {len(rows)} rows')
                hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
                seen=set();valid_seen=set();valid_count=0;ref_count=0;seconds=0;tokens=0
                for n,r in enumerate(rows,1):
                    assert r['step']==n-1 and r['contract']['method']==method
                    assert r['contract']['task_id']==tid and r['contract']['split']==split
                    assert r['error'] is None
                    h=r['outcome'];valid_count+=h['valid'];seconds+=r['seconds'];tokens+=r['usage']['total_tokens']
                    if h['in_space']:seen.add(h['key'])
                    if h['valid']:
                        valid_seen.add(h['key'])
                        if h['key'] in refs[tid]:ref_count+=1
                        else:outliers.append({'method':method,'task':tid,'seed':seed,'step':n,'expression':h['expression'],'key':h['key']})
                    rec={'method':method,'task':tid,'seed':seed,'queries':n,'VR':valid_count/n,'NR':len(seen)/n,'RR':len(valid_seen&refs[tid])/len(refs[tid]),'reference_validity':ref_count/n,'reference_size':len(refs[tid]),'recovered':len(valid_seen&refs[tid]),'outside_reference':len(valid_seen-refs[tid]),'seconds':seconds,'tokens':tokens}
                    curves.append(rec)
                    if n==16:
                        records.append(rec)
                        for label,predicate in [('simple_0_or_1',lambda c:c<=1),('complex_2_or_more',lambda c:c>=2)]:
                            subset={key for key in refs[tid] if predicate(complexity(key))}
                            if subset:strata.append({'method':method,'task':tid,'seed':seed,'stratum':label,'reference_size':len(subset),'RR':len(valid_seen&subset)/len(subset)})
                    evidence.append({k:r[k] for k in ['contract','step','task','prompt','raw_response','outcome','usage','error','seconds','request_seed']})
    metrics={m:{k:float(np.mean([r[k] for r in records if r['method']==m])) for k in ['VR','NR','RR','reference_validity','seconds','tokens','recovered','outside_reference']} for m in METHODS}
    comparisons={}
    for base in ['base_iid','balanced_sft_iid','string_coverage_iid','base_memory']:
        diffs=[]
        for tid in ids:
            a=np.mean([r['RR'] for r in records if r['method']=='canonical_coverage_iid' and r['task']==tid])
            b=np.mean([r['RR'] for r in records if r['method']==base and r['task']==tid])
            diffs.append(float(a-b))
        boot=np.random.default_rng(4300).choice(diffs,(10000,len(diffs)),replace=True).mean(axis=1)
        comparisons[base]={'RR_difference':float(np.mean(diffs)),'task_bootstrap_95_interval':np.quantile(boot,[.025,.975]).tolist(),'per_task_RR_difference':dict(zip(ids,diffs)),'paired_tasks':len(ids)}
    budget={str(q):{m:{k:float(np.mean([r[k] for r in curves if r['method']==m and r['queries']==q])) for k in ['VR','NR','RR']} for m in METHODS} for q in [1,2,4,8,10,16]}
    out=OUT/split;out.mkdir(parents=True,exist_ok=True)
    for name,items in [('episodes',records),('curves',curves)]:
        with (out/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
    # Plain JSONL is auditable; the prompts contain only benchmark observations/rules.
    with gzip.GzipFile(filename=str(out/'evidence.jsonl.gz'),mode='wb',mtime=0) as f:
        for r in evidence:f.write((json.dumps(r,ensure_ascii=False)+'\n').encode())
    write_json(out/'out_of_reference.json',outliers)
    write_json(out/'complexity_exploratory.json',{'note':'Post-hoc description; not a preregistered primary outcome. Operator counts on official mechanistic keys. Empty strata excluded.','episodes':strata})
    write_json(out/'raw_manifest.json',hashes)
    # Post-hoc diagnostic: equalize the number of reference-valid draws within
    # each paired episode, then compare expected distinct coverage. No new calls.
    valid_lists=defaultdict(list)
    for r in evidence:
        c=r['contract'];h=r['outcome']
        if h['valid'] and h['key'] in refs[c['task_id']]:
            valid_lists[(c['method'],c['task_id'],c['seed'])].append(h['key'])
    matched=[]
    for tid in ids:
        for seed in seeds:
            a=valid_lists[('canonical_coverage_iid',tid,seed)]
            b=valid_lists[('balanced_sft_iid',tid,seed)]
            m=min(len(a),len(b))
            if not m:continue
            def expected_unique(items):
                n=len(items);den=math.comb(n,m)
                return sum(1-(math.comb(n-f,m)/den if n-f>=m else 0) for f in Counter(items).values())
            matched.append({'task':tid,'seed':seed,'matched_reference_valid_draws':m,
                            'expected_RR_difference':(expected_unique(a)-expected_unique(b))/len(refs[tid])})
    by_task=defaultdict(list)
    for row in matched:by_task[row['task']].append(row['expected_RR_difference'])
    write_json(out/'matched_validity_exploratory.json',{
        'note':'Post-hoc diagnostic, not primary inference. Exact hypergeometric downsampling of recorded reference-valid outputs; zero-common-valid episodes excluded.',
        'included_episodes':len(matched),'total_paired_episodes':len(ids)*len(seeds),
        'included_tasks':len(by_task),'task_mean_expected_RR_difference':float(np.mean([np.mean(v) for v in by_task.values()])) if by_task else None,
        'episodes':matched})
    result={'split':split,'tasks':len(ids),'seeds':seeds,'calls_per_episode':16,'total_calls':len(evidence),'metrics':metrics,'budget_metrics':budget,'canonical_vs':comparisons,'scoring':'Pinned official parser/canonicalizer/validator plus disclosed fail-closed gate. Full reference denominator retained; no repaired score.'}
    write_json(out/'summary.json',result)
    return result


def main():
    v=aggregate('validation');t=aggregate('test')
    write_json(OUT/'training_summary.json',json.loads((RUN/'training_summary.json').read_text()))
    write_json(OUT/'source_manifest.json',json.loads((RUN/'source_manifest.json').read_text()))
    write_json(OUT/'evaluation_config.json',json.loads((RUN/'evaluation_config_test.json').read_text()))
    (OUT/'training_log.jsonl').write_text((RUN/'training_log.jsonl').read_text())
    ledger=json.loads((RUN/'generation_ledger.json').read_text())
    assert ledger['calls']==v['total_calls']+t['total_calls']
    write_json(OUT/'generation_ledger.json',ledger)
    print(json.dumps(t,indent=2))
if __name__=='__main__':main()
