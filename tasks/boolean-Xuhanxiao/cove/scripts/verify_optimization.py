"""Verify public optimization evidence and training labels without model inference."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from explorer.coverage_lora import dataset
from explorer.experiment import assess
from explorer.target_decode import intervene
from explorer.tasks import prepare,write_json
from explorer.upstream import evaluator,load


def rows(path):
    with gzip.open(path,'rt') as f:return [json.loads(line) for line in f]


def verify_evidence(path,summary,d,original=None):
    raw=rows(path);groups=defaultdict(list)
    for r in raw:
        c=r['contract'];groups[(c['method'],c['task_id'],c['seed'])].append(r)
    all_rows={(r['contract']['method'],r['contract']['task_id'],r['contract']['seed'],r['step']):r for r in raw}
    tasks={t['id']:t for t in d['tasks']};ev=evaluator(d['metadata'])
    stats=defaultdict(list);counts=defaultdict(int)
    for (method,tid,seed),episode in groups.items():
        episode.sort(key=lambda r:r['step']);assert len(episode)==16
        history,seen,valid_seen=[],set(),set();nv=0;active_count=0
        for i,r in enumerate(episode):
            assert r['step']==i and r['error'] is None
            assert set(r['task'])=={'id','observations'}
            assert r['task']['observations']==tasks[tid]['observations']
            h=assess(ev,d['metadata'],r['task'],r['raw_response'],history)
            for k in ['expression','key','in_space','valid','new','feedback']:assert h[k]==r['outcome'][k]
            if method.endswith('class_wor') or method.startswith('cc_class_cfg'):
                assert h['in_space'] and h['key'] not in seen
            if method.endswith('_bace') or '_target_' in method:
                active=intervene(len(seen),i)
                assert active==r['decoder']['active']
                if active:
                    active_count+=1
                    if not method.endswith('_string'):assert h['in_space'] and h['key'] not in seen
            if original is not None:
                assert r['prompt']==original[('canonical_coverage_iid',tid,seed,i)]['prompt']
            if r.get('replay'):
                assert active_count==0
                replay=r['replay']
                if 'method' in replay:
                    expected=method.removesuffix('_bace')+'_iid'
                    assert replay['method']==expected
                    source=all_rows[(expected,tid,seed,i)]
                else:
                    expected={'cc':'canonical_coverage_iid','sft':'balanced_sft_iid','base':'base_iid'}[method.split('_')[0]]
                    source=original[(expected,tid,seed,i)]
                for k in ['raw_response','prompt','request_seed','usage','seconds']:assert r[k]==source[k]
                counts['replayed_model_calls']+=1
                counts['replayed_model_tokens']+=r['usage']['total_tokens']
            elif method.startswith('uniform'):
                counts['uniform_draws']+=1
            else:
                counts['new_model_calls']+=1
                counts['new_model_tokens']+=r['usage']['total_tokens']
            if h['in_space']:seen.add(h['key'])
            if h['valid']:nv+=1;valid_seen.add(h['key'])
            history.append(h)
        if method.endswith('_bace') or method.endswith('_target_class'):assert len(seen)>=13
        refs=set(tasks[tid]['reference_keys'])
        stats[method].append([nv/16,len(seen)/16,len(valid_seen&refs)/len(refs)])
    for method,values in stats.items():
        for k,value in zip(['VR','NR','RR'],np.mean(values,axis=0)):
            assert abs(value-summary['metrics'][method][k])<1e-10,(str(path),method,k)
    if 'counts' in summary:
        assert dict(counts)==summary['counts'],(str(path),counts,summary['counts'])
    return {'outputs_rescored':len(raw),'episodes':len(groups),'metrics_match':True,'counts':dict(counts)}


def verify_training(d):
    with gzip.open(ROOT/'results/canonical_preference_v5/training_targets.json.gz','rt') as f:data=json.load(f)
    assert set(data['training_ids'])==set(d['splits']['train'])
    tasks={t['id']:t for t in d['tasks']};ev=evaluator(d['metadata']);mod,_=load()
    for g in data['groups']:
        assert g['task_id'] in d['splits']['train'] and len(g['aliases'])==4
        assert math.isfinite(g['reference_margin'])
        task=tasks[g['task_id']]
        prefix=g['aliases'][0]['tokens'][:g['aliases'][0]['prompt_length']]
        for i,a in enumerate(g['aliases']):
            assert a['tokens'][:a['prompt_length']]==prefix
            h=assess(ev,d['metadata'],task,a['answer'],[])
            assert h['in_space'] and h['valid']==(i<2)
            assert h['key']==(g['key'] if i<2 else g['negative_key'])
            if i>=2:
                expr=mod.BooleanExpression(h['expression'],d['metadata']['variables'],set(d['metadata']['operators']))
                assert sum(expr.evaluate(o['inputs'])!=o['output'] for o in task['observations'])==1
    return {'training_tasks':len(data['training_ids']),'preference_pairs_checked':len(data['groups']),
            'all_labels_train_only_and_correct':True}


def main():
    if not (ROOT/'.cache/tasks.json').exists():prepare(json.loads((ROOT/'config.json').read_text()))
    d=dataset();result={'original_source_files_unchanged':True,'studies':{}}
    manifest=json.loads((ROOT/'results/final/source_manifest.json').read_text())
    for p,digest in manifest['source_sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest,p
    originals={}
    for split in ['validation','test']:
        originals[split]={(r['contract']['method'],r['contract']['task_id'],r['contract']['seed'],r['step']):r
                          for r in rows(ROOT/'results/final'/split/'evidence.jsonl.gz')}
    specs=[('canonical_decode_v2/validation',None),('contrastive_decode_v3',None),('target_decode_v4/validation','validation'),
           ('canonical_preference_v5/validation','validation'),('canonical_preference_v5/test','test')]
    for relative,source_split in specs:
        p=ROOT/'results'/relative
        summary=json.loads((p/'summary.json').read_text())
        result['studies'][relative]=verify_evidence(p/'evidence.jsonl.gz',summary,d,originals[source_split] if source_split else None)
        print('PASS',relative,result['studies'][relative]['outputs_rescored'],'raw outputs',flush=True)
    result['training']=verify_training(d)
    for path in ['results/canonical_preference_v5/protocol.json','results/canonical_preference_v5/evaluation_protocol.json']:
        cfg=json.loads((ROOT/path).read_text())
        for name,digest in cfg['sources'].items():assert hashlib.sha256((ROOT/'explorer'/name).read_bytes()).hexdigest()==digest,name
    adapter_manifest=json.loads((ROOT/'artifacts/preference_adapters/manifest.json').read_text())
    for name,digest in adapter_manifest['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    result['adapter_hashes_match']=True;result['pass']=True
    write_json(ROOT/'results/optimization_verification.json',result)
    print('PASS: all public optimization outputs, training labels, original source hashes and new adapters verified')


if __name__=='__main__':main()
