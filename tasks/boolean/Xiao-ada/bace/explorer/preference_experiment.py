"""Fixed-budget IID/BACE evaluation of both matched preference-training arms."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

from .canonical_decode import SupportTrie, TrieLogitsProcessor, completion_tokens, grammar_support
from .canonical_preference import OUT, RUN as TRAIN
from .coverage_lora import LORA, dataset, model_load
from .experiment import assess, read_rows
from .prompt_archive import recorded_prompt
from .prefix_cache import build_prefix, fresh_prefix
from .target_decode import intervene
from .tasks import public_task, write_json
from .upstream import evaluator

RUN = TRAIN / 'evaluation'
METHODS = ['coverage_continuation_iid', 'canonical_preference_iid',
           'coverage_continuation_bace', 'canonical_preference_bace', 'uniform_bace']


def config():
    training = json.loads((OUT/'training_summary.json').read_text())
    cfg = {'methods': METHODS, 'seeds': {'validation':[101], 'test':[17,29,43]},
           'budget':16, 'bace_target':13, 'temperature':1., 'top_p':1., 'max_output_tokens':128,
           'training': training,
           'analysis_plan_sha256': hashlib.sha256((OUT/'analysis_plan.json').read_bytes()).hexdigest(),
           'sources': {p: hashlib.sha256((Path(__file__).parent/p).read_bytes()).hexdigest()
                       for p in ['preference_experiment.py', 'canonical_decode.py', 'target_decode.py', 'prompt_archive.py', 'prefix_cache.py']},
           'test_status':'Exploratory follow-up on existing v1 held-out observation sets; not a new untouched test',
           'selection_rule':'Publish both fixed final checkpoints and both inference modes. Prefer a candidate that raises validation NR above .8, increases RR and loses no more than .02 VR versus original canonical. If none passes, report the trade-off and retain v1 as main.',
           'cache_policy':'Unchanged IID prefix of the same checkpoint is replayed from this v5 evaluation, counted in all logical costs. No v1 output is replayed under a v5 checkpoint.',
           'kv_cache_policy':'Identical prompt-prefix KV states are reused per adapter/prompt; the last prompt token and every answer token are computed afresh. Actual-model seeded equivalence checked before validation. Logical prompt tokens remain fully counted; physical prefill work is logged separately.',
           'fairness':'Byte-identical immutable v1 prompt text; identical sampler, seed generation, output cap, scoring and attempt budget. No ground-truth validity filtering during proposal.'}
    p = OUT/'evaluation_protocol.json'
    if p.exists():
        assert json.loads(p.read_text()) == cfg
    else:
        write_json(p,cfg)
    return cfg


def evaluate(split):
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.tuner.utils import linear_to_lora_layers
    cfg, d = config(), dataset()
    RUN.mkdir(parents=True,exist_ok=True)
    model,tok = model_load()
    model.freeze()
    linear_to_lora_layers(model,8,LORA)
    model.eval()
    weights = {arm:mx.load(str(TRAIN/'adapters'/arm/'adapters.safetensors'))
               for arm in ['coverage_continuation','canonical_preference']}
    support = completion_tokens(tok,grammar_support(d['metadata']))
    classes = {}
    for h in support:
        classes.setdefault(h['key'],[]).append(h)
    tasks = {t['id']:t for t in d['tasks']}
    ev = evaluator(d['metadata'])
    signature = hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    ledger_path = RUN/'generation_ledger.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {
        'new_model_calls':0,'replayed_model_calls':0,'uniform_draws':0,
        'new_model_tokens':0,'replayed_model_tokens':0,'errors':0,
        'prefix_builds':0,'physical_prefix_tokens':0,'physical_last_prompt_tokens':0,
        'physical_completion_tokens':0,'prefix_build_seconds':0.}
    for tid in d['splits'][split]:
        task = public_task(tasks[tid])
        prefixes = {}
        for seed in cfg['seeds'][split]:
            prompt = recorded_prompt(split,tid,seed)
            ids = tok.apply_chat_template([{'role':'user','content':prompt}],add_generation_prompt=True,tokenize=True)
            # Randomize arm order inside each phase, preserving IID-before-BACE
            # dependency required for exact prefix caching.
            iid = METHODS[:2].copy()
            tail = METHODS[2:].copy()
            random.Random(seed+int(tid,16)).shuffle(iid)
            random.Random(seed*3+int(tid,16)).shuffle(tail)
            for method in iid+tail:
                arm = method.removesuffix('_iid').removesuffix('_bace')
                uniform = method=='uniform_bace'
                bace = method.endswith('_bace')
                path = RUN/f'{split}_{method}_{tid}_{seed}.jsonl'
                contract = {'fingerprint':signature,'split':split,'task_id':tid,'method':method,'seed':seed,'budget':16}
                rows = read_rows(path)
                assert all(r['contract']==contract and r['step']==i for i,r in enumerate(rows))
                history = [r['outcome'] for r in rows]
                replay_rows = read_rows(RUN/f'{split}_{arm}_iid_{tid}_{seed}.jsonl') if bace and not uniform else []
                if not uniform and len(rows)<16:
                    model.load_weights(list(weights[arm].items()),strict=False)
                    mx.eval(model.parameters())
                    cache_key=(arm,tuple(ids))
                    if cache_key not in prefixes:
                        start=time.monotonic()
                        prefixes[cache_key]={'cache':build_prefix(model,ids),'seconds':0.}
                        elapsed=time.monotonic()-start
                        prefixes[cache_key]['seconds']=elapsed
                        ledger['prefix_builds']+=1
                        ledger['physical_prefix_tokens']+=len(ids)-1
                        ledger['prefix_build_seconds']+=elapsed
                        write_json(ledger_path,ledger)
                for step in range(len(rows),16):
                    seen = {h['key'] for h in history if h['in_space']}
                    active = bace and intervene(len(seen),step)
                    root_seed = int.from_bytes(hashlib.sha256(f'{seed}:{step}:{tid}'.encode()).digest()[:8],'big')
                    request_seed = random.Random(root_seed).getrandbits(31)
                    replay = None
                    start = time.monotonic()
                    if uniform:
                        rng = random.Random(request_seed)
                        key = rng.choice(sorted(set(classes)-seen if active else set(classes)))
                        answer = rng.choice(classes[key])['answer']
                        usage = {'prompt_tokens':0,'completion_tokens':0,'total_tokens':0}
                        seconds = time.monotonic()-start
                        ledger['uniform_draws']+=1
                    elif bace and not active:
                        assert len(replay_rows)==16 and not any(r['decoder']['active'] for r in rows)
                        source = replay_rows[step]
                        answer,usage,seconds = source['raw_response'],source['usage'],source['seconds']
                        assert source['prompt']==prompt and source['request_seed']==request_seed
                        replay = {'method':arm+'_iid','step':step}
                        ledger['replayed_model_calls']+=1
                        ledger['replayed_model_tokens']+=usage['total_tokens']
                    else:
                        processors = [TrieLogitsProcessor(SupportTrie(support,seen),1)] if active else []
                        mx.random.seed(request_seed)
                        ledger['new_model_calls']+=1
                        ledger['physical_last_prompt_tokens']+=1
                        write_json(ledger_path,ledger)
                        parts,last = [],None
                        error = None
                        try:
                            for last in stream_generate(model,tok,[ids[-1]],max_tokens=128,
                                                        sampler=make_sampler(temp=1.,top_p=1.,top_k=0,min_p=0.),
                                                        logits_processors=processors,
                                                        prompt_cache=fresh_prefix(prefixes[(arm,tuple(ids))]['cache'])):
                                parts.append(last.text)
                        except Exception as exc:
                            ledger['errors']+=1
                            write_json(ledger_path,ledger)
                            raise RuntimeError('Model call failed; retain ledger and inspect before resume') from exc
                        answer=''.join(parts)
                        usage={'prompt_tokens':len(ids),'completion_tokens':last.generation_tokens,
                               'total_tokens':len(ids)+last.generation_tokens,'cached_prompt_tokens':len(ids)-1}
                        ledger['new_model_tokens']+=usage['total_tokens']
                        ledger['physical_completion_tokens']+=usage['completion_tokens']
                        seconds=time.monotonic()-start
                        if step==0:
                            # Charge a prompt build once per logical episode,
                            # even when physical states are shared across seeds.
                            seconds+=prefixes[(arm,tuple(ids))]['seconds']
                    h=assess(ev,d['metadata'],task,answer,history)
                    if active:
                        assert h['in_space'] and h['key'] not in seen
                    r={'contract':contract,'step':step,'task':task,'prompt':prompt,'raw_response':answer,
                       'outcome':h,'usage':usage,'seconds':seconds,'request_seed':request_seed,'error':None,
                       'replay':replay,'decoder':{'active':active,'distinct_before':len(seen),'target_distinct':13}}
                    with path.open('a') as f:
                        f.write(json.dumps(r,ensure_ascii=False)+'\n')
                    write_json(ledger_path,ledger)
                    rows.append(r);history.append(h)
                if bace:
                    assert len({r['outcome']['key'] for r in rows if r['outcome']['in_space']})>=13
            print(f'Completed preference {split} task {tid} seed {seed}; new calls {ledger["new_model_calls"]}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('split',choices=['validation','test'])
    evaluate(parser.parse_args().split)
