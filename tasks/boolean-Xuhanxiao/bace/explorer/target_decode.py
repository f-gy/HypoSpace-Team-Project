"""Budget-aware minimal intervention toward a predeclared uniqueness target.

Keep the original IID proposal until another duplicate would make the target
unreachable. Then exclude prior canonical classes at the token level. The
original IID prefix can be replayed exactly from its immutable cached run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time

from .canonical_decode import SupportTrie, TrieLogitsProcessor, completion_tokens, grammar_support, support_summary
from .coverage_lora import LORA, RUN as V1, dataset, model_load, prompt_for
from .experiment import assess, read_rows
from .tasks import public_task, write_json
from .upstream import ROOT, evaluator

RUN = ROOT / 'runs/target_decode_v4'
OUT = ROOT / 'results/target_decode_v4'
METHODS = ['cc_target_class', 'cc_target_string', 'sft_target_class', 'base_target_class', 'uniform_target_class']
ORIGINAL = {'cc': 'canonical_coverage_iid', 'sft': 'balanced_sft_iid', 'base': 'base_iid'}


def intervene(seen_count, completed, budget=16, target=13):
    return budget - completed <= max(0, target - seen_count)


def initialize():
    d = dataset()
    support = grammar_support(d['metadata'])
    cfg = {'version': 'budget_aware_canonical_intervention_v4', 'budget': 16, 'target_distinct': 13,
           'target_reason': 'Smallest integer exceeding the README .80 guide at N=16; fixed before this pilot',
           'methods': METHODS, 'seeds': {'validation': [101], 'test': [17, 29, 43]},
           'temperature': 1., 'top_p': 1., 'max_output_tokens': 128,
           'support': support_summary(support),
           'test_status': 'Follow-up on previously published v1 test observations; not untouched confirmation',
           'selection_rule': 'Report all arms and VR/NR/RR. Treat a result with more than .02 absolute VR loss versus v1 canonical as a trade-off option, not a dominance improvement. Run full follow-up only if validation reaches NR>.8, increases RR versus v1 canonical, and loses at most .02 VR.',
           'cache_replay': 'Exact original IID prefix only, before any intervention, same prompt/checkpoint/request seed. Replayed calls count in the logical 16-call budget and token cost; physical new calls and reuse separately reported.',
           'proposal_boundary': 'Only syntax/canonical keys from grammar; no reference labels, observation checking, or repair in proposal',
           'sources': {p: hashlib.sha256((ROOT / 'explorer' / p).read_bytes()).hexdigest()
                       for p in ['canonical_decode.py', 'target_decode.py']},
           'adapters': {n: hashlib.sha256((V1 / 'adapters' / n / 'adapters.safetensors').read_bytes()).hexdigest()
                        for n in ['canonical_coverage', 'balanced_sft']}}
    path = OUT / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == cfg
    else:
        write_json(path, cfg)
    return d, support, cfg


def evaluate(split):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.tuner.utils import linear_to_lora_layers
    RUN.mkdir(parents=True, exist_ok=True)
    d, support, cfg = initialize()
    model, tok = model_load()
    model.freeze()
    linear_to_lora_layers(model, 8, LORA)
    model.eval()
    weights = {'base': {n: mx.zeros_like(v) for n, v in tree_flatten(model.trainable_parameters())}}
    for prefix, name in [('cc', 'canonical_coverage'), ('sft', 'balanced_sft')]:
        weights[prefix] = mx.load(str(V1 / 'adapters' / name / 'adapters.safetensors'))
    support = completion_tokens(tok, support)
    by_key = {}
    for h in support:
        by_key.setdefault(h['key'], []).append(h)
    tasks = {t['id']: t for t in d['tasks']}
    ev = evaluator(d['metadata'])
    fingerprint = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    ledger_path = RUN / 'generation_ledger.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {'new_model_calls': 0, 'replayed_model_calls': 0, 'uniform_draws': 0, 'new_model_tokens': 0, 'replayed_model_tokens': 0}
    for tid in d['splits'][split]:
        task = public_task(tasks[tid])
        for seed in cfg['seeds'][split]:
            # Upstream formats an operator set without sorting. Its order can
            # change across Python processes. Use the immutable original text
            # so a replayed IID prefix and new masked continuation match exactly.
            template_rows = read_rows(V1 / f'{split}_canonical_coverage_iid_{tid}_{seed}.jsonl')
            assert len(template_rows) == 16
            prompt = template_rows[0]['prompt']
            ids = tok.apply_chat_template([{'role': 'user', 'content': prompt}], add_generation_prompt=True, tokenize=True)
            order = METHODS.copy()
            random.Random(seed + int(tid, 16)).shuffle(order)
            for method in order:
                prefix = method.split('_')[0]
                path = RUN / f'{split}_{method}_{tid}_{seed}.jsonl'
                rows = read_rows(path)
                contract = {'fingerprint': fingerprint, 'method': method, 'task_id': tid, 'split': split, 'seed': seed, 'budget': 16}
                assert all(r['contract'] == contract and r['step'] == i for i, r in enumerate(rows))
                history = [r['outcome'] for r in rows]
                original_path = V1 / f'{split}_{ORIGINAL[prefix]}_{tid}_{seed}.jsonl' if prefix in ORIGINAL else None
                original = read_rows(original_path) if original_path else []
                if original:
                    assert len(original) == 16
                if prefix in weights and len(rows) < 16:
                    model.load_weights(list(weights[prefix].items()), strict=False)
                    mx.eval(model.parameters())
                for step in range(len(rows), 16):
                    seen = {h['key'] for h in history if h['in_space']}
                    active = intervene(len(seen), step)
                    start = time.monotonic()
                    root_seed = int.from_bytes(hashlib.sha256(f'{seed}:{step}:{tid}'.encode()).digest()[:8], 'big')
                    request_seed = random.Random(root_seed).getrandbits(31)
                    replay = None
                    if prefix == 'uniform':
                        rng = random.Random(request_seed)
                        eligible = sorted(set(by_key) - seen if active else set(by_key))
                        key = rng.choice(eligible)
                        answer = rng.choice(by_key[key])['answer']
                        usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
                        seconds = time.monotonic() - start
                        ledger['uniform_draws'] += 1
                    elif not active:
                        assert not any(r['decoder']['active'] for r in rows)
                        r0 = original[step]
                        assert r0['prompt'] == prompt and r0['request_seed'] == request_seed and r0['error'] is None
                        answer, usage, seconds = r0['raw_response'], r0['usage'], r0['seconds']
                        replay = {'source_file': original_path.name, 'source_step': step,
                                  'source_file_sha256': hashlib.sha256(original_path.read_bytes()).hexdigest()}
                        ledger['replayed_model_calls'] += 1
                        ledger['replayed_model_tokens'] += usage['total_tokens']
                    else:
                        blocked_keys = seen if method.endswith('_class') else set()
                        blocked_strings = {r['raw_response'] for r in rows} if method.endswith('_string') else set()
                        trie = SupportTrie(support, blocked_keys, blocked_strings)
                        processor = TrieLogitsProcessor(trie, len(ids))
                        mx.random.seed(request_seed)
                        ledger['new_model_calls'] += 1
                        write_json(ledger_path, ledger)
                        parts, last = [], None
                        for last in stream_generate(model, tok, ids, max_tokens=128,
                                                    sampler=make_sampler(temp=1., top_p=1., top_k=0, min_p=0.),
                                                    logits_processors=[processor]):
                            parts.append(last.text)
                        answer = ''.join(parts)
                        usage = {'prompt_tokens': len(ids), 'completion_tokens': last.generation_tokens,
                                 'total_tokens': len(ids) + last.generation_tokens}
                        seconds = time.monotonic() - start
                        ledger['new_model_tokens'] += usage['total_tokens']
                    h = assess(ev, d['metadata'], task, answer, history)
                    if active and method.endswith('_class'):
                        assert h['in_space'] and h['key'] not in seen
                    r = {'contract': contract, 'step': step, 'task': task, 'prompt': prompt,
                         'raw_response': answer, 'outcome': h, 'usage': usage, 'seconds': seconds,
                         'request_seed': request_seed, 'error': None, 'replay': replay,
                         'decoder': {'active': active, 'distinct_before': len(seen), 'remaining_before': 16-step,
                                     'target_distinct': 13, 'exclusion': 'class' if method.endswith('_class') else 'string'}}
                    with path.open('a') as f:
                        f.write(json.dumps(r, ensure_ascii=False) + '\n')
                    write_json(ledger_path, ledger)
                    rows.append(r)
                    history.append(h)
                if method.endswith('_class'):
                    assert len({r['outcome']['key'] for r in rows if r['outcome']['in_space']}) >= 13
            print(f'Completed target {split} {tid} seed {seed}; new model calls {ledger["new_model_calls"]}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'validation', 'test'])
    args = parser.parse_args()
    if args.command == 'prepare':
        initialize()
    else:
        evaluate(args.command)
