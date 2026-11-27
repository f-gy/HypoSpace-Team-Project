"""Canonical-group DPO with observation-violating, length-matched negatives.

This is a matched continuation study from the immutable v1 coverage adapter.
Only the original 16 training observation sets supply preference labels.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import random
import time

import numpy as np

from .canonical_decode import completion_tokens, grammar_support
from .coverage_lora import LORA, RUN as V1, batch, coverage_value, dataset, model_load
from .experiment import assess
from .tasks import write_json
from .upstream import ROOT, evaluator, load as upstream_load

RUN = ROOT / 'runs/canonical_preference_v5'
OUT = ROOT / 'results/canonical_preference_v5'
BETA = .5
UPDATES = 80


def group_margin(logp):
    import mlx.core as mx
    return mx.logsumexp(logp[:2]) - mx.logsumexp(logp[2:])


def preference_loss(logp, reference_margin):
    import mlx.core as mx
    return mx.logaddexp(mx.array(0.), -BETA * (group_margin(logp) - reference_margin))


def answer_logits(model, ids, start):
    """Qwen backbone on the whole prompt; project only loss-bearing positions."""
    hidden = model.model(ids[:, :-1])[:, start:, :]
    return model.model.embed_tokens.as_linear(hidden) if model.args.tie_word_embeddings else model.lm_head(hidden)


def loss_function(use_preference, response_only=True):
    import mlx.core as mx
    import mlx.nn as nn
    def loss(model, ids, mask, ratio, reference_margin, start):
        if response_only:
            logits = answer_logits(model, ids, start)
            targets, active_mask = ids[:, start+1:], mask[:, start:]
        else:
            logits = model(ids[:, :-1])
            targets, active_mask = ids[:, 1:], mask
        ce = nn.losses.cross_entropy(logits, targets, reduction='none').astype(mx.float32) * active_mask
        nll = ce.sum(axis=1)
        anchor = (nll[:2] / active_mask[:2].sum(axis=1)).mean()
        value = anchor - coverage_value(-nll[:2], ratio, 'canonical_coverage')
        if use_preference:
            value += preference_loss(-nll, reference_margin)
        return value
    return loss


def prepare():
    if (RUN / 'training_data.json').exists():
        raise RuntimeError('Preserve existing prepared training data')
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.tuner.utils import linear_to_lora_layers
    d = dataset()
    train = json.loads((V1 / 'training_data.json').read_text())
    assert set(train['training_ids']) == set(d['splits']['train'])
    tasks = {t['id']: t for t in d['tasks'] if t['id'] in d['splits']['train']}
    model, tok = model_load()
    model.freeze()
    linear_to_lora_layers(model, 8, LORA)
    model.load_weights(list(mx.load(str(V1 / 'adapters/canonical_coverage/adapters.safetensors')).items()), strict=False)
    model.eval()
    support = completion_tokens(tok, grammar_support(d['metadata']))
    ev = evaluator(d['metadata'])
    mod, _ = upstream_load()
    negatives = {}
    for tid, task in tasks.items():
        options = []
        for h in support:
            expr = h['answer'].removeprefix('Expression: ')
            parsed = mod.BooleanExpression(expr, d['metadata']['variables'], set(d['metadata']['operators']))
            violations = sum(parsed.evaluate(o['inputs']) != o['output'] for o in task['observations'])
            if violations == 1:
                options.append(h)
        negatives[tid] = options
    groups = []
    for source in train['groups']:
        assert len(source['aliases']) == 2
        tid = source['task_id']
        positive = source['aliases'][0]
        prompt_len = positive['prompt_length']
        positive_len = len(positive['tokens']) - prompt_len
        candidates = negatives[tid]
        if not candidates:
            raise RuntimeError('No one-observation hard negatives')
        chosen = min(candidates, key=lambda h: (abs(len(h['tokens'])-positive_len),
                         -difflib.SequenceMatcher(None, positive['answer'], h['answer']).ratio(), h['answer']))
        aliases = []
        for answer in [chosen['answer'], 'Expression: (' + chosen['answer'].removeprefix('Expression: ') + ')']:
            h = assess(ev, d['metadata'], tasks[tid], answer, [])
            assert h['in_space'] and not h['valid'] and h['key'] == chosen['key']
            tail = completion_tokens(tok, [{'answer': answer, 'key': chosen['key']}])[0]['tokens']
            aliases.append({'answer': answer, 'tokens': positive['tokens'][:prompt_len] + tail, 'prompt_length': prompt_len})
        group = {**source, 'aliases': source['aliases'] + aliases,
                 'negative_key': chosen['key'], 'negative_observation_violations': 1,
                 'negative_token_length_difference': len(aliases[0]['tokens']) - len(positive['tokens'])}
        ids, mask, ratio = batch(group)
        logits = model(ids[:, :-1])
        ce = nn.losses.cross_entropy(logits, ids[:, 1:], reduction='none').astype(mx.float32) * mask
        margin = group_margin(-ce.sum(axis=1))
        mx.eval(margin)
        group['reference_margin'] = float(margin.item())
        groups.append(group)
    write_json(RUN / 'training_data.json', {'groups': groups, 'training_ids': sorted(tasks),
                                           'reference_adapter_sha256': hashlib.sha256((V1/'adapters/canonical_coverage/adapters.safetensors').read_bytes()).hexdigest()})
    plan = {'version': 'canonical_group_preference_v5', 'updates_per_arm': UPDATES,
            'arms': ['coverage_continuation', 'canonical_preference'], 'initial_adapter': 'v1 canonical_coverage',
            'positive_aliases': 2, 'negative_aliases': 2, 'microbatches_per_update': 4,
            'learning_rate': .00005, 'dpo_beta': BETA, 'preference_weight': 1.,
            'training_seed': 4305, 'training_tasks': sorted(tasks), 'training_class_pairs': len(groups),
            'negative_construction': 'Exactly one training observation violation; first minimize complete-token length difference, then maximize spelling similarity; two same-class spellings.',
            'loss': 'positive per-token SFT - C16(positive canonical mass) + optional canonical-group DPO; reference margins frozen from common initial adapter',
            'budgets': 'Both arms receive identical four-response batches and update schedule; only the preference term differs. Adam starts from zero in both arms.',
            'selection_rule': 'Use final fixed-update checkpoints. Evaluate both IID and BACE on validation; retain all outcomes. No test-based hyperparameter tuning.',
            'sources': {p: hashlib.sha256((ROOT/'explorer'/p).read_bytes()).hexdigest() for p in ['canonical_preference.py', 'canonical_decode.py']}}
    write_json(OUT/'protocol.json', plan)
    lengths = [g['negative_token_length_difference'] for g in groups]
    write_json(OUT/'training_data_audit.json', {'pairs': len(groups), 'training_tasks': len(tasks),
                                              'all_negatives_violate_exactly_one_observation': True,
                                              'mean_absolute_negative_length_difference': float(np.mean(np.abs(lengths))),
                                              'max_absolute_negative_length_difference': max(map(abs,lengths)),
                                              'uses_validation_or_test_labels': False,
                                              'training_data_sha256': hashlib.sha256((RUN/'training_data.json').read_bytes()).hexdigest()})
    print(json.dumps({'prepared_training_pairs': len(groups), 'training_tasks': len(tasks)}), flush=True)


def train():
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx_lm.tuner.trainer import grad_checkpoint
    data = json.loads((RUN/'training_data.json').read_text())
    plan = json.loads((OUT/'protocol.json').read_text())
    for name, digest in plan['sources'].items():
        assert hashlib.sha256((ROOT/'explorer'/name).read_bytes()).hexdigest() == digest
    if (RUN/'training_log.jsonl').exists():
        raise RuntimeError('Training already started; preserve run rather than overwrite')
    mx.random.seed(4305)
    model, _ = model_load()
    model.freeze()
    linear_to_lora_layers(model, 8, LORA)
    model.load_weights(list(mx.load(str(V1/'adapters/canonical_coverage/adapters.safetensors')).items()), strict=False)
    grad_checkpoint(model.layers[0])
    model.train()
    initial = {n: mx.array(v) for n,v in tree_flatten(model.trainable_parameters())}
    frozen = {n:v for n,v in tree_flatten(model.parameters()) if not n.endswith(('lora_a','lora_b'))}
    grouped = {}
    for g in data['groups']:
        grouped.setdefault(g['task_id'], []).append(g)
    rng = random.Random(4305)
    schedule = [[rng.choice(grouped[rng.choice(sorted(grouped))]) for _ in range(4)] for _ in range(UPDATES)]
    summaries = {}
    for arm in plan['arms']:
        model.load_weights(list(initial.items()), strict=False)
        optimizer = optim.Adam(learning_rate=.00005)
        fn = nn.value_and_grad(model, loss_function(arm == 'canonical_preference'))
        elapsed, total_tokens = 0., 0
        for step, examples in enumerate(schedule):
            start = time.monotonic()
            gradients, values = None, []
            for g in examples:
                ids, mask, ratio = batch(g)
                value, grad = fn(model, ids, mask, ratio, mx.array(g['reference_margin']),
                                 g['aliases'][0]['prompt_length']-1)
                gradients = grad if gradients is None else tree_map(lambda a,b:a+b, gradients, grad)
                mx.eval(value, gradients)
                values.append(float(value.item()))
                total_tokens += sum(len(a['tokens']) for a in g['aliases'])
            gradients = tree_map(lambda a:a/4, gradients)
            assert all(bool(mx.all(mx.isfinite(v)).item()) for _,v in tree_flatten(gradients))
            optimizer.update(model, gradients)
            mx.eval(model.parameters(), optimizer.state)
            seconds = time.monotonic() - start
            elapsed += seconds
            row = {'arm': arm, 'step': step+1, 'loss': float(np.mean(values)), 'seconds': seconds,
                   'sequence_tokens_so_far': total_tokens, 'peak_memory_gb': mx.get_peak_memory()/1e9}
            assert math.isfinite(row['loss'])
            with (RUN/'training_log.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
            if (step+1)%5 == 0:
                print(json.dumps(row), flush=True)
        path = RUN/'adapters'/arm
        path.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(path/'adapters.safetensors'), dict(tree_flatten(model.trainable_parameters())))
        write_json(path/'adapter_config.json', {'fine_tune_type':'lora','num_layers':8,'lora_parameters':LORA})
        summaries[arm] = {'updates':UPDATES, 'sequence_tokens': total_tokens, 'seconds': elapsed,
                          'adapter_sha256':hashlib.sha256((path/'adapters.safetensors').read_bytes()).hexdigest()}
    current = dict(tree_flatten(model.parameters()))
    assert all(bool(mx.all(v == current[n]).item()) for n,v in frozen.items())
    write_json(OUT/'training_summary.json', {'arms':summaries, 'base_weights_unchanged':True,
                                            'new_trainable_parameters':0,
                                            'trainable_parameters':sum(v.size for v in initial.values()),
                                            'peak_memory_gb':mx.get_peak_memory()/1e9})
    (OUT/'training_log.jsonl').write_text((RUN/'training_log.jsonl').read_text())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare','train'])
    args = parser.parse_args()
    {'prepare': prepare, 'train': train}[args.command]()
