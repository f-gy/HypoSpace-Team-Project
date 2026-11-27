"""CPU-only replay of COVE evidence, rankings, labels, budgets and provenance."""
from __future__ import annotations

import gzip
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from explorer.coverage_lora import dataset
from explorer.experiment import assess
from explorer.posterior_bank import bank, exact_tree_values, numpy_predictions, observation_log_likelihood
from explorer.tasks import prepare, public_task, write_json
from explorer.upstream import evaluator, load

OUT = ROOT / 'results/all80'


def rows(path):
    with gzip.open(path, 'rt') as f:
        return [json.loads(line) for line in f]


def main():
    if not (ROOT / '.cache/tasks.json').exists():
        prepare(json.loads((ROOT / 'config.json').read_text()))
    d = dataset()
    tasks = {t['id']: t for t in d['tasks']}
    ev = evaluator(d['metadata'])
    module, _ = load()
    items = bank()
    assert json.loads(json.dumps(items)) == json.loads((OUT / 'bank.json').read_text())
    keys = [h['key'] for h in items]
    assert len(keys) == len(set(keys)) == 33
    train_protocol = json.loads((OUT / 'protocol.json').read_text())
    assert set(train_protocol['training_tasks']) == set(d['splits']['train'])
    assert len(d['splits']['train']) == 16
    for a, b in [('train', 'test'), ('train', 'validation'), ('validation', 'test')]:
        assert not set(d['splits'][a]) & set(d['splits'][b])
    # These exact semantics are diagnostic only; ranking COVE below uses learned weights.
    bits = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    exact = np.stack([exact_tree_values(h['tree'], bits) for h in items])
    for i, h in enumerate(items):
        for alias in h['aliases']:
            expression = ev.parse_llm_response(alias)
            assert str(ev.get_expression_mechanistic_key(expression)) == h['key']
            parsed = module.BooleanExpression(expression, d['metadata']['variables'], set(d['metadata']['operators']))
            values = [parsed.evaluate(dict(zip(('x', 'y'), map(int, v)))) for v in bits]
            np.testing.assert_array_equal(values, exact[i])
    for tid in d['splits']['train']:
        task = tasks[tid]
        compatible = np.ones(33, dtype=bool)
        for o in task['observations']:
            compatible &= exact[:, 2 * o['inputs']['x'] + o['inputs']['y']] == o['output']
        np.testing.assert_array_equal(compatible, [k in task['reference_keys'] for k in keys])
    weights = {s: json.loads((OUT / 'executors' / str(s) / 'weights.json').read_text()) for s in [4306, 4307, 4308]}
    predictions = {s: numpy_predictions(w, items) for s, w in weights.items()}
    for w in weights.values():
        assert sum(np.asarray(v).size for net in w.values() for v in net.values()) == 179
    original_hashes = json.loads((ROOT / 'results/final/source_manifest.json').read_text())['source_sha256']
    for p, expected in original_hashes.items():
        assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == expected, p
    for n, expected in train_protocol['source_sha256'].items():
        assert hashlib.sha256((ROOT / 'explorer' / n).read_bytes()).hexdigest() == expected, n
    extra = [json.loads(line) for line in (OUT / 'iid_extras.jsonl').read_text().splitlines()]
    assert len(extra) == 42
    sources = {}
    for split in ['validation', 'test']:
        for folder, old, new in [('final', 'base_iid', 'base_iid'), ('canonical_preference_v5', 'coverage_continuation_iid', 'frozen_iid')]:
            for r in rows(ROOT / 'results' / folder / split / 'evidence.jsonl.gz'):
                c = r['contract']
                if c['method'] == old:
                    assert r['error'] is None
                    sources[(split, new, c['task_id'], c['seed'], r['step'])] = r
    for r in extra:
        assert r['error'] is None and r['step'] >= 16
        number = int.from_bytes(hashlib.sha256(f"{r['seed']}:{r['step']}:{r['task_id']}".encode()).digest()[:8], 'big')
        assert r['request_seed'] == random.Random(number).getrandbits(31)
        sources[('test', r['method'], r['task_id'], r['seed'], r['step'])] = r
    totals = {}
    for split in ['validation', 'test']:
        protocol = json.loads((OUT / f'evaluation_protocol_{split}.json').read_text())
        for n, expected in protocol['source_sha256'].items():
            assert hashlib.sha256((ROOT / 'explorer' / n).read_bytes()).hexdigest() == expected, n
        evidence = rows(OUT / split / 'evidence.jsonl.gz')
        groups = defaultdict(list)
        for r in evidence:
            assert r['split'] == split and r['task_id'] in d['splits'][split]
            assert r['seed'] in protocol['seeds']
            assert r['task'] == public_task(tasks[r['task_id']])
            assert set(r['task']) == {'id', 'observations'}
            groups[(r['protocol'], r['method'], r['task_id'], r['seed'])].append(r)
        expected_groups = len(d['splits'][split]) * len(protocol['seeds']) * len(protocol['methods']) * 2
        assert len(groups) == expected_groups
        metrics = defaultdict(list)
        checked = 0
        outside = defaultdict(int)
        for (setting, method, tid, seed), episode in groups.items():
            task = public_task(tasks[tid])
            refs = set(tasks[tid]['reference_keys'])
            n = len(refs) if setting == 'adaptive_1x' else 16
            assert len(episode) == n
            score = json.loads((OUT / 'lm_scores' / f'{split}_{tid}.json').read_text())
            assert score['class_keys'] == keys
            assert score['adapter_sha256'] == hashlib.sha256((ROOT / 'artifacts/preference_adapters/coverage_continuation/adapters.safetensors').read_bytes()).hexdigest()
            lp = np.asarray(score['alias_log_probabilities']).reshape(33, 2)
            np.testing.assert_allclose(np.logaddexp(lp[:, 0], lp[:, 1]), score['class_log_mass'], atol=1e-12)
            sampling_seed = int.from_bytes(hashlib.sha256(f'{seed}:{tid}:catalog'.encode()).digest()[:8], 'big')
            learned = {s: observation_log_likelihood(p, task['observations']) for s, p in predictions.items()}
            if method == 'llm_catalog_wor':
                logweights = np.asarray(score['class_log_mass'])
            elif method.startswith('cove_llm_'):
                logweights = np.asarray(score['class_log_mass']) + 10 * learned[int(method.rsplit('_', 1)[1])]
            elif method == 'cove_only_4306':
                logweights = 10 * learned[4306]
            elif method == 'uniform_wor':
                logweights = np.zeros(33)
            elif method == 'symbolic_wor':
                valid = np.ones(33, dtype=bool)
                for o in task['observations']:
                    valid &= exact[:, 2 * o['inputs']['x'] + o['inputs']['y']] == o['output']
                logweights = 1000 * valid.astype(float)
            else:
                assert method in ['base_iid', 'frozen_iid']
                logweights = None
            if logweights is not None:
                noise = np.random.default_rng(sampling_seed).gumbel(size=33)
                ordering = sorted(range(33), key=lambda j: (-float(logweights[j] + noise[j]), j))
            history, seen, hits = [], set(), set()
            count = 0
            for step, r in enumerate(sorted(episode, key=lambda v: v['step'])):
                assert r['step'] == step and r['budget'] == n and r['sampling_seed'] == sampling_seed
                if logweights is None:
                    assert r['raw_response'] == sources[(split, method, tid, seed, step)]['raw_response']
                else:
                    j = ordering[step]
                    assert r['proposal']['class_index'] == j
                    assert r['raw_response'] == items[j]['aliases'][0]
                    assert r['proposal']['uses_exact_constraint_solver'] == (method == 'symbolic_wor')
                    if method.startswith(('cove_llm_', 'cove_only_')):
                        np.testing.assert_allclose(r['proposal']['learned_log_likelihood'], learned[int(method.rsplit('_', 1)[1])][j])
                h = assess(ev, d['metadata'], task, r['raw_response'], history)
                assert h == r['outcome']
                if h['in_space']:
                    seen.add(h['key'])
                if h['valid']:
                    count += 1
                    hits.add(h['key'])
                history.append(h)
                checked += 1
            outside[method] += len(hits - refs)
            if logweights is not None:
                assert not hits - refs, ('catalog reference-external artifact', split, method, tid)
            metrics[(setting, method)].append([count / n, len(seen) / n, len(hits & refs) / len(refs)])
        summary = json.loads((OUT / split / 'summary.json').read_text())
        for (setting, method), values in metrics.items():
            expected = [summary['metrics'][setting][method][k] for k in ['VR', 'NR', 'RR']]
            np.testing.assert_allclose(np.mean(values, axis=0), expected, atol=1e-12, rtol=0)
        assert checked == summary['outputs']
        totals[split] = {'outputs_rescored': checked, 'episodes': len(groups), 'rankings_and_metrics_match': True,
                         'outside_reference_distinct_hits_summed_over_both_protocols': dict(outside)}
        print('PASS', split, checked, 'outputs; independent rankings and official score replay', flush=True)
    original = json.loads((ROOT / 'results/final/test/summary.json').read_text())
    # Different schema in the inherited report: extract named metrics directly.
    old_base = original['metrics']['base_iid']
    test = json.loads((OUT / 'test/summary.json').read_text())
    for k in ['VR', 'NR', 'RR']:
        assert abs(test['metrics']['fixed16']['base_iid'][k] - old_base[k]) < 1e-12
    old_frozen = json.loads((ROOT / 'results/canonical_preference_v5/test/summary.json').read_text())['metrics']['coverage_continuation_iid']
    for k in ['VR', 'NR', 'RR']:
        assert abs(test['metrics']['fixed16']['frozen_iid'][k] - old_frozen[k]) < 1e-12
    result = {'pass': True, 'splits': totals, 'all_original_source_hashes_match': True,
              'train_only_inventory_and_training_ids': True, 'training_compatibility_labels_checked': 528,
              'classes': 33, 'executor_parameters': 179,
              'no_catalog_reference_external_validity_artifacts': True, 'new_iid_calls': len(extra),
              'learned_sampler_replayed_without_exact_execution': True,
              'scope': 'CPU checks saved logits and learned weights; three on-device teacher-forcing checks separately recorded. Does not retrain or repeat all LLM forward passes.'}
    write_json(OUT / 'verification.json', result)
    print('PASS: public COVE evidence, independent score replay, frozen source and budgets')


if __name__ == '__main__':
    main()
