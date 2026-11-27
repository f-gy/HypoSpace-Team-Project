"""Post-inference scoring. This process may read gold; the runner may not."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import statistics
import sys

import networkx as nx
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'code'))
sys.path.insert(0, str(ROOT / 'vendor/HypoSpace/causal'))
from prepare import SOURCES
import sgve_core as c
from boolean_dataset import BooleanObservation
from hypospace_validator import generated_universe
from modules.models import CausalGraph
from generate_causal_dataset import CausalDatasetGenerator as CG

spec = importlib.util.spec_from_file_location('official_voxel', str(ROOT / 'vendor/HypoSpace/3d/generate_3d_dataset_complete.py'))
voxel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(voxel)
base = c.BooleanBenchmarkRefined(str(ROOT / 'judge_training_v1/validator.json'))
boolean = c.install(base, policy='enumerated', universe=generated_universe(base))


def check_key(task, query, obj):
    if task == 'causal':
        graph = CausalGraph(query['nodes'], [tuple(e) for e in obj['edges']])
        assert nx.is_directed_acyclic_graph(graph.to_networkx())
        assert all(CG.get_perturbation_effects(graph, o['perturbed_node']).effects == o['effects']
                   for o in query['observations'])
        return graph.get_hash()
    if task == 'boolean':
        formula = obj['formula']
        obs = [BooleanObservation(o['inputs'], o['output']) for o in query['observations']]
        assert boolean.validate_expression(formula, obs)[0]
        return repr(boolean.get_expression_mechanistic_key(formula))
    top = np.array(query['top_view'])
    heights = obj['heights']
    coords = list(zip(*np.where(top == 1)))
    assert len(heights) == len(coords) and all(type(h) is int and 1 <= h <= query['max_height'] for h in heights)
    layers = [np.zeros_like(top) for _ in range(max(heights))]
    for (row, col), height in zip(coords, heights):
        for level in range(height):
            layers[level][row, col] = 1
    structure = voxel.Structure3D(layers)
    assert np.array_equal(structure.get_top_view(), top)
    assert all(np.all(layers[i] <= layers[i-1]) for i in range(1, len(layers)))
    return json.dumps(obj, sort_keys=True, separators=(',', ':'))


def rates(rows, budget):
    emitted = sum(r['emitted'] for r in rows)
    valid = sum(r['valid_count'] for r in rows)
    unique = sum(r['distinct_valid'] for r in rows)
    recoveries = [r['recovery'] for r in rows if r['recovery'] is not None]
    return {
        'questions': len(rows), 'emitted': emitted, 'unique_valid': unique,
        'repeated_outputs': emitted - unique,
        'precision_emitted': valid / emitted if emitted else None,
        'uniqueness_emitted': unique / emitted if emitted else None,
        'unique_valid_per_budget': unique / budget if budget else None,
        'macro_recovery': statistics.mean(recoveries) if recoveries else None,
        'recovery_defined_questions': len(recoveries),
        'empty_solution_questions': len(rows) - len(recoveries),
    }


def summarize_task(rows, policy):
    reference = 6 * len(rows)
    allocated = sum(r['emitted'] for r in rows) if policy == 'shared_round_robin' else reference
    return {'reference_output_slots': reference, 'allocated_output_slots': allocated,
            **rates(rows, allocated)}


def score_case(task, query, result, gold, policy, split='', order='native'):
    keys = [check_key(task, query, obj) for obj in result['outputs']]
    found = set(keys)
    assert found <= gold
    assert len(found) == result['unique_count']
    assert len(keys) - len(found) == result['repeats']
    if result['complete']:
        assert found == gold
    if policy != 'fixed_repeat':
        assert len(keys) == len(found)
    return {'split': split, 'order': order, 'policy': policy,
            'id': result['id'], 'task': task, 'space_size_evaluator_only': len(gold),
            'emitted': len(keys), 'valid_count': len(keys), 'distinct_valid': len(found),
            'precision_emitted': 1.0 if keys else None,
            'uniqueness_emitted': len(found) / len(keys) if keys else None,
            'reference6_validity': len(keys) / 6 if policy != 'shared_round_robin' else None,
            'reference6_uniqueness': len(found) / 6 if policy != 'shared_round_robin' else None,
            'recovery': len(found) / len(gold) if gold else None, 'mechanism_keys': keys,
            'complete': result['complete'], 'reason': result['reason']}


def verify_frozen_success(rows, summaries, frozen):
    """Success criteria for the frozen experiment, separate from scoring."""
    assert len(summaries) == 54 and len(rows) == 720
    for r in rows:
        size = r['space_size_evaluator_only']
        if r['policy'] in {'fixed_repeat', 'fixed_stop'}:
            assert r['distinct_valid'] == min(6, size)
            expected = (6 if size else 0) if r['policy'] == 'fixed_repeat' else min(6, size)
            assert r['emitted'] == expected
        else:
            assert r['distinct_valid'] >= min(6, size), 'shared allocation lost fixed coverage'
        if r['policy'] == 'fixed_repeat':
            old = frozen[(r['split'], r['id'])]
            assert math.isclose(r['reference6_validity'], old['validity'])
            assert math.isclose(r['reference6_uniqueness'], old['uniqueness'])
            assert math.isclose(r['recovery'], old['recovery'])
    for batch in summaries:
        subset = [r for r in rows if (r['split'], r['order'], r['policy']) ==
                  (batch['split'], batch['order'], batch['policy'])]
        if batch['policy'] == 'shared_round_robin':
            assert batch['emitted'] == min(batch['budget'], sum(r['space_size_evaluator_only'] for r in subset))
    for split in SOURCES:
        for policy in ['fixed_repeat', 'fixed_stop', 'shared_round_robin']:
            variants = [r for r in summaries if r['split'] == split and r['policy'] == policy]
            for metric in ['emitted', 'unique_valid', 'macro_recovery']:
                assert all(math.isclose(r[metric], variants[0][metric]) for r in variants)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, type=pathlib.Path)
    parser.add_argument('--require-frozen-success', action='store_true')
    args = parser.parse_args()
    out = args.run_dir.resolve()
    queries = json.loads((out / 'query_only_inputs.json').read_bytes())
    inference = json.loads((out / 'inference.json').read_bytes())
    protocol = json.loads((out / 'protocol.json').read_bytes())
    for relative, digest in protocol['code_sha256'].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest
    assert inference['only_query_file_read'] and inference['llm_calls'] == 0
    assert hashlib.sha256((out / 'query_only_inputs.json').read_bytes()).hexdigest() == inference['query_input_sha256'] == protocol['query_input_sha256']
    cases = {}
    for split, relative in SOURCES.items():
        data = (ROOT / relative).read_bytes()
        assert hashlib.sha256(data).hexdigest() == protocol['source_sha256'][relative]
        cases[split] = {r['id']: r for r in json.loads(data)}
    frozen = {(r['split'], r['id']): r for r in json.load(open(ROOT / 'budget_v1/frozen_fixed6_reference.json'))}
    batch_summaries, task_summaries, rows = [], [], []
    for batch in inference['batches']:
        split, policy, order = batch['split'], batch['policy'], batch['order']
        qmap = {r['id']: r for r in queries[split]}
        scored = []
        assert batch['budget'] == 6 * len(qmap)
        assert len(batch['cases']) == len(qmap)
        assert [r['id'] for r in batch['cases']] == batch['question_order']
        assert set(batch['question_order']) == set(qmap)
        for result in batch['cases']:
            case = cases[split][result['id']]
            query = qmap[result['id']]['query']
            r = score_case(case['task'], query, result, set(case['gold']), policy, split, order)
            scored.append(r)
        emitted = sum(r['emitted'] for r in scored)
        assert emitted == batch['spent'] and batch['unspent'] == batch['budget'] - emitted
        summary = {'split': split, 'policy': policy, 'order': order,
                   'budget': batch['budget'], 'unspent': batch['unspent'],
                   'seconds': batch['seconds'], 'termination': batch['termination'],
                   **rates(scored, batch['budget']),
                   **{field: sum(q[field] for q in batch['cases']) for field in ['solver_checks', 'predicate_checks', 'construction_steps']}}
        batch_summaries.append(summary)
        for task in ['causal', 'voxel', 'boolean']:
            subset = [r for r in scored if r['task'] == task]
            if subset:
                task_summaries.append({'split': split, 'policy': policy, 'order': order,
                                       'task': task, **summarize_task(subset, policy)})
        rows.extend(scored)
    for filename, data in [('scored.json', rows), ('batch_summary.json', batch_summaries),
                           ('task_summary.json', task_summaries)]:
        with (out / filename).open('x') as handle:
            json.dump(data, handle, indent=2)
    # Keep observed scores even if strict frozen-experiment success fails.
    if args.require_frozen_success:
        verify_frozen_success(rows, batch_summaries, frozen)
    native = [r for r in batch_summaries if r['split'] == 'internal_test' and r['order'] == 'native']
    print('ALL_CHECKS_PASSED:' if args.require_frozen_success else 'SCORING_COMPLETE:',
          len(rows), 'scored question-runs;', len(batch_summaries), 'batches')
    print(json.dumps(native, indent=2))


if __name__ == '__main__':
    os.chdir(ROOT)
    main()
