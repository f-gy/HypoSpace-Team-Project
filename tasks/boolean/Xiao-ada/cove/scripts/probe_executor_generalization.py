"""Auxiliary semantic probe: unseen depth and variable compositions, no training."""
from __future__ import annotations

import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from explorer.posterior_bank import exact_tree_values, numpy_predictions
from explorer.tasks import write_json


def depth(t):
    return 0 if t[0] == 'VAR' else 1 + max(depth(v) for v in t[1:])


def main():
    rng = random.Random(4310)
    def sample(level):
        if not level or rng.random() < .2:
            return ('VAR', rng.choice(('x', 'y', 'z')))
        op = rng.choice(('AND', 'OR', 'NOT'))
        return (op, sample(level - 1)) if op == 'NOT' else (op, sample(level - 1), sample(level - 1))
    bits = np.asarray(list(itertools.product((0, 1), repeat=3)), dtype=np.float32)
    trees, labels, seen = [], [], set()
    attempted = 0
    while len(trees) < 1024:
        attempted += 1
        t = sample(6)
        if t in seen or depth(t) < 3:
            continue
        y = exact_tree_values(t, bits, ('x', 'y', 'z'))
        # Require a nonconstant function that actually depends on held-out z.
        if not np.any(y[::2] != y[1::2]):
            continue
        trees.append(t)
        labels.append(y)
        seen.add(t)
    y = np.asarray(labels)
    result = {'seed': 4310, 'programs': 1024, 'assignments_per_program': 8,
              'depth_range': [min(map(depth, trees)), max(map(depth, trees))],
              'filter': 'unique syntax; depth>=3; semantic dependence on z; nonconstant',
              'sampling_attempts': attempted,
              'scope': 'Auxiliary neural-operator semantics only. Leaf-variable mapping is supplied. Not LLM generation or a three-variable HypoSpace evaluation; no gradients or selection.',
              'metrics': {}, 'programs_and_labels': [{'tree': t, 'truth': v.astype(int).tolist()} for t, v in zip(trees, y)]}
    for seed in [4306, 4307, 4308]:
        w = json.loads((ROOT / f'results/all80/executors/{seed}/weights.json').read_text())
        p = numpy_predictions(w, [{'tree': t} for t in trees], bits, ('x', 'y', 'z'))
        correct = (p >= .5) == y
        result['metrics'][str(seed)] = {'output_accuracy': float(correct.mean()),
                                        'complete_program_accuracy': float(np.all(correct, axis=1).mean())}
    write_json(ROOT / 'results/all80/executor_generalization_filtered.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'programs_and_labels'}, indent=2))


if __name__ == '__main__':
    main()
