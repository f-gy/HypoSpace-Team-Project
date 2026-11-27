"""Allocate OUTPUT slots across query-only, persistent certified streams.

Gold sets and space sizes are deliberately absent from this interface.
The same canonical mechanism in different questions is a different item.
"""
import copy
import math
import pathlib
import sys
import time
from collections import deque

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'certified_generation_v1'))
import exact_backend as backend

ALLOWED = {
    'causal': {'nodes', 'observations', 'max_edges'},
    'voxel': {'top_view', 'max_height'},
    'boolean': {'variables', 'operators', 'max_depth', 'observations'},
}
REQUIRED = {
    'causal': {'nodes', 'observations'},
    'voxel': {'top_view', 'max_height'},
    'boolean': {'observations'},
}
GENERATORS = {'causal': backend._causal, 'voxel': backend._voxel,
              'boolean': backend._boolean}


class _Cursor:
    def __init__(self, case, deadline):
        self.id = case['id']
        self.task = case['task']
        self.deadline = deadline
        self.stats = {'solver_checks': 0, 'predicate_checks': 0,
                      'construction_steps': 0}
        self.iterator = GENERATORS[self.task](copy.deepcopy(case['query']),
                                              deadline, self.stats)
        self.seen = set()
        self.outputs = []
        self.complete = False
        self.reason = 'not_started'
        self.seconds = 0.0

    def advance(self):
        start = time.monotonic()
        try:
            while True:
                if time.monotonic() >= self.deadline:
                    raise TimeoutError('batch time limit')
                key, obj = next(self.iterator)
                if key not in self.seen:
                    self.seen.add(key)
                    self.outputs.append(obj)
                    self.reason = 'budget_reached'
                    return True
        except StopIteration:
            self.complete = True
            self.reason = 'space_exhausted'
            return False
        except TimeoutError as error:
            self.reason = 'time_limit: ' + str(error)
            return False
        finally:
            self.seconds += time.monotonic() - start

    def result(self):
        return {'id': self.id, 'task': self.task, 'outputs': self.outputs,
                'unique_count': len(self.seen),
                'repeats': len(self.outputs) - len(self.seen),
                'complete': self.complete, 'reason': self.reason,
                'generation_wall_seconds': self.seconds, **self.stats}


def allocate(cases, policy, per_case=6, total_budget=None, time_limit=30):
    """Compare fixed_repeat, fixed_stop, and shared_round_robin.

    The budget counts emitted outputs, not equal CPU compute. Time limits
    apply to the whole batch. An exhaustion proof requires StopIteration;
    reaching a quota or a time limit is never an exhaustion certificate.
    """
    if policy not in {'fixed_repeat', 'fixed_stop', 'shared_round_robin'}:
        raise ValueError('unknown policy')
    if type(per_case) is not int or per_case < 1:
        raise ValueError('positive integer per_case required')
    if not isinstance(time_limit, (float, int)) or not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError('positive finite time limit required')
    if not isinstance(cases, list):
        raise ValueError('cases must be a list')
    ids = set()
    for case in cases:
        if not isinstance(case, dict) or set(case) != {'id', 'task', 'query'}:
            raise ValueError('query-only cases require exactly id/task/query')
        qid, task, query = case['id'], case['task'], case['query']
        if not isinstance(qid, str) or not qid or qid in ids:
            raise ValueError('nonempty unique question ids required')
        if not isinstance(task, str) or task not in ALLOWED or not isinstance(query, dict):
            raise ValueError('unsupported task/query')
        if set(query) - ALLOWED[task] or REQUIRED[task] - set(query):
            raise ValueError('unexpected or missing query fields')
        ids.add(qid)
    expected = per_case * len(cases)
    budget = expected if total_budget is None else total_budget
    if type(budget) is not int or budget < 0:
        raise ValueError('nonnegative integer output budget required')
    if policy != 'shared_round_robin' and budget != expected:
        raise ValueError('fixed policy budget must equal per_case times cases')

    start = time.monotonic()
    deadline = start + time_limit
    cursors = [_Cursor(case, deadline) for case in cases]
    spent = 0
    if policy == 'shared_round_robin':
        active = deque(cursors)
        while active and spent < budget and time.monotonic() < deadline:
            cursor = active.popleft()
            if cursor.advance():
                spent += 1
                active.append(cursor)
    else:
        for cursor in cursors:
            while len(cursor.outputs) < per_case and time.monotonic() < deadline:
                if cursor.advance():
                    spent += 1
                else:
                    break
            # Repeat only a proved finite, nonempty set; never after a timeout.
            if policy == 'fixed_repeat' and cursor.complete and cursor.outputs:
                unique = list(cursor.outputs)
                missing = per_case - len(unique)
                cursor.outputs.extend(unique[i % len(unique)] for i in range(missing))
                spent += missing

    if spent == budget:
        termination = 'budget_reached'
    elif time.monotonic() >= deadline or any(c.reason.startswith('time_limit') for c in cursors):
        termination = 'time_limit'
    elif all(c.complete for c in cursors):
        termination = 'all_spaces_exhausted'
    else:
        termination = 'per_case_caps_reached'
    return {'policy': policy, 'per_case_reference': per_case, 'budget': budget,
            'spent': spent, 'unspent': budget - spent,
            'termination': termination, 'seconds': time.monotonic() - start,
            'cases': [cursor.result() for cursor in cursors]}
