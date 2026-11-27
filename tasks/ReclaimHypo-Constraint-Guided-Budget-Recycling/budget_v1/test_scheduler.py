import unittest

from scheduler import allocate


def voxel(qid, height=2, columns=1):
    return {'id': qid, 'task': 'voxel',
            'query': {'top_view': [[1] * columns], 'max_height': height}}


class SchedulerTests(unittest.TestCase):
    def test_fixed_repeat_and_stop_have_same_unique_coverage(self):
        # A two-solution problem cannot have six distinct valid outputs.
        repeated = allocate([voxel('a')], 'fixed_repeat')
        stopped = allocate([voxel('a')], 'fixed_stop')
        self.assertEqual(len(repeated['cases'][0]['outputs']), 6)
        self.assertEqual(len(stopped['cases'][0]['outputs']), 2)
        self.assertEqual(repeated['cases'][0]['unique_count'], 2)
        self.assertEqual(repeated['cases'][0]['repeats'], 4)
        self.assertEqual(stopped['unspent'], 4)

    def test_shared_budget_reclaims_small_space_slots(self):
        # S=(2,16), B=12: consume both small solutions, ten large solutions.
        r = allocate([voxel('small'), voxel('large', 4, 2)],
                     'shared_round_robin', total_budget=12)
        self.assertEqual([len(x['outputs']) for x in r['cases']], [2, 10])
        self.assertEqual(r['spent'], 12)
        self.assertEqual(r['unspent'], 0)
        self.assertTrue(r['cases'][0]['complete'])
        self.assertTrue(all(x['repeats'] == 0 for x in r['cases']))

    def test_identical_answers_in_different_questions_are_not_duplicates(self):
        r = allocate([voxel('a'), voxel('b')], 'shared_round_robin', total_budget=4)
        self.assertEqual([x['unique_count'] for x in r['cases']], [2, 2])
        self.assertEqual(r['spent'], 4)

    def test_all_spaces_exhausted_leave_budget_unspent(self):
        r = allocate([voxel('a'), voxel('b')], 'shared_round_robin', total_budget=10)
        self.assertEqual(r['spent'], 4)
        self.assertEqual(r['unspent'], 6)
        self.assertEqual(r['termination'], 'all_spaces_exhausted')
        self.assertTrue(all(x['complete'] for x in r['cases']))

    def test_no_false_exhaustion_exactly_at_budget_boundary(self):
        r = allocate([voxel('a')], 'shared_round_robin', total_budget=2)
        self.assertEqual(r['spent'], 2)
        self.assertFalse(r['cases'][0]['complete'])

    def test_zero_budget_never_pulls_or_claims_exhaustion(self):
        r = allocate([voxel('a')], 'shared_round_robin', total_budget=0)
        self.assertEqual(r['spent'], 0)
        self.assertEqual(r['cases'][0]['outputs'], [])
        self.assertFalse(r['cases'][0]['complete'])

    def test_round_robin_gives_first_pass_before_second(self):
        r = allocate([voxel('a'), voxel('b'), voxel('c')],
                     'shared_round_robin', total_budget=2)
        self.assertEqual([len(x['outputs']) for x in r['cases']], [1, 1, 0])

    def test_timeout_is_not_exhaustion_and_adds_no_repeats(self):
        r = allocate([voxel('a')], 'fixed_repeat', time_limit=1e-12)
        self.assertEqual(r['spent'], 0)
        self.assertEqual(r['termination'], 'time_limit')
        self.assertFalse(r['cases'][0]['complete'])
        self.assertEqual(r['cases'][0]['repeats'], 0)

    def test_unsatisfiable_problem_releases_all_its_slots(self):
        q = {'id': 'unsat', 'task': 'boolean', 'query': {
            'variables': ['x'], 'max_depth': 2,
            'observations': [{'inputs': {'x': 0}, 'output': 0},
                             {'inputs': {'x': 0}, 'output': 1}]}}
        r = allocate([q, voxel('b', 4)], 'shared_round_robin', total_budget=3)
        self.assertEqual([len(x['outputs']) for x in r['cases']], [0, 3])
        self.assertTrue(r['cases'][0]['complete'])

    def test_two_node_dags_are_distinct_mechanisms(self):
        q = {'id': 'dag', 'task': 'causal',
             'query': {'nodes': ['A', 'B'], 'observations': []}}
        r = allocate([q], 'fixed_repeat', per_case=4)
        self.assertEqual(r['cases'][0]['unique_count'], 3)
        self.assertEqual(r['cases'][0]['repeats'], 1)
        self.assertTrue(r['cases'][0]['complete'])

    def test_rejects_answer_fields_and_duplicate_ids(self):
        for location in ['top', 'query']:
            q = voxel('a')
            (q if location == 'top' else q['query'])['gold'] = []
            with self.assertRaises(ValueError):
                allocate([q], 'shared_round_robin')
        with self.assertRaises(ValueError):
            allocate([voxel('a'), voxel('a')], 'shared_round_robin')

    def test_rejects_invalid_budget_and_conflicting_fixed_total(self):
        for budget in [-1, True, 1.2]:
            with self.assertRaises(ValueError):
                allocate([voxel('a')], 'shared_round_robin', total_budget=budget)
        with self.assertRaises(ValueError):
            allocate([voxel('a')], 'fixed_repeat', total_budget=7)
        for limit in [0, float('inf'), float('nan')]:
            with self.assertRaises(ValueError):
                allocate([voxel('a')], 'shared_round_robin', time_limit=limit)


if __name__ == '__main__':
    unittest.main()
