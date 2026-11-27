import importlib.util
import pathlib
import json
import subprocess
import sys
import tempfile
import unittest

path = pathlib.Path(__file__).with_name('evaluate.py')
spec = importlib.util.spec_from_file_location('budget_evaluator_for_test', path)
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


class MetricTests(unittest.TestCase):
    def test_task_rate_uses_allocated_slots_not_old_fixed_quota(self):
        summarize = getattr(evaluator, 'summarize_task', None)
        self.assertIsNotNone(summarize)
        # Two questions now receive 12 outputs each, instead of 6 each.
        rows = [{'emitted': 12, 'valid_count': 12, 'distinct_valid': 12,
                 'recovery': 12/27} for _ in range(2)]
        r = summarize(rows, 'shared_round_robin')
        self.assertEqual(r['reference_output_slots'], 12)
        self.assertEqual(r['allocated_output_slots'], 24)
        self.assertEqual(r['unique_valid_per_budget'], 1.0)

    def test_empty_output_has_undefined_precision_not_perfect_precision(self):
        r = evaluator.rates([{'emitted': 0, 'valid_count': 0,
                              'distinct_valid': 0, 'recovery': 0}], 6)
        self.assertIsNone(r['precision_emitted'])
        self.assertIsNone(r['uniqueness_emitted'])
        self.assertEqual(r['unique_valid_per_budget'], 0)

    def test_preparation_ignores_appledouble_metadata_as_source_code(self):
        root = pathlib.Path(__file__).resolve().parent.parent
        probe = root / 'code/._metadata_probe.py'
        try:
            probe.write_bytes(b'not Python source')
            with tempfile.TemporaryDirectory() as temp:
                out = pathlib.Path(temp) / 'prepared'
                subprocess.run([sys.executable, str(root / 'budget_v1/prepare.py'),
                                '--run-dir', str(out)], check=True, capture_output=True)
                protocol = json.loads((out / 'protocol.json').read_text())
                self.assertNotIn('code/._metadata_probe.py', protocol['code_sha256'])
                self.assertIn('code/boolean_dataset.py', protocol['code_sha256'])
        finally:
            probe.unlink(missing_ok=True)

    def test_empty_solution_space_has_undefined_recovery_and_is_counted(self):
        score = getattr(evaluator, 'score_case', None)
        self.assertIsNotNone(score)
        result = {'id': 'empty', 'outputs': [], 'unique_count': 0,
                  'repeats': 0, 'complete': True, 'reason': 'space_exhausted'}
        row = score('boolean', {'observations': []}, result, set(), 'fixed_repeat')
        self.assertIsNone(row['recovery'])
        self.assertIsNone(row['precision_emitted'])
        other = {'emitted': 1, 'valid_count': 1, 'distinct_valid': 1, 'recovery': .5}
        r = evaluator.rates([row, other], 12)
        self.assertEqual(r['macro_recovery'], .5)
        self.assertEqual(r['recovery_defined_questions'], 1)
        self.assertEqual(r['empty_solution_questions'], 1)

    def test_honest_timeout_partial_outputs_are_scored_not_rejected(self):
        score = getattr(evaluator, 'score_case', None)
        self.assertIsNotNone(score)
        result = {'id': 'partial', 'outputs': [{'heights': [1]}], 'unique_count': 1,
                  'repeats': 0, 'complete': False, 'reason': 'time_limit'}
        q = {'top_view': [[1]], 'max_height': 3}
        gold = {'{"heights":[1]}', '{"heights":[2]}', '{"heights":[3]}'}
        for policy in ['fixed_repeat', 'shared_round_robin']:
            row = score('voxel', q, result, gold, policy)
            self.assertEqual(row['emitted'], 1)
            self.assertEqual(row['recovery'], 1/3)
            self.assertEqual(row['reason'], 'time_limit')
            self.assertFalse(row['complete'])


if __name__ == '__main__':
    unittest.main()
