"""Check the exact inference routine against independent exhaustive counting."""
import importlib.util
import itertools
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / "scripts/audit_independent_contribution.py"
spec = importlib.util.spec_from_file_location("contribution_audit", path)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class ContributionInferenceTests(unittest.TestCase):
    def test_exact_sign_flip_matches_brute_force_with_ties(self):
        weights = [1, -2, 3, 0, -1]
        target = abs(sum(weights))
        values = [sum(s * w for s, w in zip(signs, weights)) for signs in itertools.product([-1, 1], repeat=len(weights))]
        expected = sum(abs(v) >= target for v in values) / len(values)
        self.assertEqual(audit.exact_count_sign_flip(weights)["p_value"], expected)

    def test_all_zero_differences_have_probability_one(self):
        self.assertEqual(audit.exact_count_sign_flip([0, 0, 0])["p_value"], 1.)

    def test_holm_matches_known_unsorted_example(self):
        actual = audit.holm_adjust([.2, .01, .03]).tolist()
        for got, expected in zip(actual, [.2, .03, .06]):
            self.assertAlmostEqual(got, expected)


if __name__ == "__main__":
    unittest.main()
