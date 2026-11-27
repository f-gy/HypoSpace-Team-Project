import unittest

from explorer.target_decode import intervene


class TargetInterventionTests(unittest.TestCase):
    def test_all_prefix_outcomes_keep_target_feasible(self):
        states = {(0, 0)}
        for step in range(16):
            following = set()
            for completed, seen in states:
                self.assertEqual(completed, step)
                if intervene(seen, step):
                    following.add((step + 1, seen + 1))
                else:
                    following.add((step + 1, seen))
                    following.add((step + 1, seen + 1))
            states = following
        self.assertTrue(all(seen >= 13 for _, seen in states))

    def test_three_initial_failures_use_all_slack(self):
        self.assertFalse(intervene(0, 2))
        self.assertTrue(intervene(0, 3))

    def test_does_not_intervene_after_target_already_met(self):
        self.assertFalse(intervene(13, 14))


if __name__ == '__main__':
    unittest.main()
