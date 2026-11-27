import unittest

from explorer.canonical_decode import SupportTrie, grammar_support


class CanonicalDecodeTests(unittest.TestCase):
    def setUp(self):
        self.pool = [
            {"answer": "a", "key": "A", "tokens": [1, 2, 9]},
            {"answer": "alias-a", "key": "A", "tokens": [1, 3, 9]},
            {"answer": "b", "key": "B", "tokens": [1, 4, 9]},
        ]

    def test_class_exclusion_blocks_every_alias(self):
        trie = SupportTrie(self.pool, blocked_keys={"A"})
        self.assertEqual(trie.allowed([1]), (4,))
        self.assertEqual(trie.node([1, 4, 9]).terminal_key, "B")

    def test_string_exclusion_retains_alias_so_is_a_distinct_control(self):
        trie = SupportTrie(self.pool, blocked_strings={"a"})
        self.assertEqual(set(trie.allowed([1])), {3, 4})

    def test_exhaustion_and_prefix_fail_closed(self):
        with self.assertRaises(ValueError):
            SupportTrie(self.pool, blocked_keys={"A", "B"})
        with self.assertRaises(ValueError):
            SupportTrie(self.pool).allowed([7])

    def test_support_has_no_observation_or_reference_inputs(self):
        metadata = {"variables": ["x", "y"], "operators": ["AND", "OR", "NOT"], "max_depth": 2,
                    "mechanistic_opts": {"apply_commutativity": True, "apply_idempotence_and_or": True, "flatten_associativity": True}}
        pool = grammar_support(metadata)
        self.assertEqual(len(pool), 300)
        self.assertEqual(len({h['key'] for h in pool}), 33)
        self.assertTrue(all(set(h) == {"answer", "key"} for h in pool))

    def test_mask_keeps_only_active_tokens_and_handles_mlx_eos_prefetch(self):
        try:
            import mlx.core as mx
        except ImportError:
            self.skipTest("MLX only")
        from explorer.canonical_decode import TrieLogitsProcessor
        processor = TrieLogitsProcessor(SupportTrie(self.pool, blocked_keys={"A"}), 2)
        logits = mx.zeros((1, 12))
        first = processor(mx.array([7, 8, 1]), logits)
        self.assertEqual(mx.isfinite(first).sum().item(), 1)
        self.assertTrue(mx.isfinite(first[0, 4]).item())
        final = processor(mx.array([7, 8, 1, 4, 9]), logits)
        self.assertEqual(mx.isfinite(final).sum().item(), 1)
        self.assertTrue(mx.isfinite(final[0, 9]).item())


if __name__ == "__main__":
    unittest.main()
