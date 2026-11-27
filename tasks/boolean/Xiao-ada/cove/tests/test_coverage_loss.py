import unittest
try:
    import mlx.core as mx
except ImportError:
    mx=None


@unittest.skipIf(mx is None,'MLX requires Apple Silicon')
class CoverageLossTests(unittest.TestCase):
    def test_answer_mask_excludes_prompt_and_padding_and_keeps_end_token(self):
        from explorer.coverage_lora import batch
        ids,mask,_=batch({'aliases':[{'tokens':[11,12,21,99],'prompt_length':2},
                                    {'tokens':[11,12,22,23,99],'prompt_length':2}], 'class_ratio':1.})
        self.assertEqual(mask.tolist(),[[0.,1.,1.,0.],[0.,1.,1.,1.]])
        self.assertEqual(ids[0,-1].item(),0)

    def test_class_mass_is_invariant_to_alias_partition(self):
        from explorer.coverage_lora import coverage_value
        a=coverage_value(mx.log(mx.array([.1,.2])),1.,'canonical_coverage')
        b=coverage_value(mx.log(mx.array([.3])),1.,'canonical_coverage')
        self.assertAlmostEqual(a.item(),b.item(),places=6)

    def test_string_reward_can_overcount_a_single_class(self):
        from explorer.coverage_lora import coverage_value
        p=mx.log(mx.array([.2,.3]))
        self.assertGreater(coverage_value(p,1.,'string_coverage').item(),coverage_value(p,1.,'canonical_coverage').item())

    def test_extreme_log_probabilities_have_finite_gradients(self):
        from explorer.coverage_lora import coverage_value
        grad=mx.grad(lambda p:coverage_value(p,1.,'canonical_coverage'))(mx.array([-1000.,-.01]))
        self.assertTrue(mx.all(mx.isfinite(grad)).item())
