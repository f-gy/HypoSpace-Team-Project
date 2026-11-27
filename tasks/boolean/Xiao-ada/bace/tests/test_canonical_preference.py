import unittest

try:
    import mlx.core as mx
except ImportError:
    mx = None


@unittest.skipIf(mx is None, 'MLX only')
class CanonicalPreferenceTests(unittest.TestCase):
    def test_frozen_reference_has_log_two_loss(self):
        from explorer.canonical_preference import group_margin, preference_loss
        import math
        p = mx.log(mx.array([.1,.2,.05,.05]))
        self.assertAlmostEqual(preference_loss(p, group_margin(p)).item(), math.log(2), places=6)

    def test_improved_positive_group_margin_reduces_loss(self):
        from explorer.canonical_preference import preference_loss
        a = mx.log(mx.array([.1,.1,.1,.1]))
        b = mx.log(mx.array([.2,.2,.05,.05]))
        self.assertLess(preference_loss(b, mx.array(0.)).item(), preference_loss(a, mx.array(0.)).item())

    def test_within_group_probability_partition_does_not_change_preference(self):
        from explorer.canonical_preference import group_margin
        a = mx.log(mx.array([.1,.2,.04,.06]))
        b = mx.log(mx.array([.15,.15,.05,.05]))
        self.assertAlmostEqual(group_margin(a).item(), group_margin(b).item(), places=6)

    def test_extreme_values_have_finite_gradients(self):
        from explorer.canonical_preference import preference_loss
        gradient = mx.grad(lambda x: preference_loss(x, mx.array(1.)))(mx.array([-1000.,-900.,-.1,-.1]))
        self.assertTrue(mx.all(mx.isfinite(gradient)).item())

    def test_response_only_projection_preserves_loss_and_all_parameter_gradients(self):
        from mlx_lm.models.qwen2 import Model, ModelArgs
        from mlx.utils import tree_flatten
        import mlx.nn as nn
        from explorer.canonical_preference import loss_function
        mx.random.seed(4305)
        model = Model(ModelArgs(model_type='qwen2', hidden_size=16, num_hidden_layers=2,
                                intermediate_size=32, num_attention_heads=4, num_key_value_heads=2,
                                vocab_size=32, tie_word_embeddings=True, rms_norm_eps=1e-6))
        ids = mx.array([[1,2,3,4,5,6],[1,2,3,7,8,6],[1,2,3,9,10,6],[1,2,3,11,12,6]])
        mask = mx.array([[0.,0.,1.,1.,1.]]*4)
        args = (ids,mask,mx.array(1.),mx.array(0.),2)
        for preference in [False,True]:
            full,gfull = nn.value_and_grad(model,loss_function(preference,False))(model,*args)
            short,gshort = nn.value_and_grad(model,loss_function(preference,True))(model,*args)
            self.assertAlmostEqual(full.item(),short.item(),places=5)
            old=dict(tree_flatten(gfull)); new=dict(tree_flatten(gshort))
            for name in old:
                self.assertLess(mx.max(mx.abs(old[name]-new[name])).item(),1e-5,name)


if __name__ == '__main__':
    unittest.main()
