import unittest
import numpy as np
from explorer.posterior_bank import parse_tree,exact_tree_values,observation_log_likelihood


class ExecutorProtocolTests(unittest.TestCase):
    def test_parser_retains_compositional_structure(self):
        self.assertEqual(parse_tree('x AND NOT (y)'),('AND',('VAR','x'),('NOT',('VAR','y'))))

    def test_likelihood_reads_only_observed_assignments(self):
        p=np.array([[.9,.1,.2,.3],[.8,.9,.7,.6]])
        obs=[{'inputs':{'x':0,'y':0},'output':1}]
        a=observation_log_likelihood(p,obs)
        p[:,1:]=.999
        np.testing.assert_allclose(a,observation_log_likelihood(p,obs))

    def test_exact_evaluator_is_separate_and_supports_heldout_variables(self):
        tree=parse_tree('x AND NOT (z)')
        values=exact_tree_values(tree,np.array([[1,0,0],[1,1,1],[0,1,0]]),('x','y','z'))
        self.assertEqual(values.tolist(),[True,False,False])


if __name__=='__main__':unittest.main()
