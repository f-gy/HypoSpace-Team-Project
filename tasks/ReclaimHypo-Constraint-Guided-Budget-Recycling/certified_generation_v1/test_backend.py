import unittest
from exact_backend import solve
class BackendTests(unittest.TestCase):
 def test_voxel_unique_budget(self):
  r=solve('voxel',{'top_view':[[1,1]],'max_height':3},6);self.assertEqual(len(r['outputs']),6);self.assertEqual(len({tuple(x['heights']) for x in r['outputs']}),6)
 def test_exhausted_space_fills_only_proven_repeats(self):
  r=solve('voxel',{'top_view':[[1]],'max_height':2},6);self.assertTrue(r['complete']);self.assertEqual(len(r['unique_outputs']),2);self.assertEqual(len(r['outputs']),6)
 def test_causal_enumerates_all_3node_dags(self):
  r=solve('causal',{'nodes':['A','B','C'],'observations':[]},30);self.assertTrue(r['complete']);self.assertEqual(len(r['unique_outputs']),25)
 def test_causal_respects_intervention(self):
  obs=[{'perturbed_node':'A','effects':{'A':0,'B':1,'C':0}}];r=solve('causal',{'nodes':['A','B','C'],'observations':obs},12);self.assertTrue(r['complete']);self.assertEqual(len(r['unique_outputs']),4)
 def test_boolean_contradiction_is_unsat(self):
  obs=[{'inputs':{'x':0,'y':0,'z':0},'output':0},{'inputs':{'x':0,'y':0,'z':0},'output':1}];r=solve('boolean',{'variables':['x','y','z'],'observations':obs},6);self.assertTrue(r['complete']);self.assertEqual(r['outputs'],[])
 def test_bad_budget_rejected(self):
  with self.assertRaises(ValueError):solve('voxel',{'top_view':[[1]],'max_height':3},0)
if __name__=='__main__':unittest.main()
