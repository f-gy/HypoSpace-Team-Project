import unittest
from exact_backend import complete_proposals
class HybridTests(unittest.TestCase):
 def test_keeps_valid_and_repairs_invalid(self):
  q={'top_view':[[1,1]],'max_height':3}
  r=complete_proposals('voxel',q,['{"heights":[3,3]}','bad','{"heights":[0,2]}','{"heights":[3,3]}'],6)
  self.assertEqual(r['outputs'][0],{'heights':[3,3]});self.assertEqual(r['kept_proposals'],1);self.assertEqual(len({tuple(x['heights']) for x in r['outputs']}),6)
 def test_small_space_repeats_are_disclosed(self):
  r=complete_proposals('voxel',{'top_view':[[1]],'max_height':2},['{"heights":[2]}'],6)
  self.assertEqual(len(r['outputs']),6);self.assertEqual(r['unique_count'],2);self.assertEqual(r['repeats'],4)
 def test_query_gold_is_rejected(self):
  with self.assertRaises(ValueError):complete_proposals('voxel',{'top_view':[[1]],'max_height':2,'gold':[]},[],6)
if __name__=='__main__':unittest.main()
