import json,unittest
from pathlib import Path
class SplitTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  root=Path('work/lora_balanced');cls.audit=json.loads((root/'audit.json').read_text());cls.tr=[json.loads(x) for x in (root/'train.jsonl').read_text().splitlines()];cls.dev=[json.loads(x) for x in (root/'dev.jsonl').read_text().splitlines()]
 def test_no_task_overlap_or_cross_subset(self):
  self.assertEqual(self.audit['train_dev_exact_overlap'],0);self.assertEqual(self.audit['train_dev_subset_pairs'],0)
 def test_anchor_both_classes(self):
  for data in (self.tr,self.dev):
   bits={o['output'] for r in data for o in r['observations'] if tuple(o['inputs'][n] for n in ('x','y','z'))==(0,0,0)};self.assertEqual(bits,{0,1})
 def test_no_development_answers_in_generation_file(self):self.assertTrue(all('targets' not in r for r in self.dev))
 def test_generating_truths_disjoint(self):
  origin=self.audit['task_origin_truths'];a={t for r in self.tr for t in origin[r['id']]};b={t for r in self.dev for t in origin[r['id']]};self.assertFalse(a&b)
if __name__=='__main__':unittest.main()
