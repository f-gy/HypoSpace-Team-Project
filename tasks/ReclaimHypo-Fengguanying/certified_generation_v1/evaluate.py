"""Query-only inference; gold/pools used ONLY by the separate evaluator."""
import json,pathlib,sys,statistics,hashlib,importlib.util,os
import numpy as np,networkx as nx
P=pathlib.Path(__file__).resolve().parent;ROOT=P.parent
OUT=pathlib.Path(os.environ.get('HYPOSPACE_RESULT_DIR',str(P)));OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'code'))
import sgve_core as c
from hypospace_validator import generated_universe
from boolean_dataset import BooleanObservation
SOURCE=pathlib.Path(os.environ.get('HYPOSPACE_SOURCE',str(ROOT/'vendor/HypoSpace')))
if not SOURCE.exists():SOURCE=pathlib.Path('/root/hypospace-project/HypoSpace')
sys.path.insert(0,str(SOURCE/'causal'))
from modules.models import CausalGraph
from generate_causal_dataset import CausalDatasetGenerator as CG
from exact_backend import solve
spec=importlib.util.spec_from_file_location('official_voxel',str(SOURCE/'3d/generate_3d_dataset_complete.py'));vo=importlib.util.module_from_spec(spec);spec.loader.exec_module(vo)
b=c.BooleanBenchmarkRefined(str(ROOT/'judge_training_v1/validator.json'));v=c.install(b,policy='enumerated',universe=generated_universe(b))

def query(case):
 t=case['task']
 if t=='voxel':return {'top_view':case['observations'],'max_height':3}
 if t=='causal':return {'nodes':sorted(case['observations'][0]['effects']),'observations':case['observations']}
 return {'variables':['x','y','z'],'operators':['AND','OR','NOT'],'max_depth':2,'observations':case['observations']}

def key_check(t,q,obj):
 if t=='causal':
  edges=[tuple(x) for x in obj['edges']];g=CausalGraph(q['nodes'],edges)
  assert nx.is_directed_acyclic_graph(g.to_networkx())
  assert all(CG.get_perturbation_effects(g,o['perturbed_node']).effects==o['effects'] for o in q['observations'])
  return g.get_hash()
 if t=='boolean':
  f=obj['formula'];assert v.validate_expression(f,[BooleanObservation(o['inputs'],o['output']) for o in q['observations']])[0]
  return repr(v.get_expression_mechanistic_key(f))
 h=obj['heights'];top=np.array(q['top_view']);coords=list(zip(*np.where(top==1)))
 assert len(h)==len(coords) and all(type(x)==int and 1<=x<=q['max_height'] for x in h)
 layers=[np.zeros_like(top) for _ in range(max(h))]
 for (row,col),height in zip(coords,h):
  for z in range(height):layers[z][row,col]=1
 structure=vo.Structure3D(layers);assert np.array_equal(structure.get_top_view(),top)
 assert all(np.all(layers[z]<=layers[z-1]) for z in range(1,len(layers)))
 return json.dumps(obj,sort_keys=True,separators=(',',':'))

rows=[];inputs=[]
for name,path in [('dev',ROOT/'joint_three_v1/dev_evaluation.json'),('internal_test',ROOT/'joint_three_v1/test_evaluation.json'),('causal4_pressure',ROOT/'causal4_pressure/cases.json')]:
 for case in json.load(open(path)):inputs.append({'split':name,'id':case['id'],'task':case['task'],'query':query(case)})
(OUT/'query_only_inputs.json').write_text(json.dumps(inputs,indent=2))
(OUT/'protocol.json').write_text(json.dumps({'budget':6,'other_run':'S+1 to verify complete coverage; oracle-sized audit, not equal-budget comparison','inference_fields':['task','query','budget','time_limit'],'separate_inference_file_sha256':hashlib.sha256((OUT/'query_only_inputs.json').read_bytes()).hexdigest(),'solver_no_gold_or_pool':True,'boolean':'public grammar enumeration, not scan-free','causal':'Z3 reachability and acyclicity; block edge assignments','voxel':'construct supported columns','baseline_comparison':'direct_generation_v1; identical19 dev cases and output budget6; extra solver compute disclosed','test':'internal18 already used in earlier experiments; not pristine blind holdout','metrics':'V=valid/B; U=all canonical unique/B; R=distinct valid/S; repeats count','scope':'Boolean3vars depth2; voxel3x3 maxheight3; DAG3and4nodes'},indent=2))
for name,path in [('dev',ROOT/'joint_three_v1/dev_evaluation.json'),('internal_test',ROOT/'joint_three_v1/test_evaluation.json'),('causal4_pressure',ROOT/'causal4_pressure/cases.json')]:
 for case in json.load(open(path)):
  t=case['task'];q=query(case);gold=set(case['gold'])
  for mode,B in [('fixed6',6),('exhaustive_audit',len(gold)+1)]:
   result=solve(t,q,B,time_limit=30);keys=[key_check(t,q,x) for x in result['outputs']];found=set(keys)
   assert found<=gold,(name,case['id'],found-gold)
   assert len(found)==min(B,len(gold)),(case['id'],result)
   assert len(keys)==B
   if mode=='exhaustive_audit':assert result['complete'] and found==gold
   rows.append({'split':name,'id':case['id'],'task':t,'mode':mode,'budget':B,'space_size':len(gold),'validity':len(keys)/B,'uniqueness':len(found)/B,'recovery':len(found)/len(gold),'attains_feasible_bound':True,**result})
summary=[]
for split in ['dev','internal_test','causal4_pressure']:
 for task in ['causal','voxel','boolean']:
  rs=[r for r in rows if r['split']==split and r['task']==task and r['mode']=='fixed6']
  if rs:summary.append({'split':split,'task':task,'cases':len(rs),**{k:statistics.mean(r[k] for r in rs) for k in ['validity','uniqueness','recovery']},'seconds':sum(r['seconds'] for r in rs),'solver_checks':sum(r['solver_checks'] for r in rs),'predicate_checks':sum(r['predicate_checks'] for r in rs)})
(OUT/'scored.json').write_text(json.dumps(rows,indent=2));(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
# Independent public generator comparison, not instance gold.
stress=[]
for n,expected in [(3,25),(4,543)]:
 nodes=list('ABCD')[:n];r=solve('causal',{'nodes':nodes,'observations':[]},expected+1,30)
 official={g.get_hash() for g in CG.generate_all_dags(nodes)};found={CausalGraph(nodes,[tuple(e) for e in obj['edges']]).get_hash() for obj in r['unique_outputs']}
 assert r['complete'] and len(found)==expected and found==official
 stress.append({'task':'causal','nodes':n,'all_space_size':expected,'matched_official':True,'seconds':r['seconds']})
q={'nodes':list('ABCDEF'),'observations':[]};r=solve('causal',q,20,30)
assert len({key_check('causal',q,obj) for obj in r['outputs']})==20
stress.append({'task':'causal','nodes':6,'valid_distinct':20,'seconds':r['seconds'],'no_completeness_claim':True})
q={'variables':['x','y','z'],'operators':['AND','OR','NOT'],'max_depth':2,'observations':[]};r=solve('boolean',q,126,30)
keys={key_check('boolean',q,x) for x in r['unique_outputs']}
assert r['complete'] and len(keys)==125
stress.append({'task':'boolean','all_mechanisms':125,'all_independently_validated':True,'seconds':r['seconds']})
for top in [[[1,1,1],[1,0,0],[0,0,0]],[[1,0,1],[0,1,0],[0,0,0]]]:
 q={'top_view':top,'max_height':3};S=3**sum(map(sum,top));r=solve('voxel',q,S+1)
 official={x.hash for x in vo.enumerate_all_structures_by_heights(np.array(top),3)}
 # Independently reconstruct layer hashes exactly as official serialization.
 ours=set()
 for obj in r['unique_outputs']:
  key_check('voxel',q,obj);h=obj['heights'];coords=list(zip(*np.where(np.array(top)==1)));layers=[np.zeros_like(top) for _ in range(max(h))]
  for (row,col),height in zip(coords,h):
   for z in range(height):layers[z][row,col]=1
  ours.add(vo.Structure3D(layers).hash)
 assert r['complete'] and ours==official and len(ours)==S
 stress.append({'task':'voxel','all_space_size':S,'matched_official':True,'seconds':r['seconds']})
(OUT/'stress.json').write_text(json.dumps(stress,indent=2));print('ALL_CHECKS_PASSED',len(rows),'scored runs',flush=True)
