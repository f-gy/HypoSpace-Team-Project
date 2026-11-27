"""Exact symbolic backend. No model weights, answer lists, or instance GT inputs."""
import itertools,json,time,sys,pathlib
from functools import lru_cache
ROOT=pathlib.Path(__file__).resolve().parent.parent

def _voxel(q,deadline,stats):
 top=q['top_view'];H=q['max_height']
 if type(H)!=int or H<1 or not top or not top[0] or any(len(row)!=len(top[0]) or any(type(x)!=int or x not in (0,1) for x in row) for row in top):raise ValueError('invalid top view/height')
 n=sum(map(sum,top))
 if n==0:raise ValueError('empty top view is outside verified benchmark profile')
 for heights in itertools.product(range(1,H+1),repeat=n):
  stats['construction_steps']+=1;obj={'heights':list(heights)};yield json.dumps(obj,sort_keys=True),obj

def _causal(q,deadline,stats):
 import z3
 nodes=q['nodes'];obs=q['observations'];n=len(nodes)
 if not 1<=n<=8 or len(set(nodes))!=n or any(not isinstance(x,str) for x in nodes):raise ValueError('unsupported nodes')
 nodes=sorted(nodes);index={x:i for i,x in enumerate(nodes)};s=z3.Solver();E={(i,j):z3.Bool('e_%d_%d'%(i,j)) for i in range(n) for j in range(n) if i!=j};ranks=[z3.Int('rank_%d'%i) for i in range(n)]
 for rank in ranks:s.add(rank>=0,rank<n)
 for (i,j),e in E.items():s.add(z3.Implies(e,ranks[i]<ranks[j]))
 if q.get('max_edges') is not None:
  if type(q['max_edges'])!=int or q['max_edges']<0:raise ValueError('bad max_edges')
  s.add(z3.Sum([z3.If(e,1,0) for e in E.values()])<=q['max_edges'])
 R=[[E.get((i,j),z3.BoolVal(False)) for j in range(n)] for i in range(n)]
 for k in range(n):R=[[z3.Or(R[i][j],z3.And(R[i][k],R[k][j])) for j in range(n)] for i in range(n)]
 for o in obs:
  if o['perturbed_node'] not in index or set(o['effects'])!=set(nodes) or any(v not in (0,1) for v in o['effects'].values()):raise ValueError('invalid intervention')
  src=index[o['perturbed_node']]
  for target,value in o['effects'].items():
   if target==o['perturbed_node']:
    if value!=0:s.add(z3.BoolVal(False))
   else:s.add(R[src][index[target]]==bool(value))
 while True:
  remaining=deadline-time.monotonic()
  if remaining<=0:raise TimeoutError('time limit')
  s.set(timeout=max(1,int(remaining*1000)));ans=s.check();stats['solver_checks']+=1
  if ans==z3.unsat:return
  if ans==z3.unknown:raise TimeoutError('solver unknown: '+s.reason_unknown())
  model=s.model();bits={ij:z3.is_true(model.eval(e,model_completion=True)) for ij,e in E.items()};edges=sorted([[nodes[i],nodes[j]] for (i,j),bit in bits.items() if bit]);s.add(z3.Or([e!=bits[ij] for ij,e in E.items()]));obj={'edges':edges};yield json.dumps(obj,sort_keys=True),obj

@lru_cache(maxsize=16)
def _public_boolean(variables,depth,operators):
 sys.path.insert(0,str(ROOT/'code'))
 from boolean_dataset import BooleanDiscoveryGame
 opts={'apply_commutativity':True,'apply_idempotence_and_or':True,'flatten_associativity':True};es=BooleanDiscoveryGame.generate_all_expressions(list(variables),set(operators),depth,opts);out={}
 for e in es:out.setdefault(repr(e.mechanistic_key(**opts)),(e.formula,e.truth_table))
 return tuple((k,*out[k]) for k in sorted(out))

def _boolean(q,deadline,stats):
 variables=tuple(sorted(q.get('variables',['x','y','z'])));depth=q.get('max_depth',2);operators=tuple(sorted(q.get('operators',['AND','OR','NOT'])))
 if not 1<=len(variables)<=3 or len(set(variables))!=len(variables) or not 0<=depth<=2 or set(operators)-{'AND','OR','NOT'}:raise ValueError('outside verified Boolean profile')
 seen={}
 for o in q['observations']:
  if set(o['inputs'])!=set(variables) or any(v not in (0,1) for v in o['inputs'].values()) or o['output'] not in (0,1):raise ValueError('invalid Boolean observation')
  x=tuple(o['inputs'][v] for v in variables)
  if x in seen and seen[x]!=o['output']:return
  seen[x]=o['output']
 for key,formula,truth in _public_boolean(variables,depth,operators):
  if time.monotonic()>deadline:raise TimeoutError('time limit')
  stats['predicate_checks']+=1
  if all(truth[x]==y for x,y in seen.items()):yield key,{'formula':formula}

def solve(task,query,budget,time_limit=30):
 if type(budget)!=int or budget<1 or time_limit<=0:raise ValueError('positive budget/time limit required')
 allowed={'voxel':{'top_view','max_height'},'causal':{'nodes','observations','max_edges'},'boolean':{'variables','observations','operators','max_depth'}}
 if task not in allowed or set(query)-allowed[task]:raise ValueError('unsupported task or query fields')
 start=time.monotonic();deadline=start+time_limit;stats={'solver_checks':0,'predicate_checks':0,'construction_steps':0};iterator={'voxel':_voxel,'causal':_causal,'boolean':_boolean}[task](query,deadline,stats);unique=[];keys=set();complete=False;reason='budget_reached'
 try:
  while len(unique)<budget:
   if time.monotonic()>deadline:raise TimeoutError('time limit')
   key,obj=next(iterator)
   if key not in keys:keys.add(key);unique.append(obj)
 except StopIteration:complete=True;reason='space_exhausted'
 except TimeoutError as e:reason=str(e)
 outputs=list(unique)
 if complete and unique:
  outputs=[unique[i%len(unique)] for i in range(budget)]
 return {'task':task,'outputs':outputs,'unique_outputs':unique,'complete':complete,'reason':reason,'requested':budget,'emitted':len(outputs),'repeats_added_only_after_exhaustion':max(0,len(outputs)-len(unique)),'seconds':time.monotonic()-start,**stats}

def _candidate(task,q,obj):
 """Conservative acceptance; Boolean accepts public canonical surface forms."""
 if not isinstance(obj,dict):raise ValueError('not an object')
 if task=='voxel':
  h=obj['heights'];n=sum(map(sum,q['top_view']))
  if not isinstance(h,list) or len(h)!=n or any(type(x)!=int or not 1<=x<=q['max_height'] for x in h):raise ValueError('invalid height')
  obj={'heights':h};return json.dumps(obj,sort_keys=True),obj
 if task=='causal':
  import networkx as nx
  nodes=q['nodes'];edges=obj['edges']
  if not isinstance(edges,list) or any(not isinstance(e,list) or len(e)!=2 or e[0] not in nodes or e[1] not in nodes for e in edges):raise ValueError('bad edges')
  if len({tuple(e) for e in edges})!=len(edges):raise ValueError('duplicate edges')
  if q.get('max_edges') is not None and len(edges)>q['max_edges']:raise ValueError('too many edges')
  g=nx.DiGraph();g.add_nodes_from(nodes);g.add_edges_from(edges)
  if not nx.is_directed_acyclic_graph(g):raise ValueError('cycle')
  if any({v:int(v in nx.descendants(g,o['perturbed_node'])) for v in nodes}!=o['effects'] for o in q['observations']):raise ValueError('contradicts intervention')
  obj={'edges':sorted(edges)};return json.dumps(obj,sort_keys=True),obj
 import re
 f=obj['formula'];assert isinstance(f,str)
 norm=lambda x:re.sub(r'\s+','',x)
 variables=tuple(sorted(q.get('variables',['x','y','z'])));operators=tuple(sorted(q.get('operators',['AND','OR','NOT'])))
 for key,formula,truth in _public_boolean(variables,q.get('max_depth',2),operators):
  if norm(formula)==norm(f) and all(truth[tuple(o['inputs'][v] for v in variables)]==o['output'] for o in q['observations']):return key,{'formula':formula}
 raise ValueError('not a matching canonical public expression')

def complete_proposals(task,query,proposals,budget,time_limit=30):
 """Preserve valid distinct LLM proposals, fill gaps with certified construction.
 Additional symbolic compute is required; this is NOT a weight-only improvement.
 """
 allowed={'voxel':{'top_view','max_height'},'causal':{'nodes','observations','max_edges'},'boolean':{'variables','observations','operators','max_depth'}}
 if task not in allowed or set(query)-allowed[task] or type(budget)!=int or budget<1:raise ValueError('bad request')
 start=time.monotonic();kept=[];seen=set();rejected=0
 import re
 for proposal in proposals:
  if len(kept)>=budget:break
  try:
   m=re.search(r'\{.*\}',proposal,re.S);obj=json.loads(m.group() if m else proposal);key,obj=_candidate(task,query,obj)
   if key not in seen:seen.add(key);kept.append(obj)
   else:rejected+=1
  except (ValueError,KeyError,TypeError,AssertionError,IndexError):rejected+=1
 original=len(kept);backend=None;complete=False
 if len(kept)<budget:
  # At most original overlaps, so budget+original is sufficient when enough exist.
  backend=solve(task,query,budget+original,max(.001,time_limit-(time.monotonic()-start)));complete=backend['complete']
  for obj in backend['unique_outputs']:
   key,obj=_candidate(task,query,obj)
   if key not in seen:seen.add(key);kept.append(obj)
   if len(kept)>=budget:break
 unique_count=len(kept)
 if complete and kept:
  while len(kept)<budget:kept.append(kept[len(kept)%unique_count])
 return {'outputs':kept,'unique_count':unique_count,'kept_proposals':original,'rejected_or_repeated_proposals':rejected,'repeats':len(kept)-unique_count,'complete':complete,'seconds':time.monotonic()-start,'backend':backend}
