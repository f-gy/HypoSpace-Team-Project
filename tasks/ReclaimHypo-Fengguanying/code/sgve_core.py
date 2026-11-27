import sys,types,time,random,re,itertools,collections
from dataclasses import dataclass
# Scoring does not use scipy statistics or live LLM interfaces.
if 'scipy' not in sys.modules:
 s=types.ModuleType('scipy');s.stats=types.ModuleType('scipy.stats');sys.modules['scipy']=s
if 'modules.llm_interface' not in sys.modules:
 m=types.ModuleType('modules.llm_interface')
 for name in ['LLMInterface','OpenRouterLLM','OpenAILLM','AnthropicLLM']:setattr(m,name,type(name,(),{}))
 sys.modules['modules.llm_interface']=m
import sympy as sp
from boolean_benchmark import BooleanBenchmarkRefined
from boolean_dataset import BooleanObservation,BooleanDiscoveryGame,BooleanExpression
from hypospace_validator import install,generated_universe,_parse_checked,VERSION

DATA='outputs/HypoSpace_候选筛选/datasets/boolean_extended_depth2.json'
b=BooleanBenchmarkRefined(DATA); universe=generated_universe(b)
grammar=install(BooleanBenchmarkRefined(DATA),policy='grammar'); strict=install(BooleanBenchmarkRefined(DATA),policy='enumerated',universe=universe)
# Remove case-specific datasets from every validator before any generation.
for _validator in (b,grammar,strict):
 _validator.complete_dataset = None
 _validator.all_observation_sets = []

def objects(obs): return [BooleanObservation(x['inputs'],x['output']) for x in obs]
def key(expr): return repr(grammar.get_expression_mechanistic_key(expr))
def parse(expr): return _parse_checked(grammar,expr)
def serialize(node):
 if isinstance(node,sp.Symbol):return str(node)
 if isinstance(node,sp.Not):return '(NOT '+serialize(node.args[0])+')'
 if isinstance(node,sp.And):return '('+' AND '.join(serialize(x) for x in node.args)+')'
 if isinstance(node,sp.Or):return '('+' OR '.join(serialize(x) for x in node.args)+')'
 raise ValueError('unsupported AST')
def paths(node,p=()):
 yield p,node
 for i,ch in enumerate(node.args):yield from paths(ch,p+(i,))
def replace(node,path,text):
 if not path:return text
 parts=[replace(ch,path[1:],text) if i==path[0] else serialize(ch) for i,ch in enumerate(node.args)]
 if isinstance(node,sp.Not):return '(NOT '+parts[0]+')'
 if isinstance(node,sp.And):return '('+' AND '.join(parts)+')'
 if isinstance(node,sp.Or):return '('+' OR '.join(parts)+')'
 raise ValueError('invalid parent')

def descriptors(expr,group='all'):
 e=parse(expr)
 if e is None:raise ValueError('invalid parent')
 buckets={'equiv':[],'perturb':[]}
 for path,node in paths(e.sympy_expr):
  # Rules: double negation, absorption u ∨ (u ∧ v)=u, u ∧ (u ∨ v)=u, and commutation.
  buckets['equiv'].append(('double_neg',path,None))
  for v in ('x','y'):
   buckets['equiv'].append(('absorb_or',path,v));buckets['equiv'].append(('absorb_and',path,v))
  if isinstance(node,(sp.And,sp.Or)) and len(node.args)>=2:buckets['equiv'].append(('commute_reverse',path,None))
  if isinstance(node,sp.Not) and isinstance(node.args[0],(sp.And,sp.Or)):buckets['equiv'].append(('demorgan_expand',path,None))
  if isinstance(node,(sp.And,sp.Or)) and len(node.args)>=2 and all(isinstance(a,sp.Not) for a in node.args):buckets['equiv'].append(('demorgan_contract',path,None))
  if isinstance(node,sp.Symbol):buckets['perturb'].append(('replace_var',path,None))
  if isinstance(node,(sp.And,sp.Or)):buckets['perturb'].append(('flip_operator',path,None))
 buckets['perturb'].append(('swap_all',(),None))
 if group=='equiv': return buckets['equiv']
 if group=='perturb': return buckets['perturb']
 # Fixed fair interleaving between transformation classes.
 return [x for pair in itertools.zip_longest(buckets['equiv'],buckets['perturb']) for x in pair if x is not None]

def transform(expr,desc):
 rule,path,param=desc;e=parse(expr)
 if e is None:raise ValueError('invalid parent')
 node=e.sympy_expr
 for ix in path:node=node.args[ix]
 u=serialize(node)
 if rule=='double_neg':new='(NOT (NOT '+u+'))'
 elif rule=='absorb_or':new='('+u+' OR ('+u+' AND '+param+'))'
 elif rule=='absorb_and':new='('+u+' AND ('+u+' OR '+param+'))'
 elif rule=='commute_reverse':new='('+(' AND ' if isinstance(node,sp.And) else ' OR ').join(serialize(a) for a in reversed(node.args))+')'
 elif rule=='demorgan_expand':
  n=node.args[0];op=' OR ' if isinstance(n,sp.And) else ' AND ';new='('+op.join('(NOT '+serialize(x)+')' for x in n.args)+')'
 elif rule=='demorgan_contract':
  op=' OR ' if isinstance(node,sp.And) else ' AND ';new='(NOT ('+op.join(serialize(x.args[0]) for x in node.args)+'))'
 elif rule=='replace_var':new='y' if str(node)=='x' else 'x'
 elif rule=='flip_operator':new='('+(' OR ' if isinstance(node,sp.And) else ' AND ').join(serialize(x) for x in node.args)+')'
 elif rule=='swap_all':return re.sub(r'\b(?:x|y)\b',lambda m:'y' if m.group()=='x' else 'x',expr)
 else:raise ValueError('unknown transform')
 return replace(e.sympy_expr,path,new)

@dataclass
class Budgets:
 attempts:int=30
 hop:int=1
 group:str='all'
 order:str='fixed'
 seed:int=0

def expand(obs,seeds,originals,config,case_id=''):
 # Separate accepted, queued, expanded: accepted includes all valid original LLM outputs.
 accept={key(e):e for e in originals}; queue=collections.deque();queued=set();expanded=set();events=[];nvalid=0
 for index,e in enumerate(seeds):
  k=key(e)
  if k not in queued:queue.append({'expr':e,'key':k,'hop':0,'source':index,'root_key':k,'ds':None,'idx':0});queued.add(k)
 rng=random.Random(config.seed);start=time.perf_counter();cpu0=time.process_time(); validation_calls=0
 while queue and len(events)<config.attempts:
  state=queue.popleft();queued.remove(state['key']);expanded.add(state['key'])
  if state['ds'] is None:
   ds=descriptors(state['expr'],config.group)
   if config.order=='random':rng.shuffle(ds)
   state['ds']=ds
  if state['idx']>=len(state['ds']):continue
  desc=state['ds'][state['idx']];state['idx']+=1
  if state['idx']<len(state['ds']):queue.append(state);queued.add(state['key'])
  t=time.perf_counter();candidate=None;error=None
  try:candidate=transform(state['expr'],desc)
  except Exception as e:error=type(e).__name__+': '+str(e)
  transform_s=time.perf_counter()-t; ev={'case_id':case_id,'attempt':len(events)+1,'source_seed_index':state['source'],'source_seed_key':state['root_key'],'parent_expr':state['expr'],'parent_key':state['key'],'hop':state['hop']+1,'rule':desc[0],'path':list(desc[1]),'param':desc[2],'candidate':candidate,'transform_seconds':transform_s,'validation_seconds':0.,'validation_calls':0,'grammar_valid':False,'strict_valid':False,'candidate_key':None,'truth_table':None,'new':False,'reason':None,'transform_error':error}
  if candidate is None:ev['reason']='transform_failed'
  else:
   t=time.perf_counter();g,_=grammar.validate_expression(candidate,objects(obs));s,_=strict.validate_expression(candidate,objects(obs));ev['validation_seconds']=time.perf_counter()-t;ev['validation_calls']=2;validation_calls+=2;ev['grammar_valid']=bool(g);ev['strict_valid']=bool(s)
   parsed=parse(candidate)
   if parsed is not None:ev['candidate_key']=key(candidate);ev['truth_table']={str(k):v for k,v in parsed.truth_table.items()}
   if not g:ev['reason']='grammar_or_observation_invalid'
   elif not s:ev['reason']='valid_grammar_outside_enumeration'
   else:
    nvalid+=1;k=ev['candidate_key'];ev['new']=k not in accept;ev['reason']='accepted_new' if ev['new'] else 'accepted_duplicate'
    if ev['new']:accept[k]=candidate
    # Whether already accepted is independent of whether it has ever been queued or expanded.
    if state['hop']+1<config.hop and k not in queued and k not in expanded:
     queue.append({'expr':candidate,'key':k,'hop':state['hop']+1,'source':state['source'],'root_key':state['root_key'],'ds':None,'idx':0});queued.add(k)
  events.append(ev)
 return {'accepted':accept,'events':events,'attempts':len(events),'validation_calls':validation_calls,'elapsed':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu0,'expanded_keys':sorted(expanded),'queued_keys':sorted(queued),'valid_attempts':nvalid}

def public_expressions():
 return sorted((e.formula for e in BooleanDiscoveryGame.generate_all_expressions(b.variables,b.operators,b.max_depth,b.mechanistic_opts)),key=lambda s:(len(re.findall(r'\b(?:AND|OR|NOT)\b',s)),s))
