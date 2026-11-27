import json,tempfile,os,collections,statistics,csv,random
from pathlib import Path
import sgve_core as c
from boolean_dataset import BooleanObservation,BooleanDiscoveryGame,BooleanExpression
from hypospace_validator import generated_universe,_parse_checked,TOKEN_RE
from extract import extract
p=Path('work/lora_balanced');dev={r['id']:r for r in map(json.loads,(p/'dev.jsonl').read_text().splitlines())};meta={'metadata':{'variables':['x','y','z'],'operators':['AND','OR','NOT'],'max_depth':2,'mechanistic_opts':c.b.mechanistic_opts},'datasets_by_n_observations':{}}
with tempfile.NamedTemporaryFile(mode='w',delete=False) as f:json.dump(meta,f);tmp=f.name
try:
 b=c.BooleanBenchmarkRefined(tmp);v=c.install(b,policy='enumerated',universe=generated_universe(b));g=c.install(c.BooleanBenchmarkRefined(tmp),policy='grammar');es=BooleanDiscoveryGame.generate_all_expressions(['x','y','z'],{'AND','OR','NOT'},2,b.mechanistic_opts);out=[];methods=['base','oldplain','plain_1337','weighted_1337','plain_31415','weighted_31415']
 for mode in methods:
  for r in map(json.loads,(p/('dev_'+mode+'.jsonl')).read_text().splitlines()):
   obs=[BooleanObservation(o['inputs'],o['output']) for o in dev[r['id']]['observations']];anchor=next(o.output for o in obs if tuple(o.inputs[n] for n in ['x','y','z'])==(0,0,0));gold={repr(e.mechanistic_key(**b.mechanistic_opts)):e for e in es if all(e.truth_table[tuple(o.inputs[n] for n in ['x','y','z'])]==o.output for o in obs)};gb={tuple(e.truth_table.values()) for e in gold.values()};valid=[];allkeys=set();details=[]
   for raw in r['outputs']:
    x=extract(raw);toks=TOKEN_RE.findall(x);unknown=[t for t in toks if t not in {'x','y','z','AND','OR','NOT','(',')'}];parse=BooleanExpression(x,['x','y','z'],{'AND','OR','NOT'}) if x and not unknown else None;e=_parse_checked(g,x);gv=bool(e);sat=bool(gv and g.validate_expression(x,obs)[0]);sp=bool(gv and v._in_space(x));ok=sp and sat;k=repr(v.get_expression_mechanistic_key(x)) if sp else None
    if sp:allkeys.add(k)
    if ok:valid.append(k)
    if not x or (parse is not None and parse.sympy_expr is None):reason='empty_or_unparseable'
    elif unknown:reason='forbidden_token_or_prose'
    elif not gv:reason='depth_or_grammar'
    elif not sat:reason='observation_mismatch'
    elif not sp:reason='valid_grammar_outside_enumeration'
    else:reason='strict_valid'
    details.append({'raw':raw,'candidate':x,'parseable':bool(parse and parse.sympy_expr is not None),'grammar':gv,'obs_satisfied':sat,'strict_space':sp,'strict_valid':ok,'anchor_pred':e.truth_table[(0,0,0)] if gv else None,'anchor_wrong':bool(gv and e.truth_table[(0,0,0)]!=anchor),'key':k,'reason':reason})
   found=set(valid)&gold.keys();bh={tuple(gold[k].truth_table.values()) for k in found};out.append({'id':r['id'],'seed':r['seed'],'mode':mode,'anchor':anchor,'n_observations':len(obs),'validity':len(valid)/10,'recovery':len(found)/len(gold),'behavior_coverage':len(bh)/len(gb),'unique_valid':len(found),'valid_count':len(valid),'duplicate_count':len(valid)-len(set(valid)),'novelty_rate':len(allkeys)/10,'grammar_count':sum(x['grammar'] for x in details),'obs_satisfied_count':sum(x['obs_satisfied'] for x in details),'anchor_wrong_count':sum(x['anchor_wrong'] for x in details),'input_tokens':r['input_tokens_per_candidate']*10,'output_tokens':sum(r['output_tokens']),'generation_seconds':sum(r['batch_seconds']),'cpu_seconds':sum(r['batch_cpu_seconds']),'gpu_stream_seconds':sum(r['batch_gpu_stream_seconds']),'validation_calls':sum(x['grammar'] for x in details),'space_checks':sum(x['grammar'] for x in details),'details':details})
 summary={}; metric=['validity','recovery','behavior_coverage','unique_valid','novelty_rate','input_tokens','output_tokens','generation_seconds','cpu_seconds','gpu_stream_seconds']
 for m in methods:
  summary[m]={}
  for anchor in ('all',0,1):
   rows=[x for x in out if x['mode']==m and (anchor=='all' or x['anchor']==anchor)];counts=collections.Counter(d['reason'] for x in rows for d in x['details']);grammar=sum(x['grammar_count'] for x in rows);valid=sum(x['valid_count'] for x in rows)
   summary[m][str(anchor)]={**{k:statistics.mean(x[k] for x in rows) for k in metric},'records':len(rows),'candidate_count':len(rows)*10,'valid_candidates':valid,'parseable':sum(d['parseable'] for x in rows for d in x['details']),'grammar_count':grammar,'obs_satisfied_count':sum(x['obs_satisfied_count'] for x in rows),'anchor_wrong_count':sum(x['anchor_wrong_count'] for x in rows),'duplicate_valid':sum(x['duplicate_count'] for x in rows),'duplicate_rate':sum(x['duplicate_count'] for x in rows)/valid if valid else None,'validation_calls':sum(x['validation_calls'] for x in rows),'space_checks':sum(x['space_checks'] for x in rows),'failures':dict(counts),'per_generation_seed_recovery':{str(s):statistics.mean(x['recovery'] for x in rows if x['seed']==s) for s in (101,202,303)}}
 index={(x['mode'],x['id'],x['seed']):x for x in out};ids=sorted(dev);comp={}
 for tr in (1337,31415):
  vals=[statistics.mean(index['weighted_'+str(tr),i,s]['recovery']-index['plain_'+str(tr),i,s]['recovery'] for s in (101,202,303)) for i in ids];rng=random.Random(20261125);z=sorted(statistics.mean(rng.choices(vals,k=len(vals))) for _ in range(5000));comp[str(tr)]={'mean':statistics.mean(vals),'bootstrap95':[z[125],z[4874]],'positive_items':sum(x>1e-12 for x in vals),'negative_items':sum(x< -1e-12 for x in vals),'ties':sum(abs(x)<=1e-12 for x in vals),'generation_seed_delta':{str(s):statistics.mean(index['weighted_'+str(tr),i,s]['recovery']-index['plain_'+str(tr),i,s]['recovery'] for i in ids) for s in (101,202,303)}}
 (p/'dev_scored.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in out));(p/'dev_summary.json').write_text(json.dumps({'groups':summary,'comparisons':comp},indent=2));
 with (p/'dev_paired.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['id','sampling_seed','anchor',*methods]);
  for i in ids:
   for s in (101,202,303):w.writerow([i,s,index['base',i,s]['anchor'],*[index[m,i,s]['recovery'] for m in methods]])
 print(json.dumps({'groups':summary,'comparisons':comp},indent=2))
finally:os.unlink(tmp)
