import json,time,math,statistics,resource
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
root=Path('/root/hypospace-project/lora_balanced');rows=list(map(json.loads,(root/'train.jsonl').read_text().splitlines()));path='/root/hypospace-project/models/Qwen2.5-7B-Instruct';t0=time.perf_counter();tok=AutoTokenizer.from_pretrained(path,local_files_only=True);model=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.float16,local_files_only=True,low_cpu_mem_usage=True,device_map={'':0}).eval();torch.cuda.synchronize();load=time.perf_counter()-t0;torch.cuda.reset_peak_memory_stats();print('MODEL_LOADED',round(load,2),flush=True)
def prompt(row):
 lines=['x=%s, y=%s, z=%s -> %s'%(o['inputs']['x'],o['inputs']['y'],o['inputs']['z'],o['output']) for o in row['observations']];p='Find one Boolean expression consistent with every observation. Use only x, y, z, AND, OR, NOT and parentheses, maximum expression depth 2. Output only the expression.\nObservations:\n'+'\n'.join(lines)
 return tok.apply_chat_template([{'role':'system','content':'You are a precise assistant.'},{'role':'user','content':p}],tokenize=False,add_generation_prompt=True)
with (root/'likelihood.jsonl').open('w') as out:
 for i,row in enumerate(rows):
  p=prompt(row);prefix=tok.encode(p,add_special_tokens=False);records=[]
  for start in range(0,len(row['targets']),4):
   batch=row['targets'][start:start+4];items=[prefix+tok.encode(t['expression'],add_special_tokens=False)+[tok.eos_token_id] for t in batch];maxlen=max(map(len,items));tokens=torch.tensor([x+[tok.pad_token_id]*(maxlen-len(x)) for x in items],device='cuda');mask=torch.tensor([[1]*len(x)+[0]*(maxlen-len(x)) for x in items],device='cuda');torch.cuda.synchronize();t1=time.perf_counter()
   with torch.inference_mode():logits=model(input_ids=tokens,attention_mask=mask).logits
   torch.cuda.synchronize();dt=time.perf_counter()-t1
   for j,(t,ids) in enumerate(zip(batch,items)):
    ans=ids[len(prefix):];chosen=logits[j,len(prefix)-1:len(ids)-1,:].float().log_softmax(-1);target=torch.tensor(ans,device='cuda');lp=chosen.gather(1,target[:,None]).sum().item();records.append({'expression':t['expression'],'key':t['mechanistic_key'],'token_logp_sum':lp,'n_target_tokens':len(ans),'mean_logp':lp/len(ans),'input_tokens':len(ids),'forward_seconds_share':dt/len(batch)})
   del logits
  vals=[r['mean_logp'] for r in records];mean=statistics.mean(vals);sd=statistics.pstdev(vals);ws=[min(2,max(.5,math.exp(.5*(mean-r['mean_logp'])/max(sd,.25)))) for r in records];mw=statistics.mean(ws)
  for r,w in zip(records,ws):r['raw_clipped_weight']=w;r['weight']=w/mw
  out.write(json.dumps({'id':row['id'],'prompt':p,'prompt_tokens':len(prefix),'targets':records})+'\n');out.flush();print('item',i+1,'/',len(rows),'elapsed',round(time.perf_counter()-t0,1),flush=True)
print('SUMMARY',json.dumps({'wall_seconds':time.perf_counter()-t0,'load_seconds':load,'peak_gpu_allocated':torch.cuda.max_memory_allocated(),'peak_gpu_reserved':torch.cuda.max_memory_reserved(),'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}),flush=True)
