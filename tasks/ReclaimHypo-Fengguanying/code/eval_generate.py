import json,time,hashlib,sys,resource
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import PeftModel
mode=sys.argv[1];assert mode in ('base','oldplain','plain_1337','weighted_1337','plain_31415','weighted_31415');root=Path('/root/hypospace-project/lora_balanced');rows=list(map(json.loads,(root/'dev.jsonl').read_text().splitlines()));tok=AutoTokenizer.from_pretrained('/root/hypospace-project/models/Qwen2.5-7B-Instruct',local_files_only=True);start=time.perf_counter();model=AutoModelForCausalLM.from_pretrained('/root/hypospace-project/models/Qwen2.5-7B-Instruct',torch_dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=True,device_map={'':0}).eval()
if mode=='oldplain':model=PeftModel.from_pretrained(model,'/root/hypospace-project/lora_7b/adapter_plain').eval()
elif mode!='base':model=PeftModel.from_pretrained(model,root/('adapter_'+mode)).eval()
model.config.use_cache=True;torch.cuda.synchronize();load=time.perf_counter()-start;torch.cuda.reset_peak_memory_stats();print('LOADED',mode,load,flush=True)
with (root/('dev_'+mode+'.jsonl')).open('w') as f:
 for seed in [101,202,303]:
  for i,row in enumerate(rows):
   lines=['x=%s, y=%s, z=%s -> %s'%(o['inputs']['x'],o['inputs']['y'],o['inputs']['z'],o['output']) for o in row['observations']];p='Find one Boolean expression consistent with every observation. Use only x, y, z, AND, OR, NOT and parentheses, maximum expression depth 2. Output only the expression.\nObservations:\n'+'\n'.join(lines);prompt=tok.apply_chat_template([{'role':'system','content':'You are a precise assistant.'},{'role':'user','content':p}],tokenize=False,add_generation_prompt=True);inp=tok(prompt,return_tensors='pt').to('cuda');derived=int.from_bytes(hashlib.sha256((str(seed)+row['id']).encode()).digest()[:4],'big');torch.manual_seed(derived);torch.cuda.manual_seed_all(derived);output=[];outtokens=[];elapsed=[];cpu=[];gpu=[]
   for batch in (5,5):
    torch.cuda.synchronize();t0=time.perf_counter();cpu0=time.process_time();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
    with torch.inference_mode():ans=model.generate(**inp,do_sample=True,temperature=.7,top_p=1.,top_k=20,repetition_penalty=1.05,max_new_tokens=32,num_return_sequences=batch,pad_token_id=tok.eos_token_id)
    b.record();torch.cuda.synchronize();elapsed.append(time.perf_counter()-t0);cpu.append(time.process_time()-cpu0);gpu.append(a.elapsed_time(b)/1000);out=ans[:,inp['input_ids'].shape[1]:];output+=tok.batch_decode(out,skip_special_tokens=True);eos=tok.eos_token_id;outtokens.extend([(x.index(eos)+1 if eos in x else len(x)) for x in out.tolist()])
   f.write(json.dumps({'id':row['id'],'seed':seed,'sample_seed':derived,'mode':mode,'prompt':prompt,'outputs':output,'input_tokens_per_candidate':inp['input_ids'].shape[1],'output_tokens':outtokens,'batch_seconds':elapsed,'batch_cpu_seconds':cpu,'batch_gpu_stream_seconds':gpu,'load_seconds':load,'peak_gpu_allocated':torch.cuda.max_memory_allocated()})+'\n');f.flush()
  print('DONE',mode,seed,'elapsed',round(time.perf_counter()-start,1),flush=True)
