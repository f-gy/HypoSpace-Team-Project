"""Mac MLX implementation of verified canonical-coverage adapter training."""
from __future__ import annotations
import argparse, copy, gc, hashlib, json, math, random, time
from pathlib import Path
import numpy as np
from .upstream import ROOT, evaluator, load as upstream_load
from .tasks import public_task, write_json
from .experiment import assess, episode, read_rows

RUN=ROOT/'runs/final_v1'
MODEL=ROOT.parent/'runtime/models/Qwen2.5-1.5B-Instruct-4bit'
LORA={'rank':8,'scale':16.0,'dropout':0.0,'keys':['self_attn.q_proj','self_attn.v_proj']}
METHODS=['base_iid','balanced_sft_iid','canonical_coverage_iid','string_coverage_iid','base_memory']


def dataset():
    d=json.loads((ROOT/'.cache/tasks.json').read_text())
    d['splits']=json.loads((ROOT/'results/coverage_plan/split_manifest.json').read_text())['splits']
    return d


def model_load():
    from mlx_lm import load
    return load(str(MODEL))


def prompt_for(task,d):
    mod,_=upstream_load()
    return evaluator(d['metadata']).create_prompt([mod.BooleanObservation(o['inputs'],o['output']) for o in task['observations']],[])


def prepare():
    _,tok=model_load(); d=dataset(); groups=[]; empty=[]
    ev=evaluator(d['metadata']); universe={e['key']:e['formula'] for e in d['universe']}
    for task in d['tasks']:
        if task['id'] not in d['splits']['train']:continue
        prompt=prompt_for(task,d)
        messages=[{'role':'user','content':prompt}]
        prefix=tok.apply_chat_template(messages,add_generation_prompt=True,tokenize=True)
        local=[]
        for key in task['reference_keys']:
            formula=universe[key]; aliases=[]; seen=set()
            for expr in [formula,'('+formula+')']:
                result=assess(ev,d['metadata'],public_task(task),'Expression: '+expr,[])
                if not result['valid'] or result['key']!=key:continue
                answer='Expression: '+expr
                full=tok.apply_chat_template(messages+[{'role':'assistant','content':answer}],add_generation_prompt=False,tokenize=True)
                assert full[:len(prefix)]==prefix
                stop=next(i for i in range(len(prefix),len(full)) if full[i] in tok.eos_token_ids)
                full=full[:stop+1]
                if tuple(full) in seen:continue
                seen.add(tuple(full)); aliases.append({'answer':answer,'tokens':full,'prompt_length':len(prefix)})
            if not aliases:
                empty.append({'task_id':task['id'],'key':key});continue
            local.append({'task_id':task['id'],'key':key,'aliases':aliases,'reference_count':len(task['reference_keys'])})
        for g in local:g['class_ratio']=len(local)/g['reference_count']
        groups.extend(local)
    assert groups and max(len(a['tokens']) for g in groups for a in g['aliases'])<=1024
    result={'groups':groups,'empty_groups':empty,'training_ids':d['splits']['train'],'note':'Only training references are used for optimization.'}
    write_json(RUN/'training_data.json',result)
    print(json.dumps({'groups':len(groups),'aliases':sum(len(g['aliases']) for g in groups),'empty_groups':len(empty),'max_tokens':max(len(a['tokens']) for g in groups for a in g['aliases'])}),flush=True)


def batch(group):
    import mlx.core as mx
    aliases=group['aliases'];n=max(len(a['tokens']) for a in aliases)
    ids=np.zeros((len(aliases),n),dtype=np.int32); mask=np.zeros((len(aliases),n-1),dtype=np.float32)
    for i,a in enumerate(aliases):
        ids[i,:len(a['tokens'])]=a['tokens']
        mask[i,a['prompt_length']-1:len(a['tokens'])-1]=1
    return mx.array(ids),mx.array(mask),mx.array(group['class_ratio'],dtype=mx.float32)


def coverage_value(logp,ratio,kind):
    import mlx.core as mx
    probs=mx.exp(logp)
    if kind=='canonical_coverage':probs=probs.sum()[None]
    p=mx.clip(probs,0.0,1.0-1e-7)
    return (-mx.expm1(16*mx.log1p(-p))).sum()*ratio


def loss_function(kind):
    import mlx.core as mx
    import mlx.nn as nn
    def loss(model,ids,mask,ratio):
        logits=model(ids[:,:-1])
        ce=nn.losses.cross_entropy(logits,ids[:,1:],reduction='none').astype(mx.float32)*mask
        nll=ce.sum(axis=1)
        anchor=(nll/mask.sum(axis=1)).mean()
        logp=-nll
        if kind=='balanced_sft':return anchor
        coverage=coverage_value(logp,ratio,kind)
        return anchor-coverage
    return loss


def train():
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten,tree_map
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx_lm.tuner.trainer import grad_checkpoint
    data=json.loads((RUN/'training_data.json').read_text());groups=data['groups']
    mx.random.seed(4300);model,_=model_load();model.freeze()
    linear_to_lora_layers(model,8,LORA);grad_checkpoint(model.layers[0]);model.train()
    leaves=tree_flatten(model.trainable_parameters())
    assert leaves and all(n.endswith(('lora_a','lora_b')) for n,_ in leaves)
    by_task={}
    for g in groups:by_task.setdefault(g['task_id'],[]).append(g)
    rng=random.Random(4300);tasks=sorted(by_task)
    schedule=[[rng.choice(by_task[rng.choice(tasks)]) for _ in range(4)] for _ in range(200)]
    optimizer=optim.Adam(learning_rate=1e-4)
    RUN.mkdir(parents=True,exist_ok=True)
    log=RUN/'training_log.jsonl'
    if log.exists():raise RuntimeError('Training run already exists; retain it and use an explicit new version before retraining')
    base_leaves={n:v for n,v in tree_flatten(model.parameters()) if not n.endswith(('lora_a','lora_b'))}
    # Keep references to all frozen tensors; check their equality after training.
    snapshot={n:v for n,v in leaves}
    timing=[]; totals={'warmup':80,'continuation':120}; frozen_check={}
    def save(name):
        dest=RUN/'adapters'/name;dest.mkdir(parents=True,exist_ok=True)
        mx.save_safetensors(str(dest/'adapters.safetensors'),dict(tree_flatten(model.trainable_parameters())))
        write_json(dest/'adapter_config.json',{'fine_tune_type':'lora','num_layers':8,'lora_parameters':LORA})
    def updates(kind,start,end):
        fn=nn.value_and_grad(model,loss_function(kind))
        for step in range(start,end):
            tic=time.monotonic();accum=None;values=[];ntokens=0
            for g in schedule[step]:
                b=batch(g);value,grad=fn(model,*b)
                accum=grad if accum is None else tree_map(lambda x,y:x+y,accum,grad)
                values.append(value);ntokens+=sum(len(a['tokens']) for a in g['aliases'])
                mx.eval(value,accum)
            accum=tree_map(lambda x:x/4,accum)
            assert all(bool(mx.all(mx.isfinite(v)).item()) for _,v in tree_flatten(accum))
            optimizer.update(model,accum);mx.eval(model.parameters(),optimizer.state)
            value=float(mx.stack(values).mean().item());assert math.isfinite(value)
            seconds=time.monotonic()-tic;timing.append(seconds)
            row={'phase':kind,'step':step+1,'loss':value,'seconds':seconds,'sequence_tokens':ntokens,'peak_memory_gb':mx.get_peak_memory()/1e9}
            with log.open('a') as f:f.write(json.dumps(row)+'\n')
            if (step+1)%5==0:print(json.dumps(row),flush=True)
            if kind=='balanced_sft' and step+1==20 and start==0:
                projected=np.mean(timing[5:])*440
                if projected>10800:totals.update(warmup=40,continuation=60)
                write_json(RUN/'timing_probe.json',{'updates':20,'mean_update_seconds':float(np.mean(timing[5:])),'projected_440_update_seconds':float(projected),'selected_budgets':totals,'reason':'Resource-only timing decision before adapter validation or final tests.'})
    # Probe contributes to warm-up; all branches retain matched schedules.
    updates('balanced_sft',0,20);updates('balanced_sft',20,totals['warmup'])
    save('warmup')
    warm_weights={n:mx.array(v) for n,v in tree_flatten(model.trainable_parameters())}
    warm_optimizer=copy.deepcopy(optimizer.state)
    for kind in ['balanced_sft','canonical_coverage','string_coverage']:
        model.load_weights(list(warm_weights.items()),strict=False)
        optimizer=optim.Adam(learning_rate=1e-4);optimizer.state=copy.deepcopy(warm_optimizer)
        mx.random.seed(4300)
        updates(kind,totals['warmup'],totals['warmup']+totals['continuation'])
        save(kind)
    frozen_current={n:v for n,v in tree_flatten(model.parameters()) if n in base_leaves}
    assert all(bool(mx.all(v==frozen_current[n]).item()) for n,v in base_leaves.items())
    changed=any(not bool(mx.all(v==dict(tree_flatten(model.trainable_parameters()))[n]).item()) for n,v in snapshot.items())
    assert changed
    write_json(RUN/'training_summary.json',{'budgets':totals,'trainable_parameters':sum(v.size for _,v in leaves),'frozen_tensors_unchanged':True,'adapter_changed':changed,'training_tasks':len(by_task),'nonempty_classes':len(groups),'empty_classes':len(data['empty_groups']),'runtime':'mlx-lm 0.32.2 / Apple M5','peak_memory_gb':mx.get_peak_memory()/1e9})


class DirectBackend:
    kind='real'
    def __init__(self,model,tok,config,method):
        self.net=model;self.tok=tok;self.config=config;self.model=config['model_identity'];self.method=method
        self.path=RUN/'generation_ledger.json'
        self.ledger=json.loads(self.path.read_text()) if self.path.exists() else {'calls':0,'tokens':0}
    def query(self,prompt,action,task,rng):
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        seed=rng.getrandbits(31);mx.random.seed(seed)
        ids=self.tok.apply_chat_template([{'role':'user','content':prompt}],add_generation_prompt=True,tokenize=True)
        if self.ledger['calls']>=7500:raise RuntimeError('Generation attempt cap reached')
        self.ledger['calls']+=1;write_json(self.path,self.ledger)
        tic=time.monotonic();parts=[];last=None
        try:
            for last in stream_generate(self.net,self.tok,ids,max_tokens=128,sampler=make_sampler(temp=1.0,top_p=1.0,top_k=0,min_p=0.0)):
                parts.append(last.text)
            usage={'prompt_tokens':len(ids),'completion_tokens':last.generation_tokens,'total_tokens':len(ids)+last.generation_tokens}
            self.ledger['tokens']+=usage['total_tokens'];write_json(self.path,self.ledger)
            return {'text':''.join(parts),'usage':usage,'error':None,'seconds':time.monotonic()-tic,'request_seed':seed}
        except Exception as e:
            return {'text':'','usage':{},'error':type(e).__name__,'seconds':time.monotonic()-tic,'request_seed':seed}


def evaluate(split):
    import mlx.core as mx
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx.utils import tree_flatten
    d=dataset();model,tok=model_load();model.freeze();linear_to_lora_layers(model,8,LORA);model.eval()
    zero={n:mx.zeros_like(v) for n,v in tree_flatten(model.trainable_parameters())}
    weights={'base_iid':zero,'base_memory':zero}
    for kind in ['balanced_sft','canonical_coverage','string_coverage']:
        weights[kind+'_iid']=mx.load(str(RUN/'adapters'/kind/'adapters.safetensors'))
    p=json.loads((ROOT/'results/coverage_plan/plan.json').read_text()); lock=json.loads((ROOT/'model.local_mlx.lock.json').read_text())
    common={'model_identity':lock['repo_id'],'model_provenance':lock,'temperature':1.0,'max_output_tokens':128,'top_p':1.0,'top_k':0,'min_p':0.0,'send_seed':True,'adapter_training':json.loads((RUN/'training_summary.json').read_text()),'engine':'direct mlx_lm.stream_generate','split_manifest_sha256':hashlib.sha256((ROOT/'results/coverage_plan/split_manifest.json').read_bytes()).hexdigest()}
    seeds=[101] if split=='validation' else [17,29,43]
    by_id={t['id']:t for t in d['tasks']}
    configs={}
    for method in METHODS:
        cfg=dict(common,method=method)
        if method not in ['base_iid','base_memory']:
            name=method.removesuffix('_iid');cfg['adapter_sha256']=hashlib.sha256((RUN/'adapters'/name/'adapters.safetensors').read_bytes()).hexdigest()
        configs[method]=cfg
    write_json(RUN/('evaluation_config_'+split+'.json'),configs)
    for tid in d['splits'][split]:
        for seed in seeds:
            order=list(METHODS);random.Random(seed+int(tid,16)).shuffle(order)
            for method in order:
                model.load_weights(list(weights[method].items()),strict=False);mx.eval(model.parameters())
                backend=DirectBackend(model,tok,configs[method],method)
                episode(by_id[tid],d,configs[method],backend,method,seed,16,RUN,split)
            print(f'Completed {split} task {tid} repetition {seed}',flush=True)
    print(f'{split} completed',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','train','validation','test'])
    args=parser.parse_args();RUN.mkdir(parents=True,exist_ok=True)
    {'prepare':prepare,'train':train,'validation':lambda:evaluate('validation'),'test':lambda:evaluate('test')}[args.command]()
if __name__=='__main__':main()
