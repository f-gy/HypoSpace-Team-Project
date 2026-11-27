"""COVE: a compositional neural executor learned from the original train split."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from functools import partial

import numpy as np

from .coverage_lora import dataset
from .posterior_bank import bank,numpy_predictions
from .tasks import write_json
from .upstream import ROOT

OUT=ROOT/'results/all80'
RUN=ROOT/'runs/all80'
SEEDS=[4306,4307,4308]
STEPS=2000


def make_model():
    import mlx.core as mx
    import mlx.nn as nn
    class Executor(nn.Module):
        def __init__(self):
            super().__init__()
            self.op_and=nn.Sequential(nn.Linear(2,16),nn.Tanh(),nn.Linear(16,1))
            self.op_or=nn.Sequential(nn.Linear(2,16),nn.Tanh(),nn.Linear(16,1))
            self.op_not=nn.Sequential(nn.Linear(1,16),nn.Tanh(),nn.Linear(16,1))
        def __call__(self,trees,inputs):
            memo={}
            def walk(t):
                if t in memo:return memo[t]
                if t[0]=='VAR':value=inputs[:,0 if t[1]=='x' else 1]
                elif t[0]=='NOT':value=mx.sigmoid(self.op_not(walk(t[1])[:,None])[:,0])
                else:
                    a,b=walk(t[1]),walk(t[2]);x=mx.stack([a,b],axis=-1)
                    net=self.op_and if t[0]=='AND' else self.op_or
                    value=mx.sigmoid((net(x)[:,0]+net(x[:,::-1])[:,0])/2)
                memo[t]=value
                return value
            return mx.stack([walk(t) for t in trees])
    return Executor()


def training_arrays(d,items):
    by_id={t['id']:t for t in d['tasks']};keys=[h['key'] for h in items]
    positives=[];ys=[];obs=[]
    for tid in d['splits']['train']:
        task=by_id[tid];refs=set(task['reference_keys'])
        ys.append([float(key in refs) for key in keys]);obs.append(task['observations'])
        for j,key in enumerate(keys):
            if key in refs:
                for o in task['observations']:
                    positives.append([j,2*o['inputs']['x']+o['inputs']['y'],o['output']])
    return np.asarray(positives,dtype=np.int32),np.asarray(ys,dtype=np.float32),obs


def train():
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    d=dataset();items=bank();trees=[h['tree'] for h in items]
    positive,validity,observations=training_arrays(d,items)
    RUN.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'executor_training_summary.json').exists():raise RuntimeError('Retain completed executor study')
    protocol={'training_tasks':d['splits']['train'],'validation_tasks':d['splits']['validation'],
              'test_tasks':d['splits']['test'],'executor_seeds':SEEDS,'steps':STEPS,'learning_rate':.01,
              'architecture':'Shared AND/OR 2-16-1 MLPs, NOT 1-16-1 MLP; tanh then sigmoid. AND/OR average both child orders. Numerical operator rules are learned, not supplied.',
              'training_labels':'Only original training observation outputs for compatible hypotheses and train-only compatibility labels for the static 33-class inventory',
              'loss':'positive observed-output BCE + .25 task-consistency BCE',
              'posterior':'log_weight = log_q_LLM + 10*neural_observation_log_likelihood',
              'sampling':'Gumbel top-N over classes without replacement; N=K from official adaptive benchmark budget',
              'no_exact_validator_in_proposal':True,'fixed_executor_primary_seed':4306,
              'other_seeds':'Replicate executor training; report all rather than test-select',
              'limits':'Finite training-derived inventory; one frozen LLM checkpoint; existing v1 test observations, exploratory follow-up; separate from fixed16 results',
              'source_sha256':{n:hashlib.sha256((ROOT/'explorer'/n).read_bytes()).hexdigest() for n in ['learned_executor.py','posterior_bank.py']}}
    write_json(OUT/'protocol.json',protocol)
    write_json(OUT/'bank.json',items)
    pidx=mx.array(positive[:,:2]);plabel=mx.array(positive[:,2],dtype=mx.float32)
    y=mx.array(validity);bits=mx.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
    summaries=[]
    for seed in SEEDS:
        mx.random.seed(seed);model=make_model();opt=optim.Adam(.01)
        def loss_fn(net):
            p=net(trees,bits);eps=1e-6
            chosen=mx.clip(p[pidx[:,0],pidx[:,1]],eps,1-eps)
            truth=-(plabel*mx.log(chosen)+(1-plabel)*mx.log1p(-chosen)).mean()
            consistency=[]
            for obs in observations:
                factors=[]
                for o in obs:
                    q=p[:,2*o['inputs']['x']+o['inputs']['y']]
                    factors.append(q if o['output'] else 1-q)
                consistency.append(mx.prod(mx.stack(factors),axis=0))
            v=mx.clip(mx.stack(consistency),eps,1-eps)
            valid_loss=-(y*mx.log(v)+(1-y)*mx.log1p(-v)).mean()
            return truth+.25*valid_loss
        fn=nn.value_and_grad(model,loss_fn)
        state=[model.state,opt.state]
        @partial(mx.compile,inputs=state,outputs=state)
        def step():
            value,grads=fn(model);opt.update(model,grads);return value
        start=time.monotonic();log=[]
        for n in range(STEPS):
            value=step();mx.eval(state,value)
            if (n+1)%250==0:
                row={'seed':seed,'step':n+1,'loss':float(value.item()),'seconds':time.monotonic()-start}
                log.append(row);print(json.dumps(row),flush=True)
        weights={}
        for name,net in [('AND',model.op_and),('OR',model.op_or),('NOT',model.op_not)]:
            weights[name]={'W1':net.layers[0].weight.tolist(),'b1':net.layers[0].bias.tolist(),
                           'W2':net.layers[2].weight.tolist(),'b2':net.layers[2].bias.tolist()}
        native=np.array(model(trees,bits));cpu=numpy_predictions(weights,items)
        error=float(np.max(np.abs(native-cpu)));assert error<1e-5
        dest=OUT/'executors'/str(seed);write_json(dest/'weights.json',weights)
        write_json(dest/'training_log.json',log)
        positive_accuracy=float(np.mean((cpu[positive[:,0],positive[:,1]]>=.5)==positive[:,2]))
        summary={'seed':seed,'steps':STEPS,'parameters':sum(v.size for _,v in tree_flatten(model.trainable_parameters())),
                 'seconds':time.monotonic()-start,'final_loss':float(value.item()),'positive_observation_accuracy':positive_accuracy,
                 'numpy_mlx_max_difference':error,'training_observation_labels':len(positive),'training_class_task_pairs':int(validity.size)}
        summaries.append(summary);print(json.dumps(summary),flush=True)
    write_json(OUT/'executor_training_summary.json',{'runs':summaries,'no_validation_or_test_gradients':True})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['train']);parser.parse_args();train()
