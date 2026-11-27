"""Training-only canonical inventory and syntax trees; no test solution sets."""
from __future__ import annotations

import ast
import gzip
import json
from functools import lru_cache

import numpy as np

from .canonical_decode import grammar_support
from .coverage_lora import dataset
from .upstream import ROOT


def parse_tree(expression):
    node=ast.parse(expression.replace('AND','and').replace('OR','or').replace('NOT','not'),mode='eval').body
    def convert(n):
        if isinstance(n,ast.Name):return ('VAR',n.id)
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.Not):return ('NOT',convert(n.operand))
        if isinstance(n,ast.BoolOp):
            op='AND' if isinstance(n.op,ast.And) else 'OR'
            result=convert(n.values[0])
            for child in n.values[1:]:result=(op,result,convert(child))
            return result
        raise ValueError('Unsupported grammar node')
    return convert(node)


def node_count(tree):
    return 1 if tree[0]=='VAR' else 1+sum(node_count(x) for x in tree[1:])


@lru_cache(maxsize=1)
def bank():
    d=dataset()
    with gzip.open(ROOT/'results/final/training_targets.json.gz','rt') as f:training=json.load(f)
    assert set(training['training_ids'])==set(d['splits']['train'])
    aliases={}
    for group in training['groups']:
        assert group['task_id'] in d['splits']['train']
        answers=[a['answer'] for a in group['aliases']]
        assert len(answers)==2
        if group['key'] in aliases:assert aliases[group['key']]==answers
        aliases[group['key']]=answers
    candidates={}
    for item in grammar_support(d['metadata']):
        expression=item['answer'].removeprefix('Expression: ')
        tree=parse_tree(expression)
        rank=(node_count(tree),len(expression),expression)
        if item['key'] not in candidates or rank<candidates[item['key']][0]:
            candidates[item['key']]=(rank,tree,expression)
    assert set(aliases)==set(candidates)
    return [{'key':key,'aliases':aliases[key],'tree':candidates[key][1],
             'tree_expression':candidates[key][2]} for key in sorted(aliases)]


def numpy_predictions(weights,items=None,inputs=None,variables=('x','y')):
    """Execute learned neural modules. Contains no exact Boolean operator rules."""
    if items is None:items=bank()
    if inputs is None:inputs=np.array([[0,0],[0,1],[1,0],[1,1]],dtype=np.float32)
    inputs=np.asarray(inputs,dtype=np.float32)
    cache={}
    def network(name,x):
        w=weights[name]
        return (np.tanh(x@np.asarray(w['W1'],dtype=np.float32).T+np.asarray(w['b1'],dtype=np.float32))
                @np.asarray(w['W2'],dtype=np.float32).T+np.asarray(w['b2'],dtype=np.float32))[...,0]
    def walk(tree):
        tree=tuple_tree(tree)
        if tree in cache:return cache[tree]
        if tree[0]=='VAR':result=inputs[:,variables.index(tree[1])]
        elif tree[0]=='NOT':
            logits=network('NOT',walk(tree[1])[:,None])
            result=1/(1+np.exp(-np.clip(logits,-60,60)))
        else:
            a,b=walk(tree[1]),walk(tree[2]);x=np.stack([a,b],axis=-1)
            logits=(network(tree[0],x)+network(tree[0],x[:,::-1]))/2
            result=1/(1+np.exp(-np.clip(logits,-60,60)))
        cache[tree]=result
        return result
    return np.stack([walk(item['tree']) for item in items])


def tuple_tree(tree):
    return tuple(tuple_tree(x) if isinstance(x,(list,tuple)) else x for x in tree)


def observation_log_likelihood(predictions,observations):
    score=np.zeros(predictions.shape[0],dtype=np.float64)
    for o in observations:
        j=2*o['inputs']['x']+o['inputs']['y']
        p=predictions[:,j] if o['output'] else 1-predictions[:,j]
        score+=np.log(np.clip(p,1e-7,1))
    return score


def exact_tree_values(tree,inputs,variables=('x','y')):
    """Exact semantics for evaluation/solver control only; never used by COVE."""
    tree=tuple_tree(tree);op=tree[0]
    if op=='VAR':return np.asarray(inputs)[:,variables.index(tree[1])].astype(bool)
    a=exact_tree_values(tree[1],inputs,variables)
    if op=='NOT':return ~a
    b=exact_tree_values(tree[2],inputs,variables)
    return a & b if op=='AND' else a | b
