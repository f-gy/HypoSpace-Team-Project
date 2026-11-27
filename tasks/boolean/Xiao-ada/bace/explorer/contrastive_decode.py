"""Observation-contrastive canonical decoding, a validation-selected extension.

The second LM stream has the same task rules but zero observations. It never
receives reference answers. Guidance is combined with the grammar/class mask.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time

from .canonical_decode import SupportTrie, TrieLogitsProcessor, completion_tokens, grammar_support, support_summary
from .coverage_lora import LORA, RUN as V1, dataset, model_load, prompt_for
from .experiment import assess, read_rows
from .tasks import public_task, write_json
from .upstream import ROOT, evaluator

RUN = ROOT / 'runs/contrastive_decode_v3'
OUT = ROOT / 'results/contrastive_decode_v3'


class ObservationGuidance:
    def __init__(self, model, prior_ids, trie, prompt_length, scale):
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache
        self.model, self.scale = model, scale
        self.prior_ids = prior_ids
        self.mask = TrieLogitsProcessor(trie, prompt_length)
        self.prompt_length = prompt_length
        self.cache = make_prompt_cache(model)
        if len(prior_ids) > 1:
            model(mx.array(prior_ids[:-1])[None], cache=self.cache)
            mx.eval([c.state for c in self.cache])

    def __call__(self, tokens, logits):
        import mlx.core as mx
        generated = tokens[self.prompt_length:]
        next_input = self.prior_ids[-1] if len(generated) == 0 else generated[-1].item()
        prior_logits = self.model(mx.array([[next_input]]), cache=self.cache)[:, -1, :]
        guided = self.scale * logits - (self.scale - 1) * prior_logits
        return self.mask(tokens, guided)


def protocol():
    d = dataset()
    support = grammar_support(d['metadata'])
    cfg = {'version': 'observation_contrastive_canonical_v3', 'validation_scales': [2., 4.],
           'validation_seed': 101, 'budget': 16, 'temperature': 1., 'max_output_tokens': 128,
           'proposal_boundary': 'No reference keys or observation validator in proposal; prior prompt deletes observations, all other rules identical',
           'model_forward_streams_per_generated_answer': 2,
           'selection_rule': 'Maximize validation RR among fixed scales with NR > .8 and VR at least v1 canonical validation VR minus .02. If neither passes, retain original method and report failed pilots.',
           'test_status': 'Previously seen v1 observation test sets; follow-up evidence only',
           'support': support_summary(support),
           'sources': {p: hashlib.sha256((ROOT / 'explorer' / p).read_bytes()).hexdigest()
                       for p in ['canonical_decode.py', 'contrastive_decode.py']}}
    path = OUT / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == cfg
    else:
        write_json(path, cfg)
    return d, support, cfg


def validation():
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.tuner.utils import linear_to_lora_layers
    d, support, cfg = protocol()
    model, tok = model_load()
    model.freeze()
    linear_to_lora_layers(model, 8, LORA)
    model.load_weights(list(mx.load(str(V1 / 'adapters/canonical_coverage/adapters.safetensors')).items()), strict=False)
    model.eval()
    support = completion_tokens(tok, support)
    by_id = {t['id']: t for t in d['tasks']}
    prior_prompt = prompt_for({'id': 'unconditional', 'observations': []}, d)
    prior_ids = tok.apply_chat_template([{'role': 'user', 'content': prior_prompt}], add_generation_prompt=True, tokenize=True)
    fingerprint = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    ledger_path = RUN / 'generation_ledger.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {'generated_answers': 0, 'model_forward_streams': 0, 'conditional_prompt_tokens': 0, 'prior_prompt_tokens': 0, 'completion_tokens_per_stream': 0, 'errors': 0}
    ev = evaluator(d['metadata'])
    for tid in d['splits']['validation']:
        task = public_task(by_id[tid])
        prompt = prompt_for(task, d)
        ids = tok.apply_chat_template([{'role': 'user', 'content': prompt}], add_generation_prompt=True, tokenize=True)
        for scale in cfg['validation_scales']:
            method = f'cc_class_cfg{scale:g}'
            path = RUN / f'validation_{method}_{tid}_101.jsonl'
            rows = read_rows(path)
            history = [r['outcome'] for r in rows]
            for step in range(len(rows), 16):
                start = time.monotonic()
                root_seed = int.from_bytes(hashlib.sha256(f'101:{step}:{tid}'.encode()).digest()[:8], 'big')
                seed = random.Random(root_seed).getrandbits(31)
                mx.random.seed(seed)
                trie = SupportTrie(support, blocked_keys={h['key'] for h in history})
                processor = ObservationGuidance(model, prior_ids, trie, len(ids), scale)
                parts, last = [], None
                ledger['generated_answers'] += 1
                ledger['model_forward_streams'] += 2
                write_json(ledger_path, ledger)
                for last in stream_generate(model, tok, ids, max_tokens=128,
                                            sampler=make_sampler(temp=1., top_p=1., top_k=0, min_p=0.),
                                            logits_processors=[processor]):
                    parts.append(last.text)
                answer = ''.join(parts)
                h = assess(ev, d['metadata'], task, answer, history)
                assert h['in_space'] and h['key'] not in {r['key'] for r in history}
                ledger['conditional_prompt_tokens'] += len(ids)
                ledger['prior_prompt_tokens'] += len(prior_ids)
                ledger['completion_tokens_per_stream'] += last.generation_tokens
                write_json(ledger_path, ledger)
                r = {'contract': {'fingerprint': fingerprint, 'method': method, 'task_id': tid, 'split': 'validation', 'seed': 101, 'budget': 16},
                     'step': step, 'task': task, 'prompt': prompt, 'prior_prompt': prior_prompt,
                     'raw_response': answer, 'outcome': h, 'seconds': time.monotonic() - start,
                     'request_seed': seed, 'error': None,
                     'usage': {'prompt_tokens': len(ids) + len(prior_ids), 'completion_tokens': 2 * last.generation_tokens,
                               'total_tokens': len(ids) + len(prior_ids) + 2 * last.generation_tokens,
                               'model_forward_streams': 2}}
                with path.open('a') as f:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
                rows.append(r)
                history.append(h)
        print(f'Completed guided validation {tid}; answers {ledger["generated_answers"]}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'validation'])
    args = parser.parse_args()
    if args.command == 'prepare':
        protocol()
    else:
        validation()
