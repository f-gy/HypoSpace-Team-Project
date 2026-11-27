"""Isolated, resumable v2 decoding study; v1 sources and ledgers stay unchanged."""
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

RUN = ROOT / "runs/canonical_decode_v2"
OUT = ROOT / "results/canonical_decode_v2"
METHODS = ["cc_grammar_iid", "cc_string_wor", "cc_class_wor", "sft_class_wor", "base_class_wor", "uniform_class_wor"]
SEEDS = {"validation": [101], "test": [17, 29, 43]}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initialize():
    d = dataset()
    support = grammar_support(d["metadata"])
    config = {"version": "canonical_decode_v2", "methods": METHODS, "seeds": SEEDS,
              "budget": 16, "temperature": 1.0, "top_p": 1.0, "max_output_tokens": 128,
              "split_sha256": sha(ROOT / "results/coverage_plan/split_manifest.json"),
              "model_lock_sha256": sha(ROOT / "model.local_mlx.lock.json"),
              "source_sha256": {n: sha(ROOT / "explorer" / n) for n in ["canonical_decode.py", "decode_experiment.py"]},
              "adapters": {n: sha(V1 / "adapters" / n / "adapters.safetensors")
                           for n in ["canonical_coverage", "balanced_sft"]},
              "support": support_summary(support),
              "scoring": "Unchanged v1 evaluator including common fail-closed gate and full reference denominator",
              "selection_rule": "Run every listed arm on validation, then freeze all arms for follow-up test. Prioritize NR; report VR/RR and uniform/class/string controls without selective omission.",
              "test_status": "Follow-up on previously reported v1 test observations, not a new untouched test set",
              "proposal_boundary": "Grammar and canonicalizer only; no test reference keys, observation filtering, repair, or rejected hidden generation attempts",
              "uniform_cost": "Uniform samples use no LLM; zero model tokens. Shared grammar construction cost is reported separately.",
              "theory": "Class exclusion guarantees distinct canonical keys until support exhaustion; this guarantee alone does not establish learned model quality or global algorithm novelty."}
    p = OUT / "protocol.json"
    if p.exists() and json.loads(p.read_text()) != config:
        raise RuntimeError("Protocol mismatch; preserve the previous run")
    if not p.exists():
        write_json(p, config)
    return d, support, config


def evaluate(split):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.tuner.utils import linear_to_lora_layers

    tic = time.monotonic()
    d, support, config = initialize()
    grammar_seconds = time.monotonic() - tic
    model, tok = model_load()
    model.freeze()
    linear_to_lora_layers(model, 8, LORA)
    model.eval()
    support = completion_tokens(tok, support)
    assert len({h['key'] for h in support}) >= config['budget']
    assert max(len(h['tokens']) for h in support) < config['max_output_tokens']
    by_key = {}
    for h in support:
        by_key.setdefault(h["key"], []).append(h)
    weights = {"base": {n: mx.zeros_like(v) for n, v in tree_flatten(model.trainable_parameters())}}
    for name, path in [("cc", "canonical_coverage"), ("sft", "balanced_sft")]:
        weights[name] = mx.load(str(V1 / "adapters" / path / "adapters.safetensors"))
    by_id = {t["id"]: t for t in d["tasks"]}
    ev = evaluator(d["metadata"])
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    RUN.mkdir(parents=True, exist_ok=True)
    ledger_path = RUN / "generation_ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {
        "model_calls": 0, "uniform_draws": 0, "prompt_tokens": 0, "completion_tokens": 0,
        "errors": 0, "grammar_build_seconds": grammar_seconds}
    for tid in d["splits"][split]:
        task = public_task(by_id[tid])
        prompt = prompt_for(task, d)
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True)
        # Verify the tokenization boundary on every task, not only the probe.
        check = tok.apply_chat_template([{"role": "user", "content": prompt}, {"role": "assistant", "content": support[0]["answer"]}], add_generation_prompt=False, tokenize=True)
        assert check[:len(ids)] == ids
        assert check[len(ids):len(ids) + len(support[0]["tokens"])] == support[0]["tokens"]
        for seed in SEEDS[split]:
            order = METHODS.copy()
            random.Random(seed + int(tid, 16)).shuffle(order)
            for method in order:
                path = RUN / f"{split}_{method}_{tid}_{seed}.jsonl"
                rows = read_rows(path)
                contract = {"fingerprint": signature, "split": split, "task_id": tid, "method": method, "seed": seed, "budget": 16}
                assert all(r["contract"] == contract and r["step"] == i for i, r in enumerate(rows))
                history = [r["outcome"] for r in rows]
                if not method.startswith("uniform") and len(rows) < 16:
                    model.load_weights(list(weights[method.split('_')[0]].items()), strict=False)
                    mx.eval(model.parameters())
                for step in range(len(rows), 16):
                    blocked_keys = {h["key"] for h in history if h["in_space"]} if method.endswith("class_wor") else set()
                    blocked_strings = {r["raw_response"] for r in rows} if method.endswith("string_wor") else set()
                    root_seed = int.from_bytes(hashlib.sha256(f"{seed}:{step}:{tid}".encode()).digest()[:8], "big")
                    request_seed = random.Random(root_seed).getrandbits(31)
                    tic = time.monotonic()
                    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    error = None
                    processor_steps = 0
                    if method.startswith("uniform"):
                        rng = random.Random(request_seed)
                        keys = sorted(set(by_key) - blocked_keys)
                        key = rng.choice(keys)
                        answer = rng.choice(by_key[key])["answer"]
                        ledger["uniform_draws"] += 1
                    else:
                        if ledger["model_calls"] >= 7000:
                            raise RuntimeError("v2 model call cap reached")
                        ledger["model_calls"] += 1
                        write_json(ledger_path, ledger)
                        mx.random.seed(request_seed)
                        trie = SupportTrie(support, blocked_keys, blocked_strings)
                        processor = TrieLogitsProcessor(trie, len(ids))
                        parts, last = [], None
                        try:
                            for last in stream_generate(model, tok, ids, max_tokens=128,
                                                        sampler=make_sampler(temp=1.0, top_p=1.0, top_k=0, min_p=0.0),
                                                        logits_processors=[processor]):
                                parts.append(last.text)
                            answer = "".join(parts)
                            usage = {"prompt_tokens": len(ids), "completion_tokens": last.generation_tokens,
                                     "total_tokens": len(ids) + last.generation_tokens}
                            ledger["prompt_tokens"] += usage["prompt_tokens"]
                            ledger["completion_tokens"] += usage["completion_tokens"]
                            processor_steps = processor.steps
                        except Exception as exc:
                            answer, error = "".join(parts), type(exc).__name__
                            ledger["errors"] += 1
                    elapsed = time.monotonic() - tic
                    h = assess(ev, d["metadata"], task, answer, history)
                    if not error and not h["in_space"]:
                        raise RuntimeError("Trie output did not pass the shared grammar gate")
                    if blocked_keys and h["key"] in blocked_keys:
                        raise RuntimeError("Canonical exclusion invariant failed")
                    row = {"contract": contract, "step": step, "task": task, "prompt": prompt,
                           "raw_response": answer, "outcome": h, "usage": usage, "error": error,
                           "seconds": elapsed, "request_seed": request_seed,
                           "decoder": {"blocked_classes": len(blocked_keys), "blocked_strings": len(blocked_strings),
                                       "processor_steps": processor_steps}}
                    with path.open("a") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    write_json(ledger_path, ledger)
                    rows.append(row)
                    history.append(h)
                    if error:
                        raise RuntimeError(f"Retained failed model call: {error}")
            print(f"Completed {split} task {tid} seed {seed}; model calls {ledger['model_calls']}", flush=True)
    print(json.dumps(ledger), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "validation", "test"])
    args = parser.parse_args()
    if args.command == "prepare":
        _, _, cfg = initialize()
        print(json.dumps({"methods": cfg['methods'], "support_classes": cfg['support']['classes'], "support_strings": cfg['support']['strings']}))
    else:
        evaluate(args.command)


if __name__ == "__main__":
    main()
