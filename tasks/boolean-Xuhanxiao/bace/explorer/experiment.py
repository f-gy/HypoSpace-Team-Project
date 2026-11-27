from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

from .backend import RealBackend, SyntheticBackend
from .policy import ACTIONS, Predictor, available, guide
from .tasks import public_task, select_tasks, write_json
from .upstream import evaluator, load

IID_METHODS = {"base_iid", "balanced_sft_iid", "canonical_coverage_iid", "string_coverage_iid"}


def fingerprint(config, dataset):
    # Call/token safety ceilings may change without changing the experiment.
    stable = {k: v for k, v in config.items() if k not in {"max_api_calls", "max_total_tokens"}}
    blob = json.dumps({"config": stable, "metadata": dataset["metadata"], "splits": dataset["splits"]}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def read_rows(path):
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def assess(ev, metadata, task, text, history):
    module, _ = load()
    expression = ev.parse_llm_response(text)
    # Restrict model text to the task grammar before using the upstream SymPy
    # parser. The same lexical/space gate is applied to every method.
    tokens = re.findall(r"[A-Za-z]+", expression or "")
    allowed_tokens = set(metadata["variables"]) | set(metadata["operators"])
    safe = bool(expression and re.fullmatch(r"[A-Za-z\s(),&|~^]+", expression)
                and tokens and set(tokens).issubset(allowed_tokens))
    in_space = bool(safe and ev._in_space(expression))
    key = str(ev.get_expression_mechanistic_key(expression)) if in_space else None
    observations = [module.BooleanObservation(o["inputs"], o["output"]) for o in task["observations"]]
    # Explicit fail-closed gate: reject parse/space failures before validity.
    # This is recorded as a protocol difference from the upstream runner.
    valid = bool(in_space and ev.validate_expression(expression, observations)[0])
    seen_valid = {r["key"] for r in history if r["valid"]}
    feedback = ""
    if not in_space:
        feedback = "Candidate failed parsing or the allowed variable/operator/depth/constant constraints."
    elif not valid:
        parsed = module.BooleanExpression(expression, metadata["variables"], set(metadata["operators"]))
        failed = [o for o in observations if parsed.evaluate(o.inputs) != o.output]
        feedback = "Mismatched observations: " + "; ".join(o.to_string() for o in failed)
    return {"expression": expression, "key": key, "in_space": in_space,
            "valid": valid, "new": bool(valid and key not in seen_valid), "feedback": feedback}


def episode(task, dataset, config, backend, method, seed, budget, directory, split, predictors=None):
    directory = Path(directory)
    path = directory / f"{split}_{method}_{task['id']}_{seed}.jsonl"
    contract = {"fingerprint": fingerprint(config, dataset), "backend": backend.kind, "model": backend.model,
                "split": split, "task_id": task["id"], "method": method, "seed": seed, "budget": budget}
    previous = read_rows(path)
    for i, r in enumerate(previous):
        if r["contract"] != contract or r["step"] != i:
            raise ValueError("Existing trajectory does not match experiment contract; use a new run directory")
    history = [r["outcome"] for r in previous]
    visible = public_task(task)
    ev = evaluator(dataset["metadata"])
    for t in range(len(previous), budget):
        rng = random.Random(seed * 100003 + t)
        if method in IID_METHODS | {"memory", "base_memory"}:
            action = "free"
        elif method == "fixed":
            action = ACTIONS[t % 5]
        elif method == "random":
            action = rng.choice(available(history))
        elif method in {"learned", "no_history"}:
            action = predictors[method].choose(visible, history, budget)
        else:
            raise ValueError(method)
        prior = [] if method in IID_METHODS else [h["expression"] for h in history if h["expression"]]
        prompt = ev.create_prompt([load()[0].BooleanObservation(o["inputs"], o["output"]) for o in visible["observations"]], prior)
        extra = guide(action, history)
        if extra:
            prompt += "\n" + extra
        call_seed = seed * 100003 + t
        if method in IID_METHODS | {"base_memory"}:
            call_seed = int.from_bytes(hashlib.sha256(f"{seed}:{t}:{task['id']}".encode()).digest()[:8], "big")
        response = backend.query(prompt, action, visible, random.Random(call_seed))
        outcome = {**assess(ev, dataset["metadata"], visible, response["text"], history), "action": action}
        row = {"contract": contract, "step": t, "task": visible, "history": history.copy(), "action": action,
               "budget": budget, "valid": outcome["valid"], "new": outcome["new"], "outcome": outcome,
               "raw_response": response["text"], "usage": response["usage"], "error": response["error"],
               "seconds": response["seconds"], "prompt": prompt,
               "request_seed": response.get("request_seed")}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
        history.append(outcome)
        previous.append(row)
        # Fail fast on provider/config failures; the attempted call is retained.
        if response["error"]:
            raise RuntimeError(f"Provider request failed ({response['error']}); saved the attempt. Inspect endpoint before resuming.")
    return previous


def backend_for(kind, dataset, config, directory):
    if kind == "real":
        return RealBackend(config, Path(directory) / "api_ledger.json")
    if kind == "smoke":
        return SyntheticBackend(dataset["universe"], dataset["tasks"])
    raise ValueError(kind)


def collect(dataset, config, backend, directory, split="train", limit_tasks=None):
    if split not in {"train", "validation"}:
        raise ValueError("Collection is restricted to train/validation")
    count = 0
    tasks = select_tasks(dataset, split)
    if limit_tasks is not None:
        if limit_tasks < 1:
            raise ValueError("limit_tasks must be positive")
        tasks = tasks[:limit_tasks]
    for task in tasks:
        rows = episode(task, dataset, config, backend, "random", 101, config["train_queries"], directory, split)
        count += len(rows)
    print(f"Collected {count} {split} rows. Backend: {backend.kind}")


def train(dataset, config, backend, directory):
    rows = []
    for path in sorted(Path(directory).glob("train_random_*.jsonl")):
        rows.extend(read_rows(path))
    if not rows:
        raise ValueError("Collect training trajectories first")
    expected = fingerprint(config, dataset)
    for r in rows:
        c = r["contract"]
        if c["backend"] != backend.kind or c["model"] != backend.model or c["fingerprint"] != expected:
            raise ValueError("Training provenance mismatch")
        if c["task_id"] not in dataset["splits"]["train"] or c["split"] != "train":
            raise ValueError("Non-training task in training data")
    provenance = {"backend": backend.kind, "model": backend.model, "fingerprint": expected,
                  "training_task_ids": sorted({r["task"]["id"] for r in rows})}
    for name in ["learned", "no_history"]:
        model = Predictor.fit(rows, provenance, no_history=(name == "no_history"))
        model.save(Path(directory) / f"{name}.json")
    print(f"Trained two predictors from {len(rows)} rows, {len(provenance['training_task_ids'])} training tasks")


def evaluate(dataset, config, backend, directory):
    predictors = {m: Predictor.read(Path(directory) / f"{m}.json") for m in ["learned", "no_history"]}
    for model in predictors.values():
        if model.provenance["backend"] != backend.kind or model.provenance["model"] != backend.model:
            raise ValueError("Checkpoint/backend mismatch: synthetic checkpoints cannot evaluate real LLMs")
        if model.provenance["fingerprint"] != fingerprint(config, dataset):
            raise ValueError("Checkpoint experiment fingerprint mismatch")
    for task in select_tasks(dataset, "test"):
        for seed in config["eval_seeds"]:
            methods = list(config["methods"])
            random.Random(seed + int(task["id"], 16)).shuffle(methods)
            for method in methods:
                episode(task, dataset, config, backend, method, seed, config["eval_queries"], directory, "test", predictors)
            print(f"Completed test task {task['id']} repetition {seed}", flush=True)
