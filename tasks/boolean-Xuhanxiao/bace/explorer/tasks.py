from __future__ import annotations

import hashlib
import itertools
import json
import random
from pathlib import Path

from .upstream import ROOT, evaluator, load


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def prepare(config: dict) -> dict:
    module, _ = load()
    opts = dict(apply_commutativity=True, apply_idempotence_and_or=True, flatten_associativity=True)
    operators = module.BooleanDiscoveryGame.OPERATOR_SETS[config["operator_set"]]
    metadata = dict(variables=["x", "y"], operators=sorted(operators), max_depth=config["max_depth"], mechanistic_opts=opts)
    # Exact upstream enumeration; no new hypothesis language or equivalence rules.
    expressions = module.BooleanDiscoveryGame.generate_all_expressions(["x", "y"], operators, config["max_depth"], opts)
    ev = evaluator(metadata)
    universe = [{"formula": e.formula, "key": str(e.mechanistic_key(**opts)),
                 "truth": [int(e.truth_table[p]) for p in itertools.product([0, 1], repeat=2)]}
                for e in expressions]
    universe.sort(key=lambda e: e["key"])
    tasks = []
    for n in [1, 2, 3]:
        for positions in itertools.combinations(range(4), n):
            for outputs in itertools.product([0, 1], repeat=n):
                obs = [{"inputs": {"x": p // 2, "y": p % 2}, "output": y} for p, y in zip(positions, outputs)]
                keys = [e["key"] for e in universe if all(e["truth"][p] == y for p, y in zip(positions, outputs))]
                if len(keys) >= 4:
                    signature = json.dumps(obs, sort_keys=True)
                    tasks.append({"id": hashlib.sha256(signature.encode()).hexdigest()[:16], "observations": obs, "reference_keys": keys})
    tasks.sort(key=lambda t: t["id"])
    random.Random(config["split_seed"]).shuffle(tasks)
    counts = [config["train_tasks"], config["validation_tasks"], config["test_tasks"]]
    if len(tasks) < sum(counts):
        raise ValueError(f"Only {len(tasks)} eligible unique observation sets, requested {sum(counts)}")
    splits, start = {}, 0
    for label, count in zip(["train", "validation", "test"], counts):
        splits[label] = [t["id"] for t in tasks[start:start + count]]
        start += count
    # Audit generator/validator disagreement; preserve the upstream reference set.
    unaccepted = [e["formula"] for e in universe if not ev._in_space(e["formula"])]
    result = {"metadata": metadata, "preparation": {"split_seed": config["split_seed"]}, "universe": universe, "tasks": tasks, "splits": splits,
              "audit": {"reference_size": len(universe), "reference_formulas_rejected_by_validator": len(unaccepted),
                        "rejected_examples": unaccepted[:5],
                        "note": "Reference set is unchanged. Inspect mismatches before interpreting absolute RR."}}
    write_json(ROOT / ".cache/tasks.json", result)
    write_json(ROOT / "results/data_audit.json", {"metadata": metadata, "splits": splits, "eligible_tasks": len(tasks), "audit": result["audit"]})
    return result


def public_task(task: dict) -> dict:
    """Only this projection is visible to prompts and the learned policy."""
    return {"id": task["id"], "observations": task["observations"]}


def select_tasks(dataset: dict, split: str):
    by_id = {t["id"]: t for t in dataset["tasks"]}
    return [by_id[i] for i in dataset["splits"][split]]
