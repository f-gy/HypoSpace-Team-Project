"""Run an unadapted model against the new plan, independently of LoRA training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .backend import RealBackend
from .experiment import episode
from .report import summarize
from .tasks import write_json
from .upstream import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--method", choices=["base_iid", "base_memory"], default="base_iid")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/coverage_lora_v1")
    args = parser.parse_args()
    plan_path = ROOT / "results/coverage_plan/plan.json"
    split_path = ROOT / "results/coverage_plan/split_manifest.json"
    plan = json.loads(plan_path.read_text())
    manifest = json.loads(split_path.read_text())
    dataset = json.loads((ROOT / ".cache/tasks.json").read_text())
    scope, meta = plan["scope"], dataset["metadata"]
    if (meta["variables"] != scope["variables"] or sorted(meta["operators"]) != sorted(scope["operators"])
            or meta["max_depth"] != scope["max_depth"]):
        raise ValueError("Prepared data does not match the selected plan")
    dataset["splits"] = manifest["splits"]
    cfg = plan["evaluation"]
    model_lock = json.loads((ROOT / plan["model_lock"]).read_text())
    config = {k: cfg[k] for k in ["temperature", "top_p", "top_k", "min_p", "repetition_penalty", "max_output_tokens"]}
    config.update(model_identity=model_lock["repo_id"], model_provenance=model_lock,
                  max_api_calls=cfg["generation_attempt_cap"], max_total_tokens=12_000_000,
                  request_timeout_seconds=90, send_seed=True,
                  plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                  split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
                  dataset_sha256=hashlib.sha256((ROOT / ".cache/tasks.json").read_bytes()).hexdigest(),
                  source_sha256={name: hashlib.sha256((ROOT / "explorer" / name).read_bytes()).hexdigest()
                                 for name in ["baseline.py", "backend.py", "experiment.py", "report.py", "upstream.py"]})
    directory = args.run_dir
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = directory / "baseline_config.json"
    if snapshot.exists() and json.loads(snapshot.read_text()) != config:
        raise ValueError("Baseline configuration changed; retain the old run and choose a new directory")
    write_json(snapshot, config)
    backend = RealBackend(config, directory / "api_ledger.json")
    by_id = {t["id"]: t for t in dataset["tasks"]}
    seeds = cfg["validation_seeds"] if args.split == "validation" else cfg["eval_seeds"]
    for task_id in dataset["splits"][args.split]:
        for seed in seeds:
            episode(by_id[task_id], dataset, config, backend, args.method, seed,
                    cfg["calls_per_episode"], directory, args.split)
            print(f"Completed {args.split} {args.method} task {task_id} seed {seed}", flush=True)
    output = ROOT / "results/coverage_baseline" / args.split
    result = summarize(dataset, directory, output, split=args.split)
    write_json(output / "config.json", config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
