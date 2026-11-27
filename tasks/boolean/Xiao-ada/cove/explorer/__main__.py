from __future__ import annotations

import argparse
import json
from pathlib import Path

from .backend import load_env
from .experiment import backend_for, collect, evaluate, fingerprint, train
from .report import summarize
from .tasks import prepare, write_json
from .upstream import ROOT, fetch


def main():
    parser = argparse.ArgumentParser(description="HypoSpace learned exploration pilot")
    parser.add_argument("command", choices=["fetch", "prepare", "preflight", "collect", "train", "evaluate", "summarize", "smoke"])
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--backend", choices=["real", "smoke"], default="real")
    parser.add_argument("--run-dir", default=str(ROOT / "runs/pilot"))
    parser.add_argument("--split", choices=["train", "validation"], default="train")
    parser.add_argument("--limit-tasks", type=int, default=None)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.command == "fetch":
        print(json.dumps(fetch(), indent=2))
        return
    if args.command == "prepare":
        dataset = prepare(config)
        print(json.dumps({"audit": dataset["audit"], "eligible_tasks": len(dataset["tasks"]), "splits": {k: len(v) for k, v in dataset["splits"].items()}}, indent=2))
        return
    path = ROOT / ".cache/tasks.json"
    if not path.exists():
        raise SystemExit("Run python -m explorer prepare first")
    dataset = json.loads(path.read_text())
    # Guard stale data when a user edits config after preparing a dataset.
    meta = dataset["metadata"]
    from .upstream import load
    expected_ops = sorted(load()[0].BooleanDiscoveryGame.OPERATOR_SETS[config["operator_set"]])
    if meta["operators"] != expected_ops or meta["max_depth"] != config["max_depth"]:
        raise SystemExit("Prepared data does not match config; rerun prepare")
    if dataset.get("preparation", {}).get("split_seed") != config["split_seed"]:
        raise SystemExit("Prepared split seed does not match config; rerun prepare")
    if {k: len(dataset["splits"][k]) for k in ["train", "validation", "test"]} != {
        "train": config["train_tasks"], "validation": config["validation_tasks"], "test": config["test_tasks"]}:
        raise SystemExit("Prepared split sizes do not match config; rerun prepare")
    if args.command == "smoke":
        # Keep all synthetic outputs in a separate ignored directory. No real
        # backend, credentials, or performance tables are used by this command.
        tiny = dict(config, train_queries=6, eval_queries=6, eval_seeds=[7])
        subset = dict(dataset, splits={"train": dataset["splits"]["train"][:3],
                                      "validation": dataset["splits"]["validation"][:1],
                                      "test": dataset["splits"]["test"][:2]})
        directory = ROOT / ".cache/synthetic_smoke"
        backend = backend_for("smoke", subset, tiny, directory)
        collect(subset, tiny, backend, directory)
        train(subset, tiny, backend, directory)
        evaluate(subset, tiny, backend, directory)
        summary = summarize(subset, directory, directory / "report")
        print(summary["notice"])
        return
    directory = Path(args.run_dir)
    if args.command == "summarize":
        result = summarize(dataset, directory, ROOT / "results/pilot")
        print(json.dumps(result, indent=2))
        return
    backend = backend_for(args.backend, dataset, config, directory)
    if args.command == "preflight":
        calls = config["train_tasks"] * config["train_queries"] + config["test_tasks"] * len(config["eval_seeds"]) * len(config["methods"]) * config["eval_queries"]
        print(json.dumps({"backend": backend.kind, "model": backend.model, "planned_main_calls": calls,
                          "max_calls": config["max_api_calls"], "fingerprint": fingerprint(config, dataset)}, indent=2))
        return
    if args.command == "collect":
        collect(dataset, config, backend, directory, split=args.split, limit_tasks=args.limit_tasks)
    elif args.command == "train":
        train(dataset, config, backend, directory)
    elif args.command == "evaluate":
        evaluate(dataset, config, backend, directory)


if __name__ == "__main__":
    main()
