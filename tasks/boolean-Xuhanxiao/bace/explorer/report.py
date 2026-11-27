"""Export truthful, method-neutral metrics and paired task-level uncertainty."""
from __future__ import annotations

import csv
import html
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .experiment import read_rows
from .tasks import write_json


def summarize(dataset, directory, output, split="test"):
    reference = {t["id"]: set(t["reference_keys"]) for t in dataset["tasks"]}
    records = []
    for path in sorted(Path(directory).glob(f"{split}_*.jsonl")):
        rows = read_rows(path)
        if not rows:
            continue
        c = rows[0]["contract"]
        valid_keys, all_keys, valid = set(), set(), 0
        seconds, tokens, usage_complete = 0.0, 0, True
        for n, r in enumerate(rows, 1):
            h = r["outcome"]
            seconds += r.get("seconds", 0.0)
            total = r.get("usage", {}).get("total_tokens")
            usage_complete = usage_complete and total is not None
            tokens += int(total or 0)
            valid += int(h["valid"])
            if h["in_space"]:
                all_keys.add(h["key"])
            if h["valid"]:
                valid_keys.add(h["key"])
            gt = reference[c["task_id"]]
            records.append({"method": c["method"], "task": c["task_id"], "seed": c["seed"], "queries": n,
                            "VR": valid / n, "NR": len(all_keys) / n, "RR": len(valid_keys & gt) / len(gt),
                            "reference_size": len(gt), "valid_outside_reference": len(valid_keys - gt),
                            "new_valid_count": len(valid_keys), "recovered_count": len(valid_keys & gt),
                            "inference_seconds": seconds, "total_tokens": tokens if usage_complete else None,
                            "backend": c["backend"], "model": c["model"], "fingerprint": c["fingerprint"], "complete": len(rows) == c["budget"]})
    if not records:
        raise ValueError("No evaluation records")
    provenance = {(r["backend"], r["model"], r["fingerprint"]) for r in records}
    if len(provenance) != 1:
        raise ValueError("Do not mix backends or models in one report")
    backend, model, experiment_fingerprint = next(iter(provenance))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "curves.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    latest = {}
    for r in records:
        latest[(r["method"], r["task"], r["seed"])] = r
    final = [r for r in latest.values() if r["complete"]]
    metrics = {}
    for method in sorted({r["method"] for r in final}):
        subset = [r for r in final if r["method"] == method]
        metrics[method] = {k: float(np.mean([r[k] for r in subset])) for k in ["VR", "NR", "RR"]}
        metrics[method]["episodes"] = len(subset)
        metrics[method]["mean_new_valid_count"] = float(np.mean([r["new_valid_count"] for r in subset]))
        metrics[method]["mean_inference_seconds"] = float(np.mean([r["inference_seconds"] for r in subset]))
        metrics[method]["mean_total_tokens"] = (float(np.mean([r["total_tokens"] for r in subset]))
                                               if all(r["total_tokens"] is not None for r in subset) else None)
        metrics[method]["valid_outside_reference_max"] = max(r["valid_outside_reference"] for r in subset)
    task_mean = defaultdict(dict)
    for method in metrics:
        for task in {r["task"] for r in final}:
            values = [r["RR"] for r in final if r["method"] == method and r["task"] == task]
            if values:
                task_mean[method][task] = float(np.mean(values))
    comparisons = {}
    if "learned" in task_mean:
        for baseline in ["memory", "fixed", "random", "no_history"]:
            ids = sorted(set(task_mean["learned"]) & set(task_mean[baseline]))
            if not ids:
                continue
            delta = np.asarray([task_mean["learned"][i] - task_mean[baseline][i] for i in ids])
            rng = np.random.default_rng(4300)
            boot = rng.choice(delta, size=(2000, len(delta)), replace=True).mean(axis=1)
            comparisons[baseline] = {"RR_difference": float(delta.mean()), "task_bootstrap_95_interval": np.quantile(boot, [0.025, 0.975]).tolist(), "paired_tasks": len(ids)}
    note = ("SYNTHETIC PIPELINE TEST ONLY. Reference answers are used by the fixture. No LLM performance claim is supported."
            if backend != "real" else "Small pilot on held-out observation sets. Intervals resample tasks, not individual hypotheses. Not a full benchmark claim.")
    if backend == "real" and split == "validation":
        note = "Validation-set diagnostic only. Not a final held-out test score or evidence of method improvement."
    budget_metrics = {}
    for q in sorted({8, 10, max(r["queries"] for r in records)}):
        budget_metrics[str(q)] = {m: {k: float(np.mean([r[k] for r in records if r["method"] == m and r["queries"] == q and r["complete"]]))
                                     for k in ["VR", "NR", "RR"]}
                                 for m in metrics if any(r["method"] == m and r["queries"] == q and r["complete"] for r in records)}
    training_rows = [r for p in Path(directory).glob("train_*.jsonl") for r in read_rows(p)]
    training_resources = {"calls": len(training_rows), "inference_seconds": sum(r.get("seconds", 0.0) for r in training_rows),
                          "total_tokens": sum(r.get("usage", {}).get("total_tokens", 0) for r in training_rows)}
    report = {"status": backend, "split": split, "model": model, "fingerprint": experiment_fingerprint, "notice": note, "metrics": metrics,
              "metrics_by_budget": budget_metrics, "training_resources": training_resources,
              "paired_comparisons": comparisons, "complete_episodes": len(final), "all_episodes": len(latest)}
    write_json(output / "summary.json", report)
    # A standalone SVG is sufficient for the small scientific coverage plot.
    colors = ["#2563eb", "#d97706", "#16a34a", "#9333ea", "#dc2626"]
    pieces = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="530" viewBox="0 0 900 530">',
              '<rect width="900" height="530" fill="white"/>',
              '<g font-family="sans-serif" fill="#111827">',
              f'<text x="65" y="28" font-size="17">{html.escape("SYNTHETIC TEST ONLY" if backend != "real" else "HypoSpace Boolean " + split + " pilot")}</text>',
              '<text x="65" y="52" font-size="12">Mean Recovery versus counted generation calls</text>',
              '<path d="M65 80 V435 H690" stroke="#475569" fill="none"/>']
    max_q = max(r["queries"] for r in records)
    for tick in [0, .25, .5, .75, 1]:
        y = 435 - tick * 335
        pieces.append(f'<text x="20" y="{y+4}" font-size="12">{tick:.2f}</text><path d="M65 {y} H690" stroke="#e5e7eb"/>')
    for j, method in enumerate(sorted(metrics)):
        points = []
        for q in range(1, max_q + 1):
            ys = [r["RR"] for r in records if r["method"] == method and r["queries"] == q]
            if ys:
                points.append(f"{65+q/max_q*625:.2f},{435-np.mean(ys)*335:.2f}")
        color = colors[j % len(colors)]
        pieces.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        pieces.append(f'<text x="715" y="{110+j*30}" fill="{color}" font-size="13">{html.escape(method)}</text>')
    pieces += [f'<text x="65" y="460" font-size="12">0</text><text x="675" y="460" font-size="12">{max_q}</text>',
               '<text x="240" y="490" font-size="13">Counted calls per observation set</text>', '</g></svg>']
    (output / "coverage.svg").write_text("\n".join(pieces))
    return report
