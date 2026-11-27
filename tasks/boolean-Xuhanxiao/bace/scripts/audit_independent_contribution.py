"""Exploratory inference on existing v1 outputs; no training or generation.

The original primary RR analysis remains immutable. This audit reports the
complete 4-comparator x 3-metric family at the prespecified final budget of 16.
It cannot retroactively preregister secondary inference or training replicates.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/contribution_audit"
COMPARATORS = ["base_iid", "balanced_sft_iid", "string_coverage_iid", "base_memory"]
METRICS = ["VR", "NR", "RR"]


def exact_count_sign_flip(differences):
    """Two-sided sign-flip probability for an integer-lattice paired mean.

    Dynamic programming enumerates multiplicities of all signed sums exactly.
    Zero differences cancel from numerator and denominator and can be omitted.
    """
    target = abs(sum(differences))
    distribution = {0: 1}
    for weight in [abs(int(x)) for x in differences if x]:
        updated = defaultdict(int)
        for total, count in distribution.items():
            updated[total + weight] += count
            updated[total - weight] += count
        distribution = updated
    numerator = sum(n for total, n in distribution.items() if abs(total) >= target)
    denominator = sum(distribution.values())
    return {"p_value": numerator / denominator, "numerator": numerator, "denominator": denominator}


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, (len(p) - rank) * p[index]))
        adjusted[index] = previous
    return adjusted


def main():
    if OUT.exists():
        raise SystemExit("Audit exists; retain it rather than overwrite it")
    source = ROOT / "results/final/test/episodes.csv"
    rows = list(csv.DictReader(source.open()))
    ids = json.loads((ROOT / "results/coverage_plan/split_manifest.json").read_text())["splits"]["test"]
    original = json.loads((ROOT / "results/final/test/summary.json").read_text())
    assert len(ids) == 26 and len(rows) == 390
    by_pair = defaultdict(list)
    for row in rows:
        assert int(row["queries"]) == 16 and row["task"] in ids
        by_pair[(row["method"], row["task"])].append(row)
    for group in by_pair.values():
        assert sorted(int(r["seed"]) for r in group) == [17, 29, 43]
    columns = []
    records = []
    for comparator in COMPARATORS:
        for metric in METRICS:
            differences = []
            count_differences = []
            for tid in ids:
                a = by_pair[("canonical_coverage_iid", tid)]
                b = by_pair[(comparator, tid)]
                differences.append(float(np.mean([float(x[metric]) for x in a]) - np.mean([float(x[metric]) for x in b])))
                if metric in ["VR", "NR"]:
                    for row in a + b:
                        assert abs(float(row[metric]) * 16 - round(float(row[metric]) * 16)) < 1e-10
                    count_differences.append(sum(round(float(x[metric]) * 16) for x in a) - sum(round(float(x[metric]) * 16) for x in b))
            d = np.asarray(differences)
            boot = np.random.default_rng(4300).choice(d, (10000, len(d)), replace=True).mean(axis=1)
            interval = np.quantile(boot, [.025, .975])
            if metric == "RR":
                expected = original["canonical_vs"][comparator]
                assert abs(float(d.mean()) - expected["RR_difference"]) < 1e-12
                assert np.max(np.abs(interval - expected["task_bootstrap_95_interval"])) < 1e-12
            records.append({"comparator": comparator, "metric": metric,
                            "difference_percentage_points": float(100 * d.mean()),
                            "pointwise_95_bootstrap_interval_pp": (100 * interval).tolist(),
                            "task_wins": int((d > 1e-10).sum()), "task_ties": int((abs(d) <= 1e-10).sum()),
                            "task_losses": int((d < -1e-10).sum()),
                            "per_task_differences": dict(zip(ids, differences))})
            if count_differences:
                exact = exact_count_sign_flip(count_differences)
                records[-1].update(p_method="exact paired task sign-flip via integer-sum multiplicities", p_value=exact["p_value"], exact_counts=exact)
            columns.append(d)
    # RR has task-specific denominators; retain Monte Carlo sign flips there.
    d = np.stack(columns, axis=1)
    observed = abs(d.mean(axis=0))
    rng = np.random.default_rng(4300)
    counts = np.zeros(len(records), dtype=np.int64)
    for _ in range(40):
        signs = rng.integers(0, 2, size=(5000, len(ids)), dtype=np.int8) * 2 - 1
        null = abs(signs @ d / len(ids))
        counts += (null >= observed[None, :] - 1e-14).sum(axis=0)
    for index, row in enumerate(records):
        if row["metric"] == "RR":
            row.update(p_method="Monte Carlo paired task sign-flip, 200000 draws, plus-one correction", p_value=float((counts[index] + 1) / 200001))
    adjusted = holm_adjust([row["p_value"] for row in records])
    for row, p in zip(records, adjusted):
        row["holm_12_adjusted_p"] = float(p)
    result = {
        "status": "exploratory secondary inference on an already completed v1 experiment",
        "no_new_training_or_generation": True,
        "source_csv_sha256_lf_normalized": hashlib.sha256(source.read_text().encode()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tasks": 26, "inference_seeds": [17, 29, 43], "training_seeds": [4300],
        "comparisons": records,
        "assumptions_and_limits": [
            "Average inference repetitions within task before inference; the unit is the observation task.",
            "Bootstrap intervals are pointwise, not multiplicity-adjusted confidence intervals.",
            "Sign-flip inference assumes paired-task label exchangeability under its null; correlated task families may weaken generalization.",
            "Holm controls the explicitly listed 12-comparison family under valid component p-values, not every past exploratory choice.",
            "Secondary inference was added after observing primary results; correction does not turn it into a preregistered confirmatory study.",
            "Only one adapter-training seed was run; intervals do not include between-training-run uncertainty.",
            "NR includes in-space structures that violate observations; it is not valid-reference diversity or course novelty.",
            "The original RR primary outcome and its inconclusive matched-SFT comparison are unchanged."
        ]
    }
    OUT.mkdir(parents=True)
    (OUT / "secondary_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    for row in records:
        print(row["comparator"], row["metric"], f"delta={row['difference_percentage_points']:+.4f}pp", f"Holm p={row['holm_12_adjusted_p']:.6g}")


if __name__ == "__main__":
    main()
