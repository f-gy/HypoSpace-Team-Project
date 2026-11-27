"""Deterministic illustrations accompanying the analytic proofs, not LLM scores."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def hit_probability(p, n=16):
    return 1 - (1 - p) ** n


def main():
    out = ROOT / "results/contribution_audit/formal_examples.json"
    if out.exists():
        raise SystemExit("Retain the existing deterministic evidence")
    grouped_split = hit_probability(.1 + .2)
    grouped_unsplit = hit_probability(.3)
    strings = hit_probability(.1) + hit_probability(.2)
    assert abs(grouped_split - grouped_unsplit) < 1e-12 and strings > grouped_split
    pool_masses = [.2, .1]
    full_masses = [.3, .2]
    surrogate = sum(hit_probability(p) for p in pool_masses) / 2
    true_expected_rr = sum(hit_probability(p) for p in full_masses) / 2
    assert surrogate <= true_expected_rr
    # Equal-length aliases, two classes, ten answer tokens, uniform within class.
    k, t, n = 2, 10, 16
    masses = [.1, .01]
    anchor_gradient = [-1 / (k * t)] * k
    coverage_gradient = [-n / k * p * (1 - p) ** (n - 1) for p in masses]
    ratios = [b / a for a, b in zip(anchor_gradient, coverage_gradient)]
    assert abs(ratios[0] - ratios[1]) > 1e-3
    result = {
        "kind": "mathematical examples, not measured LLM results or a priority claim",
        "budget": 16,
        "mass_preserving_partition": {"parts": [.1, .2], "total": .3, "grouped_parts": grouped_split,
                                      "grouped_total": grouped_unsplit, "ungrouped_string_reward": strings},
        "lower_bound_example": {"finite_alias_masses": pool_masses, "full_valid_class_masses": full_masses,
                                "surrogate": surrogate, "true_expected_RR": true_expected_rr},
        "nonproportional_objective_gradients": {"coordinate": "log class mass; equal-length completion model",
                                                "classes": k, "answer_tokens": t, "class_masses": masses,
                                                "balanced_SFT_gradients": anchor_gradient,
                                                "negative_coverage_gradients": coverage_gradient,
                                                "coverage_to_SFT_ratios": ratios},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    }
    out.write_text(json.dumps(result, indent=2) + "\n")
    print("Verified mass-preserving invariance, finite-pool lower bound and nonproportional-gradient example")


if __name__ == "__main__":
    main()
