"""Small, trainable validity/novelty predictors; no LLM fine-tuning required."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np

ACTIONS = ["free", "root_and", "root_or", "root_not", "complex", "repair", "mutate"]
ROOTS = ["VAR", "AND", "OR", "NOT", "NOR"]
HISTORY_START = 11


def root_of(key: str | None) -> str:
    if not key:
        return "UNKNOWN"
    try:
        return ast.literal_eval(key)[0]
    except (ValueError, SyntaxError, TypeError):
        return "UNKNOWN"


def features(task: dict, history: list, action: str, budget: int, no_history=False) -> np.ndarray:
    obs = [-1.0] * 4
    for o in task["observations"]:
        obs[2 * o["inputs"]["x"] + o["inputs"]["y"]] = float(o["output"])
    t = len(history)
    base = [1.0, *obs, *[float(v >= 0) for v in obs], len(task["observations"]) / 4, t / max(1, budget)]
    valid = [h for h in history if h["valid"]]
    keys = {h["key"] for h in valid}
    base += [len(valid) / max(1, t), len(keys) / max(1, t),
             sum(h["new"] for h in history[-4:]) / max(1, min(t, 4)),
             float(bool(history) and not history[-1]["valid"]),
             float(bool(history) and history[-1]["valid"] and not history[-1]["new"])]
    base += [sum(root_of(k) == r for k in keys) / max(1, len(keys)) for r in ROOTS]
    for a in ACTIONS:
        prior = [h for h in history if h["action"] == a]
        base += [len(prior) / max(1, budget), sum(h["new"] for h in prior) / max(1, len(prior))]
    if no_history:
        # Keep elapsed budget, remove all outcome/history statistics.
        base[HISTORY_START:] = [0.0] * (len(base) - HISTORY_START)
    base = np.asarray(base, dtype=float)
    onehot = np.eye(len(ACTIONS))[ACTIONS.index(action)]
    return np.concatenate([base, onehot, np.kron(onehot, base)])


def available(history: list) -> list[str]:
    actions = ACTIONS[:5]
    if history and history[-1]["expression"]:
        actions += ["mutate" if history[-1]["valid"] else "repair"]
    return actions


class Predictor:
    """Two regularized logistic heads trained with CPU full-batch Adam."""

    def __init__(self, weights, provenance):
        self.weights = np.asarray(weights, dtype=float)
        self.provenance = provenance

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    @classmethod
    def fit(cls, records, provenance, no_history=False, epochs=600):
        X = np.asarray([features(r["task"], r["history"], r["action"], r["budget"], no_history) for r in records])
        Y = np.asarray([[r["valid"], r["new"]] for r in records], dtype=float)
        W = np.zeros((X.shape[1], 2))
        m, v = np.zeros_like(W), np.zeros_like(W)
        masks = np.column_stack([np.ones(len(records)), Y[:, 0]])
        denominators = np.maximum(1, masks.sum(axis=0))
        for epoch in range(1, epochs + 1):
            P = cls.sigmoid(X @ W)
            grad = X.T @ ((P - Y) * masks / denominators) + 0.02 * W
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad * grad
            W -= 0.03 * (m / (1 - 0.9 ** epoch)) / (np.sqrt(v / (1 - 0.999 ** epoch)) + 1e-8)
        P = cls.sigmoid(X @ W)
        loss = -((Y * np.log(P + 1e-9) + (1 - Y) * np.log(1 - P + 1e-9)) * masks).sum(axis=0) / denominators
        return cls(W, {**provenance, "no_history": no_history, "training_rows": len(records), "training_bce": loss.tolist()})

    def choose(self, task, history, budget):
        candidates = available(history)
        scores = []
        for a in candidates:
            x = features(task, history, a, budget, self.provenance["no_history"])
            p = self.sigmoid(x @ self.weights)
            scores.append(float(p[0] * p[1]))
        return candidates[int(np.argmax(scores))]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"weights": self.weights.tolist(), "provenance": self.provenance}, indent=2) + "\n")

    @classmethod
    def read(cls, path):
        value = json.loads(Path(path).read_text())
        return cls(value["weights"], value["provenance"])


def guide(action: str, history: list) -> str:
    if action == "free":
        return ""
    if action.startswith("root_"):
        return f"Additional search direction: use {action[5:].upper()} as the root operator. All original constraints still apply."
    if action == "complex":
        return "Additional search direction: seek a new expression with nested operators, using the allowed depth budget. All original constraints still apply."
    last = history[-1]
    if action == "repair":
        return (f"Additional search direction: repair this failed candidate: {last['expression']}. "
                f"Validator feedback: {last['feedback']}. Return one corrected expression under all original constraints.")
    if action == "mutate":
        return (f"Additional search direction: replace a subexpression of {last['expression']} to obtain a structurally distinct expression. "
                "Recheck every observation and every original constraint. Return one expression.")
    raise ValueError(action)
