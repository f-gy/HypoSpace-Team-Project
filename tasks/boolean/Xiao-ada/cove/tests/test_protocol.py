import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from explorer.backend import RealBackend
from explorer.experiment import assess
from explorer.policy import Predictor, features
from explorer.tasks import public_task
from explorer.report import summarize
from explorer.upstream import evaluator


META = {"variables": ["x", "y"], "operators": ["AND", "NOT", "OR"], "max_depth": 2,
        "mechanistic_opts": {"apply_commutativity": True, "apply_idempotence_and_or": True, "flatten_associativity": True}}
TASK = {"id": "test", "observations": [{"inputs": {"x": 1, "y": 1}, "output": 1}]}


class ProtocolTests(unittest.TestCase):
    def test_local_server_alias_preserves_model_identity(self):
        config = {"model_identity": "pinned-research-model"}
        with tempfile.TemporaryDirectory() as d, patch.dict("os.environ", {
            "LLM_BASE_URL": "http://127.0.0.1:8080/v1", "LLM_MODEL": "default_model", "LLM_API_KEY": ""}):
            backend = RealBackend(config, Path(d) / "ledger.json")
            self.assertEqual(backend.model, "pinned-research-model")
            self.assertEqual(backend.request_model, "default_model")

    def test_canonical_duplicates_are_not_new(self):
        ev = evaluator(META)
        a = assess(ev, META, TASK, "Expression: x AND y", [])
        b = assess(ev, META, TASK, "Expression: y AND x", [a])
        self.assertTrue(a["valid"] and a["new"])
        self.assertEqual(a["key"], b["key"])
        self.assertTrue(b["valid"])
        self.assertFalse(b["new"])

    def test_unique_invalid_is_distinct_from_valid(self):
        a = assess(evaluator(META), META, TASK, "Expression: NOT x", [])
        self.assertTrue(a["in_space"])
        self.assertFalse(a["valid"])
        self.assertFalse(a["new"])
        self.assertIn("Mismatched observations", a["feedback"])

    def test_malformed_and_unknown_symbols_fail(self):
        for text in ["Expression: (", "Expression: z", "Expression: True", "Expression: __import__('os').getcwd()"]:
            with self.subTest(text=text):
                a = assess(evaluator(META), META, TASK, text, [])
                self.assertFalse(a["valid"])

    def test_reference_answers_do_not_enter_policy_inputs(self):
        t = {**TASK, "reference_keys": ["SECRET_REFERENCE"]}
        self.assertNotIn("reference_keys", public_task(t))
        np.testing.assert_array_equal(features(public_task(t), [], "free", 16), features(TASK, [], "free", 16))

    def test_history_ablation_removes_outcome_features(self):
        a = {"action": "free", "key": "('VAR', 'x')", "valid": True, "new": True, "expression": "x"}
        b = dict(a, valid=False, new=False)
        np.testing.assert_array_equal(features(TASK, [a], "free", 16, True), features(TASK, [b], "free", 16, True))
        self.assertFalse(np.array_equal(features(TASK, [a], "free", 16), features(TASK, [b], "free", 16)))

    def test_fit_with_no_valid_examples_stays_finite(self):
        records = [{"task": TASK, "history": [], "action": "free", "budget": 16, "valid": False, "new": False} for _ in range(4)]
        model = Predictor.fit(records, {}, epochs=30)
        self.assertTrue(np.isfinite(model.weights).all())

    def test_attempted_http_error_consumes_budget_without_retry(self):
        from urllib.error import HTTPError
        config = {"max_api_calls": 1, "max_total_tokens": 100000, "max_output_tokens": 128,
                  "temperature": .8, "request_timeout_seconds": 1}
        with tempfile.TemporaryDirectory() as d, patch.dict("os.environ", {"LLM_BASE_URL": "http://127.0.0.1:8000/v1", "LLM_MODEL": "test", "LLM_API_KEY": ""}):
            b = RealBackend(config, Path(d) / "ledger.json")
            with patch("explorer.backend.urlopen", side_effect=HTTPError("", 429, "limited", None, None)) as call:
                r = b.query("hello", "free", TASK, random.Random(0))
                self.assertEqual(r["error"], "HTTP 429")
                self.assertEqual(call.call_count, 1)
            self.assertEqual(json.loads((Path(d) / "ledger.json").read_text())["calls"], 1)
            with self.assertRaises(RuntimeError):
                b.query("hello", "free", TASK, random.Random(0))

    def test_metrics_count_unique_invalid_in_nr_but_not_rr(self):
        ev = evaluator(META)
        outcomes = []
        for text in ["Expression: x AND y", "Expression: y AND x", "Expression: NOT x"]:
            outcomes.append(assess(ev, META, TASK, text, outcomes))
        key = outcomes[0]["key"]
        dataset = {"tasks": [{**TASK, "reference_keys": [key, "another_reference"]}]}
        contract = {"backend": "synthetic_smoke_only", "model": "fixture", "fingerprint": "fixture",
                    "task_id": "test", "method": "memory", "seed": 1, "budget": 3}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "test_memory_test_1.jsonl"
            p.write_text("\n".join(json.dumps({"contract": contract, "outcome": o}) for o in outcomes) + "\n")
            result = summarize(dataset, d, Path(d) / "report")
            self.assertAlmostEqual(result["metrics"]["memory"]["VR"], 2 / 3)
            self.assertAlmostEqual(result["metrics"]["memory"]["NR"], 2 / 3)
            self.assertAlmostEqual(result["metrics"]["memory"]["RR"], .5)
            self.assertIn("SYNTHETIC", result["notice"])


if __name__ == "__main__":
    unittest.main()
