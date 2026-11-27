import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from explorer.backend import RealBackend
from explorer.experiment import episode
from explorer.report import summarize
from test_protocol import META, TASK


class Stub:
    kind = "synthetic_unit_test"
    model = "unit_test_NOT_A_REAL_MODEL"

    def __init__(self):
        self.inputs = []

    def query(self, prompt, action, task, rng):
        self.inputs.append((prompt, task))
        return {"text": "Expression: x AND y", "usage": {}, "error": None, "seconds": 0}


class BaselineTests(unittest.TestCase):
    def test_iid_prompts_exclude_history_and_references_while_scoring_duplicates(self):
        task = {**TASK, "reference_keys": ["HIDDEN_REFERENCE"]}
        dataset = {"metadata": META, "splits": {"validation": ["test"]}}
        backend = Stub()
        with tempfile.TemporaryDirectory() as d:
            rows = episode(task, dataset, {}, backend, "base_iid", 101, 2, d, "validation")
        self.assertEqual(backend.inputs[0][0], backend.inputs[1][0])
        self.assertNotIn("HIDDEN_REFERENCE", backend.inputs[0][0])
        self.assertNotIn("reference_keys", backend.inputs[0][1])
        self.assertTrue(rows[0]["new"])
        self.assertFalse(rows[1]["new"])

    def test_sampling_controls_and_seed_are_sent_and_usage_is_accounted(self):
        config = dict(max_api_calls=2, max_total_tokens=100000, max_output_tokens=128,
                      temperature=1.0, request_timeout_seconds=1, top_p=1.0, top_k=0,
                      min_p=0.0, repetition_penalty=1.0, send_seed=True)
        env = {"LLM_BASE_URL": "http://127.0.0.1:8080/v1", "LLM_MODEL": "default_model", "LLM_API_KEY": ""}
        payload = {"choices": [{"message": {"content": "Expression: x"}}], "usage": {"total_tokens": 33}}
        with tempfile.TemporaryDirectory() as d, patch.dict("os.environ", env):
            backend = RealBackend(config, Path(d) / "ledger.json")
            with patch("explorer.backend.urlopen", return_value=io.BytesIO(json.dumps(payload).encode())) as call:
                response = backend.query("test", "free", TASK, random.Random(101))
                sent = json.loads(call.call_args.args[0].data)
            self.assertEqual(sent["seed"], response["request_seed"])
            self.assertEqual(sent["top_p"], 1.0)
            self.assertEqual(sent["top_k"], 0)
            self.assertEqual(sent["temperature"], 1.0)
            self.assertEqual(backend.ledger["tokens"], 33)
            self.assertEqual(backend.ledger["calls"], 1)

    def test_validation_report_does_not_include_test_records(self):
        dataset = {"tasks": [{**TASK, "reference_keys": ["key"]}]}
        contract = {"backend": "real", "model": "unit_test", "fingerprint": "unit_test",
                    "task_id": "test", "method": "base_iid", "seed": 101, "budget": 1}
        row = {"contract": contract, "outcome": {"valid": True, "in_space": True, "key": "key"},
               "usage": {"total_tokens": 5}, "seconds": 1}
        with tempfile.TemporaryDirectory() as d:
            Path(d, "validation_base.jsonl").write_text(json.dumps(row) + "\n")
            Path(d, "test_other.jsonl").write_text("not JSON; must never be read")
            report = summarize(dataset, d, Path(d) / "report", split="validation")
        self.assertEqual(report["split"], "validation")
        self.assertEqual(report["complete_episodes"], 1)
        self.assertIn("Validation-set diagnostic", report["notice"])


if __name__ == "__main__":
    unittest.main()
