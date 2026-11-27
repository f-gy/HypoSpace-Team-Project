"""Chat endpoint transport with persistent call limits and no hidden retries."""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .tasks import write_json
from .upstream import ROOT


def load_env():
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip() and not line.lstrip().startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


class RealBackend:
    kind = "real"

    def __init__(self, config, ledger_path):
        load_env()
        self.base = os.environ.get("LLM_BASE_URL", "").rstrip("/")
        self.request_model = os.environ.get("LLM_MODEL", "")
        # Local servers may expose only an alias such as 'default_model'.
        # Preserve the actual model identity in trajectories and checkpoints.
        self.model = config.get("model_identity", self.request_model)
        self.key = os.environ.get("LLM_API_KEY", "")
        parsed = urlparse(self.base)
        local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        if not self.request_model or not parsed.netloc or "YOUR_PROVIDER" in self.base:
            raise ValueError("Configure LLM_BASE_URL and LLM_MODEL in local .env first")
        if not local and (parsed.scheme != "https" or not self.key):
            raise ValueError("Remote endpoints require HTTPS and LLM_API_KEY; use an SSH tunnel for local GPU serving")
        self.config = config
        self.ledger_path = Path(ledger_path)
        self.ledger = json.loads(self.ledger_path.read_text()) if self.ledger_path.exists() else {"calls": 0, "tokens": 0, "tokens_complete": True}

    def query(self, prompt, action, task, rng):
        if self.ledger["calls"] >= self.config["max_api_calls"]:
            raise RuntimeError("Persistent API call limit reached")
        # UTF-8 bytes + output cap conservatively bounds common byte-tokenizer
        # prompt usage. Missing usage stops the next call rather than hiding cost.
        reserve = len(prompt.encode("utf-8")) + self.config["max_output_tokens"] + 256
        if not self.ledger["tokens_complete"] or self.ledger["tokens"] + reserve > self.config["max_total_tokens"]:
            raise RuntimeError("Token limit reached, or previous provider response omitted token usage")
        self.ledger["calls"] += 1
        # Reserve before sending: errors/timeouts are never silently free.
        self.ledger["tokens"] += reserve
        write_json(self.ledger_path, self.ledger)
        body = {"model": self.request_model, "messages": [{"role": "user", "content": prompt}],
                "temperature": self.config["temperature"], "max_tokens": self.config["max_output_tokens"]}
        for option in ["top_p", "top_k", "min_p", "repetition_penalty"]:
            if option in self.config:
                body[option] = self.config[option]
        request_seed = rng.getrandbits(31) if self.config.get("send_seed", False) else None
        if request_seed is not None:
            body["seed"] = request_seed
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        request = Request(self.base + "/chat/completions", data=json.dumps(body).encode(), headers=headers)
        start = time.monotonic()
        try:
            with urlopen(request, timeout=self.config["request_timeout_seconds"]) as response:
                payload = json.load(response)
            text = payload["choices"][0]["message"]["content"]
            usage = payload.get("usage", {})
            total = usage.get("total_tokens")
            if total is not None:
                self.ledger["tokens"] += int(total) - reserve
            else:
                self.ledger["tokens_complete"] = False
            write_json(self.ledger_path, self.ledger)
            return {"text": text or "", "usage": usage, "error": None, "seconds": time.monotonic() - start,
                    "request_seed": request_seed}
        except HTTPError as e:
            # Do not log remote response bodies or request headers: either may
            # contain provider/account information. Failed reservations persist.
            return {"text": "", "usage": {}, "error": f"HTTP {e.code}", "seconds": time.monotonic() - start,
                    "request_seed": request_seed}
        except (URLError, TimeoutError, KeyError, TypeError, ValueError) as e:
            return {"text": "", "usage": {}, "error": type(e).__name__, "seconds": time.monotonic() - start,
                    "request_seed": request_seed}


class SyntheticBackend:
    """Privileged test fixture. Uses reference answers; NEVER LLM evidence."""
    kind = "synthetic_smoke_only"
    model = "synthetic_fixture_NOT_A_LANGUAGE_MODEL"

    def __init__(self, universe, tasks):
        self.universe = universe
        self.references = {t['id']: set(t['reference_keys']) for t in tasks}

    def query(self, prompt, action, task, rng: random.Random):
        valid = [e for e in self.universe if e["key"] in self.references[task['id']]]
        pool = sorted(valid, key=lambda e: len(e["formula"]))
        if action.startswith("root_"):
            op = action[5:].upper()
            subset = [e for e in pool if e["key"].startswith("('" + op + "'")]
            pool = subset or pool
        if action == "free" and rng.random() < 0.8:
            pool = pool[:2]
        e = rng.choice(pool)
        return {"text": "Expression: " + e["formula"], "usage": {}, "error": None, "seconds": 0.0}
