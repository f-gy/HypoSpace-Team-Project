#!/usr/bin/env python3
"""Official HypoSpace 3D metrics for Baseline vs CPSG variants.

Uses unmodified Benchmark3D.evaluate_single_observation_set from
upstream/3d/run_3d_benchmark.py so that:
  Validity   = valid_rate
  Uniqueness = novelty_rate
  Recovery   = recovery_rate
are computed by the official code path (same validator, hash, GT recovery).

LLM adapters only change how the response text is produced:
  - official_prompt: answer the official Layer prompt as-is
  - cpsg_supportproj: CPSG height gen + SupportProj -> Structure text
  - cpsg_supportproj_retry: same + one range-feedback LLM retry (no silent fill)
    and one anti-dup attempt to protect Uniqueness

Fixed 50-obs pool (same as prior CPSG experiments). Does not modify official sources.
BAE program expansion is NOT included here (official loop only scores LLM queries).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXT_3D = ROOT / "upstream" / "3d"
sys.path.insert(0, str(EXT_3D))
sys.path.insert(0, str(ROOT / "src"))

# Official run_3d_benchmark imports scipy only for aggregate t-tests; not needed here.
if "scipy" not in sys.modules:
    import types

    _scipy = types.ModuleType("scipy")
    _stats = types.ModuleType("scipy.stats")
    _stats.ttest_1samp = lambda *a, **k: (None, None)  # type: ignore
    _scipy.stats = _stats
    sys.modules["scipy"] = _scipy
    sys.modules["scipy.stats"] = _stats

from modules.llm_interface import LLMInterface  # noqa: E402
from run_3d_benchmark import Benchmark3D  # noqa: E402
from methods.cpsg import (  # noqa: E402
    build_cpsg_prompt,
    build_cpsg_range_retry_prompt,
    cpsg_decode,
    structure_to_heights,
)

OUT_DEFAULT = ROOT / "results" / "official_hypospace"
OUT_DEFAULT.mkdir(parents=True, exist_ok=True)
OUT = OUT_DEFAULT

MODEL_CACHE = Path(os.environ.get("MODEL_CACHE_ROOT", str(ROOT / "model_cache")))
LOCAL_MODEL = Path(os.environ.get("QWEN_MODEL_PATH", str(MODEL_CACHE / "Qwen2.5-7B-Instruct")))
os.environ.setdefault("HF_HOME", str(MODEL_CACHE / "huggingface"))
# Offline only when local weights exist; otherwise allow download.
if LOCAL_MODEL.is_dir():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

OBS_POOL = ROOT / "configs" / "obs_pool_50.json"


def load_model(model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    local = Path(model_id).is_dir()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, local_files_only=local)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=local,
    )
    model.eval()
    return tok, model


def generate_text(tok, model, prompt: str, max_new_tokens: int, temperature: float) -> Tuple[str, int, int]:
    import torch

    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    kw: Dict[str, Any] = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0}
    if temperature > 0:
        kw["temperature"] = temperature
        kw["top_p"] = 0.95
    with torch.inference_mode():
        out = model.generate(**inputs, **kw)
    new = out[0][inputs["input_ids"].shape[-1] :]
    return tok.decode(new, skip_special_tokens=True), int(inputs["input_ids"].shape[-1]), int(new.shape[-1])


def extract_top_view_from_official_prompt(prompt: str) -> Optional[np.ndarray]:
    """Parse the Top view grid printed by Benchmark3D.create_prompt."""
    m = re.search(r"Top view:\s*\n((?:\s*[0-9][0-9\s]*\n)+)", prompt)
    if not m:
        return None
    rows = []
    for line in m.group(1).strip().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            rows.append([int(x) for x in parts])
        except ValueError:
            return None
    if not rows:
        return None
    arr = np.array(rows, dtype=int)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return None
    return arr


class OfficialPromptQwen(LLMInterface):
    """Answer the official HypoSpace Layer prompt with local Qwen."""

    def __init__(self, tok, model, temperature: float, max_new_tokens: int):
        self.tok = tok
        self.model = model
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def query(self, prompt: str) -> str:
        text, pin, pout = generate_text(
            self.tok, self.model, prompt, self.max_new_tokens, self.temperature
        )
        self.last_usage = {
            "prompt_tokens": pin,
            "completion_tokens": pout,
            "total_tokens": pin + pout,
        }
        return text

    def query_with_usage(self, prompt: str) -> Dict[str, Any]:
        resp = self.query(prompt)
        return {"response": resp, "usage": dict(self.last_usage), "cost": 0.0}

    def get_name(self) -> str:
        return "qwen2.5-7b-instruct_official_prompt"


class CPSGSupportProjQwen(LLMInterface):
    """CPSG height generation + SupportProj, emit official Structure text."""

    def __init__(
        self,
        tok,
        model,
        temperature: float,
        max_new_tokens: int,
        max_height: int,
        max_blocks: int,
    ):
        self.tok = tok
        self.model = model
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.max_height = max_height
        self.max_blocks = max_blocks
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def query(self, prompt: str) -> str:
        obs = extract_top_view_from_official_prompt(prompt)
        if obs is None:
            return ""
        cpsg_prompt = build_cpsg_prompt(obs, self.max_height, self.max_blocks)
        raw, pin, pout = generate_text(
            self.tok, self.model, cpsg_prompt, self.max_new_tokens, self.temperature
        )
        self.last_usage = {
            "prompt_tokens": pin,
            "completion_tokens": pout,
            "total_tokens": pin + pout,
        }
        struct, info = cpsg_decode(raw, obs, self.max_height, support_projection=True)
        if struct is None or info.get("error_class") != "OK":
            return ""  # official path: parse failure
        return "Structure:\n" + struct.to_string()

    def query_with_usage(self, prompt: str) -> Dict[str, Any]:
        resp = self.query(prompt)
        return {"response": resp, "usage": dict(self.last_usage), "cost": 0.0}

    def get_name(self) -> str:
        return "qwen2.5-7b-instruct_cpsg_supportproj"


class CPSGSupportProjRetryQwen(LLMInterface):
    """CPSG + SupportProj + one range-feedback retry; anti-dup attempt for U.

    No silent height fill. Extra LLM calls happen inside one official query slot.
    """

    def __init__(
        self,
        tok,
        model,
        temperature: float,
        max_new_tokens: int,
        max_height: int,
        max_blocks: int,
    ):
        self.tok = tok
        self.model = model
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.max_height = max_height
        self.max_blocks = max_blocks
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self._obs_key: Optional[str] = None
        self._avoid_heights: List[np.ndarray] = []
        self._avoid_hashes: set = set()
        self.stats = {
            "first_ok": 0,
            "range_retry_ok": 0,
            "range_retry_fail": 0,
            "antidup_ok": 0,
            "antidup_keep_first": 0,
            "hard_fail": 0,
        }

    def _reset_obs_if_needed(self, obs: np.ndarray) -> None:
        key = "".join(str(int(x)) for x in obs.flatten())
        if key != self._obs_key:
            self._obs_key = key
            self._avoid_heights = []
            self._avoid_hashes = set()

    def _accumulate_usage(self, pin: int, pout: int) -> None:
        self.last_usage["prompt_tokens"] += pin
        self.last_usage["completion_tokens"] += pout
        self.last_usage["total_tokens"] += pin + pout

    def _gen_decode(
        self,
        prompt: str,
        obs: np.ndarray,
    ) -> Tuple[Optional[Any], Dict[str, Any]]:
        raw, pin, pout = generate_text(
            self.tok, self.model, prompt, self.max_new_tokens, self.temperature
        )
        self._accumulate_usage(pin, pout)
        return cpsg_decode(raw, obs, self.max_height, support_projection=True)

    def query(self, prompt: str) -> str:
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        obs = extract_top_view_from_official_prompt(prompt)
        if obs is None:
            self.stats["hard_fail"] += 1
            return ""
        self._reset_obs_if_needed(obs)

        first_prompt = build_cpsg_prompt(
            obs, self.max_height, self.max_blocks, avoid_height_maps=self._avoid_heights
        )
        struct, info = self._gen_decode(first_prompt, obs)

        # One constrained LLM retry on remaining range / both failures (no silent fill)
        if struct is None or info.get("error_class") != "OK":
            err = info.get("error_class", "Parsing failure")
            if err in (
                "Height_range_violation",
                "Support_and_range_violation",
                "Parsing failure",
            ):
                bad = info.get("heights")
                bad_arr = np.array(bad, dtype=int) if bad is not None else None
                retry_prompt = build_cpsg_range_retry_prompt(
                    obs,
                    self.max_height,
                    self.max_blocks,
                    bad_arr,
                    list(info.get("problems") or []),
                    avoid_height_maps=self._avoid_heights,
                )
                struct2, info2 = self._gen_decode(retry_prompt, obs)
                if struct2 is not None and info2.get("error_class") == "OK":
                    struct, info = struct2, info2
                    self.stats["range_retry_ok"] += 1
                else:
                    self.stats["range_retry_fail"] += 1
                    self.stats["hard_fail"] += 1
                    return ""
            else:
                self.stats["hard_fail"] += 1
                return ""
        else:
            self.stats["first_ok"] += 1

        # Protect Uniqueness: if duplicate of prior valid, try one anti-dup regen; keep first if fail
        h_hash = struct.get_hash()
        if h_hash in self._avoid_hashes:
            avoid_prompt = build_cpsg_prompt(
                obs, self.max_height, self.max_blocks, avoid_height_maps=self._avoid_heights
            )
            struct3, info3 = self._gen_decode(avoid_prompt, obs)
            if (
                struct3 is not None
                and info3.get("error_class") == "OK"
                and struct3.get_hash() not in self._avoid_hashes
            ):
                struct = struct3
                self.stats["antidup_ok"] += 1
            else:
                self.stats["antidup_keep_first"] += 1

        heights = structure_to_heights(struct)
        self._avoid_heights.append(heights.copy())
        self._avoid_hashes.add(struct.get_hash())
        return "Structure:\n" + struct.to_string()

    def query_with_usage(self, prompt: str) -> Dict[str, Any]:
        resp = self.query(prompt)
        return {"response": resp, "usage": dict(self.last_usage), "cost": 0.0}

    def get_name(self) -> str:
        return "qwen2.5-7b-instruct_cpsg_supportproj_retry"


def resolve_obs_pool(bench: Benchmark3D, n_obs: int) -> List[dict]:
    if OBS_POOL.exists():
        design = json.loads(OBS_POOL.read_text(encoding="utf-8"))
        ids = design.get("obs_ids", [])
        by_id = {o["observation_id"]: o for o in bench.all_observation_sets}
        pool = [by_id[i] for i in ids if i in by_id]
        if pool:
            return pool[:n_obs]
    return bench.sample_observation_sets(n_obs, seed=7)


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(EXT_3D / "datasets" / "3d_grid3_h3_b3.json"))
    ap.add_argument("--model-id", default=str(LOCAL_MODEL), help="Local Qwen path or HF id")
    ap.add_argument("--n-obs", type=int, default=50)
    ap.add_argument("--n-queries", type=int, default=3, help="Official fixed queries per obs")
    ap.add_argument("--max-retries", type=int, default=1, help="Keep call budget = n_queries")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max-new-tokens-official", type=int, default=256)
    ap.add_argument("--max-new-tokens-cpsg", type=int, default=128)
    ap.add_argument("--seeds", default="7,17,27")
    ap.add_argument(
        "--methods",
        default="cpsg_supportproj_retry",
        help="Comma list: official_prompt,cpsg_supportproj,cpsg_supportproj_retry",
    )
    ap.add_argument(
        "--out-dir",
        default=str(ROOT / "results" / "official_hypospace_v90"),
        help="Results directory (default: official_hypospace_v90)",
    )
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    OUT = Path(args.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    if args.pilot:
        args.n_obs = 4
        args.seeds = "7"
        args.n_queries = 2

    seeds = [int(x) for x in args.seeds.split(",")]
    methods = [m.strip() for m in args.methods.split(",")]

    bench = Benchmark3D(args.dataset)
    max_height = int(bench.metadata.get("max_height", 3))
    max_blocks = int(bench.metadata.get("max_blocks", 3))
    obs_pool = resolve_obs_pool(bench, args.n_obs)

    design = {
        "protocol": "official_Benchmark3D.evaluate_single_observation_set",
        "metrics": {
            "Validity": "valid_rate",
            "Uniqueness": "novelty_rate",
            "Recovery": "recovery_rate",
        },
        "n_obs": len(obs_pool),
        "obs_ids": [o["observation_id"] for o in obs_pool],
        "n_queries": args.n_queries,
        "max_retries": args.max_retries,
        "temperature": args.temperature,
        "seeds": seeds,
        "methods": methods,
        "goal": "Push Validity toward >=90% while not dropping U/R vs cpsg_supportproj baseline (~35.6% / ~8.5%)",
        "note": (
            "Metrics computed only by official evaluate_single_observation_set. "
            "cpsg_supportproj_retry: SupportProj + one range-feedback LLM retry (no silent fill) "
            "+ optional anti-dup regen (keep first valid if anti-dup fails). "
            "BAE not included."
        ),
        "pilot": bool(args.pilot),
        "no_drop_gates": {"uniqueness_min": 0.356, "recovery_min": 0.085},
        "validity_target": 0.90,
    }
    (OUT / "experiment_design.json").write_text(json.dumps(design, indent=2) + "\n", encoding="utf-8")

    done = set()
    all_rows: List[Dict[str, Any]] = []
    ckpt = OUT / "checkpoint.json"
    if args.resume and ckpt.exists():
        prev = json.loads(ckpt.read_text(encoding="utf-8"))
        for row in prev.get("rows", []):
            if int(row.get("n_queries", -1)) != args.n_queries:
                continue
            all_rows.append(row)
            done.add((int(row["seed"]), row["method"], row["observation_set_id"]))
        print(f"Resumed {len(done)} keys", flush=True)

    print("Loading model...", flush=True)
    tok, model = load_model(args.model_id)
    t0 = time.perf_counter()

    for seed in seeds:
        import torch

        torch.manual_seed(seed)
        np.random.seed(seed)
        for method in methods:
            print(f"=== seed={seed} method={method} ===", flush=True)
            if method == "official_prompt":
                llm: LLMInterface = OfficialPromptQwen(
                    tok, model, args.temperature, args.max_new_tokens_official
                )
            elif method == "cpsg_supportproj":
                llm = CPSGSupportProjQwen(
                    tok,
                    model,
                    args.temperature,
                    args.max_new_tokens_cpsg,
                    max_height,
                    max_blocks,
                )
            elif method == "cpsg_supportproj_retry":
                llm = CPSGSupportProjRetryQwen(
                    tok,
                    model,
                    args.temperature,
                    args.max_new_tokens_cpsg,
                    max_height,
                    max_blocks,
                )
            else:
                raise ValueError(method)

            for oi, obs_set in enumerate(obs_pool):
                key = (seed, method, obs_set["observation_id"])
                if key in done:
                    continue
                # Official evaluation entrypoint
                result = bench.evaluate_single_observation_set(
                    llm,
                    obs_set,
                    n_queries=args.n_queries,
                    verbose=False,
                    max_retries=args.max_retries,
                )
                row = {
                    "seed": seed,
                    "method": method,
                    "observation_set_id": result["observation_set_id"],
                    "n_ground_truths": result["n_ground_truths"],
                    "n_queries": result["n_queries"],
                    "valid_rate": result["valid_rate"],
                    "novelty_rate": result["novelty_rate"],
                    "recovery_rate": result["recovery_rate"],
                    "parse_success_rate": result["parse_success_rate"],
                    "n_valid": result["n_valid"],
                    "n_unique": result["n_unique"],
                    "n_recovered_gts": result["n_recovered_gts"],
                    "token_usage": result["token_usage"],
                }
                all_rows.append(row)
                done.add(key)
                if (oi + 1) % 5 == 0 or oi == 0:
                    print(
                        f"  [{oi+1}/{len(obs_pool)}] {row['observation_set_id']} "
                        f"V={row['valid_rate']:.2f} U={row['novelty_rate']:.2f} "
                        f"R={row['recovery_rate']:.2f}",
                        flush=True,
                    )
                (OUT / "checkpoint.json").write_text(
                    json.dumps({"completed": len(all_rows), "design": design, "rows": all_rows}, indent=2),
                    encoding="utf-8",
                )

            if method == "cpsg_supportproj_retry" and hasattr(llm, "stats"):
                print(f"  adapter_stats={llm.stats}", flush=True)
                (OUT / f"adapter_stats_seed{seed}.json").write_text(
                    json.dumps(llm.stats, indent=2) + "\n", encoding="utf-8"
                )

    runtime = time.perf_counter() - t0

    summaries: Dict[str, Any] = {}
    for seed in seeds:
        summaries[str(seed)] = {}
        for method in methods:
            sub = [r for r in all_rows if r["seed"] == seed and r["method"] == method]
            if not sub:
                continue
            summaries[str(seed)][method] = {
                "n_obs": len(sub),
                "valid_rate": float(np.mean([r["valid_rate"] for r in sub])),
                "novelty_rate": float(np.mean([r["novelty_rate"] for r in sub])),
                "recovery_rate": float(np.mean([r["recovery_rate"] for r in sub])),
                "parse_success_rate": float(np.mean([r["parse_success_rate"] for r in sub])),
                "total_tokens": int(sum(r["token_usage"]["total_tokens"] for r in sub)),
            }

    across: Dict[str, Any] = {}
    for method in methods:
        per = [summaries[str(s)][method] for s in seeds if method in summaries.get(str(s), {})]
        if not per:
            continue
        across[method] = {
            k: {
                "mean": float(np.mean([p[k] for p in per])),
                "std": float(np.std([p[k] for p in per], ddof=0)),
            }
            for k in ["valid_rate", "novelty_rate", "recovery_rate", "parse_success_rate"]
        }

    # Compare to frozen prior official cpsg_supportproj means
    prior_path = ROOT / "reported-results" / "official_baseline_vs_cpsg_supportproj.json"
    if not prior_path.exists():
        prior_path = ROOT / "results" / "official_hypospace" / "summary.json"
    prior_ref = {"valid_rate": 0.8067, "novelty_rate": 0.3556, "recovery_rate": 0.0849}
    if prior_path.exists():
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        prev = prior.get("across_seed", {}).get("cpsg_supportproj", {})
        if prev:
            prior_ref = {k: prev[k]["mean"] for k in prior_ref if k in prev}

    gates = {}
    for method, m in across.items():
        gates[method] = {
            "validity_ge_90": m["valid_rate"]["mean"] >= 0.90,
            "uniqueness_no_drop": m["novelty_rate"]["mean"] + 1e-9 >= prior_ref["novelty_rate"] - 0.02,
            "recovery_no_drop": m["recovery_rate"]["mean"] + 1e-9 >= prior_ref["recovery_rate"] - 0.01,
            "vs_prior_cpsg_supportproj": prior_ref,
            "delta": {
                "valid_rate": m["valid_rate"]["mean"] - prior_ref["valid_rate"],
                "novelty_rate": m["novelty_rate"]["mean"] - prior_ref["novelty_rate"],
                "recovery_rate": m["recovery_rate"]["mean"] - prior_ref["recovery_rate"],
            },
        }

    payload = {
        "runtime_total_sec": runtime,
        "design": design,
        "summaries_by_seed": summaries,
        "across_seed": across,
        "gates": gates,
        "metric_names": {
            "Validity": "valid_rate (official)",
            "Uniqueness": "novelty_rate (official)",
            "Recovery": "recovery_rate (official)",
        },
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (OUT / "all_rows.json").write_text(json.dumps(all_rows, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"runtime_sec": runtime, "across_seed": across, "gates": gates}, indent=2), flush=True)
    print(f"Saved to {OUT}", flush=True)


if __name__ == "__main__":
    main()
