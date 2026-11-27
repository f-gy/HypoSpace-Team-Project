"""Load checksum-verified upstream code without its unused provider dependencies."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from urllib.request import Request, urlopen

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "create_prompt", "parse_llm_response", "get_expression_mechanistic_key",
    "_in_space", "_get_ast_depth", "_check_operators_in_ast", "validate_expression",
}
_LOADED = None


def fetch() -> dict:
    lock = json.loads((ROOT / "upstream.lock.json").read_text())
    for name, digest in lock["files"].items():
        dest = ROOT / ".cache/upstream" / name
        if not dest.exists():
            url = f"https://raw.githubusercontent.com/CTT-Pavilion/_HypoSpace/{lock['commit']}/{name}"
            req = Request(url, headers={"User-Agent": "EEC4300-HypoSpace-project"})
            data = urlopen(req, timeout=60).read()
            if hashlib.sha256(data).hexdigest() != digest:
                raise RuntimeError(f"Upstream checksum mismatch: {name}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        if hashlib.sha256(dest.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Upstream checksum mismatch: {name}")
    return lock


def load():
    global _LOADED
    if _LOADED is not None:
        return _LOADED
    fetch()
    source = ROOT / ".cache/upstream/boolean"
    spec = importlib.util.spec_from_file_location("hypospace_pinned_dataset", source / "boolean_dataset.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    tree = ast.parse((source / "boolean_benchmark.py").read_text())
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BooleanBenchmarkRefined")
    # Reuse original method ASTs unchanged. Avoid importing scipy and API SDKs
    # used only by the upstream monolithic runner, which we do not invoke.
    original.body = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in METHODS]
    if {n.name for n in original.body} != METHODS:
        raise RuntimeError("Pinned upstream adapter methods are incomplete")
    code = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), original], type_ignores=[])
    namespace = {"sp": sp, "re": re, "BooleanExpression": module.BooleanExpression}
    exec(compile(ast.fix_missing_locations(code), str(source / "boolean_benchmark.py"), "exec"), namespace)
    _LOADED = module, namespace["BooleanBenchmarkRefined"]
    return _LOADED


def evaluator(metadata: dict):
    _, cls = load()
    obj = cls.__new__(cls)
    obj.variables = metadata["variables"]
    obj.operators = set(metadata["operators"])
    obj.max_depth = metadata["max_depth"]
    obj.mechanistic_opts = metadata["mechanistic_opts"]
    obj.complete_dataset = {"metadata": metadata}
    return obj
