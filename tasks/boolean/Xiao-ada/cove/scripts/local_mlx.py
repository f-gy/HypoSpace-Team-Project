"""Download pinned safe weights and serve on loopback with a separate MLX env."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--model-dir", type=Path, default=ROOT.parent / "runtime/models/Qwen2.5-1.5B-Instruct-4bit")
    args = parser.parse_args()
    lock = json.loads((ROOT / "model.local_mlx.lock.json").read_text())
    target = args.model_dir.resolve()
    weights = target / lock["weights_file"]
    if not weights.exists():
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=lock["repo_id"], revision=lock["revision"], local_dir=target,
                          allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.txt", "tokenizer.model"])
    with weights.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    if weights.stat().st_size != lock["weights_bytes"] or digest != lock["weights_sha256"]:
        raise SystemExit("Model checksum mismatch; do not serve these weights")
    print("Verified pinned model weights", flush=True)
    if not args.download_only:
        os.execv(sys.executable, [sys.executable, "-m", "mlx_lm.server", "--model", str(target),
                 "--host", "127.0.0.1", "--port", str(args.port), "--prompt-cache-size", "2",
                 "--decode-concurrency", "1", "--prompt-concurrency", "1", "--log-level", "WARNING"])


if __name__ == "__main__":
    main()
