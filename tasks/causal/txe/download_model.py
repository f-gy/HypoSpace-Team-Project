import concurrent.futures
import hashlib
import json
import subprocess
from pathlib import Path

import requests

repo = "Qwen/Qwen3-4B-Instruct-2507"
revision = "2de2439ea21be1dc5cb21f22f88af07e43393cbb"
base = f"https://modelscope.cn/models/{repo}/resolve/{revision}"
response = requests.get(
    f"https://modelscope.cn/api/v1/models/{repo}/repo/files",
    params={"Revision": revision, "Recursive": "true"},
    timeout=25,
)
response.raise_for_status()
files = {file["Path"]: file for file in response.json()["Data"]["Files"]}
names = [
    "LICENSE", "config.json", "generation_config.json", "merges.txt",
    "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
    *(f"model-{index:05d}-of-00003.safetensors" for index in range(1, 4)),
]
destination = Path("models/Qwen3-4B-Instruct-2507")
destination.mkdir(parents=True, exist_ok=True)


def verified(path, metadata):
    if not path.is_file() or path.stat().st_size != metadata["Size"]:
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == metadata["Sha256"]


def fetch(name):
    metadata = files[name]
    target = destination / name
    if not verified(target, metadata):
        partial = target.with_name(f"{name}.part")
        if not verified(partial, metadata):
            print(f"Downloading {name} ({metadata['Size']:,} bytes)", flush=True)
            command = [
                "curl", "-fLsS", "--retry", "2", "--connect-timeout", "15",
                "--speed-limit", "1024", "--speed-time", "60",
            ]
            result = subprocess.run(
                [*command, "-C", "-", "-o", str(partial), f"{base}/{name}"],
            )
            if result.returncode == 33:
                subprocess.run(
                    [*command, "-o", str(partial), f"{base}/{name}"], check=True,
                )
            else:
                result.check_returncode()
        if not verified(partial, metadata):
            raise ValueError(f"Size or SHA-256 mismatch: {partial}")
        partial.replace(target)
    print(f"Verified {name}", flush=True)


with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(fetch, names))

(destination / "source-manifest.json").write_text(json.dumps({
    "repository": repo,
    "source": "https://modelscope.cn",
    "revision": revision,
    "files": {name: {key: files[name][key] for key in ("Size", "Sha256")} for name in names},
}, indent=2) + "\n")
