"""Restore the published adapters into a fresh private evaluation directory."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];run=ROOT/'runs/final_v1';src=ROOT/'artifacts/adapters'
if run.exists() and any(run.iterdir()):raise SystemExit('Run directory is not empty; retain it and use a fresh checkout for reproduction')
manifest=json.loads((src/'manifest.json').read_text())
for name,digest in manifest.items():
 if hashlib.sha256((src/name).read_bytes()).hexdigest()!=digest:raise SystemExit('Adapter checksum mismatch: '+name)
run.mkdir(parents=True,exist_ok=True)
for name in ['balanced_sft','canonical_coverage','string_coverage']:shutil.copytree(src/name,run/'adapters'/name)
shutil.copyfile(ROOT/'results/final/training_summary.json',run/'training_summary.json')
print('Restored reviewed adapters; original base model download is still required')
