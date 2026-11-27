"""Restore the exact published training tokens, checking the pretraining manifest."""
import gzip,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];run=ROOT/'runs/final_v1';target=run/'training_data.json'
with gzip.open(ROOT/'results/final/training_targets.json.gz','rb') as f:raw=f.read()
expected=json.loads((ROOT/'results/final/source_manifest.json').read_text())['training_data_sha256']
if hashlib.sha256(raw).hexdigest()!=expected:raise SystemExit('Training target checksum mismatch')
if target.exists() and target.read_bytes()!=raw:raise SystemExit('Different training data already exists; retain the run and use a fresh checkout')
run.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
print('Restored exact verified training tokens; no test labels included')
