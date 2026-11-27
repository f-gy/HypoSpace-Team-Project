"""Restore released v5 adapters and exact training targets; never overwrite differences."""
import gzip
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'artifacts/preference_adapters/manifest.json').read_text())
for name,digest in manifest['files'].items():
    source=ROOT/name
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    relative=source.relative_to(ROOT/'artifacts/preference_adapters')
    dest=ROOT/'runs/canonical_preference_v5/adapters'/relative
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        assert dest.read_bytes()==source.read_bytes(),str(dest)
    else:
        shutil.copyfile(source,dest)
with gzip.open(ROOT/'results/canonical_preference_v5/training_targets.json.gz','rb') as f:data=f.read()
target=ROOT/'runs/canonical_preference_v5/training_data.json'
target.parent.mkdir(parents=True,exist_ok=True)
if target.exists():assert target.read_bytes()==data
else:target.write_bytes(data)
print('Verified/restored both published v5 adapters and exact prepared training targets')
