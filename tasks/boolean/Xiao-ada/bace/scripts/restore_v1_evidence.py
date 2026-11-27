"""Restore the exact public row projection needed for reproducible IID replay.

Existing private complete trajectories are verified field-by-field, not replaced.
This creates no model calls and supplies no reference labels to a proposal.
"""
import gzip
import json
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
count=0
for split in ['validation','test']:
    grouped=defaultdict(list)
    with gzip.open(ROOT/'results/final'/split/'evidence.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line);c=r['contract']
            name=f"{split}_{c['method']}_{c['task_id']}_{c['seed']}.jsonl"
            grouped[name].append(r)
    for name,rows in grouped.items():
        path=ROOT/'runs/final_v1'/name
        if path.exists():
            existing=[json.loads(l) for l in path.read_text().splitlines()]
            assert len(existing)==len(rows)
            for old,new in zip(existing,rows):
                for k,v in new.items():
                    assert old[k]==v,(name,k)
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        count+=len(rows)
print(f'Checked/restored {count} public v1 rows; existing complete trajectories preserved')
