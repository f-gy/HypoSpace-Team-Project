"""Read prompt-only snapshots; no prior test answers enter the new proposal."""
import gzip
import json
from functools import lru_cache

from .upstream import ROOT


@lru_cache(maxsize=1)
def archive():
    with gzip.open(ROOT/'results/optimization_prompts.json.gz','rt') as f:
        return json.load(f)


def recorded_prompt(split, task_id, seed):
    return archive()['prompts'][f'{split}:{task_id}:{seed}']
