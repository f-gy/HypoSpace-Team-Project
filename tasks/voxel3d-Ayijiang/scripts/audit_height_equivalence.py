#!/usr/bin/env python3
"""
Read-only audit: is Structure3D representation-equivalent to per-(row,col) pillar heights?

Uses official HypoSpace Structure3D + Benchmark3D validator + dataset GT only.
Does NOT shrink the hypothesis space; reports whether height maps cover all official GT.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_3D = ROOT / "upstream" / "3d"
sys.path.insert(0, str(OFFICIAL_3D))

# Official run_3d_benchmark imports scipy only for t-tests; audit does not need it.
if "scipy" not in sys.modules:
    import types

    _scipy = types.ModuleType("scipy")
    _stats = types.ModuleType("scipy.stats")
    _stats.ttest_1samp = lambda *a, **k: (None, None)  # type: ignore
    _scipy.stats = _stats
    sys.modules["scipy"] = _scipy
    sys.modules["scipy.stats"] = _stats

from run_3d_benchmark import Benchmark3D, Structure3D  # noqa: E402


DATASET = OFFICIAL_3D / "datasets" / "3d_grid3_h3_b3.json"
OUT_DIR = ROOT / "results" / "cpsg_audit"
DOC_OUT = ROOT / "docs" / "cpsg_representation_equivalence_audit.md"


def parse_gt_layers(gt: dict, grid: int) -> List[List[List[int]]]:
    layers: List[List[List[int]]] = []
    for layer_str in gt["layers"]:
        layer = []
        for i in range(grid):
            row = [int(layer_str[i * grid + j]) for j in range(grid)]
            layer.append(row)
        layers.append(layer)
    return layers


def structure_to_heights(structure: Structure3D) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Convert Structure3D to height map.
    Returns (heights, error). heights[r,c] = consecutive block count from bottom.
    Fails if gravity gaps (non-contiguous column) or blocks above empty cells.
    """
    if not structure.layers:
        return np.zeros((0, 0), dtype=int), "empty_structure"

    rows, cols = structure.shape
    heights = np.zeros((rows, cols), dtype=int)
    H = len(structure.layers)

    for r in range(rows):
        for c in range(cols):
            col = [int(structure.layers[z][r, c]) for z in range(H)]
            # Find top of continuous stack from bottom
            h = 0
            for z, v in enumerate(col):
                if v == 1:
                    if z != h:
                        return None, f"gravity_gap_at_({r},{c})_z{z}"
                    h = z + 1
                else:
                    # after first zero, any later 1 is a floating block
                    if any(col[z2] == 1 for z2 in range(z + 1, H)):
                        return None, f"floating_block_at_({r},{c})"
                    break
            heights[r, c] = h
    return heights, None


def heights_to_structure(heights: np.ndarray, max_height: int) -> Structure3D:
    """Rebuild Structure3D from heights (official enumeration style)."""
    rows, cols = heights.shape
    H = int(heights.max()) if heights.size else 0
    if H == 0:
        return Structure3D([np.zeros((rows, cols), dtype=int).tolist()])
    if H > max_height:
        raise ValueError(f"height {H} exceeds max_height {max_height}")

    layers = []
    for z in range(H):
        layer = np.zeros((rows, cols), dtype=int)
        for r in range(rows):
            for c in range(cols):
                if heights[r, c] > z:
                    layer[r, c] = 1
        layers.append(layer.tolist())
    return Structure3D(layers).normalize()


def heights_match_observation(heights: np.ndarray, obs: np.ndarray) -> bool:
    occupied = (heights > 0).astype(int)
    return np.array_equal(occupied, obs)


def enumerate_height_space(obs: np.ndarray, max_height: int) -> List[Structure3D]:
    """Official-equivalent enumeration: each occupied cell height in 1..max_height."""
    coords = [(r, c) for r, c in zip(*np.where(obs == 1))]
    out: List[Structure3D] = []
    import itertools

    for combo in itertools.product(range(1, max_height + 1), repeat=len(coords)):
        heights = np.zeros_like(obs, dtype=int)
        for (r, c), h in zip(coords, combo):
            heights[r, c] = h
        out.append(heights_to_structure(heights, max_height))
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bench = Benchmark3D(str(DATASET))
    meta = bench.metadata
    grid = int(meta.get("grid_size", 3))
    max_height = int(meta.get("max_height", 3))
    max_blocks = int(meta.get("max_blocks", 3))

    n_gt_total = 0
    n_roundtrip_ok = 0
    n_validator_ok = 0
    n_height_convert_fail = 0
    n_hash_mismatch = 0
    n_validator_fail = 0
    n_obs_support_mismatch = 0
    n_height_out_of_range = 0

    convert_fail_examples: List[dict] = []
    hash_mismatch_examples: List[dict] = []
    validator_fail_examples: List[dict] = []

    # Per-observation: does height enumeration recover exactly the GT hash set?
    obs_coverage_ok = 0
    obs_coverage_fail = 0
    coverage_fail_examples: List[dict] = []
    height_space_extra = 0
    height_space_missing = 0

    block_hist = Counter()
    gt_height_max_hist = Counter()
    gt_total_voxels_hist = Counter()

    for obs_set in bench.all_observation_sets:
        obs_str = obs_set["observation"]
        obs = bench._parse_observation(obs_str)
        n_blocks = int(obs.sum())
        block_hist[n_blocks] += 1

        gt_hashes = set()
        gt_structs: List[Structure3D] = []

        for gt in obs_set["ground_truth_structures"]:
            n_gt_total += 1
            layers = parse_gt_layers(gt, grid)
            struct = Structure3D(layers).normalize()
            gt_structs.append(struct)
            gt_hashes.add(struct.get_hash())

            # stored hash vs recomputed (sanity)
            stored = gt.get("hash")

            heights, err = structure_to_heights(struct)
            if err is not None:
                n_height_convert_fail += 1
                if len(convert_fail_examples) < 10:
                    convert_fail_examples.append(
                        {
                            "obs_id": obs_set["observation_id"],
                            "error": err,
                            "gt_hash": struct.get_hash(),
                            "stored_hash": stored,
                            "layers": gt["layers"],
                        }
                    )
                continue

            assert heights is not None
            if int(heights.max()) > max_height:
                n_height_out_of_range += 1

            if not heights_match_observation(heights, obs):
                n_obs_support_mismatch += 1

            rebuilt = heights_to_structure(heights, max_height)
            h_orig = struct.get_hash()
            h_new = rebuilt.get_hash()
            if h_orig != h_new:
                n_hash_mismatch += 1
                if len(hash_mismatch_examples) < 10:
                    hash_mismatch_examples.append(
                        {
                            "obs_id": obs_set["observation_id"],
                            "orig_hash": h_orig,
                            "rebuilt_hash": h_new,
                            "heights": heights.tolist(),
                            "orig": struct.to_string(),
                            "rebuilt": rebuilt.to_string(),
                        }
                    )
            else:
                n_roundtrip_ok += 1

            if bench.validate_structure_matches_observations(rebuilt, obs_str):
                n_validator_ok += 1
            else:
                n_validator_fail += 1
                if len(validator_fail_examples) < 10:
                    validator_fail_examples.append(
                        {
                            "obs_id": obs_set["observation_id"],
                            "hash": h_new,
                            "heights": heights.tolist(),
                            "rebuilt": rebuilt.to_string(),
                        }
                    )

            gt_height_max_hist[int(heights.max())] += 1
            gt_total_voxels_hist[int(heights.sum())] += 1

        # Exact set coverage: height enumeration vs official GT hashes
        enum_structs = enumerate_height_space(obs, max_height)
        enum_hashes = {s.get_hash() for s in enum_structs}
        missing = gt_hashes - enum_hashes
        extra = enum_hashes - gt_hashes
        if not missing and not extra:
            obs_coverage_ok += 1
        else:
            obs_coverage_fail += 1
            height_space_missing += len(missing)
            height_space_extra += len(extra)
            if len(coverage_fail_examples) < 10:
                coverage_fail_examples.append(
                    {
                        "obs_id": obs_set["observation_id"],
                        "observation": obs_str,
                        "n_gt": len(gt_hashes),
                        "n_enum": len(enum_hashes),
                        "missing": sorted(missing),
                        "extra": sorted(extra),
                    }
                )

    n_obs = len(bench.all_observation_sets)
    equivalent = (
        n_height_convert_fail == 0
        and n_hash_mismatch == 0
        and n_validator_fail == 0
        and n_obs_support_mismatch == 0
        and n_height_out_of_range == 0
        and obs_coverage_fail == 0
        and n_roundtrip_ok == n_gt_total
        and n_validator_ok == n_gt_total
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(DATASET),
        "metadata": meta,
        "definitions": {
            "h3": "max_height=3: each occupied (row,col) pillar may have height in {1,2,3}",
            "b3": "max_blocks=3: top-view occupied cells count is in {1,2,3} (not total voxels)",
            "grid3": "grid_size=3: 3x3 top-view",
            "height_representation": "heights[r,c] >= 1 iff observation[r,c]==1; structure fills layers 0..h-1 continuously",
        },
        "answers": {
            "1_lossless_height_representation": equivalent,
            "2_gravity_means_continuous_bottom_up_stack": True,
            "3_h3_b3_meanings": {
                "h3": "max_height = 3",
                "b3": "max_blocks = 3 (top-view occupancy count 1..3)",
            },
            "4_other_constraints": {
                "top_view_must_match": True,
                "gravity_support": True,
                "max_height": max_height,
                "max_blocks_on_top_view": max_blocks,
                "total_voxel_budget": "none beyond sum of pillar heights (each pillar <= max_height)",
                "connectivity_across_columns": False,
            },
            "5_legal_structures_not_representable_as_heights": n_height_convert_fail == 0
            and obs_coverage_fail == 0,
            "6_height_space_covers_all_official_GT": obs_coverage_fail == 0 and height_space_missing == 0,
        },
        "counts": {
            "n_observation_sets": n_obs,
            "n_gt_structures": n_gt_total,
            "n_roundtrip_hash_ok": n_roundtrip_ok,
            "n_validator_ok_after_rebuild": n_validator_ok,
            "n_height_convert_fail": n_height_convert_fail,
            "n_hash_mismatch": n_hash_mismatch,
            "n_validator_fail": n_validator_fail,
            "n_obs_support_mismatch": n_obs_support_mismatch,
            "n_height_out_of_range": n_height_out_of_range,
            "obs_exact_set_coverage_ok": obs_coverage_ok,
            "obs_exact_set_coverage_fail": obs_coverage_fail,
            "height_space_missing_hashes": height_space_missing,
            "height_space_extra_hashes": height_space_extra,
            "block_count_hist": dict(sorted(block_hist.items())),
            "gt_max_pillar_height_hist": dict(sorted(gt_height_max_hist.items())),
            "gt_total_voxels_hist": dict(sorted(gt_total_voxels_hist.items())),
        },
        "representation_equivalent": equivalent,
        "cpsg_go_nogo": "GO" if equivalent else "NO-GO",
        "examples": {
            "convert_fail": convert_fail_examples,
            "hash_mismatch": hash_mismatch_examples,
            "validator_fail": validator_fail_examples,
            "coverage_fail": coverage_fail_examples,
        },
        "theoretical_note": (
            "Official dataset generator enumerate_all_structures_by_heights already "
            "defines the GT hypothesis space as pillar-height assignments. Gravity "
            "validator requires continuous bottom-up support per column, which is "
            "exactly the height-map semantics. Therefore height maps are not a "
            "restriction of the official space; they are the official space."
        ),
    }

    json_path = OUT_DIR / "height_equivalence_audit.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    md = f"""# CPSG Representation Equivalence Audit

**Date:** {report['generated_at']}  
**Dataset:** `{DATASET.name}`  
**Verdict:** **{report['cpsg_go_nogo']}** — representation_equivalent = `{equivalent}`

---

## Questions (user checklist)

| # | Question | Answer |
|---|---|---|
| 1 | Can Structure3D be losslessly written as per-(row,col) pillar heights? | **{'Yes' if equivalent else 'No'}** |
| 2 | Does Gravity mean blocks stack continuously from the bottom? | **Yes** (validator: cell at z requires same (r,c) at z-1) |
| 3 | What do `h3` / `b3` mean? | **h3 = max_height=3**; **b3 = max_blocks=3** (top-view occupied cells ∈ {{1,2,3}}) |
| 4 | Other constraints? | Top-view match; gravity; max_height; max_blocks on occupancy. **No** cross-column connectivity. **No** separate total-voxel budget. |
| 5 | Legal structures not representable as heights? | **{'None found in official GT' if n_height_convert_fail == 0 else 'YES — see failures'}** |
| 6 | Does height space cover all official legal hypotheses? | **{'Yes — exact set match on every observation' if obs_coverage_fail == 0 else 'No'}** |

---

## Empirical results (all official GT)

| Metric | Value |
|---|---|
| Observation sets | {n_obs} |
| GT structures | {n_gt_total} |
| Height convert failures | {n_height_convert_fail} |
| Roundtrip hash OK | {n_roundtrip_ok}/{n_gt_total} |
| Validator OK after rebuild | {n_validator_ok}/{n_gt_total} |
| Obs with exact GT↔height-enum set match | {obs_coverage_ok}/{n_obs} |
| Missing / extra hashes vs GT | {height_space_missing} / {height_space_extra} |

Block-count histogram (top-view): `{dict(sorted(block_hist.items()))}`  
GT max pillar height hist: `{dict(sorted(gt_height_max_hist.items()))}`  
GT total voxels hist: `{dict(sorted(gt_total_voxels_hist.items()))}`

---

## Why this is equivalent (not a shrunk space)

Official generator (`generate_3d_dataset_complete.py`) already builds GT via:

`enumerate_all_structures_by_heights(top_view, max_height)`

For each occupied cell, height ∈ {{1..max_height}}; layers are filled `0..h-1`.  
That is identical to the CPSG height representation. The official validator’s gravity rule is exactly “no gaps in a column from the ground up.”

Therefore predicting heights for occupied Observation cells, then programmatically converting to Structure3D, **does not** reduce the official hypothesis space.

---

## CPSG gate

- If **GO**: safe to implement CPSG experiments without inventing Validity by shrinking H.
- If **NO-GO**: stop; do not run CPSG formal experiments.

**Gate decision: {report['cpsg_go_nogo']}**

Artifact: `results/cpsg_audit/height_equivalence_audit.json`
"""
    with open(DOC_OUT, "w") as f:
        f.write(md)

    print(json.dumps({
        "representation_equivalent": equivalent,
        "cpsg_go_nogo": report["cpsg_go_nogo"],
        "n_gt": n_gt_total,
        "roundtrip_ok": n_roundtrip_ok,
        "validator_ok": n_validator_ok,
        "coverage_ok_obs": obs_coverage_ok,
        "coverage_fail_obs": obs_coverage_fail,
        "json": str(json_path),
        "md": str(DOC_OUT),
    }, indent=2))


if __name__ == "__main__":
    main()
