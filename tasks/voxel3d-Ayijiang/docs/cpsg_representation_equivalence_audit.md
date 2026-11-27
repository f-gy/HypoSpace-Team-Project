# CPSG Representation Equivalence Audit

**Date:** 2026-11-26T09:01:17.556748+00:00  
**Dataset:** `3d_grid3_h3_b3.json`  
**Verdict:** **GO** — representation_equivalent = `True`

---

## Questions (user checklist)

| # | Question | Answer |
|---|---|---|
| 1 | Can Structure3D be losslessly written as per-(row,col) pillar heights? | **Yes** |
| 2 | Does Gravity mean blocks stack continuously from the bottom? | **Yes** (validator: cell at z requires same (r,c) at z-1) |
| 3 | What do `h3` / `b3` mean? | **h3 = max_height=3**; **b3 = max_blocks=3** (top-view occupied cells ∈ {1,2,3}) |
| 4 | Other constraints? | Top-view match; gravity; max_height; max_blocks on occupancy. **No** cross-column connectivity. **No** separate total-voxel budget. |
| 5 | Legal structures not representable as heights? | **None found in official GT** |
| 6 | Does height space cover all official legal hypotheses? | **Yes — exact set match on every observation** |

---

## Empirical results (all official GT)

| Metric | Value |
|---|---|
| Observation sets | 129 |
| GT structures | 2619 |
| Height convert failures | 0 |
| Roundtrip hash OK | 2619/2619 |
| Validator OK after rebuild | 2619/2619 |
| Obs with exact GT↔height-enum set match | 129/129 |
| Missing / extra hashes vs GT | 0 / 0 |

Block-count histogram (top-view): `{1: 9, 2: 36, 3: 84}`  
GT max pillar height hist: `{1: 129, 2: 705, 3: 1785}`  
GT total voxels hist: `{1: 9, 2: 45, 3: 165, 4: 360, 5: 576, 6: 624, 7: 504, 8: 252, 9: 84}`

---

## Why this is equivalent (not a shrunk space)

Official generator (`generate_3d_dataset_complete.py`) already builds GT via:

`enumerate_all_structures_by_heights(top_view, max_height)`

For each occupied cell, height ∈ {1..max_height}; layers are filled `0..h-1`.  
That is identical to the CPSG height representation. The official validator’s gravity rule is exactly “no gaps in a column from the ground up.”

Therefore predicting heights for occupied Observation cells, then programmatically converting to Structure3D, **does not** reduce the official hypothesis space.

---

## CPSG gate

- If **GO**: safe to implement CPSG experiments without inventing Validity by shrinking H.
- If **NO-GO**: stop; do not run CPSG formal experiments.

**Gate decision: GO**

Artifact: `results/cpsg_audit/height_equivalence_audit.json`
