# Reproducibility

## Environment

- Python 3.10+
- CUDA GPU recommended for Qwen2.5-7B-Instruct (bf16)
- Dataset: `upstream/3d/datasets/3d_grid3_h3_b3.json` (grid3, max_height=3, max_blocks=3)
- Fixed observation pool: `configs/obs_pool_50.json` (50 IDs)
- Seeds: 7, 17, 27
- Official protocol: `n_queries=3`, `max_retries=1`
- Metrics: `valid_rate` / `novelty_rate` / `recovery_rate` from unmodified `evaluate_single_observation_set`

## Commands

From `tasks/voxel3d-Ayijiang`:

```bash
export QWEN_MODEL_PATH=/path/to/Qwen2.5-7B-Instruct
python scripts/audit_height_equivalence.py
python scripts/run_official_eval.py --methods official_prompt,cpsg_supportproj,cpsg_supportproj_retry \
  --n-obs 50 --seeds 7,17,27 --out-dir results/official_full
```

## Reported artifacts

| File | Content |
| --- | --- |
| `reported-results/official_baseline_vs_cpsg_supportproj.json` | Baseline vs CPSG+SupportProj |
| `reported-results/official_cpsg_supportproj_retry_v90.json` | Final retry method (V≈93.6%) |
| `reported-results/height_equivalence_audit.json` | Height ↔ Structure3D equivalence GO |

Re-runs may vary slightly with GPU / transformers versions; use the same model revision and seeds for closest match.

## Integrity notes

- Official HypoSpace validator / GT are not modified.
- SupportProj only zeros empty Observation columns.
- Range retry never silently invents occupied-column heights.
- BAE / orbit program expansion is **not** included in official V/U/R.
