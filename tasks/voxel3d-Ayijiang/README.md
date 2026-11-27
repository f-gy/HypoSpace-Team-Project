# CPSG for HypoSpace 3D Voxel Reconstruction

Constraint-Preserving Structured Generation (CPSG) for the HypoSpace **3D voxel** task.

Instead of free-form Layer text, the model predicts per-column pillar heights on Observation-occupied cells; a program converts heights to official `Structure3D`. Optional **SupportProj**, one **range-feedback retry** (no silent height fill), and **anti-dup** protect Validity / Uniqueness on the official evaluator.

## Official metrics (main result)

Same 50 observations, seeds `7/17/27`, `n_queries=3`, local Qwen2.5-7B-Instruct. Scores from unmodified `Benchmark3D.evaluate_single_observation_set`.

| Method | Validity | Uniqueness | Recovery |
| --- | ---: | ---: | ---: |
| Official Layer prompt | 0.7% | 95.8% | 0.07% |
| CPSG + SupportProj | 80.7% | 35.6% | 8.5% |
| **CPSG + SupportProj + RangeRetry + AntiDup** | **93.6%** | **76.9%** | **19.7%** |

JSON summaries: [`reported-results/`](reported-results/). Write-up: [`docs/final_project_summary.md`](docs/final_project_summary.md).

## Layout

```
voxel3d-Ayijiang/         # Ayijiang contribution (GitHub: 1741493419ayj-oss)
├── upstream/3d/          # vendored official HypoSpace 3D code + dataset
├── src/methods/cpsg.py   # height prompts, parse, SupportProj, decode
├── scripts/
│   ├── run_official_eval.py
│   └── audit_height_equivalence.py
├── configs/obs_pool_50.json
├── reported-results/
├── docs/
└── requirements.txt
```

Upstream origin: [CTT-Pavilion/_HypoSpace](https://github.com/CTT-Pavilion/_HypoSpace) (3D domain only; MIT).

## Quick start (GPU)

```bash
cd tasks/voxel3d-Ayijiang
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # match your CUDA
pip install -r requirements.txt

# Point to local Qwen2.5-7B-Instruct (or set MODEL_CACHE_ROOT)
export QWEN_MODEL_PATH=/path/to/Qwen2.5-7B-Instruct

# Representation equivalence audit (CPU OK)
python scripts/audit_height_equivalence.py

# Pilot official eval
python scripts/run_official_eval.py --pilot --methods cpsg_supportproj_retry \
  --out-dir results/pilot

# Full official eval (50 obs × 3 seeds)
python scripts/run_official_eval.py --methods cpsg_supportproj_retry \
  --n-obs 50 --n-queries 3 --seeds 7,17,27 \
  --out-dir results/official_v90
```

Methods: `official_prompt`, `cpsg_supportproj`, `cpsg_supportproj_retry`.

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).
