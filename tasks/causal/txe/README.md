# Learning Which Causal-Graph Follow-Up to Ask

This project uses a frozen Qwen3-4B-Instruct-2507 model to generate causal graphs for [HypoSpace](https://proceedings.mlr.press/v306/chen26bz.html). A deterministic verifier identifies conflicts with the observations. A lightweight selector learns which edge to ask Qwen to include or exclude next. It ranks actions by the estimated probability of getting a valid graph multiplied by the estimated probability that a valid graph is new. The final variant also uses the next generation to repair an invalid follow-up answer.

## Reported results

Both systems used the same 27 five-node questions, the same first output for each question, and 32 generation attempts per question. The final comparison used a single seed on questions already used during development.

| Method | Validity | Uniqueness | Recovery |
| --- | ---: | ---: | ---: |
| Qwen direct generation | 1.50% | 7.29% | 3.88% |
| Continued repair + learned follow-up selection | 26.74% | 30.56% | 13.65% |

[Final paired results](reported-results/causal-ppt-final-metrics.json) · [Earlier multi-arm comparison](reported-results/causal-repair-pilot-metrics.json)

## Reproduce the experiment

Use Python 3.11 and a CUDA-capable GPU. Run these commands from this directory. The weights, generated questions, answer files, and trained selector coefficients are created locally.

```bash
git clone https://github.com/CTT-Pavilion/_HypoSpace.git official
git -C official checkout c69e9318577b34b5b896996571aefd4ba6053f58
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python download_model.py
```

The downloader uses the public Qwen release on [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-4B-Instruct-2507/summary), pins the exact revision, and checks file hashes. The model downloads once. Inference runs locally.

```bash
export PYTHONHASHSEED=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

python official/causal/generate_causal_dataset.py \
  --nodes 4 --n-observations 3 --seed 33550336 \
  --output data/causal-n4-plan-a.json
python causal_followup.py prepare --run-dir results/causal-followup-pilot
python causal_followup.py train --run-dir results/causal-followup-pilot --batch-size 24

python official/causal/generate_causal_dataset_for_large.py \
  --nodes 5 --n-observations 3 --fixed --max-edges 5 \
  --n-observation-sets 140 --seed 830431 --output data/causal-n5-repair.json
python causal_repair.py --prepare-only --run-dir results/causal-repair-pilot
python run_causal_ppt_pair.py
```

Training uses four-node questions. The five-node evaluation takes one representative per observation pattern after accounting for node renaming. Complete answer sets are used for scoring after generation. The runner writes paired metrics and full trajectories to `results/causal-ppt-final/`. Retraining and generation can vary across software versions or GPU setups.
