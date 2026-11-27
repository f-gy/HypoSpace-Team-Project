# Reproducibility

## CPU evidence verification

Each release uses Python 3.12 with requirements-lock.txt. The verification scripts reconstruct observations from pinned HypoSpace rules, check source/adapter hashes, re-score saved outputs and compare aggregate metrics. They require neither model inference nor API credentials.

The BACE workflow, starting in bace, is:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python scripts/verify_results.py
.venv/bin/python scripts/verify_optimization.py
```

The COVE workflow, starting in cove, is:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python scripts/verify_all80.py
```

The source, artifact and experiment manifests remain in each release. SOURCE_PROVENANCE.json records the source commit, original file hashes before publication edits, and the exact edited-document hashes. Publication edits affect documentation, attribution wording and the adapter README hash entry, not model tensors, experimental prompts, outputs or scores.

## Apple Silicon model inference

The recorded environment is Apple M5 with 24 GB shared memory, MLX 0.32.4 and MLX LM 0.32.2. requirements-mlx-lock.txt pins the complete environment. Both packages resolve the model under ../runtime/models, allowing one local base-model copy.

The common setup from either release directory is:

```bash
python3.12 -m venv ../runtime/mlx-env
../runtime/mlx-env/bin/python -m pip install -r requirements-mlx-lock.txt
../runtime/mlx-env/bin/python -m explorer fetch
../runtime/mlx-env/bin/python -m explorer prepare
HF_HOME=../runtime/hf-cache ../runtime/mlx-env/bin/python scripts/local_mlx.py --download-only
../runtime/mlx-env/bin/python scripts/restore_preference_adapters.py
```

The model download pins the exact public revision and verifies its SHA256. BACE inference uses the released continuation adapters:

```bash
../runtime/mlx-env/bin/python -m explorer.preference_experiment validation
../runtime/mlx-env/bin/python -m explorer.preference_experiment test
```

COVE inference uses the released executor weights and frozen coverage-continuation adapter. The recorded candidate scores support a CPU ranking/metric replay:

```bash
.venv/bin/python -m explorer.all80_experiment validation
.venv/bin/python -m explorer.all80_experiment test
```

These replay commands deterministically rewrite the matching ranking/evaluation files. Recomputing candidate scores instead uses:

```bash
../runtime/mlx-env/bin/python -m explorer.catalog_scores validation
../runtime/mlx-env/bin/python -m explorer.catalog_scores test
```
 Complete executor retraining uses explorer.learned_executor and requires a separate empty output directory or preserved copy of the published results, because completed training evidence is protected from overwrite.

The task split, exact answer aliases, EOS handling, temperature, seeds and budgets are part of the protocol. Different hardware kernels or package versions can change stochastic outputs. Compressed evidence supports exact re-scoring independently of regenerated model outputs.

## Publication checks

The curated packages passed CPU re-scoring and unit tests before publication. BACE ran 32 tests and COVE ran 35; ten MLX-only tests in each suite were skipped in the CPU environment. No new LLM inference or retraining was performed during this packaging check. The recorded experimental outputs remain byte-identical to their source commits. PUBLICATION_CHECKS.json summarizes the checks.
