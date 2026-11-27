# Source and publication provenance

This directory publishes the deterministic shared-budget version under the name **ReclaimHypo: Constraint-Guided Budget Recycling**. Earlier experiment paths and raw artifacts retain the name OrbitHypo. It does not include or claim results from later witness-distillation experiments.

- Local source branch: `research/shared-budget`.
- Source HEAD: `df8640f` (full hash recorded in `publication_manifest.json`).
- `original_source_manifest.json` is the original package manifest, retained as historical evidence. Its README hash is not the hash of this renamed publication README.
- `publication_manifest.json` hashes the files actually included in this publication, excluding itself and the documented transient directories.
- `final_results/` preserves the original recorded experiment. `reproduction_publication_check/` contains a fresh run from this publication directory; it is intentionally included as publication verification evidence.

`vendor/HypoSpace/` contains upstream HypoSpace task-generation code, originally sourced from [CTT-Pavilion/_HypoSpace](https://github.com/CTT-Pavilion/_HypoSpace). The benchmark is described in the [HypoSpace paper](https://proceedings.mlr.press/v306/chen26bz.html). Existing attribution in those files is retained. No new license is asserted over upstream code; users should check upstream licensing before redistribution or reuse beyond this research submission.

The `code/` directory includes historical experiment helpers. The final CPU reproduction entry point is `reproduce_cpu.sh`, and its dependencies are `requirements-cpu.txt`; historical training helpers are not required for the reported final results.

Metric comparisons use the same constraint backend and the same total output-slot cap. They do not control for equal CPU time or equal LLM tokens. No LLM is loaded by the final CPU experiment. These are small, previously used internal cases, not a new blind evaluation of the complete benchmark.
