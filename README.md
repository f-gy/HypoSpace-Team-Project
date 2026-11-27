# HypoSpace Team Project

A four-person course project investigating hypothesis generation on the [HypoSpace benchmark](https://proceedings.mlr.press/v306/chen26bz.html).

## Team

| Member | GitHub | Contributions |
| --- | --- | --- |
| Member 1 | TBD | TBD |
| 徐晗笑 | [Xiao-ada](https://github.com/Xiao-ada) | Boolean: [BACE](tasks/boolean/Xiao-ada/bace/README.md) and [COVE](tasks/boolean/Xiao-ada/cove/README.md) |
| Member 3 | TBD | TBD |
| Member 4 | TBD | TBD |

## Planned structure

- `tasks/boolean/` — Boolean genetic interactions
- `tasks/causal/` — causal graph inference
- `tasks/voxel3d/` — 3D voxel reconstruction
- `common/` — shared interfaces, validators and metrics
- `experiments/`, `results/`, `docs/`, `tests/` — reproducible work

The [Boolean contribution from 徐晗笑 (Xiao-ada)](tasks/boolean/Xiao-ada/README.md) includes source code, configurations, result evidence, reproduction instructions and [presentation materials](tasks/boolean/Xiao-ada/presentation/README.md). BACE studies budget-aware structural exclusion during LLM decoding. COVE is a supplementary learned-executor and catalog-selection approach; its system-level results do not establish improved unrestricted LLM generation.

## Reproducibility

[ReclaimHypo: Constraint-Guided Budget Recycling](tasks/ReclaimHypo-Constraint-Guided-Budget-Recycling/README.md) provides a unified constraint-backed scheduler for causal, voxel and Boolean tasks. Its internal experiment recycles output slots across tasks; results are system-level improvements under a shared-budget protocol, not an official fixed-per-question score or evidence of improved LLM weights.

Document model versions, data splits, seeds, prompts, verification rules, budgets and commands. Distinguish development results from final tests.

## Upstream

[HypoSpace paper](https://proceedings.mlr.press/v306/chen26bz.html) · [Official code](https://github.com/CTT-Pavilion/_HypoSpace)
