# HypoSpace Team Project

A four-member course project on hypothesis generation for the [HypoSpace benchmark](https://proceedings.mlr.press/v306/chen26bz.html), covering **Boolean expressions, causal graphs, and gravity-constrained 3D voxel reconstruction**.

Each member explored their own optimization direction and contributed the method they considered most effective and efficient in their experiments. This repository brings the four contributions together, with separate code, results, and reproduction instructions for each approach.

## Team and contributions

| Member | Focus | Selected approach | Start here |
| --- | --- | --- | --- |
| Xuhanxiao | Boolean hypothesis generation | **BACE**: budget-aware canonical exclusion during LLM decoding. **COVE** is included as a supplementary learned-executor approach. | [Contribution overview](tasks/boolean-Xuhanxiao/README.md) |
| Tianxuer | Causal graph inference | **Learned follow-up selection with continued repair**: learn which edge constraint to ask about next, then repair invalid follow-up graphs. | [Method and reproduction](tasks/casual-Tianxuer/README.md) |
| Ayijiang | Gravity-constrained 3D voxel reconstruction | **CPSG**: predict structured column heights, with support projection, range-feedback retry, and duplicate control. | [Package guide](tasks/voxel3d-Ayijiang/README.md) |
| Fengguanying | All three task types | **ReclaimHypo**: constraint-guided generation with mechanism deduplication and shared output-budget recycling across tasks. | [Method and results](tasks/ReclaimHypo-Fengguanying/README.md) |

## Repository structure

```text
tasks/
├── boolean-Xuhanxiao/
│   ├── bace/                     # Main Boolean method, adapters and evidence
│   ├── cove/                     # Supplementary learned-executor method
│   ├── presentation/             # Slides and presentation scripts
│   └── REPRODUCIBILITY.md
├── casual-Tianxuer/
│   ├── causal_followup.py        # Prepare data and train the action selector
│   ├── causal_repair.py          # Causal graph verification and repair
│   ├── run_causal_ppt_pair.py    # Paired comparison runner
│   └── reported-results/        # Recorded experiment summaries
├── voxel3d-Ayijiang/
│   ├── README.md                # Package contents and reported results guide
│   └── voxel3d_for_github_tasks.zip
└── ReclaimHypo-Fengguanying/
    ├── budget_v1/               # Unified scheduler, evaluation and tests
    ├── certified_generation_v1/ # Three constraint-based task adapters
    ├── final_results/           # Recorded outputs, scores and protocol
    ├── presentation/            # Presentation script
    ├── reproduce_cpu.sh         # One-command CPU reproduction
    └── report.md                # Detailed results and metric definitions
```

## Reading and reproduction guide

- **Boolean:** begin with the [BACE README](tasks/boolean-Xuhanxiao/bace/README.md), then consult the [shared reproduction guide](tasks/boolean-Xuhanxiao/REPRODUCIBILITY.md). The [presentation directory](tasks/boolean-Xuhanxiao/presentation/README.md) explains the main findings and the supplementary COVE experiment.
- **Causal graphs:** the [contribution README](tasks/casual-Tianxuer/README.md) describes the frozen model, learned selector, paired comparison, and GPU reproduction commands. Recorded summaries are in [reported-results](tasks/casual-Tianxuer/reported-results/).
- **3D voxels:** the [contribution directory](tasks/voxel3d-Ayijiang/) retains the member's original ZIP package unchanged. It contains source, configuration, upstream task code, reports, and reproduction instructions.
- **Cross-task optimization:** start with the [ReclaimHypo report](tasks/ReclaimHypo-Fengguanying/report.md). Run `bash reproduce_cpu.sh` from that contribution directory after installing its CPU requirements; the README gives the complete setup command.

Each contribution specifies its own model, data split, budget, validator, and evaluation protocol. Read its local documentation when interpreting results: the four methods were developed under different experimental settings, so their headline scores are not a single shared leaderboard. ReclaimHypo specifically studies a total output budget that can be reassigned across task types.

## Benchmark and upstream code

[HypoSpace paper](https://proceedings.mlr.press/v306/chen26bz.html) · [Official HypoSpace repository](https://github.com/CTT-Pavilion/_HypoSpace)

Upstream attribution and contribution-specific provenance are retained within each package.
