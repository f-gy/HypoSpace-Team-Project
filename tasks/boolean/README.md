# Boolean tasks

Each team member’s Boolean method is maintained in its own subdirectory, with source code, configuration, results and reproduction instructions.

## Contributions

| Member | Method | Research role |
| --- | --- | --- |
| 徐晗笑 ([Xiao-ada](https://github.com/Xiao-ada)) | [BACE](Xiao-ada/bace/README.md) | Main method: budget-aware canonical exclusion during LLM decoding |
| 徐晗笑 ([Xiao-ada](https://github.com/Xiao-ada)) | [COVE](Xiao-ada/cove/README.md) | Supplementary approach: learned execution and fixed-catalog selection for final system outputs |

These two releases cover a controlled HypoSpace Boolean subset. The BACE comparison holds the trained checkpoint and generation budget fixed. In COVE, the executor-only control matches the coupled system, so the reported scores do not establish improved unrestricted LLM generation. Each release documents its own evaluation protocol, comparison results and limitations.

[Contribution overview](Xiao-ada/README.md) · [Reproducibility](Xiao-ada/REPRODUCIBILITY.md) · [Presentation materials and bilingual script](Xiao-ada/presentation/README.md)
