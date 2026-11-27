# Boolean hypothesis coverage experiments

Contributor: Xiao-ada.

This contribution studies two approaches on a controlled HypoSpace Boolean subset. The methods share a pinned Qwen2.5-1.5B-Instruct four-bit base model and the same 16/6/26 training, validation and test observation-task split.

| Release | Research role | Reported conclusion |
|---|---|---|
| [BACE](bace/README.md) | Main presentation method: budget-aware structural exclusion during LLM decoding | Recovery improves from 41.02% to 50.34% at the same checkpoint and 16-attempt budget, with lower validity |
| [COVE](cove/README.md) | Supplementary learned-executor and catalog-selection system | Adaptive-budget validity/recovery reach 99.27%, but the executor-only control matches the coupled system |

Both releases include source, configuration, released adapters, complete result evidence and CPU verification commands. Each directory is self-contained for evidence verification. Runtime environments and base-model downloads remain local under the ignored runtime directory.

The methods answer different questions. BACE tests the effect of structural exclusion while retaining LLM proposal probabilities. COVE tests learned observation consistency over a finite catalog. Its scores describe final system outputs rather than demonstrated gains in unrestricted LLM generation.

The original test observation tasks had already appeared in earlier experiments. Inference seeds do not represent independent model retraining. Absolute results are specific to this small Boolean setting and the disclosed common scoring gate, and are not paper-leaderboard reproductions.

[Presentation materials](presentation/README.md) contain the two BACE slides, the two-sentence COVE supplement, the English script and sentence-aligned Chinese correspondence. [Reproducibility](REPRODUCIBILITY.md) records the environment and verification workflow.

Experimental logs preserve original benchmark prompts and model outputs. Second-person language inside those immutable task inputs is part of the benchmark interface, not project narration.
