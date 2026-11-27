# COVE learned execution and catalog selection

COVE combines frozen LLM scores with a small learned executor over a fixed, training-derived Boolean hypothesis catalog. Its research target is final system-level valid coverage. It does not demonstrate that unrestricted LLM generation improves to the reported system scores.

## Results under two separate protocols

The same 26 test observation tasks and three inference seeds are retained. The adaptive protocol uses N=K, where K is the full reference-set size, following the paper's common budget. The fixed protocol uses N=16. Results from these protocols remain separate.

| Method | N=K validity | uniqueness | recovery | N=16 validity | uniqueness | recovery |
|---|---:|---:|---:|---:|---:|---:|
| Frozen adapted LLM, IID | 38.48% | 76.62% | 26.22% | 39.90% | 64.42% | 41.02% |
| LLM catalog, without replacement | 37.07% | 100.00% | 37.07% | 34.86% | 100.00% | 60.13% |
| COVE plus LLM | 99.27% | 100.00% | 99.27% | 55.53% | 100.00% | 98.28% |
| Executor only, no LLM score | 99.27% | 100.00% | 99.27% | 55.53% | 100.00% | 98.28% |
| Exact symbolic control | 99.27% | 100.00% | 99.27% | 55.53% | 100.00% | 98.28% |

The executor materially improves valid selection over the same LLM-scored catalog without learned consistency. However, the executor-only control matches the coupled system. No incremental final-metric contribution from the LLM factor is established. The exact symbolic control also matches it, limiting the necessity of the learned/LLM combination in this small formal domain.

## Method

The static catalog contains 33 structural classes and two fixed answer aliases per class, collected from the original training targets. A grammar-only procedure supplies expression trees. It does not retrieve each test task's compatible reference members.

Three shared operator networks approximate AND, OR and NOT. Their compositional architecture has 179 parameters. Training uses only the original 16 training tasks, with 260 observed-output labels and 528 training task/class compatibility labels. The primary initialization is 4306, with 4307/4308 retained as training replications.

The frozen coverage-continuation LLM scores all 66 represented complete answers, including EOS. Class mass sums the two alias probabilities. The learned observation-consistency term then reweights each class:

```text
log_weight(h) = log_q_LLM(h | O) + 10 * learned_log_match(h, O)
```

Gumbel top-N sampling selects classes without replacement. This is energy reweighting, not an exact Bayesian posterior. Proposals use learned execution and the given observations, not an exact solver or test solution-membership labels. Exact execution appears only in scoring diagnostics and the explicit symbolic control.

## Interpretation and limitations

The final output comes from a fixed catalog rather than free autoregressive answer generation. Uniqueness of 100% follows from sampling without replacement. At N=K, with all valid hits in the reference set, validity and recovery are equal, so the headline columns are not independent ability gains.

The full reference universe has 34 classes; a problematic remaining class lacks a validator-compatible training alias. The complete reference denominator is retained. Each task adds 66 teacher-forced answer scores in nine batches with shared prompt prefixes. Output budgets match within each protocol, but total computation is not matched to free generation.

The underlying LLM checkpoint is unchanged. This experiment demonstrates a learned executor's contribution to final system output, while unrestricted LLM generation gains and the LLM's incremental value remain unproven. Neural modules and Gumbel sampling are established techniques, and cross-domain or open-ended improvements are untested.

## Evidence and execution

- [All test controls](results/all80/test/summary.json), [per-episode results](results/all80/test/episodes.json), [complete compressed evidence](results/all80/test/evidence.jsonl.gz).
- [Frozen training protocol](results/all80/protocol.json), [evaluation protocol](results/all80/evaluation_protocol_test.json), [cost ledger](results/all80/costs.json).
- [Executor weights and training logs](results/all80/executors/), [catalog](results/all80/bank.json), [actual LLM scores](results/all80/lm_scores/).
- [Learned executor](explorer/learned_executor.py), [proposal and experiment](explorer/all80_experiment.py), [CPU verification](scripts/verify_all80.py).
- [Reproducibility](../REPRODUCIBILITY.md).

The package retains the original BACE/v1 baseline context because the same frozen checkpoint and recorded IID prefixes supply the catalog controls. These records are dependencies, not new COVE experiments. Earlier unfiltered executor probes are separate from the filtered composition probe and do not establish unseen-family LLM capability.

References: [HypoSpace](https://proceedings.mlr.press/v306/chen26bz.html), [Neural Module Networks](https://openaccess.thecvf.com/content_cvpr_2016/html/Andreas_Neural_Module_Networks_CVPR_2016_paper.html), [Gumbel-Top-k](https://proceedings.mlr.press/v97/kool19a.html).
