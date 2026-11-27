# BACE for Boolean hypothesis coverage

BACE, Budget-Aware Canonical Exclusion, is an inference extension to a coverage-adapted language model. It preserves ordinary sampling while a target number of distinct structural classes remains reachable. At zero slack, a token-level grammar mask excludes previously observed classes. The LLM retains its learned probabilities over the remaining token support.

## Matched test results

The same trained checkpoint is evaluated on 26 Boolean observation tasks, three inference seeds and 16 counted attempts per task. All failed, duplicate and invalid attempts remain in the budget.

| Method | Validity | Uniqueness | Recovery |
|---|---:|---:|---:|
| Same LLM, ordinary sampling | 39.90% | 64.42% | 41.02% |
| Same LLM with BACE | 36.38% | 81.49% | 50.34% |
| Uniform class proposals with BACE | 28.93% | 83.89% | 40.03% |

The matched decoder comparison yields +9.32 recovery percentage points, with a descriptive task-paired bootstrap 95% interval [5.90,12.82]. Validity falls by 3.53 points. LLM+BACE exceeds the uniform BACE control by 10.31 recovery points.

This supports a coverage/validity trade-off from structured decoding in the recorded setting. The uniqueness target is enforced by the decoder, not learned diversity. The comparison does not establish an additional recovery benefit from the coverage training loss over matched SFT.

## Algorithm

The counted budget N is 16 and the distinct-class target M is 13. S_t contains distinct space-admissible canonical keys from completed attempts, including observation-invalid structures. The slack is:

```text
D_t = (N - t) - max(0, M - |S_t|)
```

Positive slack retains ordinary sampling. Zero slack with an unmet target activates the exclusion mask. The represented grammar has 300 spellings across 33 structural classes. Observation compatibility is not checked during proposal, and no rejected answer receives a free replacement.

The target implies at least 13/16 structural uniqueness when support is available and generation completes normally. It does not guarantee validity or recovery. The controller changes the sampling process and does not contradict IID coverage bounds.

## Learned component and study boundary

The proposal is the frozen four-bit Qwen2.5-1.5B-Instruct base with a 311,296-parameter LoRA adapter. The selected coverage-continuation checkpoint adds 80 updates to the original canonical-coverage adapter. The matched IID/BACE comparison holds this checkpoint fixed. A separate preference-loss arm remains a complete, unselected control.

The study uses x/y, AND/OR/NOT and depth two, with training/validation/test observation-task counts 16/6/26. The test observations had already appeared in earlier experiments. Three inference seeds are not three independent training runs. Grammar constraints and exclusion are established ideas; the evaluated design contribution is budget-aware intervention over benchmark structural classes.

## Evidence and execution

- [Test summary](results/canonical_preference_v5/test/summary.json), [per-episode results](results/canonical_preference_v5/test/episodes.csv), [complete compressed evidence](results/canonical_preference_v5/test/evidence.jsonl.gz).
- [Frozen evaluation protocol](results/canonical_preference_v5/evaluation_protocol.json), [validation selection](results/canonical_preference_v5/selection.json), [training summary](results/canonical_preference_v5/training_summary.json).
- [Decoder](explorer/target_decode.py), [grammar masking](explorer/canonical_decode.py), [matched evaluation](explorer/preference_experiment.py).
- [Original training objective](explorer/coverage_lora.py), [continuation training](explorer/canonical_preference.py).
- [Reproducibility and CPU verification](../REPRODUCIBILITY.md).

Earlier constrained/contrastive pilots and unsuccessful preference controls remain included. They provide context for the selected system rather than additional independent confirmations. Raw evidence retains its original content.

References: [HypoSpace](https://proceedings.mlr.press/v306/chen26bz.html), [PICARD](https://aclanthology.org/2021.emnlp-main.779/), [LoRA](https://arxiv.org/abs/2106.09685).
