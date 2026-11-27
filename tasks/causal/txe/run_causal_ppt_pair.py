"""Evaluate the frozen final prompt policy and direct Qwen on the same budget."""

import collections
import json

import numpy as np

from causal_followup import ROOT, LocalQwen, State, choose, metrics, read, write
from causal_repair import feedback, repair_hint

SEED = 514271
BUDGET = 32
SOURCE = ROOT / "results" / "causal-repair-pilot"
DIRECTORY = ROOT / "results" / "causal-ppt-final"
ARMS = ("qwen_direct", "learned_followup_with_continued_repair")


def main():
    questions = read(SOURCE / "questions.json")
    selectors = read(ROOT / "results/causal-followup-pilot/selectors.json")["models"]
    model = LocalQwen(batch_size=24)
    model.torch.manual_seed(SEED)
    states = {question["id"]: [State(question, SEED + index) for _ in ARMS]
              for index, question in enumerate(questions)}
    for step in range(1, BUDGET + 1):
        jobs = []
        for pair in states.values():
            for arm, state in zip(ARMS, pair):
                last = state.records[-1] if state.records else None
                retry = False
                action = hint = None
                if arm != "qwen_direct":
                    retry = bool(last and not last["valid"] and last.get("action") and not last.get("retry"))
                    action = last["action"] if retry else choose(state, "factorized", selectors)
                    hint = repair_hint(state) if retry else (feedback(state) if not state.history else None)
                prompt = state.prompt(action)
                if hint:
                    prompt += "\n\nVerification feedback: " + hint
                jobs.append((state, action, retry, prompt))
        if step == 1:
            assert all(left[3] == right[3] for left, right in zip(jobs[::2], jobs[1::2]))
            outputs = [out for out in model.generate([job[3] for job in jobs[::2]]) for _ in range(2)]
        else:
            outputs = model.generate([job[3] for job in jobs])
        for (state, action, retry, _), output in zip(jobs, outputs, strict=True):
            state.advance(dict(state.record(action, output), retry=retry), action)
        if step % 4 == 0:
            print(f"Completed {step}/{BUDGET} generations per question and arm", flush=True)

    traces = {qid: {arm: state.records for arm, state in zip(ARMS, pair)}
              for qid, pair in states.items()}
    write(DIRECTORY / "rollouts.json", traces)
    answers = read(SOURCE / "answers.json")
    fields = ("validity", "uniqueness", "recovery", "distinct_valid", "input_tokens", "output_tokens")
    results = collections.defaultdict(dict)
    for qid, arms in traces.items():
        assert arms[ARMS[0]][0] == arms[ARMS[1]][0]
        gt = set(answers[qid])
        for arm, records in arms.items():
            assert len(records) == BUDGET
            assert all(not row["valid"] or row["graph_hash"] in gt for row in records)
            results[arm][qid] = metrics(records, gt, BUDGET)
    aggregate = {arm: {key: float(np.mean([row[key] for row in items.values()])) for key in fields}
                 for arm, items in results.items()}
    rng = np.random.default_rng(SEED)
    sample = rng.integers(0, len(questions), size=(10000, len(questions)))
    paired = {}
    for key in fields[:3]:
        differences = np.array([results[ARMS[1]][qid][key] - results[ARMS[0]][qid][key] for qid in traces])
        paired[key] = dict(
            delta=float(differences.mean()),
            bootstrap_95=np.quantile(differences[sample].mean(axis=1), [0.025, 0.975]).tolist(),
        )
    output = dict(
        settings=dict(n=len(questions), nodes=5, generation_budget=BUDGET, seed=SEED,
                      model="Qwen3-4B-Instruct-2507", common_first_generation=True,
                      temperature=0.7, max_new_tokens=64, gpu=model.torch.cuda.get_device_name(0),
                      prior_pilot_questions_reused=True),
        aggregate=aggregate, paired_improvement=paired, per_question=results,
    )
    write(DIRECTORY / "metrics.json", output)
    print(json.dumps(dict(aggregate=aggregate, paired_improvement=paired), indent=2), flush=True)


if __name__ == "__main__":
    main()
