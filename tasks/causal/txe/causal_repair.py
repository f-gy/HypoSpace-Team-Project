"""Test whether observation-conflict feedback helps start a causal-graph search."""

import argparse
import copy
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from causal_followup import (
    ROOT, SEED, CausalGraph, LocalQwen, State, choose, metrics, read, shape, write,
)

POLICIES = {
    "direct": ("direct", False),
    "repair_direct": ("direct", True),
    "rare_edge": ("rare_edge", False),
    "repair_rare_edge": ("rare_edge", True),
    "factorized": ("factorized", False),
    "repair_factorized": ("factorized", True),
    "repair_joint": ("joint", True),
}
CONTRASTS = (
    ("repair_direct", "direct"),
    ("repair_rare_edge", "rare_edge"),
    ("repair_factorized", "factorized"),
    ("repair_factorized", "repair_rare_edge"),
    ("repair_factorized", "repair_joint"),
)
DATASET = ROOT / "data/causal-n5-repair.json"
SELECTORS = ROOT / "results/causal-followup-pilot/selectors.json"


def prepare(directory):
    data = read(DATASET)
    rng = random.Random(830431)
    groups = defaultdict(list)
    for row in data["datasets"]:
        groups[shape(row, row["nodes"])].append(row)
    representatives = [rng.choice(group) for _, group in sorted(groups.items())]
    assert len(representatives) > 20
    questions, answers = [], {}
    for row in representatives:
        qid = row["observation_set_id"]
        public = {key: row[key] for key in ("observations", "nodes", "max_edges")}
        questions.append(dict(public, id=qid))
        answers[qid] = [CausalGraph.from_dict(graph).get_hash() for graph in row["ground_truth_graphs"]]
    write(directory / "questions.json", questions)
    write(directory / "answers.json", answers)
    print(f"Prepared {len(questions)} new five-node questions", flush=True)


def feedback(state):
    if state.history or not state.records:
        return None
    prior = state.records[-1]["answer"]
    graph = state.benchmark.parse_llm_response(prior)
    if graph is None:
        return "Your previous response was not a valid DAG in the required format. Use the listed nodes and output one line within the edge limit."
    # Work solely with the proposed graph and original observations.
    import networkx as nx

    network = graph.to_networkx()
    for observation in state.question["observations"]:
        source = observation["perturbed_node"]
        reachable = nx.descendants(network, source)
        for target, expected in sorted(observation["effects"].items()):
            actual = int(target in reachable)
            if actual != expected:
                instruction = "ensure there is a directed path" if expected else "ensure there is no directed path"
                return (
                    f"Previous attempt: {graph.to_string()}. When {source} is perturbed, the graph predicts "
                    f"{target}={actual}, but the observation says {target}={expected}. "
                    f"Revise the graph to {instruction} from {source} to {target}. "
                    "Check all original observations before returning the revised DAG."
                )
    raise AssertionError("A graph that matches all observations should have entered history")


def repair_hint(state):
    """Describe conflicts in the most recent graph using only observed effects."""
    if not state.records or state.records[-1]["valid"]:
        return None
    if not state.history:
        return feedback(state)
    graph = state.benchmark.parse_llm_response(state.records[-1]["answer"])
    if graph is None:
        return "The previous answer did not parse as a DAG. Return exactly one Graph: line with the listed nodes and edge limit."
    import networkx as nx

    network = graph.to_networkx()
    conflicts = []
    for observation in state.question["observations"]:
        node = observation["perturbed_node"]
        actual = nx.descendants(network, node)
        for other, expected in sorted(observation["effects"].items()):
            if int(other in actual) != expected:
                requirement = "must have a directed path" if expected else "must have no directed path"
                conflicts.append(f"{node} {requirement} to {other}")
    if not conflicts:
        return None
    edges = ", ".join(f"{u}->{v}" for u, v in sorted(graph.edges)) or "No edges"
    return (f"Previous attempt: Graph: {edges}. Correct these observation conflicts: "
            + "; ".join(conflicts[:2]) + ". Preserve the other observations and the edge requirement when given.")


def advance(model, jobs):
    prompts = []
    for state, action, hint in jobs:
        prompt = state.prompt(action)
        if hint is not None:
            prompt += "\n\nVerification feedback: " + hint
        prompts.append(prompt)
    for (state, action, hint), output in zip(jobs, model.generate(prompts), strict=True):
        state.advance(dict(state.record(action, output), repair_feedback=hint), action)


def run(model, questions, selectors, budget):
    prefix = [State(q, SEED + index) for index, q in enumerate(questions)]
    advance(model, [(state, None, None) for state in prefix])
    states = {(state.question["id"], policy): copy.deepcopy(state) for state in prefix for policy in POLICIES}
    for step in range(2, budget + 1):
        order = list(states)
        random.Random(SEED + step).shuffle(order)
        jobs = []
        for key in order:
            state = states[key]
            selector, repair = POLICIES[key[1]]
            jobs.append((state, choose(state, selector, selectors), feedback(state) if repair else None))
        advance(model, jobs)
        if step % 4 == 0 or step == budget:
            print(f"Evaluation step {step}/{budget}: {len(states)} question/policy pairs", flush=True)
    return [{"id": key[0], "policy": key[1], "rounds": state.records} for key, state in states.items()]


def score(traces, answers, budget):
    paired = defaultdict(dict)
    for trace in traces:
        rows = trace["rounds"]
        assert len(rows) == budget
        qid = trace["id"]
        assert all(not r["valid"] or r["graph_hash"] in answers[qid] for r in rows)
        result = metrics(rows, set(answers[qid]), budget)
        first = next((index for index, row in enumerate(rows, 1) if row["valid"]), None)
        result.update(
            first_valid_step=first, any_valid=int(first is not None),
            first_valid_by_8=int(first is not None and first <= 8),
            repair_rounds=sum(row["repair_feedback"] is not None for row in rows),
            total_tokens=result["input_tokens"] + result["output_tokens"],
        )
        paired[trace["policy"]][qid] = result
    aggregate = {}
    for policy, questions in paired.items():
        rows = list(questions.values())
        aggregate[policy] = {key: float(np.mean([r[key] for r in rows])) for key in rows[0]
                             if key != "first_valid_step"}
        attempts = sum(r["followups"] for r in rows)
        aggregate[policy]["compliance"] = sum(r["compliant"] for r in rows) / attempts if attempts else None
        aggregate[policy]["questions_with_valid"] = sum(r["any_valid"] for r in rows)
    comparisons = {}
    for candidate, control in CONTRASTS:
        rng = np.random.default_rng(SEED)
        qids = sorted(paired[candidate])
        indices = rng.integers(len(qids), size=(10000, len(qids)))
        difference = {}
        for key in ("any_valid", "first_valid_by_8", "recovery", "validity", "distinct_valid", "total_tokens"):
            delta = np.asarray([paired[candidate][qid][key] - paired[control][qid][key] for qid in qids])
            draws = delta[indices].mean(axis=1)
            difference[key] = dict(
                delta=float(delta.mean()), interval_95=np.quantile(draws, [0.025, 0.975]).tolist(),
                wins=int((delta > 0).sum()), ties=int((delta == 0).sum()), losses=int((delta < 0).sum()),
            )
        comparisons[f"{candidate} - {control}"] = difference
    return dict(aggregate=aggregate, comparisons=comparisons, per_question=paired)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    directory = args.run_dir
    prepare(directory)
    if args.prepare_only:
        return
    questions = read(directory / "questions.json")
    model = LocalQwen(args.batch_size)
    traces = run(model, questions, read(SELECTORS)["models"], args.budget)
    write(directory / "rollouts.json", traces)
    results = score(traces, read(directory / "answers.json"), args.budget)
    results["settings"] = dict(
        n_test=len(questions), budget=args.budget, common_warmup=1, selectors_trained_on_nodes=4,
        test_nodes=5, max_edges=5, dataset_seed=830431, generation_seed=SEED,
        model="Qwen3-4B-Instruct-2507", precision="BF16",
        gpu=model.torch.cuda.get_device_name(0),
        peak_allocated_gib=model.torch.cuda.max_memory_allocated() / 1024**3,
    )
    write(directory / "metrics.json", results)
    print(json.dumps({"settings": results["settings"], "aggregate": results["aggregate"]}, indent=2))
    for arms, values in results["comparisons"].items():
        print(arms, "recovery", values["recovery"], "any_valid", values["any_valid"])


if __name__ == "__main__":
    main()
