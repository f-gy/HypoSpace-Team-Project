"""Train an edge-follow-up selector and compare five policies on HypoSpace."""

import argparse
import copy
import itertools
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "official" / "causal"))
from run_causal_benchmark import CausalBenchmarkEnhanced, CausalGraph

SEED = 33550336
ARMS = ("direct", "random", "rare_edge", "joint", "factorized")
TRAIN_STEPS = (4, 8, 12)
WARMUP = 4
FEATURES = (
    "include", "desired_frequency", "history_size", "observations",
    "source_observed", "target_observed", "observed_source_reaches_target",
    "reverse_frequency", "mean_edges", "alternative_path_frequency",
    "prior_attempts", "prior_valid_fraction", "prior_compliance_fraction",
)


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def shape(question, nodes):
    observations = question["observations"]
    return min(tuple(sorted(
        (rename[o["perturbed_node"]], tuple(sorted(rename[n] for n, effect in o["effects"].items() if effect)))
        for o in observations
    )) for permutation in itertools.permutations(nodes) for rename in [dict(zip(nodes, permutation))])


def prepare(source, run_dir):
    data = read(source)["datasets_by_n_observations"]
    rng = random.Random(SEED)
    splits = {"train": [], "test": []}
    answers = {}
    for count, train_count, test_count in [(1, 3, 1), (2, 8, 4), (3, 13, 7)]:
        groups = defaultdict(list)
        for row in data[str(count)]:
            groups[shape(row, row["nodes"])].append(row)
        representatives = [rng.choice(rows) for _, rows in sorted(groups.items())]
        rng.shuffle(representatives)
        assert len(representatives) >= train_count + test_count
        for index, row in enumerate(representatives[:train_count + test_count]):
            split = "train" if index < train_count else "test"
            qid = row["observation_set_id"]
            public = {key: row[key] for key in ("observations", "nodes", "max_edges")}
            splits[split].append(dict(public, id=qid))
            if split == "test":
                answers[qid] = [CausalGraph.from_dict(graph).get_hash() for graph in row["ground_truth_graphs"]]
    assert not {shape(q, q["nodes"]) for q in splits["train"]} & {
        shape(q, q["nodes"]) for q in splits["test"]
    }
    write(run_dir / "questions.json", splits)
    write(run_dir / "test-answers.json", answers)


class State:
    def __init__(self, question, rng_seed):
        self.question = question
        self.benchmark = CausalBenchmarkEnhanced()
        self.benchmark.nodes = question["nodes"]
        self.benchmark.max_edges = question["max_edges"]
        self.history = {}
        self.attempts = defaultdict(list)
        self.records = []
        self.rng = random.Random(rng_seed)

    def choices(self):
        if not self.history:
            return []
        observed = {o["perturbed_node"]: o["effects"] for o in self.question["observations"]}
        choices = []
        for u, v in itertools.permutations(self.question["nodes"], 2):
            for present in (False, True):
                # Reachability supplies necessary conditions, without enumerating valid DAGs.
                impossible = any((p == u or e[u]) and (p == v or not e[v]) for p, e in observed.items())
                forced = u in observed and {n for n, effect in observed[u].items() if effect} == {v}
                if (present and impossible) or (not present and forced):
                    continue
                choices.append((u, v, present))
        return choices

    def features(self, action):
        u, v, present = action
        graphs = list(self.history.values())
        observed = {o["perturbed_node"]: o["effects"] for o in self.question["observations"]}
        attempts = self.attempts.get(action, [])
        alternate = sum(any((u, w) in g.edges and (w, v) in g.edges
                            for w in self.question["nodes"] if w not in (u, v)) for g in graphs)
        return [
            int(present), np.mean([((u, v) in g.edges) == present for g in graphs]),
            len(graphs), len(observed), int(u in observed), int(v in observed),
            observed.get(u, {}).get(v, 0), np.mean([(v, u) in g.edges for g in graphs]),
            np.mean([len(g.edges) for g in graphs]), alternate / len(graphs),
            len(attempts), (sum(r["valid"] for r in attempts) + 0.5) / (len(attempts) + 1),
            (sum(r["compliant"] for r in attempts) + 0.5) / (len(attempts) + 1),
        ]

    def prompt(self, action):
        prompt = self.benchmark.create_prompt(self.question["observations"], list(self.history.values()))
        if action is not None:
            u, v, present = action
            requirement = "include" if present else "exclude"
            prompt += f"\n\nFollow-up: Try to {requirement} the direct edge {u}->{v}. Still match every original observation."
        return prompt

    def record(self, action, output):
        answer, input_tokens, output_tokens = output
        graph = self.benchmark.parse_llm_response(answer)
        graph_hash = graph.get_hash() if graph is not None else None
        valid = graph is not None and self.benchmark.validate_hypothesis(graph, self.question["observations"])
        novel = valid and graph_hash not in self.history
        compliant = None
        if action is not None:
            u, v, present = action
            compliant = graph is not None and (((u, v) in graph.edges) == present)
        return dict(action=action, answer=answer, graph_hash=graph_hash, valid=valid,
                    novel=novel, compliant=compliant, input_tokens=input_tokens, output_tokens=output_tokens)

    def advance(self, record, action):
        self.records.append(record)
        if action is not None:
            self.attempts[action].append(record)
        if record["novel"]:
            self.history[record["graph_hash"]] = self.benchmark.parse_llm_response(record["answer"])


class LocalQwen:
    def __init__(self, batch_size):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.batch_size = batch_size
        self.path = ROOT / "models" / "Qwen3-4B-Instruct-2507"
        torch.manual_seed(SEED)
        torch.cuda.reset_peak_memory_stats()
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.path, local_files_only=True, torch_dtype=torch.bfloat16, device_map={"": 0},
        ).eval()

    def generate(self, prompts):
        records = []
        tokenizer = self.tokenizer
        eos = self.model.generation_config.eos_token_id
        eos = {eos} if isinstance(eos, int) else set(eos)
        for start in range(0, len(prompts), self.batch_size):
            text = [tokenizer.apply_chat_template(
                [{"role": "system", "content": "You are an expert in causal inference and graph theory."},
                 {"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            ) for prompt in prompts[start:start + self.batch_size]]
            inputs = tokenizer(text, padding=True, add_special_tokens=False, return_tensors="pt").to("cuda:0")
            with self.torch.inference_mode():
                outputs = self.model.generate(
                    **inputs, do_sample=True, temperature=0.7, max_new_tokens=64, pad_token_id=tokenizer.pad_token_id,
                )[:, inputs.input_ids.shape[1]:]
            for token_ids, input_length in zip(outputs.tolist(), inputs.attention_mask.sum(1).tolist()):
                generated = next((i + 1 for i, token in enumerate(token_ids) if token in eos), len(token_ids))
                records.append((tokenizer.decode(token_ids[:generated], skip_special_tokens=True).strip(),
                                input_length, generated))
        return records


def trials(model, jobs):
    prompts = [state.prompt(action) for state, action in jobs]
    return [state.record(action, output) for (state, action), output in zip(jobs, model.generate(prompts), strict=True)]


def move(model, jobs):
    for (state, action), record in zip(jobs, trials(model, jobs), strict=True):
        state.advance(record, action)


def collect(model, questions, run_dir):
    states = [State(question, SEED + i) for i, question in enumerate(questions)]
    rows = []
    for step in range(1, max(TRAIN_STEPS) + 1):
        # All branches use a fixed history and cannot update it or see other branch outputs.
        actions = []
        for state in states:
            choices = state.choices() if step > WARMUP and step % 2 == 0 else []
            actions.append((state, state.rng.choice(choices) if choices else None))
        move(model, actions)
        if step in TRAIN_STEPS:
            jobs = [(state, action) for state in states
                    for action in state.rng.sample(state.choices(), min(8, len(state.choices())))]
            for (state, action), record in zip(jobs, trials(model, jobs), strict=True):
                rows.append(dict(question=state.question["id"], step=step, features=state.features(action), **record))
            print(f"Training step {step}: {len(rows)} branch labels", flush=True)
    write(run_dir / "training-branches.json", rows)
    return rows


def fit(features, target):
    y = np.asarray(target, dtype=float)
    if len(set(target)) != 2:
        raise RuntimeError(f"Need both target classes, observed {sum(target)}/{len(target)} positives")
    matrix = np.asarray(features, dtype=float)
    center, scale = matrix.mean(axis=0), matrix.std(axis=0)
    scale[scale < 1e-8] = 1.0
    x = np.column_stack([np.ones(len(y)), (matrix - center) / scale])

    def loss(weights):
        logits = x @ weights
        penalty = weights.copy()
        penalty[0] = 0
        return (np.sum(np.logaddexp(0, logits) - y * logits) + 0.5 * penalty @ penalty,
                x.T @ (expit(logits) - y) + penalty)

    result = minimize(loss, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(result.message)
    return {"weights": result.x.tolist(), "center": center.tolist(), "scale": scale.tolist(),
            "positives": int(y.sum()), "examples": len(y)}


def probability(model, x):
    return float(expit(model["weights"][0] + np.dot(
        model["weights"][1:], (np.asarray(x) - model["center"]) / model["scale"],
    )))


def train(rows):
    valid = [row for row in rows if row["valid"]]
    return {
        "joint": fit([r["features"] for r in rows], [r["novel"] for r in rows]),
        "valid": fit([r["features"] for r in rows], [r["valid"] for r in rows]),
        "novel_given_valid": fit([r["features"] for r in valid], [r["novel"] for r in valid]),
    }


def choose(state, arm, models):
    choices = state.choices() if arm != "direct" else []
    if not choices:
        return None
    if arm == "random":
        return state.rng.choice(choices)
    state.rng.shuffle(choices)

    def score(action):
        x = state.features(action)
        if arm == "rare_edge":
            return -x[1]
        if arm == "joint":
            return probability(models["joint"], x)
        return probability(models["valid"], x) * probability(models["novel_given_valid"], x)

    return max(choices, key=score)


def evaluate(model, questions, models, budget, run_dir):
    shared = [State(question, SEED + i) for i, question in enumerate(questions)]
    for _ in range(WARMUP):
        move(model, [(state, None) for state in shared])
    states = {(state.question["id"], arm): copy.deepcopy(state) for state in shared for arm in ARMS}
    for step in range(WARMUP + 1, budget + 1):
        order = list(states)
        random.Random(SEED + step).shuffle(order)
        move(model, [(states[key], choose(states[key], key[1], models)) for key in order])
        if step % 4 == 0 or step == budget:
            print(f"Evaluation step {step}/{budget}: {len(states)} question/policy pairs", flush=True)
    traces = [{"id": qid, "arm": arm, "rounds": state.records} for (qid, arm), state in states.items()]
    write(run_dir / "test-rollouts.json", traces)
    return traces


def metrics(rounds, answer_hashes, budget):
    attempts = [row for row in rounds if row["action"] is not None]
    unique = {row["graph_hash"] for row in rounds if row["graph_hash"] is not None}
    valid = [row for row in rounds if row["valid"]]
    return dict(
        recovery=len({row["graph_hash"] for row in valid} & answer_hashes) / len(answer_hashes),
        validity=len(valid) / budget, uniqueness=len(unique) / budget,
        distinct_valid=len({row["graph_hash"] for row in valid}),
        followups=len(attempts), compliant=sum(row["compliant"] for row in attempts),
        input_tokens=sum(row["input_tokens"] for row in rounds),
        output_tokens=sum(row["output_tokens"] for row in rounds),
    )


def score(traces, answer_sets, budget):
    scored = {}
    aggregate = {}
    by_arm = {}
    for arm in ARMS:
        rows = [trace for trace in traces if trace["arm"] == arm]
        by_arm[arm] = {r["id"]: metrics(r["rounds"], set(answer_sets[r["id"]]), budget) for r in rows}
        scored[arm] = by_arm[arm]
        values = list(by_arm[arm].values())
        averages = {key: float(np.mean([row[key] for row in values])) for key in values[0]}
        n_followups = sum(row["followups"] for row in values)
        averages["compliance"] = (sum(row["compliant"] for row in values) / n_followups) if n_followups else None
        aggregate[arm] = averages
    comparisons = {}
    for control in ARMS[:-1]:
        deltas = np.array([row["recovery"] - by_arm[control][qid]["recovery"]
                           for qid, row in by_arm["factorized"].items()])
        rng = np.random.default_rng(SEED)
        draws = rng.choice(deltas, size=(10000, len(deltas)), replace=True).mean(axis=1)
        comparisons[control] = dict(delta=float(deltas.mean()), interval_95=np.quantile(draws, [0.025, 0.975]).tolist(),
                                    wins=int(sum(deltas > 0)), ties=int(sum(deltas == 0)), losses=int(sum(deltas < 0)))
    return dict(aggregate=aggregate, paired_recovery=comparisons, per_question=scored)


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("phase", choices=("prepare", "train", "evaluate"))
    args.add_argument("--run-dir", type=Path, required=True)
    args.add_argument("--budget", type=int, default=32)
    args.add_argument("--batch-size", type=int, default=16)
    config = args.parse_args()
    directory = config.run_dir
    if config.phase == "prepare":
        prepare(ROOT / "data/causal-n4-plan-a.json", directory)
        return
    assert config.budget >= WARMUP
    splits = read(directory / "questions.json")
    if config.phase == "train":
        model = LocalQwen(config.batch_size)
        rows = collect(model, splits["train"], directory)
        models = train(rows)
        write(directory / "selectors.json", {"feature_names": FEATURES, "models": models})
        print({key: (value["positives"], value["examples"]) for key, value in models.items()})
        return
    models = read(directory / "selectors.json")["models"]
    model = LocalQwen(config.batch_size)
    traces = evaluate(model, splits["test"], models, config.budget, directory)
    scores = score(traces, read(directory / "test-answers.json"), config.budget)
    scores["settings"] = dict(train_questions=len(splits["train"]), test_questions=len(splits["test"]),
                              budget=config.budget, common_warmup=WARMUP, seed=SEED,
                              gpu=model.torch.cuda.get_device_name(0), precision="BF16", temperature=0.7,
                              max_new_tokens=64, model_source=read(model.path / "source-manifest.json")["revision"],
                              peak_allocated_gib=model.torch.cuda.max_memory_allocated() / 1024**3)
    write(directory / "metrics.json", scores)
    print(json.dumps({"aggregate": scores["aggregate"], "paired_recovery": scores["paired_recovery"]}, indent=2))


if __name__ == "__main__":
    main()
