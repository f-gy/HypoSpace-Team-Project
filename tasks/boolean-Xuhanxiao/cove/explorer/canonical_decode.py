"""Grammar-only prefix masks with optional canonical-class exclusion.

No observations, reference answers, task IDs, model responses, or data splits
are inputs to support construction. Observation consistency is NOT enforced.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .experiment import assess
from .upstream import evaluator


def expressions(depth: int, variables: tuple[str, ...] = ("x", "y")) -> set[str]:
    if depth < 0:
        raise ValueError("depth must be nonnegative")
    if depth == 0:
        return set(variables)
    previous = expressions(depth - 1, variables)
    return previous | {f"NOT ({a})" for a in previous} | {
        f"({a} {op} {b})" for op in ("AND", "OR") for a in previous for b in previous
    }


def grammar_support(metadata: dict) -> list[dict]:
    if set(metadata["operators"]) != {"AND", "OR", "NOT"} or metadata["max_depth"] > 2:
        raise ValueError("This finite-support implementation is for basic depth <= 2 only")
    ev = evaluator(metadata)
    pool = []
    for formula in sorted(expressions(metadata["max_depth"], tuple(metadata["variables"]))):
        answer = "Expression: " + formula
        h = assess(ev, metadata, {"id": "grammar-only", "observations": []}, answer, [])
        if h["in_space"]:
            pool.append({"answer": answer, "key": h["key"]})
    return pool


def completion_tokens(tokenizer, support: list[dict]) -> list[dict]:
    messages = [{"role": "user", "content": "Grammar tokenization probe."}]
    prefix = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    result = []
    for h in support:
        full = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": h["answer"]}],
            add_generation_prompt=False, tokenize=True,
        )
        if full[:len(prefix)] != prefix:
            raise ValueError("Chat template does not preserve generation prefix")
        end = next(i for i in range(len(prefix), len(full)) if full[i] in tokenizer.eos_token_ids)
        result.append({**h, "tokens": full[len(prefix):end + 1]})
    return result


@dataclass
class Node:
    children: dict[int, "Node"] = field(default_factory=dict)
    terminal_key: str | None = None


class SupportTrie:
    def __init__(self, support: list[dict], blocked_keys=(), blocked_strings=()):
        blocked_keys, blocked_strings = set(blocked_keys), set(blocked_strings)
        self.root = Node()
        self.entries = 0
        for h in support:
            if h["key"] in blocked_keys or h["answer"] in blocked_strings:
                continue
            node = self.root
            for token in h["tokens"]:
                node = node.children.setdefault(token, Node())
            if node.terminal_key not in (None, h["key"]):
                raise ValueError("Token sequence maps to multiple canonical keys")
            node.terminal_key = h["key"]
            self.entries += 1
        if not self.entries:
            raise ValueError("Grammar support exhausted; do not invent extra samples")

    def node(self, generated) -> Node:
        node = self.root
        for token in generated:
            if token not in node.children:
                raise ValueError("Generated prefix left the active grammar")
            node = node.children[token]
        return node

    def allowed(self, generated) -> tuple[int, ...]:
        return tuple(self.node(generated).children)


class TrieLogitsProcessor:
    def __init__(self, trie: SupportTrie, prompt_length: int):
        self.trie = trie
        self.prompt_length = prompt_length
        self.steps = 0

    def __call__(self, tokens, logits):
        import mlx.core as mx
        generated = tokens[self.prompt_length:].tolist()
        node = self.trie.node(generated)
        allowed = tuple(node.children)
        # MLX speculatively computes one step beyond EOS before the streaming
        # layer observes it. Make the terminal an absorbing EOS state; this
        # extra prediction is never returned as a second answer token.
        if not allowed and node.terminal_key is not None:
            allowed = (generated[-1],)
        if not allowed:
            raise RuntimeError("Decoder requested a token after the grammar end token")
        indices = mx.array(allowed, dtype=mx.int32)
        masked = mx.full(logits.shape, -float("inf"), dtype=logits.dtype)
        masked[:, indices] = logits[:, indices]
        self.steps += 1
        return masked


def support_summary(support):
    counts = defaultdict(int)
    for h in support:
        counts[h["key"]] += 1
    return {"strings": len(support), "classes": len(counts),
            "alias_counts": dict(sorted(counts.items())),
            "uses_observations": False, "uses_reference_keys": False,
            "scope": "finite syntactic support, not the complete language of all spellings"}
