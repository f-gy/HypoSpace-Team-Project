"""Versioned adapter for HypoSpace Boolean validators (AND/OR/NOT only).

The upstream implementation is left untouched.  Evaluation uses its expression
parser, depth definition and mechanistic keys.  The finite-universe policy
checks structural membership in the space generated from public grammar, not
per-instance compatible answers.  Do not supply test labels as feedback.
"""
from __future__ import annotations

import re
import types
import sympy as sp
from boolean_dataset import BooleanExpression, BooleanDiscoveryGame

VERSION = 'hypospace-boolean-validator-1.0.0'
SUPPORTED_OPS = frozenset({'AND', 'OR', 'NOT'})
TOKEN_RE = re.compile(r'\s*([A-Za-z_][A-Za-z_0-9]*|[()]|\S)')


def generated_universe(benchmark):
    """Enumerate public hypothesis grammar; never use case-specific GT labels."""
    _check_supported(benchmark)
    expressions = BooleanDiscoveryGame.generate_all_expressions(
        benchmark.variables, benchmark.operators, benchmark.max_depth,
        benchmark.mechanistic_opts,
    )
    return frozenset(e.mechanistic_key(**benchmark.mechanistic_opts) for e in expressions)


def _check_supported(benchmark):
    unsupported = set(benchmark.operators) - SUPPORTED_OPS
    if unsupported:
        raise ValueError(f'{VERSION} has not been validated for operators: {sorted(unsupported)}')


def _parse_checked(benchmark, text):
    if not isinstance(text, str) or not text.strip():
        return None
    text = text.strip()
    tokens = TOKEN_RE.findall(text)
    allowed_tokens = set(benchmark.variables) | set(benchmark.operators) | {'(', ')'}
    if any(t not in allowed_tokens for t in tokens):
        return None
    try:
        expr = BooleanExpression(text, benchmark.variables, benchmark.operators)
        ast = expr.sympy_expr
        if ast is None or not getattr(ast, 'free_symbols', set()).issubset(
            {sp.Symbol(v) for v in benchmark.variables}
        ):
            return None
        nodes = [ast]
        while nodes:
            node = nodes.pop()
            if isinstance(node, sp.Symbol):
                if str(node) not in benchmark.variables:
                    return None
                continue
            if isinstance(node, sp.Not): op = 'NOT'
            elif isinstance(node, sp.And): op = 'AND'
            elif isinstance(node, sp.Or): op = 'OR'
            else: return None  # Reject numeric/Boolean constants and unknown AST nodes.
            if op not in benchmark.operators:
                return None
            nodes.extend(node.args)
        if benchmark._get_ast_depth(ast) > benchmark.max_depth:
            return None
        return expr
    except Exception:
        return None


def install(benchmark, *, policy='grammar', universe=None):
    """Install validation on an existing BooleanBenchmarkRefined instance.

    grammar: allowed syntax, variables, operators, AST depth and observations.
    enumerated: grammar plus membership in public generated structural universe.
    No method receives or looks at the instance-specific answer list.
    """
    _check_supported(benchmark)
    if policy not in ('grammar', 'enumerated'):
        raise ValueError('policy must be grammar or enumerated')
    if policy == 'enumerated':
        if universe is None:
            universe = generated_universe(benchmark)
        universe = frozenset(universe)
        if not universe:
            raise ValueError('enumerated policy requires a nonempty universe')

    def in_space(self, text):
        expr = _parse_checked(self, text)
        if expr is None:
            return False
        return policy == 'grammar' or expr.mechanistic_key(**self.mechanistic_opts) in universe

    def validate(self, text, observations):
        expr = _parse_checked(self, text)
        if expr is None:
            return False, None
        if policy == 'enumerated' and expr.mechanistic_key(**self.mechanistic_opts) not in universe:
            return False, None
        try:
            for obs in observations:
                values = tuple(obs.inputs[v] for v in expr.variables)
                if values not in expr.truth_table or expr.truth_table[values] != obs.output:
                    return False, None
            return True, expr.truth_table
        except (AttributeError, KeyError, TypeError):
            return False, None

    benchmark._in_space = types.MethodType(in_space, benchmark)
    benchmark.validate_expression = types.MethodType(validate, benchmark)
    benchmark.validator_version = VERSION
    benchmark.validator_policy = policy
    return benchmark
