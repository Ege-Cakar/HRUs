"""Parsing helpers, dispatch function, and aliases for layered FOL evaluation."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from task.layer_gen.util import tokenize_layer_fol
from task.layer_gen.util.fol_rule_bank import FOLLayerRule, FOLRuleBank, parse_clause_text

from ._rule_match import match_rule_completion_fol, evaluate_rule_matches_fol
from ._completion_path import validate_completion_path_fol, evaluate_completion_paths_fol
from ._rollout import (
    sample_rollout_examples_fol,
    run_layer_rollout_fol,
    evaluate_layer_rollouts_fol,
)


def parse_rule_statements_for_layer(
    *,
    src_layer: int,
    statement_texts: Iterable[str],
) -> list[FOLLayerRule]:
    out: list[FOLLayerRule] = []
    src_layer = int(src_layer)
    for text in statement_texts:
        try:
            lhs, rhs = parse_clause_text(str(text))
        except ValueError:
            continue
        out.append(
            FOLLayerRule(
                src_layer=src_layer,
                dst_layer=src_layer + 1,
                lhs=lhs,
                rhs=rhs,
            )
        )
    return out


def resolve_rule_sets_from_context(
    *,
    src_layer: int,
    rule_context: dict | None,
) -> tuple[list[FOLLayerRule] | None, list[FOLLayerRule] | None, list[FOLLayerRule] | None]:
    if not isinstance(rule_context, dict):
        return None, None, None
    active = (
        parse_rule_statements_for_layer(
            src_layer=src_layer,
            statement_texts=rule_context.get("active_rule_texts", []),
        )
        if "active_rule_texts" in rule_context
        else None
    )
    fixed = (
        parse_rule_statements_for_layer(
            src_layer=src_layer,
            statement_texts=rule_context.get("fixed_rule_texts", []),
        )
        if "fixed_rule_texts" in rule_context
        else None
    )
    demo = (
        parse_rule_statements_for_layer(
            src_layer=src_layer,
            statement_texts=rule_context.get("demo_schema_texts", []),
        )
        if "demo_schema_texts" in rule_context
        else None
    )
    return active, fixed, demo


def completion_is_valid_for_layer_fol(
    *,
    rule_bank: FOLRuleBank,
    src_layer: int,
    completion_tokens: list[int] | np.ndarray,
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer | None = None,
    completion_format: str = "single",
    prompt_tokens: list[int] | np.ndarray | None = None,
    active_rules: Iterable[FOLLayerRule] | None = None,
    fixed_rules: Iterable[FOLLayerRule] | None = None,
    demo_rules: Iterable[FOLLayerRule] | None = None,
) -> bool:
    completion_format = str(completion_format)
    if completion_format == "single":
        result = match_rule_completion_fol(
            rule_bank=rule_bank,
            src_layer=src_layer,
            completion_tokens=completion_tokens,
            tokenizer=tokenizer,
            active_rules=active_rules,
            fixed_rules=fixed_rules,
            demo_rules=demo_rules,
        )
        return result.is_valid_rule
    if completion_format != "full":
        raise ValueError(
            f"completion_format must be 'single' or 'full', got {completion_format!r}"
        )
    if prompt_tokens is None:
        raise ValueError("prompt_tokens are required for completion_format='full'.")
    result = validate_completion_path_fol(
        rule_bank=rule_bank,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        tokenizer=tokenizer,
        active_rules=active_rules,
        fixed_rules=fixed_rules,
        demo_rules=demo_rules,
    )
    return bool(result.success)


# Aliases
completion_is_valid_for_layer = completion_is_valid_for_layer_fol
match_rule_completion = match_rule_completion_fol
evaluate_rule_matches = evaluate_rule_matches_fol
sample_rollout_examples = sample_rollout_examples_fol
run_layer_rollout = run_layer_rollout_fol
evaluate_layer_rollouts = evaluate_layer_rollouts_fol
