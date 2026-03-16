"""Rule matching for single-statement layered FOL completions."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from task.layer_gen.util import tokenize_layer_fol
from task.layer_gen.util.fol_rule_bank import FOLLayerRule, FOLRuleBank, parse_clause_text

from ._types import (
    FAILURE_DECODE_ERROR,
    FAILURE_UNKNOWN_RULE_ERROR,
    FAILURE_WRONG_RULE_ERROR,
    FOLRuleMatchMetrics,
    FOLRuleMatchResult,
)
from ._helpers import (
    _decode_single_completion_statement,
    _expand_rules_by_example,
    _resolve_fol_tokenizer,
    _safe_rate,
)
from ._unify import _any_rule_schema_matches, _match_instantiated_rule_from_candidates


def match_rule_completion_fol(
    *,
    rule_bank: FOLRuleBank,
    src_layer: int,
    completion_tokens: list[int] | np.ndarray,
    expected_statement_text: str | None = None,
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer | None = None,
    active_rules: Iterable[FOLLayerRule] | None = None,
    fixed_rules: Iterable[FOLLayerRule] | None = None,
    demo_rules: Iterable[FOLLayerRule] | None = None,
) -> FOLRuleMatchResult:
    tokenizer = _resolve_fol_tokenizer(rule_bank=rule_bank, tokenizer=tokenizer)
    src_layer = int(src_layer)
    active_rules_list = (
        list(active_rules)
        if active_rules is not None
        else list(rule_bank.transition_rules(src_layer))
    )
    fixed_rules_list = list(fixed_rules or ())
    demo_rules_list = list(demo_rules or ())

    try:
        completion = [int(tok) for tok in np.asarray(completion_tokens, dtype=np.int32).tolist()]
    except (TypeError, ValueError):
        completion = []

    decoded = _decode_single_completion_statement(
        tokenizer=tokenizer,
        completion_tokens=completion,
    )
    if not decoded.ok or decoded.value is None:
        return FOLRuleMatchResult(
            src_layer=src_layer,
            decoded_statement=None,
            expected_statement_text=expected_statement_text,
            matched_rule=None,
            decode_error=True,
            unknown_rule_error=False,
            wrong_rule_error=False,
            is_valid_rule=False,
            is_correct=False,
            match_source=None,
        )
    decoded_statement = decoded.value
    try:
        lhs_ground, rhs_ground = parse_clause_text(decoded_statement)
    except (ValueError, TypeError):
        return FOLRuleMatchResult(
            src_layer=src_layer,
            decoded_statement=None,
            expected_statement_text=expected_statement_text,
            matched_rule=None,
            decode_error=True,
            unknown_rule_error=False,
            wrong_rule_error=False,
            is_valid_rule=False,
            is_correct=False,
            match_source=None,
        )

    matched_rule = _match_instantiated_rule_from_candidates(
        candidates=active_rules_list,
        lhs_ground=lhs_ground,
        rhs_ground=rhs_ground,
    )
    match_source: str | None = None
    if matched_rule is not None:
        match_source = "active"
    if matched_rule is None and fixed_rules_list:
        matched_rule = _match_instantiated_rule_from_candidates(
            candidates=fixed_rules_list,
            lhs_ground=lhs_ground,
            rhs_ground=rhs_ground,
        )
        if matched_rule is not None:
            match_source = "fixed"
    if matched_rule is None and demo_rules_list:
        matched_rule = _match_instantiated_rule_from_candidates(
            candidates=demo_rules_list,
            lhs_ground=lhs_ground,
            rhs_ground=rhs_ground,
        )
        if matched_rule is not None:
            match_source = "demo"
    if matched_rule is None:
        candidates = [
            *active_rules_list,
            *fixed_rules_list,
            *demo_rules_list,
        ]
        schema_matches = _any_rule_schema_matches(
            lhs_ground, rhs_ground, candidates,
        )
        return FOLRuleMatchResult(
            src_layer=src_layer,
            decoded_statement=decoded_statement,
            expected_statement_text=expected_statement_text,
            matched_rule=None,
            decode_error=False,
            unknown_rule_error=not schema_matches,
            wrong_rule_error=schema_matches,
            is_valid_rule=False,
            is_correct=False,
            match_source=None,
        )

    wrong_rule_error = (
        expected_statement_text is not None
        and decoded_statement != str(expected_statement_text)
    )
    is_correct = not wrong_rule_error
    return FOLRuleMatchResult(
        src_layer=src_layer,
        decoded_statement=decoded_statement,
        expected_statement_text=expected_statement_text,
        matched_rule=matched_rule,
        decode_error=False,
        unknown_rule_error=False,
        wrong_rule_error=bool(wrong_rule_error),
        is_valid_rule=True,
        is_correct=is_correct,
        match_source=match_source,
    )


def evaluate_rule_matches_fol(
    *,
    rule_bank: FOLRuleBank,
    src_layers: Iterable[int],
    completion_tokens: Iterable[list[int] | np.ndarray],
    expected_statement_texts: Iterable[str | None] | None = None,
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer | None = None,
    active_rules: Iterable[FOLLayerRule] | None = None,
    fixed_rules: Iterable[FOLLayerRule] | None = None,
    demo_rules: Iterable[FOLLayerRule] | None = None,
    active_rules_by_example: Iterable[Iterable[FOLLayerRule] | None] | None = None,
    fixed_rules_by_example: Iterable[Iterable[FOLLayerRule] | None] | None = None,
    demo_rules_by_example: Iterable[Iterable[FOLLayerRule] | None] | None = None,
) -> FOLRuleMatchMetrics:
    src_layers = [int(layer) for layer in src_layers]
    completion_tokens = list(completion_tokens)
    if len(src_layers) != len(completion_tokens):
        raise ValueError(
            f"src_layers and completion_tokens must have same length, got "
            f"{len(src_layers)} and {len(completion_tokens)}"
        )

    if expected_statement_texts is None:
        expected_statement_texts = [None] * len(src_layers)
    else:
        expected_statement_texts = list(expected_statement_texts)
        if len(expected_statement_texts) != len(src_layers):
            raise ValueError(
                "expected_statement_texts must match src_layers length, got "
                f"{len(expected_statement_texts)} and {len(src_layers)}"
            )

    active_rules_list = list(active_rules) if active_rules is not None else None
    fixed_rules_list = list(fixed_rules) if fixed_rules is not None else None
    demo_rules_list = list(demo_rules) if demo_rules is not None else None
    active_rules_per_example = _expand_rules_by_example(
        active_rules_by_example, len(src_layers), "active_rules_by_example",
    )
    fixed_rules_per_example = _expand_rules_by_example(
        fixed_rules_by_example, len(src_layers), "fixed_rules_by_example",
    )
    demo_rules_per_example = _expand_rules_by_example(
        demo_rules_by_example, len(src_layers), "demo_rules_by_example",
    )

    results = tuple(
        match_rule_completion_fol(
            rule_bank=rule_bank,
            src_layer=src_layer,
            completion_tokens=completion,
            expected_statement_text=expected_statement,
            tokenizer=tokenizer,
            active_rules=(
                [
                    *(active_rules_list or ()),
                    *(example_active_rules or ()),
                ]
                if active_rules_list is not None or example_active_rules is not None
                else None
            ),
            fixed_rules=(
                [
                    *(fixed_rules_list or ()),
                    *(example_fixed_rules or ()),
                ]
                if fixed_rules_list is not None or example_fixed_rules is not None
                else None
            ),
            demo_rules=(
                [
                    *(demo_rules_list or ()),
                    *(example_demo_rules or ()),
                ]
                if demo_rules_list is not None or example_demo_rules is not None
                else None
            ),
        )
        for (
            src_layer,
            completion,
            expected_statement,
            example_active_rules,
            example_fixed_rules,
            example_demo_rules,
        ) in zip(
            src_layers,
            completion_tokens,
            expected_statement_texts,
            active_rules_per_example,
            fixed_rules_per_example,
            demo_rules_per_example,
        )
    )

    n_examples = len(results)
    n_correct = sum(int(result.is_correct) for result in results)
    n_decode_error = sum(int(result.decode_error) for result in results)
    n_unknown_rule_error = sum(int(result.unknown_rule_error) for result in results)
    n_wrong_rule_error = sum(int(result.wrong_rule_error) for result in results)

    return FOLRuleMatchMetrics(
        n_examples=n_examples,
        n_correct=n_correct,
        n_decode_error=n_decode_error,
        n_unknown_rule_error=n_unknown_rule_error,
        n_wrong_rule_error=n_wrong_rule_error,
        accuracy=_safe_rate(n_correct, n_examples),
        decode_error_rate=_safe_rate(n_decode_error, n_examples),
        unknown_rule_error_rate=_safe_rate(n_unknown_rule_error, n_examples),
        wrong_rule_error_rate=_safe_rate(n_wrong_rule_error, n_examples),
        results=results,
    )
