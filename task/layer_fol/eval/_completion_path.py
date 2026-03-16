"""Completion-path validation for layered FOL evaluation."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from task.layer_gen.util import tokenize_layer_fol
from task.layer_gen.util.fol_rule_bank import FOLLayerRule, FOLRuleBank, FOLSequent

from ._types import (
    FAILURE_DECODE_ERROR,
    FAILURE_GOAL_NOT_REACHED,
    FAILURE_INAPPLICABLE_RULE_ERROR,
    FAILURE_UNKNOWN_RULE_ERROR,
    FOLCompletionPathMetrics,
    FOLCompletionPathResult,
    FOLCompletionPathStep,
    FOLRuleMatchResult,
    infer_fol_predicate_layer,
)
from ._helpers import _expand_rules_by_example, _resolve_fol_tokenizer, _safe_rate, _sorted_fol_atoms
from ._rule_match import match_rule_completion_fol


def _rule_match_unknown_for_path(result: FOLRuleMatchResult) -> bool:
    return bool(result.unknown_rule_error or result.wrong_rule_error or result.matched_rule is None)


def _decode_prompt_context(
    *,
    prompt_tokens: list[int] | np.ndarray,
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer,
) -> FOLSequent:
    try:
        prompt = [int(tok) for tok in np.asarray(prompt_tokens, dtype=np.int32).tolist()]
    except (TypeError, ValueError) as err:
        raise ValueError("Prompt tokens must be a 1D integer sequence.") from err
    return tokenizer.decode_prompt(prompt)


def validate_completion_path_fol(
    *,
    rule_bank: FOLRuleBank,
    prompt_tokens: list[int] | np.ndarray,
    completion_tokens: list[int] | np.ndarray,
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer | None = None,
    active_rules: Iterable[FOLLayerRule] | None = None,
    fixed_rules: Iterable[FOLLayerRule] | None = None,
    demo_rules: Iterable[FOLLayerRule] | None = None,
) -> FOLCompletionPathResult:
    tokenizer = _resolve_fol_tokenizer(rule_bank=rule_bank, tokenizer=tokenizer)
    sequent = _decode_prompt_context(prompt_tokens=prompt_tokens, tokenizer=tokenizer)
    if len(sequent.ants) == 0:
        raise ValueError("Prompt antecedents are empty; cannot validate completion path.")

    initial_layer = int(min(infer_fol_predicate_layer(atom.predicate) for atom in sequent.ants))
    facts = set(sequent.ants)
    goal = sequent.cons
    steps: list[FOLCompletionPathStep] = []

    try:
        completion = [int(tok) for tok in np.asarray(completion_tokens, dtype=np.int32).tolist()]
    except (TypeError, ValueError):
        completion = []

    decoded = tokenizer.try_decode_completion_texts(completion)
    if not decoded.ok or decoded.value is None:
        return FOLCompletionPathResult(
            success=False,
            failure_reason=FAILURE_DECODE_ERROR,
            n_steps=0,
            goal_reached=False,
            final_layer=initial_layer,
            final_facts=tuple(atom.text for atom in _sorted_fol_atoms(facts)),
            steps=(),
        )
    decoded_statements = decoded.value

    current_layer = int(initial_layer)
    active_rules_list = list(active_rules) if active_rules is not None else None
    fixed_rules_list = list(fixed_rules) if fixed_rules is not None else None
    demo_rules_list = list(demo_rules) if demo_rules is not None else None

    for step_idx, statement_text in enumerate(decoded_statements):
        matched = match_rule_completion_fol(
            rule_bank=rule_bank,
            src_layer=current_layer,
            completion_tokens=tokenizer.encode_completion_texts([statement_text]),
            tokenizer=tokenizer,
            active_rules=active_rules_list,
            fixed_rules=fixed_rules_list,
            demo_rules=demo_rules_list,
        )

        if matched.decode_error:
            steps.append(
                FOLCompletionPathStep(
                    step_idx=int(step_idx),
                    src_layer=int(current_layer),
                    decoded_statement=None,
                    matched_rule_statement=None,
                    decode_error=True,
                    unknown_rule_error=False,
                    inapplicable_rule_error=False,
                    goal_reached=False,
                )
            )
            return FOLCompletionPathResult(
                success=False,
                failure_reason=FAILURE_DECODE_ERROR,
                n_steps=len(steps),
                goal_reached=False,
                final_layer=int(current_layer),
                final_facts=tuple(atom.text for atom in _sorted_fol_atoms(facts)),
                steps=tuple(steps),
            )

        if _rule_match_unknown_for_path(matched):
            steps.append(
                FOLCompletionPathStep(
                    step_idx=int(step_idx),
                    src_layer=int(current_layer),
                    decoded_statement=matched.decoded_statement,
                    matched_rule_statement=None,
                    decode_error=False,
                    unknown_rule_error=True,
                    inapplicable_rule_error=False,
                    goal_reached=False,
                )
            )
            return FOLCompletionPathResult(
                success=False,
                failure_reason=FAILURE_UNKNOWN_RULE_ERROR,
                n_steps=len(steps),
                goal_reached=False,
                final_layer=int(current_layer),
                final_facts=tuple(atom.text for atom in _sorted_fol_atoms(facts)),
                steps=tuple(steps),
            )

        rule = matched.matched_rule
        assert rule is not None
        if not set(rule.lhs).issubset(facts):
            steps.append(
                FOLCompletionPathStep(
                    step_idx=int(step_idx),
                    src_layer=int(current_layer),
                    decoded_statement=matched.decoded_statement,
                    matched_rule_statement=rule.statement_text,
                    decode_error=False,
                    unknown_rule_error=False,
                    inapplicable_rule_error=True,
                    goal_reached=False,
                )
            )
            return FOLCompletionPathResult(
                success=False,
                failure_reason=FAILURE_INAPPLICABLE_RULE_ERROR,
                n_steps=len(steps),
                goal_reached=False,
                final_layer=int(current_layer),
                final_facts=tuple(atom.text for atom in _sorted_fol_atoms(facts)),
                steps=tuple(steps),
            )

        facts = set(rule.rhs)
        goal_reached = goal in facts
        steps.append(
            FOLCompletionPathStep(
                step_idx=int(step_idx),
                src_layer=int(current_layer),
                decoded_statement=matched.decoded_statement,
                matched_rule_statement=rule.statement_text,
                decode_error=False,
                unknown_rule_error=False,
                inapplicable_rule_error=False,
                goal_reached=bool(goal_reached),
            )
        )
        current_layer = int(rule.dst_layer)

        if goal_reached:
            return FOLCompletionPathResult(
                success=True,
                failure_reason=None,
                n_steps=len(steps),
                goal_reached=True,
                final_layer=int(current_layer),
                final_facts=tuple(atom.text for atom in _sorted_fol_atoms(facts)),
                steps=tuple(steps),
            )

    return FOLCompletionPathResult(
        success=False,
        failure_reason=FAILURE_GOAL_NOT_REACHED,
        n_steps=len(steps),
        goal_reached=False,
        final_layer=int(current_layer),
        final_facts=tuple(atom.text for atom in _sorted_fol_atoms(facts)),
        steps=tuple(steps),
    )


def evaluate_completion_paths_fol(
    *,
    rule_bank: FOLRuleBank,
    prompt_tokens: Iterable[list[int] | np.ndarray],
    completion_tokens: Iterable[list[int] | np.ndarray],
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer | None = None,
    active_rules: Iterable[FOLLayerRule] | None = None,
    fixed_rules: Iterable[FOLLayerRule] | None = None,
    demo_rules: Iterable[FOLLayerRule] | None = None,
    active_rules_by_example: Iterable[Iterable[FOLLayerRule] | None] | None = None,
    fixed_rules_by_example: Iterable[Iterable[FOLLayerRule] | None] | None = None,
    demo_rules_by_example: Iterable[Iterable[FOLLayerRule] | None] | None = None,
) -> FOLCompletionPathMetrics:
    prompt_tokens = list(prompt_tokens)
    completion_tokens = list(completion_tokens)
    if len(prompt_tokens) != len(completion_tokens):
        raise ValueError(
            "prompt_tokens and completion_tokens must have same length, got "
            f"{len(prompt_tokens)} and {len(completion_tokens)}"
        )

    active_rules_list = list(active_rules) if active_rules is not None else None
    fixed_rules_list = list(fixed_rules) if fixed_rules is not None else None
    demo_rules_list = list(demo_rules) if demo_rules is not None else None
    active_rules_per_example = _expand_rules_by_example(
        active_rules_by_example, len(prompt_tokens), "active_rules_by_example",
    )
    fixed_rules_per_example = _expand_rules_by_example(
        fixed_rules_by_example, len(prompt_tokens), "fixed_rules_by_example",
    )
    demo_rules_per_example = _expand_rules_by_example(
        demo_rules_by_example, len(prompt_tokens), "demo_rules_by_example",
    )

    results = tuple(
        validate_completion_path_fol(
            rule_bank=rule_bank,
            prompt_tokens=prompt,
            completion_tokens=completion,
            tokenizer=tokenizer,
            active_rules=(
                active_rules_list if active_rules_per_example[idx] is None
                else active_rules_per_example[idx]
            ),
            fixed_rules=(
                fixed_rules_list if fixed_rules_per_example[idx] is None
                else fixed_rules_per_example[idx]
            ),
            demo_rules=(
                demo_rules_list if demo_rules_per_example[idx] is None
                else demo_rules_per_example[idx]
            ),
        )
        for idx, (prompt, completion) in enumerate(zip(prompt_tokens, completion_tokens))
    )

    n_examples = len(results)
    n_success = sum(int(result.success) for result in results)
    n_failure_decode_error = sum(
        int(result.failure_reason == FAILURE_DECODE_ERROR)
        for result in results
    )
    n_failure_unknown_rule_error = sum(
        int(result.failure_reason == FAILURE_UNKNOWN_RULE_ERROR)
        for result in results
    )
    n_failure_inapplicable_rule_error = sum(
        int(result.failure_reason == FAILURE_INAPPLICABLE_RULE_ERROR)
        for result in results
    )
    n_failure_goal_not_reached = sum(
        int(result.failure_reason == FAILURE_GOAL_NOT_REACHED)
        for result in results
    )
    avg_steps = float(np.mean([result.n_steps for result in results])) if results else 0.0

    return FOLCompletionPathMetrics(
        n_examples=n_examples,
        n_success=n_success,
        n_failure_decode_error=n_failure_decode_error,
        n_failure_unknown_rule_error=n_failure_unknown_rule_error,
        n_failure_inapplicable_rule_error=n_failure_inapplicable_rule_error,
        n_failure_goal_not_reached=n_failure_goal_not_reached,
        success_rate=_safe_rate(n_success, n_examples),
        decode_error_rate=_safe_rate(n_failure_decode_error, n_examples),
        unknown_rule_error_rate=_safe_rate(n_failure_unknown_rule_error, n_examples),
        inapplicable_rule_error_rate=_safe_rate(n_failure_inapplicable_rule_error, n_examples),
        goal_not_reached_rate=_safe_rate(n_failure_goal_not_reached, n_examples),
        avg_steps=avg_steps,
        results=results,
    )
