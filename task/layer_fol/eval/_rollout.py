"""Rollout sampling and execution for layered FOL evaluation."""

from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np

from task.layer_gen.util import tokenize_layer_fol
from task.layer_gen.util.fol_rule_bank import (
    FOLLayerRule,
    FOLRuleBank,
    FOLSequent,
    parse_atom_text,
    parse_clause_text,
    sample_fol_problem,
)

from ._types import (
    FAILURE_DECODE_ERROR,
    FAILURE_GOAL_NOT_REACHED,
    FAILURE_INAPPLICABLE_RULE_ERROR,
    FAILURE_UNKNOWN_RULE_ERROR,
    FAILURE_WRONG_RULE_ERROR,
    FOLLayerPredictionAdapter,
    FOLLayerRolloutExample,
    FOLLayerRolloutMetrics,
    FOLLayerRolloutResult,
    FOLLayerRolloutTraceStep,
)
from ._helpers import _resolve_fol_tokenizer, _safe_rate, _sorted_fol_atoms
from ._rule_match import match_rule_completion_fol


def sample_rollout_examples_fol(
    *,
    rule_bank: FOLRuleBank,
    distance: int,
    n_examples: int,
    initial_ant_max: int,
    max_steps: int | None = None,
    max_unify_solutions: int = 128,
    rng: np.random.Generator | None = None,
) -> list[FOLLayerRolloutExample]:
    if n_examples < 1:
        raise ValueError(f"n_examples must be >= 1, got {n_examples}")
    if rng is None:
        rng = np.random.default_rng()

    out: list[FOLLayerRolloutExample] = []
    for _ in range(int(n_examples)):
        sampled = sample_fol_problem(
            bank=rule_bank,
            distance=int(distance),
            initial_ant_max=int(initial_ant_max),
            rng=rng,
            max_unify_solutions=int(max_unify_solutions),
        )
        if not sampled.step_ants:
            raise RuntimeError("Sampled problem contained no steps.")
        out.append(
            FOLLayerRolloutExample(
                distance=int(sampled.distance),
                start_layer=int(sampled.start_layer),
                goal_atom=str(sampled.goal_atom.text),
                initial_ants=tuple(atom.text for atom in sampled.step_ants[0]),
                max_steps=int(sampled.distance if max_steps is None else max_steps),
                oracle_rule_statements=tuple(
                    str(rule.statement_text) for rule in sampled.step_rules
                ),
            )
        )
    return out


def _adapter_last_demo_rules(
    adapter: FOLLayerPredictionAdapter,
) -> list[FOLLayerRule] | None:
    getter = getattr(adapter, "get_last_demo_rules", None)
    if not callable(getter):
        return None
    try:
        raw = getter()
    except Exception:
        return None
    if raw is None:
        return None
    out = [
        rule
        for rule in raw
        if isinstance(rule, FOLLayerRule)
    ]
    return out


def _adapter_set_oracle_rule(
    adapter: FOLLayerPredictionAdapter,
    oracle_rule: FOLLayerRule | None,
) -> None:
    setter = getattr(adapter, "set_oracle_rule", None)
    if not callable(setter):
        return
    setter(oracle_rule)


def run_layer_rollout_fol(
    *,
    rule_bank: FOLRuleBank,
    example: FOLLayerRolloutExample,
    model: Callable[[np.ndarray], Any],
    adapter: FOLLayerPredictionAdapter,
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer | None = None,
    temperature: float = 0.0,
    rng: np.random.Generator | None = None,
    demo_rules: Iterable[FOLLayerRule] | None = None,
) -> FOLLayerRolloutResult:
    tokenizer = _resolve_fol_tokenizer(rule_bank=rule_bank, tokenizer=tokenizer)
    demo_rules_list = list(demo_rules) if demo_rules is not None else None
    if rng is None:
        rng = np.random.default_rng()

    facts = {parse_atom_text(atom_text) for atom_text in example.initial_ants}
    goal = parse_atom_text(example.goal_atom)
    traces: list[FOLLayerRolloutTraceStep] = []

    for step_idx in range(int(example.max_steps)):
        src_layer = int(example.start_layer) + step_idx
        oracle_rule = None
        if step_idx < len(example.oracle_rule_statements):
            lhs, rhs = parse_clause_text(str(example.oracle_rule_statements[step_idx]))
            oracle_rule = FOLLayerRule(
                src_layer=src_layer,
                dst_layer=src_layer + 1,
                lhs=lhs,
                rhs=rhs,
            )
        _adapter_set_oracle_rule(adapter, oracle_rule)
        prompt = tokenizer.tokenize_prompt(
            FOLSequent(
                ants=_sorted_fol_atoms(facts),
                cons=goal,
            )
        )
        completion = adapter.predict_completion(
            model=model,
            prompt_tokens=prompt,
            tokenizer=tokenizer,
            temperature=temperature,
            rng=rng,
        )
        step_demo_rules = _adapter_last_demo_rules(adapter)
        combined_demo_rules = (
            [
                *(demo_rules_list or ()),
                *(step_demo_rules or ()),
            ]
            if demo_rules_list is not None or step_demo_rules is not None
            else None
        )

        matched = match_rule_completion_fol(
            rule_bank=rule_bank,
            src_layer=src_layer,
            completion_tokens=completion,
            tokenizer=tokenizer,
            demo_rules=combined_demo_rules,
        )

        if matched.decode_error:
            traces.append(
                FOLLayerRolloutTraceStep(
                    step_idx=step_idx,
                    src_layer=src_layer,
                    prompt_tokens=tuple(int(tok) for tok in prompt),
                    completion_tokens=tuple(int(tok) for tok in completion),
                    decoded_statement=None,
                    matched_rule_statement=None,
                    decode_error=True,
                    unknown_rule_error=False,
                    inapplicable_rule_error=False,
                    goal_reached=False,
                )
            )
            return FOLLayerRolloutResult(
                success=False,
                failure_reason=FAILURE_DECODE_ERROR,
                n_steps=len(traces),
                goal_reached=False,
                final_layer=src_layer,
                final_facts=tuple(atom.text for atom in _sorted_fol_atoms(facts)),
                steps=tuple(traces),
                example=example,
            )

        if matched.unknown_rule_error or matched.matched_rule is None:
            failure = (
                FAILURE_UNKNOWN_RULE_ERROR if matched.unknown_rule_error
                else FAILURE_WRONG_RULE_ERROR
            )
            traces.append(
                FOLLayerRolloutTraceStep(
                    step_idx=step_idx,
                    src_layer=src_layer,
                    prompt_tokens=tuple(int(tok) for tok in prompt),
                    completion_tokens=tuple(int(tok) for tok in completion),
                    decoded_statement=matched.decoded_statement,
                    matched_rule_statement=None,
                    decode_error=False,
                    unknown_rule_error=matched.unknown_rule_error,
                    inapplicable_rule_error=False,
                    goal_reached=False,
                )
            )
            return FOLLayerRolloutResult(
                success=False,
                failure_reason=failure,
                n_steps=len(traces),
                goal_reached=False,
                final_layer=src_layer,
                final_facts=tuple(atom.text for atom in _sorted_fol_atoms(facts)),
                steps=tuple(traces),
                example=example,
            )

        rule = matched.matched_rule
        if not set(rule.lhs).issubset(facts):
            traces.append(
                FOLLayerRolloutTraceStep(
                    step_idx=step_idx,
                    src_layer=src_layer,
                    prompt_tokens=tuple(int(tok) for tok in prompt),
                    completion_tokens=tuple(int(tok) for tok in completion),
                    decoded_statement=matched.decoded_statement,
                    matched_rule_statement=rule.statement_text,
                    decode_error=False,
                    unknown_rule_error=False,
                    inapplicable_rule_error=True,
                    goal_reached=False,
                )
            )
            return FOLLayerRolloutResult(
                success=False,
                failure_reason=FAILURE_INAPPLICABLE_RULE_ERROR,
                n_steps=len(traces),
                goal_reached=False,
                final_layer=src_layer,
                final_facts=tuple(atom.text for atom in _sorted_fol_atoms(facts)),
                steps=tuple(traces),
                example=example,
            )

        facts = set(rule.rhs)
        goal_reached = goal in facts
        traces.append(
            FOLLayerRolloutTraceStep(
                step_idx=step_idx,
                src_layer=src_layer,
                prompt_tokens=tuple(int(tok) for tok in prompt),
                completion_tokens=tuple(int(tok) for tok in completion),
                decoded_statement=matched.decoded_statement,
                matched_rule_statement=rule.statement_text,
                decode_error=False,
                unknown_rule_error=False,
                inapplicable_rule_error=False,
                goal_reached=goal_reached,
            )
        )

        if goal_reached:
            return FOLLayerRolloutResult(
                success=True,
                failure_reason=None,
                n_steps=len(traces),
                goal_reached=True,
                final_layer=int(rule.dst_layer),
                final_facts=tuple(atom.text for atom in _sorted_fol_atoms(facts)),
                steps=tuple(traces),
                example=example,
            )

    return FOLLayerRolloutResult(
        success=False,
        failure_reason=FAILURE_GOAL_NOT_REACHED,
        n_steps=len(traces),
        goal_reached=False,
        final_layer=int(example.start_layer) + len(traces),
        final_facts=tuple(atom.text for atom in _sorted_fol_atoms(facts)),
        steps=tuple(traces),
        example=example,
    )


def evaluate_layer_rollouts_fol(
    *,
    rule_bank: FOLRuleBank,
    examples: Iterable[FOLLayerRolloutExample],
    model: Callable[[np.ndarray], Any],
    adapter: FOLLayerPredictionAdapter,
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer | None = None,
    temperature: float = 0.0,
    rng: np.random.Generator | None = None,
    demo_rules: Iterable[FOLLayerRule] | None = None,
) -> FOLLayerRolloutMetrics:
    if rng is None:
        rng = np.random.default_rng()

    demo_rules_list = list(demo_rules) if demo_rules is not None else None

    results = tuple(
        run_layer_rollout_fol(
            rule_bank=rule_bank,
            example=example,
            model=model,
            adapter=adapter,
            tokenizer=tokenizer,
            temperature=temperature,
            rng=rng,
            demo_rules=demo_rules_list,
        )
        for example in examples
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
    n_failure_wrong_rule_error = sum(
        int(result.failure_reason == FAILURE_WRONG_RULE_ERROR)
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

    return FOLLayerRolloutMetrics(
        n_examples=n_examples,
        n_success=n_success,
        n_failure_decode_error=n_failure_decode_error,
        n_failure_unknown_rule_error=n_failure_unknown_rule_error,
        n_failure_wrong_rule_error=n_failure_wrong_rule_error,
        n_failure_inapplicable_rule_error=n_failure_inapplicable_rule_error,
        n_failure_goal_not_reached=n_failure_goal_not_reached,
        success_rate=_safe_rate(n_success, n_examples),
        decode_error_rate=_safe_rate(n_failure_decode_error, n_examples),
        unknown_rule_error_rate=_safe_rate(n_failure_unknown_rule_error, n_examples),
        wrong_rule_error_rate=_safe_rate(n_failure_wrong_rule_error, n_examples),
        inapplicable_rule_error_rate=_safe_rate(n_failure_inapplicable_rule_error, n_examples),
        goal_not_reached_rate=_safe_rate(n_failure_goal_not_reached, n_examples),
        avg_steps=avg_steps,
        results=results,
    )
