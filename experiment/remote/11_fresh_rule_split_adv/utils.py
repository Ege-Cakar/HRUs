"""Helpers for experiment 10 fresh-rule split evaluation."""

from __future__ import annotations

import re

import numpy as np

from task.layer_fol import (
    FOLRuleMatchMetrics,
    _find_lhs_substitutions_for_facts,
    _subst_binds_rhs_variables,
)
from task.layer_gen.util.fol_rule_bank import FOLAtom, FOLLayerRule, FOLRuleBank, FOLSequent
from task.layer_gen.util.tokenize_layer_fol import FOLLayerTokenizer


_PRED_RE = re.compile(r"r(\d+)_(\d+)$")
_FRESH_PRED_RE = re.compile(r"r_[a-z0-9]{4}$")


def _layer_from_predicate(predicate: str) -> int:
    match = _PRED_RE.fullmatch(str(predicate))
    if match is not None:
        return int(match.group(1))
    if _FRESH_PRED_RE.fullmatch(str(predicate)):
        return 0
    raise ValueError(f"Unsupported layered predicate name: {predicate}")


def _find_sequent_prompt_in_tokens(
    row_tokens,
    *,
    tokenizer: FOLLayerTokenizer,
    pad_token_id: int = 0,
) -> tuple[np.ndarray, FOLSequent]:
    row = np.asarray(row_tokens, dtype=np.int32)
    if row.ndim != 1:
        raise ValueError(f"Expected 1D row tokens, got {row.shape}")

    nonpad = row[row != int(pad_token_id)]
    sep_idx = np.where(nonpad == int(tokenizer.sep_token_id))[0]
    if sep_idx.size == 0:
        raise ValueError("Missing SEP token in row.")

    for pick in reversed(sep_idx.tolist()):
        prev = sep_idx[sep_idx < pick]
        start = int(prev[-1]) + 1 if prev.size > 0 else 0
        segment = nonpad[start : int(pick) + 1]
        if segment.size < 2:
            continue
        try:
            sequent = tokenizer.decode_prompt(segment.astype(int).tolist())
        except ValueError:
            continue
        return nonpad[: int(pick) + 1].astype(np.int32), sequent

    raise ValueError("Could not locate a valid sequent prompt segment in row tokens.")


def infer_src_layer_from_prompt_tokens(
    row_tokens,
    *,
    tokenizer: FOLLayerTokenizer,
    pad_token_id: int = 0,
) -> int:
    _, sequent = _find_sequent_prompt_in_tokens(
        row_tokens,
        tokenizer=tokenizer,
        pad_token_id=pad_token_id,
    )
    if len(sequent.ants) == 0:
        raise ValueError("Prompt antecedents are empty; cannot infer src layer.")
    return int(min(_layer_from_predicate(atom.predicate) for atom in sequent.ants))


def extract_prompt_info_from_row_tokens(
    row_tokens,
    *,
    tokenizer: FOLLayerTokenizer,
    pad_token_id: int = 0,
) -> tuple[np.ndarray, FOLSequent, int, int]:
    prompt_prefix, sequent = _find_sequent_prompt_in_tokens(
        row_tokens,
        tokenizer=tokenizer,
        pad_token_id=pad_token_id,
    )
    if len(sequent.ants) == 0:
        raise ValueError("Prompt antecedents are empty; cannot infer src layer.")
    src_layer = int(min(_layer_from_predicate(atom.predicate) for atom in sequent.ants))
    goal_layer = int(_layer_from_predicate(sequent.cons.predicate))
    return prompt_prefix, sequent, src_layer, goal_layer


def extract_ar_rule_match_inputs(
    *,
    preds,
    labels,
    xs,
    tokenizer: FOLLayerTokenizer,
) -> tuple[list[int], list[np.ndarray], list[str | None]]:
    preds = np.asarray(preds, dtype=np.int32)
    labels = np.asarray(labels, dtype=np.int32)
    xs = np.asarray(xs, dtype=np.int32)

    if preds.shape != labels.shape:
        raise ValueError(f"preds and labels must have same shape, got {preds.shape} and {labels.shape}")
    if xs.ndim != 2 or labels.ndim != 2:
        raise ValueError("Expected 2D xs/labels arrays.")

    src_layers: list[int] = []
    pred_completions: list[np.ndarray] = []
    expected_statements: list[str | None] = []

    for idx in range(labels.shape[0]):
        mask = labels[idx] != 0
        if not np.any(mask):
            continue

        _, _, src_layer, _ = extract_prompt_info_from_row_tokens(xs[idx], tokenizer=tokenizer)
        pred_completion = preds[idx][mask].astype(np.int32)
        gold_completion = labels[idx][mask].astype(np.int32)

        try:
            expected_statement = tokenizer.decode_completion_text(gold_completion.tolist())
        except (ValueError, TypeError):
            expected_statement = None

        src_layers.append(int(src_layer))
        pred_completions.append(pred_completion)
        expected_statements.append(expected_statement)

    return src_layers, pred_completions, expected_statements


def extract_ar_free_run_eval_inputs(
    *,
    xs,
    labels,
    tokenizer: FOLLayerTokenizer,
) -> tuple[list[np.ndarray], list[int], list[FOLAtom], list[int], list[str | None]]:
    xs = np.asarray(xs, dtype=np.int32)
    labels = np.asarray(labels, dtype=np.int32)
    if xs.ndim != 2 or labels.ndim != 2:
        raise ValueError(f"Expected 2D xs/labels arrays, got {xs.shape} and {labels.shape}")
    if xs.shape[0] != labels.shape[0]:
        raise ValueError(f"Batch size mismatch for xs/labels: {xs.shape} and {labels.shape}")

    prompts: list[np.ndarray] = []
    src_layers: list[int] = []
    goals: list[FOLAtom] = []
    goal_layers: list[int] = []
    expected_statements: list[str | None] = []

    for idx in range(xs.shape[0]):
        mask = labels[idx] != 0
        if not np.any(mask):
            continue

        prompt_tokens, sequent, src_layer, goal_layer = extract_prompt_info_from_row_tokens(
            xs[idx],
            tokenizer=tokenizer,
        )
        gold_completion = labels[idx][mask].astype(np.int32)
        try:
            expected_statement = tokenizer.decode_completion_text(gold_completion.tolist())
        except (ValueError, TypeError):
            expected_statement = None

        prompts.append(prompt_tokens.astype(np.int32))
        src_layers.append(int(src_layer))
        goals.append(sequent.cons)
        goal_layers.append(int(goal_layer))
        expected_statements.append(expected_statement)

    return prompts, src_layers, goals, goal_layers, expected_statements


def summarize_rule_match_metrics(metrics: FOLRuleMatchMetrics) -> dict[str, float | int]:
    n_valid = int(sum(int(result.is_valid_rule) for result in metrics.results))
    correct_given_valid_rate = float(metrics.n_correct) / float(n_valid) if n_valid > 0 else 0.0
    valid_rule_rate = float(n_valid) / float(metrics.n_examples) if metrics.n_examples > 0 else 0.0

    return {
        "n_rule_examples": int(metrics.n_examples),
        "n_valid_rule": int(n_valid),
        "n_invalid_rule": int(metrics.n_examples - n_valid),
        "n_correct_rule": int(metrics.n_correct),
        "n_decode_error": int(metrics.n_decode_error),
        "n_unknown_rule_error": int(metrics.n_unknown_rule_error),
        "n_wrong_rule_error": int(metrics.n_wrong_rule_error),
        "valid_rule_rate": valid_rule_rate,
        "invalid_rule_rate": 1.0 - valid_rule_rate,
        "correct_rule_rate": float(metrics.accuracy),
        "correct_given_valid_rate": correct_given_valid_rate,
        "decode_error_rate": float(metrics.decode_error_rate),
        "unknown_rule_error_rate": float(metrics.unknown_rule_error_rate),
        "wrong_rule_error_rate": float(metrics.wrong_rule_error_rate),
    }


def first_transition_mask(src_layers: list[int]) -> np.ndarray:
    return np.asarray([int(src_layer) == 0 for src_layer in src_layers], dtype=bool)


def summarize_first_transition_counts(
    *,
    n_examples: int,
    n_valid: int,
    n_reachable: int,
    n_decode_error: int,
    n_unknown_rule_error: int,
    n_wrong_rule_error: int,
) -> dict[str, int | float]:
    n_examples = int(n_examples)
    n_valid = int(n_valid)
    n_reachable = int(n_reachable)
    n_decode_error = int(n_decode_error)
    n_unknown_rule_error = int(n_unknown_rule_error)
    n_wrong_rule_error = int(n_wrong_rule_error)

    def _rate(n: int) -> float:
        if n_examples <= 0:
            return 0.0
        return float(n) / float(n_examples)

    reachable_given_valid_rate = float(n_reachable) / float(n_valid) if n_valid > 0 else 0.0

    return {
        "first_transition_n_examples": n_examples,
        "first_transition_n_valid_rule": n_valid,
        "first_transition_n_invalid_rule": int(max(0, n_examples - n_valid)),
        "first_transition_n_reachable_rule": n_reachable,
        "first_transition_n_decode_error": n_decode_error,
        "first_transition_n_unknown_rule_error": n_unknown_rule_error,
        "first_transition_n_wrong_rule_error": n_wrong_rule_error,
        "first_transition_rule_valid_rate": _rate(n_valid),
        "first_transition_rule_reachable_rate": _rate(n_reachable),
        "first_transition_rule_reachable_given_valid_rate": float(reachable_given_valid_rate),
        "first_transition_decode_error_rate": _rate(n_decode_error),
        "first_transition_unknown_rule_error_rate": _rate(n_unknown_rule_error),
        "first_transition_wrong_rule_error_rate": _rate(n_wrong_rule_error),
    }


def _facts_key(facts: tuple[FOLAtom, ...]) -> tuple[str, ...]:
    return tuple(sorted(atom.text for atom in facts))


def reachable_goal_exact_steps(
    *,
    rule_bank: FOLRuleBank,
    layer: int,
    facts: tuple[FOLAtom, ...],
    goal: FOLAtom,
    steps_remaining: int,
    max_unify_solutions: int,
    memo: dict[tuple[int, int, tuple[str, ...]], bool],
) -> bool:
    key = (int(layer), int(steps_remaining), _facts_key(facts))
    cached = memo.get(key)
    if cached is not None:
        return bool(cached)

    if int(steps_remaining) == 0:
        out = bool(goal in set(facts))
        memo[key] = out
        return out

    if int(layer) >= int(rule_bank.n_layers) - 1:
        memo[key] = False
        return False

    facts_tuple = tuple(facts)
    for rule in rule_bank.transition_rules(int(layer)):
        substitutions = _find_lhs_substitutions_for_facts(
            lhs=rule.lhs,
            facts=facts_tuple,
            max_solutions=int(max_unify_solutions),
        )
        if not substitutions:
            continue

        for subst in substitutions:
            if not _subst_binds_rhs_variables(rule=rule, subst=subst):
                continue
            next_rule = rule.instantiate(subst)
            next_facts = tuple(next_rule.rhs)
            if reachable_goal_exact_steps(
                rule_bank=rule_bank,
                layer=int(layer) + 1,
                facts=next_facts,
                goal=goal,
                steps_remaining=int(steps_remaining) - 1,
                max_unify_solutions=int(max_unify_solutions),
                memo=memo,
            ):
                memo[key] = True
                return True

    memo[key] = False
    return False


def predicted_rule_reaches_goal(
    *,
    rule_bank: FOLRuleBank,
    matched_rule: FOLLayerRule,
    goal: FOLAtom,
    goal_layer: int,
    max_unify_solutions: int,
) -> bool:
    dst_layer = int(matched_rule.dst_layer)
    remaining = int(goal_layer) - dst_layer
    if remaining < 0:
        return False

    memo: dict[tuple[int, int, tuple[str, ...]], bool] = {}
    return reachable_goal_exact_steps(
        rule_bank=rule_bank,
        layer=dst_layer,
        facts=tuple(matched_rule.rhs),
        goal=goal,
        steps_remaining=remaining,
        max_unify_solutions=int(max_unify_solutions),
        memo=memo,
    )
