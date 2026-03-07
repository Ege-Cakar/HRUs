"""Layer-FOL ICL sweep utility helpers."""

from __future__ import annotations

import re

import numpy as np

from task.layer_fol import FOLRuleMatchMetrics
from task.layer_gen.util.fol_rule_bank import FOLAtom, FOLSequent
from task.layer_gen.util.tokenize_layer_fol import FOLLayerTokenizer


_PRED_RE = re.compile(r"r(\d+)_(\d+)$")


def _layer_from_predicate(predicate: str) -> int:
    match = _PRED_RE.fullmatch(str(predicate))
    if match is None:
        raise ValueError(f"Unsupported layered predicate name: {predicate}")
    return int(match.group(1))


def _extract_prompt_prefix(
    row_tokens,
    *,
    tokenizer: FOLLayerTokenizer,
    pad_token_id: int = 0,
) -> np.ndarray:
    row = np.asarray(row_tokens, dtype=np.int32)
    if row.ndim != 1:
        raise ValueError(f"Expected 1D row tokens, got {row.shape}")

    nonpad = row[row != int(pad_token_id)]
    start_idx = np.where(nonpad == int(tokenizer.start_token_id))[0]
    if start_idx.size != 1:
        raise ValueError("Prompt rows must contain exactly one START token.")
    return nonpad[: int(start_idx[0]) + 1].astype(np.int32)


def infer_src_layer_from_prompt_tokens(
    row_tokens,
    *,
    tokenizer: FOLLayerTokenizer,
    pad_token_id: int = 0,
) -> int:
    sequent = tokenizer.decode_prompt(
        _extract_prompt_prefix(
            row_tokens,
            tokenizer=tokenizer,
            pad_token_id=pad_token_id,
        ).tolist()
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
    prompt_prefix = _extract_prompt_prefix(
        row_tokens,
        tokenizer=tokenizer,
        pad_token_id=pad_token_id,
    )
    sequent = tokenizer.decode_prompt(prompt_prefix.tolist())
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
            statements = tokenizer.decode_completion_texts(gold_completion.tolist())
            expected_statement = statements[0] if len(statements) == 1 else None
        except (ValueError, TypeError):
            expected_statement = None

        src_layers.append(int(src_layer))
        pred_completions.append(pred_completion)
        expected_statements.append(expected_statement)

    return src_layers, pred_completions, expected_statements


def extract_ar_free_run_inputs(
    *,
    xs,
    tokenizer: FOLLayerTokenizer,
) -> tuple[list[np.ndarray], list[int], list[FOLAtom], list[int]]:
    xs = np.asarray(xs, dtype=np.int32)
    if xs.ndim != 2:
        raise ValueError(f"Expected 2D xs array, got {xs.shape}")

    prompts: list[np.ndarray] = []
    src_layers: list[int] = []
    goals: list[FOLAtom] = []
    goal_layers: list[int] = []

    for idx in range(xs.shape[0]):
        prompt_tokens, sequent, src_layer, goal_layer = extract_prompt_info_from_row_tokens(
            xs[idx],
            tokenizer=tokenizer,
        )
        prompts.append(prompt_tokens.astype(np.int32))
        src_layers.append(int(src_layer))
        goals.append(sequent.cons)
        goal_layers.append(int(goal_layer))

    return prompts, src_layers, goals, goal_layers


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
