"""Layer-sweep-specific utility helpers."""

from __future__ import annotations

import re

import numpy as np

from task.layer import RuleMatchMetrics
from task.layer_gen.util.tokenize_layer import LayerTokenizer
from task.prop_gen.util.elem import Atom


_ATOM_RE = re.compile(r"p(\d+)_(\d+)$")


def infer_src_layer_from_prompt_tokens(
    row_tokens,
    *,
    tokenizer: LayerTokenizer,
    pad_token_id: int = 0,
) -> int:
    row = np.asarray(row_tokens, dtype=np.int32)
    if row.ndim != 1:
        raise ValueError(f"Expected 1D row tokens, got {row.shape}")

    nonpad = row[row != int(pad_token_id)]
    sep_hits = np.where(nonpad == int(tokenizer.sep_token_id))[0]
    if sep_hits.size == 0:
        raise ValueError("Missing SEP token in prompt row.")

    prompt = nonpad[: int(sep_hits[0]) + 1].tolist()
    sequent = tokenizer.decode_prompt([int(tok) for tok in prompt])

    if len(sequent.ants) == 0:
        raise ValueError("Prompt antecedents are empty; cannot infer src layer.")

    layers = []
    for ant in sequent.ants:
        if not isinstance(ant, Atom):
            raise ValueError(f"Expected atom antecedents, got {type(ant).__name__}")
        match = _ATOM_RE.fullmatch(ant.name)
        if match is None:
            raise ValueError(f"Unsupported layered atom name: {ant.name}")
        layers.append(int(match.group(1)))

    return int(min(layers))


def extract_ar_rule_match_inputs(
    *,
    preds,
    labels,
    xs,
    tokenizer: LayerTokenizer,
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

        src_layer = infer_src_layer_from_prompt_tokens(xs[idx], tokenizer=tokenizer)
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


def extract_completion_rule_match_inputs(
    *,
    preds,
    labels,
    xs,
    tokenizer: LayerTokenizer,
    eot_token_id: int,
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
        src_layer = infer_src_layer_from_prompt_tokens(xs[idx], tokenizer=tokenizer)
        pred_completion = _truncate_at_first_eot(preds[idx], eot_token_id=eot_token_id)
        gold_completion = _truncate_at_first_eot(labels[idx], eot_token_id=eot_token_id)

        try:
            expected_statement = tokenizer.decode_completion_text(gold_completion.tolist())
        except (ValueError, TypeError):
            expected_statement = None

        src_layers.append(int(src_layer))
        pred_completions.append(pred_completion.astype(np.int32))
        expected_statements.append(expected_statement)

    return src_layers, pred_completions, expected_statements


def summarize_rule_match_metrics(metrics: RuleMatchMetrics) -> dict[str, float | int]:
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


def _truncate_at_first_eot(tokens, *, eot_token_id: int) -> np.ndarray:
    row = np.asarray(tokens, dtype=np.int32)
    eot_pos = np.where(row == int(eot_token_id))[0]
    if eot_pos.size == 0:
        return row.astype(np.int32)
    return row[: int(eot_pos[0]) + 1].astype(np.int32)

