"""Shared leaf utilities for layered FOL evaluation."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from task.layer_gen.util import tokenize_layer_fol
from task.layer_gen.util.decode_result import DecodeAttempt
from task.layer_gen.util.fol_rule_bank import FOLAtom, FOLLayerRule, FOLRuleBank


def _safe_rate(num: int, den: int) -> float:
    if den <= 0:
        return 0.0
    return float(num) / float(den)


def _sorted_fol_atoms(atoms: Iterable[FOLAtom]) -> tuple[FOLAtom, ...]:
    return tuple(sorted((atom for atom in atoms), key=lambda atom: atom.text))


def _resolve_fol_tokenizer(
    *,
    rule_bank: FOLRuleBank,
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer | None,
) -> tokenize_layer_fol.FOLLayerTokenizer:
    if tokenizer is not None:
        return tokenizer
    return tokenize_layer_fol.build_tokenizer_from_rule_bank(rule_bank)


def _decode_single_completion_statement(
    *,
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer,
    completion_tokens: list[int] | np.ndarray,
) -> DecodeAttempt[str]:
    decoded = tokenizer.try_decode_completion_texts(
        [int(tok) for tok in np.asarray(completion_tokens, dtype=np.int32).tolist()]
    )
    if not decoded.ok or decoded.value is None:
        return DecodeAttempt.failure(decoded.error or "Unknown decode error.")
    statements = decoded.value
    if len(statements) != 1:
        return DecodeAttempt.failure("Expected a single completion statement.")
    return DecodeAttempt.success(statements[0])


def _expand_rules_by_example(
    by_example: Iterable[Iterable[FOLLayerRule] | None] | None,
    n_expected: int,
    param_name: str,
) -> list[list[FOLLayerRule] | None]:
    """Expand a per-example rules iterable into a fixed-length list, validating length."""
    if by_example is None:
        return [None] * n_expected
    result = [
        None if rules is None else list(rules)
        for rules in by_example
    ]
    if len(result) != n_expected:
        raise ValueError(
            f"{param_name} must match length {n_expected}, got {len(result)}"
        )
    return result
