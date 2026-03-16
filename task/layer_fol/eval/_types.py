"""Type definitions and constants for layered FOL evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Protocol

import numpy as np

from task.layer_gen.util import tokenize_layer_fol
from task.layer_gen.util.fol_rule_bank import FOLAtom, FOLLayerRule


class FOLLayerPredictionAdapter(Protocol):
    def predict_completion(
        self,
        *,
        model: Callable[[np.ndarray], Any],
        prompt_tokens: list[int] | np.ndarray,
        tokenizer: tokenize_layer_fol.FOLLayerTokenizer,
        temperature: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class FOLRuleMatchResult:
    src_layer: int
    decoded_statement: str | None
    expected_statement_text: str | None
    matched_rule: FOLLayerRule | None
    decode_error: bool
    unknown_rule_error: bool
    wrong_rule_error: bool
    is_valid_rule: bool
    is_correct: bool
    match_source: str | None = None


@dataclass(frozen=True)
class FOLRuleMatchMetrics:
    n_examples: int
    n_correct: int
    n_decode_error: int
    n_unknown_rule_error: int
    n_wrong_rule_error: int
    accuracy: float
    decode_error_rate: float
    unknown_rule_error_rate: float
    wrong_rule_error_rate: float
    results: tuple[FOLRuleMatchResult, ...]


@dataclass(frozen=True)
class FOLCompletionPathStep:
    step_idx: int
    src_layer: int
    decoded_statement: str | None
    matched_rule_statement: str | None
    decode_error: bool
    unknown_rule_error: bool
    inapplicable_rule_error: bool
    goal_reached: bool


@dataclass(frozen=True)
class FOLCompletionPathResult:
    success: bool
    failure_reason: str | None
    n_steps: int
    goal_reached: bool
    final_layer: int
    final_facts: tuple[str, ...]
    steps: tuple[FOLCompletionPathStep, ...]


@dataclass(frozen=True)
class FOLCompletionPathMetrics:
    n_examples: int
    n_success: int
    n_failure_decode_error: int
    n_failure_unknown_rule_error: int
    n_failure_inapplicable_rule_error: int
    n_failure_goal_not_reached: int
    success_rate: float
    decode_error_rate: float
    unknown_rule_error_rate: float
    inapplicable_rule_error_rate: float
    goal_not_reached_rate: float
    avg_steps: float
    results: tuple[FOLCompletionPathResult, ...]


@dataclass(frozen=True)
class FOLLayerRolloutExample:
    distance: int
    start_layer: int
    goal_atom: str
    initial_ants: tuple[str, ...]
    max_steps: int
    oracle_rule_statements: tuple[str, ...] = ()


@dataclass(frozen=True)
class FOLLayerRolloutTraceStep:
    step_idx: int
    src_layer: int
    prompt_tokens: tuple[int, ...]
    completion_tokens: tuple[int, ...]
    decoded_statement: str | None
    matched_rule_statement: str | None
    decode_error: bool
    unknown_rule_error: bool
    inapplicable_rule_error: bool
    goal_reached: bool


@dataclass(frozen=True)
class FOLLayerRolloutResult:
    success: bool
    failure_reason: str | None
    n_steps: int
    goal_reached: bool
    final_layer: int
    final_facts: tuple[str, ...]
    steps: tuple[FOLLayerRolloutTraceStep, ...]
    example: FOLLayerRolloutExample


@dataclass(frozen=True)
class FOLLayerRolloutMetrics:
    n_examples: int
    n_success: int
    n_failure_decode_error: int
    n_failure_unknown_rule_error: int
    n_failure_wrong_rule_error: int
    n_failure_inapplicable_rule_error: int
    n_failure_goal_not_reached: int
    success_rate: float
    decode_error_rate: float
    unknown_rule_error_rate: float
    wrong_rule_error_rate: float
    inapplicable_rule_error_rate: float
    goal_not_reached_rate: float
    avg_steps: float
    results: tuple[FOLLayerRolloutResult, ...]


FAILURE_DECODE_ERROR = "decode_error"
FAILURE_UNKNOWN_RULE_ERROR = "unknown_rule_error"
FAILURE_WRONG_RULE_ERROR = "wrong_rule_error"
FAILURE_INAPPLICABLE_RULE_ERROR = "inapplicable_rule_error"
FAILURE_GOAL_NOT_REACHED = "goal_not_reached"


_LAYERED_PREDICATE_RE = re.compile(r"r(\d+)_(\d+)$")
_FRESH_PREDICATE_RE = re.compile(r"r_[a-z0-9]+$")


def infer_fol_predicate_layer(predicate: str) -> int:
    """Infer logical layer from a layered or fresh predicate identifier."""
    token = str(predicate)
    match = _LAYERED_PREDICATE_RE.fullmatch(token)
    if match is not None:
        return int(match.group(1))
    if _FRESH_PREDICATE_RE.fullmatch(token):
        return 0
    raise ValueError(f"Unsupported layered predicate name: {predicate}")
