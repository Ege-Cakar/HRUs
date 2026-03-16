"""Evaluation and rollout utilities for layered FOL tasks."""

from ._types import (
    FAILURE_DECODE_ERROR,
    FAILURE_GOAL_NOT_REACHED,
    FAILURE_INAPPLICABLE_RULE_ERROR,
    FAILURE_UNKNOWN_RULE_ERROR,
    FAILURE_WRONG_RULE_ERROR,
    FOLCompletionPathMetrics,
    FOLCompletionPathResult,
    FOLCompletionPathStep,
    FOLLayerPredictionAdapter,
    FOLLayerRolloutExample,
    FOLLayerRolloutMetrics,
    FOLLayerRolloutResult,
    FOLLayerRolloutTraceStep,
    FOLRuleMatchMetrics,
    FOLRuleMatchResult,
    infer_fol_predicate_layer,
)
from ._helpers import _safe_rate, _sorted_fol_atoms
from ._unify import _find_instantiation_for_rule
from ._rule_match import match_rule_completion_fol, evaluate_rule_matches_fol
from ._completion_path import validate_completion_path_fol, evaluate_completion_paths_fol
from ._rollout import (
    sample_rollout_examples_fol,
    run_layer_rollout_fol,
    evaluate_layer_rollouts_fol,
)
from ._reachability import reachable_goal_exact_steps, predicted_rule_reaches_goal
from ._parse import (
    parse_rule_statements_for_layer,
    resolve_rule_sets_from_context,
    completion_is_valid_for_layer_fol,
    completion_is_valid_for_layer,
    match_rule_completion,
    evaluate_rule_matches,
    sample_rollout_examples,
    run_layer_rollout,
    evaluate_layer_rollouts,
)

__all__ = [
    "FAILURE_DECODE_ERROR",
    "FAILURE_GOAL_NOT_REACHED",
    "FAILURE_INAPPLICABLE_RULE_ERROR",
    "FAILURE_UNKNOWN_RULE_ERROR",
    "FAILURE_WRONG_RULE_ERROR",
    "FOLCompletionPathMetrics",
    "FOLCompletionPathResult",
    "FOLCompletionPathStep",
    "FOLLayerPredictionAdapter",
    "FOLLayerRolloutExample",
    "FOLLayerRolloutMetrics",
    "FOLLayerRolloutResult",
    "FOLLayerRolloutTraceStep",
    "FOLRuleMatchMetrics",
    "FOLRuleMatchResult",
    "_find_instantiation_for_rule",
    "_safe_rate",
    "_sorted_fol_atoms",
    "completion_is_valid_for_layer",
    "completion_is_valid_for_layer_fol",
    "evaluate_completion_paths_fol",
    "evaluate_layer_rollouts",
    "evaluate_layer_rollouts_fol",
    "evaluate_rule_matches",
    "evaluate_rule_matches_fol",
    "infer_fol_predicate_layer",
    "match_rule_completion",
    "match_rule_completion_fol",
    "parse_rule_statements_for_layer",
    "predicted_rule_reaches_goal",
    "reachable_goal_exact_steps",
    "resolve_rule_sets_from_context",
    "run_layer_rollout",
    "run_layer_rollout_fol",
    "sample_rollout_examples",
    "sample_rollout_examples_fol",
    "validate_completion_path_fol",
]
