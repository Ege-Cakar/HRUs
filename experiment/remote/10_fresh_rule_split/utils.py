"""Helpers for experiment 10 fresh-rule split evaluation."""

from __future__ import annotations

from task.layer_fol_eval import infer_fol_predicate_layer, predicted_rule_reaches_goal, reachable_goal_exact_steps
from task.layer_fol_eval_inputs import (
    _find_sequent_prompt_in_tokens,
    extract_ar_free_run_eval_inputs,
    extract_ar_rule_match_inputs,
    extract_prompt_info_from_row_tokens,
    first_transition_mask,
    infer_src_layer_from_prompt_tokens,
    summarize_first_transition_counts,
    summarize_rule_match_metrics,
)


def _layer_from_predicate(predicate: str) -> int:
    return infer_fol_predicate_layer(predicate)
