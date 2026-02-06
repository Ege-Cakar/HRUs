"""Normative baselines for rule selection on propositional proof states."""

from model.normative.bounded_prover import BoundedProverConfig, BoundedProverPolicy
from model.normative.eval import ChoiceEval, evaluate_policy_on_examples
from model.normative.kernel import KernelConfig, KernelPolicy
from model.normative.loglinear import LogLinearConfig, LogLinearPolicy
from model.normative.proof_features import (
    ACTION_SCALAR_FEATURE_NAMES,
    ChoiceExample,
    RULE_TYPE_COUNT,
    SEQUENT_FEATURE_NAMES,
    build_choice_dataset,
    feature_dim,
)

__all__ = [
    "ACTION_SCALAR_FEATURE_NAMES",
    "BoundedProverConfig",
    "BoundedProverPolicy",
    "ChoiceEval",
    "ChoiceExample",
    "KernelConfig",
    "KernelPolicy",
    "LogLinearConfig",
    "LogLinearPolicy",
    "RULE_TYPE_COUNT",
    "SEQUENT_FEATURE_NAMES",
    "build_choice_dataset",
    "evaluate_policy_on_examples",
    "feature_dim",
]
