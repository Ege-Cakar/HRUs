"""Evaluation helpers for normative rule-selection policies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model.normative.proof_features import ChoiceExample


@dataclass(frozen=True)
class ChoiceEval:
    rule_acc: float
    pos_acc: float
    joint_acc: float
    rule_membership_acc: float
    nll: float
    calibration_ece: float
    count: int


def expected_calibration_error(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10,
) -> float:
    conf = np.asarray(confidences, dtype=np.float64).reshape(-1)
    corr = np.asarray(correct).astype(bool).reshape(-1)
    if conf.size == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        low, high = bins[b], bins[b + 1]
        if b == 0:
            mask = (conf >= low) & (conf <= high)
        else:
            mask = (conf > low) & (conf <= high)
        if not np.any(mask):
            continue
        bin_conf = float(np.mean(conf[mask]))
        bin_acc = float(np.mean(corr[mask]))
        ece += abs(bin_acc - bin_conf) * (float(np.sum(mask)) / float(conf.size))
    return float(ece)


def evaluate_predictions(
    pred_actions: np.ndarray,
    target_actions: np.ndarray,
    true_action_probs: np.ndarray,
    pred_confidences: np.ndarray,
) -> ChoiceEval:
    pred = np.asarray(pred_actions, dtype=np.int32).reshape(-1, 2)
    true = np.asarray(target_actions, dtype=np.int32).reshape(-1, 2)
    true_prob = np.asarray(true_action_probs, dtype=np.float64).reshape(-1)
    pred_conf = np.asarray(pred_confidences, dtype=np.float64).reshape(-1)
    if pred.shape != true.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, true={true.shape}.")
    if true_prob.shape[0] != pred.shape[0]:
        raise ValueError("True-probability count does not match predictions.")
    if pred_conf.shape[0] != pred.shape[0]:
        raise ValueError("Pred-confidence count does not match predictions.")

    rule_ok = pred[:, 0] == true[:, 0]
    pos_ok = pred[:, 1] == true[:, 1]
    joint_ok = rule_ok & pos_ok
    ece = expected_calibration_error(pred_conf, joint_ok, n_bins=10)
    return ChoiceEval(
        rule_acc=float(np.mean(rule_ok)),
        pos_acc=float(np.mean(pos_ok)),
        joint_acc=float(np.mean(joint_ok)),
        rule_membership_acc=1.0,
        nll=float(np.mean(-np.log(np.clip(true_prob, 1e-12, 1.0)))),
        calibration_ece=ece,
        count=int(pred.shape[0]),
    )


def evaluate_policy_on_examples(
    policy,
    examples: list[ChoiceExample],
    *,
    uses_sequent_tokens: bool = False,
) -> ChoiceEval:
    if not examples:
        return ChoiceEval(0.0, 0.0, 0.0, 0.0, float("nan"), float("nan"), 0)

    pred_actions = []
    target_actions = []
    true_action_probs = []
    pred_confidences = []

    for ex in examples:
        if uses_sequent_tokens:
            probs = policy.predict_proba(ex.sequent_tokens, ex.candidate_actions)
            pred_idx = int(policy.predict_index(ex.sequent_tokens, ex.candidate_actions))
        else:
            probs = policy.predict_proba(ex.action_features)
            pred_idx = int(policy.predict_index(ex.action_features))
        pred_actions.append(ex.candidate_actions[pred_idx])
        target_actions.append(ex.target_action)
        true_action_probs.append(float(probs[ex.target_index]))
        pred_confidences.append(float(probs[pred_idx]))

    return evaluate_predictions(
        np.asarray(pred_actions, dtype=np.int32),
        np.asarray(target_actions, dtype=np.int32),
        np.asarray(true_action_probs, dtype=np.float64),
        np.asarray(pred_confidences, dtype=np.float64),
    )
