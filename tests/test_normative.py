"""Tests for normative baselines and feature extraction."""

from __future__ import annotations

import numpy as np

from common import expected_calibration_error_from_logits, multiclass_nll_from_logits
from model.normative.bounded_prover import BoundedProverPolicy
from model.normative.eval import evaluate_policy_on_examples
from model.normative.kernel import KernelPolicy
from model.normative.loglinear import LogLinearConfig, LogLinearPolicy
from model.normative.proof_features import (
    ChoiceExample,
    action_feature_vector,
    build_choice_dataset,
    build_choice_example,
    feature_dim,
    sequent_feature_vector,
)
from task.prop_gen.util.tokenize import char_to_id


def _tok(sym: str) -> int:
    return char_to_id(sym)


def _sample_tokens() -> np.ndarray:
    # p1 , ( p1 -> p2 ) |- p2
    return np.array(
        [
            _tok("p1"),
            _tok(","),
            _tok("("),
            _tok("p1"),
            _tok("\u2192"),
            _tok("p2"),
            _tok(")"),
            _tok("\u22a2"),
            _tok("p2"),
        ],
        dtype=np.int32,
    )


def test_feature_shapes_and_dim() -> None:
    sequent = _sample_tokens()
    candidates = np.array([[1, 0], [3, 2], [10, 0]], dtype=np.int32)
    seq_feat = sequent_feature_vector(sequent)
    action_feat = action_feature_vector(sequent, candidates[0], candidates)

    assert seq_feat.ndim == 1
    assert action_feat.ndim == 1
    assert action_feat.shape[0] == feature_dim()


def test_build_choice_dataset_from_masked_batch() -> None:
    seq_batch = np.stack([_sample_tokens(), _sample_tokens()], axis=0)
    target_actions = np.array([[3, 2], [1, 0]], dtype=np.int32)
    rule_set_batch = np.array(
        [
            [[1, 0], [3, 2], [0, 0]],
            [[1, 0], [10, 0], [0, 0]],
        ],
        dtype=np.int32,
    )
    rule_set_mask = np.array(
        [
            [True, True, False],
            [True, True, False],
        ],
        dtype=bool,
    )
    examples = build_choice_dataset(
        seq_batch,
        target_actions,
        rule_set_batch,
        rule_set_mask,
        skip_missing=False,
    )

    assert len(examples) == 2
    assert examples[0].target_index == 1
    assert np.array_equal(examples[1].target_action, np.array([1, 0], dtype=np.int32))


def _synthetic_examples(n: int = 40) -> list[ChoiceExample]:
    d = feature_dim()
    candidates = np.array([[1, 0], [2, 0]], dtype=np.int32)
    examples = []
    for idx in range(n):
        sign = 1.0 if idx % 2 == 0 else -1.0
        feats = np.zeros((2, d), dtype=np.float32)
        feats[0, 0] = sign
        feats[1, 0] = -sign
        target_index = 0 if sign > 0 else 1
        examples.append(
            ChoiceExample(
                sequent_tokens=np.array([1, 2, 3], dtype=np.int32),
                action_features=feats,
                target_index=target_index,
                target_action=candidates[target_index].copy(),
                candidate_actions=candidates.copy(),
            )
        )
    return examples


def test_loglinear_learns_simple_choice_rule() -> None:
    train_examples = _synthetic_examples(80)
    test_examples = _synthetic_examples(20)

    policy = LogLinearPolicy(
        LogLinearConfig(lr=0.2, weight_decay=0.0, max_steps=200, batch_size=32, seed=0)
    ).fit(train_examples)
    res = evaluate_policy_on_examples(policy, test_examples, uses_sequent_tokens=False)

    assert res.joint_acc > 0.95
    assert res.rule_acc > 0.95


def test_kernel_and_bounded_proba_are_valid() -> None:
    sequent = _sample_tokens()
    candidates = np.array([[1, 0], [2, 0], [10, 0]], dtype=np.int32)
    ex = build_choice_example(sequent, np.array([1, 0], dtype=np.int32), candidates)
    train_examples = [ex] * 4

    kernel = KernelPolicy().fit(train_examples)
    probs_k = kernel.predict_proba(ex.action_features)
    assert probs_k.shape == (3,)
    assert np.isclose(probs_k.sum(), 1.0)
    assert np.all(probs_k >= 0.0)

    bounded = BoundedProverPolicy().fit(train_examples)
    probs_b = bounded.predict_proba(sequent, candidates)
    assert probs_b.shape == (3,)
    assert np.isclose(probs_b.sum(), 1.0)
    assert np.all(probs_b >= 0.0)


def test_common_calibration_and_nll_helpers() -> None:
    logits = np.array([[8.0, -8.0], [-8.0, 8.0]], dtype=np.float32)
    labels = np.array([0, 1], dtype=np.int32)

    nll = float(multiclass_nll_from_logits(logits, labels))
    ece = float(expected_calibration_error_from_logits(logits, labels))

    assert nll < 1e-3
    assert ece < 1e-3
