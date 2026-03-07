"""Tests for layer FOL sweep utility helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from task.layer_fol import FOLRuleMatchMetrics, FOLRuleMatchResult
from task.layer_gen.util import tokenize_layer_fol as tok
from task.layer_gen.util.fol_rule_bank import (
    FOLSequent,
    build_random_fol_rule_bank,
    sample_fol_problem,
)


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiment" / "remote" / "7_layer_fol_sweep" / "utils.py"

_SPEC = importlib.util.spec_from_file_location("layer_fol_sweep_utils", MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

extract_ar_rule_match_inputs = _MODULE.extract_ar_rule_match_inputs
extract_completion_rule_match_inputs = _MODULE.extract_completion_rule_match_inputs
infer_src_layer_from_prompt_tokens = _MODULE.infer_src_layer_from_prompt_tokens
summarize_rule_match_metrics = _MODULE.summarize_rule_match_metrics


def _sampled_case(seed: int = 0):
    rng = np.random.default_rng(seed)
    bank = build_random_fol_rule_bank(
        n_layers=6,
        predicates_per_layer=4,
        rules_per_transition=8,
        arity_max=3,
        vars_per_rule_max=4,
        constants=("a", "b", "c"),
        k_in_max=2,
        k_out_max=2,
        rng=rng,
    )
    tokenizer = tok.build_tokenizer_from_rule_bank(bank)
    sampled = sample_fol_problem(
        bank=bank,
        distance=1,
        initial_ant_max=3,
        rng=rng,
    )
    return tokenizer, sampled


def test_infer_src_layer_from_prompt_tokens_fol() -> None:
    tokenizer, sampled = _sampled_case(seed=4)

    prompt = tokenizer.tokenize_prompt(
        FOLSequent(ants=sampled.step_ants[0], cons=sampled.goal_atom)
    )
    padded = np.array(prompt + [0, 0], dtype=np.int32)

    src_layer = infer_src_layer_from_prompt_tokens(padded, tokenizer=tokenizer)
    assert src_layer == sampled.step_layers[0]


def test_extract_ar_rule_match_inputs_fol() -> None:
    tokenizer, sampled = _sampled_case(seed=8)

    prompt = tokenizer.tokenize_prompt(
        FOLSequent(ants=sampled.step_ants[0], cons=sampled.goal_atom)
    )
    completion = np.array(
        tokenizer.encode_completion_texts([sampled.step_rules[0].statement_text]),
        dtype=np.int32,
    )

    seq_len = len(prompt) + len(completion) - 1
    xs = np.zeros((1, seq_len), dtype=np.int32)
    labels = np.zeros((1, seq_len), dtype=np.int32)
    preds = np.zeros((1, seq_len), dtype=np.int32)

    full = np.concatenate([np.array(prompt, dtype=np.int32), completion], axis=0)
    xs[0] = full[:-1]
    labels[0] = full[1:]
    labels[0, : len(prompt) - 1] = 0
    preds[0] = labels[0]

    src_layers, pred_completions, expected = extract_ar_rule_match_inputs(
        preds=preds,
        labels=labels,
        xs=xs,
        tokenizer=tokenizer,
    )

    assert src_layers == [sampled.step_layers[0]]
    assert np.array_equal(pred_completions[0], completion)
    assert expected == [sampled.step_rules[0].statement_text]


def test_extract_completion_rule_match_inputs_truncates_after_first_eot_fol() -> None:
    tokenizer, sampled = _sampled_case(seed=11)

    prompt = tokenizer.tokenize_prompt(
        FOLSequent(ants=sampled.step_ants[0], cons=sampled.goal_atom)
    )
    completion = np.array(
        tokenizer.encode_completion_texts([sampled.step_rules[0].statement_text]),
        dtype=np.int32,
    )
    eot_token_id = int(tokenizer.eot_token_id)

    xs = np.array([prompt + [0, 0]], dtype=np.int32)
    labels = np.full((1, completion.size + 2), eot_token_id, dtype=np.int32)
    preds = np.full((1, completion.size + 2), eot_token_id, dtype=np.int32)
    labels[0, : completion.size] = completion
    preds[0, : completion.size] = completion
    preds[0, completion.size :] = np.array([17, 18], dtype=np.int32)

    src_layers, pred_completions, expected = extract_completion_rule_match_inputs(
        preds=preds,
        labels=labels,
        xs=xs,
        tokenizer=tokenizer,
        eot_token_id=eot_token_id,
    )

    assert src_layers == [sampled.step_layers[0]]
    assert np.array_equal(pred_completions[0], completion)
    assert expected == [sampled.step_rules[0].statement_text]


def test_summarize_rule_match_metrics_fol() -> None:
    results = (
        FOLRuleMatchResult(
            src_layer=1,
            decoded_statement="r1_1(a) → r2_1(a)",
            expected_statement_text="r1_1(a) → r2_1(a)",
            matched_rule=None,
            decode_error=False,
            unknown_rule_error=False,
            wrong_rule_error=False,
            is_valid_rule=True,
            is_correct=True,
        ),
        FOLRuleMatchResult(
            src_layer=2,
            decoded_statement=None,
            expected_statement_text="r2_1(a) → r3_1(a)",
            matched_rule=None,
            decode_error=False,
            unknown_rule_error=True,
            wrong_rule_error=False,
            is_valid_rule=False,
            is_correct=False,
        ),
    )
    metrics = FOLRuleMatchMetrics(
        n_examples=2,
        n_correct=1,
        n_decode_error=0,
        n_unknown_rule_error=1,
        n_wrong_rule_error=0,
        accuracy=0.5,
        decode_error_rate=0.0,
        unknown_rule_error_rate=0.5,
        wrong_rule_error_rate=0.0,
        results=results,
    )

    summary = summarize_rule_match_metrics(metrics)
    assert summary["n_rule_examples"] == 2
    assert summary["n_valid_rule"] == 1
    assert summary["n_invalid_rule"] == 1
    assert summary["n_correct_rule"] == 1
    assert summary["valid_rule_rate"] == 0.5
    assert summary["invalid_rule_rate"] == 0.5
    assert summary["correct_given_valid_rate"] == 1.0
