"""Tests for layer-sweep local utility helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from task.layer import RuleMatchMetrics, RuleMatchResult
from task.layer_gen.util import tokenize_layer as tok
from task.prop_gen.util.elem import Atom, Sequent


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiment" / "remote" / "6_layer_sweep" / "utils.py"

_SPEC = importlib.util.spec_from_file_location("layer_sweep_utils", MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

extract_ar_rule_match_inputs = _MODULE.extract_ar_rule_match_inputs
extract_completion_rule_match_inputs = _MODULE.extract_completion_rule_match_inputs
infer_src_layer_from_prompt_tokens = _MODULE.infer_src_layer_from_prompt_tokens
summarize_rule_match_metrics = _MODULE.summarize_rule_match_metrics


def test_infer_src_layer_from_prompt_tokens() -> None:
    tokenizer = tok.build_tokenizer_from_atoms(["p2_1", "p2_3", "p5_1"])
    prompt = tokenizer.tokenize_prompt(Sequent([Atom("p2_1"), Atom("p2_3")], Atom("p5_1")))
    padded = np.array(prompt + [0, 0], dtype=np.int32)

    src_layer = infer_src_layer_from_prompt_tokens(padded, tokenizer=tokenizer)
    assert src_layer == 2


def test_extract_ar_rule_match_inputs() -> None:
    tokenizer = tok.build_tokenizer_from_atoms(["p1_1", "p2_1", "p3_1"])

    prompt = tokenizer.tokenize_prompt(Sequent([Atom("p1_1")], Atom("p3_1")))
    completion = np.array(tokenizer.encode_completion("p1_1 → p2_1"), dtype=np.int32)

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

    assert src_layers == [1]
    assert np.array_equal(pred_completions[0], completion)
    assert expected == ["p1_1 → p2_1"]


def test_extract_completion_rule_match_inputs_truncates_after_first_eot() -> None:
    tokenizer = tok.build_tokenizer_from_atoms(["p1_1", "p2_1", "p3_1"])

    prompt = tokenizer.tokenize_prompt(Sequent([Atom("p1_1")], Atom("p3_1")))
    completion = np.array(tokenizer.encode_completion("p1_1 → p2_1"), dtype=np.int32)
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

    assert src_layers == [1]
    assert np.array_equal(pred_completions[0], completion)
    assert expected == ["p1_1 → p2_1"]


def test_extract_completion_rule_match_inputs_uses_none_for_malformed_gold_completion() -> None:
    tokenizer = tok.build_tokenizer_from_atoms(["p1_1", "p2_1", "p3_1"])
    prompt = tokenizer.tokenize_prompt(Sequent([Atom("p1_1")], Atom("p3_1")))
    completion = np.array(tokenizer.encode_completion("p1_1 → p2_1"), dtype=np.int32)
    eot_token_id = int(tokenizer.eot_token_id)

    xs = np.array([prompt + [0, 0]], dtype=np.int32)
    labels = np.full((1, completion.size), int(tokenizer.start_token_id), dtype=np.int32)
    preds = np.full((1, completion.size), eot_token_id, dtype=np.int32)
    labels[0, : completion.size - 1] = completion[:-1]
    preds[0, : completion.size] = completion

    src_layers, pred_completions, expected = extract_completion_rule_match_inputs(
        preds=preds,
        labels=labels,
        xs=xs,
        tokenizer=tokenizer,
        eot_token_id=eot_token_id,
    )

    assert src_layers == [1]
    assert np.array_equal(pred_completions[0], completion)
    assert expected == [None]


def test_summarize_rule_match_metrics() -> None:
    results = (
        RuleMatchResult(
            src_layer=1,
            decoded_statement="p1_1 → p2_1",
            expected_statement_text="p1_1 → p2_1",
            matched_rule=None,
            decode_error=False,
            unknown_rule_error=False,
            wrong_rule_error=False,
            is_valid_rule=True,
            is_correct=True,
        ),
        RuleMatchResult(
            src_layer=2,
            decoded_statement=None,
            expected_statement_text="p2_1 → p3_1",
            matched_rule=None,
            decode_error=False,
            unknown_rule_error=True,
            wrong_rule_error=False,
            is_valid_rule=False,
            is_correct=False,
        ),
    )
    metrics = RuleMatchMetrics(
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
