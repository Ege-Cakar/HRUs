"""Tests for 6_layer_sweep utility helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "experiment" / "remote" / "6_layer_sweep"))

from data_utils import (
    build_prompt_only_inputs,
    extract_ar_completions,
    first_eot_indices,
    pad_completion_targets,
)
from metrics_utils import (
    extract_ar_rule_match_inputs,
    final_token_accuracy,
    infer_src_layer_from_prompt_tokens,
    last_nonzero_indices,
)

from task.layer_gen.util import tokenize_layer as tok
from task.prop_gen.util.elem import Atom, Sequent


def test_build_prompt_only_inputs_masks_post_sep_tokens() -> None:
    xs = np.array(
        [
            [11, 12, 44, 30, 31, 0, 0],
            [21, 44, 50, 0, 0, 0, 0],
        ],
        dtype=np.int32,
    )

    out = build_prompt_only_inputs(xs, n_seq=8, sep_token_id=44, pad_token_id=0)

    assert out.shape == (2, 8)
    assert np.array_equal(out[0], np.array([11, 12, 44, 0, 0, 0, 0, 0], dtype=np.int32))
    assert np.array_equal(out[1], np.array([21, 44, 0, 0, 0, 0, 0, 0], dtype=np.int32))


def test_build_prompt_only_inputs_requires_sep() -> None:
    xs = np.array([[1, 2, 3]], dtype=np.int32)

    with pytest.raises(ValueError, match="Missing SEP token"):
        build_prompt_only_inputs(xs, n_seq=4, sep_token_id=44)


def test_pad_completion_targets_and_first_eot_indices() -> None:
    ys = np.array(
        [
            [41, 42, 45, 45],
            [38, 45, 45, 45],
        ],
        dtype=np.int32,
    )
    out = pad_completion_targets(ys, max_out_len=6, eot_token_id=45)

    assert out.shape == (2, 6)
    assert np.array_equal(out[0], np.array([41, 42, 45, 45, 45, 45], dtype=np.int32))
    assert np.array_equal(out[1], np.array([38, 45, 45, 45, 45, 45], dtype=np.int32))

    idx = first_eot_indices(out, eot_token_id=45)
    assert np.array_equal(idx, np.array([2, 1], dtype=np.int32))


def test_extract_ar_completions() -> None:
    labels = np.array(
        [
            [0, 0, 10, 11, 12],
            [0, 9, 8, 0, 0],
        ],
        dtype=np.int32,
    )
    comps = extract_ar_completions(labels)
    assert np.array_equal(comps[0], np.array([10, 11, 12], dtype=np.int32))
    assert np.array_equal(comps[1], np.array([9, 8], dtype=np.int32))


def test_last_nonzero_and_final_token_accuracy() -> None:
    labels = np.array(
        [
            [0, 0, 9, 45, 0, 0],
            [0, 0, 0, 7, 8, 45],
        ],
        dtype=np.int32,
    )
    preds = np.array(
        [
            [3, 4, 9, 45, 1, 2],
            [5, 6, 7, 8, 9, 0],
        ],
        dtype=np.int32,
    )

    idx = np.asarray(last_nonzero_indices(labels))
    assert np.array_equal(idx, np.array([3, 5], dtype=np.int32))

    acc = float(final_token_accuracy(preds, labels))
    assert acc == pytest.approx(0.5)


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
