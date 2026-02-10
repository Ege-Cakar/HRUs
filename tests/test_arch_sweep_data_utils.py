"""Tests for 5_arch_sweep data utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "experiment" / "remote" / "5_arch_sweep"))

from data_utils import build_completion_targets, build_prompt_only_inputs, first_eot_indices


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


def test_build_completion_targets_left_aligned_and_eot_padded() -> None:
    labels = np.array(
        [
            [0, 0, 32, 33, 45, 0],
            [0, 0, 0, 40, 45, 0],
        ],
        dtype=np.int32,
    )

    targets, lengths = build_completion_targets(labels, max_out_len=6, eot_token_id=45)

    assert targets.shape == (2, 6)
    assert np.array_equal(lengths, np.array([3, 2], dtype=np.int32))
    assert np.array_equal(targets[0], np.array([32, 33, 45, 45, 45, 45], dtype=np.int32))
    assert np.array_equal(targets[1], np.array([40, 45, 45, 45, 45, 45], dtype=np.int32))


def test_first_eot_indices() -> None:
    targets = np.array(
        [
            [32, 33, 45, 45],
            [40, 45, 45, 45],
        ],
        dtype=np.int32,
    )
    idx = first_eot_indices(targets, eot_token_id=45)
    assert np.array_equal(idx, np.array([2, 1], dtype=np.int32))
