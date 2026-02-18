"""Utilities for LayerTask prompt/completion reshaping."""

from __future__ import annotations

import numpy as np


def _pad_or_trim_right(batch: np.ndarray, target_len: int, pad_value: int) -> np.ndarray:
    arr = np.asarray(batch, dtype=np.int32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")

    out = np.full((arr.shape[0], int(target_len)), int(pad_value), dtype=np.int32)
    clipped = min(arr.shape[1], int(target_len))
    if clipped > 0:
        out[:, :clipped] = arr[:, :clipped]
    return out


def build_prompt_only_inputs(
    xs,
    *,
    n_seq: int,
    sep_token_id: int,
    pad_token_id: int = 0,
) -> np.ndarray:
    """Build fixed-length prompt-only inputs by zeroing tokens after SEP."""
    out = _pad_or_trim_right(xs, target_len=n_seq, pad_value=pad_token_id)
    sep_hits = out == int(sep_token_id)
    has_sep = sep_hits.any(axis=1)
    if not np.all(has_sep):
        bad_rows = np.where(~has_sep)[0].tolist()
        raise ValueError(f"Missing SEP token in rows: {bad_rows}")

    sep_idx = sep_hits.argmax(axis=1)
    positions = np.arange(int(n_seq))[None, :]
    keep_mask = positions <= sep_idx[:, None]
    return np.where(keep_mask, out, int(pad_token_id)).astype(np.int32)


def pad_completion_targets(
    targets,
    *,
    max_out_len: int,
    eot_token_id: int,
) -> np.ndarray:
    """Right-pad completion rows to a fixed output length with EOT."""
    ys = np.asarray(targets, dtype=np.int32)
    if ys.ndim != 2:
        raise ValueError(f"Expected 2D targets array, got shape {ys.shape}")

    out = np.full((ys.shape[0], int(max_out_len)), int(eot_token_id), dtype=np.int32)
    clipped = min(ys.shape[1], int(max_out_len))
    if clipped > 0:
        out[:, :clipped] = ys[:, :clipped]

    for idx in range(ys.shape[0]):
        row = ys[idx]
        eot_pos = np.where(row == int(eot_token_id))[0]
        if eot_pos.size == 0:
            raise ValueError(f"Missing EOT in target row {idx}")
        if int(eot_pos[0]) >= int(max_out_len):
            raise ValueError(
                f"First EOT at position {int(eot_pos[0])} exceeds max_out_len={int(max_out_len)}"
            )

    return out


def first_eot_indices(targets, *, eot_token_id: int) -> np.ndarray:
    ys = np.asarray(targets, dtype=np.int32)
    if ys.ndim != 2:
        raise ValueError(f"Expected 2D targets array, got shape {ys.shape}")

    is_eot = ys == int(eot_token_id)
    has_eot = is_eot.any(axis=1)
    if not np.all(has_eot):
        bad_rows = np.where(~has_eot)[0].tolist()
        raise ValueError(f"Missing EOT token in rows: {bad_rows}")

    return is_eot.argmax(axis=1).astype(np.int32)


def extract_ar_completions(labels) -> list[np.ndarray]:
    """Extract nonzero completion tokens from autoregressive label rows."""
    ys = np.asarray(labels, dtype=np.int32)
    if ys.ndim != 2:
        raise ValueError(f"Expected 2D labels array, got shape {ys.shape}")

    out = []
    for idx in range(ys.shape[0]):
        completion = ys[idx][ys[idx] != 0]
        if completion.size == 0:
            raise ValueError(f"Encountered empty completion at row {idx}")
        out.append(completion.astype(np.int32))
    return out
