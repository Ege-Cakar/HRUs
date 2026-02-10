"""Utilities for prompt-only Mixer inputs and fixed-length completion targets."""

from __future__ import annotations

import numpy as np


def _pad_or_trim_right(batch: np.ndarray, target_len: int, pad_value: int) -> np.ndarray:
    """Right-pad or trim a 2D batch to ``target_len``."""
    arr = np.asarray(batch, dtype=np.int32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")

    out = np.full((arr.shape[0], target_len), pad_value, dtype=np.int32)
    clipped = min(arr.shape[1], target_len)
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
    """Build fixed-length prompt-only inputs by zeroing tokens after SEP.

    Args:
        xs: Input token batch (batch, seq_in).
        n_seq: Output fixed sequence length.
        sep_token_id: Prompt terminator token id.
        pad_token_id: Padding token id.
    """
    out = _pad_or_trim_right(xs, target_len=n_seq, pad_value=pad_token_id)
    sep_hits = out == sep_token_id
    has_sep = sep_hits.any(axis=1)
    if not np.all(has_sep):
        bad_rows = np.where(~has_sep)[0].tolist()
        raise ValueError(f"Missing SEP token in rows: {bad_rows}")

    sep_idx = sep_hits.argmax(axis=1)
    positions = np.arange(n_seq)[None, :]
    keep_mask = positions <= sep_idx[:, None]
    return np.where(keep_mask, out, pad_token_id).astype(np.int32)


def build_completion_targets(
    labels,
    *,
    max_out_len: int,
    eot_token_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract completion tokens from AR labels and pad with EOT on the right.

    Labels come from ``ImplyAutoregSizeTask`` and contain 0 for masked prompt
    positions. Non-zero labels are the completion token sequence (including EOT).
    """
    ys = np.asarray(labels, dtype=np.int32)
    if ys.ndim != 2:
        raise ValueError(f"Expected 2D labels array, got shape {ys.shape}")

    batch_size = ys.shape[0]
    targets = np.full((batch_size, max_out_len), eot_token_id, dtype=np.int32)
    lengths = np.zeros((batch_size,), dtype=np.int32)

    for idx in range(batch_size):
        completion = ys[idx][ys[idx] != 0]
        if completion.size == 0:
            raise ValueError(f"Encountered empty completion at row {idx}")
        if completion.size > max_out_len:
            raise ValueError(
                f"Completion length {completion.size} exceeds max_out_len={max_out_len}"
            )

        targets[idx, : completion.size] = completion
        lengths[idx] = int(completion.size)

    return targets, lengths


def first_eot_indices(targets, *, eot_token_id: int) -> np.ndarray:
    """Return first EOT index per row for left-aligned EOT-padded targets."""
    ys = np.asarray(targets, dtype=np.int32)
    if ys.ndim != 2:
        raise ValueError(f"Expected 2D targets array, got shape {ys.shape}")

    is_eot = ys == eot_token_id
    has_eot = is_eot.any(axis=1)
    if not np.all(has_eot):
        bad_rows = np.where(~has_eot)[0].tolist()
        raise ValueError(f"Missing EOT token in rows: {bad_rows}")

    return is_eot.argmax(axis=1).astype(np.int32)
