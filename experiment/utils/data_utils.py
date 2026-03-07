"""Shared data reshaping helpers for experiment sweeps."""

from __future__ import annotations

import numpy as np


def _pad_or_trim_right(batch: np.ndarray, target_len: int, pad_value: int) -> np.ndarray:
    """Right-pad or trim a 2D batch to ``target_len``."""
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
    start_token_id: int,
    pad_token_id: int = 0,
) -> np.ndarray:
    """Build fixed-length prompt-only inputs by zeroing tokens after START."""
    out = _pad_or_trim_right(xs, target_len=n_seq, pad_value=pad_token_id)
    start_hits = out == int(start_token_id)
    hit_counts = start_hits.sum(axis=1)
    if not np.all(hit_counts == 1):
        bad_rows = np.where(hit_counts != 1)[0].tolist()
        raise ValueError(f"Expected exactly one START token in rows: {bad_rows}")

    start_idx = start_hits.argmax(axis=1)
    positions = np.arange(int(n_seq))[None, :]
    keep_mask = positions <= start_idx[:, None]
    return np.where(keep_mask, out, int(pad_token_id)).astype(np.int32)


def first_eot_indices(targets, *, eot_token_id: int) -> np.ndarray:
    """Return first EOT index per row for left-aligned EOT-padded targets."""
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


def build_completion_targets(
    labels,
    *,
    max_out_len: int,
    eot_token_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract AR completions and pad with EOT to ``max_out_len``."""
    completions = extract_ar_completions(labels)
    batch_size = len(completions)
    targets = np.full((batch_size, int(max_out_len)), int(eot_token_id), dtype=np.int32)
    lengths = np.zeros((batch_size,), dtype=np.int32)

    for idx, completion in enumerate(completions):
        if completion.size > int(max_out_len):
            raise ValueError(
                f"Completion length {completion.size} exceeds max_out_len={int(max_out_len)}"
            )
        targets[idx, : completion.size] = completion
        lengths[idx] = int(completion.size)

    return targets, lengths
