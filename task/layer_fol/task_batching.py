"""Batch collation and fixed-length helpers for layered FOL tasks."""

from __future__ import annotations

from functools import partial

import numpy as np


def normalize_completions(completions) -> list[np.ndarray]:
    if isinstance(completions, np.ndarray):
        arr = np.asarray(completions, dtype=np.int32)
        if arr.ndim == 1:
            return [arr]
        if arr.ndim == 2:
            return [arr[idx] for idx in range(arr.shape[0])]
        raise ValueError(f"Completions array must be 1D or 2D, got {arr.shape}")

    out: list[np.ndarray] = []
    for completion in completions:
        arr = np.asarray(completion, dtype=np.int32)
        if arr.ndim != 1:
            raise ValueError(f"Completion must be 1D, got {arr.shape}")
        out.append(arr)
    return out


def pad_sequences(arrays: list[np.ndarray], *, pad_value: int = 0) -> np.ndarray:
    max_len = max(arr.shape[0] for arr in arrays)
    out = np.full((len(arrays), max_len), int(pad_value), dtype=np.int32)
    for idx, arr in enumerate(arrays):
        out[idx, : arr.shape[0]] = arr
    return out


def batch_records_autoreg(records):
    if not records:
        return np.zeros((0, 0), dtype=np.int32), np.zeros((0, 0), dtype=np.int32)

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    for rec in records:
        prompt = np.asarray(rec["prompt"], dtype=np.int32)
        if prompt.ndim != 1:
            raise ValueError(f"Prompt must be 1D, got {prompt.shape}")

        completions = normalize_completions(rec["completions"])
        if not completions:
            raise ValueError("Cannot sample from empty completion list.")

        completion = completions[np.random.randint(len(completions))]
        full = np.concatenate([prompt, completion], axis=0)
        if full.shape[0] < 2:
            raise ValueError("Prompt + completion must contain at least 2 tokens.")

        x = full[:-1].copy()
        y = full[1:].copy()
        if prompt.shape[0] > 1:
            y[: prompt.shape[0] - 1] = 0

        xs.append(x)
        ys.append(y)

    return pad_sequences(xs), pad_sequences(ys)


def batch_records_all_at_once(records, *, eot_token_id: int):
    if not records:
        return np.zeros((0, 0), dtype=np.int32), np.zeros((0, 0), dtype=np.int32)

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    for rec in records:
        prompt = np.asarray(rec["prompt"], dtype=np.int32)
        if prompt.ndim != 1:
            raise ValueError(f"Prompt must be 1D, got {prompt.shape}")

        completions = normalize_completions(rec["completions"])
        if not completions:
            raise ValueError("Cannot sample from empty completion list.")

        completion = completions[np.random.randint(len(completions))]
        if completion.shape[0] < 1:
            raise ValueError("Completion must contain at least one token.")
        if int(completion[-1]) != int(eot_token_id):
            raise ValueError(
                "Completion must terminate with EOT token for all-at-once objective."
            )

        xs.append(prompt.copy())
        ys.append(completion.copy())

    return pad_sequences(xs), pad_sequences(ys, pad_value=eot_token_id)


def make_batch_fn(*, prediction_objective: str, tokenizer):
    if str(prediction_objective) == "autoregressive":
        return batch_records_autoreg
    if tokenizer is None:
        raise RuntimeError("All-at-once objective requires tokenizer.")
    return partial(
        batch_records_all_at_once,
        eot_token_id=int(tokenizer.eot_token_id),
    )


def coerce_autoreg_batch(batch) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = batch
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    if xs.ndim != 2 or ys.ndim != 2:
        raise ValueError(
            f"Autoregressive batches must be 2D, got xs={xs.shape}, ys={ys.shape}"
        )
    if xs.shape[0] != ys.shape[0]:
        raise ValueError(
            f"Batch size mismatch for autoregressive batch: xs={xs.shape}, ys={ys.shape}"
        )
    return xs, ys


def pad_autoreg_batch_to_length(batch, *, n_seq: int):
    xs, ys = coerce_autoreg_batch(batch)
    n_seq = int(n_seq)
    if n_seq < 2:
        raise ValueError(f"Autoregressive fixed length must be >= 2, got {n_seq}")
    if xs.shape[1] > n_seq or ys.shape[1] > n_seq:
        raise ValueError(
            "Autoregressive batch sequence exceeds fixed length: "
            f"xs={xs.shape}, ys={ys.shape}, fixed_length_n_seq={n_seq}"
        )

    if xs.shape[1] == n_seq and ys.shape[1] == n_seq:
        return xs, ys

    out_x = np.full((xs.shape[0], n_seq), 0, dtype=xs.dtype)
    out_y = np.full((ys.shape[0], n_seq), 0, dtype=ys.dtype)
    if xs.shape[1] > 0:
        out_x[:, : xs.shape[1]] = xs
    if ys.shape[1] > 0:
        out_y[:, : ys.shape[1]] = ys
    return out_x, out_y


def ceil_pow2(n: int) -> int:
    n = int(n)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()
