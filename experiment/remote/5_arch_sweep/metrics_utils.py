"""Metric helpers for 5_arch_sweep."""

from __future__ import annotations

import jax.numpy as jnp


def last_nonzero_indices(labels: jnp.ndarray) -> jnp.ndarray:
    """Return the last index with nonzero label for each row.

    Labels in the autoregressive implication task use ``0`` for masked/padded
    positions, so the final completion token is the rightmost nonzero token.
    Rows with no nonzero labels are clamped to index 0.
    """
    if labels.ndim != 2:
        raise ValueError(f"Expected labels with shape (batch, seq), got {labels.shape}")

    mask = labels != 0
    pos_idx = jnp.arange(labels.shape[1])[None, :]
    last_idx = jnp.max(jnp.where(mask, pos_idx, -1), axis=1)
    return jnp.maximum(last_idx, 0)


def final_token_accuracy(preds: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Compute accuracy on the final nonzero label token per sequence."""
    if preds.shape != labels.shape:
        raise ValueError(
            f"Predictions and labels must have the same shape, got {preds.shape} and {labels.shape}"
        )
    last_idx = last_nonzero_indices(labels)
    batch_idx = jnp.arange(labels.shape[0])
    return jnp.mean(preds[batch_idx, last_idx] == labels[batch_idx, last_idx])
