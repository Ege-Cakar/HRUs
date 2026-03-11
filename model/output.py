"""Shared output projection helpers for sequence models."""

from __future__ import annotations

import jax.numpy as jnp


def validate_output_config(output_mode: str, n_pred_tokens: int) -> None:
    if output_mode not in {"last_token", "full_sequence", "last_nonpad"}:
        raise ValueError(
            "output_mode must be one of 'last_token', 'full_sequence', or 'last_nonpad', "
            f"got {output_mode!r}"
        )
    if n_pred_tokens < 1:
        raise ValueError(f"n_pred_tokens must be >= 1, got {n_pred_tokens}")


def select_last_nonpad_hidden(
    hidden: jnp.ndarray,
    tokens: jnp.ndarray,
    pad_token_id: int,
) -> jnp.ndarray:
    is_nonpad = tokens != pad_token_id
    lengths = jnp.sum(is_nonpad, axis=1)
    last_index = jnp.maximum(lengths - 1, 0)
    batch_idx = jnp.arange(hidden.shape[0])
    return hidden[batch_idx, last_index, :]


def apply_output_projection(
    x: jnp.ndarray,
    output_layer,
    *,
    output_mode: str,
    n_pred_tokens: int,
    n_out: int,
    tokens: jnp.ndarray | None = None,
    pad_token_id: int = 0,
) -> jnp.ndarray:
    if output_mode == "full_sequence":
        out = output_layer(x)
        if n_pred_tokens > 1:
            out = out.reshape(out.shape[0], out.shape[1], n_pred_tokens, n_out)
    else:
        if output_mode == "last_nonpad":
            x = select_last_nonpad_hidden(x, tokens=tokens, pad_token_id=pad_token_id)
        else:
            x = x[:, -1, :]
        out = output_layer(x)
        if n_pred_tokens > 1:
            out = out.reshape(out.shape[0], n_pred_tokens, n_out)

    if n_out == 1:
        out = out.squeeze(-1)
    return out
