"""Tests for SSM scan backend kernels."""

import jax
import jax.numpy as jnp
import numpy as np

from model.ssm_kernels import (
    resolve_scan_backend,
    selective_scan_mamba,
    selective_scan_mamba2,
)


def test_resolve_scan_backend_auto_returns_valid_backend() -> None:
    backend = resolve_scan_backend("auto")
    assert backend in {"reference", "pallas"}


def test_selective_scan_mamba_pallas_request_matches_reference() -> None:
    key = jax.random.PRNGKey(0)
    batch, seq, inner, d_state = 2, 6, 4, 3
    u = jax.random.normal(key, (batch, seq, inner))
    dt = jnp.clip(jax.random.normal(key, (batch, seq, inner)) * 0.1 + 0.2, 1e-3, 0.5)
    b = jax.random.normal(key, (batch, seq, d_state))
    c = jax.random.normal(key, (batch, seq, d_state))
    a = -jnp.exp(jax.random.normal(key, (inner, d_state)))
    d = jax.random.normal(key, (inner,))

    y_ref = selective_scan_mamba(
        u=u,
        dt=dt,
        b=b,
        c=c,
        a=a,
        d=d,
        scan_backend="reference",
    )
    y_req = selective_scan_mamba(
        u=u,
        dt=dt,
        b=b,
        c=c,
        a=a,
        d=d,
        scan_backend="pallas",
    )
    np.testing.assert_allclose(np.asarray(y_req), np.asarray(y_ref), rtol=1e-5, atol=1e-5)


def test_selective_scan_mamba2_pallas_request_matches_reference() -> None:
    key = jax.random.PRNGKey(1)
    batch, seq, n_heads, head_dim, d_state = 2, 5, 2, 3, 4
    inner = n_heads * head_dim

    u = jax.random.normal(key, (batch, seq, inner))
    dt = jnp.clip(jax.random.normal(key, (batch, seq, inner)) * 0.1 + 0.2, 1e-3, 0.5)
    b = jax.random.normal(key, (batch, seq, n_heads * d_state))
    c = jax.random.normal(key, (batch, seq, n_heads * d_state))
    a = -jnp.exp(jax.random.normal(key, (n_heads, d_state)))
    d = jax.random.normal(key, (n_heads, head_dim))

    y_ref = selective_scan_mamba2(
        u=u,
        dt=dt,
        b=b,
        c=c,
        a=a,
        d=d,
        n_heads=n_heads,
        head_dim=head_dim,
        d_state=d_state,
        scan_backend="reference",
    )
    y_req = selective_scan_mamba2(
        u=u,
        dt=dt,
        b=b,
        c=c,
        a=a,
        d=d,
        n_heads=n_heads,
        head_dim=head_dim,
        d_state=d_state,
        scan_backend="pallas",
    )
    np.testing.assert_allclose(np.asarray(y_req), np.asarray(y_ref), rtol=1e-5, atol=1e-5)
