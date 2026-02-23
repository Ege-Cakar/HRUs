"""Tests for SSM scan backend kernels."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np

import model.ssm_kernels as ssm_kernels
from model.ssm_kernels import (
    _estimate_mamba_naive_smem_bytes,
    _estimate_mamba2_tiled_smem_bytes,
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


def test_selective_scan_mamba2_chunked_pallas_matches_reference() -> None:
    key = jax.random.PRNGKey(7)
    batch, seq, n_heads, head_dim, d_state = 2, 11, 2, 5, 4
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
        scan_chunk_len=3,
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
        scan_chunk_len=3,
    )
    np.testing.assert_allclose(np.asarray(y_req), np.asarray(y_ref), rtol=1e-5, atol=1e-5)


def test_selective_scan_mamba2_pallas_gradients_match_reference() -> None:
    key = jax.random.PRNGKey(9)
    batch, seq, n_heads, head_dim, d_state = 2, 6, 2, 4, 4
    inner = n_heads * head_dim

    u = jax.random.normal(key, (batch, seq, inner))
    dt = jnp.clip(jax.random.normal(key, (batch, seq, inner)) * 0.1 + 0.2, 1e-3, 0.5)
    b = jax.random.normal(key, (batch, seq, n_heads * d_state))
    c = jax.random.normal(key, (batch, seq, n_heads * d_state))
    a = -jnp.exp(jax.random.normal(key, (n_heads, d_state)))
    d = jax.random.normal(key, (n_heads, head_dim))

    def loss_for(backend: str):
        def _loss(u_, dt_, b_, c_):
            y = selective_scan_mamba2(
                u=u_,
                dt=dt_,
                b=b_,
                c=c_,
                a=a,
                d=d,
                n_heads=n_heads,
                head_dim=head_dim,
                d_state=d_state,
                scan_backend=backend,
                scan_chunk_len=3,
            )
            return jnp.mean(y**2)

        return _loss

    grads_ref = jax.grad(loss_for("reference"), argnums=(0, 1, 2, 3))(u, dt, b, c)
    grads_req = jax.grad(loss_for("pallas"), argnums=(0, 1, 2, 3))(u, dt, b, c)

    for g_req, g_ref in zip(grads_req, grads_ref, strict=True):
        np.testing.assert_allclose(np.asarray(g_req), np.asarray(g_ref), rtol=1e-5, atol=1e-5)


def test_selective_scan_mamba2_warns_and_falls_back_on_smem_budget(monkeypatch) -> None:
    key = jax.random.PRNGKey(11)
    batch, seq, n_heads, head_dim, d_state = 2, 8, 2, 4, 4
    inner = n_heads * head_dim

    u = jax.random.normal(key, (batch, seq, inner))
    dt = jnp.clip(jax.random.normal(key, (batch, seq, inner)) * 0.1 + 0.2, 1e-3, 0.5)
    b = jax.random.normal(key, (batch, seq, n_heads * d_state))
    c = jax.random.normal(key, (batch, seq, n_heads * d_state))
    a = -jnp.exp(jax.random.normal(key, (n_heads, d_state)))
    d = jax.random.normal(key, (n_heads, head_dim))

    monkeypatch.setattr(ssm_kernels, "_DEFAULT_MAX_SMEM_BYTES", 1)
    ssm_kernels._WARNED.clear()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
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
            scan_chunk_len=3,
        )

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
        scan_chunk_len=3,
    )
    np.testing.assert_allclose(np.asarray(y_req), np.asarray(y_ref), rtol=1e-5, atol=1e-5)
    assert any("resolved scan_backend='reference'" in str(w.message) for w in caught)


def test_mamba_smem_estimate_matches_tensor_sum_rule() -> None:
    key = jax.random.PRNGKey(2)
    u = jax.random.normal(key, (2, 3, 4))
    dt = jax.random.normal(key, (2, 3, 4))
    b = jax.random.normal(key, (2, 3, 5))
    c = jax.random.normal(key, (2, 3, 5))
    a = jax.random.normal(key, (4, 5))
    d = jax.random.normal(key, (4,))
    est = _estimate_mamba_naive_smem_bytes(u=u, dt=dt, b=b, c=c, a=a, d=d)
    expected = u.nbytes + dt.nbytes + b.nbytes + c.nbytes + a.nbytes + d.nbytes + u.nbytes
    assert est == expected


def test_mamba2_tiled_smem_estimate_increases_with_chunk_len() -> None:
    key = jax.random.PRNGKey(13)
    u_h = jax.random.normal(key, (2, 64, 2, 8))
    dt_h = jax.random.normal(key, (2, 64, 2, 8))
    b_h = jax.random.normal(key, (2, 64, 2, 4))
    c_h = jax.random.normal(key, (2, 64, 2, 4))

    est_small = _estimate_mamba2_tiled_smem_bytes(
        u_h=u_h,
        dt_h=dt_h,
        b_h=b_h,
        c_h=c_h,
        scan_chunk_len=8,
    )
    est_large = _estimate_mamba2_tiled_smem_bytes(
        u_h=u_h,
        dt_h=dt_h,
        b_h=b_h,
        c_h=c_h,
        scan_chunk_len=64,
    )
    assert est_small > 0
    assert est_large > est_small
