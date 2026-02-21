"""Selective scan backends for state-space models.

This module keeps a numerically stable reference path while allowing optional
GPU-specialized kernels via Pallas.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp

try:
    from jax.experimental import pallas as pl
except Exception:  # pragma: no cover - backend-dependent import path
    pl = None


_WARNED: set[str] = set()


def _warn_once(message: str) -> None:
    if message in _WARNED:
        return
    _WARNED.add(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def resolve_scan_backend(scan_backend: str) -> str:
    if scan_backend not in {"reference", "pallas", "auto"}:
        raise ValueError(
            "scan_backend must be one of 'reference', 'pallas', or 'auto', "
            f"got {scan_backend!r}"
        )

    if scan_backend == "reference":
        return "reference"

    can_use_pallas = pl is not None and jax.default_backend() == "gpu"
    if scan_backend == "auto":
        return "pallas" if can_use_pallas else "reference"

    if can_use_pallas:
        return "pallas"

    _warn_once(
        "scan_backend='pallas' requested, but pallas GPU runtime is unavailable. "
        "Falling back to reference scan."
    )
    return "reference"


def selective_scan_mamba(
    *,
    u: jnp.ndarray,
    dt: jnp.ndarray,
    b: jnp.ndarray,
    c: jnp.ndarray,
    a: jnp.ndarray,
    d: jnp.ndarray,
    scan_backend: str,
    scan_chunk_len: int = 64,
    scan_debug_checks: bool = False,
) -> jnp.ndarray:
    if scan_chunk_len < 1:
        raise ValueError(f"scan_chunk_len must be >= 1, got {scan_chunk_len}")

    backend = resolve_scan_backend(scan_backend)
    if backend == "pallas":
        out = _selective_scan_mamba_pallas_with_ref_bwd(u, dt, b, c, a, d)
    else:
        out = _selective_scan_mamba_reference(
            u=u,
            dt=dt,
            b=b,
            c=c,
            a=a,
            d=d,
            scan_chunk_len=scan_chunk_len,
        )

    if scan_debug_checks:
        if out.shape != u.shape:
            raise ValueError(f"Unexpected selective scan shape {out.shape}, expected {u.shape}")
        if not bool(jnp.all(jnp.isfinite(out))):
            raise ValueError("Non-finite values detected in selective scan output.")
    return out


def selective_scan_mamba2(
    *,
    u: jnp.ndarray,
    dt: jnp.ndarray,
    b: jnp.ndarray,
    c: jnp.ndarray,
    a: jnp.ndarray,
    d: jnp.ndarray,
    n_heads: int,
    head_dim: int,
    d_state: int,
    scan_backend: str,
    scan_chunk_len: int = 64,
    scan_debug_checks: bool = False,
) -> jnp.ndarray:
    if scan_chunk_len < 1:
        raise ValueError(f"scan_chunk_len must be >= 1, got {scan_chunk_len}")

    backend = resolve_scan_backend(scan_backend)
    if backend == "pallas":
        u_h = u.reshape(u.shape[0], u.shape[1], n_heads, head_dim)
        dt_h = dt.reshape(dt.shape[0], dt.shape[1], n_heads, head_dim)
        b_h = b.reshape(b.shape[0], b.shape[1], n_heads, d_state)
        c_h = c.reshape(c.shape[0], c.shape[1], n_heads, d_state)
        y_h = _selective_scan_mamba2_pallas_with_ref_bwd(u_h, dt_h, b_h, c_h, a, d)
        out = y_h.reshape(y_h.shape[0], y_h.shape[1], u.shape[-1])
    else:
        out = _selective_scan_mamba2_reference(
            u=u,
            dt=dt,
            b=b,
            c=c,
            a=a,
            d=d,
            n_heads=n_heads,
            head_dim=head_dim,
            d_state=d_state,
            scan_chunk_len=scan_chunk_len,
        )

    if scan_debug_checks:
        if out.shape != u.shape:
            raise ValueError(f"Unexpected selective scan shape {out.shape}, expected {u.shape}")
        if not bool(jnp.all(jnp.isfinite(out))):
            raise ValueError("Non-finite values detected in selective scan output.")
    return out


def _selective_scan_mamba_reference(
    *,
    u: jnp.ndarray,
    dt: jnp.ndarray,
    b: jnp.ndarray,
    c: jnp.ndarray,
    a: jnp.ndarray,
    d: jnp.ndarray,
    scan_chunk_len: int,
) -> jnp.ndarray:
    inner = u.shape[-1]
    d_state = b.shape[-1]
    state0 = jnp.zeros((u.shape[0], inner, d_state), dtype=u.dtype)

    def step(state, inputs):
        u_i, dt_i, b_i, c_i = inputs
        delta_a = jnp.exp(dt_i[..., None] * a[None, :, :])
        delta_b_u = dt_i[..., None] * b_i[:, None, :] * u_i[..., None]
        state = state * delta_a + delta_b_u
        y_i = jnp.sum(state * c_i[:, None, :], axis=-1) + d[None, :] * u_i
        return state, y_i

    state = state0
    ys_chunks = []
    for start in range(0, u.shape[1], scan_chunk_len):
        stop = min(start + scan_chunk_len, u.shape[1])
        u_t = jnp.swapaxes(u[:, start:stop, :], 0, 1)
        dt_t = jnp.swapaxes(dt[:, start:stop, :], 0, 1)
        b_t = jnp.swapaxes(b[:, start:stop, :], 0, 1)
        c_t = jnp.swapaxes(c[:, start:stop, :], 0, 1)
        state, ys_chunk = jax.lax.scan(step, state, (u_t, dt_t, b_t, c_t))
        ys_chunks.append(jnp.swapaxes(ys_chunk, 0, 1))
    return jnp.concatenate(ys_chunks, axis=1)


def _selective_scan_mamba2_reference(
    *,
    u: jnp.ndarray,
    dt: jnp.ndarray,
    b: jnp.ndarray,
    c: jnp.ndarray,
    a: jnp.ndarray,
    d: jnp.ndarray,
    n_heads: int,
    head_dim: int,
    d_state: int,
    scan_chunk_len: int,
) -> jnp.ndarray:
    u_h = u.reshape(u.shape[0], u.shape[1], n_heads, head_dim)
    dt_h = dt.reshape(dt.shape[0], dt.shape[1], n_heads, head_dim)
    b_h = b.reshape(b.shape[0], b.shape[1], n_heads, d_state)
    c_h = c.reshape(c.shape[0], c.shape[1], n_heads, d_state)

    state0 = jnp.zeros(
        (u.shape[0], n_heads, head_dim, d_state),
        dtype=u.dtype,
    )

    def step(state, inputs):
        u_i, dt_i, b_i, c_i = inputs
        delta_a = jnp.exp(dt_i[..., None] * a[None, :, None, :])
        delta_b_u = dt_i[..., None] * b_i[:, :, None, :] * u_i[..., None]
        state = state * delta_a + delta_b_u
        y_i = jnp.sum(state * c_i[:, :, None, :], axis=-1) + d[None, :, :] * u_i
        return state, y_i

    state = state0
    ys_chunks = []
    for start in range(0, u_h.shape[1], scan_chunk_len):
        stop = min(start + scan_chunk_len, u_h.shape[1])
        u_t = jnp.swapaxes(u_h[:, start:stop, :, :], 0, 1)
        dt_t = jnp.swapaxes(dt_h[:, start:stop, :, :], 0, 1)
        b_t = jnp.swapaxes(b_h[:, start:stop, :, :], 0, 1)
        c_t = jnp.swapaxes(c_h[:, start:stop, :, :], 0, 1)
        state, ys_chunk = jax.lax.scan(step, state, (u_t, dt_t, b_t, c_t))
        ys_chunks.append(jnp.swapaxes(ys_chunk, 0, 1))
    ys = jnp.concatenate(ys_chunks, axis=1)
    return ys.reshape(ys.shape[0], ys.shape[1], u.shape[-1])


def _selective_scan_mamba_pallas(
    *,
    u: jnp.ndarray,
    dt: jnp.ndarray,
    b: jnp.ndarray,
    c: jnp.ndarray,
    a: jnp.ndarray,
    d: jnp.ndarray,
) -> jnp.ndarray:
    if pl is None:  # pragma: no cover - guarded by resolve_scan_backend
        return _selective_scan_mamba_reference(
            u=u,
            dt=dt,
            b=b,
            c=c,
            a=a,
            d=d,
            scan_chunk_len=u.shape[1],
        )

    batch, seq_len, inner = u.shape
    d_state = b.shape[-1]
    out_shape = jax.ShapeDtypeStruct(u.shape, u.dtype)

    def kernel(u_ref, dt_ref, b_ref, c_ref, a_ref, d_ref, y_ref):
        b_ix = pl.program_id(0)
        i_ix = pl.program_id(1)

        state = jnp.zeros((d_state,), dtype=u_ref.dtype)
        a_i = a_ref[i_ix, :]
        d_i = d_ref[i_ix]

        for t in range(seq_len):
            u_i = u_ref[b_ix, t, i_ix]
            dt_i = dt_ref[b_ix, t, i_ix]
            b_i = b_ref[b_ix, t, :]
            c_i = c_ref[b_ix, t, :]
            delta_a = jnp.exp(dt_i * a_i)
            state = state * delta_a + dt_i * b_i * u_i
            y_ref[b_ix, t, i_ix] = jnp.sum(state * c_i) + d_i * u_i

    call = pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=(batch, inner),
    )
    return call(u, dt, b, c, a, d)


@jax.custom_vjp
def _selective_scan_mamba_pallas_with_ref_bwd(
    u: jnp.ndarray,
    dt: jnp.ndarray,
    b: jnp.ndarray,
    c: jnp.ndarray,
    a: jnp.ndarray,
    d: jnp.ndarray,
) -> jnp.ndarray:
    return _selective_scan_mamba_pallas(u=u, dt=dt, b=b, c=c, a=a, d=d)


def _selective_scan_mamba_pallas_with_ref_bwd_fwd(
    u: jnp.ndarray,
    dt: jnp.ndarray,
    b: jnp.ndarray,
    c: jnp.ndarray,
    a: jnp.ndarray,
    d: jnp.ndarray,
):
    y = _selective_scan_mamba_pallas(u=u, dt=dt, b=b, c=c, a=a, d=d)
    return y, (u, dt, b, c, a, d)


def _selective_scan_mamba_pallas_with_ref_bwd_bwd(res, g):
    u, dt, b, c, a, d = res

    def _ref_fn(u_, dt_, b_, c_, a_, d_):
        return _selective_scan_mamba_reference(
            u=u_,
            dt=dt_,
            b=b_,
            c=c_,
            a=a_,
            d=d_,
            scan_chunk_len=u_.shape[1],
        )

    _, pullback = jax.vjp(_ref_fn, u, dt, b, c, a, d)
    return pullback(g)


_selective_scan_mamba_pallas_with_ref_bwd.defvjp(
    _selective_scan_mamba_pallas_with_ref_bwd_fwd,
    _selective_scan_mamba_pallas_with_ref_bwd_bwd,
)


def _selective_scan_mamba2_pallas(
    *,
    u_h: jnp.ndarray,
    dt_h: jnp.ndarray,
    b_h: jnp.ndarray,
    c_h: jnp.ndarray,
    a: jnp.ndarray,
    d: jnp.ndarray,
) -> jnp.ndarray:
    if pl is None:  # pragma: no cover - guarded by resolve_scan_backend
        return _selective_scan_mamba2_h_reference(
            u_h=u_h,
            dt_h=dt_h,
            b_h=b_h,
            c_h=c_h,
            a=a,
            d=d,
            scan_chunk_len=u_h.shape[1],
        )

    batch, seq_len, n_heads, head_dim = u_h.shape
    d_state = b_h.shape[-1]
    out_shape = jax.ShapeDtypeStruct(u_h.shape, u_h.dtype)

    def kernel(u_ref, dt_ref, b_ref, c_ref, a_ref, d_ref, y_ref):
        b_ix = pl.program_id(0)
        h_ix = pl.program_id(1)
        hd_ix = pl.program_id(2)

        state = jnp.zeros((d_state,), dtype=u_ref.dtype)
        a_h = a_ref[h_ix, :]
        d_h = d_ref[h_ix, hd_ix]

        for t in range(seq_len):
            u_i = u_ref[b_ix, t, h_ix, hd_ix]
            dt_i = dt_ref[b_ix, t, h_ix, hd_ix]
            b_i = b_ref[b_ix, t, h_ix, :]
            c_i = c_ref[b_ix, t, h_ix, :]
            delta_a = jnp.exp(dt_i * a_h)
            state = state * delta_a + dt_i * b_i * u_i
            y_ref[b_ix, t, h_ix, hd_ix] = jnp.sum(state * c_i) + d_h * u_i

    call = pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=(batch, n_heads, head_dim),
    )
    return call(u_h, dt_h, b_h, c_h, a, d)


def _selective_scan_mamba2_h_reference(
    *,
    u_h: jnp.ndarray,
    dt_h: jnp.ndarray,
    b_h: jnp.ndarray,
    c_h: jnp.ndarray,
    a: jnp.ndarray,
    d: jnp.ndarray,
    scan_chunk_len: int,
) -> jnp.ndarray:
    n_heads = u_h.shape[2]
    head_dim = u_h.shape[3]
    d_state = b_h.shape[-1]
    u = u_h.reshape(u_h.shape[0], u_h.shape[1], n_heads * head_dim)
    dt = dt_h.reshape(dt_h.shape[0], dt_h.shape[1], n_heads * head_dim)
    b = b_h.reshape(b_h.shape[0], b_h.shape[1], n_heads * d_state)
    c = c_h.reshape(c_h.shape[0], c_h.shape[1], n_heads * d_state)
    y = _selective_scan_mamba2_reference(
        u=u,
        dt=dt,
        b=b,
        c=c,
        a=a,
        d=d,
        n_heads=n_heads,
        head_dim=head_dim,
        d_state=d_state,
        scan_chunk_len=scan_chunk_len,
    )
    return y.reshape(u_h.shape)


@jax.custom_vjp
def _selective_scan_mamba2_pallas_with_ref_bwd(
    u_h: jnp.ndarray,
    dt_h: jnp.ndarray,
    b_h: jnp.ndarray,
    c_h: jnp.ndarray,
    a: jnp.ndarray,
    d: jnp.ndarray,
) -> jnp.ndarray:
    return _selective_scan_mamba2_pallas(
        u_h=u_h,
        dt_h=dt_h,
        b_h=b_h,
        c_h=c_h,
        a=a,
        d=d,
    )


def _selective_scan_mamba2_pallas_with_ref_bwd_fwd(
    u_h: jnp.ndarray,
    dt_h: jnp.ndarray,
    b_h: jnp.ndarray,
    c_h: jnp.ndarray,
    a: jnp.ndarray,
    d: jnp.ndarray,
):
    y_h = _selective_scan_mamba2_pallas(
        u_h=u_h,
        dt_h=dt_h,
        b_h=b_h,
        c_h=c_h,
        a=a,
        d=d,
    )
    return y_h, (u_h, dt_h, b_h, c_h, a, d)


def _selective_scan_mamba2_pallas_with_ref_bwd_bwd(res, g):
    u_h, dt_h, b_h, c_h, a, d = res

    def _ref_fn(u_h_, dt_h_, b_h_, c_h_, a_, d_):
        return _selective_scan_mamba2_h_reference(
            u_h=u_h_,
            dt_h=dt_h_,
            b_h=b_h_,
            c_h=c_h_,
            a=a_,
            d=d_,
            scan_chunk_len=u_h_.shape[1],
        )

    _, pullback = jax.vjp(_ref_fn, u_h, dt_h, b_h, c_h, a, d)
    return pullback(g)


_selective_scan_mamba2_pallas_with_ref_bwd.defvjp(
    _selective_scan_mamba2_pallas_with_ref_bwd_fwd,
    _selective_scan_mamba2_pallas_with_ref_bwd_bwd,
)
