"""State space sequence models using Flax NNX."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx

from model.mup import MuReadout
from model.ssm_kernels import selective_scan_mamba, selective_scan_mamba2

# TODO: all mamba models are extremely inefficient

def _resolve_dt_rank(dt_rank: int | str, n_hidden: int) -> int:
    if dt_rank == "auto":
        return max(1, (n_hidden + 15) // 16)
    if isinstance(dt_rank, int) and dt_rank >= 1:
        return dt_rank
    raise ValueError(f"dt_rank must be 'auto' or int >= 1, got {dt_rank!r}")


def _validate_output_config(output_mode: str, n_pred_tokens: int) -> None:
    if output_mode not in {"last_token", "full_sequence", "last_nonpad"}:
        raise ValueError(
            "output_mode must be one of 'last_token', 'full_sequence', or 'last_nonpad', "
            f"got {output_mode!r}"
        )
    if n_pred_tokens < 1:
        raise ValueError(f"n_pred_tokens must be >= 1, got {n_pred_tokens}")


def _validate_scan_config(scan_backend: str, scan_chunk_len: int) -> None:
    if scan_backend not in {"reference", "pallas", "auto"}:
        raise ValueError(
            "scan_backend must be one of 'reference', 'pallas', or 'auto', "
            f"got {scan_backend!r}"
        )
    if scan_chunk_len < 1:
        raise ValueError(f"scan_chunk_len must be >= 1, got {scan_chunk_len}")


def _select_last_nonpad_hidden(
    hidden: jnp.ndarray,
    tokens: jnp.ndarray,
    pad_token_id: int,
) -> jnp.ndarray:
    is_nonpad = tokens != pad_token_id
    lengths = jnp.sum(is_nonpad, axis=1)
    last_index = jnp.maximum(lengths - 1, 0)
    batch_idx = jnp.arange(hidden.shape[0])
    return hidden[batch_idx, last_index, :]


def _causal_depthwise_conv1d(
    x: jnp.ndarray,
    kernel: jnp.ndarray,
    bias: jnp.ndarray | None,
) -> jnp.ndarray:
    """Depthwise causal Conv1D with NWC layout."""
    channels = x.shape[-1]
    rhs = kernel[:, None, :]
    out = jax.lax.conv_general_dilated(
        lhs=x,
        rhs=rhs,
        window_strides=(1,),
        padding=[(kernel.shape[0] - 1, 0)],
        dimension_numbers=("NWC", "WIO", "NWC"),
        feature_group_count=channels,
    )
    if bias is not None:
        out = out + bias[None, None, :]
    return out


@dataclass
class MambaConfig:
    """Global hyperparameters for a vanilla Mamba-style SSM."""

    n_vocab: int | None = None
    n_seq: int = 128
    n_layers: int = 2
    n_hidden: int = 128
    n_out: int = 1
    n_pred_tokens: int = 1
    output_mode: str = "last_token"
    pad_token_id: int = 0
    layer_norm: bool = True
    use_bias: bool = True
    dropout_rate: float = 0.0
    use_mup: bool = False
    d_state: int = 16
    expand: int = 2
    dt_rank: int | str = "auto"
    d_conv: int = 4
    dt_min: float = 1e-3
    dt_max: float = 0.1
    scan_backend: str = "reference"
    scan_chunk_len: int = 64
    scan_debug_checks: bool = False

    def to_model(self, *, rngs: nnx.Rngs) -> "Mamba":
        return Mamba(self, rngs=rngs)


@dataclass
class Mamba2Config:
    """Global hyperparameters for a vanilla Mamba-2-style SSM."""

    n_vocab: int | None = None
    n_seq: int = 128
    n_layers: int = 2
    n_hidden: int = 128
    n_heads: int = 8
    n_out: int = 1
    n_pred_tokens: int = 1
    output_mode: str = "last_token"
    pad_token_id: int = 0
    layer_norm: bool = True
    use_bias: bool = True
    dropout_rate: float = 0.0
    use_mup: bool = False
    d_state: int = 16
    expand: int = 2
    dt_rank: int | str = "auto"
    d_conv: int = 4
    dt_min: float = 1e-3
    dt_max: float = 0.1
    scan_backend: str = "reference"
    scan_chunk_len: int = 64
    scan_debug_checks: bool = False

    def to_model(self, *, rngs: nnx.Rngs) -> "Mamba2":
        return Mamba2(self, rngs=rngs)


class MambaBlock(nnx.Module):
    """Minimal Mamba-style block using pure JAX scan."""

    def __init__(self, config: MambaConfig, *, rngs: nnx.Rngs):
        if config.expand < 1:
            raise ValueError(f"expand must be >= 1, got {config.expand}")
        if config.d_state < 1:
            raise ValueError(f"d_state must be >= 1, got {config.d_state}")
        if config.d_conv < 1:
            raise ValueError(f"d_conv must be >= 1, got {config.d_conv}")
        if config.dt_min <= 0 or config.dt_max <= 0 or config.dt_min > config.dt_max:
            raise ValueError(
                f"Invalid dt range: dt_min={config.dt_min}, dt_max={config.dt_max}"
            )
        _validate_scan_config(config.scan_backend, config.scan_chunk_len)

        self.n_hidden = config.n_hidden
        self.d_state = config.d_state
        self.inner = config.expand * config.n_hidden
        self.dt_min = config.dt_min
        self.dt_max = config.dt_max
        self.scan_backend = config.scan_backend
        self.scan_chunk_len = int(config.scan_chunk_len)
        self.scan_debug_checks = bool(config.scan_debug_checks)

        self.norm = nnx.RMSNorm(config.n_hidden, rngs=rngs) if config.layer_norm else None
        self.in_proj = nnx.Linear(
            config.n_hidden,
            2 * self.inner,
            use_bias=config.use_bias,
            rngs=rngs,
        )
        self.out_proj = nnx.Linear(
            self.inner,
            config.n_hidden,
            use_bias=config.use_bias,
            rngs=rngs,
        )

        self.dt_in = nnx.Linear(
            self.inner,
            _resolve_dt_rank(config.dt_rank, config.n_hidden),
            use_bias=config.use_bias,
            rngs=rngs,
        )
        self.dt_out = nnx.Linear(
            _resolve_dt_rank(config.dt_rank, config.n_hidden),
            self.inner,
            use_bias=config.use_bias,
            rngs=rngs,
        )
        self.b_proj = nnx.Linear(self.inner, self.d_state, use_bias=config.use_bias, rngs=rngs)
        self.c_proj = nnx.Linear(self.inner, self.d_state, use_bias=config.use_bias, rngs=rngs)

        conv_std = 1.0 / jnp.sqrt(float(config.d_conv))
        self.conv_kernel = nnx.Param(
            jax.random.normal(rngs.params(), (config.d_conv, self.inner), dtype=jnp.float32) * conv_std
        )
        self.conv_bias = nnx.Param(jnp.zeros((self.inner,), dtype=jnp.float32)) if config.use_bias else None

        a_init = jnp.log(jnp.arange(1, self.d_state + 1, dtype=jnp.float32))
        self.a_log = nnx.Param(jnp.tile(a_init[None, :], (self.inner, 1)))
        self.d = nnx.Param(jnp.ones((self.inner,), dtype=jnp.float32))
        self.dt_bias = nnx.Param(jnp.zeros((self.inner,), dtype=jnp.float32))

        self.dropout = nnx.Dropout(rate=config.dropout_rate, rngs=rngs) if config.dropout_rate > 0 else None

    def _scan(self, u, dt, b, c):
        a = -jnp.exp(self.a_log[...])
        d = self.d[...]
        return selective_scan_mamba(
            u=u,
            dt=dt,
            b=b,
            c=c,
            a=a,
            d=d,
            scan_backend=self.scan_backend,
            scan_chunk_len=self.scan_chunk_len,
            scan_debug_checks=self.scan_debug_checks,
        )

    def __call__(self, x):
        residual = x
        if self.norm is not None:
            x = self.norm(x)

        u, z = jnp.split(self.in_proj(x), 2, axis=-1)
        conv_bias = self.conv_bias[...] if self.conv_bias is not None else None
        u = _causal_depthwise_conv1d(u, self.conv_kernel[...], conv_bias)
        u = jax.nn.silu(u)

        dt = self.dt_out(self.dt_in(u)) + self.dt_bias[...]
        dt = jax.nn.softplus(dt)
        dt = jnp.clip(dt, self.dt_min, self.dt_max)
        b = jnp.tanh(self.b_proj(u))
        c = jnp.tanh(self.c_proj(u))

        y = self._scan(u, dt, b, c)
        y = y * jax.nn.silu(z)
        y = self.out_proj(y)
        if self.dropout is not None:
            y = self.dropout(y)
        return residual + y


class Mamba2Block(nnx.Module):
    """Minimal Mamba-2-style block with multi-head SSM state updates."""

    def __init__(self, config: Mamba2Config, *, rngs: nnx.Rngs):
        if config.n_hidden % config.n_heads != 0:
            raise ValueError(
                f"n_hidden ({config.n_hidden}) must be divisible by n_heads ({config.n_heads})"
            )
        if config.expand < 1:
            raise ValueError(f"expand must be >= 1, got {config.expand}")
        if config.d_state < 1:
            raise ValueError(f"d_state must be >= 1, got {config.d_state}")
        if config.d_conv < 1:
            raise ValueError(f"d_conv must be >= 1, got {config.d_conv}")
        if config.dt_min <= 0 or config.dt_max <= 0 or config.dt_min > config.dt_max:
            raise ValueError(
                f"Invalid dt range: dt_min={config.dt_min}, dt_max={config.dt_max}"
            )
        _validate_scan_config(config.scan_backend, config.scan_chunk_len)

        self.n_hidden = config.n_hidden
        self.n_heads = config.n_heads
        self.d_state = config.d_state
        self.inner = config.expand * config.n_hidden
        if self.inner % config.n_heads != 0:
            raise ValueError(
                f"expand * n_hidden ({self.inner}) must be divisible by n_heads ({config.n_heads})"
            )
        self.head_dim = self.inner // config.n_heads
        self.dt_min = config.dt_min
        self.dt_max = config.dt_max
        self.scan_backend = config.scan_backend
        self.scan_chunk_len = int(config.scan_chunk_len)
        self.scan_debug_checks = bool(config.scan_debug_checks)

        self.norm = nnx.RMSNorm(config.n_hidden, rngs=rngs) if config.layer_norm else None
        self.in_proj = nnx.Linear(
            config.n_hidden,
            2 * self.inner,
            use_bias=config.use_bias,
            rngs=rngs,
        )
        self.out_proj = nnx.Linear(
            self.inner,
            config.n_hidden,
            use_bias=config.use_bias,
            rngs=rngs,
        )

        self.dt_in = nnx.Linear(
            self.inner,
            _resolve_dt_rank(config.dt_rank, config.n_hidden),
            use_bias=config.use_bias,
            rngs=rngs,
        )
        self.dt_out = nnx.Linear(
            _resolve_dt_rank(config.dt_rank, config.n_hidden),
            self.inner,
            use_bias=config.use_bias,
            rngs=rngs,
        )
        self.b_proj = nnx.Linear(
            self.inner,
            self.n_heads * self.d_state,
            use_bias=config.use_bias,
            rngs=rngs,
        )
        self.c_proj = nnx.Linear(
            self.inner,
            self.n_heads * self.d_state,
            use_bias=config.use_bias,
            rngs=rngs,
        )

        conv_std = 1.0 / jnp.sqrt(float(config.d_conv))
        self.conv_kernel = nnx.Param(
            jax.random.normal(rngs.params(), (config.d_conv, self.inner), dtype=jnp.float32) * conv_std
        )
        self.conv_bias = nnx.Param(jnp.zeros((self.inner,), dtype=jnp.float32)) if config.use_bias else None

        a_init = jnp.log(jnp.arange(1, self.d_state + 1, dtype=jnp.float32))
        self.a_log = nnx.Param(jnp.tile(a_init[None, :], (self.n_heads, 1)))
        self.d = nnx.Param(jnp.ones((self.inner,), dtype=jnp.float32))
        self.dt_bias = nnx.Param(jnp.zeros((self.inner,), dtype=jnp.float32))

        self.dropout = nnx.Dropout(rate=config.dropout_rate, rngs=rngs) if config.dropout_rate > 0 else None

    def _scan(self, u, dt, b, c):
        a = -jnp.exp(self.a_log[...])
        d = self.d[...].reshape(self.n_heads, self.head_dim)
        return selective_scan_mamba2(
            u=u,
            dt=dt,
            b=b,
            c=c,
            a=a,
            d=d,
            n_heads=self.n_heads,
            head_dim=self.head_dim,
            d_state=self.d_state,
            scan_backend=self.scan_backend,
            scan_chunk_len=self.scan_chunk_len,
            scan_debug_checks=self.scan_debug_checks,
        )

    def __call__(self, x):
        residual = x
        if self.norm is not None:
            x = self.norm(x)

        u, z = jnp.split(self.in_proj(x), 2, axis=-1)
        conv_bias = self.conv_bias[...] if self.conv_bias is not None else None
        u = _causal_depthwise_conv1d(u, self.conv_kernel[...], conv_bias)
        u = jax.nn.silu(u)

        dt = self.dt_out(self.dt_in(u)) + self.dt_bias[...]
        dt = jax.nn.softplus(dt)
        dt = jnp.clip(dt, self.dt_min, self.dt_max)
        b = jnp.tanh(self.b_proj(u))
        c = jnp.tanh(self.c_proj(u))

        y = self._scan(u, dt, b, c)
        y = y * jax.nn.silu(z)
        y = self.out_proj(y)
        if self.dropout is not None:
            y = self.dropout(y)
        return residual + y


class Mamba(nnx.Module):
    """Stacked Mamba blocks with Transformer-compatible output modes."""

    def __init__(self, config: MambaConfig, *, rngs: nnx.Rngs):
        _validate_output_config(config.output_mode, config.n_pred_tokens)
        self.config = config

        if config.n_vocab is not None:
            self.embed = nnx.Embed(config.n_vocab, config.n_hidden, rngs=rngs)
        else:
            self.embed = None

        self.blocks = nnx.List([MambaBlock(config, rngs=rngs) for _ in range(config.n_layers)])
        self.final_ln = nnx.RMSNorm(config.n_hidden, rngs=rngs) if config.layer_norm else None

        out_features = config.n_out * config.n_pred_tokens
        if config.use_mup:
            self.output = MuReadout(
                config.n_hidden,
                out_features,
                use_bias=config.use_bias,
                rngs=rngs,
            )
        else:
            self.output = nnx.Linear(config.n_hidden, out_features, use_bias=config.use_bias, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        config = self.config
        tokens = None

        if self.embed is not None:
            assert x.ndim == 2, f"Expected 2D input (batch, seq) for token indices, got {x.shape}"
            tokens = x
            x = self.embed(x)
        else:
            assert x.ndim == 3, f"Expected 3D input (batch, seq, features), got {x.shape}"
            if config.output_mode == "last_nonpad":
                raise ValueError("output_mode='last_nonpad' requires token indices (n_vocab must be set)")

        for block in self.blocks:
            x = block(x)
        if self.final_ln is not None:
            x = self.final_ln(x)

        if config.output_mode == "full_sequence":
            out = self.output(x)
            if config.n_pred_tokens > 1:
                out = out.reshape(out.shape[0], out.shape[1], config.n_pred_tokens, config.n_out)
        else:
            if config.output_mode == "last_nonpad":
                x = _select_last_nonpad_hidden(x, tokens=tokens, pad_token_id=config.pad_token_id)
            else:
                x = x[:, -1, :]
            out = self.output(x)
            if config.n_pred_tokens > 1:
                out = out.reshape(out.shape[0], config.n_pred_tokens, config.n_out)

        if config.n_out == 1:
            out = out.squeeze(-1)
        return out


class Mamba2(nnx.Module):
    """Stacked Mamba-2 blocks with Transformer-compatible output modes."""

    def __init__(self, config: Mamba2Config, *, rngs: nnx.Rngs):
        _validate_output_config(config.output_mode, config.n_pred_tokens)
        if config.n_hidden % config.n_heads != 0:
            raise ValueError(
                f"n_hidden ({config.n_hidden}) must be divisible by n_heads ({config.n_heads})"
            )
        self.config = config

        if config.n_vocab is not None:
            self.embed = nnx.Embed(config.n_vocab, config.n_hidden, rngs=rngs)
        else:
            self.embed = None

        self.blocks = nnx.List([Mamba2Block(config, rngs=rngs) for _ in range(config.n_layers)])
        self.final_ln = nnx.RMSNorm(config.n_hidden, rngs=rngs) if config.layer_norm else None

        out_features = config.n_out * config.n_pred_tokens
        if config.use_mup:
            self.output = MuReadout(
                config.n_hidden,
                out_features,
                use_bias=config.use_bias,
                rngs=rngs,
            )
        else:
            self.output = nnx.Linear(config.n_hidden, out_features, use_bias=config.use_bias, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        config = self.config
        tokens = None

        if self.embed is not None:
            assert x.ndim == 2, f"Expected 2D input (batch, seq) for token indices, got {x.shape}"
            tokens = x
            x = self.embed(x)
        else:
            assert x.ndim == 3, f"Expected 3D input (batch, seq, features), got {x.shape}"
            if config.output_mode == "last_nonpad":
                raise ValueError("output_mode='last_nonpad' requires token indices (n_vocab must be set)")

        for block in self.blocks:
            x = block(x)
        if self.final_ln is not None:
            x = self.final_ln(x)

        if config.output_mode == "full_sequence":
            out = self.output(x)
            if config.n_pred_tokens > 1:
                out = out.reshape(out.shape[0], out.shape[1], config.n_pred_tokens, config.n_out)
        else:
            if config.output_mode == "last_nonpad":
                x = _select_last_nonpad_hidden(x, tokens=tokens, pad_token_id=config.pad_token_id)
            else:
                x = x[:, -1, :]
            out = self.output(x)
            if config.n_pred_tokens > 1:
                out = out.reshape(out.shape[0], config.n_pred_tokens, config.n_out)

        if config.n_out == 1:
            out = out.squeeze(-1)
        return out
