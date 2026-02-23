"""Bonsai-adapted Mamba2 implementation for this repository.

This module keeps the existing `model.ssm.Mamba2` path intact and offers a
separate implementation based on the chunkwise SSD algorithm used in Bonsai.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx

from model.mup import MuReadout


def _validate_output_config(output_mode: str, n_pred_tokens: int) -> None:
    if output_mode not in {"last_token", "full_sequence", "last_nonpad"}:
        raise ValueError(
            "output_mode must be one of 'last_token', 'full_sequence', or 'last_nonpad', "
            f"got {output_mode!r}"
        )
    if n_pred_tokens < 1:
        raise ValueError(f"n_pred_tokens must be >= 1, got {n_pred_tokens}")


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


@jax.tree_util.register_pytree_node_class
@dataclass
class Mamba2BonsaiCache:
    """Per-layer convolution and SSM states for incremental decoding."""

    ssm_states: list[jnp.ndarray]
    conv_states: list[jnp.ndarray]

    def tree_flatten(self):
        return (self.ssm_states, self.conv_states), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(ssm_states=list(children[0]), conv_states=list(children[1]))


@dataclass
class Mamba2BonsaiConfig:
    """Mamba2 config with repository-default SSM settings."""

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
    d_conv: int = 4
    dt_min: float = 1e-3
    dt_max: float = 0.1
    scan_chunk_len: int = 64
    dt_floor: float = 1e-4

    def to_model(self, *, rngs: nnx.Rngs) -> "Mamba2Bonsai":
        return Mamba2Bonsai(self, rngs=rngs)


def create_empty_cache(
    config: Mamba2BonsaiConfig,
    batch_size: int,
    *,
    dtype: jnp.dtype = jnp.float32,
) -> Mamba2BonsaiCache:
    if config.scan_chunk_len < 1:
        raise ValueError(f"scan_chunk_len must be >= 1, got {config.scan_chunk_len}")
    if config.n_hidden % config.n_heads != 0:
        raise ValueError(
            f"n_hidden ({config.n_hidden}) must be divisible by n_heads ({config.n_heads})"
        )
    inner = config.expand * config.n_hidden
    if inner % config.n_heads != 0:
        raise ValueError(
            f"expand * n_hidden ({inner}) must be divisible by n_heads ({config.n_heads})"
        )
    conv_dim = inner + 2 * config.d_state
    cache_len = max(0, config.d_conv - 1)
    head_dim = inner // config.n_heads

    conv_states = [
        jnp.zeros((batch_size, conv_dim, cache_len), dtype=dtype)
        for _ in range(config.n_layers)
    ]
    ssm_states = [
        jnp.zeros((batch_size, config.n_heads, head_dim, config.d_state), dtype=dtype)
        for _ in range(config.n_layers)
    ]
    return Mamba2BonsaiCache(ssm_states=ssm_states, conv_states=conv_states)


def _pad_seq_dim(x: jnp.ndarray, pad_size: int) -> jnp.ndarray:
    if pad_size == 0:
        return x
    pad_width = [(0, 0)] * x.ndim
    pad_width[1] = (0, pad_size)
    return jnp.pad(x, pad_width, mode="constant", constant_values=0.0)


def _segsum(x: jnp.ndarray) -> jnp.ndarray:
    t_len = x.shape[-1]
    x_cumsum = jnp.cumsum(x, axis=-1)
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
    mask = jnp.tril(jnp.ones((t_len, t_len), dtype=bool), k=0)
    return jnp.where(mask, x_segsum, -jnp.inf)


def _ssd_forward(
    *,
    x: jnp.ndarray,
    dt: jnp.ndarray,
    a: jnp.ndarray,
    b_mat: jnp.ndarray,
    c_mat: jnp.ndarray,
    chunk_size: int,
    d: jnp.ndarray,
    dt_bias: jnp.ndarray,
    dt_min: float,
    dt_max: float,
    initial_states: jnp.ndarray | None,
    return_final_states: bool,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    batch_size, seq_len, num_heads, _head_dim = x.shape
    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size

    dt = jax.nn.softplus(dt + dt_bias)
    dt = jnp.clip(dt, dt_min, dt_max)

    x_pad = _pad_seq_dim(x, pad_size)
    dt_pad = _pad_seq_dim(dt, pad_size)
    b_pad = _pad_seq_dim(b_mat, pad_size)
    c_pad = _pad_seq_dim(c_mat, pad_size)

    d_residual = d.reshape(1, 1, num_heads, 1) * x_pad

    x_disc = x_pad * dt_pad[..., None]
    a_disc = a.astype(x_disc.dtype) * dt_pad

    def _chunk(tensor: jnp.ndarray) -> jnp.ndarray:
        bsz, t_all, *rest = tensor.shape
        return tensor.reshape(bsz, t_all // chunk_size, chunk_size, *rest)

    x_blk = _chunk(x_disc)
    a_blk = _chunk(a_disc)
    b_blk = _chunk(b_pad)
    c_blk = _chunk(c_pad)

    a_blk2 = jnp.transpose(a_blk, (0, 3, 1, 2))
    a_cumsum = jnp.cumsum(a_blk2, axis=-1)

    l_mat = jnp.exp(_segsum(a_blk2))
    y_diag = jnp.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", c_blk, b_blk, l_mat, x_blk)

    decay_states = jnp.exp(a_cumsum[..., -1:] - a_cumsum)
    states = jnp.einsum("bclhn,bhcl,bclhp->bchpn", b_blk, decay_states, x_blk)

    if initial_states is None:
        initial_states = jnp.zeros_like(states[:, :1, ...])
    states = jnp.concatenate([initial_states, states], axis=1)

    a_end = a_cumsum[..., -1]
    a_end_padded = jnp.pad(a_end, ((0, 0), (0, 0), (1, 0)))
    decay_chunk = jnp.exp(_segsum(a_end_padded))
    new_states = jnp.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1, ...], new_states[:, -1, ...]

    state_decay_out = jnp.exp(a_cumsum)
    y_off = jnp.einsum("bclhn,bchpn,bhcl->bclhp", c_blk, states, state_decay_out)

    y = y_diag + y_off
    bsz, chunks, chunk_len, heads, head_dim = y.shape
    y = y.reshape(bsz, chunks * chunk_len, heads, head_dim)
    y = y + d_residual
    if pad_size > 0:
        y = y[:, :seq_len, :, :]

    if return_final_states:
        return y, final_state
    return y, None


class _RMSNormGate(nnx.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, *, rngs: nnx.Rngs):
        del rngs
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nnx.Param(jnp.ones((hidden_size,)))

    def __call__(self, hidden_states: jnp.ndarray, residual: jnp.ndarray) -> jnp.ndarray:
        x = hidden_states.astype(jnp.float32)
        x = x * jax.nn.silu(residual.astype(jnp.float32))
        variance = jnp.mean(x**2, axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + self.eps) * self.weight[:]
        return x.astype(hidden_states.dtype)


class _DepthwiseConv1d(nnx.Module):
    def __init__(self, features: int, kernel_size: int, use_bias: bool, *, rngs: nnx.Rngs):
        self.features = features
        self.kernel_size = kernel_size
        self.conv = nnx.Conv(
            in_features=features,
            out_features=features,
            kernel_size=(kernel_size,),
            padding=((0, 0),),
            feature_group_count=features,
            use_bias=use_bias,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        conv_state: jnp.ndarray | None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        cache_len = max(0, self.kernel_size - 1)
        if conv_state is None:
            x_padded = jnp.pad(
                x,
                ((0, 0), (cache_len, 0), (0, 0)),
                mode="constant",
                constant_values=0.0,
            )
        elif cache_len == 0:
            x_padded = x
        else:
            x_padded = jnp.concatenate([jnp.transpose(conv_state, (0, 2, 1)), x], axis=1)
        output = self.conv(x_padded)
        if cache_len == 0:
            new_conv_state = jnp.zeros((x.shape[0], x.shape[-1], 0), dtype=x.dtype)
        else:
            new_conv_state = jnp.transpose(x_padded[:, -cache_len:, :], (0, 2, 1))
        return output, new_conv_state


def _inverse_softplus(x: jnp.ndarray) -> jnp.ndarray:
    return x + jnp.log(-jnp.expm1(-x))


class Mamba2BonsaiMixer(nnx.Module):
    def __init__(self, config: Mamba2BonsaiConfig, *, rngs: nnx.Rngs):
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
        if config.scan_chunk_len < 1:
            raise ValueError(
                f"scan_chunk_len must be >= 1, got {config.scan_chunk_len}"
            )

        self.hidden_size = config.n_hidden
        self.num_heads = config.n_heads
        self.state_size = config.d_state
        self.intermediate_size = config.expand * config.n_hidden
        if self.intermediate_size % self.num_heads != 0:
            raise ValueError(
                f"expand * n_hidden ({self.intermediate_size}) must be divisible by "
                f"n_heads ({self.num_heads})"
            )
        self.head_dim = self.intermediate_size // self.num_heads
        self.chunk_size = int(config.scan_chunk_len)
        self.dt_min = float(config.dt_min)
        self.dt_max = float(config.dt_max)
        self.dt_floor = float(config.dt_floor)
        self.act = jax.nn.silu

        proj_size = 2 * (self.intermediate_size + self.state_size) + self.num_heads
        self.in_proj = nnx.Linear(
            self.hidden_size,
            proj_size,
            use_bias=config.use_bias,
            rngs=rngs,
        )

        conv_dim = self.intermediate_size + 2 * self.state_size
        self.conv1d = _DepthwiseConv1d(
            conv_dim,
            config.d_conv,
            use_bias=config.use_bias,
            rngs=rngs,
        )

        dt_init = jnp.exp(
            jax.random.uniform(rngs.params(), (self.num_heads,))
            * (jnp.log(self.dt_max) - jnp.log(self.dt_min))
            + jnp.log(self.dt_min)
        )
        dt_init = jnp.maximum(dt_init, self.dt_floor)
        self.dt_bias = nnx.Param(_inverse_softplus(dt_init))

        a_init = jax.random.uniform(rngs.params(), (self.num_heads,), minval=1.0, maxval=16.0)
        self.a_log = nnx.Param(jnp.log(a_init))
        self.d = nnx.Param(jnp.ones((self.num_heads,)))

        self.norm = _RMSNormGate(self.intermediate_size, eps=1e-5, rngs=rngs)
        self.out_proj = nnx.Linear(
            self.intermediate_size,
            self.hidden_size,
            use_bias=config.use_bias,
            rngs=rngs,
        )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        *,
        conv_state: jnp.ndarray | None,
        ssm_state: jnp.ndarray | None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        batch_size, seq_len, _ = hidden_states.shape

        zxbcdt = self.in_proj(hidden_states)
        d_mlp = (
            zxbcdt.shape[-1]
            - 2 * self.intermediate_size
            - 2 * self.state_size
            - self.num_heads
        ) // 2

        z0, x0, z, xbc, dt = jnp.split(
            zxbcdt,
            [
                d_mlp,
                2 * d_mlp,
                2 * d_mlp + self.intermediate_size,
                2 * d_mlp + self.intermediate_size + self.intermediate_size + 2 * self.state_size,
            ],
            axis=-1,
        )

        xbc, new_conv_state = self.conv1d(xbc, conv_state=conv_state)
        xbc = self.act(xbc)
        x, b_t, c_t = jnp.split(
            xbc,
            [self.intermediate_size, self.intermediate_size + self.state_size],
            axis=-1,
        )

        init_state = ssm_state[:, None, ...] if ssm_state is not None else None
        a = -jnp.exp(self.a_log[:].astype(jnp.float32))

        b_exp = jnp.broadcast_to(
            jnp.expand_dims(b_t, axis=2),
            (batch_size, seq_len, self.num_heads, self.state_size),
        )
        c_exp = jnp.broadcast_to(
            jnp.expand_dims(c_t, axis=2),
            (batch_size, seq_len, self.num_heads, self.state_size),
        )

        y, new_ssm_state = _ssd_forward(
            x=x.reshape(batch_size, seq_len, self.num_heads, self.head_dim),
            dt=dt,
            a=a,
            b_mat=b_exp,
            c_mat=c_exp,
            chunk_size=self.chunk_size,
            d=self.d[:],
            dt_bias=self.dt_bias[:],
            dt_min=self.dt_min,
            dt_max=self.dt_max,
            initial_states=init_state,
            return_final_states=True,
        )
        y = y.reshape(batch_size, seq_len, self.intermediate_size)

        y = self.norm(y, residual=z)
        if d_mlp > 0:
            y = jnp.concatenate([self.act(z0) * x0, y], axis=-1)

        return self.out_proj(y), new_conv_state, new_ssm_state


class Mamba2BonsaiBlock(nnx.Module):
    def __init__(self, config: Mamba2BonsaiConfig, *, rngs: nnx.Rngs):
        self.pre_norm = nnx.RMSNorm(config.n_hidden, rngs=rngs) if config.layer_norm else None
        self.mixer = Mamba2BonsaiMixer(config, rngs=rngs)
        self.dropout = (
            nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
            if config.dropout_rate > 0
            else None
        )

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        conv_state: jnp.ndarray | None,
        ssm_state: jnp.ndarray | None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        residual = x
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        y, new_conv_state, new_ssm_state = self.mixer(
            x,
            conv_state=conv_state,
            ssm_state=ssm_state,
        )
        if self.dropout is not None:
            y = self.dropout(y)
        return residual + y, new_conv_state, new_ssm_state


class Mamba2Bonsai(nnx.Module):
    """Stacked Bonsai-style Mamba2 blocks with our output-mode contract."""

    def __init__(self, config: Mamba2BonsaiConfig, *, rngs: nnx.Rngs):
        _validate_output_config(config.output_mode, config.n_pred_tokens)
        if config.n_hidden % config.n_heads != 0:
            raise ValueError(
                f"n_hidden ({config.n_hidden}) must be divisible by n_heads ({config.n_heads})"
            )
        if config.scan_chunk_len < 1:
            raise ValueError(
                f"scan_chunk_len must be >= 1, got {config.scan_chunk_len}"
            )
        self.config = config
        self.inner = config.expand * config.n_hidden
        if self.inner % config.n_heads != 0:
            raise ValueError(
                f"expand * n_hidden ({self.inner}) must be divisible by n_heads ({config.n_heads})"
            )

        if config.n_vocab is not None:
            self.embed = nnx.Embed(config.n_vocab, config.n_hidden, rngs=rngs)
        else:
            self.embed = None

        self.blocks = nnx.List(
            [Mamba2BonsaiBlock(config, rngs=rngs) for _ in range(config.n_layers)]
        )
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
            self.output = nnx.Linear(
                config.n_hidden,
                out_features,
                use_bias=config.use_bias,
                rngs=rngs,
            )

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        cache: Mamba2BonsaiCache | None = None,
        return_cache: bool = False,
    ):
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

        if cache is None:
            conv_states = [None] * config.n_layers
            ssm_states = [None] * config.n_layers
        else:
            if len(cache.conv_states) != config.n_layers:
                raise ValueError("cache.conv_states length must equal n_layers")
            if len(cache.ssm_states) != config.n_layers:
                raise ValueError("cache.ssm_states length must equal n_layers")
            conv_states = cache.conv_states
            ssm_states = cache.ssm_states

        new_conv_states = []
        new_ssm_states = []
        for block, conv_state, ssm_state in zip(self.blocks, conv_states, ssm_states):
            x, new_conv_state, new_ssm_state = block(
                x,
                conv_state=conv_state,
                ssm_state=ssm_state,
            )
            new_conv_states.append(new_conv_state)
            new_ssm_states.append(new_ssm_state)

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

        if return_cache:
            return out, Mamba2BonsaiCache(
                ssm_states=new_ssm_states,
                conv_states=new_conv_states,
            )
        return out
