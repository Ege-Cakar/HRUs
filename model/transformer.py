"""
Transformer model using Flax NNX
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from model.mup import MuReadout, mup_attention_scale
from model.output import apply_output_projection, validate_output_config


@jax.tree_util.register_pytree_node_class
@dataclass
class TransformerKVCache:
    """Per-layer key/value tensors for incremental Transformer decoding."""

    keys: list[jnp.ndarray]
    values: list[jnp.ndarray]
    length: jnp.ndarray

    def tree_flatten(self):
        return (self.keys, self.values, self.length), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(
            keys=list(children[0]),
            values=list(children[1]),
            length=children[2],
        )


def create_empty_kv_cache(
    config: "TransformerConfig",
    batch_size: int,
    *,
    dtype: jnp.dtype = jnp.float32,
) -> TransformerKVCache:
    """Create an empty KV cache matching the Transformer config."""
    if config.n_hidden % config.n_heads != 0:
        raise ValueError(
            f"n_hidden ({config.n_hidden}) must be divisible by n_heads ({config.n_heads})"
        )
    if config.n_layers < 1:
        raise ValueError(f"n_layers must be >= 1, got {config.n_layers}")

    head_dim = config.n_hidden // config.n_heads
    empty_shape = (int(batch_size), config.n_heads, config.n_seq, head_dim)
    keys = [jnp.zeros(empty_shape, dtype=dtype) for _ in range(config.n_layers)]
    values = [jnp.zeros(empty_shape, dtype=dtype) for _ in range(config.n_layers)]
    return TransformerKVCache(
        keys=keys,
        values=values,
        length=jnp.asarray(0, dtype=jnp.int32),
    )


def _validate_kv_cache(
    cache: TransformerKVCache,
    *,
    n_layers: int,
    batch_size: int,
    n_heads: int,
    n_seq: int,
    head_dim: int,
) -> jnp.ndarray:
    if len(cache.keys) != n_layers:
        raise ValueError(
            f"cache.keys length must equal n_layers ({n_layers}), got {len(cache.keys)}"
        )
    if len(cache.values) != n_layers:
        raise ValueError(
            f"cache.values length must equal n_layers ({n_layers}), got {len(cache.values)}"
        )

    expected_shape = (batch_size, n_heads, n_seq, head_dim)
    for layer_idx, (k, v) in enumerate(zip(cache.keys, cache.values)):
        if k.ndim != 4 or v.ndim != 4:
            raise ValueError(
                f"cache tensors must be rank-4, got key={k.shape}, value={v.shape}"
            )
        if k.shape != expected_shape:
            raise ValueError(
                f"cache key layer {layer_idx} expected shape {expected_shape}, got {k.shape}"
            )
        if v.shape != expected_shape:
            raise ValueError(
                f"cache value layer {layer_idx} expected shape {expected_shape}, got {v.shape}"
            )

    cache_len = jnp.asarray(cache.length, dtype=jnp.int32)
    if cache_len.ndim != 0:
        raise ValueError(f"cache.length must be a scalar, got shape {cache_len.shape}")
    cache_len_int = _try_int_scalar(cache_len)
    if cache_len_int is not None and (cache_len_int < 0 or cache_len_int > n_seq):
        raise ValueError(f"cache.length must be in [0, {n_seq}], got {cache_len_int}")
    return cache_len


def _try_int_scalar(x: jnp.ndarray | int) -> int | None:
    try:
        arr = np.asarray(x)
    except Exception:
        return None
    if arr.ndim != 0:
        return None
    try:
        return int(arr)
    except (TypeError, ValueError):
        return None


def sinusoidal_pos_embedding(seq_len: int, dim: int) -> jnp.ndarray:
    """Generate sinusoidal positional embeddings.
    
    Args:
        seq_len: Maximum sequence length
        dim: Embedding dimension
        
    Returns:
        Positional embeddings of shape (seq_len, dim)
    """
    position = jnp.arange(seq_len)[:, None]
    div_term = jnp.exp(jnp.arange(0, dim, 2) * (-jnp.log(10000.0) / dim))
    
    pe = jnp.zeros((seq_len, dim))
    pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
    return pe


def rotary_pos_embedding(seq_len: int, dim: int, base: float = 10000.0) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generate rotary positional embeddings (cosine/sine caches).

    Args:
        seq_len: Maximum sequence length
        dim: Head dimension
        base: RoPE base

    Returns:
        cos, sin arrays of shape (seq_len, dim // 2)
    """
    rot_dim = (dim // 2) * 2
    if rot_dim == 0:
        return jnp.zeros((seq_len, 0)), jnp.zeros((seq_len, 0))
    inv_freq = 1.0 / (base ** (jnp.arange(0, rot_dim, 2) / rot_dim))
    positions = jnp.arange(seq_len)[:, None]
    freqs = positions * inv_freq[None, :]
    return jnp.cos(freqs), jnp.sin(freqs)


def apply_rotary_pos_embedding(
    q: jnp.ndarray,
    k: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply rotary positional embeddings to query and key.

    Args:
        q: (batch, heads, seq, head_dim)
        k: (batch, heads, seq, head_dim)
        cos: (seq, head_dim // 2)
        sin: (seq, head_dim // 2)
    """
    if cos.shape[-1] == 0:
        return q, k

    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    rot_dim = cos.shape[-1] * 2

    def _rotate(x: jnp.ndarray) -> jnp.ndarray:
        x_rot = x[..., :rot_dim]
        x_pass = x[..., rot_dim:]
        x_even = x_rot[..., 0::2]
        x_odd = x_rot[..., 1::2]
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_even * sin + x_odd * cos
        x_rotated = jnp.stack([x_rotated_even, x_rotated_odd], axis=-1)
        x_rotated = x_rotated.reshape(x_rot.shape)
        if x_pass.shape[-1] == 0:
            return x_rotated
        return jnp.concatenate([x_rotated, x_pass], axis=-1)

    return _rotate(q), _rotate(k)


@dataclass
class TransformerConfig:
    """Global hyperparameters for Transformer"""
    n_vocab: int | None = None
    n_seq: int = 128  # Maximum sequence length
    n_layers: int = 2
    n_hidden: int = 128
    n_heads: int = 4
    n_mlp_hidden: int | None = None  # MLP hidden dim, defaults to 4 * n_hidden (GELU) or 8/3 * n_hidden (SwiGLU)
    n_out: int = 1
    n_pred_tokens: int = 1
    pos_encoding: str = "none"
    layer_norm: bool = True  # Uses RMSNorm when enabled
    use_swiglu: bool = False
    use_bias: bool = True
    dropout_rate: float = 0.0
    output_mode: str = "last_token"
    pad_token_id: int = 0
    use_mup: bool = False
    use_sow: bool = False
    use_bf16: bool = True

    def to_model(self, *, rngs: nnx.Rngs) -> 'Transformer':
        return Transformer(self, rngs=rngs)


class MultiHeadAttention(nnx.Module):
    """Multi-head self-attention using Flax NNX."""

    def __init__(
        self, 
        n_hidden: int, 
        n_heads: int, 
        use_bias: bool = True,
        dropout_rate: float = 0.0,
        use_mup: bool = False,
        rotary_cos: jnp.ndarray | None = None,
        rotary_sin: jnp.ndarray | None = None,
        use_sow=False,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        *, 
        rngs: nnx.Rngs
    ):
        assert n_hidden % n_heads == 0, f"n_hidden ({n_hidden}) must be divisible by n_heads ({n_heads})"
        
        self.n_hidden = n_hidden
        self.n_heads = n_heads
        self.head_dim = n_hidden // n_heads
        self.rotary_cos = rotary_cos
        self.rotary_sin = rotary_sin
        self.use_sow = use_sow
        self.compute_dtype = dtype
        self._prefer_cudnn = jax.default_backend() == "gpu"
        if use_mup:
            self.scale = mup_attention_scale(self.head_dim)
        else:
            self.scale = self.head_dim ** -0.5
        
        # Combined QKV projection for efficiency
        self.qkv = nnx.Linear(
            n_hidden,
            3 * n_hidden,
            use_bias=use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.out_proj = nnx.Linear(
            n_hidden,
            n_hidden,
            use_bias=use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        
        if dropout_rate > 0:
            self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        else:
            self.dropout = None

    def __call__(
        self,
        x: jnp.ndarray,
        mask: jnp.ndarray | None = None,
        *,
        is_causal: bool = False,
        past_key: jnp.ndarray | None = None,
        past_value: jnp.ndarray | None = None,
        pos_offset: int | jnp.ndarray = 0,
        return_kv: bool = False,
    ) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, n_hidden)
            mask: Optional causal mask of shape (seq_len, seq_len)
            is_causal: If True, use native causal masking (memory-efficient
                flash attention path). Ignored when ``mask`` is provided.

        Returns:
            Output tensor of shape (batch, seq_len, n_hidden)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        qkv = self.qkv(x)  # (batch, seq, 3 * n_hidden)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))  # (3, batch, heads, seq, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.rotary_cos is not None and self.rotary_sin is not None:
            pos_offset_i32 = jnp.asarray(pos_offset, dtype=jnp.int32)
            pos_offset_int = _try_int_scalar(pos_offset)
            if pos_offset_int is not None:
                pos_end = pos_offset_int + seq_len
                if pos_offset_int < 0 or pos_end > self.rotary_cos.shape[0]:
                    raise ValueError(
                        "RoPE position range "
                        f"[{pos_offset_int}, {pos_end}) exceeds cache size {self.rotary_cos.shape[0]}"
                    )
            cos = jax.lax.dynamic_slice_in_dim(
                self.rotary_cos,
                start_index=pos_offset_i32,
                slice_size=seq_len,
                axis=0,
            )
            sin = jax.lax.dynamic_slice_in_dim(
                self.rotary_sin,
                start_index=pos_offset_i32,
                slice_size=seq_len,
                axis=0,
            )
            q, k = apply_rotary_pos_embedding(q, k, cos=cos, sin=sin)

        if (past_key is None) != (past_value is None):
            raise ValueError("past_key and past_value must both be provided or both be None")

        if past_key is not None and past_value is not None:
            if past_key.ndim != 4 or past_value.ndim != 4:
                raise ValueError(
                    f"past_key and past_value must be rank-4, got {past_key.shape} and {past_value.shape}"
                )
            expected_prefix = (batch_size, self.n_heads)
            if past_key.shape[:2] != expected_prefix:
                raise ValueError(
                    f"past_key expected prefix {expected_prefix}, got {past_key.shape[:2]}"
                )
            if past_value.shape[:2] != expected_prefix:
                raise ValueError(
                    f"past_value expected prefix {expected_prefix}, got {past_value.shape[:2]}"
                )
            if past_key.shape[3] != self.head_dim or past_value.shape[3] != self.head_dim:
                raise ValueError(
                    f"past head_dim mismatch; expected {self.head_dim}, got {past_key.shape[3]} and {past_value.shape[3]}"
                )
            past_len_i32 = jnp.asarray(pos_offset, dtype=jnp.int32)
            k_update = k if k.dtype == past_key.dtype else k.astype(past_key.dtype)
            v_update = v if v.dtype == past_value.dtype else v.astype(past_value.dtype)
            k = jax.lax.dynamic_update_slice(past_key, k_update, (0, 0, past_len_i32, 0))
            v = jax.lax.dynamic_update_slice(past_value, v_update, (0, 0, past_len_i32, 0))

        # Use dtype-compatible K/V tensors for attention computation when cache
        # and compute dtypes differ.
        k_attn = k if k.dtype == q.dtype else k.astype(q.dtype)
        v_attn = v if v.dtype == q.dtype else v.astype(q.dtype)

        # Prefer is_causal over explicit mask when no mask is provided.
        use_is_causal = is_causal and mask is None

        use_manual_attention = self.use_sow or self.dropout is not None
        if use_manual_attention:
            if use_is_causal:
                kv_len = k_attn.shape[2]
                mask = jnp.tril(jnp.ones((seq_len, kv_len), dtype=bool))
            attn_weights = jnp.einsum("bhqd,bhkd->bhqk", q, k_attn) * self.scale
            if mask is not None:
                mask_value = jnp.finfo(attn_weights.dtype).min
                attn_weights = jnp.where(mask, attn_weights, mask_value)
            attn_weights = jax.nn.softmax(attn_weights, axis=-1)
            if self.use_sow:
                self.sow(nnx.Intermediate, "attn_weights", attn_weights)
            if self.dropout is not None:
                attn_weights = self.dropout(attn_weights)
            out = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, v_attn)
            out = jnp.transpose(out, (0, 2, 1, 3))  # (batch, seq, heads, head_dim)
        else:
            # jax.nn.dot_product_attention expects (batch, seq, heads, head_dim).
            q_attn = jnp.transpose(q, (0, 2, 1, 3))
            k_attn = jnp.transpose(k_attn, (0, 2, 1, 3))
            v_attn = jnp.transpose(v_attn, (0, 2, 1, 3))
            implementation = (
                "cudnn"
                if self._prefer_cudnn and q_attn.dtype == jnp.bfloat16
                else "xla"
            )
            out = jax.nn.dot_product_attention(
                q_attn,
                k_attn,
                v_attn,
                mask=None if use_is_causal else mask,
                is_causal=use_is_causal,
                scale=self.scale,
                implementation=implementation,
            )
        out = out.reshape(batch_size, seq_len, self.n_hidden)

        out = self.out_proj(out)
        if return_kv:
            return out, k, v
        return out


class TransformerBlock(nnx.Module):
    """Single transformer block with pre-norm RMSNorm architecture."""

    def __init__(
        self, 
        n_hidden: int, 
        n_heads: int, 
        n_mlp_hidden: int,
        use_bias: bool = True,
        layer_norm: bool = True,
        use_swiglu: bool = False,
        dropout_rate: float = 0.0,
        use_mup: bool = False,
        rotary_cos: jnp.ndarray | None = None,
        rotary_sin: jnp.ndarray | None = None,
        use_sow: bool = False,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        *, 
        rngs: nnx.Rngs
    ):
        self.layer_norm = layer_norm
        self.use_swiglu = use_swiglu
        self.use_sow = use_sow
        
        # Attention
        self.attn = MultiHeadAttention(
            n_hidden=n_hidden,
            n_heads=n_heads,
            use_bias=use_bias,
            dropout_rate=dropout_rate,
            use_mup=use_mup,
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            use_sow=self.use_sow,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs
        )
        
        # MLP
        if use_swiglu:
            self.mlp_gate = nnx.Linear(
                n_hidden,
                n_mlp_hidden,
                use_bias=use_bias,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            self.mlp_up = nnx.Linear(
                n_hidden,
                n_mlp_hidden,
                use_bias=use_bias,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            self.mlp_down = nnx.Linear(
                n_mlp_hidden,
                n_hidden,
                use_bias=use_bias,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
        else:
            self.mlp_fc1 = nnx.Linear(
                n_hidden,
                n_mlp_hidden,
                use_bias=use_bias,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            self.mlp_fc2 = nnx.Linear(
                n_mlp_hidden,
                n_hidden,
                use_bias=use_bias,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
        
        # RMS norms (pre-norm architecture)
        if layer_norm:
            self.ln1 = nnx.RMSNorm(
                n_hidden,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            self.ln2 = nnx.RMSNorm(
                n_hidden,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
        
        if dropout_rate > 0:
            self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        else:
            self.dropout = None

    def __call__(
        self,
        x: jnp.ndarray,
        mask: jnp.ndarray | None = None,
        *,
        is_causal: bool = False,
        past_key: jnp.ndarray | None = None,
        past_value: jnp.ndarray | None = None,
        pos_offset: int | jnp.ndarray = 0,
        return_kv: bool = False,
    ) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # Self-attention with residual
        residual = x
        if self.layer_norm:
            x = self.ln1(x)
        if return_kv:
            x, new_key, new_value = self.attn(
                x,
                mask=mask,
                is_causal=is_causal,
                past_key=past_key,
                past_value=past_value,
                pos_offset=pos_offset,
                return_kv=True,
            )
        else:
            x = self.attn(
                x,
                mask=mask,
                is_causal=is_causal,
                past_key=past_key,
                past_value=past_value,
                pos_offset=pos_offset,
            )
        if self.dropout is not None:
            x = self.dropout(x)
        x = x + residual
        
        # MLP with residual
        residual = x
        if self.layer_norm:
            x = self.ln2(x)
        if self.use_swiglu:
            gate = self.mlp_gate(x)
            up = self.mlp_up(x)
            x = jax.nn.silu(gate) * up
            x = self.mlp_down(x)
        else:
            x = self.mlp_fc1(x)
            x = jax.nn.gelu(x)
            x = self.mlp_fc2(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x + residual
        
        if return_kv:
            return x, new_key, new_value
        return x


class Transformer(nnx.Module):
    """Transformer model using Flax NNX."""

    def __init__(self, config: TransformerConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.compute_dtype = jnp.bfloat16 if config.use_bf16 else jnp.float32
        self.param_dtype = jnp.float32
        if config.n_mlp_hidden is None:
            if config.use_swiglu:
                n_mlp_hidden = int(4 * config.n_hidden * 2 / 3)
            else:
                n_mlp_hidden = 4 * config.n_hidden
        else:
            n_mlp_hidden = config.n_mlp_hidden

        if config.pos_encoding not in {"absolute", "rope", "none"}:
            raise ValueError(
                f"pos_encoding must be one of 'absolute', 'rope', or 'none', got {config.pos_encoding!r}"
            )
        validate_output_config(config.output_mode, config.n_pred_tokens)

        # Token embedding
        if config.n_vocab is not None:
            self.embed = nnx.Embed(
                num_embeddings=config.n_vocab,
                features=config.n_hidden,
                dtype=self.compute_dtype,
                param_dtype=self.param_dtype,
                rngs=rngs
            )
        else:
            self.embed = None
        
        # Positional embeddings (stored as buffer, not parameter)
        if config.pos_encoding == "absolute":
            self.pos_embedding = sinusoidal_pos_embedding(config.n_seq, config.n_hidden).astype(
                self.compute_dtype
            )
            self.rotary_cos = None
            self.rotary_sin = None
        elif config.pos_encoding == "rope":
            self.pos_embedding = None
            head_dim = config.n_hidden // config.n_heads
            self.rotary_cos, self.rotary_sin = rotary_pos_embedding(config.n_seq, head_dim)
            self.rotary_cos = self.rotary_cos.astype(self.compute_dtype)
            self.rotary_sin = self.rotary_sin.astype(self.compute_dtype)
        else:
            self.pos_embedding = None
            self.rotary_cos = None
            self.rotary_sin = None
        
        # Transformer blocks
        self.blocks = nnx.List([
            TransformerBlock(
                n_hidden=config.n_hidden,
                n_heads=config.n_heads,
                n_mlp_hidden=n_mlp_hidden,
                use_bias=config.use_bias,
                layer_norm=config.layer_norm,
                use_swiglu=config.use_swiglu,
                dropout_rate=config.dropout_rate,
                use_mup=config.use_mup,
                rotary_cos=self.rotary_cos,
                rotary_sin=self.rotary_sin,
                use_sow=config.use_sow,
                dtype=self.compute_dtype,
                param_dtype=self.param_dtype,
                rngs=rngs,
            )
            for _ in range(config.n_layers)
        ])
        
        # Final RMS norm (if using normalization)
        if config.layer_norm:
            self.final_ln = nnx.RMSNorm(
                config.n_hidden,
                dtype=self.compute_dtype,
                param_dtype=self.param_dtype,
                rngs=rngs,
            )
        else:
            self.final_ln = None
        
        # Output projection
        out_features = config.n_out * config.n_pred_tokens
        if config.use_mup:
            self.output = MuReadout(
                config.n_hidden,
                out_features,
                use_bias=config.use_bias,
                dtype=self.compute_dtype,
                param_dtype=self.param_dtype,
                rngs=rngs,
            )
        else:
            self.output = nnx.Linear(
                config.n_hidden,
                out_features,
                use_bias=config.use_bias,
                dtype=self.compute_dtype,
                param_dtype=self.param_dtype,
                rngs=rngs,
            )
        
        # Precompute causal mask for max sequence length
        self._causal_mask = jnp.tril(jnp.ones((config.n_seq, config.n_seq), dtype=bool))

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        cache: TransformerKVCache | None = None,
        return_cache: bool = False,
    ):
        """
        Args:
            x: Input tensor
               - If n_vocab is set: (batch, seq_len) of token indices
               - If n_vocab is None: (batch, seq_len, n_hidden) pre-embedded
               
        Returns:
            Output logits:
               - If output_mode='last_token' and n_pred_tokens=1 and n_out=1: (batch,)
               - If output_mode='last_token' and n_pred_tokens=1 and n_out>1: (batch, n_out)
               - If output_mode='last_token' and n_pred_tokens>1: (batch, n_pred_tokens, n_out)
               - If output_mode='last_nonpad' and n_pred_tokens=1 and n_out=1: (batch,)
               - If output_mode='last_nonpad' and n_pred_tokens=1 and n_out>1: (batch, n_out)
               - If output_mode='last_nonpad' and n_pred_tokens>1: (batch, n_pred_tokens, n_out)
               - If output_mode='full_sequence' and n_pred_tokens=1: (batch, seq_len, n_out) or squeezed
               - If output_mode='full_sequence' and n_pred_tokens>1: (batch, seq_len, n_pred_tokens, n_out)
        """
        config = self.config
        tokens = None
        
        # Embed tokens if needed
        if self.embed is not None:
            assert x.ndim == 2, f"Expected 2D input (batch, seq) for token indices, got {x.shape}"
            tokens = x
            x = self.embed(x)
        else:
            assert x.ndim == 3, f"Expected 3D input (batch, seq, features), got {x.shape}"
            if config.output_mode == "last_nonpad":
                raise ValueError("output_mode='last_nonpad' requires token indices (n_vocab must be set)")

        x = x.astype(self.compute_dtype)
        
        batch_size, seq_len, _ = x.shape
        head_dim = config.n_hidden // config.n_heads

        use_cache = cache is not None or return_cache

        if use_cache:
            if config.output_mode == "last_nonpad":
                raise ValueError("output_mode='last_nonpad' does not support cache-based inference")
            if cache is None:
                cache = create_empty_kv_cache(config, batch_size=batch_size, dtype=x.dtype)
            past_len = _validate_kv_cache(
                cache,
                n_layers=config.n_layers,
                batch_size=batch_size,
                n_heads=config.n_heads,
                n_seq=config.n_seq,
                head_dim=head_dim,
            )
            past_len_int = _try_int_scalar(past_len)
            if past_len_int is not None and past_len_int + seq_len > config.n_seq:
                raise ValueError(
                    f"cache length ({past_len_int}) + input seq_len ({seq_len}) exceeds n_seq ({config.n_seq})"
                )
            past_keys = cache.keys
            past_values = cache.values
        else:
            if seq_len > config.n_seq:
                raise ValueError(
                    f"input seq_len ({seq_len}) exceeds configured n_seq ({config.n_seq})"
                )
            past_len = jnp.asarray(0, dtype=jnp.int32)
            past_keys = [None] * config.n_layers
            past_values = [None] * config.n_layers

        # Add positional embeddings
        if self.pos_embedding is not None:
            if use_cache:
                pos = jax.lax.dynamic_slice_in_dim(
                    self.pos_embedding,
                    start_index=jnp.asarray(past_len, dtype=jnp.int32),
                    slice_size=seq_len,
                    axis=0,
                )
            else:
                pos = self.pos_embedding[:seq_len]
            x = x + pos

        # Use precomputed causal mask (sliced to actual sequence length).
        # When not using cache, prefer is_causal=True to avoid materializing
        # the O(n²) attention mask, enabling memory-efficient flash attention.
        if use_cache:
            mask = jax.lax.dynamic_slice(
                self._causal_mask,
                (jnp.asarray(past_len, dtype=jnp.int32), 0),
                (seq_len, config.n_seq),
            )
            causal_flag = False
        else:
            mask = None
            causal_flag = True

        # Apply transformer blocks
        if return_cache:
            new_keys = []
            new_values = []
            for block, past_key, past_value in zip(self.blocks, past_keys, past_values):
                x, new_key, new_value = block(
                    x,
                    mask=mask,
                    is_causal=causal_flag,
                    past_key=past_key,
                    past_value=past_value,
                    pos_offset=past_len,
                    return_kv=True,
                )
                new_keys.append(new_key)
                new_values.append(new_value)
        else:
            for block, past_key, past_value in zip(self.blocks, past_keys, past_values):
                x = block(
                    x,
                    mask=mask,
                    is_causal=causal_flag,
                    past_key=past_key,
                    past_value=past_value,
                    pos_offset=past_len,
                )
        
        # Final RMS norm
        if self.final_ln is not None:
            x = self.final_ln(x)

        # Output projection
        out = apply_output_projection(
            x, self.output,
            output_mode=config.output_mode,
            n_pred_tokens=config.n_pred_tokens,
            n_out=config.n_out,
            tokens=tokens,
            pad_token_id=config.pad_token_id,
        )

        if return_cache:
            return out, TransformerKVCache(
                keys=new_keys,
                values=new_values,
                length=jnp.asarray(past_len, dtype=jnp.int32) + jnp.asarray(seq_len, dtype=jnp.int32),
            )
        return out
