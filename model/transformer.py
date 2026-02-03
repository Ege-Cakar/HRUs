"""
Transformer model using Flax NNX
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx

from model.mup import MuReadout, mup_attention_scale

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
        *, 
        rngs: nnx.Rngs
    ):
        assert n_hidden % n_heads == 0, f"n_hidden ({n_hidden}) must be divisible by n_heads ({n_heads})"
        
        self.n_hidden = n_hidden
        self.n_heads = n_heads
        self.head_dim = n_hidden // n_heads
        self.rotary_cos = rotary_cos
        self.rotary_sin = rotary_sin
        if use_mup:
            self.scale = mup_attention_scale(self.head_dim)
        else:
            self.scale = self.head_dim ** -0.5
        
        # Combined QKV projection for efficiency
        self.qkv = nnx.Linear(n_hidden, 3 * n_hidden, use_bias=use_bias, rngs=rngs)
        self.out_proj = nnx.Linear(n_hidden, n_hidden, use_bias=use_bias, rngs=rngs)
        
        if dropout_rate > 0:
            self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        else:
            self.dropout = None

    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray | None = None) -> jnp.ndarray:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, n_hidden)
            mask: Optional causal mask of shape (seq_len, seq_len)
            
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
            cos = self.rotary_cos[:seq_len]
            sin = self.rotary_sin[:seq_len]
            q, k = apply_rotary_pos_embedding(q, k, cos=cos, sin=sin)
        
        # Scaled dot-product attention
        attn_weights = jnp.einsum('bhqd,bhkd->bhqk', q, k) * self.scale
        
        if mask is not None:
            attn_weights = jnp.where(mask, attn_weights, -1e9)
        
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        
        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, v)
        out = jnp.transpose(out, (0, 2, 1, 3))  # (batch, seq, heads, head_dim)
        out = out.reshape(batch_size, seq_len, self.n_hidden)
        
        return self.out_proj(out)


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
        *, 
        rngs: nnx.Rngs
    ):
        self.layer_norm = layer_norm
        self.use_swiglu = use_swiglu
        
        # Attention
        self.attn = MultiHeadAttention(
            n_hidden=n_hidden,
            n_heads=n_heads,
            use_bias=use_bias,
            dropout_rate=dropout_rate,
            use_mup=use_mup,
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            rngs=rngs
        )
        
        # MLP
        if use_swiglu:
            self.mlp_gate = nnx.Linear(n_hidden, n_mlp_hidden, use_bias=use_bias, rngs=rngs)
            self.mlp_up = nnx.Linear(n_hidden, n_mlp_hidden, use_bias=use_bias, rngs=rngs)
            self.mlp_down = nnx.Linear(n_mlp_hidden, n_hidden, use_bias=use_bias, rngs=rngs)
        else:
            self.mlp_fc1 = nnx.Linear(n_hidden, n_mlp_hidden, use_bias=use_bias, rngs=rngs)
            self.mlp_fc2 = nnx.Linear(n_mlp_hidden, n_hidden, use_bias=use_bias, rngs=rngs)
        
        # RMS norms (pre-norm architecture)
        if layer_norm:
            self.ln1 = nnx.RMSNorm(n_hidden, rngs=rngs)
            self.ln2 = nnx.RMSNorm(n_hidden, rngs=rngs)
        
        if dropout_rate > 0:
            self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        else:
            self.dropout = None

    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray | None = None) -> jnp.ndarray:
        # Self-attention with residual
        residual = x
        if self.layer_norm:
            x = self.ln1(x)
        x = self.attn(x, mask=mask)
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
        
        return x


class Transformer(nnx.Module):
    """Transformer model using Flax NNX."""

    def __init__(self, config: TransformerConfig, *, rngs: nnx.Rngs):
        self.config = config
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
        if config.output_mode not in {"last_token", "full_sequence", "last_nonpad"}:
            raise ValueError(
                "output_mode must be one of 'last_token', 'full_sequence', or 'last_nonpad', "
                f"got {config.output_mode!r}"
            )
        if config.n_pred_tokens < 1:
            raise ValueError(f"n_pred_tokens must be >= 1, got {config.n_pred_tokens}")
        
        # Token embedding
        if config.n_vocab is not None:
            self.embed = nnx.Embed(
                num_embeddings=config.n_vocab,
                features=config.n_hidden,
                rngs=rngs
            )
        else:
            self.embed = None
        
        # Positional embeddings (stored as buffer, not parameter)
        if config.pos_encoding == "absolute":
            self.pos_embedding = sinusoidal_pos_embedding(config.n_seq, config.n_hidden)
            self.rotary_cos = None
            self.rotary_sin = None
        elif config.pos_encoding == "rope":
            self.pos_embedding = None
            head_dim = config.n_hidden // config.n_heads
            self.rotary_cos, self.rotary_sin = rotary_pos_embedding(config.n_seq, head_dim)
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
                rngs=rngs,
            )
            for _ in range(config.n_layers)
        ])
        
        # Final RMS norm (if using normalization)
        if config.layer_norm:
            self.final_ln = nnx.RMSNorm(config.n_hidden, rngs=rngs)
        else:
            self.final_ln = None
        
        # Output projection
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
        
        # Precompute causal mask for max sequence length
        self._causal_mask = jnp.tril(jnp.ones((config.n_seq, config.n_seq), dtype=bool))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
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
        
        batch_size, seq_len, _ = x.shape
        
        # Add positional embeddings
        if self.pos_embedding is not None:
            x = x + self.pos_embedding[:seq_len]
        
        # Use precomputed causal mask (sliced to actual sequence length)
        mask = self._causal_mask[:seq_len, :seq_len]
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask=mask)
        
        # Final RMS norm
        if self.final_ln is not None:
            x = self.final_ln(x)
        
        # Output projection
        if config.output_mode == "full_sequence":
            out = self.output(x)  # (batch, seq_len, n_out * n_pred_tokens)
            if config.n_pred_tokens > 1:
                out = out.reshape(out.shape[0], out.shape[1], config.n_pred_tokens, config.n_out)
        else:
            if config.output_mode == "last_nonpad":
                is_nonpad = tokens != config.pad_token_id
                lengths = jnp.sum(is_nonpad, axis=1)
                last_index = jnp.maximum(lengths - 1, 0)
                batch_idx = jnp.arange(x.shape[0])
                x = x[batch_idx, last_index, :]
            else:
                x = x[:, -1, :]  # (batch, n_hidden)
            out = self.output(x)  # (batch, n_out * n_pred_tokens)
            if config.n_pred_tokens > 1:
                out = out.reshape(out.shape[0], config.n_pred_tokens, config.n_out)
        
        # Squeeze single output dimension
        if config.n_out == 1:
            out = out.squeeze(-1)
        
        return out
