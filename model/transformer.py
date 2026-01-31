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


@dataclass
class TransformerConfig:
    """Global hyperparameters for Transformer"""
    n_vocab: int | None = None
    n_seq: int = 128  # Maximum sequence length
    n_layers: int = 2
    n_hidden: int = 128
    n_heads: int = 4
    n_mlp_hidden: int | None = None  # MLP hidden dim, defaults to 4 * n_hidden
    n_out: int = 1
    pos_emb: bool = True
    layer_norm: bool = True
    use_bias: bool = True
    dropout_rate: float = 0.0
    last_token_only: bool = True
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
        *, 
        rngs: nnx.Rngs
    ):
        assert n_hidden % n_heads == 0, f"n_hidden ({n_hidden}) must be divisible by n_heads ({n_heads})"
        
        self.n_hidden = n_hidden
        self.n_heads = n_heads
        self.head_dim = n_hidden // n_heads
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
    """Single transformer block with pre-norm architecture."""

    def __init__(
        self, 
        n_hidden: int, 
        n_heads: int, 
        n_mlp_hidden: int,
        use_bias: bool = True,
        layer_norm: bool = True,
        dropout_rate: float = 0.0,
        use_mup: bool = False,
        *, 
        rngs: nnx.Rngs
    ):
        self.layer_norm = layer_norm
        
        # Attention
        self.attn = MultiHeadAttention(
            n_hidden=n_hidden,
            n_heads=n_heads,
            use_bias=use_bias,
            dropout_rate=dropout_rate,
            use_mup=use_mup,
            rngs=rngs
        )
        
        # MLP
        self.mlp_fc1 = nnx.Linear(n_hidden, n_mlp_hidden, use_bias=use_bias, rngs=rngs)
        self.mlp_fc2 = nnx.Linear(n_mlp_hidden, n_hidden, use_bias=use_bias, rngs=rngs)
        
        # Layer norms (pre-norm architecture)
        if layer_norm:
            self.ln1 = nnx.LayerNorm(n_hidden, rngs=rngs)
            self.ln2 = nnx.LayerNorm(n_hidden, rngs=rngs)
        
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
        n_mlp_hidden = config.n_mlp_hidden or (4 * config.n_hidden)
        
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
        if config.pos_emb:
            self.pos_embedding = sinusoidal_pos_embedding(config.n_seq, config.n_hidden)
        else:
            self.pos_embedding = None
        
        # Transformer blocks
        self.blocks = nnx.List([
            TransformerBlock(
                n_hidden=config.n_hidden,
                n_heads=config.n_heads,
                n_mlp_hidden=n_mlp_hidden,
                use_bias=config.use_bias,
                layer_norm=config.layer_norm,
                dropout_rate=config.dropout_rate,
                use_mup=config.use_mup,
                rngs=rngs
            )
            for _ in range(config.n_layers)
        ])
        
        # Final layer norm (if using layer norm)
        if config.layer_norm:
            self.final_ln = nnx.LayerNorm(config.n_hidden, rngs=rngs)
        else:
            self.final_ln = None
        
        # Output projection
        if config.use_mup:
            self.output = MuReadout(
                config.n_hidden,
                config.n_out,
                use_bias=config.use_bias,
                rngs=rngs,
            )
        else:
            self.output = nnx.Linear(config.n_hidden, config.n_out, use_bias=config.use_bias, rngs=rngs)
        
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
               - If last_token_only=True and n_out=1: (batch,)
               - If last_token_only=True and n_out>1: (batch, n_out)
               - If last_token_only=False: (batch, seq_len, n_out) or squeezed
        """
        config = self.config
        
        # Embed tokens if needed
        if self.embed is not None:
            assert x.ndim == 2, f"Expected 2D input (batch, seq) for token indices, got {x.shape}"
            x = self.embed(x)
        else:
            assert x.ndim == 3, f"Expected 3D input (batch, seq, features), got {x.shape}"
        
        batch_size, seq_len, _ = x.shape
        
        # Add positional embeddings
        if self.pos_embedding is not None:
            x = x + self.pos_embedding[:seq_len]
        
        # Use precomputed causal mask (sliced to actual sequence length)
        mask = self._causal_mask[:seq_len, :seq_len]
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask=mask)
        
        # Final layer norm
        if self.final_ln is not None:
            x = self.final_ln(x)
        
        # Output projection
        if config.last_token_only:
            x = x[:, -1, :]  # (batch, n_hidden)
            out = self.output(x)  # (batch, n_out)
        else:
            out = self.output(x)  # (batch, seq_len, n_out)
        
        # Squeeze single output dimension
        if config.n_out == 1:
            out = out.squeeze(-1)
        
        return out
