"""
Gated DeltaNet (GDN) linear attention block using Flax NNX.

Implements the gated delta rule recurrence from Yang et al. (2024):
    S_t = S_{t-1} + beta_t * (v_t - S_{t-1} @ k_t) @ k_t^T

where beta_t is a learned per-head gate controlling the update strength.
This is a linear-complexity alternative to softmax attention that supports
efficient sequential (scan-based) computation.

The GDNBlock follows the same pre-norm residual interface as TransformerBlock:
    x -> RMSNorm -> GDN_Attention -> residual -> RMSNorm -> MLP -> residual
"""

import jax
import jax.numpy as jnp
from flax import nnx


class GatedDeltaNetAttention(nnx.Module):
    """Gated DeltaNet linear attention.

    Uses the delta rule recurrence with a learned gate per head.
    Sequential implementation via jax.lax.scan — correct and sufficient
    for small/medium models. Chunkwise parallel can be added later.
    """

    def __init__(
        self,
        n_hidden: int,
        n_heads: int,
        use_bias: bool = True,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        *,
        rngs: nnx.Rngs,
    ):
        assert n_hidden % n_heads == 0, (
            f"n_hidden ({n_hidden}) must be divisible by n_heads ({n_heads})"
        )

        self.n_hidden = n_hidden
        self.n_heads = n_heads
        self.head_dim = n_hidden // n_heads

        # Q, K, V projections (separate for clarity; can fuse later)
        self.q_proj = nnx.Linear(
            n_hidden, n_hidden, use_bias=use_bias,
            dtype=dtype, param_dtype=param_dtype, rngs=rngs,
        )
        self.k_proj = nnx.Linear(
            n_hidden, n_hidden, use_bias=use_bias,
            dtype=dtype, param_dtype=param_dtype, rngs=rngs,
        )
        self.v_proj = nnx.Linear(
            n_hidden, n_hidden, use_bias=use_bias,
            dtype=dtype, param_dtype=param_dtype, rngs=rngs,
        )

        # Per-head gate: projects to n_heads scalars
        self.beta_proj = nnx.Linear(
            n_hidden, n_heads, use_bias=True,
            dtype=dtype, param_dtype=param_dtype, rngs=rngs,
        )

        # Output projection
        self.out_proj = nnx.Linear(
            n_hidden, n_hidden, use_bias=use_bias,
            dtype=dtype, param_dtype=param_dtype, rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: (batch, seq, n_hidden)

        Returns:
            (batch, seq, n_hidden)
        """
        batch, seq, _ = x.shape
        d = self.head_dim
        h = self.n_heads

        # Project to q, k, v: (batch, seq, n_heads, head_dim)
        q = self.q_proj(x).reshape(batch, seq, h, d)
        k = self.k_proj(x).reshape(batch, seq, h, d)
        v = self.v_proj(x).reshape(batch, seq, h, d)

        # Normalize k for numerical stability (L2 norm per head)
        k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)

        # Per-head gate: (batch, seq, n_heads, 1)
        beta = jax.nn.sigmoid(self.beta_proj(x))  # (batch, seq, n_heads)
        beta = beta[..., None]  # (batch, seq, n_heads, 1)

        # Transpose for scan: (seq, batch, heads, dim)
        q = jnp.transpose(q, (1, 0, 2, 3))
        k = jnp.transpose(k, (1, 0, 2, 3))
        v = jnp.transpose(v, (1, 0, 2, 3))
        beta = jnp.transpose(beta, (1, 0, 2, 3))

        # Delta rule recurrence via scan
        def _step(S, inputs):
            """Single recurrence step.

            S: (batch, heads, head_dim, head_dim) — recurrent state matrix
            inputs: (q_t, k_t, v_t, beta_t) each (batch, heads, dim) or (batch, heads, 1)
            """
            q_t, k_t, v_t, beta_t = inputs

            # Delta update: S = S + beta * (v - S @ k) @ k^T
            Sk = jnp.einsum("bhde,bhe->bhd", S, k_t)  # (batch, heads, head_dim)
            delta = v_t - Sk  # (batch, heads, head_dim)
            # Outer product: delta @ k^T, scaled by beta
            # beta_t is (batch, heads, 1), expand to (batch, heads, 1, 1) for broadcast
            update = beta_t[..., None] * jnp.einsum("bhd,bhe->bhde", delta, k_t)
            S = S + update

            # Output: o_t = S @ q_t
            o_t = jnp.einsum("bhde,bhe->bhd", S, q_t)  # (batch, heads, head_dim)
            return S, o_t

        # Initial state: zero matrix per head
        S0 = jnp.zeros((batch, h, d, d), dtype=x.dtype)

        # Run scan over sequence dimension
        _, outputs = jax.lax.scan(_step, S0, (q, k, v, beta))
        # outputs: (seq, batch, heads, head_dim)

        # Transpose back: (batch, seq, heads, head_dim)
        outputs = jnp.transpose(outputs, (1, 0, 2, 3))
        # Reshape to (batch, seq, n_hidden)
        outputs = outputs.reshape(batch, seq, self.n_hidden)

        return self.out_proj(outputs)


class GDNBlock(nnx.Module):
    """GDN + MLP block with pre-norm residual connections.

    Same interface as TransformerBlock (minus KV cache / causal mask args,
    which are not applicable to linear attention).
    """

    def __init__(
        self,
        n_hidden: int,
        n_heads: int,
        n_mlp_hidden: int,
        use_bias: bool = True,
        layer_norm: bool = True,
        use_swiglu: bool = False,
        dropout_rate: float = 0.0,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        *,
        rngs: nnx.Rngs,
    ):
        self.layer_norm = layer_norm
        self.use_swiglu = use_swiglu

        # GDN attention
        self.attn = GatedDeltaNetAttention(
            n_hidden=n_hidden,
            n_heads=n_heads,
            use_bias=use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        # MLP (same structure as TransformerBlock)
        if use_swiglu:
            self.mlp_gate = nnx.Linear(
                n_hidden, n_mlp_hidden, use_bias=use_bias,
                dtype=dtype, param_dtype=param_dtype, rngs=rngs,
            )
            self.mlp_up = nnx.Linear(
                n_hidden, n_mlp_hidden, use_bias=use_bias,
                dtype=dtype, param_dtype=param_dtype, rngs=rngs,
            )
            self.mlp_down = nnx.Linear(
                n_mlp_hidden, n_hidden, use_bias=use_bias,
                dtype=dtype, param_dtype=param_dtype, rngs=rngs,
            )
        else:
            self.mlp_fc1 = nnx.Linear(
                n_hidden, n_mlp_hidden, use_bias=use_bias,
                dtype=dtype, param_dtype=param_dtype, rngs=rngs,
            )
            self.mlp_fc2 = nnx.Linear(
                n_mlp_hidden, n_hidden, use_bias=use_bias,
                dtype=dtype, param_dtype=param_dtype, rngs=rngs,
            )

        # RMS norms
        if layer_norm:
            self.ln1 = nnx.RMSNorm(
                n_hidden, dtype=dtype, param_dtype=param_dtype, rngs=rngs,
            )
            self.ln2 = nnx.RMSNorm(
                n_hidden, dtype=dtype, param_dtype=param_dtype, rngs=rngs,
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
    ) -> jnp.ndarray:
        """Forward pass. Accepts same kwargs as TransformerBlock for interface
        compatibility, but ignores mask/cache args (linear attention is causal
        by construction via the scan recurrence)."""
        # GDN attention with residual
        residual = x
        if self.layer_norm:
            x = self.ln1(x)
        x = self.attn(x)
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
            # Return dummy KV for interface compat; GDN has no KV cache
            dummy = jnp.zeros((0,), dtype=x.dtype)
            return x, dummy, dummy
        return x
