"""
Gated DeltaNet (GDN) linear attention block using Flax NNX.

Implements the gated delta rule from Yang et al. (ICLR 2025):
    S_t = S_{t-1} * (α_t * (I - β_t * k_t * k_t^T)) + β_t * v_t * k_t^T   (Eq. 10)

where:
    α_t ∈ (0,1) is a data-dependent decay/gating term (controls memory clearance)
    β_t ∈ (0,1) is the delta write strength (controls update precision)

This combines Mamba2's gated decay with DeltaNet's delta rule for complementary
memory management: α enables rapid memory erasure, β enables targeted updates.

Block design follows the paper's Figure 1 (Section 3.4):
    - q, k: Linear → short conv → SiLU → L2 norm
    - v:    Linear → short conv → SiLU
    - α, β: Linear (no conv, no activation) → sigmoid
    - output: GroupNorm → output_gate(SiLU) * normalized_output → Linear

The GDNBlock follows Llama's macro architecture with SwiGLU MLP layers,
matching the Ouro/LoopLM transformer block design.

Sequential implementation via jax.lax.scan. Chunkwise parallel (Section 3.3)
can be added for GPU efficiency without changing the recurrence semantics.
"""

import jax
import jax.numpy as jnp
from flax import nnx


def _short_conv_1d(x: jnp.ndarray, weight: jnp.ndarray) -> jnp.ndarray:
    """Causal 1D depthwise convolution with kernel size 4.

    Args:
        x: (batch, seq, features)
        weight: (features, kernel_size) convolution weights

    Returns:
        (batch, seq, features) — causally convolved
    """
    features, kernel_size = weight.shape
    # Pad causally: (kernel_size - 1) on the left, 0 on the right
    x_padded = jnp.pad(x, ((0, 0), (kernel_size - 1, 0), (0, 0)))
    # Depthwise conv: manual sliding window via einsum for clarity
    # Stack shifted versions: (batch, seq, features, kernel_size)
    shifts = jnp.stack(
        [x_padded[:, i:i + x.shape[1], :] for i in range(kernel_size)],
        axis=-1,
    )
    # Elementwise multiply by weight and sum over kernel dim
    return jnp.sum(shifts * weight[None, None, :, :], axis=-1)


class GatedDeltaNetAttention(nnx.Module):
    """Gated DeltaNet linear attention (Yang et al., ICLR 2025).

    Recurrence: S_t = α_t * S_{t-1} * (I - β_t * k_t @ k_t^T) + β_t * v_t @ k_t^T
    Output:     o_t = S_t @ q_t

    q, k path: Linear → ShortConv(4) → SiLU → L2 norm
    v path:    Linear → ShortConv(4) → SiLU
    α, β:      Linear → sigmoid (per-head scalars)
    output:    GroupNorm → gate(SiLU) * output → Linear
    """

    def __init__(
        self,
        n_hidden: int,
        n_heads: int,
        use_bias: bool = True,
        conv_kernel_size: int = 4,
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
        self.conv_kernel_size = conv_kernel_size

        # Q, K, V linear projections
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

        # Short convolution weights for q, k, v (depthwise, causal)
        self.q_conv = nnx.Param(
            jax.random.normal(rngs.params(), (n_hidden, conv_kernel_size)) * 0.02
        )
        self.k_conv = nnx.Param(
            jax.random.normal(rngs.params(), (n_hidden, conv_kernel_size)) * 0.02
        )
        self.v_conv = nnx.Param(
            jax.random.normal(rngs.params(), (n_hidden, conv_kernel_size)) * 0.02
        )

        # α (decay gate) and β (write strength): project to n_heads scalars each
        self.alpha_proj = nnx.Linear(
            n_hidden, n_heads, use_bias=True,
            dtype=dtype, param_dtype=param_dtype, rngs=rngs,
        )
        self.beta_proj = nnx.Linear(
            n_hidden, n_heads, use_bias=True,
            dtype=dtype, param_dtype=param_dtype, rngs=rngs,
        )

        # Output gate: Linear → SiLU, applied elementwise to output
        self.output_gate_proj = nnx.Linear(
            n_hidden, n_hidden, use_bias=use_bias,
            dtype=dtype, param_dtype=param_dtype, rngs=rngs,
        )

        # Output normalization (GroupNorm with groups = n_heads)
        self.output_norm = nnx.GroupNorm(
            num_groups=n_heads, num_features=n_hidden,
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

        # ── Q, K, V: Linear → ShortConv → SiLU ──────────────────────
        q = self.q_proj(x)
        q = _short_conv_1d(q, self.q_conv.value)
        q = jax.nn.silu(q)
        q = q.reshape(batch, seq, h, d)
        # L2 normalize q and k per head
        q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)

        k = self.k_proj(x)
        k = _short_conv_1d(k, self.k_conv.value)
        k = jax.nn.silu(k)
        k = k.reshape(batch, seq, h, d)
        k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)

        v = self.v_proj(x)
        v = _short_conv_1d(v, self.v_conv.value)
        v = jax.nn.silu(v)
        v = v.reshape(batch, seq, h, d)
        # No L2 norm on v

        # ── α (decay) and β (write strength): Linear → sigmoid ──────
        alpha = jax.nn.sigmoid(self.alpha_proj(x))  # (batch, seq, n_heads)
        beta = jax.nn.sigmoid(self.beta_proj(x))    # (batch, seq, n_heads)

        # ── Output gate: Linear → SiLU ──────────────────────────────
        out_gate = jax.nn.silu(self.output_gate_proj(x))  # (batch, seq, n_hidden)

        # ── Transpose for scan: (seq, batch, heads, dim) ────────────
        q = jnp.transpose(q, (1, 0, 2, 3))
        k = jnp.transpose(k, (1, 0, 2, 3))
        v = jnp.transpose(v, (1, 0, 2, 3))
        alpha = jnp.transpose(alpha, (1, 0, 2))      # (seq, batch, heads)
        beta = jnp.transpose(beta, (1, 0, 2))         # (seq, batch, heads)

        # ── Gated delta rule recurrence via scan ─────────────────────
        def _step(S, inputs):
            """Gated delta rule (Eq. 10):
            S_t = α_t * S_{t-1} * (I - β_t * k_t @ k_t^T) + β_t * v_t @ k_t^T

            Equivalently:
            S_t = α_t * (S_{t-1} - β_t * (S_{t-1} @ k_t) @ k_t^T) + β_t * v_t @ k_t^T

            S:      (batch, heads, d_v, d_k) — recurrent state
            q_t:    (batch, heads, d_k)
            k_t:    (batch, heads, d_k)
            v_t:    (batch, heads, d_v)
            alpha_t: (batch, heads) — decay
            beta_t:  (batch, heads) — write strength
            """
            q_t, k_t, v_t, alpha_t, beta_t = inputs

            # Expand scalars for broadcasting: (batch, heads, 1, 1)
            a = alpha_t[:, :, None, None]
            b = beta_t[:, :, None, None]

            # S_{t-1} @ k_t: (batch, heads, d_v)
            Sk = jnp.einsum("bhde,bhe->bhd", S, k_t)

            # delta = v_t - S_{t-1} @ k_t
            delta = v_t - Sk

            # Gated delta update:
            # S_t = α * S_{t-1} + β * delta @ k^T
            # = α * S_{t-1} + β * (v - S@k) @ k^T
            # = α * S_{t-1} - α * β * (S@k) @ k^T + β * v @ k^T  [rearranged]
            # But the paper's Eq. 10 is:
            # S_t = S_{t-1} * α * (I - β * k @ k^T) + β * v @ k^T
            # = α * S_{t-1} - α * β * S_{t-1} @ k @ k^T + β * v @ k^T
            # Which equals: α * S_{t-1} + β * (v - α * S_{t-1} @ k) @ k^T
            Sk_alpha = alpha_t[:, :, None] * Sk  # α * (S @ k)
            delta_gated = v_t - Sk_alpha  # v - α * S @ k
            update = b * jnp.einsum("bhd,bhe->bhde", delta_gated, k_t)
            S = a * S + update

            # Output: o_t = S_t @ q_t
            o_t = jnp.einsum("bhde,bhe->bhd", S, q_t)
            return S, o_t

        # Initial state: zero matrix per head
        S0 = jnp.zeros((batch, h, d, d), dtype=x.dtype)

        # Run scan over sequence dimension
        _, outputs = jax.lax.scan(_step, S0, (q, k, v, alpha, beta))
        # outputs: (seq, batch, heads, head_dim)

        # ── Transpose back: (batch, seq, heads * head_dim) ──────────
        outputs = jnp.transpose(outputs, (1, 0, 2, 3))
        outputs = outputs.reshape(batch, seq, self.n_hidden)

        # ── Output: GroupNorm → gate * output → Linear ───────────────
        outputs = self.output_norm(outputs)
        outputs = out_gate * outputs
        return self.out_proj(outputs)


class GDNBlock(nnx.Module):
    """GDN + SwiGLU MLP block with pre-norm residual connections.

    Follows Llama/Ouro macro architecture:
        x → RMSNorm → GDN → residual → RMSNorm → SwiGLU_MLP → residual

    Same call interface as TransformerBlock for drop-in use in Architecture
    and RecursiveArchitecture.
    """

    def __init__(
        self,
        n_hidden: int,
        n_heads: int,
        n_mlp_hidden: int,
        use_bias: bool = True,
        layer_norm: bool = True,
        use_swiglu: bool = True,  # Default True to match Ouro
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

        # MLP: SwiGLU by default (matching Ouro/Llama)
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

        # RMS norms (pre-norm architecture)
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
        compatibility, but ignores mask/cache args (GDN is causal by construction
        via the scan recurrence)."""
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
            dummy = jnp.zeros((0,), dtype=x.dtype)
            return x, dummy, dummy
        return x
