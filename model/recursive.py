"""
Recursive (depth-recurrent) architecture: prelude / looped core / coda.

Structure: Attn + [core_pattern] × T (weight-tied) + Attn
  - Recursive Transformer:  Attn + [Attn Attn Attn Attn] × T + Attn
  - Recursive Hybrid (HRU): Attn + [GDN  GDN  GDN  Attn] × T + Attn

The core blocks are instantiated ONCE and looped T times (weight sharing).
T (n_recurrences) can differ between train and test time — this is the key
mechanism for depth extrapolation.

Input injection: at each loop iteration, the original prelude output is
concatenated with the current hidden state and projected back to n_hidden.
This prevents information decay across loop iterations (Geiping et al., 2025).
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx

from model.gated_delta_net import GDNBlock
from model.mup import MuReadout
from model.output import apply_output_projection, validate_output_config
from model.transformer import (
    TransformerBlock,
    rotary_pos_embedding,
    sinusoidal_pos_embedding,
)


def _make_block(
    btype: str,
    *,
    n_hidden: int,
    n_heads: int,
    n_mlp_hidden: int,
    use_bias: bool,
    layer_norm: bool,
    use_swiglu: bool,
    dropout_rate: float,
    use_mup: bool,
    use_sow: bool,
    rotary_cos,
    rotary_sin,
    compute_dtype,
    param_dtype,
    rngs: nnx.Rngs,
) -> nnx.Module:
    """Instantiate a single block by type string."""
    if btype == "attn":
        return TransformerBlock(
            n_hidden=n_hidden,
            n_heads=n_heads,
            n_mlp_hidden=n_mlp_hidden,
            use_bias=use_bias,
            layer_norm=layer_norm,
            use_swiglu=use_swiglu,
            dropout_rate=dropout_rate,
            use_mup=use_mup,
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            use_sow=use_sow,
            dtype=compute_dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
    elif btype == "gdn":
        return GDNBlock(
            n_hidden=n_hidden,
            n_heads=n_heads,
            n_mlp_hidden=n_mlp_hidden,
            use_bias=use_bias,
            layer_norm=layer_norm,
            use_swiglu=use_swiglu,
            dropout_rate=dropout_rate,
            dtype=compute_dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
    else:
        raise ValueError(f"Unknown block type: {btype!r}")


@dataclass
class RecursiveArchConfig:
    """Config for recursive (depth-recurrent) architectures.

    Architecture layout:
        1 prelude Attn + [core_pattern × 4] looped n_recurrences times + 1 coda Attn

    The core is weight-tied: the same 4 blocks are reused across all loop
    iterations. n_recurrences can be overridden at inference time.

    Effective depth comparison:
        - RecursiveArchConfig(core_block="attn", n_recurrences=T)
          is the recursive counterpart of
          ArchConfig(core_block="attn", n_core_repeats=T)
        - Same structure, but the recursive version shares core weights.
    """
    # Vocabulary / sequence
    n_vocab: int | None = None
    n_seq: int = 128
    n_out: int = 1
    n_pred_tokens: int = 1
    pad_token_id: int = 0

    # Hidden dimensions
    n_hidden: int = 128
    n_heads: int = 4
    n_mlp_hidden: int | None = None

    # Architecture structure
    core_block: str = "attn"        # "attn" | "hybrid"
    n_recurrences: int = 1          # T — default loop count (can override at call time)

    # Block options
    pos_encoding: str = "rope"
    layer_norm: bool = True
    use_swiglu: bool = False
    use_bias: bool = True
    dropout_rate: float = 0.0
    output_mode: str = "full_sequence"
    use_mup: bool = False
    use_sow: bool = False
    use_bf16: bool = True

    @property
    def n_layers_unrolled(self) -> int:
        """Total blocks when unrolled (for comparison with non-recursive)."""
        return 4 * self.n_recurrences + 2

    @property
    def n_params_blocks(self) -> int:
        """Number of unique block parameter sets (prelude + core + coda)."""
        return 1 + 4 + 1  # always 6, regardless of n_recurrences

    def to_model(self, *, rngs: nnx.Rngs) -> "RecursiveArchitecture":
        return RecursiveArchitecture(self, rngs=rngs)


class RecursiveArchitecture(nnx.Module):
    """Recursive architecture: Attn + [core] × T (weight-tied) + Attn.

    The core blocks are instantiated once and applied T times per forward pass.
    Input injection concatenates the prelude output with the current hidden
    state at each iteration and projects back to n_hidden.

    Use n_recurrences kwarg in __call__ to override T at test time.
    """

    def __init__(self, config: RecursiveArchConfig, *, rngs: nnx.Rngs):
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
                f"pos_encoding must be 'absolute', 'rope', or 'none', got {config.pos_encoding!r}"
            )
        validate_output_config(config.output_mode, config.n_pred_tokens)

        # Shared block kwargs
        block_kw = dict(
            n_hidden=config.n_hidden,
            n_heads=config.n_heads,
            n_mlp_hidden=n_mlp_hidden,
            use_bias=config.use_bias,
            layer_norm=config.layer_norm,
            use_swiglu=config.use_swiglu,
            dropout_rate=config.dropout_rate,
            use_mup=config.use_mup,
            use_sow=config.use_sow,
            compute_dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
        )

        # ── Token embedding ──────────────────────────────────────────
        if config.n_vocab is not None:
            self.embed = nnx.Embed(
                num_embeddings=config.n_vocab,
                features=config.n_hidden,
                dtype=self.compute_dtype,
                param_dtype=self.param_dtype,
                rngs=rngs,
            )
        else:
            self.embed = None

        # ── Positional embeddings ────────────────────────────────────
        if config.pos_encoding == "absolute":
            self.pos_embedding = sinusoidal_pos_embedding(
                config.n_seq, config.n_hidden
            ).astype(self.compute_dtype)
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

        block_kw["rotary_cos"] = self.rotary_cos
        block_kw["rotary_sin"] = self.rotary_sin

        # ── Prelude: 1 attention block ───────────────────────────────
        self.prelude = _make_block("attn", **block_kw, rngs=rngs)

        # ── Core: 4 blocks (instantiated ONCE, looped T times) ───────
        if config.core_block == "attn":
            core_types = ["attn", "attn", "attn", "attn"]
        elif config.core_block == "hybrid":
            core_types = ["gdn", "gdn", "gdn", "attn"]
        else:
            raise ValueError(
                f"core_block must be 'attn' or 'hybrid', got {config.core_block!r}"
            )
        self.core = nnx.List([
            _make_block(bt, **block_kw, rngs=rngs) for bt in core_types
        ])
        self._core_types = core_types

        # ── Input injection: concat(prelude_out, hidden) → n_hidden ──
        self.input_injection = nnx.Linear(
            2 * config.n_hidden,
            config.n_hidden,
            use_bias=config.use_bias,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            rngs=rngs,
        )

        # ── Coda: 1 attention block ─────────────────────────────────
        self.coda = _make_block("attn", **block_kw, rngs=rngs)

        # ── Final norm + output ──────────────────────────────────────
        if config.layer_norm:
            self.final_ln = nnx.RMSNorm(
                config.n_hidden,
                dtype=self.compute_dtype,
                param_dtype=self.param_dtype,
                rngs=rngs,
            )
        else:
            self.final_ln = None

        out_features = config.n_out * config.n_pred_tokens
        if config.use_mup:
            self.output = MuReadout(
                config.n_hidden, out_features,
                use_bias=config.use_bias,
                dtype=self.compute_dtype,
                param_dtype=self.param_dtype,
                rngs=rngs,
            )
        else:
            self.output = nnx.Linear(
                config.n_hidden, out_features,
                use_bias=config.use_bias,
                dtype=self.compute_dtype,
                param_dtype=self.param_dtype,
                rngs=rngs,
            )

        # Causal mask
        self._causal_mask = jnp.tril(jnp.ones((config.n_seq, config.n_seq), dtype=bool))

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        n_recurrences: int | None = None,
        cache=None,
        return_cache: bool = False,
    ):
        """Forward pass.

        Args:
            x: (batch, seq) token IDs or (batch, seq, n_hidden) embeddings.
            n_recurrences: Override loop count T. If None, uses config default.
                Set higher at test time to extrapolate beyond training depth.
            cache: Not supported for recursive models (raises if provided).
            return_cache: Not supported for recursive models.

        Returns:
            Output logits shaped per output_mode / n_pred_tokens / n_out.
        """
        if cache is not None or return_cache:
            raise NotImplementedError(
                "KV cache is not supported for recursive architectures. "
                "Use the non-recursive Architecture for cached inference."
            )

        config = self.config
        T = n_recurrences if n_recurrences is not None else config.n_recurrences
        tokens = None

        # ── Embed ────────────────────────────────────────────────────
        if self.embed is not None:
            assert x.ndim == 2, f"Expected (batch, seq) for token IDs, got {x.shape}"
            tokens = x
            x = self.embed(x)
        else:
            assert x.ndim == 3, f"Expected (batch, seq, features), got {x.shape}"
            if config.output_mode == "last_nonpad":
                raise ValueError("last_nonpad requires n_vocab to be set")

        x = x.astype(self.compute_dtype)
        batch_size, seq_len, _ = x.shape

        if seq_len > config.n_seq:
            raise ValueError(f"seq_len ({seq_len}) exceeds n_seq ({config.n_seq})")

        # ── Positional embeddings ────────────────────────────────────
        if self.pos_embedding is not None:
            x = x + self.pos_embedding[:seq_len]

        # Causal mask for attention blocks
        mask = None
        causal_flag = True

        # ── Prelude ──────────────────────────────────────────────────
        x = self.prelude(x, mask=mask, is_causal=causal_flag)
        prelude_out = x  # saved for input injection

        # ── Looped core ──────────────────────────────────────────────
        for _t in range(T):
            # Input injection: concat prelude output with current state
            x = self.input_injection(
                jnp.concatenate([prelude_out, x], axis=-1)
            )

            # Apply the 4 core blocks (same weights each iteration)
            for block, btype in zip(self.core, self._core_types):
                if btype == "attn":
                    x = block(x, mask=mask, is_causal=causal_flag)
                else:
                    x = block(x)

        # ── Coda ─────────────────────────────────────────────────────
        x = self.coda(x, mask=mask, is_causal=causal_flag)

        # ── Final norm + output ──────────────────────────────────────
        if self.final_ln is not None:
            x = self.final_ln(x)

        return apply_output_projection(
            x, self.output,
            output_mode=config.output_mode,
            n_pred_tokens=config.n_pred_tokens,
            n_out=config.n_out,
            tokens=tokens,
            pad_token_id=config.pad_token_id,
        )
