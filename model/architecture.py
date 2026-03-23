"""
Unified architecture model supporting Standard Transformer and Hybrid (GDN+Attn).

Structure: Attn + [core_pattern] × k + Attn
  - Standard:  Attn + [Attn Attn Attn Attn] × k + Attn   (4k+2 blocks)
  - Hybrid:    Attn + [GDN  GDN  GDN  Attn] × k + Attn   (4k+2 blocks)

The prelude (first Attn) and coda (last Attn) are always full attention blocks.
The core_pattern is repeated k times with independent weights (non-recursive).
Recursive variants will weight-tie the core repetitions.
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx

from model.gated_delta_net import GDNBlock
from model.mup import MuReadout, mup_attention_scale
from model.output import apply_output_projection, validate_output_config
from model.transformer import (
    MultiHeadAttention,
    TransformerBlock,
    TransformerKVCache,
    create_empty_kv_cache,
    rotary_pos_embedding,
    sinusoidal_pos_embedding,
    _try_int_scalar,
    _validate_kv_cache,
)


@dataclass
class ArchConfig:
    """Unified config for Standard Transformer and Hybrid architectures.

    Architecture layout:
        1 prelude Attn block + [core_pattern × 4] × n_core_repeats + 1 coda Attn block
        Total blocks = 4 * n_core_repeats + 2

    The core_block parameter determines the pattern of each 4-block core unit:
        "attn":   [Attn, Attn, Attn, Attn]  — pure transformer
        "hybrid": [GDN, GDN, GDN, Attn]     — hybrid linear + full attention
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
    n_mlp_hidden: int | None = None  # defaults to 4*n_hidden (GELU) or 8/3*n_hidden (SwiGLU)

    # Architecture structure
    core_block: str = "attn"        # "attn" | "hybrid"
    n_core_repeats: int = 1         # k — number of core unit repetitions

    # Block options
    pos_encoding: str = "rope"      # "none" | "absolute" | "rope"
    layer_norm: bool = True
    use_swiglu: bool = False
    use_bias: bool = True
    dropout_rate: float = 0.0
    output_mode: str = "full_sequence"
    use_mup: bool = False
    use_sow: bool = False
    use_bf16: bool = True

    @property
    def n_layers(self) -> int:
        """Total number of blocks in the architecture."""
        return 4 * self.n_core_repeats + 2

    def to_model(self, *, rngs: nnx.Rngs) -> "Architecture":
        return Architecture(self, rngs=rngs)


def _build_block_sequence(config: ArchConfig, *, rngs: nnx.Rngs) -> list[str]:
    """Return the ordered list of block types for the architecture.

    Returns:
        List of "attn" or "gdn" strings, one per block.
    """
    if config.core_block == "attn":
        core_unit = ["attn", "attn", "attn", "attn"]
    elif config.core_block == "hybrid":
        core_unit = ["gdn", "gdn", "gdn", "attn"]
    else:
        raise ValueError(f"core_block must be 'attn' or 'hybrid', got {config.core_block!r}")

    # prelude + core × k + coda
    return ["attn"] + core_unit * config.n_core_repeats + ["attn"]


class Architecture(nnx.Module):
    """Unified model: Attn + [core_pattern] × k + Attn.

    Follows the same forward interface as Transformer for drop-in compatibility
    with the existing training loop.
    """

    def __init__(self, config: ArchConfig, *, rngs: nnx.Rngs):
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

        # Token embedding
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

        # Positional embeddings
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

        # Build block sequence
        block_types = _build_block_sequence(config, rngs=rngs)
        self._block_types = block_types  # store for introspection

        blocks = []
        for btype in block_types:
            if btype == "attn":
                blocks.append(TransformerBlock(
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
                ))
            elif btype == "gdn":
                blocks.append(GDNBlock(
                    n_hidden=config.n_hidden,
                    n_heads=config.n_heads,
                    n_mlp_hidden=n_mlp_hidden,
                    use_bias=config.use_bias,
                    layer_norm=config.layer_norm,
                    use_swiglu=config.use_swiglu,
                    dropout_rate=config.dropout_rate,
                    dtype=self.compute_dtype,
                    param_dtype=self.param_dtype,
                    rngs=rngs,
                ))
            else:
                raise ValueError(f"Unknown block type: {btype!r}")
        self.blocks = nnx.List(blocks)

        # Final RMS norm
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
        cache: TransformerKVCache | None = None,
        return_cache: bool = False,
    ):
        """Forward pass.

        Args:
            x: (batch, seq) token IDs if n_vocab set, else (batch, seq, n_hidden).
            cache: Optional KV cache for incremental decoding (attn blocks only).
            return_cache: If True, return (output, updated_cache).

        Returns:
            Output logits shaped according to output_mode / n_pred_tokens / n_out.
        """
        config = self.config
        tokens = None

        # Embed tokens
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
        head_dim = config.n_hidden // config.n_heads

        # Count attention blocks for cache sizing
        n_attn_blocks = sum(1 for bt in self._block_types if bt == "attn")

        use_cache = cache is not None or return_cache
        if use_cache:
            if config.output_mode == "last_nonpad":
                raise ValueError("last_nonpad does not support cache-based inference")
            if cache is None:
                # Create cache only for attention blocks
                from dataclasses import dataclass as _dc
                class _FakeConfig:
                    pass
                _fc = _FakeConfig()
                _fc.n_hidden = config.n_hidden
                _fc.n_heads = config.n_heads
                _fc.n_seq = config.n_seq
                _fc.n_layers = n_attn_blocks
                cache = create_empty_kv_cache(_fc, batch_size=batch_size, dtype=x.dtype)
            past_len = _validate_kv_cache(
                cache,
                n_layers=n_attn_blocks,
                batch_size=batch_size,
                n_heads=config.n_heads,
                n_seq=config.n_seq,
                head_dim=head_dim,
            )
            past_keys = list(cache.keys)
            past_values = list(cache.values)
        else:
            if seq_len > config.n_seq:
                raise ValueError(
                    f"seq_len ({seq_len}) exceeds n_seq ({config.n_seq})"
                )
            past_len = jnp.asarray(0, dtype=jnp.int32)
            past_keys = [None] * n_attn_blocks
            past_values = [None] * n_attn_blocks

        # Positional embeddings
        if self.pos_embedding is not None:
            if use_cache:
                pos = jax.lax.dynamic_slice_in_dim(
                    self.pos_embedding,
                    start_index=jnp.asarray(past_len, dtype=jnp.int32),
                    slice_size=seq_len, axis=0,
                )
            else:
                pos = self.pos_embedding[:seq_len]
            x = x + pos

        # Causal mask (only for attention blocks)
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

        # Forward through blocks
        attn_idx = 0
        new_keys = []
        new_values = []

        for block, btype in zip(self.blocks, self._block_types):
            if btype == "attn":
                if return_cache:
                    x, new_key, new_value = block(
                        x, mask=mask, is_causal=causal_flag,
                        past_key=past_keys[attn_idx],
                        past_value=past_values[attn_idx],
                        pos_offset=past_len, return_kv=True,
                    )
                    new_keys.append(new_key)
                    new_values.append(new_value)
                else:
                    x = block(
                        x, mask=mask, is_causal=causal_flag,
                        past_key=past_keys[attn_idx],
                        past_value=past_values[attn_idx],
                        pos_offset=past_len,
                    )
                attn_idx += 1
            else:
                # GDN block — no mask/cache needed
                x = block(x)

        # Final norm
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
