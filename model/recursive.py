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

ACT (Adaptive Computation Time):
When use_act=True, a learned halt gate predicts p_halt per token per iteration.
During training, the output is a weighted mixture of per-iteration outputs using
a geometric halting distribution, with a ponder cost penalty. During eval, the
model can early-exit when cumulative halt probability exceeds a threshold.

Training loss with ACT:
    total_loss = task_loss(act_weighted_logits, y) + ponder_weight * ponder_cost

Use return_aux=True in __call__ to get per-iteration logits and ponder cost.
"""
from dataclasses import dataclass, field

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

    # ACT (Adaptive Computation Time)
    use_act: bool = False             # Enable learned halting
    act_ponder_weight: float = 0.01   # Lambda for ponder cost in total loss
    act_halt_threshold: float = 0.5   # Cumulative halt threshold for early exit at inference

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


@dataclass
class RecursiveOutput:
    """Output container for recursive architecture with ACT.

    Attributes:
        logits: Final output logits (ACT-weighted if use_act, else last iteration).
        per_iteration_logits: Logits at each core iteration (for aux losses / ACT).
        halt_probs: Per-iteration halt probabilities (batch, seq) if ACT enabled.
        ponder_cost: Scalar mean ponder cost for ACT regularization.
        n_iterations: Number of core iterations executed.
    """
    logits: jnp.ndarray
    per_iteration_logits: list[jnp.ndarray] = field(default_factory=list)
    halt_probs: list[jnp.ndarray] = field(default_factory=list)
    ponder_cost: jnp.ndarray | None = None
    n_iterations: int = 0


class HaltGate(nnx.Module):
    """Per-token halting probability for ACT."""

    def __init__(self, n_hidden: int, *, dtype, param_dtype, rngs: nnx.Rngs):
        self.linear = nnx.Linear(
            n_hidden, 1, use_bias=True,
            dtype=dtype, param_dtype=param_dtype, rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x: (batch, seq, n_hidden) -> (batch, seq) halt probabilities."""
        return jax.nn.sigmoid(self.linear(x).squeeze(-1))


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

        # ACT halt gate (optional)
        if config.use_act:
            self.halt_gate = HaltGate(
                config.n_hidden,
                dtype=self.compute_dtype,
                param_dtype=self.param_dtype,
                rngs=rngs,
            )
        else:
            self.halt_gate = None

    def _apply_coda_and_output(self, x: jnp.ndarray, tokens: jnp.ndarray | None) -> jnp.ndarray:
        """Apply coda block, final norm, and output projection."""
        h = self.coda(x, is_causal=True)
        if self.final_ln is not None:
            h = self.final_ln(h)
        return apply_output_projection(
            h, self.output,
            output_mode=self.config.output_mode,
            n_pred_tokens=self.config.n_pred_tokens,
            n_out=self.config.n_out,
            tokens=tokens,
            pad_token_id=self.config.pad_token_id,
        )

    def _act_aggregate(
        self,
        per_iteration_logits: list[jnp.ndarray],
        halt_probs: list[jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Aggregate per-iteration outputs using geometric halting distribution.

        p(exit at t) = halt_t * prod(1 - halt_k, k < t)
        Remainder probability assigned to the final iteration.

        Returns:
            (weighted_logits, ponder_cost)
        """
        T = len(halt_probs)
        running_prob = jnp.ones_like(halt_probs[0])  # (batch, seq)
        exit_weights = []

        for t in range(T - 1):
            p_exit_t = running_prob * halt_probs[t]
            exit_weights.append(p_exit_t)
            running_prob = running_prob * (1.0 - halt_probs[t])

        # Remainder to last iteration
        exit_weights.append(running_prob)

        # Weighted sum of logits
        logits = jnp.zeros_like(per_iteration_logits[0])
        for t in range(T):
            w = exit_weights[t]
            while w.ndim < logits.ndim:
                w = w[..., None]
            logits = logits + w * per_iteration_logits[t]

        # Ponder cost: expected number of iterations
        ponder_cost = jnp.zeros_like(halt_probs[0])
        for t in range(T):
            ponder_cost = ponder_cost + (t + 1) * exit_weights[t]
        ponder_cost = jnp.mean(ponder_cost)

        return logits, ponder_cost

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        n_recurrences: int | None = None,
        return_aux: bool = False,
        cache=None,
        return_cache: bool = False,
    ) -> jnp.ndarray | RecursiveOutput:
        """Forward pass.

        Args:
            x: (batch, seq) token IDs or (batch, seq, n_hidden) embeddings.
            n_recurrences: Override loop count T. If None, uses config default.
                Set higher at test time to extrapolate beyond training depth.
            return_aux: If True, return RecursiveOutput with per-iteration
                logits, halt probs, and ponder cost. Required for ACT training.
            cache: Not supported for recursive models (raises if provided).
            return_cache: Not supported for recursive models.

        Returns:
            If return_aux=False: output logits (standard shape).
            If return_aux=True: RecursiveOutput with full iteration info.
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

        # ── Prelude ──────────────────────────────────────────────────
        x = self.prelude(x, is_causal=True)
        prelude_out = x  # saved for input injection

        # ── Looped core ──────────────────────────────────────────────
        per_iteration_logits = []
        halt_probs = []

        for _t in range(T):
            # Input injection: concat prelude output with current state
            x = self.input_injection(
                jnp.concatenate([prelude_out, x], axis=-1)
            )

            # Apply the 4 core blocks (same weights each iteration)
            for block, btype in zip(self.core, self._core_types):
                if btype == "attn":
                    x = block(x, is_causal=True)
                else:
                    x = block(x)

            # Per-iteration output (for ACT / aux losses)
            if return_aux or config.use_act:
                iter_logits = self._apply_coda_and_output(x, tokens)
                per_iteration_logits.append(iter_logits)

                if self.halt_gate is not None:
                    halt_probs.append(self.halt_gate(x))

        # ── Final output ─────────────────────────────────────────────
        if config.use_act and self.halt_gate is not None and len(halt_probs) > 0:
            # ACT: weighted mixture of per-iteration outputs
            logits, ponder_cost = self._act_aggregate(per_iteration_logits, halt_probs)
        else:
            # Fixed depth: single pass through coda + output
            logits = self._apply_coda_and_output(x, tokens)
            ponder_cost = None

        if return_aux:
            return RecursiveOutput(
                logits=logits,
                per_iteration_logits=per_iteration_logits,
                halt_probs=halt_probs,
                ponder_cost=ponder_cost,
                n_iterations=T,
            )
        return logits
