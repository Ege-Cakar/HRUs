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

ACT (Adaptive Computation Time), following Ouro/LoopLM (Zhu et al., 2025):
When use_act=True, a learned exit gate λ_t(x) = σ(Linear(h^(t))) predicts
per-token instantaneous exit probability at each iteration t.

Exit distribution (geometric with remainder):
  p_φ(t|x) = λ_t * S_{t-1}   for t = 1, ..., T_max - 1
  p_φ(T_max|x) = S_{T_max-1}  (remainder)
  where S_t = Π_{j=1}^{t} (1 - λ_j)  is the survival probability.

Stage I training (ELBO with uniform prior):
  L = Σ_t p_φ(t|x) * L^(t) - β * H(p_φ(·|x))

  where L^(t) is the CE loss at iteration t and H is entropy.
  The entropy term prevents collapse to always using T_max.

Stage II training (adaptive gate fine-tuning, LM frozen):
  Compute loss improvement I_i^(t) = max(0, L^(t-1) - L^(t)) per token i.
  Ideal continuation label w_i^(t) = σ(k * (I_i^(t) - γ)).
  Train gate via BCE between (1 - λ_i^(t)) and w_i^(t).

Inference (CDF-based early exit):
  Exit at first step where CDF(n|x) = 1 - S_n(x) >= threshold q.

Use return_aux=True in __call__ to get per-iteration logits and exit distribution.
Loss helpers: compute_act_stage1_loss(), compute_act_stage2_loss().
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


# ---------------------------------------------------------------------------
# Loss helpers (pure functions, called from training loop)
# ---------------------------------------------------------------------------

def compute_exit_distribution(
    halt_probs: list[jnp.ndarray],
) -> list[jnp.ndarray]:
    """Compute the discrete exit distribution from per-step halt probabilities.

    Args:
        halt_probs: List of T arrays, each (batch, seq) with values in (0, 1).
            halt_probs[t] = λ_t(x), the instantaneous exit probability at step t.

    Returns:
        List of T arrays (batch, seq): p_φ(t|x) for t = 1, ..., T.
        Sums to 1 over t for each (batch, seq) position.
    """
    T = len(halt_probs)
    running_survival = jnp.ones_like(halt_probs[0])  # S_0 = 1
    exit_probs = []

    for t in range(T - 1):
        # p(exit at t) = λ_t * S_{t-1}
        p_exit_t = halt_probs[t] * running_survival
        exit_probs.append(p_exit_t)
        # S_t = S_{t-1} * (1 - λ_t)
        running_survival = running_survival * (1.0 - halt_probs[t])

    # Remainder to last step: p(exit at T) = S_{T-1}
    exit_probs.append(running_survival)
    return exit_probs


def compute_exit_entropy(exit_probs: list[jnp.ndarray], eps: float = 1e-8) -> jnp.ndarray:
    """Entropy of the exit distribution H(p_φ(·|x)).

    Args:
        exit_probs: List of T arrays (batch, seq) from compute_exit_distribution.

    Returns:
        Scalar: mean entropy over batch and sequence positions.
    """
    entropy = jnp.zeros_like(exit_probs[0])
    for p_t in exit_probs:
        entropy = entropy - p_t * jnp.log(p_t + eps)
    return jnp.mean(entropy)


def compute_act_stage1_loss(
    per_iteration_logits: list[jnp.ndarray],
    labels: jnp.ndarray,
    halt_probs: list[jnp.ndarray],
    beta: float = 0.1,
    loss_fn=None,
    pad_token_id: int = 0,
) -> tuple[jnp.ndarray, dict]:
    """Stage I ELBO loss: expected task loss - β * entropy.

    L = Σ_t p_φ(t|x) * L^(t) - β * H(p_φ(·|x))

    Args:
        per_iteration_logits: List of T arrays (batch, seq, vocab).
        labels: (batch, seq) integer labels. Positions with pad_token_id are masked.
        halt_probs: List of T arrays (batch, seq) — instantaneous exit probs.
        beta: Entropy regularization coefficient.
        loss_fn: Per-token loss function(logits, labels) -> (batch, seq).
            Defaults to softmax CE with integer labels.
        pad_token_id: Token ID to mask in loss computation.

    Returns:
        (total_loss, metrics_dict) where metrics_dict has 'expected_task_loss',
        'entropy', 'per_iteration_losses'.
    """
    if loss_fn is None:
        def loss_fn(logits, labels):
            return jax.vmap(jax.vmap(
                lambda l, y: -jax.nn.log_softmax(l)[y]
            ))(logits, labels)

    T = len(per_iteration_logits)
    exit_probs = compute_exit_distribution(halt_probs)

    # Compute per-iteration masked losses: L^(t) scalar
    mask = (labels != pad_token_id).astype(jnp.float32)
    mask_sum = jnp.maximum(jnp.sum(mask), 1.0)

    per_iter_losses = []
    for t in range(T):
        token_losses = loss_fn(per_iteration_logits[t], labels)  # (batch, seq)
        masked_loss = jnp.sum(token_losses * mask) / mask_sum
        per_iter_losses.append(masked_loss)

    # Expected task loss: Σ_t E[p_φ(t|x)] * L^(t)
    # p_φ(t|x) is per-token, L^(t) is per-token, so we weight per-token
    expected_task_loss = jnp.zeros(())
    for t in range(T):
        token_losses = loss_fn(per_iteration_logits[t], labels)  # (batch, seq)
        # Weight each token's loss by its exit probability at this step
        weighted = token_losses * exit_probs[t] * mask
        expected_task_loss = expected_task_loss + jnp.sum(weighted) / mask_sum

    # Entropy regularization
    entropy = compute_exit_entropy(exit_probs)

    total_loss = expected_task_loss - beta * entropy

    metrics = {
        "expected_task_loss": expected_task_loss,
        "entropy": entropy,
        "per_iteration_losses": jnp.stack(per_iter_losses),
    }
    return total_loss, metrics


def compute_act_stage2_loss(
    per_iteration_logits: list[jnp.ndarray],
    labels: jnp.ndarray,
    halt_probs: list[jnp.ndarray],
    loss_fn=None,
    pad_token_id: int = 0,
    k: float = 50.0,
    gamma: float = 0.005,
) -> tuple[jnp.ndarray, dict]:
    """Stage II adaptive gate loss (LM frozen, only gate trained).

    For each token i and step t >= 2, compute:
      I_i^(t) = max(0, L_i^(t-1) - L_i^(t))     (loss improvement)
      w_i^(t) = σ(k * (I_i^(t) - γ))             (ideal continuation label)

    Then BCE between gate's continuation probability (1 - λ_i^(t)) and w_i^(t).

    Args:
        per_iteration_logits: List of T arrays (batch, seq, vocab).
            These should be DETACHED from the LM (stop gradient).
        labels: (batch, seq) integer labels.
        halt_probs: List of T arrays (batch, seq) — instantaneous exit probs.
        loss_fn: Per-token loss function(logits, labels) -> (batch, seq).
        pad_token_id: Token ID to mask.
        k: Sigmoid slope for ideal label (default 50.0 from Ouro).
        gamma: Improvement threshold (default 0.005 from Ouro).

    Returns:
        (adaptive_loss, metrics_dict)
    """
    if loss_fn is None:
        def loss_fn(logits, labels):
            return jax.vmap(jax.vmap(
                lambda l, y: -jax.nn.log_softmax(l)[y]
            ))(logits, labels)

    T = len(per_iteration_logits)
    mask = (labels != pad_token_id).astype(jnp.float32)
    mask_sum = jnp.maximum(jnp.sum(mask), 1.0)

    # Compute per-token losses at each step (detached from LM)
    per_token_losses = []
    for t in range(T):
        logits_stopped = jax.lax.stop_gradient(per_iteration_logits[t])
        per_token_losses.append(loss_fn(logits_stopped, labels))

    # Adaptive BCE loss averaged over steps t=2..T
    total_bce = jnp.zeros(())
    n_steps = 0

    for t in range(1, T):  # t >= 2 in 1-indexed = t >= 1 in 0-indexed
        # Loss improvement
        improvement = jnp.maximum(0.0, per_token_losses[t - 1] - per_token_losses[t])
        # Ideal continuation label
        w = jax.nn.sigmoid(k * (improvement - gamma))

        # Gate's predicted continuation probability = 1 - λ_t
        cont_prob = 1.0 - halt_probs[t]

        # BCE: w * log(cont_prob) + (1-w) * log(λ_t)
        eps = 1e-8
        bce = -(w * jnp.log(cont_prob + eps) + (1.0 - w) * jnp.log(halt_probs[t] + eps))
        total_bce = total_bce + jnp.sum(bce * mask) / mask_sum
        n_steps += 1

    adaptive_loss = total_bce / max(n_steps, 1)

    metrics = {
        "adaptive_loss": adaptive_loss,
        "n_steps": n_steps,
    }
    return adaptive_loss, metrics


# ---------------------------------------------------------------------------
# Block builder
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

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

    # ACT (Adaptive Computation Time) — following Ouro (Zhu et al., 2025)
    use_act: bool = False             # Enable learned exit gate
    act_beta: float = 0.1            # β for entropy regularization in Stage I
    act_halt_threshold: float = 0.5   # CDF threshold q for early exit at inference

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


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class RecursiveOutput:
    """Output container for recursive architecture with ACT.

    Attributes:
        logits: Final output logits. If use_act: last iteration's logits (use
            per_iteration_logits + exit_distribution for the proper ACT loss).
            If not use_act: logits from the final iteration.
        per_iteration_logits: List of T arrays (batch, seq, vocab) — logits at
            each core iteration, obtained by applying coda + output at each step.
        halt_probs: List of T arrays (batch, seq) — instantaneous exit probability
            λ_t(x) = σ(Linear(h^(t))) at each iteration. Only if use_act=True.
        exit_distribution: List of T arrays (batch, seq) — the proper discrete
            exit distribution p_φ(t|x) computed from halt_probs. Sums to 1.
            Only if use_act=True.
        n_iterations: Number of core iterations executed.
    """
    logits: jnp.ndarray
    per_iteration_logits: list[jnp.ndarray] = field(default_factory=list)
    halt_probs: list[jnp.ndarray] = field(default_factory=list)
    exit_distribution: list[jnp.ndarray] = field(default_factory=list)
    n_iterations: int = 0


# ---------------------------------------------------------------------------
# Halt gate
# ---------------------------------------------------------------------------

class HaltGate(nnx.Module):
    """Per-token instantaneous exit probability for ACT.

    λ_t(x) = σ(Linear_φ(h^(t))) ∈ (0, 1)
    """

    def __init__(self, n_hidden: int, *, dtype, param_dtype, rngs: nnx.Rngs):
        self.linear = nnx.Linear(
            n_hidden, 1, use_bias=True,
            dtype=dtype, param_dtype=param_dtype, rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x: (batch, seq, n_hidden) -> (batch, seq) exit probabilities."""
        return jax.nn.sigmoid(self.linear(x).squeeze(-1))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

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
                logits, halt probs, and exit distribution. Required for ACT
                training loss computation.
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
        # Always use the last iteration's coda output as the returned logits.
        # For ACT training, use per_iteration_logits + exit_distribution
        # with compute_act_stage1_loss() to get the proper ELBO.
        logits = self._apply_coda_and_output(x, tokens)

        if return_aux:
            exit_dist = []
            if halt_probs:
                exit_dist = compute_exit_distribution(halt_probs)

            return RecursiveOutput(
                logits=logits,
                per_iteration_logits=per_iteration_logits,
                halt_probs=halt_probs,
                exit_distribution=exit_dist,
                n_iterations=T,
            )
        return logits
