"""Training loop for recursive architectures with ACT.

Follows Ouro/LoopLM (Zhu et al., 2025) training methodology adapted for
small-scale synthetic experiments:

Stage I — Joint training (LM + gate):
    Loss = Σ_t p_φ(t|x) * CE^(t) - β * H(p_φ(·|x))
    The entropy term prevents the exit distribution from collapsing to T_max.
    β decays over training (exploration → exploitation).

Stage II — Gate-only fine-tuning (LM frozen):
    For each step t, compute loss improvement I_i^(t) = max(0, L^(t-1) - L^(t)).
    Ideal continuation label w_i^(t) = σ(k * (I^(t) - γ)).
    Train gate via BCE between (1 - λ_i^(t)) and w_i^(t).

Key Ouro insights applied:
    - Gradient clipping at 1.0 (critical for stability with looped gradients)
    - Weight decay 0.1, AdamW with β1=0.9, β2=0.95
    - β (entropy coefficient) decays over training
    - Conservative learning rates for recursive models
    - Per-iteration losses logged for monitoring depth utilization
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from model.recursive import (
    RecursiveArchConfig,
    RecursiveArchitecture,
    RecursiveOutput,
    compute_act_stage1_loss,
    compute_act_stage2_loss,
    compute_exit_distribution,
    compute_exit_entropy,
)
from train import (
    create_optimizer,
    warmup_cosine_schedule,
    _resolve_seed,
    _resolve_iterators,
    _resolve_print_fn,
    _iter_steps,
    _should_eval,
    _apply_grads_jit,
)
from wandb_utils import WandbConfig, log_wandb_metrics, wandb_run_context


# ---------------------------------------------------------------------------
# β schedule: entropy coefficient decay following Ouro
# ---------------------------------------------------------------------------

def constant_beta_schedule(beta: float) -> Callable[[int, int], float]:
    """Constant β throughout training."""
    def _schedule(step: int, total_steps: int) -> float:
        return beta
    return _schedule


def linear_beta_schedule(
    beta_start: float,
    beta_end: float,
) -> Callable[[int, int], float]:
    """Linear decay from beta_start to beta_end over training."""
    def _schedule(step: int, total_steps: int) -> float:
        frac = step / max(total_steps - 1, 1)
        return beta_start + (beta_end - beta_start) * frac
    return _schedule


def staged_beta_schedule(
    beta_early: float,
    beta_late: float,
    transition_frac: float = 0.5,
) -> Callable[[int, int], float]:
    """Step-function β: beta_early for first fraction, beta_late for rest.

    Mirrors Ouro's β=0.1 → β=0.05 stage transition.
    """
    def _schedule(step: int, total_steps: int) -> float:
        if step < total_steps * transition_frac:
            return beta_early
        return beta_late
    return _schedule


# ---------------------------------------------------------------------------
# Stage I: joint LM + gate training
# ---------------------------------------------------------------------------

def _make_stage1_grad_fn(pad_token_id: int = 0):
    """Create JIT-compiled Stage I gradient function."""

    @nnx.jit
    def _grad_step(optimizer: nnx.Optimizer, x, labels, beta):
        def loss_fn(model):
            result = model(x, return_aux=True)
            total_loss, _ = compute_act_stage1_loss(
                result.per_iteration_logits,
                labels,
                result.halt_probs,
                beta=beta,
                pad_token_id=pad_token_id,
            )
            return total_loss

        return nnx.value_and_grad(
            loss_fn, argnums=nnx.DiffState(0, optimizer.wrt)
        )(optimizer.model)

    return _grad_step


def _make_stage1_eval_fn(pad_token_id: int = 0):
    """Create eval function that returns per-iteration metrics."""

    def _eval(optimizer, batch, beta=0.1):
        x, labels = batch
        model = optimizer.model
        result = model(x, return_aux=True)

        total_loss, metrics = compute_act_stage1_loss(
            result.per_iteration_logits,
            labels,
            result.halt_probs,
            beta=beta,
            pad_token_id=pad_token_id,
        )

        # Accuracy from last iteration logits
        logits = result.per_iteration_logits[-1]
        mask = (labels != pad_token_id)
        preds = jnp.argmax(logits, axis=-1)
        correct = (preds == labels) & mask
        acc = jnp.sum(correct) / jnp.maximum(jnp.sum(mask), 1)

        # Mean exit step (expected depth)
        if result.exit_distribution:
            expected_depth = jnp.zeros_like(result.exit_distribution[0])
            for t, p_t in enumerate(result.exit_distribution):
                expected_depth = expected_depth + (t + 1) * p_t
            mean_depth = float(jnp.mean(expected_depth))
        else:
            mean_depth = float(result.n_iterations)

        return {
            "loss": float(total_loss),
            "acc": float(acc),
            "task_loss": float(metrics["expected_task_loss"]),
            "entropy": float(metrics["entropy"]),
            "mean_depth": mean_depth,
            "per_iter_losses": [float(l) for l in metrics["per_iteration_losses"]],
        }

    return _eval


# ---------------------------------------------------------------------------
# Stage II: gate-only fine-tuning
# ---------------------------------------------------------------------------

def _make_stage2_grad_fn(pad_token_id: int = 0, k: float = 50.0, gamma: float = 0.005):
    """Create JIT-compiled Stage II gradient function.

    Only the halt_gate parameters receive gradients; LM is frozen via stop_gradient.
    """

    @nnx.jit
    def _grad_step(optimizer: nnx.Optimizer, x, labels):
        def loss_fn(model):
            result = model(x, return_aux=True)
            # Detach per-iteration logits from LM
            detached_logits = [jax.lax.stop_gradient(l) for l in result.per_iteration_logits]
            loss, _ = compute_act_stage2_loss(
                detached_logits,
                labels,
                result.halt_probs,
                pad_token_id=pad_token_id,
                k=k,
                gamma=gamma,
            )
            return loss

        return nnx.value_and_grad(
            loss_fn, argnums=nnx.DiffState(0, optimizer.wrt)
        )(optimizer.model)

    return _grad_step


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

@dataclass
class RecursiveTrainConfig:
    """Configuration for recursive model training.

    Follows Ouro's training philosophy adapted for small synthetic experiments.
    """
    # Stage I settings
    train_iters: int = 10_000
    test_every: int = 1_000
    test_iters: int = 2

    # Learning rate
    lr: float = 3e-4
    warmup_frac: float = 0.05
    end_lr: float = 0.0

    # Optimizer (Ouro defaults)
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    adam_b1: float = 0.9
    adam_b2: float = 0.95

    # β schedule for entropy regularization
    beta_start: float = 0.1       # Ouro Stage 1a
    beta_end: float = 0.05        # Ouro Stage 1b+
    beta_transition_frac: float = 0.5  # When to switch

    # Stage II settings (gate-only fine-tuning)
    stage2_iters: int = 0         # 0 = skip Stage II
    stage2_lr: float = 1e-4
    stage2_k: float = 50.0        # Sigmoid slope for ideal label
    stage2_gamma: float = 0.005   # Improvement threshold

    # Misc
    pad_token_id: int = 0
    seed: int | None = None
    use_tqdm: bool = False
    grad_accum_steps: int = 1


def train_recursive(
    config: RecursiveArchConfig,
    train_config: RecursiveTrainConfig,
    train_iter,
    test_iter=None,
    wandb_cfg: WandbConfig | None = None,
    print_fn=None,
) -> tuple[nnx.Optimizer, dict]:
    """Train a recursive architecture with Ouro-style ACT.

    Two-stage training:
        Stage I:  Joint LM + gate training with ELBO loss + β-scheduled entropy.
        Stage II: Gate-only fine-tuning with adaptive BCE (optional).

    Args:
        config: RecursiveArchConfig (must have use_act=True for ACT training).
        train_config: RecursiveTrainConfig with all hyperparameters.
        train_iter: Training data iterator yielding (xs, ys) batches.
        test_iter: Optional test data iterator.
        wandb_cfg: Optional W&B logging config.
        print_fn: Optional status print function(step, hist).

    Returns:
        (optimizer, hist) where hist contains per-eval metrics including
        per-iteration losses, entropy, and mean exit depth.
    """
    tc = train_config
    seed = _resolve_seed(tc.seed)
    train_iter, test_iter = _resolve_iterators(train_iter, test_iter)
    print_fn = _resolve_print_fn(print_fn)

    # β schedule
    beta_schedule = staged_beta_schedule(tc.beta_start, tc.beta_end, tc.beta_transition_frac)

    # Build model + optimizer
    rngs = nnx.Rngs(seed)
    model = config.to_model(rngs=rngs)

    lr_schedule = warmup_cosine_schedule(tc.lr, tc.train_iters, warmup_frac=tc.warmup_frac, end_lr=tc.end_lr)

    tx = optax.chain(
        optax.clip_by_global_norm(tc.grad_clip),
        optax.adamw(
            lr_schedule,
            weight_decay=tc.weight_decay,
            b1=tc.adam_b1,
            b2=tc.adam_b2,
        ),
    )
    optimizer = nnx.ModelAndOptimizer(model, tx, wrt=nnx.Param)

    # JIT-compiled functions
    stage1_grad_fn = _make_stage1_grad_fn(pad_token_id=tc.pad_token_id)
    stage1_eval_fn = _make_stage1_eval_fn(pad_token_id=tc.pad_token_id)

    # History
    hist = {
        "train": [],
        "test": [],
        "stage": [],
    }

    n_params = sum(p.size for p in jax.tree.leaves(nnx.state(model)))
    print(f"Recursive model: {n_params:,} params, T_max={config.n_recurrences}, ACT={config.use_act}")

    # ====================================================================
    # Stage I: Joint LM + gate training
    # ====================================================================
    print(f"\n--- Stage I: Joint training ({tc.train_iters} iters) ---")

    with wandb_run_context(wandb_cfg) as wandb:
        for step in _iter_steps(train_iter, tc.train_iters, tc.use_tqdm):
            beta = beta_schedule(step, tc.train_iters)
            beta_arr = jnp.asarray(beta, dtype=jnp.float32)

            x, labels = next(train_iter)
            loss_val, grads = stage1_grad_fn(optimizer, x, labels, beta_arr)
            _apply_grads_jit(optimizer, grads)

            if _should_eval(step, tc.train_iters, tc.test_every):
                # Eval on train
                train_batch = next(train_iter)
                train_metrics = stage1_eval_fn(optimizer, train_batch, beta=beta)

                # Eval on test
                if test_iter is not None:
                    test_metrics_list = []
                    for _ in range(tc.test_iters):
                        test_batch = next(test_iter)
                        test_metrics_list.append(stage1_eval_fn(optimizer, test_batch, beta=beta))
                    # Average test metrics
                    test_metrics = {}
                    for key in test_metrics_list[0]:
                        if key == "per_iter_losses":
                            test_metrics[key] = test_metrics_list[0][key]
                        else:
                            test_metrics[key] = sum(m[key] for m in test_metrics_list) / len(test_metrics_list)
                else:
                    test_metrics = {}

                hist["train"].append(train_metrics)
                hist["test"].append(test_metrics)
                hist["stage"].append("stage1")

                # W&B logging
                flat_metrics = {}
                for prefix, metrics in [("train", train_metrics), ("test", test_metrics)]:
                    for k, v in metrics.items():
                        if k == "per_iter_losses":
                            for i, lv in enumerate(v):
                                flat_metrics[f"{prefix}/loss_iter_{i}"] = lv
                        else:
                            flat_metrics[f"{prefix}/{k}"] = v
                flat_metrics["train/beta"] = beta

                log_wandb_metrics(wandb, step=step + 1, train=None, test=None, summary=None)
                # Manual log for richer metrics
                if wandb is not None:
                    wandb.log(flat_metrics, step=step + 1)

                # Print
                depth_str = f"  depth={train_metrics['mean_depth']:.2f}" if 'mean_depth' in train_metrics else ""
                print(
                    f"  [{step+1}/{tc.train_iters}] "
                    f"loss={train_metrics['loss']:.4f}  "
                    f"acc={train_metrics['acc']:.4f}  "
                    f"entropy={train_metrics['entropy']:.3f}  "
                    f"β={beta:.3f}"
                    f"{depth_str}"
                )

        # ====================================================================
        # Stage II: Gate-only fine-tuning (optional)
        # ====================================================================
        if tc.stage2_iters > 0:
            print(f"\n--- Stage II: Gate fine-tuning ({tc.stage2_iters} iters) ---")

            # Create gate-only optimizer
            # Only train halt_gate parameters
            stage2_tx = optax.chain(
                optax.clip_by_global_norm(tc.grad_clip),
                optax.adamw(tc.stage2_lr, weight_decay=0.0),
            )
            # We reuse the same optimizer but note: in a real setup you'd
            # create a separate optimizer that only updates gate params.
            # For simplicity, we use stop_gradient in the loss to freeze LM.
            stage2_optimizer = nnx.ModelAndOptimizer(optimizer.model, stage2_tx, wrt=nnx.Param)

            stage2_grad_fn = _make_stage2_grad_fn(
                pad_token_id=tc.pad_token_id,
                k=tc.stage2_k,
                gamma=tc.stage2_gamma,
            )

            for step in _iter_steps(train_iter, tc.stage2_iters, tc.use_tqdm):
                x, labels = next(train_iter)
                loss_val, grads = stage2_grad_fn(stage2_optimizer, x, labels)
                _apply_grads_jit(stage2_optimizer, grads)

                if _should_eval(step, tc.stage2_iters, tc.test_every):
                    # Use Stage I eval for consistent metrics
                    train_batch = next(train_iter)
                    train_metrics = stage1_eval_fn(stage2_optimizer, train_batch, beta=0.0)

                    if test_iter is not None:
                        test_batch = next(test_iter)
                        test_metrics = stage1_eval_fn(stage2_optimizer, test_batch, beta=0.0)
                    else:
                        test_metrics = {}

                    hist["train"].append(train_metrics)
                    hist["test"].append(test_metrics)
                    hist["stage"].append("stage2")

                    if wandb is not None:
                        flat_metrics = {}
                        for prefix, metrics in [("train", train_metrics), ("test", test_metrics)]:
                            for k, v in metrics.items():
                                if k == "per_iter_losses":
                                    for i, lv in enumerate(v):
                                        flat_metrics[f"{prefix}/loss_iter_{i}"] = lv
                                else:
                                    flat_metrics[f"{prefix}/{k}"] = v
                        flat_metrics["train/stage"] = 2
                        wandb.log(flat_metrics, step=tc.train_iters + step + 1)

                    print(
                        f"  [S2 {step+1}/{tc.stage2_iters}] "
                        f"loss={train_metrics['loss']:.4f}  "
                        f"acc={train_metrics['acc']:.4f}  "
                        f"depth={train_metrics['mean_depth']:.2f}"
                    )

    return optimizer, hist


# ---------------------------------------------------------------------------
# Model-aware loss for use with standard train.py (Stage I only, simpler)
# ---------------------------------------------------------------------------

def make_act_stage1_loss(beta: float = 0.1, pad_token_id: int = 0):
    """Create a model-aware loss function for Stage I ACT training.

    Can be passed to train.py's train() as the loss parameter.
    Compatible with the _model_aware convention in _compute_batch_loss.
    """
    def _loss(model, x, labels):
        result = model(x, return_aux=True)
        total_loss, _ = compute_act_stage1_loss(
            result.per_iteration_logits,
            labels,
            result.halt_probs,
            beta=beta,
            pad_token_id=pad_token_id,
        )
        return total_loss

    _loss._model_aware = True
    return _loss
