"""Compute-optimal frontier sweep for depth3_fresh_icl FOL task."""

# <codecell>
from __future__ import annotations

from collections import Counter
import itertools
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
LOCAL_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
sys.path.append(str(LOCAL_DIR))

from common import new_seed, split_cases
from model.compute import compute_metrics_from_info, training_flops_total
from model.eval_adapters import make_model_callable
from model.ssm_bonsai import Mamba2BonsaiConfig
from model.transformer import TransformerConfig
from task.layer_fol import (
    AutoregressiveLogitsAdapter,
    FOLLayerTask,
    _build_tokenizer_for_fresh_icl,
    _fresh_predicate_sentinels,
    compute_fol_dims,
    predicted_rule_reaches_goal,
    print_task_preview,
    run_layer_rollout_fol,
)
from task.layer_gen.util.fol_rule_bank import (
    FOLLayerRule,
    build_random_fol_rule_bank,
    parse_atom_text,
    parse_clause_text,
)
from train import Case, ce_mask, warmup_cosine_schedule

from experiment.utils.metrics_utils import final_token_accuracy
from wandb_utils import make_experiment_wandb_config


RUN_ID = new_seed()
print("RUN ID", RUN_ID)

WANDB_PROJECT = Path(__file__).resolve().parent.name
WANDB_API_KEY_PATH = ROOT / "key" / "wandb.txt"
USE_WANDB = True

# --- Fixed task/demo settings (no sweep) ---
EVAL_ROLES = ["eval"]

DEMO_DISTRIBUTION = "zipf_per_rule"
TRAIN_ALPHA = 0
EVAL_ALPHA = 1.0
TRAIN_MIN_N_DEMOS = 1
TRAIN_MAX_N_DEMOS = 32
EVAL_MAX_N_DEMOS = 8
TRAIN_DEMO_RANKED = True
EVAL_DEMO_RANKED = True

BATCH_SIZE = 32
GRAD_ACCUM_STEPS = 1
EFFECTIVE_BATCH_SIZE = int(BATCH_SIZE) * int(GRAD_ACCUM_STEPS)

TRAIN_ITERS_SWEEP = [6400, 25600]
LR_SWEEP = [5e-5, 1e-4, 5e-4]

TEST_ITERS = 2
ROLLOUT_EXAMPLES_PER_ROLE = 50
EVAL_ITERS_PER_ROLE = 128 // BATCH_SIZE
PREVIEW_EXAMPLES_PER_SPLIT = 3

# --- Single task shape ---
BASE_NUM_PRED = 16
MID_PRED = 256

TASK_SHAPE = {
    "predicates_per_layer": (BASE_NUM_PRED, MID_PRED, BASE_NUM_PRED),
    "rules_per_transition": (BASE_NUM_PRED**2, BASE_NUM_PRED**2),
}

N_LAYERS = 3
ARITY_MIN = 0
ARITY_MAX = 0
VARS_PER_RULE_MAX = 6
K_IN_MAX = 1
K_OUT_MAX = 1
INITIAL_ANT_MAX = 1
CONSTANTS = [f"p{i}" for i in range(1)]
SAMPLE_MAX_ATTEMPTS = 4096
MAX_UNIFY_SOLUTIONS = 128
BASE_BANK_SEED = 3053
PREDICATE_NAME_LEN = 4
TRAIN_INCLUDE_ORACLE = False

TRAIN_FIXED_LENGTH_MODE = "next_pow2"
EVAL_FIXED_LENGTH_MODE = "next_pow2"

# --- Model architecture grids ---
# (n_layers, n_hidden, n_heads)
TRANSFORMER_CONFIGS = [
    (2, 128, 4),
    # (2, 256, 8),
    (4, 256, 8),
    # (4, 512, 8),
    (4, 1024, 16),
    # (8, 1024, 16),
    (8, 1536, 16),
    # (8, 2048, 32),
    (12, 2048, 32),
]

MAMBA2_BONSAI_CONFIGS = [
    (2, 128, 4),
    # (2, 256, 8),
    (4, 384, 12),
    # (4, 768, 12),
    (4, 1536, 12),
    # (8, 1536, 12),
    (8, 2048, 16),
    # (12, 2048, 16),
    (12, 3072, 24),
]

# Fixed Mamba2 hyperparams
MAMBA2_D_STATE = 32
MAMBA2_D_CONV = 4
MAMBA2_SCAN_CHUNK_LEN = 64

### START TEST CONFIGS
# TRANSFORMER_CONFIGS = [(2, 128, 4)]
# MAMBA2_BONSAI_CONFIGS = [(2, 128, 4)]
# TRAIN_ITERS_SWEEP = [20]
# LR_SWEEP = [3e-4]
# BATCH_SIZE = 2
# GRAD_ACCUM_STEPS = 1
# EFFECTIVE_BATCH_SIZE = int(BATCH_SIZE) * int(GRAD_ACCUM_STEPS)
# TEST_ITERS = 1
# EVAL_ITERS_PER_ROLE = 1
# ROLLOUT_EXAMPLES_PER_ROLE = 4
# PREVIEW_EXAMPLES_PER_SPLIT = 1
# USE_WANDB = False
### END TEST CONFIGS

RUN_SPLIT = (
    len(TRANSFORMER_CONFIGS) * len(TRAIN_ITERS_SWEEP) * len(LR_SWEEP)
    + len(MAMBA2_BONSAI_CONFIGS) * len(TRAIN_ITERS_SWEEP) * len(LR_SWEEP)
)

RUN_SPLIT


# <codecell>

def _compute_test_every(train_iters: int) -> int:
    return max(1, train_iters // 16)


def _make_case_wandb_cfg(*, case_name, model_config, train_args, info):
    return make_experiment_wandb_config(
        enabled=USE_WANDB,
        project=WANDB_PROJECT,
        run_id=RUN_ID,
        run_name=f"{case_name}-{RUN_ID}",
        api_key_path=WANDB_API_KEY_PATH,
        model_config=model_config,
        train_args=train_args,
        info=info,
    )


def _safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def _ceil_pow2_int(n: int) -> int:
    n = int(n)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _lr_tag(lr: float) -> str:
    return str(lr).replace(".", "p")


def _mean_metrics(metrics_list):
    if not metrics_list:
        return {}

    keys = sorted(metrics_list[0].keys())
    out = {}
    for key in keys:
        vals = [float(m[key]) for m in metrics_list if key in m]
        if vals:
            out[key] = float(np.mean(vals))
    return out


def _normalize_task_shape(task_shape: dict) -> dict:
    return {
        "predicates_per_layer": tuple(int(v) for v in task_shape["predicates_per_layer"]),
        "rules_per_transition": tuple(int(v) for v in task_shape["rules_per_transition"]),
    }


def _task_shape_mid_pred(task_shape: dict) -> int:
    predicates = tuple(int(v) for v in task_shape["predicates_per_layer"])
    if len(predicates) < 2:
        raise ValueError(f"expected at least 2 predicate-layer entries, got {predicates}")
    return int(predicates[1])


def _build_base_bank_and_tokenizer(*, task_shape: dict):
    """Build a 3-layer base bank and the fresh-ICL tokenizer."""
    task_shape = _normalize_task_shape(task_shape)
    base_bank = build_random_fol_rule_bank(
        n_layers=int(N_LAYERS),
        predicates_per_layer=task_shape["predicates_per_layer"],
        rules_per_transition=task_shape["rules_per_transition"],
        arity_min=int(ARITY_MIN),
        arity_max=int(ARITY_MAX),
        vars_per_rule_max=int(VARS_PER_RULE_MAX),
        k_in_max=int(K_IN_MAX),
        k_out_max=int(K_OUT_MAX),
        constants=tuple(str(c) for c in CONSTANTS),
        rng=np.random.default_rng(int(BASE_BANK_SEED)),
    )
    tokenizer = _build_tokenizer_for_fresh_icl(
        base_bank=base_bank,
        predicate_name_len=int(PREDICATE_NAME_LEN),
    )
    return base_bank, tokenizer


def _compute_dims(base_bank, tokenizer, *, max_n_demos_for_shapes: int):
    """Compute tensor dims from the base bank + fresh predicate estimates."""
    sentinels = _fresh_predicate_sentinels(name_len=int(PREDICATE_NAME_LEN))
    extra_arities = {s: int(base_bank.arity_max) for s in sentinels}
    return compute_fol_dims(
        rule_banks=[base_bank],
        tokenizer=tokenizer,
        initial_ant_max=int(INITIAL_ANT_MAX),
        max_n_demos=int(max_n_demos_for_shapes),
        extra_predicate_arities=extra_arities,
        fresh_k_in_max=int(K_IN_MAX),
        fresh_k_out_max=int(K_OUT_MAX),
    )


def _make_layer_task(
    *,
    task_shape: dict,
    split_role: str,
    seed: int,
    drop_remainder: bool,
    shuffle: bool,
    min_n_demos: int,
    max_n_demos: int,
    fixed_length_mode: str,
    fixed_length_n_seq: int,
    include_oracle: bool = False,
    demo_distribution: str = "uniform",
    demo_distribution_alpha: float = 1.0,
    demo_ranked: bool = True,
    batch_size: int | None = None,
):
    task_shape = _normalize_task_shape(task_shape)
    return FOLLayerTask(
        distance_range=(2, 2),
        batch_size=int(batch_size if batch_size is not None else BATCH_SIZE),
        mode="online",
        task_split="depth3_fresh_icl",
        split_role=str(split_role),
        shuffle=shuffle,
        seed=seed,
        worker_count=0,
        drop_remainder=drop_remainder,
        prediction_objective="autoregressive",
        predicates_per_layer=task_shape["predicates_per_layer"],
        rules_per_transition=task_shape["rules_per_transition"],
        fresh_icl_base_bank_seed=int(BASE_BANK_SEED),
        arity_min=int(ARITY_MIN),
        arity_max=int(ARITY_MAX),
        vars_per_rule_max=int(VARS_PER_RULE_MAX),
        constants=tuple(str(tok) for tok in CONSTANTS),
        k_in_max=int(K_IN_MAX),
        k_out_max=int(K_OUT_MAX),
        initial_ant_max=int(INITIAL_ANT_MAX),
        min_n_demos=int(min_n_demos),
        max_n_demos=int(max_n_demos),
        include_oracle=bool(include_oracle),
        sample_max_attempts=int(SAMPLE_MAX_ATTEMPTS),
        max_unify_solutions=int(MAX_UNIFY_SOLUTIONS),
        fixed_length_mode=str(fixed_length_mode),
        fixed_length_n_seq=int(fixed_length_n_seq),
        predicate_name_len=int(PREDICATE_NAME_LEN),
        demo_distribution=str(demo_distribution),
        demo_distribution_alpha=float(demo_distribution_alpha),
        demo_ranked=bool(demo_ranked),
    )


def _estimate_batch_size(
    *,
    n_seq: int,
    base_n_seq: int,
    base_batch_size: int,
) -> int:
    """Scale batch size to keep total token count roughly constant."""
    if n_seq <= base_n_seq:
        return base_batch_size
    return max(1, base_batch_size * base_n_seq // n_seq)


def _estimate_batch_and_iters(
    *,
    n_seq: int,
    base_n_seq: int,
    base_batch_size: int,
    base_n_iters: int,
) -> tuple[int, int]:
    """Scale batch size down and iterations up to keep total samples constant."""
    batch_size = _estimate_batch_size(
        n_seq=n_seq,
        base_n_seq=base_n_seq,
        base_batch_size=base_batch_size,
    )
    if batch_size >= base_batch_size:
        return base_batch_size, base_n_iters
    target_samples = base_batch_size * base_n_iters
    n_iters = max(1, -(-target_samples // batch_size))  # ceil division
    return batch_size, n_iters


def _make_task_shape_bundle(*, task_shape: dict) -> dict:
    """Build bank, tokenizer, and dims for the single fixed task shape."""
    task_shape = _normalize_task_shape(task_shape)
    base_bank, tokenizer = _build_base_bank_and_tokenizer(task_shape=task_shape)

    dims_train = _compute_dims(
        base_bank, tokenizer,
        max_n_demos_for_shapes=int(TRAIN_MAX_N_DEMOS),
    )
    dims_eval = _compute_dims(
        base_bank, tokenizer,
        max_n_demos_for_shapes=int(EVAL_MAX_N_DEMOS),
    )

    max_completion_len = max(
        int(dims_train["max_completion_len"]),
        int(dims_eval["max_completion_len"]),
    )

    train_n_seq_raw = int(dims_train["n_seq_ar"])
    eval_n_seq_raw = int(dims_eval["n_seq_ar"])

    train_n_seq = int(_ceil_pow2_int(train_n_seq_raw))
    eval_n_seq = int(_ceil_pow2_int(eval_n_seq_raw))
    model_n_seq = max(train_n_seq, eval_n_seq)

    n_vocab = max(int(dims_train["n_vocab"]), int(dims_eval["n_vocab"]))

    return {
        "task_shape": task_shape,
        "mid_pred": int(_task_shape_mid_pred(task_shape)),
        "base_bank": base_bank,
        "tokenizer": tokenizer,
        "dims": {"train": dims_train, "eval": dims_eval},
        "train_n_seq_raw": train_n_seq_raw,
        "eval_n_seq_raw": eval_n_seq_raw,
        "train_n_seq": train_n_seq,
        "eval_n_seq": eval_n_seq,
        "model_n_seq": model_n_seq,
        "n_vocab": n_vocab,
        "max_completion_len": int(max_completion_len),
    }


# <codecell>

def make_ar_light_metrics_fn():
    def _metrics(optimizer, batch, loss=None):
        _ = loss
        xs, labels = batch
        logits = optimizer.model(xs)

        loss_val = ce_mask(logits, labels)
        preds = jnp.argmax(logits, axis=-1)
        mask = labels != 0
        total = jnp.maximum(jnp.sum(mask), 1)
        token_acc = jnp.sum((preds == labels) & mask) / total
        final_acc = final_token_accuracy(preds, labels)
        seq_correct = (preds == labels) | (~mask)
        seq_exact_acc = jnp.mean(jnp.all(seq_correct, axis=1))

        return {
            "loss": loss_val,
            "token_acc": token_acc,
            "final_token_acc": final_acc,
            "seq_exact_acc": seq_exact_acc,
        }

    return _metrics


def make_print_fn(metric_key: str):
    def _print(step, hist):
        train_metrics = hist["train"][-1]
        test_metrics = hist["test"][-1]
        print(
            "ITER {}: train_loss={:.4f} train_{}={:.4f} test_loss={:.4f} test_{}={:.4f}".format(
                step,
                train_metrics["loss"],
                metric_key,
                train_metrics[metric_key],
                test_metrics["loss"],
                metric_key,
                test_metrics[metric_key],
            )
        )

    return _print


def _evaluate_role_for_demo(
    optimizer,
    *,
    task_shape: dict,
    role: str,
    tokenizer,
    rule_bank,
    n_seq_ar: int,
    max_completion_len: int,
    n_iters: int,
    eval_max_n_demos: int,
    eval_alpha: float,
    eval_demo_ranked: bool = True,
    demo_distribution: str = DEMO_DISTRIBUTION,
    batch_size: int | None = None,
    model_fn=None,
    shared_adapter=None,
):
    if model_fn is None:
        model_fn = make_model_callable(optimizer, to_numpy=False)
    if shared_adapter is None:
        shared_adapter = AutoregressiveLogitsAdapter(
            n_seq=int(n_seq_ar),
            max_completion_len=int(max_completion_len),
            pad_token_id=0,
            jit_step=True,
        )

    metrics_fn = make_ar_light_metrics_fn()

    seq_len_counts: Counter[int] = Counter()
    n_eval_batches = 0

    eval_task = _make_layer_task(
        task_shape=task_shape,
        split_role=str(role),
        seed=new_seed(),
        drop_remainder=False,
        shuffle=True,
        min_n_demos=int(eval_max_n_demos),
        max_n_demos=int(eval_max_n_demos),
        fixed_length_mode=EVAL_FIXED_LENGTH_MODE,
        fixed_length_n_seq=int(n_seq_ar),
        demo_distribution=str(demo_distribution),
        demo_distribution_alpha=float(eval_alpha),
        demo_ranked=bool(eval_demo_ranked),
        batch_size=batch_size,
    )
    try:
        all_batch_metrics = []
        for _ in range(int(n_iters)):
            batch = next(eval_task)
            xs = np.asarray(batch[0])
            if xs.ndim == 2:
                seq_len_counts[int(xs.shape[1])] += 1
            n_eval_batches += 1
            all_batch_metrics.append(metrics_fn(optimizer, batch))
        agg = _mean_metrics(all_batch_metrics)
    finally:
        close = getattr(eval_task, "close", None)
        if callable(close):
            close()

    # --- Per-example rollouts with fresh temp banks ---
    rollout_rng = np.random.default_rng(
        int(RUN_ID)
        + 1_000 * int(eval_max_n_demos)
        + int(round(eval_alpha * 100))
        + (17 if role == "train" else 31)
    )

    n_rollout_success = 0
    n_rollout_decode_error = 0
    n_rollout_unknown_rule_error = 0
    n_rollout_wrong_rule_error = 0
    n_rollout_inapplicable_rule_error = 0
    n_rollout_goal_not_reached = 0
    rollout_steps: list[int] = []

    step0_reachable_count = 0
    step1_reachable_count = 0

    rollout_demo_adapter = None
    for _ in range(int(ROLLOUT_EXAMPLES_PER_ROLE)):
        temp_bank = eval_task.build_fresh_temp_bank(rollout_rng)

        if rollout_demo_adapter is None:
            rollout_demo_adapter = eval_task.make_demo_adapter(
                shared_adapter,
                temp_bank,
                min_n_demos=int(eval_max_n_demos),
                max_n_demos=int(eval_max_n_demos),
                demo_distribution_alpha=float(eval_alpha),
                demo_ranked=bool(eval_demo_ranked),
            )
        else:
            rollout_demo_adapter.rule_bank = temp_bank

        example = eval_task.sample_rollout_example(rollout_rng, rule_bank=temp_bank)

        result = run_layer_rollout_fol(
            rule_bank=temp_bank,
            example=example,
            model=model_fn,
            adapter=rollout_demo_adapter,
            tokenizer=tokenizer,
            temperature=0.0,
            rng=rollout_rng,
        )

        rollout_steps.append(int(result.n_steps))
        if result.success:
            n_rollout_success += 1
        elif result.failure_reason == "decode_error":
            n_rollout_decode_error += 1
        elif result.failure_reason == "unknown_rule_error":
            n_rollout_unknown_rule_error += 1
        elif result.failure_reason == "wrong_rule_error":
            n_rollout_wrong_rule_error += 1
        elif result.failure_reason == "inapplicable_rule_error":
            n_rollout_inapplicable_rule_error += 1
        elif result.failure_reason == "goal_not_reached":
            n_rollout_goal_not_reached += 1

        # Per-step reachability tracking
        goal_atom = parse_atom_text(example.goal_atom)
        goal_layer = int(example.start_layer) + int(example.distance)
        for step in result.steps:
            if step.matched_rule_statement is None or step.inapplicable_rule_error:
                continue
            try:
                lhs, rhs = parse_clause_text(str(step.matched_rule_statement))
                matched_rule = FOLLayerRule(
                    src_layer=int(step.src_layer),
                    dst_layer=int(step.src_layer) + 1,
                    lhs=lhs,
                    rhs=rhs,
                )
                reachable = predicted_rule_reaches_goal(
                    rule_bank=temp_bank,
                    matched_rule=matched_rule,
                    goal=goal_atom,
                    goal_layer=goal_layer,
                    max_unify_solutions=int(MAX_UNIFY_SOLUTIONS),
                )
            except (ValueError, RuntimeError):
                reachable = False

            if int(step.step_idx) == 0:
                if reachable:
                    step0_reachable_count += 1
            elif int(step.step_idx) == 1:
                if reachable:
                    step1_reachable_count += 1

    n_rollout_total = int(ROLLOUT_EXAMPLES_PER_ROLE)

    def _rollout_rate(n: int) -> float:
        return float(n) / float(n_rollout_total) if n_rollout_total > 0 else 0.0

    agg.update(
        {
            "eval_role": str(role),
            "eval_max_n_demos": int(eval_max_n_demos),
            "eval_alpha": float(eval_alpha),
            "eval_demo_ranked": bool(eval_demo_ranked),
            "eval_n_batches": int(n_eval_batches),
            "eval_n_unique_seq_lens": int(len(seq_len_counts)),
            "eval_top_seq_lens": [
                {"seq_len": int(seq_len), "count": int(count)}
                for seq_len, count in seq_len_counts.most_common(10)
            ],
            "rollout_n_examples": n_rollout_total,
            "rollout_success_rate": _rollout_rate(n_rollout_success),
            "rollout_decode_error_rate": _rollout_rate(n_rollout_decode_error),
            "rollout_unknown_rule_error_rate": _rollout_rate(n_rollout_unknown_rule_error),
            "rollout_wrong_rule_error_rate": _rollout_rate(n_rollout_wrong_rule_error),
            "rollout_inapplicable_rule_error_rate": _rollout_rate(
                n_rollout_inapplicable_rule_error
            ),
            "rollout_goal_not_reached_rate": _rollout_rate(n_rollout_goal_not_reached),
            "rollout_avg_steps": float(np.mean(rollout_steps)) if rollout_steps else 0.0,
            "rollout_step0_reachable_rate": _rollout_rate(step0_reachable_count),
            "rollout_step1_reachable_rate": _rollout_rate(step1_reachable_count),
        }
    )

    return agg


# <codecell>

save_dir = Path("set")
save_dir.mkdir(parents=True, exist_ok=True)

bundle = _make_task_shape_bundle(task_shape=TASK_SHAPE)
task_shape = bundle["task_shape"]
train_n_seq = int(bundle["train_n_seq"])
eval_n_seq = int(bundle["eval_n_seq"])
n_vocab = int(bundle["n_vocab"])

all_cases = []

ar_light_metrics_fn = make_ar_light_metrics_fn()

# --- Transformer configs ---
for (n_layers, n_hidden, n_heads), train_iters, lr in itertools.product(
    TRANSFORMER_CONFIGS,
    TRAIN_ITERS_SWEEP,
    LR_SWEEP,
):
    test_every = _compute_test_every(train_iters)

    train_batch_size = _estimate_batch_size(
        n_seq=train_n_seq,
        base_n_seq=train_n_seq,
        base_batch_size=BATCH_SIZE,
    )
    train_grad_accum_steps = max(1, EFFECTIVE_BATCH_SIZE // train_batch_size)

    config = TransformerConfig(
        n_vocab=n_vocab,
        n_seq=int(bundle["model_n_seq"]),
        n_layers=n_layers,
        n_hidden=n_hidden,
        n_heads=n_heads,
        n_out=n_vocab,
        n_pred_tokens=1,
        pos_encoding="rope",
        layer_norm=True,
        use_swiglu=True,
        use_bias=True,
        dropout_rate=0.0,
        output_mode="full_sequence",
        pad_token_id=0,
    )

    train_task = _make_layer_task(
        task_shape=task_shape,
        split_role="train",
        seed=new_seed(),
        drop_remainder=True,
        shuffle=True,
        min_n_demos=int(TRAIN_MIN_N_DEMOS),
        max_n_demos=int(TRAIN_MAX_N_DEMOS),
        fixed_length_mode=TRAIN_FIXED_LENGTH_MODE,
        fixed_length_n_seq=train_n_seq,
        include_oracle=bool(TRAIN_INCLUDE_ORACLE),
        demo_distribution=DEMO_DISTRIBUTION,
        demo_distribution_alpha=float(TRAIN_ALPHA),
        demo_ranked=bool(TRAIN_DEMO_RANKED),
        batch_size=train_batch_size,
    )
    test_task = _make_layer_task(
        task_shape=task_shape,
        split_role="eval",
        seed=new_seed(),
        drop_remainder=False,
        shuffle=True,
        min_n_demos=int(TRAIN_MIN_N_DEMOS),
        max_n_demos=int(TRAIN_MAX_N_DEMOS),
        fixed_length_mode=EVAL_FIXED_LENGTH_MODE,
        fixed_length_n_seq=int(bundle["model_n_seq"]),
        demo_distribution=DEMO_DISTRIBUTION,
        demo_distribution_alpha=float(TRAIN_ALPHA),
        demo_ranked=bool(TRAIN_DEMO_RANKED),
        batch_size=train_batch_size,
    )

    train_args = {
        "loss": "ce_mask",
        "eval_fns": [ar_light_metrics_fn],
        "print_fn": make_print_fn("final_token_acc"),
        "train_iters": int(train_iters),
        "test_iters": TEST_ITERS,
        "test_every": test_every,
        "grad_accum_steps": int(train_grad_accum_steps),
        "lr": warmup_cosine_schedule(lr, int(train_iters)),
    }

    info = {
        "model_family": "transformer",
        "target_format": "next_token_full_sequence",
        "task_split": "depth3_fresh_icl",
        "eval_roles": list(EVAL_ROLES),
        "distance_range": [2],
        "mid_pred": int(bundle["mid_pred"]),
        "predicates_per_layer": task_shape["predicates_per_layer"],
        "rules_per_transition": task_shape["rules_per_transition"],
        "train_max_n_demos": int(TRAIN_MAX_N_DEMOS),
        "eval_max_n_demos": int(EVAL_MAX_N_DEMOS),
        "predicate_name_len": int(PREDICATE_NAME_LEN),
        "train_include_oracle": bool(TRAIN_INCLUDE_ORACLE),
        "train_alpha": float(TRAIN_ALPHA),
        "eval_alpha": float(EVAL_ALPHA),
        "train_demo_ranked": bool(TRAIN_DEMO_RANKED),
        "demo_distribution": DEMO_DISTRIBUTION,
        "n_layers": n_layers,
        "n_hidden": n_hidden,
        "n_heads": n_heads,
        "pos_encoding": "rope",
        "use_swiglu": True,
        "lr": lr,
        "n_vocab": n_vocab,
        "n_seq": int(bundle["model_n_seq"]),
        "train_fixed_length_mode": TRAIN_FIXED_LENGTH_MODE,
        "train_fixed_length_n_seq": train_n_seq,
        "eval_fixed_length_mode": EVAL_FIXED_LENGTH_MODE,
        "eval_fixed_length_n_seq": eval_n_seq,
        "train_eval_profile": "light",
        "train_iters": int(train_iters),
        "grad_accum_steps": int(train_grad_accum_steps),
        "microbatch_size": int(train_batch_size),
        "effective_batch_size": int(train_batch_size) * int(train_grad_accum_steps),
    }

    case_name = (
        f"14_co_transformer_"
        f"l{int(n_layers)}_h{int(n_hidden)}_heads{int(n_heads)}_"
        f"lr{_lr_tag(lr)}_ti{int(train_iters)}"
    )
    train_args["wandb_cfg"] = _make_case_wandb_cfg(
        case_name=case_name,
        model_config=config,
        train_args=train_args,
        info=info,
    )

    case = Case(
        case_name,
        config,
        train_task=train_task,
        test_task=test_task,
        train_args=train_args,
        info=info,
    )
    all_cases.append(case)


# --- Mamba2 Bonsai configs ---
for (n_layers, n_hidden, n_heads), train_iters, lr in itertools.product(
    MAMBA2_BONSAI_CONFIGS,
    TRAIN_ITERS_SWEEP,
    LR_SWEEP,
):
    test_every = _compute_test_every(train_iters)

    train_batch_size = _estimate_batch_size(
        n_seq=train_n_seq,
        base_n_seq=train_n_seq,
        base_batch_size=BATCH_SIZE,
    )
    train_grad_accum_steps = max(1, EFFECTIVE_BATCH_SIZE // train_batch_size)

    config = Mamba2BonsaiConfig(
        n_vocab=n_vocab,
        n_seq=int(bundle["model_n_seq"]),
        n_layers=n_layers,
        n_hidden=n_hidden,
        n_heads=n_heads,
        n_out=n_vocab,
        n_pred_tokens=1,
        output_mode="full_sequence",
        pad_token_id=0,
        layer_norm=True,
        use_bias=True,
        dropout_rate=0.0,
        d_state=MAMBA2_D_STATE,
        d_conv=MAMBA2_D_CONV,
        expand=2,
        scan_chunk_len=MAMBA2_SCAN_CHUNK_LEN,
    )

    train_task = _make_layer_task(
        task_shape=task_shape,
        split_role="train",
        seed=new_seed(),
        drop_remainder=True,
        shuffle=True,
        min_n_demos=int(TRAIN_MIN_N_DEMOS),
        max_n_demos=int(TRAIN_MAX_N_DEMOS),
        fixed_length_mode=TRAIN_FIXED_LENGTH_MODE,
        fixed_length_n_seq=train_n_seq,
        include_oracle=bool(TRAIN_INCLUDE_ORACLE),
        demo_distribution=DEMO_DISTRIBUTION,
        demo_distribution_alpha=float(TRAIN_ALPHA),
        demo_ranked=bool(TRAIN_DEMO_RANKED),
        batch_size=train_batch_size,
    )
    test_task = _make_layer_task(
        task_shape=task_shape,
        split_role="eval",
        seed=new_seed(),
        drop_remainder=False,
        shuffle=True,
        min_n_demos=int(TRAIN_MIN_N_DEMOS),
        max_n_demos=int(TRAIN_MAX_N_DEMOS),
        fixed_length_mode=EVAL_FIXED_LENGTH_MODE,
        fixed_length_n_seq=int(bundle["model_n_seq"]),
        demo_distribution=DEMO_DISTRIBUTION,
        demo_distribution_alpha=float(TRAIN_ALPHA),
        demo_ranked=bool(TRAIN_DEMO_RANKED),
        batch_size=train_batch_size,
    )

    train_args = {
        "loss": "ce_mask",
        "eval_fns": [ar_light_metrics_fn],
        "print_fn": make_print_fn("final_token_acc"),
        "train_iters": int(train_iters),
        "test_iters": TEST_ITERS,
        "test_every": test_every,
        "grad_accum_steps": int(train_grad_accum_steps),
        "lr": warmup_cosine_schedule(lr, int(train_iters)),
    }

    info = {
        "model_family": "mamba2_bonsai",
        "target_format": "next_token_full_sequence",
        "task_split": "depth3_fresh_icl",
        "eval_roles": list(EVAL_ROLES),
        "distance_range": [2],
        "mid_pred": int(bundle["mid_pred"]),
        "predicates_per_layer": task_shape["predicates_per_layer"],
        "rules_per_transition": task_shape["rules_per_transition"],
        "train_max_n_demos": int(TRAIN_MAX_N_DEMOS),
        "eval_max_n_demos": int(EVAL_MAX_N_DEMOS),
        "predicate_name_len": int(PREDICATE_NAME_LEN),
        "train_include_oracle": bool(TRAIN_INCLUDE_ORACLE),
        "train_alpha": float(TRAIN_ALPHA),
        "eval_alpha": float(EVAL_ALPHA),
        "train_demo_ranked": bool(TRAIN_DEMO_RANKED),
        "demo_distribution": DEMO_DISTRIBUTION,
        "n_layers": n_layers,
        "n_hidden": n_hidden,
        "n_heads": n_heads,
        "d_state": MAMBA2_D_STATE,
        "d_conv": MAMBA2_D_CONV,
        "scan_chunk_len": MAMBA2_SCAN_CHUNK_LEN,
        "lr": lr,
        "n_vocab": n_vocab,
        "n_seq": int(bundle["model_n_seq"]),
        "train_fixed_length_mode": TRAIN_FIXED_LENGTH_MODE,
        "train_fixed_length_n_seq": train_n_seq,
        "eval_fixed_length_mode": EVAL_FIXED_LENGTH_MODE,
        "eval_fixed_length_n_seq": eval_n_seq,
        "train_eval_profile": "light",
        "train_iters": int(train_iters),
        "grad_accum_steps": int(train_grad_accum_steps),
        "microbatch_size": int(train_batch_size),
        "effective_batch_size": int(train_batch_size) * int(train_grad_accum_steps),
    }

    case_name = (
        f"14_co_mamba2_bonsai_"
        f"l{int(n_layers)}_h{int(n_hidden)}_heads{int(n_heads)}_"
        f"lr{_lr_tag(lr)}_ti{int(train_iters)}"
    )
    train_args["wandb_cfg"] = _make_case_wandb_cfg(
        case_name=case_name,
        model_config=config,
        train_args=train_args,
        info=info,
    )

    case = Case(
        case_name,
        config,
        train_task=train_task,
        test_task=test_task,
        train_args=train_args,
        info=info,
    )
    all_cases.append(case)


print("TOTAL CASES:", len(all_cases))
if len(all_cases) != int(RUN_SPLIT):
    raise ValueError(f"RUN_SPLIT={RUN_SPLIT} does not match total cases={len(all_cases)}")
all_cases = split_cases(all_cases, RUN_SPLIT, shuffle_seed=200)
print("CASES IN THIS RUN:", len(all_cases))
print("CASE NAMES", [case.name for case in all_cases])


# <codecell>

dims = bundle["dims"]
print("TRAIN DIMS", dims["train"])
print("EVAL DIMS", dims["eval"])
print(
    "SEQUENCE SHAPE POLICY",
    {
        "train_fixed_length_mode": TRAIN_FIXED_LENGTH_MODE,
        "train_fixed_length_n_seq": train_n_seq,
        "eval_fixed_length_mode": EVAL_FIXED_LENGTH_MODE,
        "eval_fixed_length_n_seq": eval_n_seq,
        "train_raw_n_seq_ar": int(bundle["train_n_seq_raw"]),
        "eval_raw_n_seq_ar": int(bundle["eval_n_seq_raw"]),
        "model_n_seq": int(bundle["model_n_seq"]),
    },
)

preview_train_task = _make_layer_task(
    task_shape=task_shape,
    split_role="train",
    seed=101,
    drop_remainder=True,
    shuffle=True,
    min_n_demos=int(TRAIN_MIN_N_DEMOS),
    max_n_demos=int(TRAIN_MAX_N_DEMOS),
    fixed_length_mode=TRAIN_FIXED_LENGTH_MODE,
    fixed_length_n_seq=train_n_seq,
    include_oracle=bool(TRAIN_INCLUDE_ORACLE),
    demo_distribution=DEMO_DISTRIBUTION,
    demo_distribution_alpha=float(TRAIN_ALPHA),
)
preview_eval_task = _make_layer_task(
    task_shape=task_shape,
    split_role="eval",
    seed=202,
    drop_remainder=False,
    shuffle=True,
    min_n_demos=int(EVAL_MAX_N_DEMOS),
    max_n_demos=int(EVAL_MAX_N_DEMOS),
    fixed_length_mode=EVAL_FIXED_LENGTH_MODE,
    fixed_length_n_seq=eval_n_seq,
    demo_distribution=DEMO_DISTRIBUTION,
    demo_distribution_alpha=float(EVAL_ALPHA),
)
try:
    print_task_preview(
        preview_train_task,
        role="train",
        n_examples=PREVIEW_EXAMPLES_PER_SPLIT,
    )
    print_task_preview(
        preview_eval_task,
        role="eval",
        n_examples=PREVIEW_EXAMPLES_PER_SPLIT,
    )
finally:
    for task in (preview_train_task, preview_eval_task):
        close = getattr(task, "close", None)
        if callable(close):
            close()


# <codecell>

rows = []
for case in tqdm(all_cases, desc="cases", leave=True):
    print("RUNNING", case.name, case.info)
    train_start = time.perf_counter()
    case.run()
    train_wall_s = time.perf_counter() - train_start

    post_eval_start = time.perf_counter()

    model_fn = make_model_callable(case.optimizer, to_numpy=False)

    # --- Single eval pass ---
    eval_n_seq_for_eval = eval_n_seq
    eval_batch_size, eval_n_iters = _estimate_batch_and_iters(
        n_seq=eval_n_seq_for_eval,
        base_n_seq=train_n_seq,
        base_batch_size=BATCH_SIZE,
        base_n_iters=EVAL_ITERS_PER_ROLE,
    )
    eval_adapter = AutoregressiveLogitsAdapter(
        n_seq=eval_n_seq_for_eval,
        max_completion_len=int(bundle["max_completion_len"]),
        pad_token_id=0,
        jit_step=True,
    )

    eval_metrics = _evaluate_role_for_demo(
        case.optimizer,
        task_shape=task_shape,
        role="eval",
        tokenizer=bundle["tokenizer"],
        rule_bank=bundle["base_bank"],
        n_seq_ar=eval_n_seq_for_eval,
        max_completion_len=int(bundle["max_completion_len"]),
        n_iters=eval_n_iters,
        eval_max_n_demos=int(EVAL_MAX_N_DEMOS),
        eval_alpha=float(EVAL_ALPHA),
        eval_demo_ranked=bool(EVAL_DEMO_RANKED),
        demo_distribution=DEMO_DISTRIBUTION,
        batch_size=eval_batch_size,
        model_fn=model_fn,
        shared_adapter=eval_adapter,
    )

    post_eval_wall_s = time.perf_counter() - post_eval_start

    # --- Compute tracking ---
    compute_info = compute_metrics_from_info(case.info)
    n_params = compute_info["n_params"]
    forward_flops = compute_info["forward_flops"]

    train_n_seq_for_flops = int(case.info.get("train_fixed_length_n_seq", train_n_seq))
    train_compute_info = compute_metrics_from_info(
        case.info, n_seq_override=train_n_seq_for_flops
    )
    total_training_flops = training_flops_total(
        train_compute_info["forward_flops"],
        train_iters=int(case.info["train_iters"]),
        batch_size=int(case.info["microbatch_size"]),
        grad_accum_steps=int(case.info["grad_accum_steps"]),
    )

    rollout_success_rate = float(eval_metrics.get("rollout_success_rate", float("nan")))

    row = {
        "run_id": RUN_ID,
        "name": case.name,
        "model_family": case.info["model_family"],
        "n_layers": int(case.info["n_layers"]),
        "n_hidden": int(case.info["n_hidden"]),
        "n_heads": int(case.info["n_heads"]),
        "lr": float(case.info["lr"]),
        "train_iters": int(case.info["train_iters"]),
        "n_params": int(n_params),
        "forward_flops": int(forward_flops),
        "total_training_flops": int(total_training_flops),
        "info": case.info,
        "train_args": {
            "loss": case.train_args["loss"],
            "train_iters": case.train_args["train_iters"],
            "test_iters": case.train_args["test_iters"],
            "test_every": case.train_args["test_every"],
            "grad_accum_steps": case.train_args["grad_accum_steps"],
            "lr": case.info["lr"],
            "eval_profile": case.info.get("train_eval_profile", "light"),
        },
        "metrics_final": case.hist["test"][-1] if case.hist and case.hist.get("test") else {},
        "eval_metrics": eval_metrics,
        "rollout_success_rate": rollout_success_rate,
        "train_wall_s": float(train_wall_s),
        "hist": case.hist,
    }
    rows.append(row)

    for task in (case.train_task, case.test_task):
        close = getattr(task, "close", None)
        if callable(close):
            close()
    case.optimizer = None
    case.hist = None
    case.train_task = None
    case.test_task = None
    case.train_args["eval_fns"] = None

pd.DataFrame(rows).to_pickle(save_dir / f"res.{RUN_ID}.pkl")
print("done!")
