"""Architecture sweep for the depth3_fresh_icl fresh-rule split."""

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
from model.eval_adapters import make_model_callable
from model.ssm_bonsai import Mamba2BonsaiConfig
from model.transformer import TransformerConfig
from task.layer_fol import (
    AutoregressiveLogitsAdapter,
    FOLDemoAugmentedAdapter,
    FOLLayerTask,
    _build_tokenizer_for_fresh_icl,
    _fresh_predicate_sentinels,
    compute_fol_dims,
    print_task_preview,
    run_layer_rollout_fol,
    sample_rollout_examples,
)
from task.layer_gen.util.fol_rule_bank import (
    build_fresh_layer0_bank,
    build_random_fol_rule_bank,
    generate_fresh_predicate_names,
)
from train import Case, ce_mask, warmup_cosine_schedule

from experiment.utils.metrics_utils import final_token_accuracy
from wandb_utils import make_experiment_wandb_config


RUN_ID = new_seed()
print("RUN ID", RUN_ID)

WANDB_PROJECT = Path(__file__).resolve().parent.name
WANDB_API_KEY_PATH = ROOT / "key" / "wandb.txt"
USE_WANDB = True

EVAL_ROLES = ["train", "eval"]

TRAIN_MIN_N_DEMOS = 4
TRAIN_MAX_N_DEMOS = 8
EVAL_MAX_N_DEMOS_SWEEP = [1, 2, 4, 8, 12, 16, 24, 32]
SELECTION_EVAL_MAX_N_DEMOS = 8

BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 8
EFFECTIVE_BATCH_SIZE = int(BATCH_SIZE) * int(GRAD_ACCUM_STEPS)
TRAIN_ITERS_SWEEP = [400, 6400, 25600]
# TRAIN_ITERS_SWEEP = [1600, 6400, 25600, 102400]
TEST_EVERY = 100
TEST_ITERS = 2
EVAL_ITERS_PER_ROLE = 2
ROLLOUT_EXAMPLES_PER_ROLE = 64
PREVIEW_EXAMPLES_PER_SPLIT = 3

# RUN_SPLIT = 8
RUN_SPLIT = 6

PREDICATES_PER_LAYER = 8
RULES_PER_TRANSITION = 16
FRESH_ICL_N_PREDICATES = 8
N_LAYERS = 3
# ARITY_MAX = 3
ARITY_MAX = 1
VARS_PER_RULE_MAX = 6
K_IN_MAX = 1
K_OUT_MAX = 1
INITIAL_ANT_MAX = 1
# CONSTANTS = [f"p{i}" for i in range(16)]
CONSTANTS = [f"p{i}" for i in range(1)]
SAMPLE_MAX_ATTEMPTS = 4096
MAX_UNIFY_SOLUTIONS = 128
BASE_BANK_SEED = 2042

TRAIN_FIXED_LENGTH_MODE = "next_pow2"
EVAL_FIXED_LENGTH_MODE = "next_pow2"

TRANSFORMER_LAYERS = [48]
TRANSFORMER_WIDTH_HEADS = [(1600, 25)]
TRANSFORMER_LRS = [7e-5]
TRANSFORMER_POS = ["rope"]
TRANSFORMER_SWIGLU = [True]

MAMBA2_BONSAI_LAYERS = [48]
MAMBA2_BONSAI_WIDTH_HEADS = [(2304, 18)]
MAMBA2_BONSAI_D_STATE = [64]
MAMBA2_BONSAI_D_CONV = [4]
MAMBA2_BONSAI_SCAN_CHUNK_LEN = [64]
MAMBA2_BONSAI_LRS = [7e-5]

### START TEST CONFIGS
# BATCH_SIZE = 2
# GRAD_ACCUM_STEPS = 2
# TRAIN_ITERS_SWEEP = [20]
# TEST_EVERY = 10
# TEST_ITERS = 1
# EVAL_ITERS_PER_ROLE = 1
# ROLLOUT_EXAMPLES_PER_ROLE = 4
# RUN_SPLIT = 1
# TRAIN_MAX_N_DEMOS = 4
# EVAL_MAX_N_DEMOS_SWEEP = [4]
# SELECTION_EVAL_MAX_N_DEMOS = 4
# PREDICATES_PER_LAYER = 10
# RULES_PER_TRANSITION = 18
# FRESH_ICL_N_PREDICATES = 10
# TRANSFORMER_LAYERS = [2]
# TRANSFORMER_WIDTH_HEADS = [(128, 4)]
# TRANSFORMER_LRS = [3e-4]
# MAMBA2_BONSAI_LAYERS = [2]
# MAMBA2_BONSAI_WIDTH_HEADS = [(128, 4)]
# MAMBA2_BONSAI_D_STATE = [8]
# MAMBA2_BONSAI_D_CONV = [4]
# MAMBA2_BONSAI_SCAN_CHUNK_LEN = [16]
# MAMBA2_BONSAI_LRS = [3e-4]
# USE_WANDB = False
### END TEST CONFIGS


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


def _build_base_bank_and_tokenizer():
    """Build a 3-layer base bank and the fresh-ICL tokenizer."""
    base_bank = build_random_fol_rule_bank(
        n_layers=int(N_LAYERS),
        predicates_per_layer=int(PREDICATES_PER_LAYER),
        rules_per_transition=int(RULES_PER_TRANSITION),
        arity_max=int(ARITY_MAX),
        vars_per_rule_max=int(VARS_PER_RULE_MAX),
        k_in_max=int(K_IN_MAX),
        k_out_max=int(K_OUT_MAX),
        constants=tuple(str(c) for c in CONSTANTS),
        rng=np.random.default_rng(int(BASE_BANK_SEED)),
    )
    tokenizer = _build_tokenizer_for_fresh_icl(base_bank=base_bank)
    return base_bank, tokenizer


def _compute_dims(
    base_bank,
    tokenizer,
    *,
    max_n_demos_for_shapes: int,
):
    """Compute tensor dims from the base bank + fresh predicate estimates."""
    sentinels = _fresh_predicate_sentinels()
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
    split_role: str,
    seed: int,
    drop_remainder: bool,
    shuffle: bool,
    min_n_demos: int,
    max_n_demos: int,
    fixed_length_mode: str,
    fixed_length_n_seq: int,
):
    return FOLLayerTask(
        distance_range=(2, 2),
        batch_size=int(BATCH_SIZE),
        mode="online",
        task_split="depth3_fresh_icl",
        split_role=str(split_role),
        shuffle=shuffle,
        seed=seed,
        worker_count=0,
        drop_remainder=drop_remainder,
        prediction_objective="autoregressive",
        predicates_per_layer=int(PREDICATES_PER_LAYER),
        rules_per_transition=int(RULES_PER_TRANSITION),
        fresh_icl_n_predicates=int(FRESH_ICL_N_PREDICATES),
        fresh_icl_base_bank_seed=int(BASE_BANK_SEED),
        arity_max=int(ARITY_MAX),
        vars_per_rule_max=int(VARS_PER_RULE_MAX),
        constants=tuple(str(tok) for tok in CONSTANTS),
        k_in_max=int(K_IN_MAX),
        k_out_max=int(K_OUT_MAX),
        initial_ant_max=int(INITIAL_ANT_MAX),
        min_n_demos=int(min_n_demos),
        max_n_demos=int(max_n_demos),
        sample_max_attempts=int(SAMPLE_MAX_ATTEMPTS),
        max_unify_solutions=int(MAX_UNIFY_SOLUTIONS),
        fixed_length_mode=str(fixed_length_mode),
        fixed_length_n_seq=int(fixed_length_n_seq),
    )


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
    role: str,
    tokenizer,
    rule_bank,
    n_seq_ar: int,
    max_completion_len: int,
    n_iters: int,
    eval_max_n_demos: int,
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
        split_role=str(role),
        seed=new_seed(),
        drop_remainder=False,
        shuffle=True,
        min_n_demos=int(eval_max_n_demos),
        max_n_demos=int(eval_max_n_demos),
        fixed_length_mode=EVAL_FIXED_LENGTH_MODE,
        fixed_length_n_seq=int(n_seq_ar),
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
        + (17 if role == "train" else 31)
    )

    n_rollout_success = 0
    n_rollout_decode_error = 0
    n_rollout_unknown_rule_error = 0
    n_rollout_wrong_rule_error = 0
    n_rollout_inapplicable_rule_error = 0
    n_rollout_goal_not_reached = 0
    rollout_steps: list[int] = []

    rollout_demo_adapter = None
    for _ in range(int(ROLLOUT_EXAMPLES_PER_ROLE)):
        # Build a per-example fresh bank
        fresh_preds = generate_fresh_predicate_names(int(FRESH_ICL_N_PREDICATES), rollout_rng)
        temp_bank = build_fresh_layer0_bank(
            base_bank=rule_bank,
            fresh_predicates=fresh_preds,
            rules_per_transition=int(RULES_PER_TRANSITION),
            k_in_min=1,
            k_in_max=int(K_IN_MAX),
            k_out_min=1,
            k_out_max=int(K_OUT_MAX),
            rng=rollout_rng,
        )

        # Build/reuse a rollout adapter that uses the temp bank for demos
        if rollout_demo_adapter is None:
            rollout_demo_adapter = FOLDemoAugmentedAdapter(
                base_adapter=shared_adapter,
                rule_bank=temp_bank,
                tokenizer=tokenizer,
                min_n_demos=int(eval_max_n_demos),
                max_n_demos=int(eval_max_n_demos),
                max_unify_solutions=int(MAX_UNIFY_SOLUTIONS),
            )
        else:
            rollout_demo_adapter.rule_bank = temp_bank

        # Sample one rollout example from this bank
        examples = sample_rollout_examples(
            rule_bank=temp_bank,
            distance=2,
            n_examples=1,
            initial_ant_max=int(INITIAL_ANT_MAX),
            max_steps=2,
            max_unify_solutions=int(MAX_UNIFY_SOLUTIONS),
            rng=rollout_rng,
        )

        # Run rollout with the temp bank
        result = run_layer_rollout_fol(
            rule_bank=temp_bank,
            example=examples[0],
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

    n_rollout_total = int(ROLLOUT_EXAMPLES_PER_ROLE)

    def _rollout_rate(n: int) -> float:
        return float(n) / float(n_rollout_total) if n_rollout_total > 0 else 0.0

    agg.update(
        {
            "eval_role": str(role),
            "eval_max_n_demos": int(eval_max_n_demos),
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
        }
    )

    return agg


def _metric_by_role_demo(
    metrics_by_role_eval_demo: dict,
    *,
    role: str,
    eval_max_n_demos: int,
    metric_name: str,
) -> float:
    role_metrics = (metrics_by_role_eval_demo or {}).get(str(role), {})
    demo_metrics = (role_metrics or {}).get(int(eval_max_n_demos), {})
    value = demo_metrics.get(metric_name)
    if value is None:
        return float("nan")
    return float(value)


save_dir = Path("set")
save_dir.mkdir(parents=True, exist_ok=True)

BASE_BANK, SHARED_TOKENIZER = _build_base_bank_and_tokenizer()

DIMS_TRAIN = _compute_dims(
    BASE_BANK,
    SHARED_TOKENIZER,
    max_n_demos_for_shapes=TRAIN_MAX_N_DEMOS,
)
DIMS_EVAL = _compute_dims(
    BASE_BANK,
    SHARED_TOKENIZER,
    max_n_demos_for_shapes=max(EVAL_MAX_N_DEMOS_SWEEP),
)
N_VOCAB = max(int(DIMS_TRAIN["n_vocab"]), int(DIMS_EVAL["n_vocab"]))
MAX_COMPLETION_LEN = max(
    int(DIMS_TRAIN["max_completion_len"]),
    int(DIMS_EVAL["max_completion_len"]),
)
TRAIN_N_SEQ_AR_RAW = int(DIMS_TRAIN["n_seq_ar"])
EVAL_N_SEQ_AR_RAW = int(DIMS_EVAL["n_seq_ar"])
MODEL_N_SEQ_AR = int(max(2, _ceil_pow2_int(EVAL_N_SEQ_AR_RAW)))
TRAIN_N_SEQ_AR = int(MODEL_N_SEQ_AR)
N_SEQ_AR = int(MODEL_N_SEQ_AR)
DIMS = {"train": DIMS_TRAIN, "eval": DIMS_EVAL}

print("TRAIN DIMS", DIMS["train"])
print("EVAL DIMS", DIMS["eval"])
print(
    "SEQUENCE SHAPE POLICY",
    {
        "train_fixed_length_mode": TRAIN_FIXED_LENGTH_MODE,
        "train_fixed_length_n_seq": TRAIN_N_SEQ_AR,
        "eval_fixed_length_mode": EVAL_FIXED_LENGTH_MODE,
        "eval_fixed_length_n_seq": N_SEQ_AR,
        "train_raw_n_seq_ar": TRAIN_N_SEQ_AR_RAW,
        "eval_raw_n_seq_ar": EVAL_N_SEQ_AR_RAW,
        "model_n_seq": N_SEQ_AR,
        "causal_mask_tokens": int(N_SEQ_AR) * int(N_SEQ_AR),
    },
)
print(
    "FRESH ICL CONFIG",
    {
        "base_bank_seed": BASE_BANK_SEED,
        "predicates_per_layer": PREDICATES_PER_LAYER,
        "rules_per_transition": RULES_PER_TRANSITION,
        "fresh_icl_n_predicates": FRESH_ICL_N_PREDICATES,
        "n_layers": N_LAYERS,
    },
)

preview_train_task = _make_layer_task(
    split_role="train",
    seed=101,
    drop_remainder=True,
    shuffle=True,
    min_n_demos=int(TRAIN_MIN_N_DEMOS),
    max_n_demos=int(TRAIN_MAX_N_DEMOS),
    fixed_length_mode=TRAIN_FIXED_LENGTH_MODE,
    fixed_length_n_seq=TRAIN_N_SEQ_AR,
)
preview_eval_task = _make_layer_task(
    split_role="eval",
    seed=202,
    drop_remainder=False,
    shuffle=True,
    min_n_demos=int(TRAIN_MIN_N_DEMOS),
    max_n_demos=int(TRAIN_MAX_N_DEMOS),
    fixed_length_mode=EVAL_FIXED_LENGTH_MODE,
    fixed_length_n_seq=N_SEQ_AR,
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
all_cases = []

ar_light_metrics_fn = make_ar_light_metrics_fn()

for n_layers, (n_hidden, n_heads), lr, pos_encoding, use_swiglu, train_iters in itertools.product(
    TRANSFORMER_LAYERS,
    TRANSFORMER_WIDTH_HEADS,
    TRANSFORMER_LRS,
    TRANSFORMER_POS,
    TRANSFORMER_SWIGLU,
    TRAIN_ITERS_SWEEP,
):
    config = TransformerConfig(
        n_vocab=N_VOCAB,
        n_seq=N_SEQ_AR,
        n_layers=n_layers,
        n_hidden=n_hidden,
        n_heads=n_heads,
        n_out=N_VOCAB,
        n_pred_tokens=1,
        pos_encoding=pos_encoding,
        layer_norm=True,
        use_swiglu=use_swiglu,
        use_bias=True,
        dropout_rate=0.0,
        output_mode="full_sequence",
        pad_token_id=0,
    )

    train_task = _make_layer_task(
        split_role="train",
        seed=new_seed(),
        drop_remainder=True,
        shuffle=True,
        min_n_demos=int(TRAIN_MIN_N_DEMOS),
        max_n_demos=int(TRAIN_MAX_N_DEMOS),
        fixed_length_mode=TRAIN_FIXED_LENGTH_MODE,
        fixed_length_n_seq=TRAIN_N_SEQ_AR,
    )
    test_task = _make_layer_task(
        split_role="eval",
        seed=new_seed(),
        drop_remainder=False,
        shuffle=True,
        min_n_demos=int(TRAIN_MIN_N_DEMOS),
        max_n_demos=int(TRAIN_MAX_N_DEMOS),
        fixed_length_mode=EVAL_FIXED_LENGTH_MODE,
        fixed_length_n_seq=N_SEQ_AR,
    )

    train_args = {
        "loss": "ce_mask",
        "eval_fns": [ar_light_metrics_fn],
        "print_fn": make_print_fn("final_token_acc"),
        "train_iters": int(train_iters),
        "test_iters": TEST_ITERS,
        "test_every": TEST_EVERY,
        "grad_accum_steps": int(GRAD_ACCUM_STEPS),
        "lr": warmup_cosine_schedule(lr, int(train_iters)),
    }

    info = {
        "model_family": "transformer",
        "target_format": "next_token_full_sequence",
        "task_split": "depth3_fresh_icl",
        "eval_roles": list(EVAL_ROLES),
        "distance_range": [2],
        "train_max_n_demos": int(TRAIN_MAX_N_DEMOS),
        "eval_max_n_demos_sweep": [int(v) for v in EVAL_MAX_N_DEMOS_SWEEP],
        "n_layers": n_layers,
        "n_hidden": n_hidden,
        "n_heads": n_heads,
        "pos_encoding": pos_encoding,
        "use_swiglu": use_swiglu,
        "lr": lr,
        "n_vocab": N_VOCAB,
        "n_seq": N_SEQ_AR,
        "train_fixed_length_mode": TRAIN_FIXED_LENGTH_MODE,
        "train_fixed_length_n_seq": TRAIN_N_SEQ_AR,
        "eval_fixed_length_mode": EVAL_FIXED_LENGTH_MODE,
        "eval_fixed_length_n_seq": N_SEQ_AR,
        "train_eval_profile": "light",
        "train_iters": int(train_iters),
        "grad_accum_steps": int(GRAD_ACCUM_STEPS),
        "microbatch_size": int(BATCH_SIZE),
        "effective_batch_size": int(EFFECTIVE_BATCH_SIZE),
    }
    case_name = (
        f"10_fresh_rule_split_transformer_"
        f"l{int(n_layers)}_h{int(n_hidden)}_heads{int(n_heads)}_"
        f"lr{_lr_tag(lr)}_ga{int(GRAD_ACCUM_STEPS)}_ti{int(train_iters)}"
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

for n_layers, (n_hidden, n_heads), d_state, d_conv, scan_chunk_len, lr, train_iters in itertools.product(
    MAMBA2_BONSAI_LAYERS,
    MAMBA2_BONSAI_WIDTH_HEADS,
    MAMBA2_BONSAI_D_STATE,
    MAMBA2_BONSAI_D_CONV,
    MAMBA2_BONSAI_SCAN_CHUNK_LEN,
    MAMBA2_BONSAI_LRS,
    TRAIN_ITERS_SWEEP,
):
    config = Mamba2BonsaiConfig(
        n_vocab=N_VOCAB,
        n_seq=N_SEQ_AR,
        n_layers=n_layers,
        n_hidden=n_hidden,
        n_heads=n_heads,
        n_out=N_VOCAB,
        n_pred_tokens=1,
        output_mode="full_sequence",
        pad_token_id=0,
        layer_norm=True,
        use_bias=True,
        dropout_rate=0.0,
        d_state=d_state,
        d_conv=d_conv,
        expand=2,
        scan_chunk_len=scan_chunk_len,
    )

    train_task = _make_layer_task(
        split_role="train",
        seed=new_seed(),
        drop_remainder=True,
        shuffle=True,
        min_n_demos=int(TRAIN_MIN_N_DEMOS),
        max_n_demos=int(TRAIN_MAX_N_DEMOS),
        fixed_length_mode=TRAIN_FIXED_LENGTH_MODE,
        fixed_length_n_seq=TRAIN_N_SEQ_AR,
    )
    test_task = _make_layer_task(
        split_role="eval",
        seed=new_seed(),
        drop_remainder=False,
        shuffle=True,
        min_n_demos=int(TRAIN_MIN_N_DEMOS),
        max_n_demos=int(TRAIN_MAX_N_DEMOS),
        fixed_length_mode=EVAL_FIXED_LENGTH_MODE,
        fixed_length_n_seq=N_SEQ_AR,
    )

    train_args = {
        "loss": "ce_mask",
        "eval_fns": [ar_light_metrics_fn],
        "print_fn": make_print_fn("final_token_acc"),
        "train_iters": int(train_iters),
        "test_iters": TEST_ITERS,
        "test_every": TEST_EVERY,
        "grad_accum_steps": int(GRAD_ACCUM_STEPS),
        "lr": warmup_cosine_schedule(lr, int(train_iters)),
    }

    info = {
        "model_family": "mamba2_bonsai",
        "target_format": "next_token_full_sequence",
        "task_split": "depth3_fresh_icl",
        "eval_roles": list(EVAL_ROLES),
        "distance_range": [2],
        "train_max_n_demos": int(TRAIN_MAX_N_DEMOS),
        "eval_max_n_demos_sweep": [int(v) for v in EVAL_MAX_N_DEMOS_SWEEP],
        "n_layers": n_layers,
        "n_hidden": n_hidden,
        "n_heads": n_heads,
        "d_state": d_state,
        "d_conv": d_conv,
        "scan_chunk_len": scan_chunk_len,
        "lr": lr,
        "n_vocab": N_VOCAB,
        "n_seq": N_SEQ_AR,
        "train_fixed_length_mode": TRAIN_FIXED_LENGTH_MODE,
        "train_fixed_length_n_seq": TRAIN_N_SEQ_AR,
        "eval_fixed_length_mode": EVAL_FIXED_LENGTH_MODE,
        "eval_fixed_length_n_seq": N_SEQ_AR,
        "train_eval_profile": "light",
        "train_iters": int(train_iters),
        "grad_accum_steps": int(GRAD_ACCUM_STEPS),
        "microbatch_size": int(BATCH_SIZE),
        "effective_batch_size": int(EFFECTIVE_BATCH_SIZE),
    }
    case_name = (
        "10_fresh_rule_split_mamba2_bonsai_"
        f"l{int(n_layers)}_h{int(n_hidden)}_heads{int(n_heads)}_"
        f"ds{int(d_state)}_lr{_lr_tag(lr)}_ga{int(GRAD_ACCUM_STEPS)}_ti{int(train_iters)}"
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
all_cases = split_cases(all_cases, RUN_SPLIT, shuffle_seed=200)
print("CASES IN THIS RUN:", len(all_cases))
print("CASE NAMES", [case.name for case in all_cases])


# <codecell>
rows = []
for case in tqdm(all_cases, desc="cases", leave=True):
    print("RUNNING", case.name, case.info)
    train_start = time.perf_counter()
    case.run()
    train_wall_s = time.perf_counter() - train_start

    metrics_by_role_eval_demo = {}
    post_eval_start = time.perf_counter()

    model_fn = make_model_callable(case.optimizer, to_numpy=False)
    shared_adapter = AutoregressiveLogitsAdapter(
        n_seq=int(N_SEQ_AR),
        max_completion_len=int(MAX_COMPLETION_LEN),
        pad_token_id=0,
        jit_step=True,
    )

    eval_job_bar = tqdm(
        total=int(len(EVAL_ROLES) * len(EVAL_MAX_N_DEMOS_SWEEP)),
        desc=f"{case.name} eval role/demo sweep",
        leave=False,
    )
    for role in EVAL_ROLES:
        role_metrics = {}
        for eval_max_n_demos in EVAL_MAX_N_DEMOS_SWEEP:
            eval_job_bar.set_postfix(
                role=str(role),
                demos=int(eval_max_n_demos),
            )
            role_metrics[int(eval_max_n_demos)] = _evaluate_role_for_demo(
                case.optimizer,
                role=str(role),
                tokenizer=SHARED_TOKENIZER,
                rule_bank=BASE_BANK,
                n_seq_ar=N_SEQ_AR,
                max_completion_len=MAX_COMPLETION_LEN,
                n_iters=EVAL_ITERS_PER_ROLE,
                eval_max_n_demos=int(eval_max_n_demos),
                model_fn=model_fn,
                shared_adapter=shared_adapter,
            )
            eval_job_bar.update(1)
        metrics_by_role_eval_demo[str(role)] = role_metrics
    eval_job_bar.close()

    post_eval_wall_s = time.perf_counter() - post_eval_start

    selection_metric_name = "rollout_success_rate"
    selection_metric_value = _metric_by_role_demo(
        metrics_by_role_eval_demo,
        role="eval",
        eval_max_n_demos=int(SELECTION_EVAL_MAX_N_DEMOS),
        metric_name=selection_metric_name,
    )

    row = {
        "run_id": RUN_ID,
        "name": case.name,
        "model_family": case.info["model_family"],
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
        "metrics_by_role_eval_demo": metrics_by_role_eval_demo,
        "selection_role": "eval",
        "selection_eval_max_n_demos": int(SELECTION_EVAL_MAX_N_DEMOS),
        "selection_metric_name": selection_metric_name,
        "selection_metric_value": float(selection_metric_value),
        "eval_rollout_success_rate": _metric_by_role_demo(
            metrics_by_role_eval_demo,
            role="eval",
            eval_max_n_demos=int(SELECTION_EVAL_MAX_N_DEMOS),
            metric_name="rollout_success_rate",
        ),
        "train_rollout_success_rate": _metric_by_role_demo(
            metrics_by_role_eval_demo,
            role="train",
            eval_max_n_demos=int(SELECTION_EVAL_MAX_N_DEMOS),
            metric_name="rollout_success_rate",
        ),
        "fresh_icl_config": {
            "base_bank_seed": BASE_BANK_SEED,
            "predicates_per_layer": PREDICATES_PER_LAYER,
            "rules_per_transition": RULES_PER_TRANSITION,
            "fresh_icl_n_predicates": FRESH_ICL_N_PREDICATES,
            "n_layers": N_LAYERS,
            "arity_max": ARITY_MAX,
            "vars_per_rule_max": VARS_PER_RULE_MAX,
            "k_in_max": K_IN_MAX,
            "k_out_max": K_OUT_MAX,
            "initial_ant_max": INITIAL_ANT_MAX,
            "constants": list(CONSTANTS),
            "train_max_n_demos": int(TRAIN_MAX_N_DEMOS),
            "eval_max_n_demos_sweep": [int(v) for v in EVAL_MAX_N_DEMOS_SWEEP],
            "sample_max_attempts": SAMPLE_MAX_ATTEMPTS,
            "max_unify_solutions": MAX_UNIFY_SOLUTIONS,
        },
        "dims": DIMS,
        "train_wall_s": float(train_wall_s),
        "post_eval_wall_s": float(post_eval_wall_s),
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
