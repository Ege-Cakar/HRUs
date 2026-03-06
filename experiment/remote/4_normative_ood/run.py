"""
Run Transformer + normative baselines on implication-size OOD regimes.
"""

from __future__ import annotations

from dataclasses import asdict
import itertools
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

from common import (
    expected_calibration_error_from_logits,
    multiclass_nll_from_logits,
    new_seed,
    rule_membership_accuracy,
    split_cases,
)
from model.normative import (
    BoundedProverConfig,
    BoundedProverPolicy,
    KernelConfig,
    KernelPolicy,
    LogLinearConfig,
    LogLinearPolicy,
    build_choice_dataset,
    evaluate_policy_on_examples,
)
from model.transformer import TransformerConfig
from task.prop import ImplySizeTask
from train import Case
from wandb_utils import make_experiment_wandb_config


RUN_ID = new_seed()
print("RUN ID", RUN_ID)

WANDB_PROJECT = Path(__file__).resolve().parent.name
WANDB_API_KEY_PATH = ROOT / "key" / "wandb.txt"
USE_WANDB = True

DS_PATH = Path("/n/netscratch/pehlevan_lab/Lab/wlt/data/math/toy_imply")

TRAIN_MAXIMA = [3, 5, 10]
TEST_MAX = 25
SIZE_REGIMES = [
    {
        "train_sizes": (2, train_max),
        "test_sizes": (train_max + 1, TEST_MAX),
        "train_max": train_max,
        "test_max": TEST_MAX,
    }
    for train_max in TRAIN_MAXIMA
]

POS_ENCODINGS = ["none", "absolute", "rope"]
N_LAYERS = [2, 4]
HEAD_DIMS = [16, 32]
N_HEADS = [2, 4, 8]
LRS = [3e-4, 1e-3]
NONLINEARITIES = [("gelu", False), ("swiglu", True)]

BATCH_SIZE = 64
TRAIN_ITERS = 5_000
TEST_EVERY = 500
TEST_ITERS = 3

RULE_CLASSES = 12
RUN_SPLIT = 54

NORM_TRAIN_BATCHES = 128
NORM_TEST_BATCHES = 64

### START TEST CONFIGS
# DS_PATH = Path(ROOT / "task" / "prop_gen" / "data" / "toy_imply")

# TRAIN_MAXIMA = [5]
# TEST_MAX = 10
# SIZE_REGIMES = [
#     {
#         "train_sizes": (2, train_max),
#         "test_sizes": (train_max + 1, TEST_MAX),
#         "train_max": train_max,
#         "test_max": TEST_MAX,
#     }
#     for train_max in TRAIN_MAXIMA
# ]

# POS_ENCODINGS = ["none"]
# N_LAYERS = [2]
# HEAD_DIMS = [16]
# N_HEADS = [2]
# LRS = [1e-3]
# NONLINEARITIES = [("gelu", False)]

# BATCH_SIZE = 64
# TRAIN_ITERS = 200
# TEST_EVERY = 100
# TEST_ITERS = 1

# RUN_SPLIT = 1
# NORM_TRAIN_BATCHES = 8
# NORM_TEST_BATCHES = 4
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


def make_transformer_metrics_fn():
    def _metrics(optimizer, batch, loss=None):
        if len(batch) == 2:
            xs, labels = batch
            rule_set_batch = None
            rule_set_mask = None
        elif len(batch) == 4:
            xs, labels, rule_set_batch, rule_set_mask = batch
        else:
            raise ValueError(f"Unexpected batch size: {len(batch)}")

        logits = optimizer.model(xs)
        loss_val = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        preds = jnp.argmax(logits, axis=-1)
        rule_pred = preds[:, 0]
        pos_pred = preds[:, 1]
        rule_true = labels[:, 0]
        pos_true = labels[:, 1]
        rule_acc = jnp.mean(rule_pred == rule_true)
        pos_acc = jnp.mean(pos_pred == pos_true)
        joint_acc = jnp.mean((rule_pred == rule_true) & (pos_pred == pos_true))

        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_labels = labels.reshape(-1)
        metrics = {
            "loss": loss_val,
            "rule_acc": rule_acc,
            "pos_acc": pos_acc,
            "joint_acc": joint_acc,
            "nll": multiclass_nll_from_logits(flat_logits, flat_labels),
            "calibration_ece": expected_calibration_error_from_logits(flat_logits, flat_labels),
        }
        if rule_set_batch is not None and rule_set_mask is not None:
            pred_rules = jnp.stack([rule_pred, pos_pred], axis=-1)
            metrics["rule_membership_acc"] = rule_membership_accuracy(
                pred_rules, rule_set_batch, rule_set_mask
            )
        return metrics

    return _metrics


def make_print_fn():
    def _print(step, hist):
        train_metrics = hist["train"][-1]
        test_metrics = hist["test"][-1]
        msg = (
            "ITER {}: train_joint={:.4f} train_nll={:.4f} "
            "test_joint={:.4f} test_nll={:.4f}"
        ).format(
            step,
            train_metrics["joint_acc"],
            train_metrics["nll"],
            test_metrics["joint_acc"],
            test_metrics["nll"],
        )
        if "rule_membership_acc" in test_metrics:
            msg += " test_member={:.4f}".format(test_metrics["rule_membership_acc"])
        print(msg)

    return _print


def _stats_for_sizes(size_range):
    return ImplySizeTask.stats_from_metadata(DS_PATH, size_range)


def _compute_vocab_seq_out(train_sizes, test_sizes):
    train_stats = _stats_for_sizes(train_sizes)
    test_stats = _stats_for_sizes(test_sizes)

    max_token = max(train_stats["max_token"], test_stats["max_token"])
    max_pos = max(train_stats["max_pos"], test_stats["max_pos"])
    max_seq = max(train_stats["max_seq"], test_stats["max_seq"])

    n_vocab = max(128, max_token + 1)
    n_seq = max(128, max_seq)
    n_out = max(RULE_CLASSES, max_pos + 1, TEST_MAX + 1)
    return n_vocab, n_seq, n_out


def _make_task(size_range, *, return_rule_sets, drop_remainder):
    return ImplySizeTask(
        DS_PATH,
        size_range=size_range,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_remainder=drop_remainder,
        worker_count=0,
        return_rule_sets=return_rule_sets,
    )


def _collect_choice_examples(task, n_batches):
    all_examples = []
    for _ in range(n_batches):
        batch = next(task)
        if len(batch) == 2:
            raise ValueError("Need return_rule_sets=True to collect normative examples.")
        xs, ys, rule_set_batch, rule_set_mask = batch
        examples = build_choice_dataset(xs, ys, rule_set_batch, rule_set_mask, skip_missing=True)
        all_examples.extend(examples)
    return all_examples


metrics_fn = make_transformer_metrics_fn()
print_fn = make_print_fn()

all_cases = []
for regime in SIZE_REGIMES:
    train_sizes = regime["train_sizes"]
    test_sizes = regime["test_sizes"]
    n_vocab, n_seq, n_out = _compute_vocab_seq_out(train_sizes, test_sizes)

    for (
        pos_encoding,
        n_layers,
        head_dim,
        n_heads,
        lr,
        (nonlin_name, use_swiglu),
    ) in itertools.product(
        POS_ENCODINGS,
        N_LAYERS,
        HEAD_DIMS,
        N_HEADS,
        LRS,
        NONLINEARITIES,
    ):
        n_hidden = head_dim * n_heads
        config = TransformerConfig(
            n_vocab=n_vocab,
            n_seq=n_seq,
            n_layers=n_layers,
            n_hidden=n_hidden,
            n_heads=n_heads,
            n_out=n_out,
            n_pred_tokens=2,
            pos_encoding=pos_encoding,
            layer_norm=True,
            use_bias=True,
            dropout_rate=0.0,
            output_mode="last_nonpad",
            pad_token_id=0,
            use_swiglu=use_swiglu,
        )

        train_task = _make_task(train_sizes, return_rule_sets=False, drop_remainder=True)
        test_task = _make_task(test_sizes, return_rule_sets=True, drop_remainder=False)

        train_args = {
            "loss": "ce",
            "eval_fns": [metrics_fn],
            "print_fn": print_fn,
            "train_iters": TRAIN_ITERS,
            "test_iters": TEST_ITERS,
            "test_every": TEST_EVERY,
            "lr": lr,
        }
        info = {
            "train_sizes": train_sizes,
            "test_sizes": test_sizes,
            "train_max": regime["train_max"],
            "test_max": regime["test_max"],
            "pos_encoding": pos_encoding,
            "n_layers": n_layers,
            "head_dim": head_dim,
            "n_heads": n_heads,
            "n_hidden": n_hidden,
            "lr": lr,
            "nonlinearity": nonlin_name,
        }
        train_args["wandb_cfg"] = _make_case_wandb_cfg(
            case_name="ImplySizeNormative",
            model_config=config,
            train_args=train_args,
            info=info,
        )
        all_cases.append(
            Case(
                "ImplySizeNormative",
                config,
                train_task=train_task,
                test_task=test_task,
                train_args=train_args,
                info=info,
            )
        )

print("TOTAL CASES:", len(all_cases))
all_cases = split_cases(all_cases, RUN_SPLIT, shuffle_seed=200)
print("CASES", all_cases)

rows = []
for case in all_cases:
    print("RUNNING", case.name, case.info)
    case.run()
    transformer_test = case.hist["test"][-1] if case.hist and case.hist["test"] else {}
    rows.append(
        {
            "run_id": RUN_ID,
            "model_family": "transformer",
            "name": case.name,
            "info": case.info,
            "metrics": transformer_test,
            "n_examples": np.nan,
        }
    )

    train_norm_task = _make_task(case.info["train_sizes"], return_rule_sets=True, drop_remainder=False)
    test_norm_task = _make_task(case.info["test_sizes"], return_rule_sets=True, drop_remainder=False)
    train_examples = _collect_choice_examples(train_norm_task, NORM_TRAIN_BATCHES)
    test_examples = _collect_choice_examples(test_norm_task, NORM_TEST_BATCHES)
    print("NORM EXAMPLES", len(train_examples), len(test_examples))

    loglinear = LogLinearPolicy(
        LogLinearConfig(lr=0.05, weight_decay=1e-4, max_steps=800, batch_size=128, seed=0)
    ).fit(train_examples)
    kernel = KernelPolicy(KernelConfig(gamma=0.01)).fit(train_examples)
    bounded = BoundedProverPolicy(BoundedProverConfig(temperature=1.0, fit_temperature=True)).fit(
        train_examples
    )

    norm_evals = {
        "loglinear": asdict(evaluate_policy_on_examples(loglinear, test_examples, uses_sequent_tokens=False)),
        "kernel": asdict(evaluate_policy_on_examples(kernel, test_examples, uses_sequent_tokens=False)),
        "bounded_prover": asdict(evaluate_policy_on_examples(bounded, test_examples, uses_sequent_tokens=True)),
    }
    for family, metrics in norm_evals.items():
        rows.append(
            {
                "run_id": RUN_ID,
                "model_family": family,
                "name": case.name,
                "info": case.info,
                "metrics": metrics,
                "n_examples": len(test_examples),
            }
        )

    case.optimizer = None
    case.train_task = None
    case.test_task = None
    case.train_args["eval_fns"] = None
    case.train_args["print_fn"] = None

df = pd.DataFrame(rows)
save_dir = Path("set")
save_dir.mkdir(parents=True, exist_ok=True)
df.to_pickle(save_dir / f"res.{RUN_ID}.pkl")
print("done!")
