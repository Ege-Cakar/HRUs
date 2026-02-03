"""
Sweep Transformer hyperparameters on ImplySizeTask.
"""

# <codecell>
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

from common import new_seed, split_cases, rule_membership_accuracy
from model.transformer import TransformerConfig
from task.prop import ImplySizeTask
from train import Case

RUN_ID = new_seed()
print("RUN ID", RUN_ID)

# Placeholder: update when dataset is available.
DS_PATH = Path("/n/netscratch/pehlevan_lab/Lab/wlt/data/math/toy_imply")

# Sweep sizes: train ranges (2..max), test ranges (max+1..25).
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

# Sweep hyperparameters (medium scale).
POS_ENCODINGS = ["none", "absolute", "rope"]
N_LAYERS = [2, 4, 8]
HEAD_DIMS = [16, 32]
N_HEADS = [2, 4, 8, 16, 32]
LRS = [1e-4, 3e-4, 1e-3, 3e-3]
NONLINEARITIES = [("gelu", False), ("swiglu", True)]

# Training schedule (fast).
BATCH_SIZE = 64
TRAIN_ITERS = 5_000
TEST_EVERY = 500
TEST_ITERS = 3

RULE_CLASSES = 12
RUN_SPLIT = 216

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
# LRS = [3e-3]
# NONLINEARITIES = [("gelu", False)]

# BATCH_SIZE = 64
# TRAIN_ITERS = 100
# TEST_EVERY = 50
# TEST_ITERS = 1

# RULE_CLASSES = 12
# RUN_SPLIT = 1
### END TEST CONFIGS


def make_metrics_fn():
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
        loss_val = optax.softmax_cross_entropy_with_integer_labels(
            logits, labels
        ).mean()
        preds = jnp.argmax(logits, axis=-1)
        rule_pred = preds[:, 0]
        pos_pred = preds[:, 1]
        rule_true = labels[:, 0]
        pos_true = labels[:, 1]
        rule_acc = jnp.mean(rule_pred == rule_true)
        pos_acc = jnp.mean(pos_pred == pos_true)
        joint_acc = jnp.mean((rule_pred == rule_true) & (pos_pred == pos_true))

        metrics = {
            "loss": loss_val,
            "rule_acc": rule_acc,
            "pos_acc": pos_acc,
            "joint_acc": joint_acc,
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
            "ITER {}: train_loss={:.4f} train_joint={:.4f} "
            "test_loss={:.4f} test_joint={:.4f}"
        ).format(
            step,
            train_metrics["loss"],
            train_metrics["joint_acc"],
            test_metrics["loss"],
            test_metrics["joint_acc"],
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


metrics_fn = make_metrics_fn()
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

        train_task = _make_task(
            train_sizes, return_rule_sets=False, drop_remainder=True
        )
        test_task = _make_task(
            test_sizes, return_rule_sets=True, drop_remainder=False
        )

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

        all_cases.append(
            Case(
                "ImplySize",
                config,
                train_task=train_task,
                test_task=test_task,
                train_args=train_args,
                info=info,
            )
        )

print('TOTAL CASES:', len(all_cases))

all_cases = split_cases(all_cases, RUN_SPLIT, shuffle_seed=200)
print("CASES", all_cases)

for case in all_cases:
    print("RUNNING", case.name)
    case.run()
    case.optimizer = None
    case.train_task = None
    case.test_task = None
    case.train_args["eval_fns"] = None
    case.train_args["print_fn"] = None

df = pd.DataFrame(all_cases)
df.to_pickle(f"res.{RUN_ID}.pkl")
print("done!")
