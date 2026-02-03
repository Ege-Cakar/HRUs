"""Playground: small Transformer on ImplySizeTask."""

#<codecell>
from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import optax

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from model.transformer import TransformerConfig
from task.prop import ImplySizeTask
from train import train
from common import rule_membership_accuracy

DS_PATH = ROOT / "task" / "prop_gen" / "data" / "toy_imply"
TRAIN_SIZES = (2, 5)
TEST_SIZES = (6, 12)

BATCH_SIZE = 64
TRAIN_ITERS = 1800
TEST_EVERY = 200
TEST_ITERS = 5
SEED = 42

RULE_CLASSES = 12  # rule_type_to_id max (11) + 1 for pad


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
        loss_val = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
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


def eval_on_sizes(
    optimizer,
    ds_path: Path,
    sizes,
    batch_size: int,
    n_iters: int = 5,
):
    metrics_fn = make_metrics_fn()
    for size in sizes:
        task = ImplySizeTask(
            ds_path,
            size_range=(size, size),
            batch_size=batch_size,
            shuffle=True,
            drop_remainder=False,
            worker_count=0,
            return_rule_sets=True,
        )
        metrics = []
        for _ in range(n_iters):
            metrics.append(metrics_fn(optimizer, next(task)))
        avg = {
            k: float(np.mean([float(m[k]) for m in metrics])) for k in metrics[0]
        }
        msg = (
            "SIZE {:02d}: loss={:.4f} rule_acc={:.4f} pos_acc={:.4f} joint_acc={:.4f}"
        ).format(
            size,
            avg["loss"],
            avg["rule_acc"],
            avg["pos_acc"],
            avg["joint_acc"],
        )
        if "rule_membership_acc" in avg:
            msg += " rule_member={:.4f}".format(avg["rule_membership_acc"])
        print(msg)


def main() -> None:
    train_task = ImplySizeTask(
        DS_PATH,
        size_range=TRAIN_SIZES,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_remainder=True,
        worker_count=0,
    )

    test_task = ImplySizeTask(
        DS_PATH,
        size_range=TEST_SIZES,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_remainder=False,
        worker_count=0,
        return_rule_sets=True,
    )

    train_stats = train_task.stats
    test_stats = test_task.stats

    max_token = max(train_stats["max_token"], test_stats["max_token"])
    max_pos = max(train_stats["max_pos"], test_stats["max_pos"])
    max_seq = max(train_stats["max_seq"], test_stats["max_seq"])

    n_vocab = max(128, max_token + 1)
    n_seq = max(128, max_seq)
    n_out = max(RULE_CLASSES, max_pos + 1, TEST_SIZES[1] + 1)

    print(
        "DATA STATS: max_token={} max_pos={} max_seq={} n_vocab={} n_out={} n_seq={}".format(
            max_token, max_pos, max_seq, n_vocab, n_out, n_seq
        )
    )

    config = TransformerConfig(
        n_vocab=n_vocab,
        n_seq=n_seq,
        n_layers=4,
        n_hidden=64,
        n_heads=4,
        n_out=n_out,
        n_pred_tokens=2,
        pos_encoding="none",
        layer_norm=True,
        use_bias=True,
        dropout_rate=0.0,
        output_mode="last_nonpad",
        pad_token_id=0,
        use_swiglu=True
    )

    metrics_fn = make_metrics_fn()

    optimizer, hist = train(
        config,
        train_iter=train_task,
        test_iter=test_task,
        loss="ce",
        eval_fns=[metrics_fn],
        print_fn=make_print_fn(),
        train_iters=TRAIN_ITERS,
        test_iters=TEST_ITERS,
        test_every=TEST_EVERY,
        lr=5e-4,
        seed=SEED,
        use_tqdm=False,
    )

    print("\nEval on longer sizes:")
    eval_on_sizes(
        optimizer,
        DS_PATH,
        range(TEST_SIZES[0], TEST_SIZES[1] + 1),
        BATCH_SIZE,
        n_iters=TEST_ITERS,
    )


if __name__ == "__main__":
    main()

# %%
