"""Architecture sweep on ImplyAutoregSizeTask length generalization."""

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

from common import new_seed, split_cases
from model.mlp import CompletionMixerConfig
from model.ssm_bonsai import Mamba2BonsaiConfig
from model.transformer import TransformerConfig
from task.prop_ar import ImplyAutoregSizeTask
from task.prop_gen.util.tokenize_ar import eot_token_id, sep_token_id
from train import Case, ce_mask

from experiment.utils.data_utils import build_completion_targets, build_prompt_only_inputs
from experiment.utils.metrics_utils import final_token_accuracy


RUN_ID = new_seed()
print("RUN ID", RUN_ID)

DS_PATH = Path("/n/netscratch/pehlevan_lab/Lab/wlt/data/math/toy_imply_ar")

TRAIN_MAX_SIZES = [5, 10, 15]
EVAL_SIZES = list(range(5, 26))

BATCH_SIZE = 64
TRAIN_ITERS = 5_000
TEST_EVERY = 500
TEST_ITERS = 3
EVAL_ITERS_PER_SIZE = 3

RUN_SPLIT = 12

TRANSFORMER_LAYERS = [4, 8]
TRANSFORMER_WIDTH_HEADS = [(128, 4), (256, 8)]
TRANSFORMER_LRS = [1e-4, 3e-4, 1e-3]
TRANSFORMER_POS = ["rope"]
TRANSFORMER_SWIGLU = [True]

MAMBA2_BONSAI_LAYERS = [4, 8]
MAMBA2_BONSAI_WIDTH_HEADS = [(128, 4), (256, 8)]
MAMBA2_BONSAI_D_STATE = [16, 32]
MAMBA2_BONSAI_D_CONV = [4, 8]
MAMBA2_BONSAI_SCAN_CHUNK_LEN = [64]
MAMBA2_BONSAI_LRS = [1e-4, 3e-4, 1e-3]

MIXER_LAYERS = [4, 8]
MIXER_HIDDEN = [128, 256]
MIXER_CHANNELS = [64, 128]
MIXER_LRS = [3e-4, 1e-3, 3e-3]

### START TEST CONFIGS
# DS_PATH = Path(ROOT / "task" / "prop_gen" / "data" / "test")
# TRAIN_MAX_SIZES = [5, 10, 15]
# EVAL_SIZES = list(range(5, 21))

# BATCH_SIZE = 32
# TRAIN_ITERS = 100
# TEST_EVERY = 50
# TEST_ITERS = 1
# EVAL_ITERS_PER_SIZE = 1

# RUN_SPLIT = 1

# TRANSFORMER_LAYERS = [4]
# TRANSFORMER_WIDTH_HEADS = [(128, 4)]
# TRANSFORMER_LRS = [3e-4]
# TRANSFORMER_POS = ["none"]
# TRANSFORMER_SWIGLU = [False]

# MAMBA2_BONSAI_LAYERS = [4]
# MAMBA2_BONSAI_WIDTH_HEADS = [(128, 4)]
# MAMBA2_BONSAI_D_STATE = [16]
# MAMBA2_BONSAI_D_CONV = [4]
# MAMBA2_BONSAI_SCAN_CHUNK_LEN = [32]
# MAMBA2_BONSAI_LRS = [3e-4]

# MIXER_LAYERS = [4]
# MIXER_HIDDEN = [128]
# MIXER_CHANNELS = [64]
# MIXER_LRS = [1e-3]
### END TEST CONFIGS


def _stats_for_sizes(size_range):
    return ImplyAutoregSizeTask.stats_from_metadata(DS_PATH, size_range)


def _train_sizes_for_max(train_max_size: int) -> list[int]:
    return [int(size) for size in EVAL_SIZES if int(size) <= int(train_max_size)]


def _ood_sizes_for_max(train_max_size: int) -> list[int]:
    return [int(size) for size in EVAL_SIZES if int(size) > int(train_max_size)]


def _compute_dims(train_max_sizes, eval_sizes):
    max_train = int(max(train_max_sizes))
    train_sizes = [int(size) for size in eval_sizes if int(size) <= max_train]
    train_stats = _stats_for_sizes(train_sizes)
    eval_stats = _stats_for_sizes(eval_sizes)
    max_token = max(train_stats["max_token"], eval_stats["max_token"])
    max_seq = max(train_stats["max_seq"], eval_stats["max_seq"])
    max_out_len = max(train_stats["max_completion_seq"], eval_stats["max_completion_seq"])

    n_vocab = max(128, max_token + 1)
    n_seq = int(max_seq)
    max_out_len = int(max_out_len)
    return n_vocab, n_seq, max_out_len


def _make_task(size_range, *, drop_remainder, shuffle):
    return ImplyAutoregSizeTask(
        DS_PATH,
        size_range=size_range,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        drop_remainder=drop_remainder,
        worker_count=0,
    )


class MixerBatchAdapter:
    """Wrap AR batches into prompt-only inputs and fixed completion targets."""

    def __init__(self, base_task, *, n_seq: int, max_out_len: int):
        self.base_task = base_task
        self.n_seq = n_seq
        self.max_out_len = max_out_len

    def __iter__(self):
        return self

    def __next__(self):
        xs, ys = next(self.base_task)
        prompt_xs = build_prompt_only_inputs(
            xs,
            n_seq=self.n_seq,
            sep_token_id=sep_token_id,
            pad_token_id=0,
        )
        targets, _ = build_completion_targets(
            ys,
            max_out_len=self.max_out_len,
            eot_token_id=eot_token_id,
        )
        return prompt_xs, targets


def _mean_metrics(metrics_list):
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    out = {}
    for key in keys:
        out[key] = float(np.mean([float(m[key]) for m in metrics_list]))
    return out


def make_ar_metrics_fn():
    def _metrics(optimizer, batch, loss=None):
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


def make_mixer_metrics_fn():
    def _metrics(optimizer, batch, loss=None):
        xs, labels = batch
        logits = optimizer.model(xs)

        loss_val = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        preds = jnp.argmax(logits, axis=-1)

        token_acc_full = jnp.mean(preds == labels)

        is_eot = labels == eot_token_id
        first_eot = jnp.argmax(is_eot, axis=1)
        batch_idx = jnp.arange(labels.shape[0])
        pos_idx = jnp.arange(labels.shape[1])[None, :]
        unpadded_mask = pos_idx <= first_eot[:, None]

        unpadded_total = jnp.maximum(jnp.sum(unpadded_mask), 1)
        token_acc_unpadded = jnp.sum((preds == labels) & unpadded_mask) / unpadded_total

        seq_correct_unpadded = (preds == labels) | (~unpadded_mask)
        seq_exact_unpadded = jnp.mean(jnp.all(seq_correct_unpadded, axis=1))

        eot_pos_acc = jnp.mean(preds[batch_idx, first_eot] == eot_token_id)

        return {
            "loss": loss_val,
            "token_acc_full": token_acc_full,
            "token_acc_unpadded": token_acc_unpadded,
            "seq_exact_acc_unpadded": seq_exact_unpadded,
            "eot_pos_acc": eot_pos_acc,
            "final_token_acc": eot_pos_acc,
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


def _evaluate_by_size(optimizer, *, family: str, n_seq: int, max_out_len: int, n_iters: int):
    metrics_fn = make_mixer_metrics_fn() if family == "mixer_completion" else make_ar_metrics_fn()

    size_metrics = {}
    for size in EVAL_SIZES:
        base_task = _make_task((size, size), drop_remainder=False, shuffle=True)
        eval_task = (
            MixerBatchAdapter(base_task, n_seq=n_seq, max_out_len=max_out_len)
            if family == "mixer_completion"
            else base_task
        )

        all_metrics = []
        for _ in range(n_iters):
            all_metrics.append(metrics_fn(optimizer, next(eval_task)))
        size_metrics[int(size)] = _mean_metrics(all_metrics)

    return size_metrics


def _avg_ood_metric(size_metrics, metric_name, ood_sizes):
    vals = []
    for size in ood_sizes:
        metrics = size_metrics.get(int(size), {})
        val = metrics.get(metric_name)
        if val is not None:
            vals.append(float(val))
    return float(np.mean(vals)) if vals else float("nan")


n_vocab, n_seq, max_out_len = _compute_dims(TRAIN_MAX_SIZES, EVAL_SIZES)
print("DATA DIMS", {"n_vocab": n_vocab, "n_seq": n_seq, "max_out_len": max_out_len})

all_cases = []

ar_metrics_fn = make_ar_metrics_fn()
mixer_metrics_fn = make_mixer_metrics_fn()

for train_max_size in TRAIN_MAX_SIZES:
    train_sizes = _train_sizes_for_max(train_max_size)
    ood_sizes = _ood_sizes_for_max(train_max_size)
    if not train_sizes:
        raise ValueError(f"No train sizes found for train_max_size={train_max_size}")
    if not ood_sizes:
        raise ValueError(
            f"No OOD sizes found for train_max_size={train_max_size}; "
            "ensure train_max_size is smaller than max(EVAL_SIZES)."
        )

    # Transformer cases.
    for n_layers, (n_hidden, n_heads), lr, pos_encoding, use_swiglu in itertools.product(
        TRANSFORMER_LAYERS,
        TRANSFORMER_WIDTH_HEADS,
        TRANSFORMER_LRS,
        TRANSFORMER_POS,
        TRANSFORMER_SWIGLU,
    ):
        config = TransformerConfig(
            n_vocab=n_vocab,
            n_seq=n_seq,
            n_layers=n_layers,
            n_hidden=n_hidden,
            n_heads=n_heads,
            n_out=n_vocab,
            n_pred_tokens=1,
            pos_encoding=pos_encoding,
            layer_norm=True,
            use_swiglu=use_swiglu,
            use_bias=True,
            dropout_rate=0.0,
            output_mode="full_sequence",
            pad_token_id=0,
        )

        train_task = _make_task(train_sizes, drop_remainder=True, shuffle=True)
        test_task = _make_task(ood_sizes, drop_remainder=False, shuffle=True)

        train_args = {
            "loss": "ce_mask",
            "eval_fns": [ar_metrics_fn],
            "print_fn": make_print_fn("final_token_acc"),
            "train_iters": TRAIN_ITERS,
            "test_iters": TEST_ITERS,
            "test_every": TEST_EVERY,
            "lr": lr,
        }

        info = {
            "model_family": "transformer",
            "target_format": "next_token_full_sequence",
            "train_max_size": int(train_max_size),
            "train_sizes": train_sizes,
            "eval_sizes": EVAL_SIZES,
            "ood_sizes": ood_sizes,
            "n_layers": n_layers,
            "n_hidden": n_hidden,
            "n_heads": n_heads,
            "pos_encoding": pos_encoding,
            "use_swiglu": use_swiglu,
            "lr": lr,
            "n_vocab": n_vocab,
            "n_seq": n_seq,
        }

        all_cases.append(
            Case(
                f"5_arch_sweep_transformer_tmax{int(train_max_size):02d}",
                config,
                train_task=train_task,
                test_task=test_task,
                train_args=train_args,
                info=info,
            )
        )

    # Mamba-2 Bonsai cases.
    for n_layers, (n_hidden, n_heads), d_state, d_conv, scan_chunk_len, lr in itertools.product(
        MAMBA2_BONSAI_LAYERS,
        MAMBA2_BONSAI_WIDTH_HEADS,
        MAMBA2_BONSAI_D_STATE,
        MAMBA2_BONSAI_D_CONV,
        MAMBA2_BONSAI_SCAN_CHUNK_LEN,
        MAMBA2_BONSAI_LRS,
    ):
        config = Mamba2BonsaiConfig(
            n_vocab=n_vocab,
            n_seq=n_seq,
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
            d_state=d_state,
            d_conv=d_conv,
            expand=2,
            scan_chunk_len=scan_chunk_len,
        )

        train_task = _make_task(train_sizes, drop_remainder=True, shuffle=True)
        test_task = _make_task(ood_sizes, drop_remainder=False, shuffle=True)

        train_args = {
            "loss": "ce_mask",
            "eval_fns": [ar_metrics_fn],
            "print_fn": make_print_fn("final_token_acc"),
            "train_iters": TRAIN_ITERS,
            "test_iters": TEST_ITERS,
            "test_every": TEST_EVERY,
            "lr": lr,
        }

        info = {
            "model_family": "mamba2",
            "target_format": "next_token_full_sequence",
            "train_max_size": int(train_max_size),
            "train_sizes": train_sizes,
            "eval_sizes": EVAL_SIZES,
            "ood_sizes": ood_sizes,
            "n_layers": n_layers,
            "n_hidden": n_hidden,
            "n_heads": n_heads,
            "d_state": d_state,
            "d_conv": d_conv,
            "scan_chunk_len": scan_chunk_len,
            "lr": lr,
            "n_vocab": n_vocab,
            "n_seq": n_seq,
        }

        all_cases.append(
            Case(
                f"5_arch_sweep_mamba2_tmax{int(train_max_size):02d}",
                config,
                train_task=train_task,
                test_task=test_task,
                train_args=train_args,
                info=info,
            )
        )

    # Mixer completion-at-once cases.
    for n_layers, n_hidden, n_channels, lr in itertools.product(
        MIXER_LAYERS,
        MIXER_HIDDEN,
        MIXER_CHANNELS,
        MIXER_LRS,
    ):
        config = CompletionMixerConfig(
            n_vocab=n_vocab,
            n_seq=n_seq,
            n_layers=n_layers,
            n_hidden=n_hidden,
            n_channels=n_channels,
            n_out_seq=max_out_len,
            n_out_vocab=n_vocab,
            act_fn="gelu",
            layer_norm=True,
            use_bias=True,
        )

        train_base = _make_task(train_sizes, drop_remainder=True, shuffle=True)
        test_base = _make_task(ood_sizes, drop_remainder=False, shuffle=True)

        train_task = MixerBatchAdapter(train_base, n_seq=n_seq, max_out_len=max_out_len)
        test_task = MixerBatchAdapter(test_base, n_seq=n_seq, max_out_len=max_out_len)

        train_args = {
            "loss": "ce",
            "eval_fns": [mixer_metrics_fn],
            "print_fn": make_print_fn("eot_pos_acc"),
            "train_iters": TRAIN_ITERS,
            "test_iters": TEST_ITERS,
            "test_every": TEST_EVERY,
            "lr": lr,
        }

        info = {
            "model_family": "mixer_completion",
            "target_format": "completion_left_aligned_eot_padded",
            "train_max_size": int(train_max_size),
            "train_sizes": train_sizes,
            "eval_sizes": EVAL_SIZES,
            "ood_sizes": ood_sizes,
            "n_layers": n_layers,
            "n_hidden": n_hidden,
            "n_channels": n_channels,
            "lr": lr,
            "n_vocab": n_vocab,
            "n_seq": n_seq,
            "max_out_len": max_out_len,
        }

        all_cases.append(
            Case(
                f"5_arch_sweep_mixer_completion_tmax{int(train_max_size):02d}",
                config,
                train_task=train_task,
                test_task=test_task,
                train_args=train_args,
                info=info,
            )
        )

print("TOTAL CASES:", len(all_cases))
all_cases = split_cases(all_cases, RUN_SPLIT, shuffle_seed=200)
print("CASES IN THIS RUN:", len(all_cases))


# <codecell>
rows = []
for case in all_cases:
    print("RUNNING", case.name, case.info)
    case.run()

    family = case.info["model_family"]
    final_test = case.hist["test"][-1] if case.hist and case.hist.get("test") else {}

    size_metrics = _evaluate_by_size(
        case.optimizer,
        family=family,
        n_seq=n_seq,
        max_out_len=max_out_len,
        n_iters=EVAL_ITERS_PER_SIZE,
    )

    key_metric = "eot_pos_acc" if family == "mixer_completion" else "final_token_acc"
    row = {
        "run_id": RUN_ID,
        "name": case.name,
        "model_family": family,
        "info": case.info,
        "train_args": {
            "loss": case.train_args["loss"],
            "train_iters": case.train_args["train_iters"],
            "test_iters": case.train_args["test_iters"],
            "test_every": case.train_args["test_every"],
            "lr": case.train_args["lr"],
        },
        "metrics_final": final_test,
        "metrics_by_size": size_metrics,
        "ood_metric_name": key_metric,
        "ood_metric_avg": _avg_ood_metric(
            size_metrics,
            key_metric,
            ood_sizes=case.info.get("ood_sizes", []),
        ),
        "hist": case.hist,
    }
    rows.append(row)

    case.optimizer = None
    case.hist = None
    case.train_task = None
    case.test_task = None

save_dir = Path("set")
save_dir.mkdir(parents=True, exist_ok=True)

pd.DataFrame(rows).to_pickle(save_dir / f"res.{RUN_ID}.pkl")
print("done!")
