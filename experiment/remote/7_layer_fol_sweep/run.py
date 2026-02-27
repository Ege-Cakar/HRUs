"""Architecture sweep on FOLLayerTask distance generalization in online mode."""

# <codecell>
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
LOCAL_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
sys.path.append(str(LOCAL_DIR))

from common import new_seed, split_cases
from model.eval_adapters import make_model_callable
from model.mlp import CompletionMixerConfig
from model.ssm_bonsai import Mamba2BonsaiConfig
from model.transformer import TransformerConfig
from task.layer_fol import (
    AutoregressiveLogitsAdapter,
    CompletionLogitsAdapter,
    FOLLayerTask,
    evaluate_layer_rollouts,
    evaluate_rule_matches,
    sample_rollout_examples,
)
from task.layer_gen.util import tokenize_layer_fol
from task.layer_gen.util.fol_rule_bank import (
    build_random_fol_rule_bank,
    save_fol_rule_bank,
)
from train import Case, ce_mask

from experiment.utils.batch_adapters import MixerBatchAdapter
from experiment.utils.data_utils import (
    build_prompt_only_inputs,
    pad_completion_targets,
)
from utils import (
    extract_ar_rule_match_inputs,
    extract_completion_rule_match_inputs,
    summarize_rule_match_metrics,
)
from experiment.utils.metrics_utils import final_token_accuracy


RUN_ID = new_seed()
print("RUN ID", RUN_ID)

TRAIN_MAX_DISTANCES = [3, 5, 10]
EVAL_DISTANCES = list(range(1, 21))

BATCH_SIZE = 64
TRAIN_ITERS = 5_000
TEST_EVERY = 1000
TEST_ITERS = 3
EVAL_ITERS_PER_DISTANCE = 3
ROLLOUT_EXAMPLES_PER_DISTANCE = 64

RUN_SPLIT = 96

RULE_BANK_SEED = 2026
N_LAYERS = 24
PREDICATES_PER_LAYER = 16
RULES_PER_TRANSITION = 32
K_IN_MAX = 3
K_OUT_MAX = 5
INITIAL_ANT_MAX = 3
ARITY_MAX = 3
VARS_PER_RULE_MAX = 4
CONSTANTS = ("a", "b", "c", "d")
SAMPLE_MAX_ATTEMPTS = 4096
MAX_UNIFY_SOLUTIONS = 128

TRANSFORMER_LAYERS = [4, 8]
TRANSFORMER_WIDTH_HEADS = [(128, 4), (256, 8)]
TRANSFORMER_LRS = [3e-4, 1e-3]
TRANSFORMER_POS = ["rope"]
TRANSFORMER_SWIGLU = [True]

MAMBA2_BONSAI_LAYERS = [4, 8]
MAMBA2_BONSAI_WIDTH_HEADS = [(128, 4), (256, 8)]
MAMBA2_BONSAI_D_STATE = [16, 32]
MAMBA2_BONSAI_D_CONV = [4]
MAMBA2_BONSAI_SCAN_CHUNK_LEN = [64]
MAMBA2_BONSAI_LRS = [3e-4, 1e-3]

MIXER_LAYERS = [4, 8]
MIXER_HIDDEN = [128, 256]
MIXER_CHANNELS = [128]
MIXER_LRS = [1e-3, 3e-3]

### START TEST CONFIGS
# TRAIN_MAX_DISTANCES = [2]
# EVAL_DISTANCES = [1, 2, 8]
# BATCH_SIZE = 8
# TRAIN_ITERS = 20
# TEST_EVERY = 10
# TEST_ITERS = 1
# EVAL_ITERS_PER_DISTANCE = 1
# ROLLOUT_EXAMPLES_PER_DISTANCE = 4
# RUN_SPLIT = 1
# TRANSFORMER_LAYERS = [2]
# TRANSFORMER_WIDTH_HEADS = [(64, 4)]
# TRANSFORMER_LRS = [3e-4]
# MAMBA2_BONSAI_LAYERS = [2]
# MAMBA2_BONSAI_WIDTH_HEADS = [(64, 4)]
# MAMBA2_BONSAI_D_STATE = [8]
# MAMBA2_BONSAI_D_CONV = [4]
# MAMBA2_BONSAI_SCAN_CHUNK_LEN = [16]
# MAMBA2_BONSAI_LRS = [3e-4]
# MIXER_LAYERS = [2]
# MIXER_HIDDEN = [64]
# MIXER_CHANNELS = [64]
# MIXER_LRS = [1e-3]
### END TEST CONFIGS


def _build_shared_rule_bank(save_dir: Path):
    rng = np.random.default_rng(RULE_BANK_SEED)
    rule_bank = build_random_fol_rule_bank(
        n_layers=N_LAYERS,
        predicates_per_layer=PREDICATES_PER_LAYER,
        rules_per_transition=RULES_PER_TRANSITION,
        arity_max=ARITY_MAX,
        vars_per_rule_max=VARS_PER_RULE_MAX,
        k_in_max=K_IN_MAX,
        k_out_max=K_OUT_MAX,
        constants=CONSTANTS,
        rng=rng,
    )

    rule_bank_payload = rule_bank.to_dict()
    canonical = json.dumps(rule_bank_payload, sort_keys=True, separators=(",", ":"))
    rule_bank_sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    rule_bank_path = save_dir / f"rule_bank.{RUN_ID}.json"
    save_fol_rule_bank(rule_bank_path, rule_bank)

    tokenizer = tokenize_layer_fol.build_tokenizer_from_rule_bank(rule_bank)
    return rule_bank, tokenizer, rule_bank_payload, rule_bank_sha256, rule_bank_path


def _compute_dims(rule_bank, tokenizer):
    max_rhs_atoms = max(
        len(rule.rhs)
        for rules in rule_bank.transitions.values()
        for rule in rules
    )
    max_prompt_facts = max(int(INITIAL_ANT_MAX), int(max_rhs_atoms))
    max_atom_len = 1
    first_const = str(rule_bank.constants[0])
    for predicate, arity in rule_bank.predicate_arities.items():
        atom_text = f"{str(predicate)}({','.join(first_const for _ in range(int(arity)))})"
        max_atom_len = max(
            int(max_atom_len),
            len(tokenizer.encode_completion(atom_text)) - 1,
        )
    max_prompt_len = (
        max_prompt_facts * max_atom_len
        + max(0, max_prompt_facts - 1)
        + 1
        + max_atom_len
        + 1
    )
    max_completion_len = 1
    for rules in rule_bank.transitions.values():
        for rule in rules:
            max_completion_len = max(
                max_completion_len,
                len(tokenizer.encode_completion(rule.statement_text)),
            )

    n_seq_ar = int(max_prompt_len + max_completion_len - 1)
    n_seq_completion = max(32, n_seq_ar)
    n_vocab = max(512, int(tokenizer.vocab_size))

    return {
        "n_vocab": int(n_vocab),
        "max_prompt_len": int(max_prompt_len),
        "max_completion_len": int(max_completion_len),
        "n_seq_ar": int(n_seq_ar),
        "n_seq_completion": int(n_seq_completion),
    }


def _train_distances_for_k(k: int) -> list[int]:
    return list(range(1, int(k) + 1))


def _ood_distances_for_k(k: int) -> list[int]:
    return [int(d) for d in EVAL_DISTANCES if int(d) > int(k)]


def _make_layer_task(
    distance_range,
    *,
    prediction_objective: str,
    rule_bank_path: Path,
    seed: int,
    drop_remainder: bool,
    shuffle: bool,
):
    return FOLLayerTask(
        distance_range=distance_range,
        batch_size=BATCH_SIZE,
        mode="online",
        shuffle=shuffle,
        seed=seed,
        worker_count=0,
        drop_remainder=drop_remainder,
        prediction_objective=prediction_objective,
        rule_bank_path=rule_bank_path,
        predicates_per_layer=PREDICATES_PER_LAYER,
        arity_max=ARITY_MAX,
        vars_per_rule_max=VARS_PER_RULE_MAX,
        constants=CONSTANTS,
        initial_ant_max=INITIAL_ANT_MAX,
        sample_max_attempts=SAMPLE_MAX_ATTEMPTS,
        max_unify_solutions=MAX_UNIFY_SOLUTIONS,
    )


def _safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


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


def make_ar_metrics_fn(*, tokenizer, rule_bank):
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

        src_layers, pred_completions, expected_statements = extract_ar_rule_match_inputs(
            preds=np.asarray(preds),
            labels=np.asarray(labels),
            xs=np.asarray(xs),
            tokenizer=tokenizer,
        )
        rule_matches = evaluate_rule_matches(
            rule_bank=rule_bank,
            src_layers=src_layers,
            completion_tokens=pred_completions,
            expected_statement_texts=expected_statements,
            tokenizer=tokenizer,
        )
        rule_metrics = summarize_rule_match_metrics(rule_matches)

        return {
            "loss": loss_val,
            "token_acc": token_acc,
            "final_token_acc": final_acc,
            "seq_exact_acc": seq_exact_acc,
            **rule_metrics,
        }

    return _metrics


def make_mixer_metrics_fn(*, tokenizer, rule_bank, eot_token_id: int):
    def _metrics(optimizer, batch, loss=None):
        _ = loss
        xs, labels = batch
        logits = optimizer.model(xs)

        loss_val = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        preds = jnp.argmax(logits, axis=-1)

        token_acc_full = jnp.mean(preds == labels)

        is_eot = labels == int(eot_token_id)
        first_eot = jnp.argmax(is_eot, axis=1)
        batch_idx = jnp.arange(labels.shape[0])
        pos_idx = jnp.arange(labels.shape[1])[None, :]
        unpadded_mask = pos_idx <= first_eot[:, None]

        unpadded_total = jnp.maximum(jnp.sum(unpadded_mask), 1)
        token_acc_unpadded = jnp.sum((preds == labels) & unpadded_mask) / unpadded_total

        seq_correct_unpadded = (preds == labels) | (~unpadded_mask)
        seq_exact_unpadded = jnp.mean(jnp.all(seq_correct_unpadded, axis=1))

        eot_pos_acc = jnp.mean(preds[batch_idx, first_eot] == int(eot_token_id))

        src_layers, pred_completions, expected_statements = extract_completion_rule_match_inputs(
            preds=np.asarray(preds),
            labels=np.asarray(labels),
            xs=np.asarray(xs),
            tokenizer=tokenizer,
            eot_token_id=eot_token_id,
        )
        rule_matches = evaluate_rule_matches(
            rule_bank=rule_bank,
            src_layers=src_layers,
            completion_tokens=pred_completions,
            expected_statement_texts=expected_statements,
            tokenizer=tokenizer,
        )
        rule_metrics = summarize_rule_match_metrics(rule_matches)

        return {
            "loss": loss_val,
            "token_acc_full": token_acc_full,
            "token_acc_unpadded": token_acc_unpadded,
            "seq_exact_acc_unpadded": seq_exact_unpadded,
            "eot_pos_acc": eot_pos_acc,
            "final_token_acc": eot_pos_acc,
            **rule_metrics,
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


def _evaluate_by_distance(
    optimizer,
    *,
    family: str,
    tokenizer,
    rule_bank,
    rule_bank_path: Path,
    n_seq_ar: int,
    n_seq_completion: int,
    max_completion_len: int,
    n_iters: int,
):
    metrics_fn = (
        make_mixer_metrics_fn(
            tokenizer=tokenizer,
            rule_bank=rule_bank,
            eot_token_id=int(tokenizer.eot_token_id),
        )
        if family == "mixer_completion"
        else make_ar_metrics_fn(tokenizer=tokenizer, rule_bank=rule_bank)
    )

    model_fn = make_model_callable(optimizer, to_numpy=False)
    rollout_adapter = (
        CompletionLogitsAdapter(n_seq=n_seq_completion, pad_token_id=0)
        if family == "mixer_completion"
        else AutoregressiveLogitsAdapter(
            n_seq=n_seq_ar,
            max_completion_len=max_completion_len,
            pad_token_id=0,
            jit_step=True,
        )
    )

    metrics_by_distance = {}
    for distance in EVAL_DISTANCES:
        if family == "mixer_completion":
            base_task = _make_layer_task(
                (distance, distance),
                prediction_objective="all_at_once",
                rule_bank_path=rule_bank_path,
                seed=new_seed(),
                drop_remainder=False,
                shuffle=True,
            )
            eval_task = MixerBatchAdapter(
                base_task,
                prompt_builder=lambda xs: build_prompt_only_inputs(
                    xs,
                    n_seq=n_seq_completion,
                    sep_token_id=int(tokenizer.sep_token_id),
                    pad_token_id=0,
                ),
                target_builder=lambda ys: pad_completion_targets(
                    ys,
                    max_out_len=max_completion_len,
                    eot_token_id=int(tokenizer.eot_token_id),
                ),
            )
        else:
            eval_task = _make_layer_task(
                (distance, distance),
                prediction_objective="autoregressive",
                rule_bank_path=rule_bank_path,
                seed=new_seed(),
                drop_remainder=False,
                shuffle=True,
            )

        all_batch_metrics = []
        for _ in range(int(n_iters)):
            all_batch_metrics.append(metrics_fn(optimizer, next(eval_task)))
        agg = _mean_metrics(all_batch_metrics)

        rollout_examples = sample_rollout_examples(
            rule_bank=rule_bank,
            distance=int(distance),
            n_examples=int(ROLLOUT_EXAMPLES_PER_DISTANCE),
            initial_ant_max=int(INITIAL_ANT_MAX),
            max_steps=int(distance),
            max_unify_solutions=int(MAX_UNIFY_SOLUTIONS),
            rng=np.random.default_rng(int(RUN_ID) + 1000 * int(distance)),
        )
        rollout_metrics = evaluate_layer_rollouts(
            rule_bank=rule_bank,
            examples=rollout_examples,
            model=model_fn,
            adapter=rollout_adapter,
            tokenizer=tokenizer,
            temperature=0.0,
            rng=np.random.default_rng(int(RUN_ID) + 2000 * int(distance)),
        )

        agg.update(
            {
                "rollout_n_examples": int(rollout_metrics.n_examples),
                "rollout_success_rate": float(rollout_metrics.success_rate),
                "rollout_decode_error_rate": float(rollout_metrics.decode_error_rate),
                "rollout_unknown_rule_error_rate": float(rollout_metrics.unknown_rule_error_rate),
                "rollout_inapplicable_rule_error_rate": float(
                    rollout_metrics.inapplicable_rule_error_rate
                ),
                "rollout_goal_not_reached_rate": float(rollout_metrics.goal_not_reached_rate),
                "rollout_avg_steps": float(rollout_metrics.avg_steps),
            }
        )

        metrics_by_distance[int(distance)] = agg

    return metrics_by_distance


def _avg_ood_metric(metrics_by_distance, metric_name: str, ood_distances: list[int]) -> float:
    vals = []
    for distance in ood_distances:
        metrics = metrics_by_distance.get(int(distance), {})
        value = metrics.get(metric_name)
        if value is not None and not np.isnan(float(value)):
            vals.append(float(value))
    return _safe_mean(vals)


save_dir = Path("set")
save_dir.mkdir(parents=True, exist_ok=True)

(
    SHARED_RULE_BANK,
    SHARED_TOKENIZER,
    SHARED_RULE_BANK_PAYLOAD,
    SHARED_RULE_BANK_SHA256,
    SHARED_RULE_BANK_PATH,
) = _build_shared_rule_bank(save_dir)

DIMS = _compute_dims(SHARED_RULE_BANK, SHARED_TOKENIZER)
N_VOCAB = DIMS["n_vocab"]
MAX_COMPLETION_LEN = DIMS["max_completion_len"]
N_SEQ_AR = DIMS["n_seq_ar"]
N_SEQ_COMPLETION = DIMS["n_seq_completion"]

print("DATA DIMS", DIMS)
print("RULE BANK", {"path": str(SHARED_RULE_BANK_PATH), "sha256": SHARED_RULE_BANK_SHA256})

all_cases = []

ar_metrics_fn = make_ar_metrics_fn(tokenizer=SHARED_TOKENIZER, rule_bank=SHARED_RULE_BANK)
mixer_metrics_fn = make_mixer_metrics_fn(
    tokenizer=SHARED_TOKENIZER,
    rule_bank=SHARED_RULE_BANK,
    eot_token_id=int(SHARED_TOKENIZER.eot_token_id),
)

for train_max_distance in TRAIN_MAX_DISTANCES:
    train_distances = _train_distances_for_k(train_max_distance)
    ood_distances = _ood_distances_for_k(train_max_distance)
    if not train_distances:
        raise ValueError(f"No train distances for k={train_max_distance}")
    if not ood_distances:
        raise ValueError(
            f"No OOD distances for k={train_max_distance}; ensure k < max(EVAL_DISTANCES)."
        )

    for n_layers, (n_hidden, n_heads), lr, pos_encoding, use_swiglu in itertools.product(
        TRANSFORMER_LAYERS,
        TRANSFORMER_WIDTH_HEADS,
        TRANSFORMER_LRS,
        TRANSFORMER_POS,
        TRANSFORMER_SWIGLU,
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
            train_distances,
            prediction_objective="autoregressive",
            rule_bank_path=SHARED_RULE_BANK_PATH,
            seed=new_seed(),
            drop_remainder=True,
            shuffle=True,
        )
        test_task = _make_layer_task(
            ood_distances,
            prediction_objective="autoregressive",
            rule_bank_path=SHARED_RULE_BANK_PATH,
            seed=new_seed(),
            drop_remainder=False,
            shuffle=True,
        )

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
            "train_max_distance": int(train_max_distance),
            "train_distances": train_distances,
            "eval_distances": EVAL_DISTANCES,
            "ood_distances": ood_distances,
            "n_layers": n_layers,
            "n_hidden": n_hidden,
            "n_heads": n_heads,
            "pos_encoding": pos_encoding,
            "use_swiglu": use_swiglu,
            "lr": lr,
            "n_vocab": N_VOCAB,
            "n_seq": N_SEQ_AR,
        }

        all_cases.append(
            Case(
                f"7_layer_fol_sweep_transformer_k{int(train_max_distance):02d}",
                config,
                train_task=train_task,
                test_task=test_task,
                train_args=train_args,
                info=info,
            )
        )

    for n_layers, (n_hidden, n_heads), d_state, d_conv, scan_chunk_len, lr in itertools.product(
        MAMBA2_BONSAI_LAYERS,
        MAMBA2_BONSAI_WIDTH_HEADS,
        MAMBA2_BONSAI_D_STATE,
        MAMBA2_BONSAI_D_CONV,
        MAMBA2_BONSAI_SCAN_CHUNK_LEN,
        MAMBA2_BONSAI_LRS,
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
            train_distances,
            prediction_objective="autoregressive",
            rule_bank_path=SHARED_RULE_BANK_PATH,
            seed=new_seed(),
            drop_remainder=True,
            shuffle=True,
        )
        test_task = _make_layer_task(
            ood_distances,
            prediction_objective="autoregressive",
            rule_bank_path=SHARED_RULE_BANK_PATH,
            seed=new_seed(),
            drop_remainder=False,
            shuffle=True,
        )

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
            "train_max_distance": int(train_max_distance),
            "train_distances": train_distances,
            "eval_distances": EVAL_DISTANCES,
            "ood_distances": ood_distances,
            "n_layers": n_layers,
            "n_hidden": n_hidden,
            "n_heads": n_heads,
            "d_state": d_state,
            "d_conv": d_conv,
            "scan_chunk_len": scan_chunk_len,
            "lr": lr,
            "n_vocab": N_VOCAB,
            "n_seq": N_SEQ_AR,
        }

        all_cases.append(
            Case(
                f"7_layer_fol_sweep_mamba2_k{int(train_max_distance):02d}",
                config,
                train_task=train_task,
                test_task=test_task,
                train_args=train_args,
                info=info,
            )
        )

    for n_layers, n_hidden, n_channels, lr in itertools.product(
        MIXER_LAYERS,
        MIXER_HIDDEN,
        MIXER_CHANNELS,
        MIXER_LRS,
    ):
        config = CompletionMixerConfig(
            n_vocab=N_VOCAB,
            n_seq=N_SEQ_COMPLETION,
            n_layers=n_layers,
            n_hidden=n_hidden,
            n_channels=n_channels,
            n_out_seq=MAX_COMPLETION_LEN,
            n_out_vocab=N_VOCAB,
            act_fn="gelu",
            layer_norm=True,
            use_bias=True,
        )

        train_base = _make_layer_task(
            train_distances,
            prediction_objective="all_at_once",
            rule_bank_path=SHARED_RULE_BANK_PATH,
            seed=new_seed(),
            drop_remainder=True,
            shuffle=True,
        )
        test_base = _make_layer_task(
            ood_distances,
            prediction_objective="all_at_once",
            rule_bank_path=SHARED_RULE_BANK_PATH,
            seed=new_seed(),
            drop_remainder=False,
            shuffle=True,
        )

        train_task = MixerBatchAdapter(
            train_base,
            prompt_builder=lambda xs: build_prompt_only_inputs(
                xs,
                n_seq=N_SEQ_COMPLETION,
                sep_token_id=int(SHARED_TOKENIZER.sep_token_id),
                pad_token_id=0,
            ),
            target_builder=lambda ys: pad_completion_targets(
                ys,
                max_out_len=MAX_COMPLETION_LEN,
                eot_token_id=int(SHARED_TOKENIZER.eot_token_id),
            ),
        )
        test_task = MixerBatchAdapter(
            test_base,
            prompt_builder=lambda xs: build_prompt_only_inputs(
                xs,
                n_seq=N_SEQ_COMPLETION,
                sep_token_id=int(SHARED_TOKENIZER.sep_token_id),
                pad_token_id=0,
            ),
            target_builder=lambda ys: pad_completion_targets(
                ys,
                max_out_len=MAX_COMPLETION_LEN,
                eot_token_id=int(SHARED_TOKENIZER.eot_token_id),
            ),
        )

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
            "train_max_distance": int(train_max_distance),
            "train_distances": train_distances,
            "eval_distances": EVAL_DISTANCES,
            "ood_distances": ood_distances,
            "n_layers": n_layers,
            "n_hidden": n_hidden,
            "n_channels": n_channels,
            "lr": lr,
            "n_vocab": N_VOCAB,
            "n_seq": N_SEQ_COMPLETION,
            "max_out_len": MAX_COMPLETION_LEN,
        }

        all_cases.append(
            Case(
                f"7_layer_fol_sweep_mixer_completion_k{int(train_max_distance):02d}",
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
print("CASE NAMES", [case.name for case in all_cases])


# <codecell>
rows = []
for case in tqdm(all_cases):
    print("RUNNING", case.name, case.info)
    case.run()

    family = case.info["model_family"]
    final_test = case.hist["test"][-1] if case.hist and case.hist.get("test") else {}

    distance_metrics = _evaluate_by_distance(
        case.optimizer,
        family=family,
        tokenizer=SHARED_TOKENIZER,
        rule_bank=SHARED_RULE_BANK,
        rule_bank_path=SHARED_RULE_BANK_PATH,
        n_seq_ar=N_SEQ_AR,
        n_seq_completion=N_SEQ_COMPLETION,
        max_completion_len=MAX_COMPLETION_LEN,
        n_iters=EVAL_ITERS_PER_DISTANCE,
    )

    ood_distances = case.info.get("ood_distances", [])
    ood_rollout_success_avg = _avg_ood_metric(
        distance_metrics,
        "rollout_success_rate",
        ood_distances,
    )

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
        "metrics_by_distance": distance_metrics,
        "selection_metric_name": "ood_rollout_success_avg",
        "selection_metric_value": ood_rollout_success_avg,
        "ood_rollout_success_avg": ood_rollout_success_avg,
        "ood_final_token_acc_avg": _avg_ood_metric(
            distance_metrics,
            "final_token_acc",
            ood_distances,
        ),
        "ood_valid_rule_rate_avg": _avg_ood_metric(
            distance_metrics,
            "valid_rule_rate",
            ood_distances,
        ),
        "ood_correct_rule_rate_avg": _avg_ood_metric(
            distance_metrics,
            "correct_rule_rate",
            ood_distances,
        ),
        "ood_rollout_unknown_rule_error_rate_avg": _avg_ood_metric(
            distance_metrics,
            "rollout_unknown_rule_error_rate",
            ood_distances,
        ),
        "ood_rollout_decode_error_rate_avg": _avg_ood_metric(
            distance_metrics,
            "rollout_decode_error_rate",
            ood_distances,
        ),
        "rule_bank_path": str(SHARED_RULE_BANK_PATH.resolve()),
        "rule_bank_sha256": SHARED_RULE_BANK_SHA256,
        "rule_bank_config": {
            "seed": RULE_BANK_SEED,
            "n_layers": N_LAYERS,
            "predicates_per_layer": PREDICATES_PER_LAYER,
            "rules_per_transition": RULES_PER_TRANSITION,
            "arity_max": ARITY_MAX,
            "vars_per_rule_max": VARS_PER_RULE_MAX,
            "constants": list(CONSTANTS),
            "k_in_max": K_IN_MAX,
            "k_out_max": K_OUT_MAX,
            "initial_ant_max": INITIAL_ANT_MAX,
            "sample_max_attempts": SAMPLE_MAX_ATTEMPTS,
            "max_unify_solutions": MAX_UNIFY_SOLUTIONS,
        },
        "rule_bank": SHARED_RULE_BANK_PAYLOAD,
        "dims": DIMS,
        "hist": case.hist,
    }
    rows.append(row)

    case.optimizer = None
    case.hist = None
    case.train_task = None
    case.test_task = None
    case.train_args["eval_fns"] = None

pd.DataFrame(rows).to_pickle(save_dir / f"res.{RUN_ID}.pkl")
print("done!")
