"""Train a small local Transformer and inspect teacher-forced vs AR rule validity."""

# <codecell>
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

from common import new_seed
from experiment.utils.metrics_utils import final_token_accuracy
from model.eval_adapters import make_model_callable
from model.transformer import TransformerConfig
from task.layer import (
    AutoregressiveLogitsAdapter,
    LayerTask,
    evaluate_layer_rollouts,
    evaluate_rule_matches,
    match_rule_completion,
    sample_rollout_examples,
)
from task.layer_gen.util import tokenize_layer
from task.layer_gen.util.rule_bank import build_random_rule_bank, save_rule_bank
from task.prop_gen.util.elem import Atom
from train import ce_mask, train


RUN_ID = int(new_seed())

EVAL_DISTANCES = list(range(1, 11))

RULE_BANK_SEED = 2026
N_LAYERS = 24
PROPS_PER_LAYER = 16
RULES_PER_TRANSITION = 32
K_IN_MAX = 3
K_OUT_MAX = 5
INITIAL_ANT_MAX = 3

REMOTE_REFERENCE_TF_VALID_RULE_RATE = 0.60
REMOTE_REFERENCE_HIGH_AR_ROLLOUT_SUCCESS = 0.90

_ATOM_RE = re.compile(r"p(\d+)_(\d+)$")


@dataclass
class InspectConfig:
    batch_size: int = 32
    train_iters: int = 5000
    test_every: int = 250
    test_iters: int = 2
    train_max_distance: int = 5
    inspect_distance: int = 10
    max_inspect_batches: int = 20
    min_target_examples: int = 5
    print_examples: int = 8
    n_layers: int = 2
    n_hidden: int = 64
    n_heads: int = 4
    lr: float = 1e-3
    rollout_examples: int = 64


CFG = InspectConfig()


def _build_rule_bank(save_dir: Path):
    rng = np.random.default_rng(RULE_BANK_SEED)
    rule_bank = build_random_rule_bank(
        n_layers=N_LAYERS,
        props_per_layer=PROPS_PER_LAYER,
        rules_per_transition=RULES_PER_TRANSITION,
        k_in_max=K_IN_MAX,
        k_out_max=K_OUT_MAX,
        rng=rng,
    )
    rule_bank_path = save_dir / f"rule_bank.{RUN_ID}.json"
    save_rule_bank(rule_bank_path, rule_bank)
    tokenizer = tokenize_layer.build_tokenizer_from_rule_bank(rule_bank)
    return rule_bank, tokenizer, rule_bank_path


def _compute_dims(rule_bank, tokenizer) -> dict[str, int]:
    max_prompt_len = 2 * int(rule_bank.props_per_layer) + 2
    max_completion_len = 1
    for rules in rule_bank.transitions.values():
        for rule in rules:
            max_completion_len = max(
                max_completion_len,
                len(tokenizer.encode_completion(rule.statement_text)),
            )

    n_seq_ar = int(max_prompt_len + max_completion_len - 1)
    n_vocab = max(512, int(tokenizer.vocab_size))
    return {
        "n_vocab": int(n_vocab),
        "max_completion_len": int(max_completion_len),
        "n_seq_ar": int(n_seq_ar),
    }


def _train_distances_for_k(k: int) -> list[int]:
    return list(range(1, int(k) + 1))


def _ood_distances_for_k(k: int) -> list[int]:
    return [int(d) for d in EVAL_DISTANCES if int(d) > int(k)]


def _make_layer_task(
    distance_range,
    *,
    batch_size: int,
    prediction_objective: str,
    rule_bank_path: Path,
    seed: int,
    drop_remainder: bool,
    shuffle: bool,
):
    return LayerTask(
        distance_range=distance_range,
        batch_size=int(batch_size),
        mode="online",
        shuffle=bool(shuffle),
        seed=int(seed),
        worker_count=0,
        drop_remainder=bool(drop_remainder),
        prediction_objective=prediction_objective,
        rule_bank_path=rule_bank_path,
        initial_ant_max=INITIAL_ANT_MAX,
    )


def infer_src_layer_from_prompt_tokens(
    row_tokens,
    *,
    tokenizer,
    pad_token_id: int = 0,
) -> int:
    row = np.asarray(row_tokens, dtype=np.int32)
    if row.ndim != 1:
        raise ValueError(f"Expected 1D row tokens, got {row.shape}")

    nonpad = row[row != int(pad_token_id)]
    sep_hits = np.where(nonpad == int(tokenizer.sep_token_id))[0]
    if sep_hits.size == 0:
        raise ValueError("Missing SEP token in prompt row.")

    prompt = nonpad[: int(sep_hits[0]) + 1].tolist()
    sequent = tokenizer.decode_prompt([int(tok) for tok in prompt])
    if len(sequent.ants) == 0:
        raise ValueError("Prompt antecedents are empty; cannot infer src layer.")

    layers = []
    for ant in sequent.ants:
        if not isinstance(ant, Atom):
            raise ValueError(f"Expected atom antecedents, got {type(ant).__name__}")
        match = _ATOM_RE.fullmatch(ant.name)
        if match is None:
            raise ValueError(f"Unsupported layered atom name: {ant.name}")
        layers.append(int(match.group(1)))
    return int(min(layers))


def _extract_prompt_tokens(row_tokens, *, tokenizer, pad_token_id: int = 0) -> np.ndarray:
    row = np.asarray(row_tokens, dtype=np.int32)
    nonpad = row[row != int(pad_token_id)]
    sep_hits = np.where(nonpad == int(tokenizer.sep_token_id))[0]
    if sep_hits.size == 0:
        raise ValueError("Missing SEP token while extracting prompt.")
    return nonpad[: int(sep_hits[0]) + 1].astype(np.int32)


def _safe_decode_prompt_text(prompt_tokens: np.ndarray, *, tokenizer) -> str | None:
    try:
        return str(tokenizer.decode_prompt([int(tok) for tok in prompt_tokens.tolist()]))
    except (ValueError, TypeError):
        return None


def _safe_decode_completion_text(tokens: np.ndarray, *, tokenizer) -> str | None:
    try:
        return tokenizer.decode_completion_text([int(tok) for tok in tokens.tolist()])
    except (ValueError, TypeError):
        return None


def extract_ar_rule_match_inputs(
    *,
    preds,
    labels,
    xs,
    tokenizer,
) -> tuple[list[int], list[np.ndarray], list[str | None]]:
    preds = np.asarray(preds, dtype=np.int32)
    labels = np.asarray(labels, dtype=np.int32)
    xs = np.asarray(xs, dtype=np.int32)

    if preds.shape != labels.shape:
        raise ValueError(f"preds and labels must have same shape, got {preds.shape} and {labels.shape}")
    if xs.ndim != 2 or labels.ndim != 2:
        raise ValueError("Expected 2D xs/labels arrays.")

    src_layers: list[int] = []
    pred_completions: list[np.ndarray] = []
    expected_statements: list[str | None] = []

    for idx in range(labels.shape[0]):
        mask = labels[idx] != 0
        if not np.any(mask):
            continue

        src_layer = infer_src_layer_from_prompt_tokens(xs[idx], tokenizer=tokenizer)
        pred_completion = preds[idx][mask].astype(np.int32)
        gold_completion = labels[idx][mask].astype(np.int32)

        expected_statement = _safe_decode_completion_text(gold_completion, tokenizer=tokenizer)

        src_layers.append(int(src_layer))
        pred_completions.append(pred_completion)
        expected_statements.append(expected_statement)

    return src_layers, pred_completions, expected_statements


def summarize_rule_match_metrics(metrics) -> dict[str, float | int]:
    n_valid = int(sum(int(result.is_valid_rule) for result in metrics.results))
    correct_given_valid_rate = float(metrics.n_correct) / float(n_valid) if n_valid > 0 else 0.0
    valid_rule_rate = float(n_valid) / float(metrics.n_examples) if metrics.n_examples > 0 else 0.0
    return {
        "n_rule_examples": int(metrics.n_examples),
        "n_valid_rule": int(n_valid),
        "n_invalid_rule": int(metrics.n_examples - n_valid),
        "n_correct_rule": int(metrics.n_correct),
        "n_decode_error": int(metrics.n_decode_error),
        "n_unknown_rule_error": int(metrics.n_unknown_rule_error),
        "n_wrong_rule_error": int(metrics.n_wrong_rule_error),
        "valid_rule_rate": valid_rule_rate,
        "invalid_rule_rate": 1.0 - valid_rule_rate,
        "correct_rule_rate": float(metrics.accuracy),
        "correct_given_valid_rate": correct_given_valid_rate,
        "decode_error_rate": float(metrics.decode_error_rate),
        "unknown_rule_error_rate": float(metrics.unknown_rule_error_rate),
        "wrong_rule_error_rate": float(metrics.wrong_rule_error_rate),
    }


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


def make_print_fn(metric_key: str):
    def _print(step, hist):
        train_metrics = hist["train"][-1]
        test_metrics = hist["test"][-1]
        print(
            "ITER {}: train_loss={:.4f} train_{}={:.4f} test_loss={:.4f} test_{}={:.4f}".format(
                step,
                float(train_metrics["loss"]),
                metric_key,
                float(train_metrics[metric_key]),
                float(test_metrics["loss"]),
                metric_key,
                float(test_metrics[metric_key]),
            )
        )

    return _print


def _match_error_label(match_result) -> str:
    if bool(match_result.decode_error):
        return "decode_error"
    if bool(match_result.unknown_rule_error):
        return "unknown_rule_error"
    if bool(match_result.wrong_rule_error):
        return "wrong_rule_error"
    if bool(match_result.is_valid_rule):
        return "none"
    return "unknown"


def _match_to_dict(match_result) -> dict[str, object]:
    return {
        "decoded_statement": match_result.decoded_statement,
        "expected_statement_text": match_result.expected_statement_text,
        "is_valid_rule": bool(match_result.is_valid_rule),
        "is_correct": bool(match_result.is_correct),
        "decode_error": bool(match_result.decode_error),
        "unknown_rule_error": bool(match_result.unknown_rule_error),
        "wrong_rule_error": bool(match_result.wrong_rule_error),
        "error_type": _match_error_label(match_result),
        "matched_rule_statement": (
            None if match_result.matched_rule is None else str(match_result.matched_rule.statement_text)
        ),
    }


def _transition_flags(tf_is_valid: bool, ar_is_valid: bool) -> dict[str, bool]:
    return {
        "tf_invalid_ar_valid": (not tf_is_valid) and ar_is_valid,
        "tf_valid_ar_invalid": tf_is_valid and (not ar_is_valid),
        "both_valid": tf_is_valid and ar_is_valid,
        "both_invalid": (not tf_is_valid) and (not ar_is_valid),
    }


def inspect_examples(
    *,
    optimizer,
    inspect_task,
    tokenizer,
    rule_bank,
    n_seq_ar: int,
    max_completion_len: int,
    max_batches: int,
    min_target_examples: int,
) -> tuple[list[dict[str, object]], int]:
    model_fn = make_model_callable(optimizer, to_numpy=False)
    adapter = AutoregressiveLogitsAdapter(
        n_seq=int(n_seq_ar),
        max_completion_len=int(max_completion_len),
        pad_token_id=0,
        jit_step=True,
    )
    rng = np.random.default_rng(int(RUN_ID) + 73_001)

    records: list[dict[str, object]] = []
    target_count = 0
    batches_used = 0

    for batch_idx in range(int(max_batches)):
        batches_used += 1
        xs, labels = next(inspect_task)
        logits = optimizer.model(xs)
        preds = np.asarray(jnp.argmax(logits, axis=-1))

        xs_np = np.asarray(xs, dtype=np.int32)
        labels_np = np.asarray(labels, dtype=np.int32)

        for example_idx in range(xs_np.shape[0]):
            mask = labels_np[example_idx] != 0
            if not np.any(mask):
                continue

            try:
                src_layer = int(infer_src_layer_from_prompt_tokens(xs_np[example_idx], tokenizer=tokenizer))
                prompt_tokens = _extract_prompt_tokens(xs_np[example_idx], tokenizer=tokenizer)
            except ValueError:
                continue

            gold_completion = labels_np[example_idx][mask].astype(np.int32)
            tf_completion = preds[example_idx][mask].astype(np.int32)
            ar_completion = adapter.predict_completion(
                model=model_fn,
                prompt_tokens=prompt_tokens,
                tokenizer=tokenizer,
                temperature=0.0,
                rng=rng,
            ).astype(np.int32)

            gold_statement = _safe_decode_completion_text(gold_completion, tokenizer=tokenizer)
            tf_match = match_rule_completion(
                rule_bank=rule_bank,
                src_layer=src_layer,
                completion_tokens=tf_completion,
                expected_statement_text=gold_statement,
                tokenizer=tokenizer,
            )
            ar_match = match_rule_completion(
                rule_bank=rule_bank,
                src_layer=src_layer,
                completion_tokens=ar_completion,
                expected_statement_text=gold_statement,
                tokenizer=tokenizer,
            )

            tf_result = _match_to_dict(tf_match)
            ar_result = _match_to_dict(ar_match)
            transitions = _transition_flags(
                tf_is_valid=bool(tf_result["is_valid_rule"]),
                ar_is_valid=bool(ar_result["is_valid_rule"]),
            )
            if transitions["tf_invalid_ar_valid"]:
                target_count += 1

            records.append(
                {
                    "run_id": int(RUN_ID),
                    "batch_index": int(batch_idx),
                    "example_index": int(example_idx),
                    "src_layer": int(src_layer),
                    "prompt_tokens": [int(tok) for tok in prompt_tokens.tolist()],
                    "prompt_text": _safe_decode_prompt_text(prompt_tokens, tokenizer=tokenizer),
                    "gold_completion_tokens": [int(tok) for tok in gold_completion.tolist()],
                    "gold_statement_text": gold_statement,
                    "teacher_forced_completion_tokens": [int(tok) for tok in tf_completion.tolist()],
                    "autoregressive_completion_tokens": [int(tok) for tok in ar_completion.tolist()],
                    "teacher_forced": tf_result,
                    "autoregressive": ar_result,
                    "transition": transitions,
                }
            )

        if target_count >= int(min_target_examples):
            break

    return records, batches_used


def aggregate_example_stats(records: list[dict[str, object]]) -> dict[str, object]:
    n_examples = int(len(records))
    if n_examples == 0:
        return {
            "n_examples": 0,
            "teacher_forced_valid_rule_rate": 0.0,
            "autoregressive_valid_rule_rate": 0.0,
            "teacher_forced_correct_rule_rate": 0.0,
            "autoregressive_correct_rule_rate": 0.0,
            "transition_counts": {},
        }

    tf_valid = np.array(
        [bool(rec["teacher_forced"]["is_valid_rule"]) for rec in records],
        dtype=bool,
    )
    ar_valid = np.array(
        [bool(rec["autoregressive"]["is_valid_rule"]) for rec in records],
        dtype=bool,
    )
    tf_correct = np.array(
        [bool(rec["teacher_forced"]["is_correct"]) for rec in records],
        dtype=bool,
    )
    ar_correct = np.array(
        [bool(rec["autoregressive"]["is_correct"]) for rec in records],
        dtype=bool,
    )

    transition_keys = ["tf_invalid_ar_valid", "tf_valid_ar_invalid", "both_valid", "both_invalid"]
    transition_counts = {
        key: int(sum(int(bool(rec["transition"].get(key, False))) for rec in records))
        for key in transition_keys
    }

    return {
        "n_examples": n_examples,
        "teacher_forced_valid_rule_rate": float(tf_valid.mean()),
        "autoregressive_valid_rule_rate": float(ar_valid.mean()),
        "teacher_forced_correct_rule_rate": float(tf_correct.mean()),
        "autoregressive_correct_rule_rate": float(ar_correct.mean()),
        "transition_counts": transition_counts,
    }


def evaluate_rollout_summary(
    *,
    optimizer,
    tokenizer,
    rule_bank,
    inspect_distance: int,
    n_seq_ar: int,
    max_completion_len: int,
    rollout_examples: int,
) -> dict[str, object]:
    model_fn = make_model_callable(optimizer, to_numpy=False)
    adapter = AutoregressiveLogitsAdapter(
        n_seq=int(n_seq_ar),
        max_completion_len=int(max_completion_len),
        pad_token_id=0,
        jit_step=True,
    )
    examples = sample_rollout_examples(
        rule_bank=rule_bank,
        distance=int(inspect_distance),
        n_examples=int(rollout_examples),
        initial_ant_max=int(INITIAL_ANT_MAX),
        max_steps=int(inspect_distance),
        rng=np.random.default_rng(int(RUN_ID) + 8101),
    )
    metrics = evaluate_layer_rollouts(
        rule_bank=rule_bank,
        examples=examples,
        model=model_fn,
        adapter=adapter,
        tokenizer=tokenizer,
        temperature=0.0,
        rng=np.random.default_rng(int(RUN_ID) + 9101),
    )
    return {
        "rollout_n_examples": int(metrics.n_examples),
        "rollout_success_rate": float(metrics.success_rate),
        "rollout_decode_error_rate": float(metrics.decode_error_rate),
        "rollout_unknown_rule_error_rate": float(metrics.unknown_rule_error_rate),
        "rollout_inapplicable_rule_error_rate": float(metrics.inapplicable_rule_error_rate),
        "rollout_goal_not_reached_rate": float(metrics.goal_not_reached_rate),
        "rollout_avg_steps": float(metrics.avg_steps),
    }


def build_consistency_report(
    example_stats: dict[str, object],
    rollout_summary: dict[str, object],
) -> dict[str, object]:
    tf_valid = float(example_stats.get("teacher_forced_valid_rule_rate", 0.0))
    ar_valid = float(example_stats.get("autoregressive_valid_rule_rate", 0.0))
    rollout_success = float(rollout_summary.get("rollout_success_rate", 0.0))

    return {
        "teacher_forced_vs_remote_0p60_delta": float(tf_valid - REMOTE_REFERENCE_TF_VALID_RULE_RATE),
        "rollout_success_vs_remote_0p90_delta": float(
            rollout_success - REMOTE_REFERENCE_HIGH_AR_ROLLOUT_SUCCESS
        ),
        "autoregressive_valid_minus_teacher_forced_valid": float(ar_valid - tf_valid),
        "rollout_success_minus_teacher_forced_valid": float(rollout_success - tf_valid),
        "teacher_forced_close_to_remote_0p60": bool(abs(tf_valid - 0.60) <= 0.20),
        "autoregressive_valid_higher_than_teacher_forced": bool(ar_valid > tf_valid),
        "rollout_success_higher_than_teacher_forced": bool(rollout_success > tf_valid),
    }


def _json_dump(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _jsonl_dump(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _print_example(record: dict[str, object]) -> None:
    tf = record["teacher_forced"]
    ar = record["autoregressive"]
    print("=" * 80)
    print(f"batch={record['batch_index']} example={record['example_index']} src_layer={record['src_layer']}")
    print(f"prompt: {record['prompt_text']}")
    print(f"gold:   {record['gold_statement_text']}")
    print(
        f"TF: valid={tf['is_valid_rule']} correct={tf['is_correct']} "
        f"error={tf['error_type']} decoded={tf['decoded_statement']}"
    )
    print(
        f"AR: valid={ar['is_valid_rule']} correct={ar['is_correct']} "
        f"error={ar['error_type']} decoded={ar['decoded_statement']}"
    )


if int(CFG.n_hidden) % int(CFG.n_heads) != 0:
    raise ValueError(f"n_hidden ({CFG.n_hidden}) must be divisible by n_heads ({CFG.n_heads})")

save_dir = Path(__file__).resolve().parent / "set"
save_dir.mkdir(parents=True, exist_ok=True)

rule_bank, tokenizer, rule_bank_path = _build_rule_bank(save_dir)
dims = _compute_dims(rule_bank, tokenizer)
n_vocab = int(dims["n_vocab"])
n_seq_ar = int(dims["n_seq_ar"])
max_completion_len = int(dims["max_completion_len"])

train_distances = _train_distances_for_k(int(CFG.train_max_distance))
ood_distances = _ood_distances_for_k(int(CFG.train_max_distance))

print("RUN ID", RUN_ID)
print("RULE BANK PATH", rule_bank_path)
print("DIMS", dims)
print("TRAIN DISTANCES", train_distances)
print("OOD DISTANCES", ood_distances)

config = TransformerConfig(
    n_vocab=n_vocab,
    n_seq=n_seq_ar,
    n_layers=int(CFG.n_layers),
    n_hidden=int(CFG.n_hidden),
    n_heads=int(CFG.n_heads),
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
    train_distances,
    batch_size=int(CFG.batch_size),
    prediction_objective="autoregressive",
    rule_bank_path=rule_bank_path,
    seed=new_seed(),
    drop_remainder=True,
    shuffle=True,
)
test_task = _make_layer_task(
    ood_distances,
    batch_size=int(CFG.batch_size),
    prediction_objective="autoregressive",
    rule_bank_path=rule_bank_path,
    seed=new_seed(),
    drop_remainder=False,
    shuffle=True,
)
metrics_fn = make_ar_metrics_fn(tokenizer=tokenizer, rule_bank=rule_bank)

optimizer, hist = train(
    config,
    train_iter=train_task,
    test_iter=test_task,
    loss="ce_mask",
    eval_fns=[metrics_fn],
    print_fn=make_print_fn("final_token_acc"),
    train_iters=int(CFG.train_iters),
    test_iters=int(CFG.test_iters),
    test_every=int(CFG.test_every),
    lr=float(CFG.lr),
)

final_train = hist["train"][-1] if hist.get("train") else {}
final_test = hist["test"][-1] if hist.get("test") else {}

inspect_task = _make_layer_task(
    (int(CFG.inspect_distance), int(CFG.inspect_distance)),
    batch_size=int(CFG.batch_size),
    prediction_objective="autoregressive",
    rule_bank_path=rule_bank_path,
    seed=new_seed(),
    drop_remainder=False,
    shuffle=True,
)
records, batches_used = inspect_examples(
    optimizer=optimizer,
    inspect_task=inspect_task,
    tokenizer=tokenizer,
    rule_bank=rule_bank,
    n_seq_ar=n_seq_ar,
    max_completion_len=max_completion_len,
    max_batches=int(CFG.max_inspect_batches),
    min_target_examples=int(CFG.min_target_examples),
)
example_stats = aggregate_example_stats(records)

rollout_summary = evaluate_rollout_summary(
    optimizer=optimizer,
    tokenizer=tokenizer,
    rule_bank=rule_bank,
    inspect_distance=int(CFG.inspect_distance),
    n_seq_ar=n_seq_ar,
    max_completion_len=max_completion_len,
    rollout_examples=int(CFG.rollout_examples),
)
consistency = build_consistency_report(example_stats, rollout_summary)

jsonl_path = save_dir / f"inspect_rules.{RUN_ID}.jsonl"
summary_path = save_dir / f"inspect_summary.{RUN_ID}.json"

_jsonl_dump(jsonl_path, records)
summary_payload = {
    "run_id": int(RUN_ID),
    "config": {
        "batch_size": int(CFG.batch_size),
        "train_iters": int(CFG.train_iters),
        "test_every": int(CFG.test_every),
        "test_iters": int(CFG.test_iters),
        "train_max_distance": int(CFG.train_max_distance),
        "inspect_distance": int(CFG.inspect_distance),
        "n_layers": int(CFG.n_layers),
        "n_hidden": int(CFG.n_hidden),
        "n_heads": int(CFG.n_heads),
        "lr": float(CFG.lr),
        "rollout_examples": int(CFG.rollout_examples),
        "max_inspect_batches": int(CFG.max_inspect_batches),
        "min_target_examples": int(CFG.min_target_examples),
    },
    "rule_bank_path": str(rule_bank_path.resolve()),
    "dims": dims,
    "train_distances": train_distances,
    "ood_distances": ood_distances,
    "batches_used_for_inspection": int(batches_used),
    "train_final_metrics": final_train,
    "test_final_metrics": final_test,
    "example_stats": example_stats,
    "rollout_summary": rollout_summary,
    "consistency_report": consistency,
    "records_jsonl_path": str(jsonl_path.resolve()),
}
_json_dump(summary_path, summary_payload)

print("\nSummary")
print("  records:", len(records))
print("  batches_used:", batches_used)
print(
    "  tf_valid_rule_rate={:.4f} ar_valid_rule_rate={:.4f} rollout_success_rate={:.4f}".format(
        float(example_stats["teacher_forced_valid_rule_rate"]),
        float(example_stats["autoregressive_valid_rule_rate"]),
        float(rollout_summary["rollout_success_rate"]),
    )
)
print("  transition_counts:", example_stats["transition_counts"])
print("  consistency_report:", consistency)

target_examples = [rec for rec in records if bool(rec["transition"].get("tf_invalid_ar_valid", False))]
print(f"\nExamples where TF invalid but AR valid: {len(target_examples)}")
for record in target_examples[: int(CFG.print_examples)]:
    _print_example(record)

if not target_examples:
    print("No tf_invalid_ar_valid examples found in inspected records.")

print("\nSaved:")
print(" ", jsonl_path)
print(" ", summary_path)
