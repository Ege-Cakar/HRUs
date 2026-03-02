"""Architecture sweep for the depth3_icl_transfer disjoint-rule split."""

# <codecell>
from __future__ import annotations

from collections import Counter
import hashlib
import itertools
import json
import re
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
    FOLLayerTask,
    _augment_prompt_with_demos,
    evaluate_layer_rollouts,
    evaluate_rule_matches,
    match_rule_completion_fol,
    sample_rollout_examples,
)
from task.layer_gen.util import tokenize_layer_fol
from task.layer_gen.util.fol_rule_bank import (
    FOLDepth3ICLSplitBundle,
    build_depth3_icl_split_bundle,
    save_fol_depth3_icl_split_bundle,
)
from train import Case, ce_mask

from utils import (
    extract_ar_free_run_eval_inputs,
    extract_ar_rule_match_inputs,
    predicted_rule_reaches_goal,
    summarize_first_transition_counts,
    summarize_rule_match_metrics,
)
from experiment.utils.metrics_utils import final_token_accuracy


_LAYERED_PRED_RE = re.compile(r"r(\d+)_(\d+)$")

RUN_ID = new_seed()
print("RUN ID", RUN_ID)

EVAL_ROLES = ["train", "eval"]

TRAIN_MAX_N_DEMOS = 8
EVAL_MAX_N_DEMOS_SWEEP = [0, 2, 4, 8, 12, 16, 24, 32]
SELECTION_EVAL_MAX_N_DEMOS = 8

BATCH_SIZE = 32
TRAIN_ITERS_SWEEP = [400, 1600, 6400, 25600, 102400]
TEST_EVERY = 1000
TEST_ITERS = 3
EVAL_ITERS_PER_ROLE = 3
ROLLOUT_EXAMPLES_PER_ROLE = 64

RUN_SPLIT = 120

SPLIT_SEED = 2032
# PREDICATES_PER_LAYER = 64
# RULES_01_TRAIN = 256
# RULES_01_EVAL = 256
# RULES_12_SHARED = 256
PREDICATES_PER_LAYER = 1024
RULES_01_TRAIN = 1024
RULES_01_EVAL = 1024
RULES_12_SHARED = 1024
ARITY_MAX = 3
VARS_PER_RULE_MAX = 6
K_IN_MAX = 3
K_OUT_MAX = 5
INITIAL_ANT_MAX = 5
CONSTANTS = [f"p{i}" for i in range(256)]
SAMPLE_MAX_ATTEMPTS = 4096
MAX_UNIFY_SOLUTIONS = 128

TRAIN_FIXED_LENGTH_MODE = "next_pow2"
EVAL_FIXED_LENGTH_MODE = "next_pow2"

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

### START TEST CONFIGS
# BATCH_SIZE = 8
# TRAIN_ITERS_SWEEP = [20]
# TEST_EVERY = 10
# TEST_ITERS = 1
# EVAL_ITERS_PER_ROLE = 1
# ROLLOUT_EXAMPLES_PER_ROLE = 4
# RUN_SPLIT = 1
# TRAIN_MAX_N_DEMOS = 4
# EVAL_MAX_N_DEMOS_SWEEP = [0, 4, 8]
# SELECTION_EVAL_MAX_N_DEMOS = 4
# PREDICATES_PER_LAYER = 10
# RULES_01_TRAIN = 18
# RULES_01_EVAL = 18
# RULES_12_SHARED = 18
# TRANSFORMER_LAYERS = [2]
# TRANSFORMER_WIDTH_HEADS = [(64, 4)]
# TRANSFORMER_LRS = [3e-4]
# MAMBA2_BONSAI_LAYERS = [2]
# MAMBA2_BONSAI_WIDTH_HEADS = [(64, 4)]
# MAMBA2_BONSAI_D_STATE = [8]
# MAMBA2_BONSAI_D_CONV = [4]
# MAMBA2_BONSAI_SCAN_CHUNK_LEN = [16]
# MAMBA2_BONSAI_LRS = [3e-4]
### END TEST CONFIGS


def _layer_from_predicate(predicate: str) -> int:
    match = _LAYERED_PRED_RE.fullmatch(str(predicate))
    if match is None:
        raise ValueError(f"Unsupported layered predicate name: {predicate}")
    return int(match.group(1))


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


def _build_tokenizer_for_split_bundle(
    bundle: FOLDepth3ICLSplitBundle,
) -> tokenize_layer_fol.FOLLayerTokenizer:
    max_vars = max(
        int(bundle.train_bank.vars_per_rule_max),
        int(bundle.eval_bank.vars_per_rule_max),
    )
    identifiers: set[str] = set(bundle.train_bank.constants) | set(bundle.eval_bank.constants)
    identifiers.update(bundle.train_bank.predicate_arities)
    identifiers.update(bundle.eval_bank.predicate_arities)
    identifiers.update(f"x{idx}" for idx in range(1, int(max_vars) + 1))
    predicate_identifiers = set(bundle.train_bank.predicate_arities)
    predicate_identifiers.update(bundle.eval_bank.predicate_arities)
    return tokenize_layer_fol.build_tokenizer_from_identifiers(
        sorted(identifiers),
        predicate_identifiers=sorted(predicate_identifiers),
    )


def _iter_bundle_rules(bundle: FOLDepth3ICLSplitBundle):
    seen: set[tuple[str, str, str]] = set()
    for bank_name, bank in (("train", bundle.train_bank), ("eval", bundle.eval_bank)):
        for src_layer, rules in bank.transitions.items():
            for rule in rules:
                key = (str(bank_name), str(src_layer), str(rule.statement_text))
                if key in seen:
                    continue
                seen.add(key)
                yield rule


def _build_shared_split_bundle(save_dir: Path):
    bundle = build_depth3_icl_split_bundle(
        predicates_per_layer=int(PREDICATES_PER_LAYER),
        rules_01_train=int(RULES_01_TRAIN),
        rules_01_eval=int(RULES_01_EVAL),
        rules_12_shared=int(RULES_12_SHARED),
        arity_max=int(ARITY_MAX),
        vars_per_rule_max=int(VARS_PER_RULE_MAX),
        k_in_max=int(K_IN_MAX),
        k_out_max=int(K_OUT_MAX),
        constants=tuple(str(c) for c in CONSTANTS),
        rng=np.random.default_rng(int(SPLIT_SEED)),
    )

    bundle_payload = bundle.to_dict()
    canonical = json.dumps(bundle_payload, sort_keys=True, separators=(",", ":"))
    bundle_sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    bundle_path = save_dir / f"depth3_icl_split_bundle.{RUN_ID}.json"
    save_fol_depth3_icl_split_bundle(bundle_path, bundle)

    tokenizer = _build_tokenizer_for_split_bundle(bundle)
    return bundle, tokenizer, bundle_payload, bundle_sha256, bundle_path


def _compute_dims(
    bundle: FOLDepth3ICLSplitBundle,
    tokenizer,
    *,
    max_n_demos_for_shapes: int,
):
    all_rules = list(_iter_bundle_rules(bundle))
    if not all_rules:
        raise ValueError("Split bundle has no rules.")

    max_rhs_atoms = max(len(rule.rhs) for rule in all_rules)
    max_prompt_facts = max(int(INITIAL_ANT_MAX), int(max_rhs_atoms))

    merged_predicate_arities = dict(bundle.train_bank.predicate_arities)
    merged_predicate_arities.update(bundle.eval_bank.predicate_arities)

    first_const = str(bundle.train_bank.constants[0])
    max_atom_len = 1
    for predicate, arity in merged_predicate_arities.items():
        atom_text = f"{str(predicate)}({','.join(first_const for _ in range(int(arity)))})"
        max_atom_len = max(int(max_atom_len), len(tokenizer.encode_completion(atom_text)) - 1)

    max_prompt_len = (
        max_prompt_facts * max_atom_len
        + max(0, max_prompt_facts - 1)
        + 1
        + max_atom_len
        + 1
    )

    if int(max_n_demos_for_shapes) > 0:
        max_demo_clause_len = max(
            len(tokenizer.encode_completion(rule.statement_text)) - 1
            for rule in all_rules
        )
        max_prompt_len += int(max_n_demos_for_shapes) * (int(max_demo_clause_len) + 1)

    max_completion_len = max(
        len(tokenizer.encode_completion(rule.statement_text))
        for rule in all_rules
    )

    n_seq_ar = int(max_prompt_len + max_completion_len - 1)
    n_vocab = max(512, int(tokenizer.vocab_size))

    return {
        "n_vocab": int(n_vocab),
        "max_prompt_len": int(max_prompt_len),
        "max_completion_len": int(max_completion_len),
        "n_seq_ar": int(n_seq_ar),
    }


def _make_layer_task(
    *,
    split_role: str,
    split_rule_bundle_path: Path,
    seed: int,
    drop_remainder: bool,
    shuffle: bool,
    max_n_demos: int,
    fixed_length_mode: str,
    fixed_length_n_seq: int,
):
    return FOLLayerTask(
        distance_range=(2, 2),
        batch_size=int(BATCH_SIZE),
        mode="online",
        task_split="depth3_icl_transfer",
        split_role=str(split_role),
        split_rule_bundle_path=split_rule_bundle_path,
        shuffle=shuffle,
        seed=seed,
        worker_count=0,
        drop_remainder=drop_remainder,
        prediction_objective="autoregressive",
        predicates_per_layer=int(PREDICATES_PER_LAYER),
        arity_max=int(ARITY_MAX),
        vars_per_rule_max=int(VARS_PER_RULE_MAX),
        constants=tuple(str(tok) for tok in CONSTANTS),
        initial_ant_max=int(INITIAL_ANT_MAX),
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


def make_ar_metrics_fn(*, tokenizer, rule_bank, model_fn, n_seq: int, max_completion_len: int):
    adapter = AutoregressiveLogitsAdapter(
        n_seq=int(n_seq),
        max_completion_len=int(max_completion_len),
        pad_token_id=0,
        jit_step=True,
    )

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

        (
            prompt_tokens,
            fr_src_layers,
            fr_goals,
            fr_goal_layers,
            fr_expected_statements,
        ) = extract_ar_free_run_eval_inputs(
            xs=np.asarray(xs),
            labels=np.asarray(labels),
            tokenizer=tokenizer,
        )
        n_examples = 0
        n_valid = 0
        n_reachable = 0
        n_decode_error = 0
        n_unknown_rule_error = 0
        n_wrong_rule_error = 0
        n_correct_rule = 0

        for prompt, src_layer, goal, goal_layer, expected_statement in zip(
            prompt_tokens,
            fr_src_layers,
            fr_goals,
            fr_goal_layers,
            fr_expected_statements,
        ):
            if int(src_layer) != 0:
                continue
            n_examples += 1

            completion = adapter.predict_completion(
                model=model_fn,
                prompt_tokens=prompt,
                tokenizer=tokenizer,
                temperature=0.0,
                rng=None,
            )
            matched = match_rule_completion_fol(
                rule_bank=rule_bank,
                src_layer=int(src_layer),
                completion_tokens=completion,
                expected_statement_text=expected_statement,
                tokenizer=tokenizer,
            )

            if matched.decode_error:
                n_decode_error += 1
                continue
            if matched.unknown_rule_error or matched.matched_rule is None:
                n_unknown_rule_error += 1
                continue

            n_valid += 1
            if matched.wrong_rule_error:
                n_wrong_rule_error += 1
            else:
                n_correct_rule += 1

            if predicted_rule_reaches_goal(
                rule_bank=rule_bank,
                matched_rule=matched.matched_rule,
                goal=goal,
                goal_layer=int(goal_layer),
                max_unify_solutions=MAX_UNIFY_SOLUTIONS,
            ):
                n_reachable += 1

        first_transition_metrics = summarize_first_transition_counts(
            n_examples=n_examples,
            n_valid=n_valid,
            n_reachable=n_reachable,
            n_decode_error=n_decode_error,
            n_unknown_rule_error=n_unknown_rule_error,
            n_wrong_rule_error=n_wrong_rule_error,
        )
        first_transition_metrics["first_transition_n_correct_rule"] = int(n_correct_rule)
        first_transition_metrics["first_transition_correct_rule_rate"] = (
            float(n_correct_rule) / float(n_examples) if n_examples > 0 else 0.0
        )

        return {
            "loss": loss_val,
            "token_acc": token_acc,
            "final_token_acc": final_acc,
            "seq_exact_acc": seq_exact_acc,
            **rule_metrics,
            **first_transition_metrics,
        }

    return _metrics


class DemoAugmentedAdapter:
    """Prepend sampled demo completions before calling a base adapter."""

    def __init__(
        self,
        *,
        base_adapter,
        rule_bank,
        tokenizer,
        max_n_demos: int,
        max_unify_solutions: int,
    ) -> None:
        self.base_adapter = base_adapter
        self.rule_bank = rule_bank
        self.tokenizer = tokenizer
        self.max_n_demos = int(max_n_demos)
        self.max_unify_solutions = int(max_unify_solutions)

    def predict_completion(
        self,
        *,
        model,
        prompt_tokens,
        tokenizer,
        temperature: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if self.max_n_demos <= 0:
            return self.base_adapter.predict_completion(
                model=model,
                prompt_tokens=prompt_tokens,
                tokenizer=tokenizer,
                temperature=temperature,
                rng=rng,
            )

        if rng is None:
            rng = np.random.default_rng()

        prompt = np.asarray(prompt_tokens, dtype=np.int32).tolist()
        try:
            sequent = self.tokenizer.decode_prompt(prompt)
            src_layer = int(min(_layer_from_predicate(atom.predicate) for atom in sequent.ants))
            prompt = _augment_prompt_with_demos(
                prompt_tokens=prompt,
                rule_bank=self.rule_bank,
                tokenizer=self.tokenizer,
                rng=rng,
                src_layer=src_layer,
                ants=tuple(sequent.ants),
                max_n_demos=self.max_n_demos,
                max_unify_solutions=self.max_unify_solutions,
            )
        except ValueError:
            pass

        return self.base_adapter.predict_completion(
            model=model,
            prompt_tokens=prompt,
            tokenizer=tokenizer,
            temperature=temperature,
            rng=rng,
        )


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
    split_rule_bundle_path: Path,
    n_seq_ar: int,
    max_completion_len: int,
    n_iters: int,
    eval_max_n_demos: int,
):
    model_fn = make_model_callable(optimizer, to_numpy=False)
    metrics_fn = make_ar_metrics_fn(
        tokenizer=tokenizer,
        rule_bank=rule_bank,
        model_fn=model_fn,
        n_seq=int(n_seq_ar),
        max_completion_len=int(max_completion_len),
    )

    base_rollout_adapter = AutoregressiveLogitsAdapter(
        n_seq=int(n_seq_ar),
        max_completion_len=int(max_completion_len),
        pad_token_id=0,
        jit_step=True,
    )
    rollout_adapter = DemoAugmentedAdapter(
        base_adapter=base_rollout_adapter,
        rule_bank=rule_bank,
        tokenizer=tokenizer,
        max_n_demos=int(eval_max_n_demos),
        max_unify_solutions=int(MAX_UNIFY_SOLUTIONS),
    )

    seq_len_counts: Counter[int] = Counter()
    n_eval_batches = 0

    eval_task = _make_layer_task(
        split_role=str(role),
        split_rule_bundle_path=split_rule_bundle_path,
        seed=new_seed(),
        drop_remainder=False,
        shuffle=True,
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

    rollout_examples = sample_rollout_examples(
        rule_bank=rule_bank,
        distance=2,
        n_examples=int(ROLLOUT_EXAMPLES_PER_ROLE),
        initial_ant_max=int(INITIAL_ANT_MAX),
        max_steps=2,
        max_unify_solutions=int(MAX_UNIFY_SOLUTIONS),
        rng=np.random.default_rng(
            int(RUN_ID)
            + 1_000 * int(eval_max_n_demos)
            + (17 if role == "train" else 31)
        ),
    )
    rollout_metrics = evaluate_layer_rollouts(
        rule_bank=rule_bank,
        examples=rollout_examples,
        model=model_fn,
        adapter=rollout_adapter,
        tokenizer=tokenizer,
        temperature=0.0,
        rng=np.random.default_rng(
            int(RUN_ID)
            + 2_000 * int(eval_max_n_demos)
            + (17 if role == "train" else 31)
        ),
    )

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

(
    SHARED_SPLIT_BUNDLE,
    SHARED_TOKENIZER,
    SHARED_SPLIT_BUNDLE_PAYLOAD,
    SHARED_SPLIT_BUNDLE_SHA256,
    SHARED_SPLIT_BUNDLE_PATH,
) = _build_shared_split_bundle(save_dir)

DIMS_TRAIN = _compute_dims(
    SHARED_SPLIT_BUNDLE,
    SHARED_TOKENIZER,
    max_n_demos_for_shapes=TRAIN_MAX_N_DEMOS,
)
DIMS_EVAL = _compute_dims(
    SHARED_SPLIT_BUNDLE,
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
    "SPLIT BUNDLE",
    {
        "path": str(SHARED_SPLIT_BUNDLE_PATH),
        "sha256": SHARED_SPLIT_BUNDLE_SHA256,
    },
)

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
        split_rule_bundle_path=SHARED_SPLIT_BUNDLE_PATH,
        seed=new_seed(),
        drop_remainder=True,
        shuffle=True,
        max_n_demos=int(TRAIN_MAX_N_DEMOS),
        fixed_length_mode=TRAIN_FIXED_LENGTH_MODE,
        fixed_length_n_seq=TRAIN_N_SEQ_AR,
    )
    test_task = _make_layer_task(
        split_role="eval",
        split_rule_bundle_path=SHARED_SPLIT_BUNDLE_PATH,
        seed=new_seed(),
        drop_remainder=False,
        shuffle=True,
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
        "lr": lr,
    }

    info = {
        "model_family": "transformer",
        "target_format": "next_token_full_sequence",
        "task_split": "depth3_icl_transfer",
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
    }

    case = Case(
        (
            f"9_disjoint_rule_split_transformer_"
            f"l{int(n_layers)}_h{int(n_hidden)}_heads{int(n_heads)}_"
            f"lr{_lr_tag(lr)}_ti{int(train_iters)}"
        ),
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
        split_rule_bundle_path=SHARED_SPLIT_BUNDLE_PATH,
        seed=new_seed(),
        drop_remainder=True,
        shuffle=True,
        max_n_demos=int(TRAIN_MAX_N_DEMOS),
        fixed_length_mode=TRAIN_FIXED_LENGTH_MODE,
        fixed_length_n_seq=TRAIN_N_SEQ_AR,
    )
    test_task = _make_layer_task(
        split_role="eval",
        split_rule_bundle_path=SHARED_SPLIT_BUNDLE_PATH,
        seed=new_seed(),
        drop_remainder=False,
        shuffle=True,
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
        "lr": lr,
    }

    info = {
        "model_family": "mamba2_bonsai",
        "target_format": "next_token_full_sequence",
        "task_split": "depth3_icl_transfer",
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
    }

    case = Case(
        (
            "9_disjoint_rule_split_mamba2_bonsai_"
            f"l{int(n_layers)}_h{int(n_hidden)}_heads{int(n_heads)}_"
            f"ds{int(d_state)}_lr{_lr_tag(lr)}_ti{int(train_iters)}"
        ),
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

    eval_job_bar = tqdm(
        total=int(len(EVAL_ROLES) * len(EVAL_MAX_N_DEMOS_SWEEP)),
        desc=f"{case.name} eval role/demo sweep",
        leave=False,
    )
    for role in EVAL_ROLES:
        role_bank = (
            SHARED_SPLIT_BUNDLE.train_bank if str(role) == "train" else SHARED_SPLIT_BUNDLE.eval_bank
        )
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
                rule_bank=role_bank,
                split_rule_bundle_path=SHARED_SPLIT_BUNDLE_PATH,
                n_seq_ar=N_SEQ_AR,
                max_completion_len=MAX_COMPLETION_LEN,
                n_iters=EVAL_ITERS_PER_ROLE,
                eval_max_n_demos=int(eval_max_n_demos),
            )
            eval_job_bar.update(1)
        metrics_by_role_eval_demo[str(role)] = role_metrics
    eval_job_bar.close()

    post_eval_wall_s = time.perf_counter() - post_eval_start

    selection_metric_name = "first_transition_rule_reachable_rate"
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
            "lr": case.train_args["lr"],
            "eval_profile": case.info.get("train_eval_profile", "light"),
        },
        "metrics_final": case.hist["test"][-1] if case.hist and case.hist.get("test") else {},
        "metrics_by_role_eval_demo": metrics_by_role_eval_demo,
        "selection_role": "eval",
        "selection_eval_max_n_demos": int(SELECTION_EVAL_MAX_N_DEMOS),
        "selection_metric_name": selection_metric_name,
        "selection_metric_value": float(selection_metric_value),
        "eval_first_transition_rule_reachable_rate": _metric_by_role_demo(
            metrics_by_role_eval_demo,
            role="eval",
            eval_max_n_demos=int(SELECTION_EVAL_MAX_N_DEMOS),
            metric_name="first_transition_rule_reachable_rate",
        ),
        "eval_first_transition_rule_valid_rate": _metric_by_role_demo(
            metrics_by_role_eval_demo,
            role="eval",
            eval_max_n_demos=int(SELECTION_EVAL_MAX_N_DEMOS),
            metric_name="first_transition_rule_valid_rate",
        ),
        "eval_rollout_success_rate": _metric_by_role_demo(
            metrics_by_role_eval_demo,
            role="eval",
            eval_max_n_demos=int(SELECTION_EVAL_MAX_N_DEMOS),
            metric_name="rollout_success_rate",
        ),
        "train_first_transition_rule_reachable_rate": _metric_by_role_demo(
            metrics_by_role_eval_demo,
            role="train",
            eval_max_n_demos=int(SELECTION_EVAL_MAX_N_DEMOS),
            metric_name="first_transition_rule_reachable_rate",
        ),
        "train_rollout_success_rate": _metric_by_role_demo(
            metrics_by_role_eval_demo,
            role="train",
            eval_max_n_demos=int(SELECTION_EVAL_MAX_N_DEMOS),
            metric_name="rollout_success_rate",
        ),
        "split_bundle_path": str(SHARED_SPLIT_BUNDLE_PATH.resolve()),
        "split_bundle_sha256": SHARED_SPLIT_BUNDLE_SHA256,
        "split_bundle_config": {
            "seed": SPLIT_SEED,
            "predicates_per_layer": PREDICATES_PER_LAYER,
            "rules_01_train": RULES_01_TRAIN,
            "rules_01_eval": RULES_01_EVAL,
            "rules_12_shared": RULES_12_SHARED,
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
        "split_bundle": SHARED_SPLIT_BUNDLE_PAYLOAD,
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
