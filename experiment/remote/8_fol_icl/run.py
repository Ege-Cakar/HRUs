"""Architecture sweep on FOLLayerTask with in-context demonstrations."""

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
    FOLAtom,
    FOLLayerRule,
    FOLRuleBank,
    _augment_prompt_with_demos,
    _find_lhs_substitutions_for_facts,
    _subst_binds_rhs_variables,
    evaluate_layer_rollouts,
    evaluate_rule_matches,
    match_rule_completion_fol,
    sample_rollout_examples,
)
from task.layer_gen.util import tokenize_layer_fol
from task.layer_gen.util.fol_rule_bank import (
    build_random_fol_rule_bank,
    save_fol_rule_bank,
)
from train import Case, ce_mask

from utils import (
    extract_ar_free_run_inputs,
    extract_ar_rule_match_inputs,
    summarize_rule_match_metrics,
)
from experiment.utils.metrics_utils import final_token_accuracy


_LAYERED_PRED_RE = re.compile(r"r(\d+)_(\d+)$")

RUN_ID = new_seed()
print("RUN ID", RUN_ID)

TRAIN_MAX_DISTANCES = [4]
EVAL_DISTANCES = list(range(1, 9))

TRAIN_MAX_N_DEMOS = 8
EVAL_MAX_N_DEMOS_SWEEP = [0, 2, 4, 8, 12, 16, 24, 32]
SELECTION_EVAL_MAX_N_DEMOS = 8

BATCH_SIZE = 32
TRAIN_ITERS_SWEEP = [400, 1600, 6400, 25600, 102400]
TEST_EVERY = 1000
TEST_ITERS = 3
EVAL_ITERS_PER_DISTANCE = 3
ROLLOUT_EXAMPLES_PER_DISTANCE = 64

RUN_SPLIT = 10

RULE_BANK_SEED = 2029
N_LAYERS = 12
PREDICATES_PER_LAYER = 32
RULES_PER_TRANSITION = 64
K_IN_MAX = 3
K_OUT_MAX = 5
INITIAL_ANT_MAX = 5
ARITY_MAX = 3
VARS_PER_RULE_MAX = 6
CONSTANTS = [f'p{i}' for i in range(512)]
SAMPLE_MAX_ATTEMPTS = 4096
MAX_UNIFY_SOLUTIONS = 128

TRAIN_FIXED_LENGTH_MODE = "next_pow2"
EVAL_FIXED_LENGTH_MODE = "next_pow2"

# TRANSFORMER_LAYERS = [4, 8]
TRANSFORMER_LAYERS = [4]
# TRANSFORMER_WIDTH_HEADS = [(128, 4), (256, 8)]
TRANSFORMER_WIDTH_HEADS = [(256, 8)]
# TRANSFORMER_LRS = [3e-4, 1e-3]
TRANSFORMER_LRS = [5e-4]
TRANSFORMER_POS = ["rope"]
TRANSFORMER_SWIGLU = [True]

# MAMBA2_BONSAI_LAYERS = [4, 8]
MAMBA2_BONSAI_LAYERS = [4]
# MAMBA2_BONSAI_WIDTH_HEADS = [(128, 4), (256, 8)]
MAMBA2_BONSAI_WIDTH_HEADS = [(256, 8)]
# MAMBA2_BONSAI_D_STATE = [16, 32]
MAMBA2_BONSAI_D_STATE = [32]
MAMBA2_BONSAI_D_CONV = [4]
MAMBA2_BONSAI_SCAN_CHUNK_LEN = [64]
# MAMBA2_BONSAI_LRS = [3e-4, 1e-3]
MAMBA2_BONSAI_LRS = [5e-4]

### START TEST CONFIGS
# TRAIN_MAX_DISTANCES = [2]
# EVAL_DISTANCES = [1, 2, 8]
# TRAIN_MAX_N_DEMOS = 2
# EVAL_MAX_N_DEMOS_SWEEP = [0, 2, 8]
# SELECTION_EVAL_MAX_N_DEMOS = 2
# BATCH_SIZE = 8
# TRAIN_ITERS_SWEEP = [20]
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


class _TaskProbe:
    def __init__(self, name: str) -> None:
        self.name = str(name)
        self.n_batches = 0
        self.total_fetch_s = 0.0
        self.seq_len_counts: Counter[int] = Counter()

    def observe(self, batch, fetch_s: float) -> None:
        self.n_batches += 1
        self.total_fetch_s += float(fetch_s)
        if isinstance(batch, tuple) and len(batch) == 2:
            xs = np.asarray(batch[0])
            if xs.ndim == 2:
                self.seq_len_counts[int(xs.shape[1])] += 1


class _TaskProbeIterator:
    def __init__(self, task, probe: _TaskProbe) -> None:
        self._task = task
        self.probe = probe

    def __iter__(self):
        return self

    def __next__(self):
        start = time.perf_counter()
        batch = next(self._task)
        self.probe.observe(batch, time.perf_counter() - start)
        return batch

    def close(self) -> None:
        close = getattr(self._task, "close", None)
        if callable(close):
            close()

    def __getattr__(self, name):
        return getattr(self._task, name)


def _summarize_task_probe(probe: _TaskProbe) -> dict:
    top_seq_lens = [
        {"seq_len": int(seq_len), "count": int(count)}
        for seq_len, count in probe.seq_len_counts.most_common(10)
    ]
    seq_lens = list(probe.seq_len_counts.keys())
    avg_fetch_ms = 1_000.0 * probe.total_fetch_s / probe.n_batches if probe.n_batches > 0 else 0.0

    return {
        "name": probe.name,
        "n_batches": int(probe.n_batches),
        "total_fetch_s": float(probe.total_fetch_s),
        "avg_fetch_ms": float(avg_fetch_ms),
        "n_unique_seq_lens": int(len(probe.seq_len_counts)),
        "min_seq_len": int(min(seq_lens)) if seq_lens else None,
        "max_seq_len": int(max(seq_lens)) if seq_lens else None,
        "top_seq_lens": top_seq_lens,
    }


def _facts_key(facts: tuple[FOLAtom, ...]) -> tuple[str, ...]:
    return tuple(sorted(atom.text for atom in facts))


def _reachable_goal_exact_steps(
    *,
    rule_bank: FOLRuleBank,
    layer: int,
    facts: tuple[FOLAtom, ...],
    goal: FOLAtom,
    steps_remaining: int,
    max_unify_solutions: int,
    memo: dict[tuple[int, int, tuple[str, ...]], bool],
) -> bool:
    key = (int(layer), int(steps_remaining), _facts_key(facts))
    cached = memo.get(key)
    if cached is not None:
        return bool(cached)

    if int(steps_remaining) == 0:
        out = bool(goal in set(facts))
        memo[key] = out
        return out

    if int(layer) >= int(rule_bank.n_layers) - 1:
        memo[key] = False
        return False

    facts_tuple = tuple(facts)
    for rule in rule_bank.transition_rules(int(layer)):
        substitutions = _find_lhs_substitutions_for_facts(
            lhs=rule.lhs,
            facts=facts_tuple,
            max_solutions=int(max_unify_solutions),
        )
        if not substitutions:
            continue

        for subst in substitutions:
            if not _subst_binds_rhs_variables(rule=rule, subst=subst):
                continue
            next_rule = rule.instantiate(subst)
            next_facts = tuple(next_rule.rhs)
            if _reachable_goal_exact_steps(
                rule_bank=rule_bank,
                layer=int(layer) + 1,
                facts=next_facts,
                goal=goal,
                steps_remaining=int(steps_remaining) - 1,
                max_unify_solutions=max_unify_solutions,
                memo=memo,
            ):
                memo[key] = True
                return True

    memo[key] = False
    return False


def _predicted_rule_reaches_goal(
    *,
    rule_bank: FOLRuleBank,
    matched_rule: FOLLayerRule,
    goal: FOLAtom,
    goal_layer: int,
    max_unify_solutions: int,
) -> bool:
    dst_layer = int(matched_rule.dst_layer)
    remaining = int(goal_layer) - dst_layer
    if remaining < 0:
        return False

    memo: dict[tuple[int, int, tuple[str, ...]], bool] = {}
    return _reachable_goal_exact_steps(
        rule_bank=rule_bank,
        layer=dst_layer,
        facts=tuple(matched_rule.rhs),
        goal=goal,
        steps_remaining=remaining,
        max_unify_solutions=int(max_unify_solutions),
        memo=memo,
    )


class DemoAugmentedAdapter:
    """Prepend sampled demo completions before calling a base adapter."""

    def __init__(
        self,
        *,
        base_adapter,
        rule_bank: FOLRuleBank,
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


def _compute_dims(rule_bank, tokenizer, *, max_n_demos_for_shapes: int):
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

    if int(max_n_demos_for_shapes) > 0:
        max_demo_clause_len = 1
        for rules in rule_bank.transitions.values():
            for rule in rules:
                max_demo_clause_len = max(
                    max_demo_clause_len,
                    len(tokenizer.encode_completion(rule.statement_text)) - 1,
                )
        max_prompt_len += int(max_n_demos_for_shapes) * (int(max_demo_clause_len) + 1)

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
        "max_prompt_len": int(max_prompt_len),
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
    prediction_objective: str,
    rule_bank_path: Path,
    seed: int,
    drop_remainder: bool,
    shuffle: bool,
    max_n_demos: int,
    fixed_length_mode: str = "batch_max",
    fixed_length_n_seq: int | None = None,
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
        max_n_demos=int(max_n_demos),
        sample_max_attempts=SAMPLE_MAX_ATTEMPTS,
        max_unify_solutions=MAX_UNIFY_SOLUTIONS,
        fixed_length_mode=str(fixed_length_mode),
        fixed_length_n_seq=(
            None if fixed_length_n_seq is None else int(fixed_length_n_seq)
        ),
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

        prompt_tokens, fr_src_layers, fr_goals, fr_goal_layers = extract_ar_free_run_inputs(
            xs=np.asarray(xs),
            tokenizer=tokenizer,
        )
        n_examples = len(prompt_tokens)
        n_valid = 0
        n_reachable = 0
        n_decode_error = 0
        n_unknown_rule_error = 0

        for prompt, src_layer, goal, goal_layer in zip(
            prompt_tokens,
            fr_src_layers,
            fr_goals,
            fr_goal_layers,
        ):
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
                tokenizer=tokenizer,
            )

            if matched.decode_error:
                n_decode_error += 1
                continue
            if matched.unknown_rule_error or matched.matched_rule is None:
                n_unknown_rule_error += 1
                continue

            n_valid += 1
            if _predicted_rule_reaches_goal(
                rule_bank=rule_bank,
                matched_rule=matched.matched_rule,
                goal=goal,
                goal_layer=int(goal_layer),
                max_unify_solutions=MAX_UNIFY_SOLUTIONS,
            ):
                n_reachable += 1

        free_run_metrics = {
            "free_run_rule_valid_rate": float(n_valid) / float(n_examples) if n_examples > 0 else 0.0,
            "free_run_rule_reachable_rate": float(n_reachable) / float(n_examples)
            if n_examples > 0
            else 0.0,
            "free_run_decode_error_rate": float(n_decode_error) / float(n_examples)
            if n_examples > 0
            else 0.0,
            "free_run_unknown_rule_error_rate": float(n_unknown_rule_error) / float(n_examples)
            if n_examples > 0
            else 0.0,
        }

        return {
            "loss": loss_val,
            "token_acc": token_acc,
            "final_token_acc": final_acc,
            "seq_exact_acc": seq_exact_acc,
            **rule_metrics,
            **free_run_metrics,
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


def _evaluate_by_distance_for_demo(
    optimizer,
    *,
    tokenizer,
    rule_bank,
    rule_bank_path: Path,
    n_seq_ar: int,
    max_completion_len: int,
    n_iters: int,
    eval_max_n_demos: int,
    perf_stats: dict | None = None,
    progress_desc_prefix: str = "",
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

    eval_fetch_s = 0.0
    eval_metrics_s = 0.0
    eval_rollout_s = 0.0
    seq_len_counts: Counter[int] = Counter()
    n_eval_batches = 0

    metrics_by_distance = {}
    distance_iter = tqdm(
        EVAL_DISTANCES,
        desc=f"{progress_desc_prefix}distances@demos={int(eval_max_n_demos)}",
        leave=False,
    )
    for distance in distance_iter:
        eval_task = _make_layer_task(
            (distance, distance),
            prediction_objective="autoregressive",
            rule_bank_path=rule_bank_path,
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
                fetch_start = time.perf_counter()
                batch = next(eval_task)
                eval_fetch_s += time.perf_counter() - fetch_start

                xs = np.asarray(batch[0])
                if xs.ndim == 2:
                    seq_len_counts[int(xs.shape[1])] += 1
                n_eval_batches += 1

                metrics_start = time.perf_counter()
                all_batch_metrics.append(metrics_fn(optimizer, batch))
                eval_metrics_s += time.perf_counter() - metrics_start
            agg = _mean_metrics(all_batch_metrics)
        finally:
            close = getattr(eval_task, "close", None)
            if callable(close):
                close()

        rollout_examples = sample_rollout_examples(
            rule_bank=rule_bank,
            distance=int(distance),
            n_examples=int(ROLLOUT_EXAMPLES_PER_DISTANCE),
            initial_ant_max=int(INITIAL_ANT_MAX),
            max_steps=int(distance),
            max_unify_solutions=int(MAX_UNIFY_SOLUTIONS),
            rng=np.random.default_rng(
                int(RUN_ID) + 1000 * int(distance) + 100_000 * int(eval_max_n_demos)
            ),
        )
        rollout_start = time.perf_counter()
        rollout_metrics = evaluate_layer_rollouts(
            rule_bank=rule_bank,
            examples=rollout_examples,
            model=model_fn,
            adapter=rollout_adapter,
            tokenizer=tokenizer,
            temperature=0.0,
            rng=np.random.default_rng(
                int(RUN_ID) + 2000 * int(distance) + 100_000 * int(eval_max_n_demos)
            ),
        )
        eval_rollout_s += time.perf_counter() - rollout_start

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

    if perf_stats is not None:
        perf_stats.update(
            {
                "n_eval_batches": int(n_eval_batches),
                "eval_fetch_s": float(eval_fetch_s),
                "eval_metrics_s": float(eval_metrics_s),
                "eval_rollout_s": float(eval_rollout_s),
                "n_unique_seq_lens": int(len(seq_len_counts)),
                "top_seq_lens": [
                    {"seq_len": int(seq_len), "count": int(count)}
                    for seq_len, count in seq_len_counts.most_common(10)
                ],
            }
        )

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

DIMS = _compute_dims(
    SHARED_RULE_BANK,
    SHARED_TOKENIZER,
    max_n_demos_for_shapes=TRAIN_MAX_N_DEMOS,
)
EVAL_DIMS = _compute_dims(
    SHARED_RULE_BANK,
    SHARED_TOKENIZER,
    max_n_demos_for_shapes=max(EVAL_MAX_N_DEMOS_SWEEP),
)
N_VOCAB = max(int(DIMS["n_vocab"]), int(EVAL_DIMS["n_vocab"]))
MAX_COMPLETION_LEN = max(int(DIMS["max_completion_len"]), int(EVAL_DIMS["max_completion_len"]))
TRAIN_N_SEQ_AR_RAW = int(DIMS["n_seq_ar"])
EVAL_N_SEQ_AR_RAW = int(EVAL_DIMS["n_seq_ar"])
MODEL_N_SEQ_AR = int(max(2, _ceil_pow2_int(EVAL_N_SEQ_AR_RAW)))
TRAIN_N_SEQ_AR = int(MODEL_N_SEQ_AR)
N_SEQ_AR = int(MODEL_N_SEQ_AR)
DIMS = {"train": DIMS, "eval": EVAL_DIMS}

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
print("RULE BANK", {"path": str(SHARED_RULE_BANK_PATH), "sha256": SHARED_RULE_BANK_SHA256})

# <codecell>

all_cases = []

ar_light_metrics_fn = make_ar_light_metrics_fn()

for train_max_distance in TRAIN_MAX_DISTANCES:
    train_distances = _train_distances_for_k(train_max_distance)
    ood_distances = _ood_distances_for_k(train_max_distance)
    if not train_distances:
        raise ValueError(f"No train distances for k={train_max_distance}")
    if not ood_distances:
        raise ValueError(
            f"No OOD distances for k={train_max_distance}; ensure k < max(EVAL_DISTANCES)."
        )

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

        train_probe = _TaskProbe(name="train")
        test_probe = _TaskProbe(name="test")
        train_task = _TaskProbeIterator(
            _make_layer_task(
                train_distances,
                prediction_objective="autoregressive",
                rule_bank_path=SHARED_RULE_BANK_PATH,
                seed=new_seed(),
                drop_remainder=True,
                shuffle=True,
                max_n_demos=int(TRAIN_MAX_N_DEMOS),
                fixed_length_mode=TRAIN_FIXED_LENGTH_MODE,
                fixed_length_n_seq=TRAIN_N_SEQ_AR,
            ),
            train_probe,
        )
        test_task = _TaskProbeIterator(
            _make_layer_task(
                ood_distances,
                prediction_objective="autoregressive",
                rule_bank_path=SHARED_RULE_BANK_PATH,
                seed=new_seed(),
                drop_remainder=False,
                shuffle=True,
                max_n_demos=int(TRAIN_MAX_N_DEMOS),
                fixed_length_mode=EVAL_FIXED_LENGTH_MODE,
                fixed_length_n_seq=N_SEQ_AR,
            ),
            test_probe,
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
            "train_max_distance": int(train_max_distance),
            "train_distances": train_distances,
            "eval_distances": EVAL_DISTANCES,
            "ood_distances": ood_distances,
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
            f"8_fol_icl_transformer_k{int(train_max_distance):02d}_ti{int(train_iters)}",
            config,
            train_task=train_task,
            test_task=test_task,
            train_args=train_args,
            info=info,
        )
        case._train_probe = train_probe
        case._test_probe = test_probe
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

        train_probe = _TaskProbe(name="train")
        test_probe = _TaskProbe(name="test")
        train_task = _TaskProbeIterator(
            _make_layer_task(
                train_distances,
                prediction_objective="autoregressive",
                rule_bank_path=SHARED_RULE_BANK_PATH,
                seed=new_seed(),
                drop_remainder=True,
                shuffle=True,
                max_n_demos=int(TRAIN_MAX_N_DEMOS),
                fixed_length_mode=TRAIN_FIXED_LENGTH_MODE,
                fixed_length_n_seq=TRAIN_N_SEQ_AR,
            ),
            train_probe,
        )
        test_task = _TaskProbeIterator(
            _make_layer_task(
                ood_distances,
                prediction_objective="autoregressive",
                rule_bank_path=SHARED_RULE_BANK_PATH,
                seed=new_seed(),
                drop_remainder=False,
                shuffle=True,
                max_n_demos=int(TRAIN_MAX_N_DEMOS),
                fixed_length_mode=EVAL_FIXED_LENGTH_MODE,
                fixed_length_n_seq=N_SEQ_AR,
            ),
            test_probe,
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
            "model_family": "mamba2",
            "target_format": "next_token_full_sequence",
            "train_max_distance": int(train_max_distance),
            "train_distances": train_distances,
            "eval_distances": EVAL_DISTANCES,
            "ood_distances": ood_distances,
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
            f"8_fol_icl_mamba2_k{int(train_max_distance):02d}_ti{int(train_iters)}",
            config,
            train_task=train_task,
            test_task=test_task,
            train_args=train_args,
            info=info,
        )
        case._train_probe = train_probe
        case._test_probe = test_probe
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

    train_probe_summary = _summarize_task_probe(case._train_probe)
    test_probe_summary = _summarize_task_probe(case._test_probe)

    metrics_by_eval_demo = {}
    eval_perf_by_demo = {}
    post_eval_start = time.perf_counter()
    eval_demo_iter = tqdm(
        EVAL_MAX_N_DEMOS_SWEEP,
        desc=f"{case.name} eval-demo sweep",
        leave=False,
    )
    for eval_max_n_demos in eval_demo_iter:
        demo_perf = {}
        demo_start = time.perf_counter()
        metrics_by_eval_demo[int(eval_max_n_demos)] = _evaluate_by_distance_for_demo(
            case.optimizer,
            tokenizer=SHARED_TOKENIZER,
            rule_bank=SHARED_RULE_BANK,
            rule_bank_path=SHARED_RULE_BANK_PATH,
            n_seq_ar=N_SEQ_AR,
            max_completion_len=MAX_COMPLETION_LEN,
            n_iters=EVAL_ITERS_PER_DISTANCE,
            eval_max_n_demos=int(eval_max_n_demos),
            perf_stats=demo_perf,
            progress_desc_prefix=f"{case.name} ",
        )
        demo_perf["wall_s"] = float(time.perf_counter() - demo_start)
        eval_perf_by_demo[int(eval_max_n_demos)] = demo_perf
    post_eval_wall_s = time.perf_counter() - post_eval_start

    eval_fetch_s = float(sum(v.get("eval_fetch_s", 0.0) for v in eval_perf_by_demo.values()))
    eval_metrics_s = float(sum(v.get("eval_metrics_s", 0.0) for v in eval_perf_by_demo.values()))
    eval_rollout_s = float(sum(v.get("eval_rollout_s", 0.0) for v in eval_perf_by_demo.values()))
    train_fetch_s = float(train_probe_summary["total_fetch_s"])
    test_fetch_s = float(test_probe_summary["total_fetch_s"])
    train_non_fetch_s_est = float(max(0.0, train_wall_s - train_fetch_s - test_fetch_s))
    perf = {
        "train_wall_s": float(train_wall_s),
        "train_fetch_s": train_fetch_s,
        "test_fetch_s": test_fetch_s,
        "train_non_fetch_s_est": train_non_fetch_s_est,
        "post_eval_wall_s": float(post_eval_wall_s),
        "eval_fetch_s": eval_fetch_s,
        "eval_metrics_s": eval_metrics_s,
        "eval_rollout_s": eval_rollout_s,
        "train_task": train_probe_summary,
        "test_task": test_probe_summary,
        "eval_by_demo": eval_perf_by_demo,
    }

    print(
        "PERF SUMMARY",
        {
            "case": case.name,
            "train_wall_s": round(float(train_wall_s), 2),
            "train_fetch_s": round(train_fetch_s, 2),
            "train_non_fetch_s_est": round(train_non_fetch_s_est, 2),
            "post_eval_wall_s": round(float(post_eval_wall_s), 2),
            "train_unique_seq_lens": train_probe_summary["n_unique_seq_lens"],
            "test_unique_seq_lens": test_probe_summary["n_unique_seq_lens"],
        },
    )

    selected_distance_metrics = metrics_by_eval_demo[int(SELECTION_EVAL_MAX_N_DEMOS)]
    ood_distances = case.info.get("ood_distances", [])

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
        "metrics_by_eval_demo": metrics_by_eval_demo,
        "perf": perf,
        "metrics_by_distance": selected_distance_metrics,
        "selection_eval_max_n_demos": int(SELECTION_EVAL_MAX_N_DEMOS),
        "selection_metric_name": "ood_rollout_success_avg",
        "selection_metric_value": _avg_ood_metric(
            selected_distance_metrics,
            "rollout_success_rate",
            ood_distances,
        ),
        "ood_rollout_success_avg": _avg_ood_metric(
            selected_distance_metrics,
            "rollout_success_rate",
            ood_distances,
        ),
        "ood_final_token_acc_avg": _avg_ood_metric(
            selected_distance_metrics,
            "final_token_acc",
            ood_distances,
        ),
        "ood_valid_rule_rate_avg": _avg_ood_metric(
            selected_distance_metrics,
            "valid_rule_rate",
            ood_distances,
        ),
        "ood_correct_rule_rate_avg": _avg_ood_metric(
            selected_distance_metrics,
            "correct_rule_rate",
            ood_distances,
        ),
        "ood_free_run_rule_reachable_rate_avg": _avg_ood_metric(
            selected_distance_metrics,
            "free_run_rule_reachable_rate",
            ood_distances,
        ),
        "ood_rollout_unknown_rule_error_rate_avg": _avg_ood_metric(
            selected_distance_metrics,
            "rollout_unknown_rule_error_rate",
            ood_distances,
        ),
        "ood_rollout_decode_error_rate_avg": _avg_ood_metric(
            selected_distance_metrics,
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
            "train_max_n_demos": int(TRAIN_MAX_N_DEMOS),
            "eval_max_n_demos_sweep": [int(v) for v in EVAL_MAX_N_DEMOS_SWEEP],
            "sample_max_attempts": SAMPLE_MAX_ATTEMPTS,
            "max_unify_solutions": MAX_UNIFY_SOLUTIONS,
        },
        "rule_bank": SHARED_RULE_BANK_PAYLOAD,
        "dims": DIMS,
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
