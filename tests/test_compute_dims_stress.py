"""Stress test: verify _compute_dims does not underestimate sequence lengths.

Samples many batches from FOLLayerTask with depth3_fresh_icl and checks that
actual (unpadded) sequence lengths never exceed the estimates from _compute_dims.

The main concern is ARITY_MAX=1, where fresh predicates (char-tokenized, e.g.,
r_abcd -> 6 tokens) are much longer than base-bank predicates (single-token).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiment" / "remote" / "10_fresh_rule_split"))

from task.layer_fol import (
    FOLLayerTask,
    _build_tokenizer_for_fresh_icl,
    _fresh_predicate_sentinels,
)
from task.layer_gen.util.fol_rule_bank import (
    build_fresh_layer0_bank,
    build_random_fol_rule_bank,
    generate_fresh_predicate_names,
)


# ── Config matching run.py defaults (ARITY_MAX=1) ──────────────────────────
PREDICATES_PER_LAYER = 10
RULES_PER_TRANSITION = 18
FRESH_ICL_N_PREDICATES = 10
N_LAYERS = 3
ARITY_MAX = 1
VARS_PER_RULE_MAX = 6
K_IN_MAX = 1
K_OUT_MAX = 3
INITIAL_ANT_MAX = 3
CONSTANTS = ("p0",)
SAMPLE_MAX_ATTEMPTS = 4096
MAX_UNIFY_SOLUTIONS = 128
BASE_BANK_SEED = 2042
BATCH_SIZE = 8


def _ceil_pow2_int(n: int) -> int:
    n = int(n)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _build_base_bank_and_tokenizer():
    base_bank = build_random_fol_rule_bank(
        n_layers=N_LAYERS,
        predicates_per_layer=PREDICATES_PER_LAYER,
        rules_per_transition=RULES_PER_TRANSITION,
        arity_max=ARITY_MAX,
        vars_per_rule_max=VARS_PER_RULE_MAX,
        k_in_max=K_IN_MAX,
        k_out_max=K_OUT_MAX,
        constants=CONSTANTS,
        rng=np.random.default_rng(BASE_BANK_SEED),
    )
    tokenizer = _build_tokenizer_for_fresh_icl(base_bank=base_bank)
    return base_bank, tokenizer


def _compute_dims(base_bank, tokenizer, *, max_n_demos_for_shapes):
    """Replica of run.py _compute_dims (the function under test)."""
    all_rules = []
    for _src_layer, rules in base_bank.transitions.items():
        all_rules.extend(rules)
    if not all_rules:
        raise ValueError("Base bank has no rules.")

    max_rhs_atoms = max(len(rule.rhs) for rule in all_rules)
    max_prompt_facts = max(INITIAL_ANT_MAX, max_rhs_atoms)

    sentinels = _fresh_predicate_sentinels()
    merged_predicate_arities = dict(base_bank.predicate_arities)
    for s in sentinels:
        if s not in merged_predicate_arities:
            merged_predicate_arities[s] = base_bank.arity_max

    first_const = str(base_bank.constants[0])
    max_atom_len = 1
    for predicate, arity in merged_predicate_arities.items():
        atom_text = f"{predicate}({','.join(first_const for _ in range(arity))})"
        max_atom_len = max(max_atom_len, len(tokenizer.encode_completion(atom_text)) - 1)

    max_prompt_len = (
        max_prompt_facts * max_atom_len
        + max(0, max_prompt_facts - 1)
        + 1
        + max_atom_len
        + 1
    )

    if max_n_demos_for_shapes > 0:
        max_demo_clause_len = max(
            len(tokenizer.encode_completion(rule.statement_text)) - 1
            for rule in all_rules
        )
        fresh_clause_estimate = max_atom_len * (K_IN_MAX + K_OUT_MAX) + 5
        max_demo_clause_len = max(max_demo_clause_len, fresh_clause_estimate)
        max_prompt_len += max_n_demos_for_shapes * (max_demo_clause_len + 1)

    max_completion_len = max(
        len(tokenizer.encode_completion(rule.statement_text))
        for rule in all_rules
    )
    fresh_completion_estimate = max_atom_len * (K_IN_MAX + K_OUT_MAX) + 5
    max_completion_len = max(max_completion_len, fresh_completion_estimate)

    n_seq_ar = max_prompt_len + max_completion_len - 1
    return {
        "max_prompt_len": max_prompt_len,
        "max_completion_len": max_completion_len,
        "n_seq_ar": n_seq_ar,
        "max_atom_len": max_atom_len,
    }


def _actual_seq_lengths(batch):
    """Return list of per-example non-pad lengths from a batch (xs, ys)."""
    xs, _ = batch
    lengths = []
    for i in range(xs.shape[0]):
        nonzero = np.where(xs[i] != 0)[0]
        if len(nonzero) > 0:
            lengths.append(int(nonzero[-1]) + 1)
        else:
            lengths.append(0)
    return lengths


def _make_task(*, max_n_demos, n_seq_cap, seed=42):
    return FOLLayerTask(
        distance_range=(2, 2),
        batch_size=BATCH_SIZE,
        mode="online",
        task_split="depth3_fresh_icl",
        split_role="train",
        shuffle=True,
        seed=seed,
        worker_count=0,
        drop_remainder=False,
        prediction_objective="autoregressive",
        predicates_per_layer=PREDICATES_PER_LAYER,
        rules_per_transition=RULES_PER_TRANSITION,
        fresh_icl_n_predicates=FRESH_ICL_N_PREDICATES,
        arity_max=ARITY_MAX,
        vars_per_rule_max=VARS_PER_RULE_MAX,
        constants=CONSTANTS,
        k_in_max=K_IN_MAX,
        k_out_max=K_OUT_MAX,
        initial_ant_max=INITIAL_ANT_MAX,
        min_n_demos=max_n_demos,
        max_n_demos=max_n_demos,
        sample_max_attempts=SAMPLE_MAX_ATTEMPTS,
        max_unify_solutions=MAX_UNIFY_SOLUTIONS,
        fixed_length_mode="next_pow2",
        fixed_length_n_seq=n_seq_cap,
        online_prefetch_backend="sync",
    )


# ── Sentinel coverage: verify fresh predicates are never longer ─────────────
def test_sentinel_covers_all_fresh_names():
    """All fresh predicate names should be the same length as the sentinels."""
    sentinels = _fresh_predicate_sentinels()
    sentinel_lens = {len(s) for s in sentinels}

    rng = np.random.default_rng(123)
    for _ in range(1000):
        names = generate_fresh_predicate_names(64, rng)
        for name in names:
            assert len(name) in sentinel_lens, (
                f"Fresh name {name!r} (len={len(name)}) not covered by sentinel "
                f"lengths {sentinel_lens}"
            )


# ── Atom tokenization: verify fresh atoms are no longer than estimated ──────
def test_fresh_atom_tokenization_bounded():
    """Fresh atom tokens should not exceed the max_atom_len estimate."""
    base_bank, tokenizer = _build_base_bank_and_tokenizer()
    dims = _compute_dims(base_bank, tokenizer, max_n_demos_for_shapes=0)
    max_atom_len = dims["max_atom_len"]

    rng = np.random.default_rng(456)
    for _ in range(500):
        names = generate_fresh_predicate_names(64, rng)
        for name in names:
            for const in CONSTANTS:
                atom_text = f"{name}({const})"
                encoded_len = len(tokenizer.encode_completion(atom_text)) - 1
                assert encoded_len <= max_atom_len, (
                    f"Atom {atom_text!r} encodes to {encoded_len} tokens, "
                    f"exceeds estimate {max_atom_len}"
                )


# ── Fresh clause tokenization: verify completions are bounded ───────────────
def test_fresh_clause_tokenization_bounded():
    """Fresh rule statement tokens should not exceed max_completion_len."""
    base_bank, tokenizer = _build_base_bank_and_tokenizer()
    dims = _compute_dims(base_bank, tokenizer, max_n_demos_for_shapes=8)
    max_completion_len = dims["max_completion_len"]

    rng = np.random.default_rng(789)
    for _ in range(200):
        fresh_preds = generate_fresh_predicate_names(FRESH_ICL_N_PREDICATES, rng)
        temp_bank = build_fresh_layer0_bank(
            base_bank=base_bank,
            fresh_predicates=fresh_preds,
            rules_per_transition=RULES_PER_TRANSITION,
            k_in_min=1,
            k_in_max=K_IN_MAX,
            k_out_min=1,
            k_out_max=K_OUT_MAX,
            rng=rng,
        )
        for rules in temp_bank.transitions.values():
            for rule in rules:
                encoded_len = len(tokenizer.encode_completion(rule.statement_text))
                assert encoded_len <= max_completion_len, (
                    f"Rule {rule.statement_text!r} encodes to {encoded_len} "
                    f"tokens (incl EOT), exceeds estimate {max_completion_len}"
                )


# ── End-to-end: sample batches and verify lengths stay within bounds ────────
@pytest.mark.parametrize("max_n_demos", [0, 4, 8])
def test_actual_batch_lengths_within_bounds(max_n_demos):
    """Sample batches from FOLLayerTask and check no sequence exceeds estimate."""
    base_bank, tokenizer = _build_base_bank_and_tokenizer()
    dims = _compute_dims(base_bank, tokenizer, max_n_demos_for_shapes=max_n_demos)
    n_seq_ar = dims["n_seq_ar"]
    n_seq_cap = max(2, _ceil_pow2_int(n_seq_ar))

    task = _make_task(
        max_n_demos=max_n_demos,
        n_seq_cap=n_seq_cap,
        seed=1234 + max_n_demos,
    )
    n_batches = 50
    max_seen = 0
    try:
        for _ in range(n_batches):
            batch = next(task)
            lengths = _actual_seq_lengths(batch)
            batch_max = max(lengths)
            max_seen = max(max_seen, batch_max)
            assert batch_max <= n_seq_cap, (
                f"Batch sequence length {batch_max} exceeds cap {n_seq_cap} "
                f"(raw estimate n_seq_ar={n_seq_ar}, demos={max_n_demos})"
            )
    finally:
        close = getattr(task, "close", None)
        if callable(close):
            close()

    print(
        f"\n  demos={max_n_demos}: max_seen={max_seen}, "
        f"n_seq_ar={n_seq_ar}, cap={n_seq_cap}, "
        f"headroom={n_seq_cap - max_seen}"
    )


# ── Stress: high demo counts to push length estimates ───────────────────────
@pytest.mark.parametrize("max_n_demos", [16, 32])
def test_high_demo_count_lengths(max_n_demos):
    """With many demos, verify sequence lengths don't overflow the estimate."""
    base_bank, tokenizer = _build_base_bank_and_tokenizer()
    dims = _compute_dims(base_bank, tokenizer, max_n_demos_for_shapes=max_n_demos)
    n_seq_ar = dims["n_seq_ar"]
    n_seq_cap = max(2, _ceil_pow2_int(n_seq_ar))

    task = _make_task(
        max_n_demos=max_n_demos,
        n_seq_cap=n_seq_cap,
        seed=9999 + max_n_demos,
    )
    n_batches = 30
    max_seen = 0
    try:
        for _ in range(n_batches):
            batch = next(task)
            lengths = _actual_seq_lengths(batch)
            batch_max = max(lengths)
            max_seen = max(max_seen, batch_max)
            assert batch_max <= n_seq_cap, (
                f"Batch sequence length {batch_max} exceeds cap {n_seq_cap} "
                f"(raw estimate n_seq_ar={n_seq_ar}, demos={max_n_demos})"
            )
    finally:
        close = getattr(task, "close", None)
        if callable(close):
            close()

    print(
        f"\n  demos={max_n_demos}: max_seen={max_seen}, "
        f"n_seq_ar={n_seq_ar}, cap={n_seq_cap}, "
        f"headroom={n_seq_cap - max_seen}"
    )
