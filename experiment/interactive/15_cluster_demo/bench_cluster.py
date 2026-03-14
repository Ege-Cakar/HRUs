"""Quick benchmark: cluster vs zipf_per_rule per-sample data generation cost.

Measures the dominant cost centers in the cluster demo distribution to
guide optimization decisions for experiment 15.
"""
# <codecell>
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

from task.layer_fol import FOLLayerTask
from task.layer_fol.demos import (
    _classify_rules_by_rank,
    _precompute_cluster_candidate_rankings,
    _sample_cluster,
)
from task.layer_gen.util.fol_rule_bank import (
    FOLRuleBank,
    build_random_fol_rule_bank,
    build_fresh_layer0_bank,
    generate_fresh_predicate_names,
    sample_fol_problem,
)


# ── Use the same configuration as experiment 15 ──
BASE_BANK_SEED = 2047
BASE_NUM_PRED = 16
MID_PRED = 256
N_LAYERS = 3
ARITY_MIN = 0
ARITY_MAX = 0
VARS_PER_RULE_MAX = 6
K_IN_MAX = 1
K_OUT_MAX = 1
INITIAL_ANT_MAX = 1
CONSTANTS = ("c0",)
PREDICATE_NAME_LEN = 4
MAX_UNIFY_SOLUTIONS = 128
CLUSTER_N_SAMPLES = 500

PREDICATES_PER_LAYER = (BASE_NUM_PRED, MID_PRED, BASE_NUM_PRED)
RULES_PER_TRANSITION = (BASE_NUM_PRED ** 2, BASE_NUM_PRED ** 2)


# <codecell>
def build_base_bank():
    return build_random_fol_rule_bank(
        n_layers=N_LAYERS,
        predicates_per_layer=PREDICATES_PER_LAYER,
        rules_per_transition=RULES_PER_TRANSITION,
        arity_min=ARITY_MIN,
        arity_max=ARITY_MAX,
        vars_per_rule_max=VARS_PER_RULE_MAX,
        k_in_max=K_IN_MAX,
        k_out_max=K_OUT_MAX,
        constants=CONSTANTS,
        rng=np.random.default_rng(BASE_BANK_SEED),
    )


def build_temp_bank(base_bank, rng):
    fresh_preds = generate_fresh_predicate_names(
        BASE_NUM_PRED, rng, name_len=PREDICATE_NAME_LEN,
    )
    # Match the actual task: fresh bank gets same number of rules as base bank
    n_fresh_rules = len(base_bank.transition_rules(0))
    return build_fresh_layer0_bank(
        base_bank=base_bank,
        fresh_predicates=fresh_preds,
        rules_per_transition=n_fresh_rules,
        k_in_min=1,
        k_in_max=K_IN_MAX,
        k_out_min=1,
        k_out_max=K_OUT_MAX,
        rng=rng,
    )


def time_fn(fn, n_calls=1, label=""):
    """Time a callable, return (total_seconds, per_call_seconds)."""
    t0 = time.perf_counter()
    for _ in range(n_calls):
        fn()
    elapsed = time.perf_counter() - t0
    per_call = elapsed / n_calls
    if label:
        print(f"  {label}: {elapsed:.4f}s total, {per_call*1000:.2f}ms/call  (n={n_calls})")
    return elapsed, per_call


# <codecell>
print("=" * 70)
print("BENCHMARK: cluster vs zipf_per_rule data generation cost")
print("=" * 70)

base_bank = build_base_bank()
rng = np.random.default_rng(42)

# ── 1. Build fresh temp bank ──
print("\n1. Build fresh temp bank")
time_fn(lambda: build_temp_bank(base_bank, np.random.default_rng(rng.integers(1 << 31))),
        n_calls=50, label="build_temp_bank")

# ── 2. sample_fol_problem ──
print("\n2. sample_fol_problem (distance=2)")
temp_bank = build_temp_bank(base_bank, np.random.default_rng(100))


def _do_sample_problem():
    return sample_fol_problem(
        bank=temp_bank, distance=2, initial_ant_max=INITIAL_ANT_MAX,
        rng=np.random.default_rng(rng.integers(1 << 31)),
        max_attempts=128, max_unify_solutions=MAX_UNIFY_SOLUTIONS,
    )


time_fn(_do_sample_problem, n_calls=500, label="sample_fol_problem")

# ── 3. _classify_rules_by_rank at each src_layer ──
print("\n3. _classify_rules_by_rank (256 rules)")
sampled = _do_sample_problem()

for step_idx in range(len(sampled.step_layers)):
    src_layer = int(sampled.step_layers[step_idx])
    ants = sampled.step_ants[step_idx]
    rules = list(temp_bank.transition_rules(src_layer))

    def _do_classify(sl=src_layer, a=ants, r=rules):
        return _classify_rules_by_rank(
            rules=r, ants=a, goal_atom=sampled.goal_atom,
            rule_bank=temp_bank, max_unify_solutions=MAX_UNIFY_SOLUTIONS,
        )

    n = 100 if src_layer == 0 else 500
    time_fn(_do_classify, n_calls=n,
            label=f"classify src_layer={src_layer} ({len(rules)} rules)")


# ── 4. _precompute_cluster_candidate_rankings ──
print("\n4. _precompute_cluster_candidate_rankings")
src_layer_0 = int(sampled.step_layers[0])
ants_0 = sampled.step_ants[0]
rules_0 = list(temp_bank.transition_rules(src_layer_0))
ranked_0 = _classify_rules_by_rank(
    rules=rules_0, ants=ants_0, goal_atom=sampled.goal_atom,
    rule_bank=temp_bank, max_unify_solutions=MAX_UNIFY_SOLUTIONS,
)

src_layer_1 = int(sampled.step_layers[1])
ants_1 = sampled.step_ants[1]
rules_1 = list(temp_bank.transition_rules(src_layer_1))
ranked_1 = _classify_rules_by_rank(
    rules=rules_1, ants=ants_1, goal_atom=sampled.goal_atom,
    rule_bank=temp_bank, max_unify_solutions=MAX_UNIFY_SOLUTIONS,
)

for src_layer, rules, ranked in [(src_layer_0, rules_0, ranked_0),
                                   (src_layer_1, rules_1, ranked_1)]:
    def _do_precompute(sl=src_layer, r=rules, rk=ranked):
        return _precompute_cluster_candidate_rankings(
            rule_bank=temp_bank, src_layer=sl, rules=r, actual_ranked=rk,
            rng=np.random.default_rng(rng.integers(1 << 31)),
            cluster_n_samples=CLUSTER_N_SAMPLES,
            max_unify_solutions=MAX_UNIFY_SOLUTIONS,
            distance=2, initial_ant_max=INITIAL_ANT_MAX,
        )

    n = 3 if src_layer == 0 else 10
    time_fn(_do_precompute, n_calls=n,
            label=f"precompute_candidates src_layer={src_layer} (n_samples={CLUSTER_N_SAMPLES})")


# ── 5. Full _sample_cluster ──
print("\n5. Full _sample_cluster")
for src_layer, ants in [(src_layer_0, ants_0), (src_layer_1, ants_1)]:
    def _do_cluster(sl=src_layer, a=ants):
        return _sample_cluster(
            rule_bank=temp_bank, src_layer=sl, ants=a,
            max_unify_solutions=MAX_UNIFY_SOLUTIONS,
            rng=np.random.default_rng(rng.integers(1 << 31)),
            n_demos=64, include_oracle=False, oracle_rule=None,
            alpha=2.0, goal_atom=sampled.goal_atom,
            demo_ranked=True, demo_unique=True,
            cluster_n_samples=CLUSTER_N_SAMPLES, cluster_k=5,
            cluster_base_dist="zipf_per_rule",
            cluster_unselected_rank=None,
            distance=2, initial_ant_max=INITIAL_ANT_MAX,
        )

    n = 3 if src_layer == 0 else 10
    time_fn(_do_cluster, n_calls=n,
            label=f"_sample_cluster src_layer={src_layer}")


# ── 6. Full task __next__() comparison ──
print("\n6. Full task __next__() — cluster vs zipf_per_rule")
print("   (batch_size=4, max_n_demos=64, online_prefetch=False)")
print("   NOTE: cluster task now precomputes candidates for base-bank layers at init")

for dist_name in ["zipf_per_rule", "cluster"]:
    task = FOLLayerTask(
        mode="online",
        task_split="depth3_fresh_icl",
        distance_range=2,
        batch_size=4,
        seed=999,
        predicates_per_layer=PREDICATES_PER_LAYER,
        rules_per_transition=RULES_PER_TRANSITION,
        fresh_icl_base_bank_seed=BASE_BANK_SEED,
        max_n_demos=64,
        min_n_demos=64,
        online_prefetch=False,
        demo_distribution=dist_name,
        demo_distribution_alpha=2.0,
        demo_ranked=True,
        cluster_k=5,
        cluster_n_samples=CLUSTER_N_SAMPLES,
        cluster_base_dist="zipf_per_rule",
        arity_min=ARITY_MIN,
        arity_max=ARITY_MAX,
        vars_per_rule_max=VARS_PER_RULE_MAX,
        k_in_min=1,
        k_in_max=K_IN_MAX,
        k_out_min=1,
        k_out_max=K_OUT_MAX,
        constants=CONSTANTS,
        initial_ant_max=INITIAL_ANT_MAX,
        sample_max_attempts=4096,
        max_unify_solutions=MAX_UNIFY_SOLUTIONS,
        predicate_name_len=PREDICATE_NAME_LEN,
    )
    n = 20 if dist_name == "zipf_per_rule" else 8
    time_fn(lambda t=task: next(t), n_calls=n,
            label=f"next() [{dist_name}] eval (batch=4, forced step=0)")
    task.close()

# Also test train role (random step_idx, benefits from precomputed layer-1)
print("\n7. Train-role task (random step, precomputed layer-1 candidates)")
for dist_name in ["zipf_per_rule", "cluster"]:
    task = FOLLayerTask(
        mode="online",
        task_split="depth3_fresh_icl",
        split_role="train",
        distance_range=2,
        batch_size=4,
        seed=999,
        predicates_per_layer=PREDICATES_PER_LAYER,
        rules_per_transition=RULES_PER_TRANSITION,
        fresh_icl_base_bank_seed=BASE_BANK_SEED,
        max_n_demos=64,
        min_n_demos=64,
        online_prefetch=False,
        demo_distribution=dist_name,
        demo_distribution_alpha=2.0,
        demo_ranked=True,
        cluster_k=5,
        cluster_n_samples=CLUSTER_N_SAMPLES,
        cluster_base_dist="zipf_per_rule",
        arity_min=ARITY_MIN,
        arity_max=ARITY_MAX,
        vars_per_rule_max=VARS_PER_RULE_MAX,
        k_in_min=1,
        k_in_max=K_IN_MAX,
        k_out_min=1,
        k_out_max=K_OUT_MAX,
        constants=CONSTANTS,
        initial_ant_max=INITIAL_ANT_MAX,
        sample_max_attempts=4096,
        max_unify_solutions=MAX_UNIFY_SOLUTIONS,
        predicate_name_len=PREDICATE_NAME_LEN,
    )
    n = 20 if dist_name == "zipf_per_rule" else 8
    time_fn(lambda t=task: next(t), n_calls=n,
            label=f"next() [{dist_name}] train (batch=4, random step)")
    task.close()


# ── 8. Projected training time ──
print("\n" + "=" * 70)
print("PROJECTED TRAINING TIME (25,600 iters, effective_batch=32)")
print("=" * 70)
print("(Use the per-call times above to estimate.)")
print("  zipf: 25600 × (per-iter data gen + GPU) ")
print("  cluster: 25600 × (per-iter data gen + GPU)")
print("  The GPU portion is the same for both.")
