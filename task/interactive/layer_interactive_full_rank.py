"""Interactive sweep: Kendall tau over beta for the full_rank demo distribution
under the depth3_fresh_icl split.

In this mode, every example gets completely fresh layer-0 predicates and fresh
layer-0→1 rules, while layer-1→2 transitions remain fixed from a shared base
bank.  This forces the model to rely entirely on in-context demonstrations.

For each trial in the sweep, a new temp bank is constructed with fresh layer-0
predicates, a problem is sampled, rules are classified, and ranked demos are
drawn.  `beta` controls the Plackett-Luce ordering noise:
  - beta=0   -> random permutation
  - beta=inf -> deterministic best-first
  - intermediate -> noisy ranking
"""
# <codecell>
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from scipy.stats import kendalltau
from task.layer_fol import (
    FOLLayerTask,
    _build_tokenizer_for_fresh_icl,
    _fresh_predicate_sentinels,
    compute_fol_dims,
    print_task_preview,
    split_prompt_row_segments,
)
from task.layer_fol.demos import (
    _classify_rules_by_rank,
    augment_prompt_with_demos,
    sample_ranked_demos,
)
from task.layer_gen.util import tokenize_layer_fol as tok
from task.layer_gen.util.fol_rule_bank import (
    FOLSequent,
    build_fresh_layer0_bank,
    build_random_fol_rule_bank,
    generate_fresh_predicate_names,
    sample_fol_problem,
)


# <codecell>  Configuration (inline editable)
BETA_VALUES = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, float('inf')]
N_SAMPLES = 200
N_DEMOS = 32
SEED = 43
DEMO_DISTRIBUTION = "full_rank"

# Base bank config (shared layer 1→2)
BASE_BANK_SEED = 2043
N_LAYERS = 3
PREDICATES_PER_LAYER = [8, 64, 8]
RULES_PER_TRANSITION = [64, 64]
ARITY_MIN = 0
ARITY_MAX = 0
VARS_PER_RULE_MAX = 1
CONSTANTS = ("a", "b", "c", "d")
K_IN_MIN = 1
K_IN_MAX = 1
K_OUT_MIN = 1
K_OUT_MAX = 1
INITIAL_ANT_MAX = 1
MAX_UNIFY_SOLUTIONS = 128
PREDICATE_NAME_LEN = 4

# <codecell>  Build base bank + fresh-ICL tokenizer
base_bank_rng = np.random.default_rng(BASE_BANK_SEED)
base_bank = build_random_fol_rule_bank(
    n_layers=N_LAYERS,
    predicates_per_layer=PREDICATES_PER_LAYER,
    rules_per_transition=RULES_PER_TRANSITION,
    arity_min=ARITY_MIN,
    arity_max=ARITY_MAX,
    vars_per_rule_max=VARS_PER_RULE_MAX,
    constants=CONSTANTS,
    k_in_min=K_IN_MIN,
    k_in_max=K_IN_MAX,
    k_out_min=K_OUT_MIN,
    k_out_max=K_OUT_MAX,
    rng=base_bank_rng,
)

tokenizer = _build_tokenizer_for_fresh_icl(
    base_bank=base_bank,
    predicate_name_len=PREDICATE_NAME_LEN,
)

print(f"Base bank: {N_LAYERS} layers")
for layer in range(N_LAYERS - 1):
    n_rules = len(list(base_bank.transition_rules(layer)))
    print(f"  Layer {layer}→{layer+1}: {n_rules} rules")
print(f"Tokenizer vocab size: {tokenizer.vocab_size}")


# <codecell>  Helper: build a fresh temp bank and sample a problem
def _fresh_bank_and_problem(rng: np.random.Generator):
    """Generate fresh layer-0 predicates, build temp bank, sample a distance-2 problem."""
    fresh_preds = generate_fresh_predicate_names(
        PREDICATES_PER_LAYER[0],
        rng,
        name_len=PREDICATE_NAME_LEN,
    )
    temp_bank = build_fresh_layer0_bank(
        base_bank=base_bank,
        fresh_predicates=fresh_preds,
        rules_per_transition=RULES_PER_TRANSITION[0],
        k_in_min=K_IN_MIN,
        k_in_max=K_IN_MAX,
        k_out_min=K_OUT_MIN,
        k_out_max=K_OUT_MAX,
        rng=rng,
    )
    sampled = sample_fol_problem(
        bank=temp_bank,
        distance=2,
        initial_ant_max=INITIAL_ANT_MAX,
        rng=rng,
        max_attempts=4096,
        max_unify_solutions=MAX_UNIFY_SOLUTIONS,
    )
    return temp_bank, sampled


# <codecell>  Sweep loop (fresh bank per trial)
results = []

for beta in BETA_VALUES:
    taus = []
    rank_1_fracs = []
    for trial in range(N_SAMPLES):
        rng = np.random.default_rng(SEED * 1000 + trial)
        try:
            temp_bank, sampled = _fresh_bank_and_problem(rng)
        except RuntimeError:
            continue

        # Use step_idx=0 -> src_layer=0 (the fresh layer)
        src_layer = sampled.step_layers[0]
        ants = sampled.step_ants[0]
        goal_atom = sampled.goal_atom
        rules = list(temp_bank.transition_rules(int(src_layer)))

        ranked = _classify_rules_by_rank(
            rules=rules,
            ants=ants,
            goal_atom=goal_atom,
            rule_bank=temp_bank,
            max_unify_solutions=MAX_UNIFY_SOLUTIONS,
        )

        schemas, ranks = sample_ranked_demos(
            ranked_rules=ranked,
            rng=rng,
            n_demos=min(N_DEMOS, len(rules)),
            demo_distribution=DEMO_DISTRIBUTION,
            alpha=1.0,  # unused by full_rank
            include_oracle=False,
            oracle_rule=None,
            demo_ranked=True,
            demo_unique=True,
            demo_ranking_beta=beta,
        )
        if len(ranks) < 2:
            continue
        ideal = sorted(ranks, reverse=True)
        tau, _ = kendalltau(ideal, ranks)
        taus.append(tau)
        rank_1_fracs.append(sum(1 for r in ranks if r == 1) / len(ranks))

    mean_tau = float(np.nanmean(taus)) if taus else float('nan')
    std_tau = float(np.nanstd(taus)) if taus else float('nan')
    mean_r1 = float(np.nanmean(rank_1_fracs)) if rank_1_fracs else float('nan')
    n_valid = int(np.sum(np.isfinite(taus))) if taus else 0
    results.append({
        "beta": beta,
        "mean_tau": mean_tau,
        "std_tau": std_tau,
        "mean_rank1_frac": mean_r1,
        "n": n_valid,
    })
    beta_str = "inf" if not np.isfinite(beta) else f"{beta:.2f}"
    print(
        f"  beta={beta_str:>5s}  tau={mean_tau:+.3f} +/- {std_tau:.3f}  "
        f"rank1_frac={mean_r1:.2f}  (n={n_valid})"
    )

# <codecell>  Display: table of results
import pandas as pd

df = pd.DataFrame(results)
print("\n=== Full-rank sweep (depth3_fresh_icl): mean Kendall tau by beta ===")
print(df.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

# <codecell>  Preview samples at representative beta points
PREVIEW_POINTS = [
    (0.0, "random (beta=0)"),
    (0.5, "noisy (beta=0.5)"),
    (1.0, "moderate (beta=1)"),
    (5.0, "strong (beta=5)"),
    (float('inf'), "deterministic (beta=inf)"),
]
PREVIEW_N_DEMOS = 8
PREVIEW_N_SAMPLES = 3

print("\n" + "=" * 70)
print("SAMPLE PREVIEWS (depth3_fresh_icl)")
print("=" * 70)

for p_beta, label in PREVIEW_POINTS:
    beta_str = "inf" if not np.isfinite(p_beta) else f"{p_beta:.1f}"
    print(f"\n--- beta={beta_str}  ({label}) ---")

    for trial in range(PREVIEW_N_SAMPLES):
        preview_rng = np.random.default_rng(SEED * 100 + trial)
        try:
            temp_bank, sampled = _fresh_bank_and_problem(preview_rng)
        except RuntimeError:
            print(f"  sample {trial}: (failed to sample problem)")
            continue

        src_layer = sampled.step_layers[0]
        ants = sampled.step_ants[0]
        goal_atom = sampled.goal_atom
        sequent = FOLSequent(ants=ants, cons=goal_atom)
        n_rules = len(list(temp_bank.transition_rules(int(src_layer))))

        result = augment_prompt_with_demos(
            prompt_tokens=tokenizer.tokenize_prompt(sequent),
            rule_bank=temp_bank,
            tokenizer=tokenizer,
            rng=preview_rng,
            src_layer=int(src_layer),
            ants=ants,
            max_n_demos=min(PREVIEW_N_DEMOS, n_rules),
            min_n_demos=min(PREVIEW_N_DEMOS, n_rules),
            max_unify_solutions=MAX_UNIFY_SOLUTIONS,
            include_oracle=False,
            oracle_rule=sampled.step_rules[0],
            demo_distribution=DEMO_DISTRIBUTION,
            demo_distribution_alpha=1.0,
            goal_atom=goal_atom,
            demo_ranked=True,
            demo_unique=True,
            demo_ranking_beta=p_beta,
        )
        rank_str = " ".join(f"{r}" for r in result.demo_ranks)
        print(f"  sample {trial}: src_layer={src_layer} n_rules={n_rules} ranks=[{rank_str}]")
        print(f"    query: {sequent.text}")
        for schema, inst, rank in zip(
            result.demo_schemas, result.demo_instances, result.demo_ranks
        ):
            oracle_tag = " *oracle*" if schema == sampled.step_rules[0] else ""
            print(f"    [{rank}] {schema.statement_text}  ->  {inst}{oracle_tag}")

# <codecell>  Preview: oracle injection
print("\n" + "=" * 70)
print("ORACLE INJECTION PREVIEW (depth3_fresh_icl)")
print("=" * 70)

for p_beta, label in [(1.0, "noisy"), (float('inf'), "deterministic")]:
    beta_str = "inf" if not np.isfinite(p_beta) else f"{p_beta:.1f}"
    print(f"\n--- beta={beta_str}  ({label}, include_oracle=True) ---")
    for trial in range(PREVIEW_N_SAMPLES):
        preview_rng = np.random.default_rng(SEED * 100 + trial)
        try:
            temp_bank, sampled = _fresh_bank_and_problem(preview_rng)
        except RuntimeError:
            print(f"  sample {trial}: (failed to sample problem)")
            continue

        src_layer = sampled.step_layers[0]
        ants = sampled.step_ants[0]
        goal_atom = sampled.goal_atom
        sequent = FOLSequent(ants=ants, cons=goal_atom)
        n_rules = len(list(temp_bank.transition_rules(int(src_layer))))

        result = augment_prompt_with_demos(
            prompt_tokens=tokenizer.tokenize_prompt(sequent),
            rule_bank=temp_bank,
            tokenizer=tokenizer,
            rng=preview_rng,
            src_layer=int(src_layer),
            ants=ants,
            max_n_demos=min(PREVIEW_N_DEMOS, n_rules),
            min_n_demos=min(PREVIEW_N_DEMOS, n_rules),
            max_unify_solutions=MAX_UNIFY_SOLUTIONS,
            include_oracle=True,
            oracle_rule=sampled.step_rules[0],
            demo_distribution=DEMO_DISTRIBUTION,
            demo_distribution_alpha=1.0,
            goal_atom=goal_atom,
            demo_ranked=True,
            demo_unique=True,
            demo_ranking_beta=p_beta,
        )
        rank_str = " ".join(f"{r}" for r in result.demo_ranks)
        has_oracle = any(s == sampled.step_rules[0] for s in result.demo_schemas)
        print(f"  sample {trial}: ranks=[{rank_str}]  oracle_present={has_oracle}")

# <codecell>  FOLLayerTask-level oracle preview
#
# The sections above used the low-level `augment_prompt_with_demos` API.
# This section creates fully initialized FOLLayerTask instances with
# include_oracle=True so we can see exactly where the oracle ends up in
# the final tokenized prompt after passing through the complete task pipeline.

TASK_ORACLE_BETA_POINTS = [0.0, 1.0, float('inf')]
TASK_ORACLE_N_PREVIEW = 10
TASK_ORACLE_N_DEMOS = 8

sentinels = _fresh_predicate_sentinels(name_len=PREDICATE_NAME_LEN)
extra_arities = {s: int(base_bank.arity_max) for s in sentinels}

_task_dims = compute_fol_dims(
    rule_banks=[base_bank],
    tokenizer=tokenizer,
    initial_ant_max=INITIAL_ANT_MAX,
    max_n_demos=TASK_ORACLE_N_DEMOS,
    extra_predicate_arities=extra_arities,
    fresh_k_in_max=K_IN_MAX,
    fresh_k_out_max=K_OUT_MAX,
)


def _ceil_pow2(n: int) -> int:
    n = int(n)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


_task_n_seq = max(2, _ceil_pow2(int(_task_dims["n_seq_ar"])))

print("\n" + "=" * 70)
print("FOLLayerTask ORACLE POSITION PREVIEW (depth3_fresh_icl, include_oracle=True)")
print("=" * 70)

for p_beta in TASK_ORACLE_BETA_POINTS:
    beta_str = "inf" if not np.isfinite(p_beta) else f"{p_beta:.1f}"
    print(f"\n--- beta={beta_str} ---")

    task = FOLLayerTask(
        mode="online",
        task_split="depth3_fresh_icl",
        split_role="train",
        seed=SEED + 1,
        distance_range=(2, 2),
        batch_size=1,
        initial_ant_max=INITIAL_ANT_MAX,
        prediction_objective="autoregressive",
        fixed_length_mode="next_pow2",
        fixed_length_n_seq=_task_n_seq,
        n_layers=N_LAYERS,
        predicates_per_layer=PREDICATES_PER_LAYER,
        rules_per_transition=RULES_PER_TRANSITION,
        fresh_icl_base_bank_seed=BASE_BANK_SEED,
        arity_min=ARITY_MIN,
        arity_max=ARITY_MAX,
        vars_per_rule_max=VARS_PER_RULE_MAX,
        constants=CONSTANTS,
        k_in_max=K_IN_MAX,
        k_out_max=K_OUT_MAX,
        predicate_name_len=PREDICATE_NAME_LEN,
        sample_max_attempts=4096,
        max_unify_solutions=MAX_UNIFY_SOLUTIONS,
        min_n_demos=TASK_ORACLE_N_DEMOS,
        max_n_demos=TASK_ORACLE_N_DEMOS,
        include_oracle=True,
        demo_distribution=DEMO_DISTRIBUTION,
        demo_ranking_beta=p_beta,
        online_prefetch=False,
    )

    for i in range(TASK_ORACLE_N_PREVIEW):
        record = task._sample_online_record()
        rule_context = record.get("rule_context", {})
        prompt = np.asarray(record["prompt"], dtype=np.int32)
        src_layer = int(record["src_layer"])
        gt_statements = record["statement_texts"]

        demo_segments, main_segment = split_prompt_row_segments(
            prompt, tokenizer=task.tokenizer,
        )
        sequent = task.tokenizer.decode_prompt(main_segment.tolist())

        # Decode each demo to text
        demo_texts = []
        for demo in demo_segments:
            demo_text = task.tokenizer.decode_completion_texts(
                list(demo) + [int(task.tokenizer.eot_token_id)]
            )[0]
            demo_texts.append(demo_text)

        # Identify which demo slots contain the oracle rule.
        # Use exact equality to avoid substring false positives
        # (e.g. "r1_2" matching inside "r1_21").
        oracle_positions = []
        for idx, dt in enumerate(demo_texts):
            for gt in gt_statements:
                if dt == gt:
                    oracle_positions.append(idx)
                    break

        n_demos = len(demo_segments)
        oracle_pos_str = (
            ", ".join(str(p) for p in oracle_positions) if oracle_positions else "absent"
        )
        print(
            f"  sample {i}: n_demos={n_demos}  "
            f"oracle_at=[{oracle_pos_str}] (0=first shown, {n_demos - 1}=closest to query)"
        )
        print(f"    query: {sequent.text}")
        print(f"    gt:    {gt_statements}")
        for d_idx, dt in enumerate(demo_texts):
            tag = " <-- ORACLE" if d_idx in oracle_positions else ""
            print(f"    demo[{d_idx}]: {dt}{tag}")

    task.close()

# <codecell>  Optional: matplotlib tau-vs-beta curve
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    betas_finite = [r["beta"] for r in results if np.isfinite(r["beta"])]
    taus_finite = [r["mean_tau"] for r in results if np.isfinite(r["beta"])]
    stds_finite = [r["std_tau"] for r in results if np.isfinite(r["beta"])]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        betas_finite, taus_finite, yerr=stds_finite,
        marker="o", capsize=4, linewidth=1.5, label="full_rank (fresh_icl)",
    )
    # Mark beta=inf as a horizontal dashed line
    inf_result = [r for r in results if not np.isfinite(r["beta"])]
    if inf_result:
        ax.axhline(
            inf_result[0]["mean_tau"], color="tab:red",
            linestyle="--", linewidth=1, label=f"beta=inf ({inf_result[0]['mean_tau']:+.3f})",
        )

    ax.set_xlabel("beta (ranking strength)")
    ax.set_ylabel("Mean Kendall tau")
    ax.set_title("full_rank (depth3_fresh_icl): demo order correlation vs ideal")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_path = Path(__file__).parent / "set" / "full_rank_beta_tau.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nTau-vs-beta plot saved to {out_path}")
    plt.close(fig)

    # Second plot: rank-1 fraction vs beta
    r1_finite = [r["mean_rank1_frac"] for r in results if np.isfinite(r["beta"])]
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(betas_finite, r1_finite, marker="s", linewidth=1.5, color="tab:green")
    if inf_result:
        ax2.axhline(
            inf_result[0]["mean_rank1_frac"], color="tab:red",
            linestyle="--", linewidth=1, label=f"beta=inf ({inf_result[0]['mean_rank1_frac']:.2f})",
        )
    ax2.set_xlabel("beta (ranking strength)")
    ax2.set_ylabel("Mean fraction of rank-1 rules in demos")
    ax2.set_title("full_rank (depth3_fresh_icl): rank-1 concentration vs beta")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    out_path2 = Path(__file__).parent / "set" / "full_rank_beta_rank1_frac.png"
    fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
    print(f"Rank-1 fraction plot saved to {out_path2}")
    plt.close(fig2)
except ImportError:
    print("\nmatplotlib not available; skipping plots.")
