"""Interactive sweep: Kendall tau over (alpha, beta) grid for demo ranking.

Measures how the combination of retrieval quality (alpha) and ranking
strength (beta) affects the correlation between returned demo order and
the ideal descending-rank ordering.
"""
# <codecell>
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from scipy.stats import kendalltau
from task.layer_fol.demos import (
    _classify_rules_by_rank,
    sample_ranked_demos,
)
from task.layer_gen.util.fol_rule_bank import (
    build_random_fol_rule_bank,
    sample_fol_problem,
)


# <codecell>  Configuration (inline editable)
ALPHA_VALUES = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
BETA_VALUES = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, float('inf')]
N_SAMPLES = 200
N_DEMOS = 32
SEED = 42
DEMO_DISTRIBUTION = "zipf_per_rule"

# Rule bank config
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

# <codecell>  Build rule bank, sample a problem, classify rules by rank
bank_rng = np.random.default_rng(SEED)
bank = build_random_fol_rule_bank(
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
    rng=bank_rng,
)

problem_rng = np.random.default_rng(SEED + 1)
sampled = sample_fol_problem(
    bank=bank,
    distance=1,
    initial_ant_max=INITIAL_ANT_MAX,
    rng=problem_rng,
    max_attempts=4096,
    max_unify_solutions=MAX_UNIFY_SOLUTIONS,
)
src_layer = sampled.step_layers[0]
ants = sampled.step_ants[0]
goal_atom = sampled.goal_atom
rules = list(bank.transition_rules(int(src_layer)))

ranked = _classify_rules_by_rank(
    rules=rules,
    ants=ants,
    goal_atom=goal_atom,
    rule_bank=bank,
    max_unify_solutions=MAX_UNIFY_SOLUTIONS,
)

print(f"Rule bank: {len(rules)} rules at layer {src_layer}")
for rank in sorted(ranked):
    print(f"  Rank {rank}: {len(ranked[rank])} rules")

# <codecell>  Sweep loop
results = []

for alpha in ALPHA_VALUES:
    for beta in BETA_VALUES:
        taus = []
        for trial in range(N_SAMPLES):
            rng = np.random.default_rng(SEED * 1000 + trial)
            schemas, ranks = sample_ranked_demos(
                ranked_rules=ranked,
                rng=rng,
                n_demos=min(N_DEMOS, len(rules)),
                demo_distribution=DEMO_DISTRIBUTION,
                alpha=alpha,
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

        mean_tau = float(np.mean(taus)) if taus else float('nan')
        std_tau = float(np.std(taus)) if taus else float('nan')
        results.append({
            "alpha": alpha,
            "beta": beta,
            "mean_tau": mean_tau,
            "std_tau": std_tau,
            "n": len(taus),
        })
        beta_str = "inf" if not np.isfinite(beta) else f"{beta:.2f}"
        print(f"  alpha={alpha:.1f}  beta={beta_str:>5s}  tau={mean_tau:+.3f} ± {std_tau:.3f}  (n={len(taus)})")

# <codecell>  Display: pandas pivot table of mean Kendall tau
import pandas as pd

df = pd.DataFrame(results)
pivot = df.pivot_table(
    index="alpha",
    columns="beta",
    values="mean_tau",
    aggfunc="first",
)
print("\n=== Mean Kendall tau (alpha × beta) ===")
print(pivot.to_string(float_format=lambda x: f"{x:+.3f}"))

# <codecell>  Preview samples at representative (alpha, beta) points
from task.layer_fol.demos import (
    augment_prompt_with_demos,
)
from task.layer_gen.util import tokenize_layer_fol as tok
from task.layer_gen.util.fol_rule_bank import FOLSequent

PREVIEW_POINTS = [
    # (alpha, beta, label)
    (1.0, 0.0, "uniform retrieval, random order"),
    (1.0, 1.0, "uniform retrieval, noisy order"),
    (1.0, float('inf'), "uniform retrieval, perfect order"),
    (5.0, 0.0, "strong retrieval, random order"),
    (5.0, 1.0, "strong retrieval, noisy order"),
    (5.0, float('inf'), "strong retrieval, perfect order"),
]
PREVIEW_N_DEMOS = min(8, len(rules))
PREVIEW_N_SAMPLES = 3

tokenizer = tok.build_tokenizer_from_rule_bank(bank)
sequent = FOLSequent(ants=ants, cons=goal_atom)

print("\n" + "=" * 70)
print("SAMPLE PREVIEWS")
print("=" * 70)
print(f"\nQuery: {sequent.text}")
print(f"Oracle rule: {sampled.step_rules[0].statement_text}")
print(f"Layer {src_layer}, {len(rules)} candidate rules")

for p_alpha, p_beta, label in PREVIEW_POINTS:
    beta_str = "inf" if not np.isfinite(p_beta) else f"{p_beta:.1f}"
    print(f"\n--- alpha={p_alpha}, beta={beta_str}  ({label}) ---")

    for trial in range(PREVIEW_N_SAMPLES):
        preview_rng = np.random.default_rng(SEED * 100 + trial)
        result = augment_prompt_with_demos(
            prompt_tokens=tokenizer.tokenize_prompt(sequent),
            rule_bank=bank,
            tokenizer=tokenizer,
            rng=preview_rng,
            src_layer=int(src_layer),
            ants=ants,
            max_n_demos=PREVIEW_N_DEMOS,
            min_n_demos=PREVIEW_N_DEMOS,
            max_unify_solutions=MAX_UNIFY_SOLUTIONS,
            include_oracle=False,
            oracle_rule=sampled.step_rules[0],
            demo_distribution=DEMO_DISTRIBUTION,
            demo_distribution_alpha=p_alpha,
            goal_atom=goal_atom,
            demo_ranked=True,
            demo_unique=True,
            demo_ranking_beta=p_beta,
        )
        rank_str = " ".join(f"{r}" for r in result.demo_ranks)
        print(f"  sample {trial}: ranks=[{rank_str}]")
        for d_idx, (schema, inst, rank) in enumerate(
            zip(result.demo_schemas, result.demo_instances, result.demo_ranks)
        ):
            oracle_tag = " *oracle*" if schema == sampled.step_rules[0] else ""
            print(f"    [{rank}] {schema.statement_text}  ->  {inst}{oracle_tag}")

# <codecell>  Optional: matplotlib heatmap
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    beta_labels = [("inf" if not np.isfinite(b) else f"{b:.2f}") for b in BETA_VALUES]
    im = ax.imshow(
        pivot.values,
        aspect="auto",
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
    )
    ax.set_xticks(range(len(beta_labels)))
    ax.set_xticklabels(beta_labels)
    ax.set_yticks(range(len(ALPHA_VALUES)))
    ax.set_yticklabels([f"{a:.1f}" for a in ALPHA_VALUES])
    ax.set_xlabel("beta (ranking strength)")
    ax.set_ylabel("alpha (retrieval quality)")
    ax.set_title("Mean Kendall tau: demo order vs ideal descending rank")
    fig.colorbar(im, ax=ax, label="Kendall tau")

    for i in range(len(ALPHA_VALUES)):
        for j in range(len(BETA_VALUES)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=8)

    out_path = Path(__file__).parent / "set" / "alpha_beta_tau_heatmap.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nHeatmap saved to {out_path}")
    plt.close(fig)
except ImportError:
    print("\nmatplotlib not available; skipping heatmap.")
