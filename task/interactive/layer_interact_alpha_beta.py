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
    _build_per_rule_pool_and_weights,
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
N_LAYERS = 2
PREDICATES_PER_LAYER = 8
RULES_PER_TRANSITION = 32
ARITY_MIN = 1
ARITY_MAX = 3
VARS_PER_RULE_MAX = 4
CONSTANTS = ("a", "b", "c", "d")
K_IN_MIN = 1
K_IN_MAX = 3
K_OUT_MIN = 1
K_OUT_MAX = 3
INITIAL_ANT_MAX = 3
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
