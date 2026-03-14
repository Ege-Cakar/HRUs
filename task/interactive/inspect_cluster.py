# <codecell>
"""Interactive inspection of the `cluster` demo distribution pipeline.

Walks through: rule bank construction, rule classification by rank,
ranking vector construction, Spearman's footrule distance matrix,
k-medoids clustering, medoid selection, and final demo assignment.
Compares cluster vs. zipf_per_rule demo distributions on a full task.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from task.layer_fol.demos import (
    _classify_rules_by_rank,
    _batch_build_ranking_vectors,
    _spearman_footrule_distance_matrix,
    _k_medoids,
    _sample_cluster,
    _sample_fresh_query_at_layer,
    augment_prompt_with_demos,
    sample_ranked_demos,
)
from task.layer_fol import FOLLayerTask, print_task_preview
from task.layer_gen.util.fol_rule_bank import (
    build_random_fol_rule_bank,
    sample_fol_problem,
)

# --- Config ---
BANK_CFG = {
    "seed": 42,
    "n_layers": 3,
    "predicates_per_layer": (3, 9, 3),
    "rules_per_transition": (9, 9),
    "arity_max": 0,
    "arity_min": 0,
    "vars_per_rule_max": 1,
    "k_in_min": 1,
    "k_in_max": 1,
    "k_out_min": 1,
    "k_out_max": 1,
    "constants": ("a", "b", "c", "d"),
}
CLUSTER_CFG = {
    "n_demos": 4,
    "cluster_n_samples": 8,
    "cluster_k": 1,
    "alpha": 10,
    "cluster_base_dist": "zipf_per_rule",
    "cluster_unselected_rank": None,  # defaults to n_demos + 1 = 5
    "include_oracle": True,
    "demo_ranked": True,
    "demo_unique": True,
}
PROBLEM_CFG = {
    "distance": 1,
    "initial_ant_max": 1,
    "max_unify_solutions": 128,
}
TASK_PREVIEW_CFG = {"n_examples": 5, "batch_size": 4}
WALKTHROUGH_SEED = 7


def format_ranking_matrix(
    vectors: np.ndarray,
    rule_labels: list[str],
    row_labels: list[str],
) -> str:
    """Aligned text table for ranking vectors (rows=contexts, cols=rules)."""
    col_width = max(len(lbl) for lbl in rule_labels + ["ctx"])
    col_width = max(col_width, 4) + 1
    header = "ctx".ljust(col_width) + "".join(
        lbl.rjust(col_width) for lbl in rule_labels
    )
    lines = [header, "-" * len(header)]
    for i, row_label in enumerate(row_labels):
        row = row_label.ljust(col_width) + "".join(
            str(int(vectors[i, j])).rjust(col_width)
            for j in range(vectors.shape[1])
        )
        lines.append(row)
    return "\n".join(lines)


def format_dist_matrix(matrix: np.ndarray, labels: list[str]) -> str:
    """Aligned text table for a distance matrix."""
    col_width = max(len(lbl) for lbl in labels + [""]) + 1
    col_width = max(col_width, 5)
    header = "".ljust(col_width) + "".join(
        lbl.rjust(col_width) for lbl in labels
    )
    lines = [header, "-" * len(header)]
    for i, lbl in enumerate(labels):
        row = lbl.ljust(col_width) + "".join(
            str(int(matrix[i, j])).rjust(col_width)
            for j in range(matrix.shape[1])
        )
        lines.append(row)
    return "\n".join(lines)


# <codecell>
# --- Cell 1: Build and Print Rule Bank ---

rng = np.random.default_rng(BANK_CFG["seed"])
bank = build_random_fol_rule_bank(
    n_layers=BANK_CFG["n_layers"],
    predicates_per_layer=BANK_CFG["predicates_per_layer"],
    rules_per_transition=BANK_CFG["rules_per_transition"],
    arity_max=BANK_CFG["arity_max"],
    arity_min=BANK_CFG["arity_min"],
    vars_per_rule_max=BANK_CFG["vars_per_rule_max"],
    k_in_min=BANK_CFG["k_in_min"],
    k_in_max=BANK_CFG["k_in_max"],
    k_out_min=BANK_CFG["k_out_min"],
    k_out_max=BANK_CFG["k_out_max"],
    constants=BANK_CFG["constants"],
    rng=rng,
)

print("=" * 60)
print("RULE BANK")
print("=" * 60)
for layer in range(bank.n_layers):
    preds = bank.predicates_for_layer(layer)
    print(f"  layer {layer}: {preds}")

for src_layer in sorted(bank.transitions):
    rules = bank.transition_rules(src_layer)
    print(f"\n  transition {src_layer} -> {src_layer + 1}  ({len(rules)} rules)")
    for idx, rule in enumerate(rules):
        print(f"    [{idx}] {rule.statement_text}")


# <codecell>
# --- Cell 2: Sample a Problem, Classify Rules by Rank ---

walkthrough_rng = np.random.default_rng(WALKTHROUGH_SEED)
sampled = sample_fol_problem(
    bank=bank,
    distance=PROBLEM_CFG["distance"],
    initial_ant_max=PROBLEM_CFG["initial_ant_max"],
    rng=walkthrough_rng,
    max_unify_solutions=PROBLEM_CFG["max_unify_solutions"],
)

src_layer = sampled.step_layers[0]
ants = sampled.step_ants[0]
goal_atom = sampled.goal_atom
oracle_rule = sampled.step_rule_templates[0]

print("\n" + "=" * 60)
print("SAMPLED PROBLEM")
print("=" * 60)
print(f"  src_layer:  {src_layer}")
print(f"  antecedents: {[str(a) for a in ants]}")
print(f"  goal_atom:   {goal_atom}")
print(f"  oracle_rule: {oracle_rule.statement_text}")

rules = list(bank.transition_rules(src_layer))
ranked = _classify_rules_by_rank(
    rules=rules,
    ants=ants,
    goal_atom=goal_atom,
    rule_bank=bank,
    max_unify_solutions=PROBLEM_CFG["max_unify_solutions"],
)

rank_descriptions = {
    1: "applicable + goal-reachable",
    2: "applicable + NOT goal-reachable",
    3: "NOT applicable + goal-reachable",
    4: "NOT applicable + NOT goal-reachable",
}
print("\nRULE CLASSIFICATION BY RANK")
for rank in (1, 2, 3, 4):
    desc = rank_descriptions[rank]
    print(f"  rank {rank} ({desc}):")
    for rule in ranked[rank]:
        print(f"    {rule.statement_text}")
    if not ranked[rank]:
        print(f"    (none)")

rule_to_idx = {rule: i for i, rule in enumerate(rules)}
print("\nrule_to_idx mapping:")
for rule, idx in rule_to_idx.items():
    print(f"  [{idx}] {rule.statement_text}")


# <codecell>
# --- Cell 3: Run Clustering Step-by-Step ---

n_demos = CLUSTER_CFG["n_demos"]
cluster_n_samples = CLUSTER_CFG["cluster_n_samples"]
cluster_k = CLUSTER_CFG["cluster_k"]
alpha = CLUSTER_CFG["alpha"]
cluster_base_dist = CLUSTER_CFG["cluster_base_dist"]
include_oracle = CLUSTER_CFG["include_oracle"]
demo_ranked = CLUSTER_CFG["demo_ranked"]
demo_unique = CLUSTER_CFG["demo_unique"]
n_rules = len(rules)
unselected_rank = (
    int(CLUSTER_CFG["cluster_unselected_rank"])
    if CLUSTER_CFG["cluster_unselected_rank"] is not None
    else n_demos + 1
)

# Reset walkthrough RNG for reproducible step-by-step
walkthrough_rng = np.random.default_rng(WALKTHROUGH_SEED)
# Advance past the sample_fol_problem calls
_throwaway = sample_fol_problem(
    bank=bank,
    distance=PROBLEM_CFG["distance"],
    initial_ant_max=PROBLEM_CFG["initial_ant_max"],
    rng=walkthrough_rng,
    max_unify_solutions=PROBLEM_CFG["max_unify_solutions"],
)

print("\n" + "=" * 60)
print("CLUSTERING STEP-BY-STEP")
print("=" * 60)

# 3a: Sample N fresh queries and classify each; actual query is the +1
total_samples = cluster_n_samples + 1
all_schemas: list[list] = []
all_ranks: list[list] = []
fresh_query_info: list[str] = []
for i in range(cluster_n_samples):
    fresh_query = _sample_fresh_query_at_layer(
        rule_bank=bank,
        src_layer=src_layer,
        distance=PROBLEM_CFG["distance"],
        initial_ant_max=PROBLEM_CFG["initial_ant_max"],
        rng=walkthrough_rng,
        max_unify_solutions=PROBLEM_CFG["max_unify_solutions"],
    )
    if fresh_query is not None:
        fresh_ants, fresh_goal = fresh_query
        fresh_ranked = _classify_rules_by_rank(
            rules=rules,
            ants=fresh_ants,
            goal_atom=fresh_goal,
            rule_bank=bank,
            max_unify_solutions=PROBLEM_CFG["max_unify_solutions"],
        )
        fresh_query_info.append(
            f"ants={[str(a) for a in fresh_ants]}, goal={fresh_goal}"
        )
    else:
        fresh_ranked = ranked
        fresh_query_info.append("(fallback to actual query)")

    schemas_i, ranks_i = sample_ranked_demos(
        ranked_rules=fresh_ranked,
        rng=walkthrough_rng,
        n_demos=n_demos,
        demo_distribution=cluster_base_dist,
        alpha=alpha,
        include_oracle=False,
        oracle_rule=None,
        demo_ranked=demo_ranked,
        demo_unique=demo_unique,
    )
    all_schemas.append(schemas_i)
    all_ranks.append(ranks_i)

# Actual query (the +1)
actual_schemas, actual_ranks = sample_ranked_demos(
    ranked_rules=ranked,
    rng=walkthrough_rng,
    n_demos=n_demos,
    demo_distribution=cluster_base_dist,
    alpha=alpha,
    include_oracle=False,
    oracle_rule=None,
    demo_ranked=demo_ranked,
    demo_unique=demo_unique,
)
all_schemas.append(actual_schemas)
all_ranks.append(actual_ranks)

n_candidates = len(all_schemas) - 1
print(f"\n3a: Sampled {n_candidates} candidate contexts + 1 query")
for i in range(n_candidates):
    rule_strs = [r.statement_text for r in all_schemas[i]]
    print(f"  cand[{i}]: query={fresh_query_info[i]}")
    print(f"           rules={rule_strs}  ranks={all_ranks[i]}")
query_idx = n_candidates
rule_strs = [r.statement_text for r in all_schemas[query_idx]]
print(f"  QUERY: rules={rule_strs}  ranks={all_ranks[query_idx]}")

# 3b: Build ranking vectors
n_total = n_candidates + 1
ranking_vectors = _batch_build_ranking_vectors(
    all_schemas=all_schemas,
    all_ranks=all_ranks,
    rule_to_idx=rule_to_idx,
    n_rules=n_rules,
    unselected_rank=unselected_rank,
    rng=walkthrough_rng,
)

rule_labels = [f"r{idx}" for idx in range(n_rules)]
row_labels = [f"cand[{i}]" for i in range(n_candidates)] + ["QUERY"]
print(f"\n3b: Ranking vectors (unselected_rank={unselected_rank})")
print(format_ranking_matrix(ranking_vectors, rule_labels, row_labels))

# 3c: Compute footrule distance matrix
candidate_vectors = ranking_vectors[:n_candidates]
dist_matrix = _spearman_footrule_distance_matrix(candidate_vectors)

cand_labels = [f"c{i}" for i in range(n_candidates)]
print(f"\n3c: Spearman's footrule distance matrix ({n_candidates}x{n_candidates})")
print(format_dist_matrix(dist_matrix, cand_labels))
print(f"  symmetric: {np.allclose(dist_matrix, dist_matrix.T)}")
print(f"  diagonal zeros: {np.all(np.diag(dist_matrix) == 0)}")

# 3d: Run k-medoids
effective_k = min(cluster_k, n_candidates)
medoids = _k_medoids(dist_matrix, effective_k, walkthrough_rng)
assignments = np.argmin(dist_matrix[medoids], axis=0)

print(f"\n3d: k-medoids (k={effective_k})")
print(f"  medoid indices: {medoids.tolist()}")
for ci in range(effective_k):
    members = np.where(assignments == ci)[0]
    print(f"  cluster {ci} (medoid=cand[{medoids[ci]}]): members={members.tolist()}")

# 3e: Find closest medoid for query
query_vector = ranking_vectors[n_candidates]
medoid_vectors = candidate_vectors[medoids]
dists_to_medoids = np.sum(np.abs(medoid_vectors - query_vector[None, :]), axis=1)
closest_idx = int(np.argmin(dists_to_medoids))
closest_medoid_cand = medoids[closest_idx]

print(f"\n3e: Closest medoid for query")
for ci in range(effective_k):
    print(f"  dist to medoid cand[{medoids[ci]}]: {int(dists_to_medoids[ci])}")
print(f"  -> selected medoid: cand[{closest_medoid_cand}]")
print(f"  -> demos: {[r.statement_text for r in all_schemas[closest_medoid_cand]]}")
print(f"  -> ranks (medoid's query): {all_ranks[closest_medoid_cand]}")
# Re-evaluate ranks against the actual query
actual_rank_for_rule = {}
for rank, rule_list in ranked.items():
    for rule in rule_list:
        actual_rank_for_rule[rule] = rank
manual_actual_ranks = [actual_rank_for_rule.get(s, 4) for s in all_schemas[closest_medoid_cand]]
print(f"  -> ranks (actual query):   {manual_actual_ranks}")


# <codecell>
# --- Cell 4: Verify Against `_sample_cluster` ---

print("\n" + "=" * 60)
print("VERIFICATION: _sample_cluster()")
print("=" * 60)

# Re-seed to match the walkthrough
verify_rng = np.random.default_rng(WALKTHROUGH_SEED)
# Advance past the same sample_fol_problem call
_throwaway2 = sample_fol_problem(
    bank=bank,
    distance=PROBLEM_CFG["distance"],
    initial_ant_max=PROBLEM_CFG["initial_ant_max"],
    rng=verify_rng,
    max_unify_solutions=PROBLEM_CFG["max_unify_solutions"],
)

verify_schemas, verify_ranks = _sample_cluster(
    rule_bank=bank,
    src_layer=src_layer,
    ants=ants,
    max_unify_solutions=PROBLEM_CFG["max_unify_solutions"],
    rng=verify_rng,
    n_demos=n_demos,
    include_oracle=include_oracle,
    oracle_rule=oracle_rule,
    alpha=alpha,
    goal_atom=goal_atom,
    demo_ranked=demo_ranked,
    demo_unique=demo_unique,
    cluster_n_samples=cluster_n_samples,
    cluster_k=cluster_k,
    cluster_base_dist=cluster_base_dist,
    cluster_unselected_rank=CLUSTER_CFG["cluster_unselected_rank"],
    distance=PROBLEM_CFG["distance"],
    initial_ant_max=PROBLEM_CFG["initial_ant_max"],
)

manual_schemas = all_schemas[closest_medoid_cand]
# Use actual-query ranks to match the fixed _sample_cluster behavior
manual_ranks = manual_actual_ranks

verify_texts = [r.statement_text for r in verify_schemas]
manual_texts = [r.statement_text for r in manual_schemas]

print(f"  _sample_cluster schemas: {verify_texts}")
print(f"  _sample_cluster ranks:   {verify_ranks}")
print(f"  manual walkthrough schemas: {manual_texts}")
print(f"  manual walkthrough ranks:   {manual_ranks}")

match = (verify_texts == manual_texts) and (verify_ranks == manual_ranks)
if match:
    print("  MATCH: manual walkthrough matches _sample_cluster output")
else:
    print("  MISMATCH: manual walkthrough does NOT match _sample_cluster output")
    print("    (This may be due to RNG state divergence from _build_ranking_vector tiebreakers)")


# <codecell>
# --- Cell 5: Full Task Preview — Cluster Distribution ---

print("\n" + "=" * 60)
print("FULL TASK PREVIEW: cluster distribution")
print("=" * 60)

cluster_task = FOLLayerTask(
    mode="online",
    task_split="none",
    demo_distribution="cluster",
    distance_range=(1, 1),
    batch_size=TASK_PREVIEW_CFG["batch_size"],
    seed=100,
    n_layers=BANK_CFG["n_layers"],
    predicates_per_layer=BANK_CFG["predicates_per_layer"],
    rules_per_transition=BANK_CFG["rules_per_transition"],
    arity_max=BANK_CFG["arity_max"],
    arity_min=BANK_CFG["arity_min"],
    vars_per_rule_max=BANK_CFG["vars_per_rule_max"],
    k_in_min=BANK_CFG["k_in_min"],
    k_in_max=BANK_CFG["k_in_max"],
    k_out_min=BANK_CFG["k_out_min"],
    k_out_max=BANK_CFG["k_out_max"],
    constants=BANK_CFG["constants"],
    initial_ant_max=PROBLEM_CFG["initial_ant_max"],
    max_n_demos=CLUSTER_CFG["n_demos"],
    min_n_demos=CLUSTER_CFG["n_demos"],
    include_oracle=CLUSTER_CFG["include_oracle"],
    demo_distribution_alpha=CLUSTER_CFG["alpha"],
    demo_ranked=CLUSTER_CFG["demo_ranked"],
    demo_unique=CLUSTER_CFG["demo_unique"],
    cluster_n_samples=CLUSTER_CFG["cluster_n_samples"],
    cluster_k=CLUSTER_CFG["cluster_k"],
    cluster_base_dist=CLUSTER_CFG["cluster_base_dist"],
    cluster_unselected_rank=CLUSTER_CFG["cluster_unselected_rank"],
    online_prefetch=False,
)

print_task_preview(cluster_task, role="cluster", n_examples=TASK_PREVIEW_CFG["n_examples"])


# <codecell>
# --- Cell 6: Comparison — Cluster vs. Baseline (zipf_per_rule) ---

print("\n" + "=" * 60)
print("COMPARISON: cluster vs. zipf_per_rule")
print("=" * 60)

baseline_task = FOLLayerTask(
    mode="online",
    task_split="none",
    demo_distribution="zipf_per_rule",
    distance_range=(1, 1),
    batch_size=TASK_PREVIEW_CFG["batch_size"],
    seed=100,
    n_layers=BANK_CFG["n_layers"],
    predicates_per_layer=BANK_CFG["predicates_per_layer"],
    rules_per_transition=BANK_CFG["rules_per_transition"],
    arity_max=BANK_CFG["arity_max"],
    arity_min=BANK_CFG["arity_min"],
    vars_per_rule_max=BANK_CFG["vars_per_rule_max"],
    k_in_min=BANK_CFG["k_in_min"],
    k_in_max=BANK_CFG["k_in_max"],
    k_out_min=BANK_CFG["k_out_min"],
    k_out_max=BANK_CFG["k_out_max"],
    constants=BANK_CFG["constants"],
    initial_ant_max=PROBLEM_CFG["initial_ant_max"],
    max_n_demos=CLUSTER_CFG["n_demos"],
    min_n_demos=CLUSTER_CFG["n_demos"],
    include_oracle=CLUSTER_CFG["include_oracle"],
    demo_distribution_alpha=CLUSTER_CFG["alpha"],
    demo_ranked=CLUSTER_CFG["demo_ranked"],
    demo_unique=CLUSTER_CFG["demo_unique"],
    online_prefetch=False,
)

print_task_preview(baseline_task, role="zipf_per_rule", n_examples=TASK_PREVIEW_CFG["n_examples"])

# --- Aggregate rank distributions over ~50 samples ---
# Call augment_prompt_with_demos directly to get demo_ranks, since
# task._sample_online_record() discards them for the non-fresh path.
from collections import Counter
from task.layer_gen.util.fol_rule_bank import FOLSequent

N_AGG = 50
agg_rng = np.random.default_rng(999)


def _collect_demo_ranks(task, dist_name, n_samples, rng):
    """Sample problems and call augment_prompt_with_demos to get demo_ranks."""
    rank_counts: Counter = Counter()
    task_bank = task.rule_bank
    task_tok = task.tokenizer
    for _ in range(n_samples):
        try:
            sampled = sample_fol_problem(
                bank=task_bank,
                distance=1,
                initial_ant_max=PROBLEM_CFG["initial_ant_max"],
                rng=rng,
                max_unify_solutions=PROBLEM_CFG["max_unify_solutions"],
            )
        except RuntimeError:
            continue
        ants = sampled.step_ants[0]
        src_layer = sampled.step_layers[0]
        sequent = FOLSequent(ants=ants, cons=sampled.goal_atom)
        prompt = task_tok.tokenize_prompt(sequent)
        result = augment_prompt_with_demos(
            prompt_tokens=prompt,
            rule_bank=task_bank,
            tokenizer=task_tok,
            rng=rng,
            src_layer=src_layer,
            ants=ants,
            max_n_demos=CLUSTER_CFG["n_demos"],
            min_n_demos=CLUSTER_CFG["n_demos"],
            max_unify_solutions=PROBLEM_CFG["max_unify_solutions"],
            include_oracle=CLUSTER_CFG["include_oracle"],
            oracle_rule=sampled.step_rules[0],
            demo_distribution=dist_name,
            demo_distribution_alpha=CLUSTER_CFG["alpha"],
            goal_atom=sampled.goal_atom,
            demo_ranked=CLUSTER_CFG["demo_ranked"],
            demo_unique=CLUSTER_CFG["demo_unique"],
            cluster_n_samples=CLUSTER_CFG["cluster_n_samples"],
            cluster_k=CLUSTER_CFG["cluster_k"],
            cluster_base_dist=CLUSTER_CFG["cluster_base_dist"],
            cluster_unselected_rank=CLUSTER_CFG["cluster_unselected_rank"],
            cluster_distance=PROBLEM_CFG["distance"],
            cluster_initial_ant_max=PROBLEM_CFG["initial_ant_max"],
        )
        for r in result.demo_ranks:
            rank_counts[int(r)] += 1
    return rank_counts


cluster_rank_counts = _collect_demo_ranks(cluster_task, "cluster", N_AGG, agg_rng)
baseline_rank_counts = _collect_demo_ranks(baseline_task, "zipf_per_rule", N_AGG, agg_rng)

all_ranks_seen = sorted(set(cluster_rank_counts) | set(baseline_rank_counts))
print(f"\nDEMO RANK FREQUENCY (over {N_AGG} samples)")
print(f"  {'rank':<6} {'cluster':>10} {'zipf_per_rule':>15}")
print(f"  {'-' * 6} {'-' * 10} {'-' * 15}")
for rank in all_ranks_seen:
    c_count = cluster_rank_counts.get(rank, 0)
    b_count = baseline_rank_counts.get(rank, 0)
    print(f"  {rank:<6} {c_count:>10} {b_count:>15}")

cluster_total = sum(cluster_rank_counts.values())
baseline_total = sum(baseline_rank_counts.values())
print(f"  {'total':<6} {cluster_total:>10} {baseline_total:>15}")

cluster_task.close()
baseline_task.close()

# %%
