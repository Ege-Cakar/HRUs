"""Cluster-based demo selection via k-medoids on Spearman's footrule."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.spatial.distance import cdist as _scipy_cdist

from task.layer_gen.util.fol_rule_bank import FOLAtom, FOLLayerRule, FOLRuleBank, sample_fol_problem

from ._ranking import _classify_rules_by_rank, _precompute_reachable_sets
from ._sampling import (
    _resolve_demo_ranking_beta,
    _rank_order_demos,
    sample_ranked_demos,
)
from ._unify import _find_oracle_schema_or_raise


def _spearman_footrule_distance_matrix(
    ranking_vectors: np.ndarray,
) -> np.ndarray:
    """Pairwise Spearman's footrule: D(A,B) = sum|rank_A[i] - rank_B[i]|."""
    return _scipy_cdist(
        ranking_vectors.astype(np.float64),
        ranking_vectors.astype(np.float64),
        metric="cityblock",
    )


def _k_medoids(
    dist_matrix: np.ndarray,
    k: int,
    rng: np.random.Generator,
    max_iter: int = 100,
) -> np.ndarray:
    """Return indices of k medoids via PAM (Partitioning Around Medoids)."""
    n = dist_matrix.shape[0]
    medoids = rng.choice(n, size=k, replace=False)
    for _ in range(max_iter):
        assignments = np.argmin(dist_matrix[medoids], axis=0)
        changed = False
        for ci in range(k):
            members = np.where(assignments == ci)[0]
            if len(members) == 0:
                continue
            costs = dist_matrix[np.ix_(members, members)].sum(axis=1)
            best = members[np.argmin(costs)]
            if best != medoids[ci]:
                medoids[ci] = best
                changed = True
        if not changed:
            break
    return medoids


def _batch_build_ranking_vectors(
    *,
    all_schemas: list[list[FOLLayerRule]],
    all_ranks: list[list[int]],
    rule_to_idx: dict[FOLLayerRule, int],
    n_rules: int,
    unselected_rank: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build ranking vectors for multiple contexts in one call."""
    n = len(all_schemas)
    vectors = np.full((n, n_rules), unselected_rank, dtype=np.int64)
    for ctx_i in range(n):
        schemas_i = all_schemas[ctx_i]
        ranks_i = all_ranks[ctx_i]
        if not schemas_i:
            continue
        tiebreakers = rng.random(len(schemas_i))
        paired = list(zip(schemas_i, ranks_i, tiebreakers))
        paired.sort(key=lambda x: (-x[1], x[2]))
        for position_0, (rule, _, _) in enumerate(paired):
            idx = rule_to_idx.get(rule)
            if idx is not None:
                vectors[ctx_i, idx] = position_0 + 1
    return vectors


def _sample_fresh_query_at_layer(
    *,
    rule_bank: FOLRuleBank,
    src_layer: int,
    distance: int,
    initial_ant_max: int,
    rng: np.random.Generator,
    max_attempts: int = 128,
    max_unify_solutions: int = 128,
) -> tuple[tuple[FOLAtom, ...], FOLAtom] | None:
    """Generate a fresh ``(ants, goal_atom)`` at *src_layer*.

    Samples a new proof via ``sample_fol_problem`` and extracts the step
    whose layer matches *src_layer*.  Returns ``None`` on failure.
    """
    try:
        sampled = sample_fol_problem(
            bank=rule_bank,
            distance=int(distance),
            initial_ant_max=int(initial_ant_max),
            rng=rng,
            max_attempts=int(max_attempts),
            max_unify_solutions=int(max_unify_solutions),
        )
    except RuntimeError:
        return None

    for step_idx, layer in enumerate(sampled.step_layers):
        if int(layer) == int(src_layer):
            return tuple(sampled.step_ants[step_idx]), sampled.goal_atom
    return None


def _precompute_cluster_candidate_rankings(
    *,
    rule_bank: FOLRuleBank,
    src_layer: int,
    rules: list[FOLLayerRule],
    actual_ranked: dict[int, list[FOLLayerRule]],
    rng: np.random.Generator,
    cluster_n_samples: int,
    max_unify_solutions: int = 64,
    distance: int | None = None,
    initial_ant_max: int | None = None,
    server=None,
) -> list[dict[int, list[FOLLayerRule]]]:
    """Precompute ranked-rule dicts for cluster candidate queries.

    Samples ``cluster_n_samples`` fresh queries via
    ``_sample_fresh_query_at_layer`` and classifies all *rules* for each.
    Returns a list of ranked-rule dictionaries (one per candidate).  Falls
    back to *actual_ranked* when a fresh sample fails.

    This is the expensive part of ``_sample_cluster`` that is independent of
    ``alpha``, ``n_demos``, and ``cluster_k``, and can therefore be computed
    once and reused across those sweep dimensions.

    When *server* is provided (a ``ClusterPrecomputeClient``), computation
    is offloaded to a subprocess pool for parallelism.
    """
    can_sample_fresh = (distance is not None and initial_ant_max is not None)

    # If a server is available and we can sample fresh queries, offload.
    if server is not None and can_sample_fresh:
        seed = int(rng.integers(1 << 63))
        return server.precompute(
            rule_bank=rule_bank,
            src_layer=src_layer,
            cluster_n_samples=cluster_n_samples,
            seed=seed,
            max_unify_solutions=max_unify_solutions,
            distance=distance,
            initial_ant_max=initial_ant_max,
            fallback_ranked=actual_ranked,
        )

    # Sequential path with reachability caching.
    reachable_sets = _precompute_reachable_sets(rules, rule_bank) if can_sample_fresh else None
    candidate_rankings: list[dict[int, list[FOLLayerRule]]] = []

    for _ in range(int(cluster_n_samples)):
        fresh_ranked = None
        if can_sample_fresh:
            fresh_query = _sample_fresh_query_at_layer(
                rule_bank=rule_bank,
                src_layer=int(src_layer),
                distance=int(distance),
                initial_ant_max=int(initial_ant_max),
                rng=rng,
                max_unify_solutions=int(max_unify_solutions),
            )
            if fresh_query is not None:
                fresh_ants, fresh_goal = fresh_query
                fresh_ranked = _classify_rules_by_rank(
                    rules=rules,
                    ants=fresh_ants,
                    goal_atom=fresh_goal,
                    rule_bank=rule_bank,
                    max_unify_solutions=int(max_unify_solutions),
                    reachable_sets=reachable_sets,
                )
        candidate_rankings.append(
            fresh_ranked if fresh_ranked is not None else actual_ranked
        )

    return candidate_rankings


def _sample_cluster_from_precomputed(
    *,
    candidate_rankings: list[dict[int, list[FOLLayerRule]]],
    actual_ranked: dict[int, list[FOLLayerRule]],
    rules: list[FOLLayerRule],
    rule_to_idx: dict[FOLLayerRule, int],
    rng: np.random.Generator,
    n_demos: int,
    alpha: float,
    cluster_k: int,
    cluster_base_dist: str,
    cluster_unselected_rank: int | None,
    demo_ranked: bool,
    demo_unique: bool,
    include_oracle: bool = False,
    oracle_rule: FOLLayerRule | None = None,
    demo_ranking_beta: float | None = None,
) -> tuple[list[FOLLayerRule], list[int]]:
    """Sample demos using precomputed candidate rankings.

    Equivalent to ``_sample_cluster`` but skips the expensive fresh-query
    sampling.  *candidate_rankings* should come from
    ``_precompute_cluster_candidate_rankings``.
    """
    n_rules = len(rules)
    if n_rules == 0:
        return [], []

    unselected_rank = (
        int(cluster_unselected_rank)
        if cluster_unselected_rank is not None
        else n_demos + 1
    )

    # --- Per-candidate demo sampling (cheap) ---
    all_schemas: list[list[FOLLayerRule]] = []
    all_ranks: list[list[int]] = []

    for candidate_ranked in candidate_rankings:
        schemas_i, ranks_i = sample_ranked_demos(
            ranked_rules=candidate_ranked,
            rng=rng,
            n_demos=n_demos,
            demo_distribution=str(cluster_base_dist),
            alpha=alpha,
            include_oracle=False,
            oracle_rule=None,
            demo_ranked=demo_ranked,
            demo_unique=demo_unique,
        )
        all_schemas.append(schemas_i)
        all_ranks.append(ranks_i)

    # Actual query sample (the +1)
    actual_schemas, actual_ranks = sample_ranked_demos(
        ranked_rules=actual_ranked,
        rng=rng,
        n_demos=n_demos,
        demo_distribution=str(cluster_base_dist),
        alpha=alpha,
        include_oracle=False,
        oracle_rule=None,
        demo_ranked=demo_ranked,
        demo_unique=demo_unique,
    )
    all_schemas.append(actual_schemas)
    all_ranks.append(actual_ranks)

    # --- Build ranking vectors & select ---
    n_candidates = len(all_schemas) - 1
    effective_k = int(cluster_k)
    if n_candidates < effective_k:
        warnings.warn(
            f"cluster candidates ({n_candidates}) < cluster_k ({effective_k}); "
            f"clamping k to {n_candidates}."
        )
        effective_k = max(1, n_candidates)

    if n_candidates < 1:
        selected_schemas = actual_schemas
        selected_ranks = actual_ranks
    else:
        ranking_vectors = _batch_build_ranking_vectors(
            all_schemas=all_schemas,
            all_ranks=all_ranks,
            rule_to_idx=rule_to_idx,
            n_rules=n_rules,
            unselected_rank=unselected_rank,
            rng=rng,
        )

        candidate_vectors = ranking_vectors[:n_candidates]
        dist_matrix = _spearman_footrule_distance_matrix(candidate_vectors)
        medoids = _k_medoids(dist_matrix, effective_k, rng)

        query_vector = ranking_vectors[n_candidates]
        medoid_vectors = candidate_vectors[medoids]
        dists_to_medoids = np.sum(
            np.abs(medoid_vectors - query_vector[None, :]),
            axis=1,
        )
        closest_medoid_idx = medoids[int(np.argmin(dists_to_medoids))]

        selected_schemas = all_schemas[closest_medoid_idx]
        # Re-evaluate ranks against the *actual* query, not the medoid's query.
        actual_rank_for_rule: dict[FOLLayerRule, int] = {}
        for rank, rule_list in actual_ranked.items():
            for rule in rule_list:
                actual_rank_for_rule[rule] = rank
        selected_ranks = [
            actual_rank_for_rule.get(s, 4) for s in selected_schemas
        ]

    # --- Oracle injection ---
    if include_oracle:
        if oracle_rule is None:
            raise ValueError("include_oracle=True requires oracle_rule.")
        oracle_schema = _find_oracle_schema_or_raise(actual_ranked, oracle_rule)
        if oracle_schema not in selected_schemas:
            if len(selected_schemas) >= n_demos:
                selected_schemas[-1] = oracle_schema
                selected_ranks[-1] = 1
            else:
                selected_schemas.append(oracle_schema)
                selected_ranks.append(1)
        beta = _resolve_demo_ranking_beta(demo_ranking_beta, demo_ranked)
        selected_schemas, selected_ranks = _rank_order_demos(
            selected_schemas, selected_ranks, rng, beta,
        )

    return selected_schemas, selected_ranks


def _build_ranks_matrix(
    all_ranked: list[dict[int, list[FOLLayerRule]]],
    rule_to_idx: dict[FOLLayerRule, int],
    n_rules: int,
) -> np.ndarray:
    """Build (n_contexts × n_rules) matrix of per-rule rank assignments."""
    mat = np.full((len(all_ranked), n_rules), 4, dtype=np.float64)
    for i, ranked_dict in enumerate(all_ranked):
        for rank, rule_list in ranked_dict.items():
            for rule in rule_list:
                j = rule_to_idx.get(rule)
                if j is not None:
                    mat[i, j] = rank
    return mat


def _batch_cluster_select(
    *,
    candidate_rankings: list[dict[int, list[FOLLayerRule]]],
    actual_ranked: dict[int, list[FOLLayerRule]],
    rules: list[FOLLayerRule],
    rule_to_idx: dict[FOLLayerRule, int],
    rng: np.random.Generator,
    alpha_values: list[float],
    n_demos_values: list[int],
    cluster_k_values: list[int],
    cluster_base_dist: str,
    cluster_unselected_rank: int | None,
    demo_unique: bool,
) -> dict[tuple[int, float, int], tuple[list[FOLLayerRule], list[int]]]:
    """Batch cluster selection over a parameter sweep.

    Equivalent to calling ``_sample_cluster_from_precomputed`` for each
    ``(cluster_k, alpha, n_demos)`` combination, but:

    * Shares ranking vectors and distance matrix across ``cluster_k`` values.
    * Uses vectorised numpy weight computation instead of per-call Python
      pool-building.

    Returns ``{(cluster_k, alpha, n_demos): (selected_schemas, selected_ranks)}``.
    """
    n_rules = len(rules)
    if n_rules == 0:
        return {
            (ck, a, nd): ([], [])
            for ck in cluster_k_values
            for a in alpha_values
            for nd in n_demos_values
        }

    n_candidates = len(candidate_rankings)
    n_total = n_candidates + 1  # +1 for actual query
    headless = cluster_base_dist in ("zipf_headless", "zipf_per_rule_headless")

    # Build ranks matrix once: ranks_mat[i, j] = rank of rule j for context i.
    # Contexts 0..n_candidates-1 are candidates; context n_candidates is actual.
    all_ranked = list(candidate_rankings) + [actual_ranked]
    ranks_mat = _build_ranks_matrix(all_ranked, rule_to_idx, n_rules)

    results: dict[tuple[int, float, int], tuple[list[FOLLayerRule], list[int]]] = {}

    for alpha in alpha_values:
        # Vectorised weight computation: (n_total, n_rules)
        weights = np.power(ranks_mat, -alpha)
        if headless:
            weights[ranks_mat == 1] = 0.0
        row_sums = weights.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        weights /= row_sums

        for n_demos in n_demos_values:
            unselected_rank = (
                int(cluster_unselected_rank)
                if cluster_unselected_rank is not None
                else n_demos + 1
            )

            # --- Sample rule indices per context ---
            sampled_idx = np.empty((n_total, n_demos), dtype=np.intp)
            for i in range(n_total):
                if demo_unique:
                    n_elig = int((weights[i] > 0).sum())
                    n = min(n_demos, n_elig)
                else:
                    n = n_demos
                sampled_idx[i, :n] = rng.choice(
                    n_rules, size=n,
                    replace=not demo_unique, p=weights[i],
                )
                if n < n_demos:
                    sampled_idx[i, n:] = sampled_idx[i, n - 1]

            # Look up ranks of sampled rules: (n_total, n_demos)
            sampled_rnk = ranks_mat[
                np.arange(n_total)[:, None], sampled_idx
            ].astype(np.int64)

            # --- Build ranking vectors via vectorised sort ---
            tiebreakers = rng.random((n_total, n_demos))
            sort_key = -sampled_rnk.astype(np.float64) + tiebreakers
            order = np.argsort(sort_key, axis=1)

            vectors = np.full(
                (n_total, n_rules), unselected_rank, dtype=np.int64,
            )
            positions = np.arange(1, n_demos + 1)
            for i in range(n_total):
                vectors[i, sampled_idx[i, order[i]]] = positions

            # --- Distance matrix (shared across ck) ---
            candidate_vectors = vectors[:n_candidates]
            query_vector = vectors[n_candidates]

            dist_matrix = None
            if n_candidates >= 1:
                dist_matrix = _spearman_footrule_distance_matrix(
                    candidate_vectors,
                )

            # --- Per-ck: k-medoids + closest selection ---
            for ck in cluster_k_values:
                effective_k = min(ck, max(1, n_candidates))

                if n_candidates < 1 or dist_matrix is None:
                    closest = n_candidates
                else:
                    medoids = _k_medoids(dist_matrix, effective_k, rng)
                    dists = np.sum(
                        np.abs(
                            candidate_vectors[medoids]
                            - query_vector[None, :]
                        ),
                        axis=1,
                    )
                    closest = medoids[int(np.argmin(dists))]

                sel_schemas = [rules[int(j)] for j in sampled_idx[closest]]
                # Use the *actual* query's ranks, not the candidate's own ranks.
                sel_ranks = [
                    int(ranks_mat[n_candidates, int(j)])
                    for j in sampled_idx[closest]
                ]
                results[(ck, alpha, n_demos)] = (sel_schemas, sel_ranks)

    return results


def _sample_cluster(
    *,
    rule_bank: FOLRuleBank,
    src_layer: int,
    ants: tuple[FOLAtom, ...] = (),
    max_unify_solutions: int = 64,
    rng: np.random.Generator,
    n_demos: int,
    include_oracle: bool,
    oracle_rule: FOLLayerRule | None,
    alpha: float,
    goal_atom: FOLAtom | None = None,
    demo_ranked: bool,
    demo_unique: bool,
    cluster_n_samples: int,
    cluster_k: int,
    cluster_base_dist: str,
    cluster_unselected_rank: int | None,
    ranked_rules: dict[int, list[FOLLayerRule]] | None = None,
    distance: int | None = None,
    initial_ant_max: int | None = None,
    precomputed_cluster_candidates: list | None = None,
    demo_ranking_beta: float | None = None,
) -> tuple[list[FOLLayerRule], list[int]]:
    """Sample demos via k-medoids clustering on Spearman's footrule distance.

    Each of the N candidate samples derives its ranking from a freshly
    sampled query (via ``_sample_fresh_query_at_layer``), producing genuine
    diversity across query types rather than just sampling noise.

    If *ranked_rules* is provided, rule classification for the actual query
    is skipped.
    """
    rules = list(rule_bank.transition_rules(int(src_layer)))
    if not rules:
        return [], []

    # Actual query classification
    if ranked_rules is not None:
        actual_ranked = ranked_rules
    else:
        actual_ranked = _classify_rules_by_rank(
            rules=rules,
            ants=ants,
            goal_atom=goal_atom,
            rule_bank=rule_bank,
            max_unify_solutions=int(max_unify_solutions),
        )

    rule_to_idx = {rule: i for i, rule in enumerate(rules)}

    # Use precomputed candidate rankings if available, otherwise compute them.
    if precomputed_cluster_candidates is not None:
        candidate_rankings = precomputed_cluster_candidates
    else:
        candidate_rankings = _precompute_cluster_candidate_rankings(
            rule_bank=rule_bank,
            src_layer=int(src_layer),
            rules=rules,
            actual_ranked=actual_ranked,
            rng=rng,
            cluster_n_samples=int(cluster_n_samples),
            max_unify_solutions=int(max_unify_solutions),
            distance=distance,
            initial_ant_max=initial_ant_max,
        )

    # Sample demos & select via clustering (cheap)
    return _sample_cluster_from_precomputed(
        candidate_rankings=candidate_rankings,
        actual_ranked=actual_ranked,
        rules=rules,
        rule_to_idx=rule_to_idx,
        rng=rng,
        n_demos=n_demos,
        alpha=alpha,
        cluster_k=cluster_k,
        cluster_base_dist=cluster_base_dist,
        cluster_unselected_rank=cluster_unselected_rank,
        demo_ranked=demo_ranked,
        demo_unique=demo_unique,
        include_oracle=include_oracle,
        oracle_rule=oracle_rule,
        demo_ranking_beta=demo_ranking_beta,
    )
