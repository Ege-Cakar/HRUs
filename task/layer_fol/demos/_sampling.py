"""Demo sampling strategies (uniform, Zipf, Zipf-per-rule)."""

from __future__ import annotations

import numpy as np

from task.layer_gen.util.fol_rule_bank import FOLAtom, FOLLayerRule, FOLRuleBank

from ._ranking import _classify_rules_by_rank
from ._unify import (
    _collect_applicable_demo_schemas,
    _find_matching_demo_schema_for_rule,
    _find_oracle_schema_or_raise,
)


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------

def _resolve_demo_ranking_beta(
    demo_ranking_beta: float | None,
    demo_ranked: bool,
) -> float:
    """Resolve *demo_ranking_beta*, defaulting from *demo_ranked* flag."""
    if demo_ranking_beta is not None:
        return float(demo_ranking_beta)
    return float('inf') if demo_ranked else 0.0


def _rank_order_demos(
    schemas: list[FOLLayerRule],
    ranks: list[int],
    rng: np.random.Generator,
    beta: float,
) -> tuple[list[FOLLayerRule], list[int]]:
    """Reorder demos using a Plackett-Luce (Gumbel noise) ranking model.

    - ``beta = 0`` → random shuffle (equivalent to ``demo_ranked=False``)
    - ``beta = inf`` → deterministic descending sort (equivalent to ``demo_ranked=True``)
    - Intermediate ``beta`` → noisy partial ordering via ``rank + Gumbel / beta``
    """
    n = len(schemas)
    if n <= 1:
        return schemas, ranks
    if beta == 0.0:
        order = rng.permutation(n)
    elif not np.isfinite(beta):
        paired = sorted(zip(ranks, schemas), key=lambda x: x[0], reverse=True)
        return [s for _, s in paired], [r for r, _ in paired]
    else:
        noisy = np.array(ranks, dtype=np.float64) + rng.gumbel(size=n) / beta
        order = np.argsort(-noisy)
    return [schemas[int(i)] for i in order], [ranks[int(i)] for i in order]


# ------------------------------------------------------------------
# Pool / weight builders
# ------------------------------------------------------------------

def _build_per_rule_pool_and_weights(
    ranked_rules: dict[int, list[FOLLayerRule]],
    alpha: float,
    exclude_rank_1: bool,
) -> tuple[list[tuple[FOLLayerRule, int]], np.ndarray]:
    """Build flat (rule, rank) list and per-rule Zipf weights."""
    pool: list[tuple[FOLLayerRule, int]] = []
    for rank in sorted(ranked_rules):
        if exclude_rank_1 and rank == 1:
            continue
        for rule in ranked_rules[rank]:
            pool.append((rule, rank))
    if not pool:
        return pool, np.array([])
    weights = np.array([1.0 / (rank ** alpha) for _, rank in pool])
    weights /= weights.sum()
    return pool, weights


def _build_rank_level_pool_and_weights(
    ranked_rules: dict[int, list[FOLLayerRule]],
    alpha: float,
    sampling_ranks: list[int],
) -> tuple[list[tuple[FOLLayerRule, int]], np.ndarray]:
    """Build flat (rule, rank) list with rank-level Zipf weights spread across rules."""
    pool: list[tuple[FOLLayerRule, int]] = []
    weights_list: list[float] = []
    for k in sampling_ranks:
        rules_in_rank = ranked_rules[k]
        per_rule_w = (1.0 / (k ** alpha)) / len(rules_in_rank)
        for rule in rules_in_rank:
            pool.append((rule, k))
            weights_list.append(per_rule_w)
    if not pool:
        return pool, np.array([])
    w = np.array(weights_list)
    w /= w.sum()
    return pool, w


# ------------------------------------------------------------------
# Shared weighted-pool sampling (Phase 0d)
# ------------------------------------------------------------------

def _sample_from_weighted_pool(
    *,
    pool: list[tuple[FOLLayerRule, int]],
    weights: np.ndarray,
    rng: np.random.Generator,
    n_demos: int,
    include_oracle: bool,
    oracle_rule: FOLLayerRule | None,
    ranked_rules: dict[int, list[FOLLayerRule]],
    demo_unique: bool,
    demo_ranking_beta: float,
) -> tuple[list[FOLLayerRule], list[int]]:
    """Sample demos from a pre-built weighted pool, handling oracle and reorder."""
    if not include_oracle:
        if not pool:
            return [], []
        if demo_unique:
            n = min(int(n_demos), len(pool))
            indices = rng.choice(len(pool), size=n, replace=False, p=weights)
            sampled_schemas = [pool[int(i)][0] for i in indices]
            sampled_ranks = [pool[int(i)][1] for i in indices]
        else:
            sampled_schemas: list[FOLLayerRule] = []
            sampled_ranks: list[int] = []
            for _ in range(int(n_demos)):
                idx = int(rng.choice(len(pool), p=weights))
                rule, rank = pool[idx]
                sampled_schemas.append(rule)
                sampled_ranks.append(rank)
        sampled_schemas, sampled_ranks = _rank_order_demos(
            sampled_schemas, sampled_ranks, rng, demo_ranking_beta,
        )
        return sampled_schemas, sampled_ranks

    # --- Oracle path ---
    if oracle_rule is None:
        raise ValueError("include_oracle=True requires oracle_rule.")

    oracle_schema = _find_oracle_schema_or_raise(ranked_rules, oracle_rule)

    sampled_schemas = [oracle_schema]
    sampled_ranks = [1]

    if int(n_demos) > 1 and pool:
        if demo_unique:
            remaining_indices = [
                i for i, (rule, _) in enumerate(pool)
                if rule is not oracle_schema
            ]
            if remaining_indices:
                remaining_pool = [pool[i] for i in remaining_indices]
                remaining_weights = np.array([weights[i] for i in remaining_indices])
                remaining_weights /= remaining_weights.sum()
                n = min(int(n_demos) - 1, len(remaining_pool))
                indices = rng.choice(
                    len(remaining_pool), size=n, replace=False, p=remaining_weights,
                )
                for i in indices:
                    sampled_schemas.append(remaining_pool[int(i)][0])
                    sampled_ranks.append(remaining_pool[int(i)][1])
        else:
            for _ in range(int(n_demos) - 1):
                idx = int(rng.choice(len(pool), p=weights))
                rule, rank = pool[idx]
                sampled_schemas.append(rule)
                sampled_ranks.append(rank)

    sampled_schemas, sampled_ranks = _rank_order_demos(
        sampled_schemas, sampled_ranks, rng, demo_ranking_beta,
    )
    return sampled_schemas, sampled_ranks


# ------------------------------------------------------------------
# Flat-list uniform sampling
# ------------------------------------------------------------------

def _sample_demo_schemas(
    *,
    rng: np.random.Generator,
    schemas: list[FOLLayerRule],
    n_demos: int,
    include_oracle: bool = False,
    oracle_rule: FOLLayerRule | None = None,
    demo_unique: bool = False,
) -> list[FOLLayerRule]:
    if n_demos < 1 or not schemas:
        return []

    if not include_oracle:
        if demo_unique:
            n = min(int(n_demos), len(schemas))
            picks = rng.choice(len(schemas), size=n, replace=False)
        else:
            picks = rng.integers(0, len(schemas), size=int(n_demos))
        return [schemas[int(idx)] for idx in picks]

    if oracle_rule is None:
        raise ValueError("include_oracle=True requires oracle_rule.")

    oracle_schema = _find_matching_demo_schema_for_rule(
        schemas=schemas,
        oracle_rule=oracle_rule,
    )
    if oracle_schema is None:
        raise RuntimeError("Oracle rule schema was not found among applicable demo schemas.")

    sampled = [oracle_schema]
    if int(n_demos) > 1:
        if demo_unique:
            oracle_idx = next(
                i for i, s in enumerate(schemas) if s is oracle_schema
            )
            remaining = [s for i, s in enumerate(schemas) if i != oracle_idx]
            if remaining:
                n = min(int(n_demos) - 1, len(remaining))
                picks = rng.choice(len(remaining), size=n, replace=False)
                sampled.extend(remaining[int(idx)] for idx in picks)
        else:
            picks = rng.integers(0, len(schemas), size=int(n_demos) - 1)
            sampled.extend(schemas[int(idx)] for idx in picks)
    order = rng.permutation(len(sampled))
    return [sampled[int(idx)] for idx in order]


# ------------------------------------------------------------------
# Zipf sampling (rank-level and per-rule variants)
# ------------------------------------------------------------------

def _sample_demo_schemas_zipf(
    *,
    rng: np.random.Generator,
    ranked_rules: dict[int, list[FOLLayerRule]],
    n_demos: int,
    alpha: float,
    include_oracle: bool,
    oracle_rule: FOLLayerRule | None,
    headless: bool = False,
    demo_ranked: bool = True,
    demo_unique: bool = False,
    demo_ranking_beta: float | None = None,
) -> tuple[list[FOLLayerRule], list[int]]:
    non_empty_ranks = sorted(k for k, v in ranked_rules.items() if v)
    if not non_empty_ranks:
        return [], []

    beta = _resolve_demo_ranking_beta(demo_ranking_beta, demo_ranked)

    sampling_ranks = [k for k in non_empty_ranks if k != 1] if headless else list(non_empty_ranks)

    pool, weights = _build_rank_level_pool_and_weights(ranked_rules, alpha, sampling_ranks)
    return _sample_from_weighted_pool(
        pool=pool,
        weights=weights,
        rng=rng,
        n_demos=n_demos,
        include_oracle=include_oracle,
        oracle_rule=oracle_rule,
        ranked_rules=ranked_rules,
        demo_unique=demo_unique,
        demo_ranking_beta=beta,
    )


def _sample_demo_schemas_zipf_per_rule(
    *,
    rng: np.random.Generator,
    ranked_rules: dict[int, list[FOLLayerRule]],
    n_demos: int,
    alpha: float,
    include_oracle: bool,
    oracle_rule: FOLLayerRule | None,
    headless: bool = False,
    demo_ranked: bool = True,
    demo_unique: bool = False,
    demo_ranking_beta: float | None = None,
) -> tuple[list[FOLLayerRule], list[int]]:
    non_empty_ranks = sorted(k for k, v in ranked_rules.items() if v)
    if not non_empty_ranks:
        return [], []

    beta = _resolve_demo_ranking_beta(demo_ranking_beta, demo_ranked)

    pool, weights = _build_per_rule_pool_and_weights(
        ranked_rules, alpha, exclude_rank_1=headless,
    )
    return _sample_from_weighted_pool(
        pool=pool,
        weights=weights,
        rng=rng,
        n_demos=n_demos,
        include_oracle=include_oracle,
        oracle_rule=oracle_rule,
        ranked_rules=ranked_rules,
        demo_unique=demo_unique,
        demo_ranking_beta=beta,
    )


# ------------------------------------------------------------------
# Public dispatcher
# ------------------------------------------------------------------

def sample_ranked_demos(
    *,
    ranked_rules: dict[int, list[FOLLayerRule]],
    rng: np.random.Generator,
    n_demos: int,
    demo_distribution: str,
    alpha: float,
    include_oracle: bool = False,
    oracle_rule: FOLLayerRule | None = None,
    demo_ranked: bool = True,
    demo_unique: bool = True,
    demo_ranking_beta: float | None = None,
) -> tuple[list[FOLLayerRule], list[int]]:
    """Dispatch demo sampling on pre-classified rules by distribution name.

    Accepts the same rank-based distribution names as
    ``augment_prompt_with_demos`` (``"zipf"``, ``"zipf_headless"``,
    ``"zipf_per_rule"``, ``"zipf_per_rule_headless"``, ``"full_rank"``).

    Returns ``(sampled_schemas, sampled_ranks)``.
    """
    demo_distribution = str(demo_distribution)
    if demo_distribution in ("zipf", "zipf_headless"):
        return _sample_demo_schemas_zipf(
            rng=rng,
            ranked_rules=ranked_rules,
            n_demos=n_demos,
            alpha=alpha,
            include_oracle=include_oracle,
            oracle_rule=oracle_rule,
            headless=(demo_distribution == "zipf_headless"),
            demo_ranked=demo_ranked,
            demo_unique=demo_unique,
            demo_ranking_beta=demo_ranking_beta,
        )
    elif demo_distribution in ("zipf_per_rule", "zipf_per_rule_headless"):
        return _sample_demo_schemas_zipf_per_rule(
            rng=rng,
            ranked_rules=ranked_rules,
            n_demos=n_demos,
            alpha=alpha,
            include_oracle=include_oracle,
            oracle_rule=oracle_rule,
            headless=(demo_distribution == "zipf_per_rule_headless"),
            demo_ranked=demo_ranked,
            demo_unique=demo_unique,
            demo_ranking_beta=demo_ranking_beta,
        )
    elif demo_distribution == "full_rank":
        beta = _resolve_demo_ranking_beta(demo_ranking_beta, demo_ranked)
        return _sample_demo_schemas_full_rank(
            ranked_rules=ranked_rules,
            n_demos=n_demos,
            rng=rng,
            demo_ranking_beta=beta,
            include_oracle=include_oracle,
            oracle_rule=oracle_rule,
        )
    else:
        raise ValueError(
            f"sample_ranked_demos does not support demo_distribution={demo_distribution!r}; "
            f"expected 'zipf', 'zipf_headless', 'zipf_per_rule', 'zipf_per_rule_headless', "
            f"or 'full_rank'."
        )


# ------------------------------------------------------------------
# Thin wrappers that classify then sample
# ------------------------------------------------------------------

def _sample_uniform(
    *,
    rule_bank: FOLRuleBank,
    src_layer: int,
    ants: tuple[FOLAtom, ...],
    max_unify_solutions: int,
    rng: np.random.Generator,
    n_demos: int,
    include_oracle: bool,
    oracle_rule: FOLLayerRule | None,
    demo_unique: bool = False,
) -> tuple[list[FOLLayerRule], list[int]]:
    schemas = _collect_applicable_demo_schemas(
        rule_bank=rule_bank,
        src_layer=int(src_layer),
        ants=ants,
        max_unify_solutions=int(max_unify_solutions),
    )
    if not schemas:
        return [], []
    sampled = _sample_demo_schemas(
        rng=rng,
        schemas=schemas,
        n_demos=n_demos,
        include_oracle=include_oracle,
        oracle_rule=oracle_rule,
        demo_unique=demo_unique,
    )
    return sampled, []


def _sample_zipf_ranked(
    *,
    rule_bank: FOLRuleBank,
    src_layer: int,
    ants: tuple[FOLAtom, ...],
    max_unify_solutions: int,
    rng: np.random.Generator,
    n_demos: int,
    include_oracle: bool,
    oracle_rule: FOLLayerRule | None,
    alpha: float,
    goal_atom: FOLAtom,
    headless: bool = False,
    demo_ranked: bool = True,
    demo_unique: bool = False,
    demo_ranking_beta: float | None = None,
    per_rule: bool = False,
) -> tuple[list[FOLLayerRule], list[int]]:
    """Classify rules then sample with zipf or zipf_per_rule distribution."""
    rules = list(rule_bank.transition_rules(int(src_layer)))
    ranked = _classify_rules_by_rank(
        rules=rules,
        ants=ants,
        goal_atom=goal_atom,
        rule_bank=rule_bank,
        max_unify_solutions=int(max_unify_solutions),
    )
    if per_rule:
        dist_name = "zipf_per_rule_headless" if headless else "zipf_per_rule"
    else:
        dist_name = "zipf_headless" if headless else "zipf"
    return sample_ranked_demos(
        ranked_rules=ranked,
        rng=rng,
        n_demos=n_demos,
        demo_distribution=dist_name,
        alpha=alpha,
        include_oracle=include_oracle,
        oracle_rule=oracle_rule,
        demo_ranked=demo_ranked,
        demo_unique=demo_unique,
        demo_ranking_beta=demo_ranking_beta,
    )


# ------------------------------------------------------------------
# Full-rank Plackett-Luce sampling
# ------------------------------------------------------------------

def _sample_demo_schemas_full_rank(
    *,
    ranked_rules: dict[int, list[FOLLayerRule]],
    n_demos: int,
    rng: np.random.Generator,
    demo_ranking_beta: float,
    include_oracle: bool,
    oracle_rule: FOLLayerRule | None,
) -> tuple[list[FOLLayerRule], list[int]]:
    """Order all rules via Plackett-Luce on inverted ranks, take top k.

    Scores are ``5 - rank`` so rank-1 rules get score 4 (best).
    The PL ordering IS the final presentation order — no separate
    ``_rank_order_demos`` call is needed.

    - ``beta = 0`` → random permutation
    - ``beta = inf`` → deterministic worst-first, best-last (best closest to query)
    - Intermediate ``beta`` → noisy: ``score + Gumbel(0,1) / beta``
    """
    # Flatten all rules with their ranks
    pool: list[tuple[FOLLayerRule, int]] = []
    for rank in sorted(ranked_rules):
        for rule in ranked_rules[rank]:
            pool.append((rule, rank))
    if not pool:
        return [], []

    n = len(pool)
    k = min(int(n_demos), n)
    if k <= 0:
        return [], []

    scores = np.array([5 - rank for _, rank in pool], dtype=np.float64)

    # Plackett-Luce via Gumbel trick
    if demo_ranking_beta == 0.0:
        order = rng.permutation(n)
    elif not np.isfinite(demo_ranking_beta):
        # Deterministic: sort by score descending (stable to preserve
        # within-rank order from the sorted ranked_rules iteration)
        order = np.argsort(-scores, kind="stable")
    else:
        noisy = scores + rng.gumbel(size=n) / demo_ranking_beta
        order = np.argsort(-noisy)

    # Take top k positions (preserving PL order)
    selected_indices = list(order[:k])

    # Oracle injection
    if include_oracle:
        if oracle_rule is None:
            raise ValueError("include_oracle=True requires oracle_rule.")
        oracle_schema = _find_oracle_schema_or_raise(ranked_rules, oracle_rule)
        # Find oracle's position in pool
        oracle_pool_idx = None
        for i, (rule, _) in enumerate(pool):
            if rule is oracle_schema:
                oracle_pool_idx = i
                break
        if oracle_pool_idx is not None and oracle_pool_idx not in selected_indices:
            # Replace weakest selected item (last position) with oracle
            selected_indices[-1] = oracle_pool_idx
            # Re-sort selected positions to preserve PL order
            # (sort by their position in the PL ordering)
            order_list = list(order)
            selected_indices.sort(key=lambda idx: order_list.index(idx))

    selected_indices = selected_indices[::-1]
    schemas = [pool[i][0] for i in selected_indices]
    ranks = [pool[i][1] for i in selected_indices]
    return schemas, ranks


def _sample_full_rank(
    *,
    rule_bank: FOLRuleBank,
    src_layer: int,
    ants: tuple[FOLAtom, ...],
    max_unify_solutions: int,
    rng: np.random.Generator,
    n_demos: int,
    include_oracle: bool,
    oracle_rule: FOLLayerRule | None,
    goal_atom: FOLAtom,
    demo_ranked: bool = True,
    demo_ranking_beta: float | None = None,
) -> tuple[list[FOLLayerRule], list[int]]:
    """Classify rules then sample with full_rank Plackett-Luce distribution."""
    beta = _resolve_demo_ranking_beta(demo_ranking_beta, demo_ranked)
    rules = list(rule_bank.transition_rules(int(src_layer)))
    ranked = _classify_rules_by_rank(
        rules=rules,
        ants=ants,
        goal_atom=goal_atom,
        rule_bank=rule_bank,
        max_unify_solutions=int(max_unify_solutions),
    )
    return _sample_demo_schemas_full_rank(
        ranked_rules=ranked,
        n_demos=n_demos,
        rng=rng,
        demo_ranking_beta=beta,
        include_oracle=include_oracle,
        oracle_rule=oracle_rule,
    )
