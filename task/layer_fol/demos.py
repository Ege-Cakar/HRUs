"""Demo augmentation utilities for layered FOL tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy.spatial.distance import cdist as _scipy_cdist

import warnings

from task.layer_gen.util import tokenize_layer_fol
from task.layer_gen.util.fol_rule_bank import (
    FOLAtom,
    FOLLayerRule,
    FOLRuleBank,
    sample_fol_problem,
)
from .eval_inputs import extract_prompt_info_from_row_tokens


def _is_variable(token: str) -> bool:
    return token.startswith("x")


def _unify_template_atom_with_ground(
    template: FOLAtom,
    ground: FOLAtom,
    subst: dict[str, str],
) -> dict[str, str] | None:
    if template.predicate != ground.predicate:
        return None
    if len(template.args) != len(ground.args):
        return None

    out = dict(subst)
    for templ_term, ground_term in zip(template.args, ground.args):
        if _is_variable(templ_term):
            bound = out.get(templ_term)
            if bound is None:
                out[templ_term] = ground_term
            elif bound != ground_term:
                return None
        elif templ_term != ground_term:
            return None
    return out


@dataclass(frozen=True)
class FOLDemoAugmentationResult:
    prompt_tokens: list[int]
    demo_schemas: tuple[FOLLayerRule, ...]
    demo_instances: tuple[str, ...]
    demo_ranks: tuple[int, ...] = ()


def augment_prompt_with_demos(
    *,
    prompt_tokens: list[int],
    rule_bank: FOLRuleBank,
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer,
    rng: np.random.Generator,
    src_layer: int,
    ants: tuple[FOLAtom, ...],
    max_n_demos: int,
    min_n_demos: int,
    max_unify_solutions: int,
    include_oracle: bool = False,
    oracle_rule: FOLLayerRule | None = None,
    demo_distribution: str = "uniform",
    demo_distribution_alpha: float = 1.0,
    goal_atom: FOLAtom | None = None,
    demo_ranked: bool = True,
    demo_all: bool = False,
    demo_unique: bool = True,
    cluster_n_samples: int = 100,
    cluster_k: int = 5,
    cluster_base_dist: str = "zipf_per_rule",
    cluster_unselected_rank: int | None = None,
    cluster_distance: int | None = None,
    cluster_initial_ant_max: int | None = None,
    precomputed_cluster_candidates: list | None = None,
    demo_ranking_beta: float | None = None,
) -> FOLDemoAugmentationResult:
    max_n_demos = int(max_n_demos)
    min_n_demos = int(min_n_demos)
    include_oracle = bool(include_oracle)
    demo_distribution = str(demo_distribution)
    demo_distribution_alpha = float(demo_distribution_alpha)
    demo_ranked = bool(demo_ranked)
    demo_all = bool(demo_all)
    demo_unique = bool(demo_unique)
    if demo_ranking_beta is None:
        demo_ranking_beta = float('inf') if demo_ranked else 0.0

    _empty = FOLDemoAugmentationResult(
        prompt_tokens=list(int(tok) for tok in prompt_tokens),
        demo_schemas=(),
        demo_instances=(),
    )

    if demo_all:
        all_rules = list(rule_bank.transition_rules(int(src_layer)))
        if not all_rules:
            return _empty
        demo_statements = [
            _instantiate_demo_schema_with_random_constants(
                rule=rule,
                constants=rule_bank.constants,
                rng=rng,
            )
            for rule in all_rules
        ]
        augmented_prompt = _prepend_demo_statements_to_prompt(
            prompt_tokens=prompt_tokens,
            demo_statements=demo_statements,
            tokenizer=tokenizer,
        )
        return FOLDemoAugmentationResult(
            prompt_tokens=augmented_prompt,
            demo_schemas=tuple(all_rules),
            demo_instances=tuple(str(s) for s in demo_statements),
            demo_ranks=(),
        )

    if demo_distribution not in {"uniform", "zipf", "zipf_headless", "zipf_per_rule", "zipf_per_rule_headless", "cluster"}:
        raise ValueError(
            f"demo_distribution must be 'uniform', 'zipf', 'zipf_headless', "
            f"'zipf_per_rule', 'zipf_per_rule_headless', or 'cluster', "
            f"got {demo_distribution!r}"
        )
    if demo_distribution in {"zipf", "zipf_headless", "zipf_per_rule", "zipf_per_rule_headless", "cluster"} and goal_atom is None:
        raise ValueError(f"demo_distribution={demo_distribution!r} requires goal_atom.")
    if min_n_demos > max_n_demos:
        raise ValueError(
            f"min_n_demos must be <= max_n_demos, got {min_n_demos} > {max_n_demos}"
        )
    if include_oracle and max_n_demos < 1:
        raise ValueError("include_oracle=True requires max_n_demos >= 1.")
    if include_oracle and oracle_rule is None:
        raise ValueError("include_oracle=True requires oracle_rule.")

    if max_n_demos <= 0:
        return _empty

    n_demos = int(rng.integers(min_n_demos, max_n_demos + 1))
    if include_oracle and n_demos < 1:
        raise ValueError("include_oracle=True requires sampling at least one demo.")
    if n_demos <= 0:
        return _empty

    if demo_distribution == "cluster":
        sampled_schemas, sampled_ranks = _sample_cluster(
            rule_bank=rule_bank,
            src_layer=int(src_layer),
            ants=ants,
            max_unify_solutions=int(max_unify_solutions),
            rng=rng,
            n_demos=n_demos,
            include_oracle=include_oracle,
            oracle_rule=oracle_rule,
            alpha=demo_distribution_alpha,
            goal_atom=goal_atom,
            demo_ranked=demo_ranked,
            demo_unique=demo_unique,
            cluster_n_samples=int(cluster_n_samples),
            cluster_k=int(cluster_k),
            cluster_base_dist=str(cluster_base_dist),
            cluster_unselected_rank=cluster_unselected_rank,
            distance=cluster_distance,
            initial_ant_max=cluster_initial_ant_max,
            precomputed_cluster_candidates=precomputed_cluster_candidates,
            demo_ranking_beta=demo_ranking_beta,
        )
    elif demo_distribution == "uniform":
        sampled_schemas, sampled_ranks = _sample_uniform(
            rule_bank=rule_bank,
            src_layer=int(src_layer),
            ants=ants,
            max_unify_solutions=int(max_unify_solutions),
            rng=rng,
            n_demos=n_demos,
            include_oracle=include_oracle,
            oracle_rule=oracle_rule,
            demo_unique=demo_unique,
        )
    elif demo_distribution in {"zipf_per_rule", "zipf_per_rule_headless"}:
        sampled_schemas, sampled_ranks = _sample_zipf_per_rule(
            rule_bank=rule_bank,
            src_layer=int(src_layer),
            ants=ants,
            max_unify_solutions=int(max_unify_solutions),
            rng=rng,
            n_demos=n_demos,
            include_oracle=include_oracle,
            oracle_rule=oracle_rule,
            alpha=demo_distribution_alpha,
            goal_atom=goal_atom,
            headless=(demo_distribution == "zipf_per_rule_headless"),
            demo_ranked=demo_ranked,
            demo_unique=demo_unique,
            demo_ranking_beta=demo_ranking_beta,
        )
    else:
        sampled_schemas, sampled_ranks = _sample_zipf(
            rule_bank=rule_bank,
            src_layer=int(src_layer),
            ants=ants,
            max_unify_solutions=int(max_unify_solutions),
            rng=rng,
            n_demos=n_demos,
            include_oracle=include_oracle,
            oracle_rule=oracle_rule,
            alpha=demo_distribution_alpha,
            goal_atom=goal_atom,
            headless=(demo_distribution == "zipf_headless"),
            demo_ranked=demo_ranked,
            demo_unique=demo_unique,
            demo_ranking_beta=demo_ranking_beta,
        )

    if not sampled_schemas:
        return _empty

    demo_statements = [
        _instantiate_demo_schema_with_random_constants(
            rule=schema,
            constants=rule_bank.constants,
            rng=rng,
        )
        for schema in sampled_schemas
    ]

    augmented_prompt = _prepend_demo_statements_to_prompt(
        prompt_tokens=prompt_tokens,
        demo_statements=demo_statements,
        tokenizer=tokenizer,
    )
    return FOLDemoAugmentationResult(
        prompt_tokens=augmented_prompt,
        demo_schemas=tuple(sampled_schemas),
        demo_instances=tuple(str(statement) for statement in demo_statements),
        demo_ranks=tuple(sampled_ranks),
    )


def _augment_prompt_with_demos(
    *,
    prompt_tokens: list[int],
    rule_bank: FOLRuleBank,
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer,
    rng: np.random.Generator,
    src_layer: int,
    ants: tuple[FOLAtom, ...],
    max_n_demos: int,
    min_n_demos: int,
    max_unify_solutions: int,
    include_oracle: bool = False,
    oracle_rule: FOLLayerRule | None = None,
    demo_distribution: str = "uniform",
    demo_distribution_alpha: float = 1.0,
    goal_atom: FOLAtom | None = None,
    demo_ranked: bool = True,
    demo_all: bool = False,
    demo_unique: bool = True,
    cluster_n_samples: int = 100,
    cluster_k: int = 5,
    cluster_base_dist: str = "zipf_per_rule",
    cluster_unselected_rank: int | None = None,
    cluster_distance: int | None = None,
    cluster_initial_ant_max: int | None = None,
    demo_ranking_beta: float | None = None,
) -> list[int]:
    return augment_prompt_with_demos(
        prompt_tokens=prompt_tokens,
        rule_bank=rule_bank,
        tokenizer=tokenizer,
        rng=rng,
        src_layer=src_layer,
        ants=ants,
        max_n_demos=max_n_demos,
        min_n_demos=min_n_demos,
        max_unify_solutions=max_unify_solutions,
        include_oracle=include_oracle,
        oracle_rule=oracle_rule,
        demo_distribution=demo_distribution,
        demo_distribution_alpha=demo_distribution_alpha,
        goal_atom=goal_atom,
        demo_ranked=demo_ranked,
        demo_all=demo_all,
        demo_unique=demo_unique,
        cluster_n_samples=cluster_n_samples,
        cluster_k=cluster_k,
        cluster_base_dist=cluster_base_dist,
        cluster_unselected_rank=cluster_unselected_rank,
        cluster_distance=cluster_distance,
        cluster_initial_ant_max=cluster_initial_ant_max,
        demo_ranking_beta=demo_ranking_beta,
    ).prompt_tokens


def _collect_applicable_demo_schemas(
    *,
    rule_bank: FOLRuleBank,
    src_layer: int,
    ants: tuple[FOLAtom, ...],
    max_unify_solutions: int,
) -> list[FOLLayerRule]:
    schemas: list[FOLLayerRule] = []
    seen_schema_keys: set[str] = set()
    ground_ants = tuple(ants)
    for rule in rule_bank.transition_rules(int(src_layer)):
        substitutions = _find_lhs_substitutions_for_facts(
            lhs=rule.lhs,
            facts=ground_ants,
            max_solutions=int(max_unify_solutions),
        )
        if not any(_subst_binds_rhs_variables(rule=rule, subst=subst) for subst in substitutions):
            continue

        schema_key = str(rule.statement_text)
        if schema_key in seen_schema_keys:
            continue
        seen_schema_keys.add(schema_key)
        schemas.append(rule)
    return schemas


def _subst_binds_rhs_variables(*, rule: FOLLayerRule, subst: dict[str, str]) -> bool:
    rhs_vars = {
        term
        for atom in rule.rhs
        for term in atom.args
        if _is_variable(term)
    }
    return rhs_vars.issubset(set(subst))


def _find_lhs_substitutions_for_facts(
    *,
    lhs: tuple[FOLAtom, ...],
    facts: tuple[FOLAtom, ...],
    max_solutions: int,
) -> list[dict[str, str]]:
    solutions: list[dict[str, str]] = []

    def _search(idx: int, subst: dict[str, str]) -> None:
        if len(solutions) >= max_solutions:
            return
        if idx >= len(lhs):
            solutions.append(dict(subst))
            return

        templ = lhs[idx]
        for fact in facts:
            maybe = _unify_template_atom_with_ground(templ, fact, subst)
            if maybe is None:
                continue
            _search(idx + 1, maybe)
            if len(solutions) >= max_solutions:
                return

    _search(0, {})
    return solutions


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


def _find_matching_demo_schema_for_rule(
    *,
    schemas: list[FOLLayerRule],
    oracle_rule: FOLLayerRule,
) -> FOLLayerRule | None:
    for schema in schemas:
        if _find_demo_schema_instantiation(
            schema=schema,
            ground_rule=oracle_rule,
        ) is not None:
            return schema
    return None


def _find_demo_schema_instantiation(
    *,
    schema: FOLLayerRule,
    ground_rule: FOLLayerRule,
) -> dict[str, str] | None:
    if int(schema.src_layer) != int(ground_rule.src_layer):
        return None
    if int(schema.dst_layer) != int(ground_rule.dst_layer):
        return None
    if len(schema.lhs) != len(ground_rule.lhs):
        return None
    if len(schema.rhs) != len(ground_rule.rhs):
        return None

    subst: dict[str, str] = {}
    for templ_atom, ground_atom in zip(schema.lhs, ground_rule.lhs):
        maybe = _unify_template_atom_with_ground(templ_atom, ground_atom, subst)
        if maybe is None:
            return None
        subst = maybe
    for templ_atom, ground_atom in zip(schema.rhs, ground_rule.rhs):
        maybe = _unify_template_atom_with_ground(templ_atom, ground_atom, subst)
        if maybe is None:
            return None
        subst = maybe
    return subst


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


def _sample_zipf(
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
) -> tuple[list[FOLLayerRule], list[int]]:
    rules = list(rule_bank.transition_rules(int(src_layer)))
    ranked = _classify_rules_by_rank(
        rules=rules,
        ants=ants,
        goal_atom=goal_atom,
        rule_bank=rule_bank,
        max_unify_solutions=int(max_unify_solutions),
    )
    return sample_ranked_demos(
        ranked_rules=ranked,
        rng=rng,
        n_demos=n_demos,
        demo_distribution="zipf_headless" if headless else "zipf",
        alpha=alpha,
        include_oracle=include_oracle,
        oracle_rule=oracle_rule,
        demo_ranked=demo_ranked,
        demo_unique=demo_unique,
        demo_ranking_beta=demo_ranking_beta,
    )


def _rhs_predicates(rule: FOLLayerRule) -> set[str]:
    return {atom.predicate for atom in rule.rhs}


def _lhs_predicates(rule: FOLLayerRule) -> set[str]:
    return {atom.predicate for atom in rule.lhs}


def _is_goal_reachable_from_rule_rhs(
    rule: FOLLayerRule,
    goal_atom: FOLAtom,
    rule_bank: FOLRuleBank,
) -> bool:
    available_preds = _rhs_predicates(rule)
    current_src = int(rule.dst_layer)

    while True:
        if goal_atom.predicate in available_preds:
            return True
        next_rules = rule_bank.transition_rules(current_src)
        if not next_rules:
            break
        next_preds: set[str] = set()
        for t in next_rules:
            if _lhs_predicates(t).issubset(available_preds):
                next_preds.update(_rhs_predicates(t))
        if not next_preds:
            break
        available_preds = next_preds
        current_src += 1

    return goal_atom.predicate in available_preds


def _is_applicable(
    rule: FOLLayerRule,
    ants: tuple[FOLAtom, ...],
    max_unify_solutions: int,
) -> bool:
    substitutions = _find_lhs_substitutions_for_facts(
        lhs=rule.lhs,
        facts=ants,
        max_solutions=int(max_unify_solutions),
    )
    return any(
        _subst_binds_rhs_variables(rule=rule, subst=subst)
        for subst in substitutions
    )


def _reachable_predicates_from_rule(
    rule: FOLLayerRule,
    rule_bank: FOLRuleBank,
) -> frozenset[str]:
    """Compute predicate names reachable from rule's RHS through the rule bank."""
    available_preds = _rhs_predicates(rule)
    current_src = int(rule.dst_layer)
    while True:
        next_rules = rule_bank.transition_rules(current_src)
        if not next_rules:
            break
        next_preds: set[str] = set()
        for t in next_rules:
            if _lhs_predicates(t).issubset(available_preds):
                next_preds.update(_rhs_predicates(t))
        if not next_preds:
            break
        available_preds = next_preds
        current_src += 1
    return frozenset(available_preds)


def _precompute_reachable_sets(
    rules: list[FOLLayerRule],
    rule_bank: FOLRuleBank,
) -> dict[FOLLayerRule, frozenset[str]]:
    """Precompute reachable predicate sets for all rules at a layer."""
    return {rule: _reachable_predicates_from_rule(rule, rule_bank) for rule in rules}


def _classify_rules_by_rank(
    *,
    rules: list[FOLLayerRule],
    ants: tuple[FOLAtom, ...],
    goal_atom: FOLAtom,
    rule_bank: FOLRuleBank,
    max_unify_solutions: int,
    reachable_sets: dict[FOLLayerRule, frozenset[str]] | None = None,
) -> dict[int, list[FOLLayerRule]]:
    ranked: dict[int, list[FOLLayerRule]] = {1: [], 2: [], 3: [], 4: []}
    for rule in rules:
        applicable = _is_applicable(rule, ants, max_unify_solutions)
        if reachable_sets is not None:
            reachable = goal_atom.predicate in reachable_sets[rule]
        else:
            reachable = _is_goal_reachable_from_rule_rhs(rule, goal_atom, rule_bank)
        if applicable and reachable:
            ranked[1].append(rule)
        elif applicable and not reachable:
            ranked[2].append(rule)
        elif not applicable and reachable:
            ranked[3].append(rule)
        else:
            ranked[4].append(rule)
    return ranked


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
    ``"zipf_per_rule"``, ``"zipf_per_rule_headless"``).

    Returns ``(sampled_schemas, sampled_ranks)``.
    """
    demo_distribution = str(demo_distribution)
    if demo_ranking_beta is None:
        demo_ranking_beta = float('inf') if demo_ranked else 0.0
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
    else:
        raise ValueError(
            f"sample_ranked_demos does not support demo_distribution={demo_distribution!r}; "
            f"expected 'zipf', 'zipf_headless', 'zipf_per_rule', or 'zipf_per_rule_headless'."
        )


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

    if demo_ranking_beta is None:
        demo_ranking_beta = float('inf') if demo_ranked else 0.0

    if headless:
        sampling_ranks = [k for k in non_empty_ranks if k != 1]
    else:
        sampling_ranks = list(non_empty_ranks)

    if not include_oracle:
        if not sampling_ranks:
            return [], []
        if demo_unique:
            flat_pool: list[tuple[FOLLayerRule, int]] = []
            flat_weights: list[float] = []
            for k in sampling_ranks:
                rules_in_rank = ranked_rules[k]
                per_rule_w = (1.0 / (k ** alpha)) / len(rules_in_rank)
                for rule in rules_in_rank:
                    flat_pool.append((rule, k))
                    flat_weights.append(per_rule_w)
            if not flat_pool:
                return [], []
            w = np.array(flat_weights)
            w /= w.sum()
            n = min(int(n_demos), len(flat_pool))
            indices = rng.choice(len(flat_pool), size=n, replace=False, p=w)
            sampled_schemas = [flat_pool[int(i)][0] for i in indices]
            sampled_ranks = [flat_pool[int(i)][1] for i in indices]
        else:
            weights = np.array([1.0 / (k ** alpha) for k in sampling_ranks])
            weights /= weights.sum()
            sampled_schemas: list[FOLLayerRule] = []
            sampled_ranks: list[int] = []
            for _ in range(int(n_demos)):
                rank_idx = int(rng.choice(len(sampling_ranks), p=weights))
                rank = sampling_ranks[rank_idx]
                pool = ranked_rules[rank]
                rule = pool[int(rng.integers(0, len(pool)))]
                sampled_schemas.append(rule)
                sampled_ranks.append(rank)
        sampled_schemas, sampled_ranks = _rank_order_demos(
            sampled_schemas, sampled_ranks, rng, demo_ranking_beta,
        )
        return sampled_schemas, sampled_ranks

    if oracle_rule is None:
        raise ValueError("include_oracle=True requires oracle_rule.")

    oracle_schema = _find_matching_demo_schema_for_rule(
        schemas=ranked_rules.get(1, []),
        oracle_rule=oracle_rule,
    )
    if oracle_schema is None:
        all_schemas = [r for rules in ranked_rules.values() for r in rules]
        oracle_schema = _find_matching_demo_schema_for_rule(
            schemas=all_schemas,
            oracle_rule=oracle_rule,
        )
    if oracle_schema is None:
        raise RuntimeError(
            "Oracle rule schema was not found among rules in any rank."
        )

    sampled_schemas = [oracle_schema]
    sampled_ranks = [1]

    if int(n_demos) > 1:
        if demo_unique:
            flat_pool = []
            flat_weights = []
            for k in sampling_ranks:
                rules_in_rank = ranked_rules[k]
                per_rule_w = (1.0 / (k ** alpha)) / len(rules_in_rank)
                for rule in rules_in_rank:
                    if rule is oracle_schema:
                        continue
                    flat_pool.append((rule, k))
                    flat_weights.append(per_rule_w)
            if flat_pool:
                w = np.array(flat_weights)
                w /= w.sum()
                n = min(int(n_demos) - 1, len(flat_pool))
                indices = rng.choice(len(flat_pool), size=n, replace=False, p=w)
                for i in indices:
                    sampled_schemas.append(flat_pool[int(i)][0])
                    sampled_ranks.append(flat_pool[int(i)][1])
        else:
            if not sampling_ranks:
                extra_ranks = [k for k in non_empty_ranks if k != 1]
                if not extra_ranks:
                    extra_ranks = list(non_empty_ranks)
                sampling_ranks = extra_ranks if extra_ranks else list(non_empty_ranks)
            weights = np.array([1.0 / (k ** alpha) for k in sampling_ranks])
            weights /= weights.sum()
            for _ in range(int(n_demos) - 1):
                rank_idx = int(rng.choice(len(sampling_ranks), p=weights))
                rank = sampling_ranks[rank_idx]
                pool = ranked_rules[rank]
                rule = pool[int(rng.integers(0, len(pool)))]
                sampled_schemas.append(rule)
                sampled_ranks.append(rank)

    sampled_schemas, sampled_ranks = _rank_order_demos(
        sampled_schemas, sampled_ranks, rng, demo_ranking_beta,
    )
    return sampled_schemas, sampled_ranks


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

    if demo_ranking_beta is None:
        demo_ranking_beta = float('inf') if demo_ranked else 0.0

    if not include_oracle:
        pool, weights = _build_per_rule_pool_and_weights(
            ranked_rules, alpha, exclude_rank_1=headless,
        )
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

    if oracle_rule is None:
        raise ValueError("include_oracle=True requires oracle_rule.")

    oracle_schema = _find_matching_demo_schema_for_rule(
        schemas=ranked_rules.get(1, []),
        oracle_rule=oracle_rule,
    )
    if oracle_schema is None:
        all_schemas = [r for rules in ranked_rules.values() for r in rules]
        oracle_schema = _find_matching_demo_schema_for_rule(
            schemas=all_schemas,
            oracle_rule=oracle_rule,
        )
    if oracle_schema is None:
        raise RuntimeError(
            "Oracle rule schema was not found among rules in any rank."
        )

    sampled_schemas = [oracle_schema]
    sampled_ranks = [1]

    if int(n_demos) > 1:
        pool, weights = _build_per_rule_pool_and_weights(
            ranked_rules, alpha, exclude_rank_1=headless,
        )
        if not pool:
            # Fallback: try non-rank-1, then all ranks
            fallback_pool, fallback_weights = _build_per_rule_pool_and_weights(
                ranked_rules, alpha, exclude_rank_1=True,
            )
            if not fallback_pool:
                fallback_pool, fallback_weights = _build_per_rule_pool_and_weights(
                    ranked_rules, alpha, exclude_rank_1=False,
                )
            pool, weights = fallback_pool, fallback_weights
        if pool:
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


def _sample_zipf_per_rule(
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
) -> tuple[list[FOLLayerRule], list[int]]:
    rules = list(rule_bank.transition_rules(int(src_layer)))
    ranked = _classify_rules_by_rank(
        rules=rules,
        ants=ants,
        goal_atom=goal_atom,
        rule_bank=rule_bank,
        max_unify_solutions=int(max_unify_solutions),
    )
    return sample_ranked_demos(
        ranked_rules=ranked,
        rng=rng,
        n_demos=n_demos,
        demo_distribution="zipf_per_rule_headless" if headless else "zipf_per_rule",
        alpha=alpha,
        include_oracle=include_oracle,
        oracle_rule=oracle_rule,
        demo_ranked=demo_ranked,
        demo_unique=demo_unique,
        demo_ranking_beta=demo_ranking_beta,
    )


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
        oracle_schema = _find_matching_demo_schema_for_rule(
            schemas=actual_ranked.get(1, []),
            oracle_rule=oracle_rule,
        )
        if oracle_schema is None:
            all_flat = [r for rs in actual_ranked.values() for r in rs]
            oracle_schema = _find_matching_demo_schema_for_rule(
                schemas=all_flat, oracle_rule=oracle_rule,
            )
        if oracle_schema is None:
            raise RuntimeError(
                "Oracle rule schema was not found among rules in any rank."
            )
        if oracle_schema not in selected_schemas:
            if len(selected_schemas) >= n_demos:
                selected_schemas[-1] = oracle_schema
                selected_ranks[-1] = 1
            else:
                selected_schemas.append(oracle_schema)
                selected_ranks.append(1)
        if demo_ranking_beta is None:
            _beta = float('inf') if demo_ranked else 0.0
        else:
            _beta = demo_ranking_beta
        selected_schemas, selected_ranks = _rank_order_demos(
            selected_schemas, selected_ranks, rng, _beta,
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


def _instantiate_demo_schema_with_random_constants(
    *,
    rule: FOLLayerRule,
    constants: tuple[str, ...],
    rng: np.random.Generator,
) -> str:
    if not constants:
        raise ValueError("Cannot instantiate demo rule schema without constants.")

    variables = tuple(sorted(rule.variables()))
    if not variables:
        return str(rule.statement_text)

    substitution = {
        var: str(constants[int(rng.integers(0, len(constants)))])
        for var in variables
    }
    return str(rule.instantiate(substitution).statement_text)


def _prepend_demo_statements_to_prompt(
    *,
    prompt_tokens: list[int],
    demo_statements: list[str],
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer,
) -> list[int]:
    out: list[int] = []
    for statement in demo_statements:
        demo_completion = tokenizer.encode_completion_texts([statement])
        out.extend(int(tok) for tok in demo_completion[:-1])
        out.append(int(tokenizer.sep_token_id))
    out.extend(int(tok) for tok in prompt_tokens)
    return out


class FOLDemoAugmentedAdapter:
    """Adapter wrapper that prepends sampled demonstration rules to prompts."""

    def __init__(
        self,
        *,
        base_adapter,
        rule_bank: FOLRuleBank,
        tokenizer,
        min_n_demos: int,
        max_n_demos: int,
        max_unify_solutions: int,
        include_oracle: bool = False,
        demo_distribution: str = "uniform",
        demo_distribution_alpha: float = 1.0,
        demo_ranked: bool = True,
        demo_all: bool = False,
        demo_unique: bool = True,
        cluster_n_samples: int = 100,
        cluster_k: int = 5,
        cluster_base_dist: str = "zipf_per_rule",
        cluster_unselected_rank: int | None = None,
        cluster_distance: int | None = None,
        cluster_initial_ant_max: int | None = None,
        demo_ranking_beta: float | None = None,
    ) -> None:
        self.base_adapter = base_adapter
        self.rule_bank = rule_bank
        self.tokenizer = tokenizer
        self.min_n_demos = int(min_n_demos)
        self.max_n_demos = int(max_n_demos)
        self.max_unify_solutions = int(max_unify_solutions)
        self.include_oracle = bool(include_oracle)
        self.demo_distribution = str(demo_distribution)
        self.demo_distribution_alpha = float(demo_distribution_alpha)
        self.demo_ranked = bool(demo_ranked)
        self.demo_all = bool(demo_all)
        self.demo_unique = bool(demo_unique)
        if demo_ranking_beta is None:
            self.demo_ranking_beta = float('inf') if demo_ranked else 0.0
        else:
            self.demo_ranking_beta = float(demo_ranking_beta)
        self.cluster_n_samples = int(cluster_n_samples)
        self.cluster_k = int(cluster_k)
        self.cluster_base_dist = str(cluster_base_dist)
        self.cluster_unselected_rank = (
            None if cluster_unselected_rank is None
            else int(cluster_unselected_rank)
        )
        self.cluster_distance = (
            None if cluster_distance is None
            else int(cluster_distance)
        )
        self.cluster_initial_ant_max = (
            None if cluster_initial_ant_max is None
            else int(cluster_initial_ant_max)
        )
        self._last_demo_rules: list[FOLLayerRule] = []
        self._oracle_rule: FOLLayerRule | None = None

    def get_last_demo_rules(self) -> list[FOLLayerRule]:
        return list(self._last_demo_rules)

    def set_oracle_rule(self, oracle_rule: FOLLayerRule | None) -> None:
        self._oracle_rule = oracle_rule

    def predict_completion(
        self,
        *,
        model: Callable[[np.ndarray], Any],
        prompt_tokens,
        tokenizer,
        temperature: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        self._last_demo_rules = []
        if self.max_n_demos <= 0 and not self.demo_all:
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
        oracle_rule = self._oracle_rule
        self._oracle_rule = None
        try:
            _, sequent, src_layer, _ = extract_prompt_info_from_row_tokens(
                np.asarray(prompt, dtype=np.int32),
                tokenizer=self.tokenizer,
            )
        except ValueError:
            self._last_demo_rules = []
        else:
            augmented = augment_prompt_with_demos(
                prompt_tokens=prompt,
                rule_bank=self.rule_bank,
                tokenizer=self.tokenizer,
                rng=rng,
                src_layer=int(src_layer),
                ants=tuple(sequent.ants),
                min_n_demos=self.min_n_demos,
                max_n_demos=self.max_n_demos,
                max_unify_solutions=self.max_unify_solutions,
                include_oracle=self.include_oracle,
                oracle_rule=oracle_rule,
                demo_distribution=self.demo_distribution,
                demo_distribution_alpha=self.demo_distribution_alpha,
                goal_atom=sequent.cons,
                demo_ranked=self.demo_ranked,
                demo_all=self.demo_all,
                demo_unique=self.demo_unique,
                cluster_n_samples=self.cluster_n_samples,
                cluster_k=self.cluster_k,
                cluster_base_dist=self.cluster_base_dist,
                cluster_unselected_rank=self.cluster_unselected_rank,
                cluster_distance=self.cluster_distance,
                cluster_initial_ant_max=self.cluster_initial_ant_max,
                demo_ranking_beta=self.demo_ranking_beta,
            )
            prompt = augmented.prompt_tokens
            self._last_demo_rules = list(augmented.demo_schemas)

        return self.base_adapter.predict_completion(
            model=model,
            prompt_tokens=prompt,
            tokenizer=tokenizer,
            temperature=temperature,
            rng=rng,
        )
