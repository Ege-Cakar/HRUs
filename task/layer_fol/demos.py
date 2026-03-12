"""Demo augmentation utilities for layered FOL tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from task.layer_gen.util import tokenize_layer_fol
from task.layer_gen.util.fol_rule_bank import FOLAtom, FOLLayerRule, FOLRuleBank
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
) -> FOLDemoAugmentationResult:
    max_n_demos = int(max_n_demos)
    min_n_demos = int(min_n_demos)
    include_oracle = bool(include_oracle)
    demo_distribution = str(demo_distribution)
    demo_distribution_alpha = float(demo_distribution_alpha)
    demo_ranked = bool(demo_ranked)
    demo_all = bool(demo_all)
    demo_unique = bool(demo_unique)

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

    if demo_distribution not in {"uniform", "zipf", "zipf_headless", "zipf_per_rule", "zipf_per_rule_headless"}:
        raise ValueError(
            f"demo_distribution must be 'uniform', 'zipf', 'zipf_headless', "
            f"'zipf_per_rule', or 'zipf_per_rule_headless', "
            f"got {demo_distribution!r}"
        )
    if demo_distribution in {"zipf", "zipf_headless", "zipf_per_rule", "zipf_per_rule_headless"} and goal_atom is None:
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

    if demo_distribution == "uniform":
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
) -> tuple[list[FOLLayerRule], list[int]]:
    rules = list(rule_bank.transition_rules(int(src_layer)))
    ranked = _classify_rules_by_rank(
        rules=rules,
        ants=ants,
        goal_atom=goal_atom,
        rule_bank=rule_bank,
        max_unify_solutions=int(max_unify_solutions),
    )
    return _sample_demo_schemas_zipf(
        rng=rng,
        ranked_rules=ranked,
        n_demos=n_demos,
        alpha=alpha,
        include_oracle=include_oracle,
        oracle_rule=oracle_rule,
        headless=headless,
        demo_ranked=demo_ranked,
        demo_unique=demo_unique,
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


def _classify_rules_by_rank(
    *,
    rules: list[FOLLayerRule],
    ants: tuple[FOLAtom, ...],
    goal_atom: FOLAtom,
    rule_bank: FOLRuleBank,
    max_unify_solutions: int,
) -> dict[int, list[FOLLayerRule]]:
    ranked: dict[int, list[FOLLayerRule]] = {1: [], 2: [], 3: [], 4: []}
    for rule in rules:
        applicable = _is_applicable(rule, ants, max_unify_solutions)
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
) -> tuple[list[FOLLayerRule], list[int]]:
    """Dispatch demo sampling on pre-classified rules by distribution name.

    Accepts the same rank-based distribution names as
    ``augment_prompt_with_demos`` (``"zipf"``, ``"zipf_headless"``,
    ``"zipf_per_rule"``, ``"zipf_per_rule_headless"``).

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
) -> tuple[list[FOLLayerRule], list[int]]:
    non_empty_ranks = sorted(k for k, v in ranked_rules.items() if v)
    if not non_empty_ranks:
        return [], []

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
        if demo_ranked:
            paired = sorted(
                zip(sampled_ranks, sampled_schemas),
                key=lambda x: x[0],
                reverse=True,
            )
            sampled_schemas = [s for _, s in paired]
            sampled_ranks = [r for r, _ in paired]
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

    if demo_ranked:
        paired = sorted(
            zip(sampled_ranks, sampled_schemas),
            key=lambda x: x[0],
            reverse=True,
        )
        sampled_schemas = [s for _, s in paired]
        sampled_ranks = [r for r, _ in paired]
    else:
        order = rng.permutation(len(sampled_schemas))
        sampled_schemas = [sampled_schemas[int(i)] for i in order]
        sampled_ranks = [sampled_ranks[int(i)] for i in order]
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
) -> tuple[list[FOLLayerRule], list[int]]:
    non_empty_ranks = sorted(k for k, v in ranked_rules.items() if v)
    if not non_empty_ranks:
        return [], []

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
        if demo_ranked:
            paired = sorted(
                zip(sampled_ranks, sampled_schemas),
                key=lambda x: x[0],
                reverse=True,
            )
            sampled_schemas = [s for _, s in paired]
            sampled_ranks = [r for r, _ in paired]
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

    if demo_ranked:
        paired = sorted(
            zip(sampled_ranks, sampled_schemas),
            key=lambda x: x[0],
            reverse=True,
        )
        sampled_schemas = [s for _, s in paired]
        sampled_ranks = [r for r, _ in paired]
    else:
        order = rng.permutation(len(sampled_schemas))
        sampled_schemas = [sampled_schemas[int(i)] for i in order]
        sampled_ranks = [sampled_ranks[int(i)] for i in order]
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
) -> tuple[list[FOLLayerRule], list[int]]:
    rules = list(rule_bank.transition_rules(int(src_layer)))
    ranked = _classify_rules_by_rank(
        rules=rules,
        ants=ants,
        goal_atom=goal_atom,
        rule_bank=rule_bank,
        max_unify_solutions=int(max_unify_solutions),
    )
    return _sample_demo_schemas_zipf_per_rule(
        rng=rng,
        ranked_rules=ranked,
        n_demos=n_demos,
        alpha=alpha,
        include_oracle=include_oracle,
        oracle_rule=oracle_rule,
        headless=headless,
        demo_ranked=demo_ranked,
        demo_unique=demo_unique,
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
