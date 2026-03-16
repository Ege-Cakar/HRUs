"""Rule classification by rank (applicability × goal-reachability)."""

from __future__ import annotations

from task.layer_gen.util.fol_rule_bank import FOLAtom, FOLLayerRule, FOLRuleBank

from ._unify import _find_lhs_substitutions_for_facts, _subst_binds_rhs_variables


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
