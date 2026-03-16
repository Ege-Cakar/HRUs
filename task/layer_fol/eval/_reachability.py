"""Reachability analysis for layered FOL evaluation."""

from __future__ import annotations

from task.layer_fol._unify import _find_lhs_substitutions_for_facts, _subst_binds_rhs_variables
from task.layer_gen.util.fol_rule_bank import FOLAtom, FOLRuleBank


def _facts_key(facts: tuple[FOLAtom, ...]) -> tuple[str, ...]:
    return tuple(sorted(atom.text for atom in facts))


def reachable_goal_exact_steps(
    *,
    rule_bank: FOLRuleBank,
    layer: int,
    facts: tuple[FOLAtom, ...],
    goal: FOLAtom,
    steps_remaining: int,
    max_unify_solutions: int,
    memo: dict[tuple[int, int, tuple[str, ...]], bool],
) -> bool:
    key = (int(layer), int(steps_remaining), _facts_key(facts))
    cached = memo.get(key)
    if cached is not None:
        return bool(cached)

    if int(steps_remaining) == 0:
        out = bool(goal in set(facts))
        memo[key] = out
        return out

    if int(layer) >= int(rule_bank.n_layers) - 1:
        memo[key] = False
        return False

    facts_tuple = tuple(facts)
    for rule in rule_bank.transition_rules(int(layer)):
        substitutions = _find_lhs_substitutions_for_facts(
            lhs=rule.lhs,
            facts=facts_tuple,
            max_solutions=int(max_unify_solutions),
        )
        if not substitutions:
            continue

        for subst in substitutions:
            if not _subst_binds_rhs_variables(rule=rule, subst=subst):
                continue
            next_rule = rule.instantiate(subst)
            next_facts = tuple(next_rule.rhs)
            if reachable_goal_exact_steps(
                rule_bank=rule_bank,
                layer=int(layer) + 1,
                facts=next_facts,
                goal=goal,
                steps_remaining=int(steps_remaining) - 1,
                max_unify_solutions=int(max_unify_solutions),
                memo=memo,
            ):
                memo[key] = True
                return True

    memo[key] = False
    return False


def predicted_rule_reaches_goal(
    *,
    rule_bank: FOLRuleBank,
    matched_rule: FOLLayerRule,
    goal: FOLAtom,
    goal_layer: int,
    max_unify_solutions: int,
) -> bool:
    dst_layer = int(matched_rule.dst_layer)
    remaining = int(goal_layer) - dst_layer
    if remaining < 0:
        return False

    memo: dict[tuple[int, int, tuple[str, ...]], bool] = {}
    return reachable_goal_exact_steps(
        rule_bank=rule_bank,
        layer=dst_layer,
        facts=tuple(matched_rule.rhs),
        goal=goal,
        steps_remaining=remaining,
        max_unify_solutions=int(max_unify_solutions),
        memo=memo,
    )
