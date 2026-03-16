"""Eval-specific unification and matching for layered FOL rules."""

from __future__ import annotations

from typing import Iterable

from task.layer_fol._unify import _is_variable, _unify_template_atom_with_ground
from task.layer_gen.util.fol_rule_bank import FOLAtom, FOLLayerRule


def _find_multiset_matches(
    *,
    templates: tuple[FOLAtom, ...],
    grounds: tuple[FOLAtom, ...],
    seed_subst: dict[str, str],
    max_solutions: int,
) -> list[dict[str, str]]:
    if len(templates) != len(grounds):
        return []

    solutions: list[dict[str, str]] = []
    used = [False] * len(grounds)

    def _search(idx: int, subst: dict[str, str]) -> None:
        if len(solutions) >= max_solutions:
            return
        if idx >= len(templates):
            solutions.append(dict(subst))
            return

        template = templates[idx]
        for ground_idx, ground in enumerate(grounds):
            if used[ground_idx]:
                continue
            maybe = _unify_template_atom_with_ground(template, ground, subst)
            if maybe is None:
                continue
            used[ground_idx] = True
            _search(idx + 1, maybe)
            used[ground_idx] = False
            if len(solutions) >= max_solutions:
                return

    _search(0, dict(seed_subst))
    return solutions


def _find_instantiation_for_rule(
    *,
    template: FOLLayerRule,
    lhs_ground: tuple[FOLAtom, ...],
    rhs_ground: tuple[FOLAtom, ...],
    max_solutions: int = 64,
) -> dict[str, str] | None:
    lhs_subs = _find_multiset_matches(
        templates=template.lhs,
        grounds=lhs_ground,
        seed_subst={},
        max_solutions=max_solutions,
    )
    if not lhs_subs:
        return None

    for lhs_sub in lhs_subs:
        rhs_subs = _find_multiset_matches(
            templates=template.rhs,
            grounds=rhs_ground,
            seed_subst=lhs_sub,
            max_solutions=1,
        )
        if rhs_subs:
            return rhs_subs[0]
    return None


def _unify_template_atom_with_ground_schema(
    template: FOLAtom,
    ground: FOLAtom,
    subst: dict[str, str],
    pred_map: dict[str, str],
) -> tuple[dict[str, str], dict[str, str]] | None:
    """Like ``_unify_template_atom_with_ground`` but matches structurally.

    Instead of requiring exact predicate name equality, maintains an injective
    mapping *pred_map* from template predicate names to ground predicate names.
    Arity must still match, and variable bindings are consistent as before.

    Returns ``(updated_subst, updated_pred_map)`` on success, ``None`` on failure.
    """
    if len(template.args) != len(ground.args):
        return None

    # Check / extend predicate mapping
    new_pred_map = dict(pred_map)
    bound_pred = new_pred_map.get(template.predicate)
    if bound_pred is None:
        # Injective: ground predicate must not already be a target of another template pred
        if ground.predicate in new_pred_map.values():
            return None
        new_pred_map[template.predicate] = ground.predicate
    elif bound_pred != ground.predicate:
        return None

    # Unify variable / constant arguments (identical to the exact version)
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
    return out, new_pred_map


def _find_multiset_matches_schema(
    *,
    templates: tuple[FOLAtom, ...],
    grounds: tuple[FOLAtom, ...],
    seed_subst: dict[str, str],
    seed_pred_map: dict[str, str],
    max_solutions: int,
) -> list[tuple[dict[str, str], dict[str, str]]]:
    """Like ``_find_multiset_matches`` but threads *pred_map* alongside *subst*."""
    if len(templates) != len(grounds):
        return []

    solutions: list[tuple[dict[str, str], dict[str, str]]] = []
    used = [False] * len(grounds)

    def _search(idx: int, subst: dict[str, str], pred_map: dict[str, str]) -> None:
        if len(solutions) >= max_solutions:
            return
        if idx >= len(templates):
            solutions.append((dict(subst), dict(pred_map)))
            return

        template = templates[idx]
        for ground_idx, ground in enumerate(grounds):
            if used[ground_idx]:
                continue
            maybe = _unify_template_atom_with_ground_schema(
                template, ground, subst, pred_map,
            )
            if maybe is None:
                continue
            new_subst, new_pred_map = maybe
            used[ground_idx] = True
            _search(idx + 1, new_subst, new_pred_map)
            used[ground_idx] = False
            if len(solutions) >= max_solutions:
                return

    _search(0, dict(seed_subst), dict(seed_pred_map))
    return solutions


def _find_schema_match_for_rule(
    *,
    template: FOLLayerRule,
    lhs_ground: tuple[FOLAtom, ...],
    rhs_ground: tuple[FOLAtom, ...],
    max_solutions: int = 64,
) -> bool:
    """Like ``_find_instantiation_for_rule`` but uses schema unification.

    Returns ``True`` if the structural pattern (atom counts, arities,
    variable-binding pattern) of *template* can be matched against the ground
    atoms, allowing arbitrary predicate-name renaming.
    """
    lhs_sols = _find_multiset_matches_schema(
        templates=template.lhs,
        grounds=lhs_ground,
        seed_subst={},
        seed_pred_map={},
        max_solutions=max_solutions,
    )
    if not lhs_sols:
        return False

    for lhs_subst, lhs_pred_map in lhs_sols:
        rhs_sols = _find_multiset_matches_schema(
            templates=template.rhs,
            grounds=rhs_ground,
            seed_subst=lhs_subst,
            seed_pred_map=lhs_pred_map,
            max_solutions=1,
        )
        if rhs_sols:
            return True
    return False


def _any_rule_schema_matches(
    lhs_ground: tuple[FOLAtom, ...],
    rhs_ground: tuple[FOLAtom, ...],
    candidates: Iterable[FOLLayerRule],
) -> bool:
    """Return ``True`` if any candidate rule's schema matches the ground atoms."""
    for rule in candidates:
        if _find_schema_match_for_rule(
            template=rule,
            lhs_ground=lhs_ground,
            rhs_ground=rhs_ground,
        ):
            return True
    return False


def _match_instantiated_rule_from_candidates(
    *,
    candidates: Iterable[FOLLayerRule],
    lhs_ground: tuple[FOLAtom, ...],
    rhs_ground: tuple[FOLAtom, ...],
) -> FOLLayerRule | None:
    for rule in candidates:
        subst = _find_instantiation_for_rule(
            template=rule,
            lhs_ground=lhs_ground,
            rhs_ground=rhs_ground,
        )
        if subst is not None:
            return rule.instantiate(subst)
    return None
