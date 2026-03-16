"""Problem sampling and unification for FOL rule banks."""

from __future__ import annotations

import numpy as np

from ._types import (
    FOLAtom,
    FOLLayerRule,
    FOLRuleBank,
    FOLSampledProblem,
    _is_variable,
    _sorted_atoms,
)


def _unify_template_atom(
    template: FOLAtom,
    ground: FOLAtom,
    subst: dict[str, str],
) -> dict[str, str] | None:
    # Unify one template atom with one ground fact under the current substitution.
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


def _find_lhs_substitutions(
    *,
    lhs: tuple[FOLAtom, ...],
    facts: tuple[FOLAtom, ...],
    max_solutions: int,
) -> list[dict[str, str]]:
    # Backtracking search for substitutions that make every LHS atom match facts.
    solutions: list[dict[str, str]] = []

    def _search(idx: int, subst: dict[str, str]) -> None:
        if len(solutions) >= max_solutions:
            return
        if idx >= len(lhs):
            solutions.append(dict(subst))
            return

        templ = lhs[idx]
        # Try matching this LHS atom against every available fact.
        for fact in facts:
            maybe = _unify_template_atom(templ, fact, subst)
            if maybe is None:
                continue
            _search(idx + 1, maybe)
            if len(solutions) >= max_solutions:
                return

    _search(0, {})
    return solutions


def _has_rhs_support(
    *,
    rule: FOLLayerRule,
    subst: dict[str, str],
) -> bool:
    # Keep only substitutions that bind all variables used on the RHS.
    rhs_vars = {
        term
        for atom in rule.rhs
        for term in atom.args
        if _is_variable(term)
    }
    return rhs_vars.issubset(set(subst))


def sample_fol_problem(
    *,
    bank: FOLRuleBank,
    distance: int,
    initial_ant_max: int,
    rng: np.random.Generator,
    max_attempts: int = 1024,
    max_unify_solutions: int = 128,
) -> FOLSampledProblem:
    if distance < 1:
        raise ValueError(f"distance must be >= 1, got {distance}")
    if distance >= bank.n_layers:
        raise ValueError(
            f"distance {distance} is too large for n_layers={bank.n_layers}."
        )
    if max_unify_solutions < 1:
        raise ValueError("max_unify_solutions must be >= 1")

    max_start = bank.n_layers - distance - 1
    if max_start < 0:
        raise ValueError(
            f"No valid start layer for distance {distance} with n_layers={bank.n_layers}."
        )

    constants = np.asarray(bank.constants, dtype=object)

    for _ in range(max_attempts):
        start_layer = int(rng.integers(0, max_start + 1))
        predicates = np.asarray(bank.predicates_for_layer(start_layer), dtype=object)
        if predicates.size < 1:
            continue
        max_initial = max(1, min(int(initial_ant_max), int(predicates.size)))
        initial_size = int(rng.integers(1, max_initial + 1))
        picked = [
            str(tok)
            for tok in rng.choice(predicates, size=initial_size, replace=False)
        ]

        initial_facts = {
            FOLAtom(
                predicate=pred,
                args=tuple(
                    str(constants[int(rng.integers(0, len(constants)))])
                    for _ in range(bank.predicate_arities[pred])
                ),
            )
            for pred in picked
        }

        facts_by_layer: dict[int, set[FOLAtom]] = {start_layer: set(initial_facts)}
        step_layers: list[int] = []
        step_ants: list[tuple[FOLAtom, ...]] = []
        step_templates: list[FOLLayerRule] = []
        step_rules: list[FOLLayerRule] = []
        step_substitutions: list[dict[str, str]] = []

        feasible = True
        for step in range(distance):
            src_layer = start_layer + step
            dst_layer = src_layer + 1
            src_facts = _sorted_atoms(facts_by_layer.get(src_layer, set()))
            if not src_facts:
                feasible = False
                break

            candidates: list[tuple[FOLLayerRule, dict[str, str]]] = []
            for rule in bank.transition_rules(src_layer):
                subs = _find_lhs_substitutions(
                    lhs=rule.lhs,
                    facts=src_facts,
                    max_solutions=max_unify_solutions,
                )
                if not subs:
                    continue

                valid_subs = [
                    sub
                    for sub in subs
                    if _has_rhs_support(rule=rule, subst=sub)
                ]
                if not valid_subs:
                    continue

                pick_sub = valid_subs[int(rng.integers(0, len(valid_subs)))]
                candidates.append((rule, pick_sub))

            if not candidates:
                feasible = False
                break

            pick_idx = int(rng.integers(0, len(candidates)))
            template, subst = candidates[pick_idx]
            instantiated = template.instantiate(subst)

            step_layers.append(src_layer)
            step_ants.append(src_facts)
            step_templates.append(template)
            step_rules.append(instantiated)
            step_substitutions.append(dict(subst))

            facts_by_layer.setdefault(dst_layer, set()).update(instantiated.rhs)

        final_layer = start_layer + distance
        final_facts = _sorted_atoms(facts_by_layer.get(final_layer, set()))
        if not feasible or not final_facts:
            continue

        goal_atom = final_facts[int(rng.integers(0, len(final_facts)))]
        return FOLSampledProblem(
            distance=int(distance),
            start_layer=int(start_layer),
            goal_atom=goal_atom,
            step_layers=tuple(step_layers),
            step_ants=tuple(step_ants),
            step_rule_templates=tuple(step_templates),
            step_rules=tuple(step_rules),
            step_substitutions=tuple(step_substitutions),
        )

    raise RuntimeError(
        "Failed to sample feasible FOL problem after "
        f"{max_attempts} attempts for distance={distance}."
    )
