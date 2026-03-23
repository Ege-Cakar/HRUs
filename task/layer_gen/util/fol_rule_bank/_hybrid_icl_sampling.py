"""Problem sampling for hybrid ICL tasks (internalized + fresh rules).

Each derivation step is randomly assigned to use either an internalized
rule (from the base bank) or a fresh rule (from the appropriate pool).
Fresh rules used in a problem become required in-context demonstrations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._hybrid_icl import HybridICLBank
from ._sampling import (
    _find_lhs_substitutions,
    _has_rhs_support,
)
from ._types import (
    FOLAtom,
    FOLLayerRule,
    FOLSampledProblem,
    _is_variable,
    _sorted_atoms,
)


@dataclass(frozen=True)
class HybridICLSampledProblem:
    """A sampled FOL problem with mixed internalized/fresh transitions.

    Extends FOLSampledProblem with information about which transitions
    used fresh rules and which fresh rules are required as in-context demos.
    """

    distance: int
    start_layer: int
    goal_atom: FOLAtom
    step_layers: tuple[int, ...]
    step_ants: tuple[tuple[FOLAtom, ...], ...]
    step_rule_templates: tuple[FOLLayerRule, ...]
    step_rules: tuple[FOLLayerRule, ...]
    step_substitutions: tuple[dict[str, str], ...]
    # Hybrid-specific fields:
    transition_sources: tuple[str, ...]  # "internalized" or "fresh" per step
    fresh_rules_used: tuple[FOLLayerRule, ...]  # template rules needing in-context demos
    fresh_rules_instantiated: tuple[FOLLayerRule, ...]  # instantiated fresh rules

    def to_fol_sampled_problem(self) -> FOLSampledProblem:
        """Convert to a standard FOLSampledProblem (drops hybrid metadata)."""
        return FOLSampledProblem(
            distance=self.distance,
            start_layer=self.start_layer,
            goal_atom=self.goal_atom,
            step_layers=self.step_layers,
            step_ants=self.step_ants,
            step_rule_templates=self.step_rule_templates,
            step_rules=self.step_rules,
            step_substitutions=self.step_substitutions,
        )


def sample_hybrid_icl_problem(
    *,
    hybrid_bank: HybridICLBank,
    distance: int,
    eval_mode: str = "train",
    initial_ant_max: int = 3,
    rng: np.random.Generator,
    max_attempts: int = 1024,
    max_unify_solutions: int = 128,
) -> HybridICLSampledProblem:
    """Sample a problem mixing internalized and fresh transitions.

    Parameters
    ----------
    hybrid_bank : HybridICLBank
        The hybrid bank containing base + fresh rule pools.
    distance : int
        Number of derivation steps.
    eval_mode : str
        Which fresh pool to use: "train", "rule_gen", or "pred_gen".
    initial_ant_max : int
        Maximum number of initial facts.
    rng : np.random.Generator
        Random number generator.
    max_attempts : int
        Maximum sampling attempts before giving up.
    max_unify_solutions : int
        Maximum unification solutions per rule.

    Returns
    -------
    HybridICLSampledProblem
    """
    if distance < 1:
        raise ValueError(f"distance must be >= 1, got {distance}")

    base_bank = hybrid_bank.base_bank
    if distance >= base_bank.n_layers:
        raise ValueError(
            f"distance {distance} too large for n_layers={base_bank.n_layers}."
        )

    fresh_rules_pool = hybrid_bank.fresh_rules_for_mode(eval_mode)
    fresh_preds_pool = hybrid_bank.fresh_predicates_for_mode(eval_mode)
    constants = np.asarray(base_bank.constants, dtype=object)
    p_fresh = float(hybrid_bank.p_fresh)

    max_start = base_bank.n_layers - distance - 1
    if max_start < 0:
        raise ValueError(
            f"No valid start layer for distance {distance} "
            f"with n_layers={base_bank.n_layers}."
        )

    for _ in range(max_attempts):
        start_layer = int(rng.integers(0, max_start + 1))

        # Decide which transitions are fresh vs internalized.
        transition_sources: list[str] = []
        for step in range(distance):
            src_layer = start_layer + step
            has_fresh = (
                src_layer in fresh_rules_pool
                and len(fresh_rules_pool[src_layer]) > 0
            )
            if has_fresh and rng.random() < p_fresh:
                transition_sources.append("fresh")
            else:
                transition_sources.append("internalized")

        # Generate initial facts.
        # Include fresh predicates for the start layer if needed.
        base_predicates = list(base_bank.predicates_for_layer(start_layer))
        if transition_sources[0] == "fresh" and start_layer in fresh_preds_pool:
            # Include fresh predicates so fresh rules can match.
            initial_predicates = base_predicates + list(fresh_preds_pool[start_layer])
        else:
            initial_predicates = base_predicates

        if not initial_predicates:
            continue

        predicates_arr = np.asarray(initial_predicates, dtype=object)
        max_initial = max(1, min(initial_ant_max, len(initial_predicates)))
        initial_size = int(rng.integers(1, max_initial + 1))
        picked = [
            str(tok)
            for tok in rng.choice(predicates_arr, size=initial_size, replace=False)
        ]

        # Build predicate arities including fresh predicates.
        all_arities = dict(base_bank.predicate_arities)
        for layer_preds in fresh_preds_pool.values():
            for pred in layer_preds:
                if pred not in all_arities:
                    # Use a default arity from any rule that uses this pred.
                    all_arities[pred] = _infer_fresh_predicate_arity(
                        pred, fresh_rules_pool
                    )

        initial_facts = set()
        for pred in picked:
            arity = all_arities.get(pred, 0)
            args = tuple(
                str(constants[int(rng.integers(0, len(constants)))])
                for _ in range(arity)
            )
            initial_facts.add(FOLAtom(predicate=pred, args=args))

        # Run the derivation chain.
        facts_by_layer: dict[int, set[FOLAtom]] = {start_layer: set(initial_facts)}
        step_layers: list[int] = []
        step_ants: list[tuple[FOLAtom, ...]] = []
        step_templates: list[FOLLayerRule] = []
        step_rules: list[FOLLayerRule] = []
        step_substitutions: list[dict[str, str]] = []
        fresh_templates_used: list[FOLLayerRule] = []
        fresh_instantiated_used: list[FOLLayerRule] = []

        feasible = True
        for step in range(distance):
            src_layer = start_layer + step
            dst_layer = src_layer + 1
            source = transition_sources[step]

            src_facts = _sorted_atoms(facts_by_layer.get(src_layer, set()))
            if not src_facts:
                feasible = False
                break

            # Choose rule pool based on source type.
            if source == "fresh":
                rules = list(fresh_rules_pool.get(src_layer, ()))
            else:
                rules = list(base_bank.transition_rules(src_layer))

            if not rules:
                # Fallback: try the other pool.
                if source == "fresh":
                    rules = list(base_bank.transition_rules(src_layer))
                    transition_sources[step] = "internalized"
                    source = "internalized"
                else:
                    rules = list(fresh_rules_pool.get(src_layer, ()))
                    transition_sources[step] = "fresh"
                    source = "fresh"

            if not rules:
                feasible = False
                break

            # Find applicable rules via unification.
            candidates: list[tuple[FOLLayerRule, dict[str, str]]] = []
            for rule in rules:
                subs = _find_lhs_substitutions(
                    lhs=rule.lhs,
                    facts=src_facts,
                    max_solutions=max_unify_solutions,
                )
                valid_subs = [
                    sub for sub in subs
                    if _has_rhs_support(rule=rule, subst=sub)
                ]
                if valid_subs:
                    pick_sub = valid_subs[int(rng.integers(0, len(valid_subs)))]
                    candidates.append((rule, pick_sub))

            if not candidates:
                # No applicable rule in chosen pool — try fallback.
                if source == "fresh":
                    alt_rules = list(base_bank.transition_rules(src_layer))
                    transition_sources[step] = "internalized"
                    source = "internalized"
                else:
                    alt_rules = list(fresh_rules_pool.get(src_layer, ()))
                    transition_sources[step] = "fresh"
                    source = "fresh"

                for rule in alt_rules:
                    subs = _find_lhs_substitutions(
                        lhs=rule.lhs,
                        facts=src_facts,
                        max_solutions=max_unify_solutions,
                    )
                    valid_subs = [
                        sub for sub in subs
                        if _has_rhs_support(rule=rule, subst=sub)
                    ]
                    if valid_subs:
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

            if transition_sources[step] == "fresh":
                fresh_templates_used.append(template)
                fresh_instantiated_used.append(instantiated)

            # Add derived facts to the next layer.
            # Include both standard AND fresh predicates as available facts.
            next_facts = facts_by_layer.setdefault(dst_layer, set())
            next_facts.update(instantiated.rhs)

            # Also propagate any fresh-predicate facts for the next layer
            # so fresh rules at the next step can match.
            if step + 1 < distance:
                next_src = dst_layer
                next_source = transition_sources[step + 1]
                if next_source == "fresh" and next_src in fresh_preds_pool:
                    # Generate some fresh-predicate facts for the next layer
                    # so fresh rules have something to match against.
                    fresh_preds = fresh_preds_pool[next_src]
                    for fp in fresh_preds:
                        fp_arity = all_arities.get(fp, 0)
                        fp_args = tuple(
                            str(constants[int(rng.integers(0, len(constants)))])
                            for _ in range(fp_arity)
                        )
                        next_facts.add(FOLAtom(predicate=fp, args=fp_args))

        if not feasible:
            continue

        final_layer = start_layer + distance
        final_facts = _sorted_atoms(facts_by_layer.get(final_layer, set()))
        if not final_facts:
            continue

        goal_atom = final_facts[int(rng.integers(0, len(final_facts)))]

        return HybridICLSampledProblem(
            distance=int(distance),
            start_layer=int(start_layer),
            goal_atom=goal_atom,
            step_layers=tuple(step_layers),
            step_ants=tuple(step_ants),
            step_rule_templates=tuple(step_templates),
            step_rules=tuple(step_rules),
            step_substitutions=tuple(step_substitutions),
            transition_sources=tuple(transition_sources),
            fresh_rules_used=tuple(fresh_templates_used),
            fresh_rules_instantiated=tuple(fresh_instantiated_used),
        )

    raise RuntimeError(
        f"Failed to sample hybrid ICL problem after {max_attempts} attempts "
        f"for distance={distance}, eval_mode={eval_mode!r}."
    )


def _infer_fresh_predicate_arity(
    predicate: str,
    rules_pool: dict[int, tuple[FOLLayerRule, ...]],
) -> int:
    """Infer the arity of a fresh predicate from its usage in rules."""
    for layer_rules in rules_pool.values():
        for rule in layer_rules:
            for atom in rule.lhs + rule.rhs:
                if atom.predicate == predicate:
                    return len(atom.args)
    return 0
