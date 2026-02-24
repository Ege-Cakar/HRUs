from __future__ import annotations

import numpy as np

from task.layer_gen.util.fol_rule_bank import (
    FOLAtom,
    FOLSequent,
    build_random_fol_rule_bank,
    parse_clause_text,
    parse_sequent_text,
    sample_fol_problem,
)


def _sample_bank(seed: int = 0):
    rng = np.random.default_rng(seed)
    return build_random_fol_rule_bank(
        n_layers=6,
        predicates_per_layer=4,
        rules_per_transition=8,
        arity_max=3,
        vars_per_rule_max=4,
        constants=("a", "b", "c"),
        k_in_max=3,
        k_out_max=3,
        rng=rng,
    )


def test_parse_clause_and_sequent_roundtrip() -> None:
    clause = "r0_1(a,b) ∧ r0_2(b,c) → r1_1(a,c) ∧ r1_2(c,c)"
    lhs, rhs = parse_clause_text(clause)
    reconstructed = " ∧ ".join(atom.text for atom in lhs) + " → " + " ∧ ".join(
        atom.text for atom in rhs
    )
    assert reconstructed == clause

    sequent = parse_sequent_text("r0_1(a,b),r0_2(b,c)⊢r1_1(a,c)")
    assert sequent == FOLSequent(
        ants=(FOLAtom("r0_1", ("a", "b")), FOLAtom("r0_2", ("b", "c"))),
        cons=FOLAtom("r1_1", ("a", "c")),
    )


def test_generated_rules_are_range_restricted() -> None:
    bank = _sample_bank(seed=7)
    assert bank.transitions

    for rules in bank.transitions.values():
        for rule in rules:
            lhs_vars = {
                term
                for atom in rule.lhs
                for term in atom.args
                if term.startswith("x")
            }
            rhs_vars = {
                term
                for atom in rule.rhs
                for term in atom.args
                if term.startswith("x")
            }
            assert rhs_vars.issubset(lhs_vars)


def test_sample_problem_outputs_ground_instantiations() -> None:
    bank = _sample_bank(seed=11)
    rng = np.random.default_rng(11)

    sampled = sample_fol_problem(
        bank=bank,
        distance=3,
        initial_ant_max=3,
        rng=rng,
    )

    assert sampled.distance == 3
    assert len(sampled.step_layers) == 3
    assert len(sampled.step_ants) == 3
    assert len(sampled.step_rules) == 3

    for src_layer, ants, rule in zip(sampled.step_layers, sampled.step_ants, sampled.step_rules):
        assert src_layer + 1 == rule.dst_layer
        assert set(rule.lhs).issubset(set(ants))
        for atom in rule.lhs + rule.rhs:
            assert not any(term.startswith("x") for term in atom.args)
