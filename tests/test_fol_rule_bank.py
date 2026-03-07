from __future__ import annotations

import numpy as np

from task.layer_gen.util.fol_rule_bank import (
    FOLAtom,
    FOLDepth3ICLSplitBundle,
    FOLRuleBank,
    FOLSequent,
    build_depth3_icl_split_bundle,
    build_fresh_layer0_bank,
    build_random_fol_rule_bank,
    generate_fresh_predicate_names,
    load_fol_depth3_icl_split_bundle,
    parse_clause_text,
    parse_sequent_text,
    sample_fol_problem,
    save_fol_depth3_icl_split_bundle,
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


def test_generated_rules_respect_k_in_out_min() -> None:
    bank = build_random_fol_rule_bank(
        n_layers=5,
        predicates_per_layer=5,
        rules_per_transition=8,
        arity_max=3,
        vars_per_rule_max=4,
        k_in_max=4,
        k_out_max=4,
        k_in_min=2,
        k_out_min=3,
        constants=("a", "b", "c"),
        rng=np.random.default_rng(23),
    )

    for rules in bank.transitions.values():
        for rule in rules:
            assert len(rule.lhs) >= 2
            assert len(rule.rhs) >= 3


def test_build_random_fol_rule_bank_supports_per_layer_and_transition_counts() -> None:
    bank = build_random_fol_rule_bank(
        n_layers=3,
        predicates_per_layer=(2, 3, 4),
        rules_per_transition=(5, 6),
        arity_max=3,
        vars_per_rule_max=4,
        k_in_max=2,
        k_out_max=2,
        constants=("a", "b", "c"),
        rng=np.random.default_rng(29),
    )

    assert bank.predicates_for_layer(0) == ("r0_1", "r0_2")
    assert bank.predicates_for_layer(1) == ("r1_1", "r1_2", "r1_3")
    assert bank.predicates_for_layer(2) == ("r2_1", "r2_2", "r2_3", "r2_4")
    assert len(bank.transition_rules(0)) == 5
    assert len(bank.transition_rules(1)) == 6

    payload = bank.to_dict()
    assert payload["predicates_per_layer"] == [2, 3, 4]
    roundtrip = FOLRuleBank.from_dict(payload)
    assert roundtrip.to_dict() == payload


def test_build_random_fol_rule_bank_validates_per_transition_minima() -> None:
    import pytest

    with pytest.raises(ValueError, match="transition 0->1"):
        build_random_fol_rule_bank(
            n_layers=3,
            predicates_per_layer=(1, 1, 4),
            rules_per_transition=(4, 4),
            arity_max=3,
            vars_per_rule_max=4,
            k_in_max=2,
            k_out_max=2,
            k_out_min=2,
            constants=("a", "b", "c"),
            rng=np.random.default_rng(31),
        )


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


def test_build_depth3_split_bundle_disjoint_layer0_predicates() -> None:
    bundle = build_depth3_icl_split_bundle(
        predicates_per_layer=4,
        rules_01_train=8,
        rules_01_eval=8,
        rules_12_shared=8,
        arity_max=3,
        vars_per_rule_max=4,
        k_in_max=2,
        k_out_max=2,
        constants=("a", "b", "c"),
        rng=np.random.default_rng(7),
    )
    assert isinstance(bundle, FOLDepth3ICLSplitBundle)
    assert bundle.train_bank.n_layers == 3
    assert bundle.eval_bank.n_layers == 3
    assert len(bundle.train_layer0_indices) == 4
    assert len(bundle.eval_layer0_indices) == 4
    assert set(bundle.train_layer0_indices).isdisjoint(set(bundle.eval_layer0_indices))
    assert set(bundle.train_layer0_indices) | set(bundle.eval_layer0_indices) == set(
        range(1, 9)
    )
    assert any(idx <= 4 for idx in bundle.train_layer0_indices)
    assert any(idx > 4 for idx in bundle.train_layer0_indices)
    assert any(idx <= 4 for idx in bundle.eval_layer0_indices)
    assert any(idx > 4 for idx in bundle.eval_layer0_indices)
    assert set(bundle.train_layer0_predicates).isdisjoint(set(bundle.eval_layer0_predicates))


def test_build_depth3_split_bundle_shared_1_to_2_transitions_identical() -> None:
    bundle = build_depth3_icl_split_bundle(
        predicates_per_layer=4,
        rules_01_train=8,
        rules_01_eval=8,
        rules_12_shared=8,
        arity_max=3,
        vars_per_rule_max=4,
        k_in_max=2,
        k_out_max=2,
        constants=("a", "b", "c"),
        rng=np.random.default_rng(11),
    )
    assert bundle.train_bank.statement_set(1) == bundle.eval_bank.statement_set(1)


def test_split_bundle_roundtrip_save_load_preserves_banks_and_sets(tmp_path) -> None:
    bundle = build_depth3_icl_split_bundle(
        predicates_per_layer=3,
        rules_01_train=4,
        rules_01_eval=5,
        rules_12_shared=6,
        arity_max=3,
        vars_per_rule_max=4,
        k_in_max=2,
        k_out_max=2,
        constants=("a", "b", "c"),
        rng=np.random.default_rng(13),
    )
    path = tmp_path / "split_bundle.json"
    save_fol_depth3_icl_split_bundle(path, bundle)
    loaded = load_fol_depth3_icl_split_bundle(path)

    assert loaded.train_bank.to_dict() == bundle.train_bank.to_dict()
    assert loaded.eval_bank.to_dict() == bundle.eval_bank.to_dict()
    assert loaded.train_layer0_indices == bundle.train_layer0_indices
    assert loaded.eval_layer0_indices == bundle.eval_layer0_indices
    assert loaded.train_layer0_predicates == bundle.train_layer0_predicates
    assert loaded.eval_layer0_predicates == bundle.eval_layer0_predicates
    assert loaded.shared_layer1_predicates == bundle.shared_layer1_predicates
    assert loaded.shared_layer2_predicates == bundle.shared_layer2_predicates


def test_split_bundle_from_dict_v1_backfills_layer0_indices() -> None:
    bundle = build_depth3_icl_split_bundle(
        predicates_per_layer=3,
        rules_01_train=4,
        rules_01_eval=4,
        rules_12_shared=4,
        arity_max=3,
        vars_per_rule_max=4,
        k_in_max=2,
        k_out_max=2,
        constants=("a", "b", "c"),
        rng=np.random.default_rng(17),
    )
    payload = bundle.to_dict()
    payload["version"] = "fol_depth3_icl_split_v1"
    payload.pop("train_layer0_indices")
    payload.pop("eval_layer0_indices")

    loaded = FOLDepth3ICLSplitBundle.from_dict(payload)
    assert loaded.train_layer0_indices == bundle.train_layer0_indices
    assert loaded.eval_layer0_indices == bundle.eval_layer0_indices
    assert loaded.train_layer0_predicates == bundle.train_layer0_predicates
    assert loaded.eval_layer0_predicates == bundle.eval_layer0_predicates


def test_fol_rule_bank_layer_predicates_override_predicates_for_layer() -> None:
    bank = FOLRuleBank(
        n_layers=3,
        predicates_per_layer=2,
        arity_max=1,
        constants=("a",),
        vars_per_rule_max=1,
        predicate_arities={
            "r0_9": 1,
            "r1_9": 1,
            "r2_9": 1,
        },
        transitions={},
        layer_predicates={
            0: ("r0_9",),
            1: ("r1_9",),
            2: ("r2_9",),
        },
    )
    assert bank.predicates_for_layer(0) == ("r0_9",)
    assert bank.predicates_for_layer(1) == ("r1_9",)
    assert bank.predicates_for_layer(2) == ("r2_9",)


def test_generate_fresh_predicate_names_format_and_uniqueness() -> None:
    rng = np.random.default_rng(42)
    # Default name_len=1 produces 3-char names: r_ + 1 char.
    names = generate_fresh_predicate_names(10, rng)
    assert len(names) == 10
    assert len(set(names)) == 10
    for name in names:
        assert name.startswith("r_")
        assert len(name) == 3
        suffix = name[2:]
        assert all(ch in "abcdefghijklmnopqrstuvwxyz0123456789" for ch in suffix)

    # name_len=4 produces 6-char names (legacy behavior).
    rng2 = np.random.default_rng(42)
    names4 = generate_fresh_predicate_names(10, rng2, name_len=4)
    assert len(names4) == 10
    for name in names4:
        assert name.startswith("r_")
        assert len(name) == 6


def test_generate_fresh_predicate_names_rejects_zero() -> None:
    rng = np.random.default_rng(0)
    try:
        generate_fresh_predicate_names(0, rng)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for n=0")


def test_build_fresh_layer0_bank_produces_valid_3_layer_bank() -> None:
    rng = np.random.default_rng(7)
    base_bank = build_random_fol_rule_bank(
        n_layers=3,
        predicates_per_layer=4,
        rules_per_transition=8,
        arity_max=3,
        vars_per_rule_max=4,
        constants=("a", "b", "c"),
        k_in_max=2,
        k_out_max=2,
        rng=rng,
    )
    fresh_preds = generate_fresh_predicate_names(5, rng)
    fresh_bank = build_fresh_layer0_bank(
        base_bank=base_bank,
        fresh_predicates=fresh_preds,
        rules_per_transition=6,
        k_in_min=1,
        k_in_max=2,
        k_out_min=1,
        k_out_max=2,
        rng=rng,
    )
    assert fresh_bank.n_layers == 3
    assert fresh_bank.predicates_for_layer(0) == fresh_preds
    assert fresh_bank.predicates_for_layer(1) == base_bank.predicates_for_layer(1)
    assert fresh_bank.predicates_for_layer(2) == base_bank.predicates_for_layer(2)

    # Layer 0->1 rules use fresh predicates on LHS.
    for rule in fresh_bank.transition_rules(0):
        for atom in rule.lhs:
            assert atom.predicate in set(fresh_preds)
        for atom in rule.rhs:
            assert atom.predicate in set(base_bank.predicates_for_layer(1))

    # Layer 1->2 rules are identical to base.
    assert fresh_bank.statement_set(1) == base_bank.statement_set(1)


def test_build_fresh_layer0_bank_supports_sampling() -> None:
    rng = np.random.default_rng(11)
    base_bank = build_random_fol_rule_bank(
        n_layers=3,
        predicates_per_layer=4,
        rules_per_transition=8,
        arity_max=3,
        vars_per_rule_max=4,
        constants=("a", "b", "c"),
        k_in_max=2,
        k_out_max=2,
        rng=rng,
    )
    fresh_preds = generate_fresh_predicate_names(4, rng)
    fresh_bank = build_fresh_layer0_bank(
        base_bank=base_bank,
        fresh_predicates=fresh_preds,
        rules_per_transition=6,
        k_in_min=1,
        k_in_max=2,
        k_out_min=1,
        k_out_max=2,
        rng=rng,
    )
    sampled = sample_fol_problem(
        bank=fresh_bank,
        distance=2,
        initial_ant_max=3,
        rng=rng,
    )
    assert sampled.distance == 2
    assert sampled.start_layer == 0
    assert len(sampled.step_rules) == 2


def test_fol_atom_arity_zero() -> None:
    atom = FOLAtom(predicate="p", args=())
    assert atom.arity == 0
    assert atom.text == "p"
    # Roundtrip through parse_atom_text.
    from task.layer_gen.util.fol_rule_bank import parse_atom_text
    parsed = parse_atom_text("p")
    assert parsed == atom
    assert parsed.text == "p"
    # Also test arity-0 with parentheses (empty args).
    parsed_empty = parse_atom_text("q()")
    assert parsed_empty == FOLAtom(predicate="q", args=())
    assert parsed_empty.text == "q"


def test_build_random_fol_rule_bank_with_arity_min_zero() -> None:
    rng = np.random.default_rng(99)
    bank = build_random_fol_rule_bank(
        n_layers=3,
        predicates_per_layer=6,
        rules_per_transition=8,
        arity_max=2,
        arity_min=0,
        vars_per_rule_max=4,
        k_in_max=2,
        k_out_max=2,
        constants=("a", "b"),
        rng=rng,
    )
    assert bank.arity_min == 0
    assert bank.arity_max == 2
    # Should contain some arity-0 predicates.
    zero_arity_preds = [
        pred for pred, arity in bank.predicate_arities.items() if arity == 0
    ]
    assert len(zero_arity_preds) > 0, "Expected at least one arity-0 predicate with arity_min=0"
    # Rules should still be valid.
    for rules in bank.transitions.values():
        for rule in rules:
            lhs_vars = {t for a in rule.lhs for t in a.args if t.startswith("x")}
            rhs_vars = {t for a in rule.rhs for t in a.args if t.startswith("x")}
            assert rhs_vars.issubset(lhs_vars)


def test_build_fresh_layer0_bank_with_arity_min_zero() -> None:
    rng = np.random.default_rng(42)
    base_bank = build_random_fol_rule_bank(
        n_layers=3,
        predicates_per_layer=6,
        rules_per_transition=8,
        arity_max=2,
        arity_min=0,
        vars_per_rule_max=4,
        k_in_max=2,
        k_out_max=2,
        constants=("a", "b"),
        rng=rng,
    )
    fresh_preds = generate_fresh_predicate_names(4, rng)
    fresh_bank = build_fresh_layer0_bank(
        base_bank=base_bank,
        fresh_predicates=fresh_preds,
        rules_per_transition=6,
        k_in_min=1,
        k_in_max=2,
        k_out_min=1,
        k_out_max=2,
        rng=rng,
    )
    assert fresh_bank.arity_min == 0
    assert fresh_bank.n_layers == 3
    assert fresh_bank.predicates_for_layer(0) == fresh_preds


def test_fol_rule_bank_from_dict_defaults_arity_min() -> None:
    """Backward compat: missing arity_min in payload defaults to 1."""
    rng = np.random.default_rng(7)
    bank = build_random_fol_rule_bank(
        n_layers=3,
        predicates_per_layer=4,
        rules_per_transition=4,
        arity_max=2,
        vars_per_rule_max=3,
        k_in_max=2,
        k_out_max=2,
        constants=("a",),
        rng=rng,
    )
    payload = bank.to_dict()
    # Remove arity_min to simulate legacy payload.
    payload.pop("arity_min")
    loaded = FOLRuleBank.from_dict(payload)
    assert loaded.arity_min == 1
