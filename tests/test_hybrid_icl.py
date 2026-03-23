"""Tests for hybrid ICL bank construction and sampling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from task.layer_gen.util.fol_rule_bank import (
    FOLRuleBank,
    HybridICLBank,
    HybridICLSampledProblem,
    build_hybrid_icl_bank,
    build_random_fol_rule_bank,
    load_hybrid_icl_bank,
    sample_hybrid_icl_problem,
    save_hybrid_icl_bank,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _small_base_bank(seed: int = 42) -> FOLRuleBank:
    """Build a small base bank for fast tests."""
    return build_random_fol_rule_bank(
        n_layers=6,
        predicates_per_layer=4,
        rules_per_transition=8,
        arity_min=1,
        arity_max=2,
        vars_per_rule_max=3,
        constants=("a", "b", "c"),
        k_in_min=1,
        k_in_max=2,
        k_out_min=1,
        k_out_max=2,
        rng=np.random.default_rng(seed),
    )


def _small_hybrid_bank(seed: int = 42, p_fresh: float = 0.5) -> HybridICLBank:
    """Build a small hybrid bank for fast tests."""
    base = _small_base_bank(seed)
    return build_hybrid_icl_bank(
        base_bank=base,
        fresh_predicates_per_layer=4,
        fresh_rules_per_transition=4,
        pred_train_frac=0.5,
        p_fresh=p_fresh,
        predicate_name_len=3,
        rng=np.random.default_rng(seed + 1000),
    )


# ── Construction ─────────────────────────────────────────────────────────────

class TestHybridICLBankConstruction:
    def test_builds_successfully(self):
        bank = _small_hybrid_bank()
        assert isinstance(bank, HybridICLBank)
        assert bank.n_layers == 6
        assert bank.n_transitions == 5

    def test_train_and_eval_pred_predicates_disjoint(self):
        bank = _small_hybrid_bank()
        for layer in bank.train_predicates_by_layer:
            train_set = set(bank.train_predicates_by_layer[layer])
            if layer in bank.eval_pred_predicates_by_layer:
                eval_set = set(bank.eval_pred_predicates_by_layer[layer])
                overlap = train_set & eval_set
                assert not overlap, (
                    f"Layer {layer}: train and eval-pred predicates overlap: {overlap}"
                )

    def test_fresh_predicates_not_in_base_bank(self):
        bank = _small_hybrid_bank()
        base_preds = set(bank.base_bank.predicate_arities.keys())
        fresh_preds = bank.all_fresh_predicates()
        overlap = base_preds & fresh_preds
        assert not overlap, f"Fresh predicates overlap with base bank: {overlap}"

    def test_train_rules_use_train_predicates_on_lhs(self):
        bank = _small_hybrid_bank()
        for layer, rules in bank.train_rules.items():
            train_preds = set(bank.train_predicates_by_layer.get(layer, ()))
            for rule in rules:
                for atom in rule.lhs:
                    assert atom.predicate in train_preds or atom.predicate in bank.base_bank.predicate_arities, (
                        f"Train rule LHS uses unexpected predicate: {atom.predicate}"
                    )

    def test_eval_pred_rules_use_eval_predicates_on_lhs(self):
        bank = _small_hybrid_bank()
        for layer, rules in bank.eval_pred_rules.items():
            eval_preds = set(bank.eval_pred_predicates_by_layer.get(layer, ()))
            for rule in rules:
                for atom in rule.lhs:
                    assert atom.predicate in eval_preds or atom.predicate in bank.base_bank.predicate_arities, (
                        f"Eval-pred rule LHS uses unexpected predicate: {atom.predicate}"
                    )

    def test_fresh_rules_have_standard_rhs(self):
        """Fresh rules should produce standard predicates on RHS (composability)."""
        bank = _small_hybrid_bank()
        base_preds = set(bank.base_bank.predicate_arities.keys())
        for rule in bank.all_fresh_rules():
            for atom in rule.rhs:
                assert atom.predicate in base_preds, (
                    f"Fresh rule RHS uses non-standard predicate: {atom.predicate}"
                )

    def test_eval_rule_rules_differ_from_train(self):
        """Eval-rule rules should have different structure than train rules."""
        bank = _small_hybrid_bank()
        for layer in bank.train_rules:
            train_stmts = {r.statement_text for r in bank.train_rules[layer]}
            if layer in bank.eval_rule_rules:
                eval_stmts = {r.statement_text for r in bank.eval_rule_rules[layer]}
                # They should be largely different (allow some accidental overlap).
                if len(train_stmts) > 1 and len(eval_stmts) > 1:
                    # At least some rules should differ.
                    assert eval_stmts != train_stmts, (
                        f"Layer {layer}: eval-rule rules identical to train rules"
                    )

    def test_each_transition_has_rules(self):
        bank = _small_hybrid_bank()
        for src_layer in range(bank.n_transitions):
            assert src_layer in bank.train_rules, f"Missing train rules for layer {src_layer}"
            assert len(bank.train_rules[src_layer]) > 0

    def test_deterministic_from_seed(self):
        b1 = _small_hybrid_bank(seed=99)
        b2 = _small_hybrid_bank(seed=99)
        assert b1.train_predicates_by_layer == b2.train_predicates_by_layer
        assert b1.eval_pred_predicates_by_layer == b2.eval_pred_predicates_by_layer

    def test_rejects_too_few_fresh_predicates(self):
        base = _small_base_bank()
        with pytest.raises(ValueError, match="fresh_predicates_per_layer"):
            build_hybrid_icl_bank(
                base_bank=base,
                fresh_predicates_per_layer=1,
                fresh_rules_per_transition=4,
                rng=np.random.default_rng(0),
            )


# ── Sampling ─────────────────────────────────────────────────────────────────

class TestHybridICLSampling:
    def test_sample_train_mode(self):
        bank = _small_hybrid_bank(p_fresh=0.5)
        problem = sample_hybrid_icl_problem(
            hybrid_bank=bank,
            distance=2,
            eval_mode="train",
            rng=np.random.default_rng(42),
            max_attempts=2048,
        )
        assert isinstance(problem, HybridICLSampledProblem)
        assert problem.distance == 2
        assert len(problem.step_layers) == 2
        assert len(problem.transition_sources) == 2

    def test_sample_rule_gen_mode(self):
        bank = _small_hybrid_bank(p_fresh=0.5)
        problem = sample_hybrid_icl_problem(
            hybrid_bank=bank,
            distance=2,
            eval_mode="rule_gen",
            rng=np.random.default_rng(42),
            max_attempts=2048,
        )
        assert isinstance(problem, HybridICLSampledProblem)

    def test_sample_pred_gen_mode(self):
        bank = _small_hybrid_bank(p_fresh=0.5)
        problem = sample_hybrid_icl_problem(
            hybrid_bank=bank,
            distance=2,
            eval_mode="pred_gen",
            rng=np.random.default_rng(42),
            max_attempts=2048,
        )
        assert isinstance(problem, HybridICLSampledProblem)

    def test_transition_sources_valid(self):
        bank = _small_hybrid_bank(p_fresh=0.5)
        rng = np.random.default_rng(42)
        for _ in range(10):
            problem = sample_hybrid_icl_problem(
                hybrid_bank=bank,
                distance=2,
                eval_mode="train",
                rng=rng,
                max_attempts=2048,
            )
            for source in problem.transition_sources:
                assert source in ("internalized", "fresh")

    def test_fresh_rules_tracked(self):
        """When a fresh transition is used, the template should be in fresh_rules_used."""
        bank = _small_hybrid_bank(p_fresh=1.0)  # Force all fresh
        rng = np.random.default_rng(42)
        problem = sample_hybrid_icl_problem(
            hybrid_bank=bank,
            distance=2,
            eval_mode="train",
            rng=rng,
            max_attempts=4096,
        )
        n_fresh = sum(1 for s in problem.transition_sources if s == "fresh")
        assert len(problem.fresh_rules_used) == n_fresh

    def test_to_fol_sampled_problem(self):
        bank = _small_hybrid_bank()
        problem = sample_hybrid_icl_problem(
            hybrid_bank=bank,
            distance=2,
            eval_mode="train",
            rng=np.random.default_rng(42),
            max_attempts=2048,
        )
        fol_problem = problem.to_fol_sampled_problem()
        assert fol_problem.distance == problem.distance
        assert fol_problem.goal_atom == problem.goal_atom

    def test_composability_fresh_outputs_standard_predicates(self):
        """Fresh rule outputs should use standard predicates so
        internalized rules at the next step can consume them."""
        bank = _small_hybrid_bank(p_fresh=0.8)
        base_preds = set(bank.base_bank.predicate_arities.keys())
        rng = np.random.default_rng(42)
        for _ in range(20):
            problem = sample_hybrid_icl_problem(
                hybrid_bank=bank,
                distance=3,
                eval_mode="train",
                rng=rng,
                max_attempts=4096,
            )
            for i, source in enumerate(problem.transition_sources):
                if source == "fresh":
                    rule = problem.step_rules[i]
                    for atom in rule.rhs:
                        assert atom.predicate in base_preds, (
                            f"Fresh rule RHS predicate {atom.predicate} not in base bank"
                        )

    def test_higher_depth(self):
        bank = _small_hybrid_bank(p_fresh=0.3)
        problem = sample_hybrid_icl_problem(
            hybrid_bank=bank,
            distance=4,
            eval_mode="train",
            rng=np.random.default_rng(42),
            max_attempts=4096,
        )
        assert problem.distance == 4
        assert len(problem.step_layers) == 4


# ── Persistence ──────────────────────────────────────────────────────────────

class TestHybridICLPersistence:
    def test_save_load_roundtrip(self, tmp_path: Path):
        bank = _small_hybrid_bank()
        path = tmp_path / "hybrid_bank.json"
        save_hybrid_icl_bank(path, bank)
        loaded = load_hybrid_icl_bank(path)

        assert loaded.base_bank.n_layers == bank.base_bank.n_layers
        assert loaded.p_fresh == bank.p_fresh
        assert loaded.train_predicates_by_layer == bank.train_predicates_by_layer
        assert loaded.eval_pred_predicates_by_layer == bank.eval_pred_predicates_by_layer
        # Rules should roundtrip.
        for layer in bank.train_rules:
            assert len(loaded.train_rules[layer]) == len(bank.train_rules[layer])

    def test_to_dict_from_dict_roundtrip(self):
        bank = _small_hybrid_bank()
        payload = bank.to_dict()
        loaded = HybridICLBank.from_dict(payload)
        assert loaded.base_bank.n_layers == bank.base_bank.n_layers
        assert loaded.fresh_predicate_name_len == bank.fresh_predicate_name_len

    def test_loaded_bank_can_sample(self, tmp_path: Path):
        bank = _small_hybrid_bank()
        path = tmp_path / "hybrid_bank.json"
        save_hybrid_icl_bank(path, bank)
        loaded = load_hybrid_icl_bank(path)

        problem = sample_hybrid_icl_problem(
            hybrid_bank=loaded,
            distance=2,
            eval_mode="train",
            rng=np.random.default_rng(42),
            max_attempts=2048,
        )
        assert problem.distance == 2
