"""Tests for propositional logic inference rules (Gentzen's NJ system)."""

import pytest
from task.prop_gen.util.elem import (
    Atom, And, Or, Implies, PTrue, PFalse,
    Sequent, Rule,
    Axiom, ImpliesRight, ImpliesLeft,
    AndRight, AndLeft,
    OrRight1, OrRight2, OrLeft,
    TrueRight, FalseLeft, NegationRight,
)


# =============================================================================
# Helpers
# =============================================================================

A = Atom("A")
B = Atom("B")
C = Atom("C")
D = Atom("D")


# =============================================================================
# Test Proposition Types
# =============================================================================

class TestPropositionTypes:
    def test_atom_str(self):
        assert str(A) == "A"
    
    def test_and_str(self):
        assert str(And(A, B)) == "(A ∧ B)"
    
    def test_or_str(self):
        assert str(Or(A, B)) == "(A ∨ B)"
    
    def test_implies_str(self):
        assert str(Implies(A, B)) == "(A → B)"
    
    def test_true_str(self):
        assert str(PTrue()) == "⊤"
    
    def test_false_str(self):
        assert str(PFalse()) == "⊥"
    
    def test_sequent_str(self):
        seq = Sequent([A, B], C)
        assert str(seq) == "A, B ⊢ C"
    
    def test_sequent_empty_ants(self):
        seq = Sequent([], A)
        assert str(seq) == " ⊢ A"
    
    def test_proposition_equality(self):
        """Frozen dataclasses should be hashable and comparable."""
        assert And(A, B) == And(A, B)
        assert And(A, B) != And(B, A)
        assert Implies(A, B) == Implies(A, B)
        assert {And(A, B), And(A, B)} == {And(A, B)}


# =============================================================================
# Test Axiom Rule
# =============================================================================

class TestAxiom:
    def test_axiom_applies(self):
        """A ⊢ A should be immediately proven."""
        seq = Sequent([A], A)
        result = Axiom().apply(seq)
        assert result == []  # No subgoals - proven
    
    def test_axiom_with_extra_ants(self):
        """A, B ⊢ A should be immediately proven."""
        seq = Sequent([A, B], A)
        result = Axiom().apply(seq)
        assert result == []
    
    def test_axiom_not_applicable(self):
        """A ⊢ B should not be provable by axiom."""
        seq = Sequent([A], B)
        result = Axiom().apply(seq)
        assert result is None
    
    def test_axiom_complex_proposition(self):
        """Axiom works with complex propositions."""
        prop = And(Implies(A, B), Or(C, D))
        seq = Sequent([prop, A], prop)
        result = Axiom().apply(seq)
        assert result == []


# =============================================================================
# Test Implication Rules
# =============================================================================

class TestImpliesRight:
    def test_implies_right_basic(self):
        """⊢ A → B becomes A ⊢ B"""
        seq = Sequent([], Implies(A, B))
        result = ImpliesRight().apply(seq)
        assert len(result) == 1
        assert result[0].ants == (A,)
        assert result[0].cons == B
    
    def test_implies_right_with_context(self):
        """C ⊢ A → B becomes C, A ⊢ B"""
        seq = Sequent([C], Implies(A, B))
        result = ImpliesRight().apply(seq)
        assert len(result) == 1
        assert result[0].ants == (C, A)
        assert result[0].cons == B
    
    def test_implies_right_not_applicable(self):
        """Rule doesn't apply to non-implication consequent."""
        seq = Sequent([A], B)
        result = ImpliesRight().apply(seq)
        assert result is None
    
    def test_implies_right_nested(self):
        """⊢ A → (B → C) becomes A ⊢ B → C"""
        seq = Sequent([], Implies(A, Implies(B, C)))
        result = ImpliesRight().apply(seq)
        assert len(result) == 1
        assert result[0].ants == (A,)
        assert result[0].cons == Implies(B, C)


class TestImpliesLeft:
    def test_implies_left_basic(self):
        """A → B ⊢ C splits into ⊢ A and B ⊢ C"""
        impl = Implies(A, B)
        seq = Sequent([impl], C)
        result = ImpliesLeft(impl).apply(seq)
        assert len(result) == 2
        # First subgoal: prove A
        assert result[0].ants == ()
        assert result[0].cons == A
        # Second subgoal: with B, prove C
        assert result[1].ants == (B,)
        assert result[1].cons == C
    
    def test_implies_left_with_context(self):
        """D, A → B ⊢ C splits into D ⊢ A and D, B ⊢ C"""
        impl = Implies(A, B)
        seq = Sequent([D, impl], C)
        result = ImpliesLeft(impl).apply(seq)
        assert len(result) == 2
        assert D in result[0].ants
        assert D in result[1].ants
    
    def test_implies_left_not_in_context(self):
        """Rule not applicable if implication not in antecedents."""
        impl = Implies(A, B)
        seq = Sequent([C], D)
        result = ImpliesLeft(impl).apply(seq)
        assert result is None


# =============================================================================
# Test Conjunction Rules
# =============================================================================

class TestAndRight:
    def test_and_right_basic(self):
        """⊢ A ∧ B splits into ⊢ A and ⊢ B"""
        seq = Sequent([], And(A, B))
        result = AndRight().apply(seq)
        assert len(result) == 2
        assert result[0].cons == A
        assert result[1].cons == B
    
    def test_and_right_with_context(self):
        """C ⊢ A ∧ B splits into C ⊢ A and C ⊢ B"""
        seq = Sequent([C], And(A, B))
        result = AndRight().apply(seq)
        assert len(result) == 2
        assert result[0].ants == (C,)
        assert result[1].ants == (C,)
    
    def test_and_right_not_applicable(self):
        """Rule doesn't apply to non-conjunction consequent."""
        seq = Sequent([], A)
        result = AndRight().apply(seq)
        assert result is None


class TestAndLeft:
    def test_and_left_basic(self):
        """A ∧ B ⊢ C becomes A, B ⊢ C"""
        conj = And(A, B)
        seq = Sequent([conj], C)
        result = AndLeft(conj).apply(seq)
        assert len(result) == 1
        assert A in result[0].ants
        assert B in result[0].ants
        assert conj not in result[0].ants
    
    def test_and_left_with_context(self):
        """D, A ∧ B ⊢ C becomes D, A, B ⊢ C"""
        conj = And(A, B)
        seq = Sequent([D, conj], C)
        result = AndLeft(conj).apply(seq)
        assert len(result) == 1
        assert D in result[0].ants
        assert A in result[0].ants
        assert B in result[0].ants
    
    def test_and_left_not_in_context(self):
        """Rule not applicable if conjunction not in antecedents."""
        conj = And(A, B)
        seq = Sequent([C], D)
        result = AndLeft(conj).apply(seq)
        assert result is None


# =============================================================================
# Test Disjunction Rules
# =============================================================================

class TestOrRight1:
    def test_or_right1_basic(self):
        """⊢ A ∨ B becomes ⊢ A (prove left disjunct)"""
        seq = Sequent([], Or(A, B))
        result = OrRight1().apply(seq)
        assert len(result) == 1
        assert result[0].cons == A
    
    def test_or_right1_not_applicable(self):
        """Rule doesn't apply to non-disjunction consequent."""
        seq = Sequent([], A)
        result = OrRight1().apply(seq)
        assert result is None


class TestOrRight2:
    def test_or_right2_basic(self):
        """⊢ A ∨ B becomes ⊢ B (prove right disjunct)"""
        seq = Sequent([], Or(A, B))
        result = OrRight2().apply(seq)
        assert len(result) == 1
        assert result[0].cons == B
    
    def test_or_right2_not_applicable(self):
        """Rule doesn't apply to non-disjunction consequent."""
        seq = Sequent([], A)
        result = OrRight2().apply(seq)
        assert result is None


class TestOrLeft:
    def test_or_left_basic(self):
        """A ∨ B ⊢ C splits into A ⊢ C and B ⊢ C"""
        disj = Or(A, B)
        seq = Sequent([disj], C)
        result = OrLeft(disj).apply(seq)
        assert len(result) == 2
        # First case: assume A
        assert A in result[0].ants
        assert disj not in result[0].ants
        assert result[0].cons == C
        # Second case: assume B
        assert B in result[1].ants
        assert disj not in result[1].ants
        assert result[1].cons == C
    
    def test_or_left_not_in_context(self):
        """Rule not applicable if disjunction not in antecedents."""
        disj = Or(A, B)
        seq = Sequent([C], D)
        result = OrLeft(disj).apply(seq)
        assert result is None


# =============================================================================
# Test Truth Rules
# =============================================================================

class TestTrueRight:
    def test_true_right_basic(self):
        """⊢ ⊤ is immediately proven."""
        seq = Sequent([], PTrue())
        result = TrueRight().apply(seq)
        assert result == []
    
    def test_true_right_with_context(self):
        """A, B ⊢ ⊤ is immediately proven."""
        seq = Sequent([A, B], PTrue())
        result = TrueRight().apply(seq)
        assert result == []
    
    def test_true_right_not_applicable(self):
        """Rule doesn't apply to non-⊤ consequent."""
        seq = Sequent([], A)
        result = TrueRight().apply(seq)
        assert result is None


# =============================================================================
# Test Falsity Rules
# =============================================================================

class TestFalseLeft:
    def test_false_left_basic(self):
        """⊥ ⊢ A is immediately proven (ex falso)."""
        seq = Sequent([PFalse()], A)
        result = FalseLeft().apply(seq)
        assert result == []
    
    def test_false_left_with_context(self):
        """B, ⊥ ⊢ A is immediately proven."""
        seq = Sequent([B, PFalse()], A)
        result = FalseLeft().apply(seq)
        assert result == []
    
    def test_false_left_any_consequent(self):
        """⊥ ⊢ (complex) is immediately proven."""
        complex_prop = And(Implies(A, B), Or(C, D))
        seq = Sequent([PFalse()], complex_prop)
        result = FalseLeft().apply(seq)
        assert result == []
    
    def test_false_left_not_applicable(self):
        """Rule doesn't apply without ⊥ in antecedents."""
        seq = Sequent([A], B)
        result = FalseLeft().apply(seq)
        assert result is None


# =============================================================================
# Test Negation Rules
# =============================================================================

class TestNegationRight:
    def test_negation_right_basic(self):
        """⊢ A → ⊥ becomes A ⊢ ⊥"""
        seq = Sequent([], Implies(A, PFalse()))
        result = NegationRight().apply(seq)
        assert len(result) == 1
        assert result[0].ants == (A,)
        assert result[0].cons == PFalse()
    
    def test_negation_right_with_context(self):
        """B ⊢ A → ⊥ becomes B, A ⊢ ⊥"""
        seq = Sequent([B], Implies(A, PFalse()))
        result = NegationRight().apply(seq)
        assert len(result) == 1
        assert B in result[0].ants
        assert A in result[0].ants
        assert result[0].cons == PFalse()
    
    def test_negation_right_not_applicable_regular_implies(self):
        """Rule doesn't apply to regular implications (not ending in ⊥)."""
        seq = Sequent([], Implies(A, B))
        result = NegationRight().apply(seq)
        assert result is None


# =============================================================================
# Integration Tests - Complete Proofs
# =============================================================================

class TestCompleteProofs:
    """Test that rules can be composed to prove theorems."""
    
    def test_identity_proof(self):
        """Prove: ⊢ A → A"""
        # Start: ⊢ A → A
        seq = Sequent([], Implies(A, A))
        
        # Apply ImpliesRight: A ⊢ A
        result = ImpliesRight().apply(seq)
        assert len(result) == 1
        
        # Apply Axiom: proven!
        final = Axiom().apply(result[0])
        assert final == []
    
    def test_modus_ponens_derivable(self):
        """Prove: A, A → B ⊢ B"""
        impl = Implies(A, B)
        seq = Sequent([A, impl], B)
        
        # Apply ImpliesLeft on A → B
        result = ImpliesLeft(impl).apply(seq)
        assert len(result) == 2
        
        # First subgoal: A ⊢ A (axiom)
        assert Axiom().apply(result[0]) == []
        
        # Second subgoal: A, B ⊢ B (axiom)
        assert Axiom().apply(result[1]) == []
    
    def test_and_commutativity_left_to_right(self):
        """Prove: A ∧ B ⊢ B ∧ A"""
        conj_ab = And(A, B)
        conj_ba = And(B, A)
        seq = Sequent([conj_ab], conj_ba)
        
        # Apply AndLeft: A, B ⊢ B ∧ A
        result = AndLeft(conj_ab).apply(seq)
        assert len(result) == 1
        
        # Apply AndRight: A, B ⊢ B and A, B ⊢ A
        result2 = AndRight().apply(result[0])
        assert len(result2) == 2
        
        # Both are axioms
        assert Axiom().apply(result2[0]) == []
        assert Axiom().apply(result2[1]) == []
    
    def test_or_introduction(self):
        """Prove: A ⊢ A ∨ B"""
        seq = Sequent([A], Or(A, B))
        
        # Apply OrRight1: A ⊢ A
        result = OrRight1().apply(seq)
        assert len(result) == 1
        
        # Axiom
        assert Axiom().apply(result[0]) == []
    
    def test_hypothetical_syllogism(self):
        """Prove: A → B, B → C ⊢ A → C"""
        impl_ab = Implies(A, B)
        impl_bc = Implies(B, C)
        seq = Sequent([impl_ab, impl_bc], Implies(A, C))
        
        # Apply ImpliesRight: A → B, B → C, A ⊢ C
        result = ImpliesRight().apply(seq)
        assert len(result) == 1
        new_seq = result[0]
        assert A in new_seq.ants
        
        # Apply ImpliesLeft on A → B
        result2 = ImpliesLeft(impl_ab).apply(new_seq)
        assert len(result2) == 2
        
        # First subgoal has A in antecedents (without A → B)
        # Axiom should work
        assert Axiom().apply(result2[0]) == []
        
        # Second subgoal: B → C, A, B ⊢ C
        # Apply ImpliesLeft on B → C
        result3 = ImpliesLeft(impl_bc).apply(result2[1])
        assert len(result3) == 2
        
        # Both subgoals are axioms
        assert Axiom().apply(result3[0]) == []
        assert Axiom().apply(result3[1]) == []
    
    def test_ex_falso(self):
        """Prove: ⊥ ⊢ A ∧ B"""
        seq = Sequent([PFalse()], And(A, B))
        result = FalseLeft().apply(seq)
        assert result == []  # Immediately proven
