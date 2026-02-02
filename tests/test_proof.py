"""Tests for proof search environment."""

import pytest
from task.prop_gen.util.elem import (
    Atom, And, Or, Implies, PTrue, PFalse, Sequent,
    Axiom, ImpliesRight, ImpliesLeft, AndRight, AndLeft,
    OrRight1, OrRight2, OrLeft, TrueRight, FalseLeft,
)
from task.prop_gen.util.proof import (
    ProofNode, CompletedProofNode, FailedProofNode, InternalProofNode,
    get_next_rules, build_proof_tree,
)


# =============================================================================
# Helpers
# =============================================================================

A = Atom("A")
B = Atom("B")
C = Atom("C")
D = Atom("D")


# =============================================================================
# Test ProofNode Types
# =============================================================================

class TestProofNodeTypes:
    def test_completed_proof_node(self):
        """CompletedProofNode represents a successfully proven sequent."""
        seq = Sequent([A], A)
        node = CompletedProofNode(sequent=seq, rule=Axiom())
        
        assert node.is_provable is True
        assert node.depth == 1
        assert node.size == 1
        assert "✓" in str(node)
        assert "Ax" in str(node)
    
    def test_failed_proof_node(self):
        """FailedProofNode represents a failed proof attempt."""
        seq = Sequent([], A)
        node = FailedProofNode(sequent=seq, reason="no applicable rules")
        
        assert node.is_provable is False
        assert node.depth == 1
        assert node.size == 1
        assert "✗" in str(node)
    
    def test_internal_proof_node_provable(self):
        """InternalProofNode is provable if any branch succeeds."""
        seq = Sequent([A, B], And(A, B))
        child1 = CompletedProofNode(Sequent([A, B], A), Axiom())
        child2 = CompletedProofNode(Sequent([A, B], B), Axiom())
        node = InternalProofNode(seq, [(AndRight(), [child1, child2])])
        
        assert node.is_provable is True
        assert node.depth == 2
        assert node.size == 3
    
    def test_internal_proof_node_not_provable(self):
        """InternalProofNode is not provable if no branch succeeds."""
        seq = Sequent([A], B)
        child = FailedProofNode(Sequent([A], B), reason="no match")
        node = InternalProofNode(seq, [(ImpliesRight(), [child])])
        
        assert node.is_provable is False
    
    def test_internal_proof_node_successful_branch(self):
        """successful_branch() returns the first working branch."""
        seq = Sequent([A], A)
        child_success = CompletedProofNode(Sequent([A], A), Axiom())
        child_fail = FailedProofNode(Sequent([A], B), "no match")
        
        node = InternalProofNode(seq, [
            (OrRight1(), [child_fail]),
            (Axiom(), [child_success]),
        ])
        
        branch = node.successful_branch()
        assert branch is not None
        rule, children = branch
        assert isinstance(rule, Axiom)


# =============================================================================
# Test get_next_rules
# =============================================================================

class TestGetNextRules:
    def test_always_includes_axiom(self):
        """Axiom is always a candidate rule."""
        seq = Sequent([A], B)
        rules = get_next_rules(seq)
        assert any(isinstance(r, Axiom) for r in rules)
    
    def test_always_includes_structural(self):
        """TrueRight and FalseLeft are always candidates."""
        seq = Sequent([A], B)
        rules = get_next_rules(seq)
        assert any(isinstance(r, TrueRight) for r in rules)
        assert any(isinstance(r, FalseLeft) for r in rules)
    
    def test_implies_right_for_implication_consequent(self):
        """ImpliesRight is a candidate when consequent is an implication."""
        seq = Sequent([], Implies(A, B))
        rules = get_next_rules(seq)
        assert any(isinstance(r, ImpliesRight) for r in rules)
    
    def test_and_right_for_conjunction_consequent(self):
        """AndRight is a candidate when consequent is a conjunction."""
        seq = Sequent([], And(A, B))
        rules = get_next_rules(seq)
        assert any(isinstance(r, AndRight) for r in rules)
    
    def test_or_right_for_disjunction_consequent(self):
        """Both OrRight rules are candidates when consequent is a disjunction."""
        seq = Sequent([], Or(A, B))
        rules = get_next_rules(seq)
        assert any(isinstance(r, OrRight1) for r in rules)
        assert any(isinstance(r, OrRight2) for r in rules)
    
    def test_and_left_for_conjunction_antecedent(self):
        """AndLeft is a candidate for each conjunction in antecedents."""
        conj = And(A, B)
        seq = Sequent([conj], C)
        rules = get_next_rules(seq)
        and_lefts = [r for r in rules if isinstance(r, AndLeft)]
        assert len(and_lefts) == 1
        assert and_lefts[0].conjunction == conj
    
    def test_or_left_for_disjunction_antecedent(self):
        """OrLeft is a candidate for each disjunction in antecedents."""
        disj = Or(A, B)
        seq = Sequent([disj], C)
        rules = get_next_rules(seq)
        or_lefts = [r for r in rules if isinstance(r, OrLeft)]
        assert len(or_lefts) == 1
        assert or_lefts[0].disjunction == disj
    
    def test_implies_left_for_implication_antecedent(self):
        """ImpliesLeft is a candidate for each implication in antecedents."""
        impl = Implies(A, B)
        seq = Sequent([impl], C)
        rules = get_next_rules(seq)
        impl_lefts = [r for r in rules if isinstance(r, ImpliesLeft)]
        assert len(impl_lefts) == 1
        assert impl_lefts[0].implication == impl
    
    def test_multiple_left_rules(self):
        """Multiple left rules generated for multiple antecedents."""
        impl = Implies(A, B)
        conj = And(B, C)
        seq = Sequent([impl, conj], D)
        rules = get_next_rules(seq)
        
        impl_lefts = [r for r in rules if isinstance(r, ImpliesLeft)]
        and_lefts = [r for r in rules if isinstance(r, AndLeft)]
        
        assert len(impl_lefts) == 1
        assert len(and_lefts) == 1


# =============================================================================
# Test build_proof_tree - Simple Proofs
# =============================================================================

class TestBuildProofTreeSimple:
    def test_axiom(self):
        """A ⊢ A is provable by axiom."""
        seq = Sequent([A], A)
        node = build_proof_tree(seq)
        
        assert node.is_provable
        assert isinstance(node, CompletedProofNode)
        assert isinstance(node.rule, Axiom)
    
    def test_true_introduction(self):
        """⊢ ⊤ is provable by TrueRight."""
        seq = Sequent([], PTrue())
        node = build_proof_tree(seq)
        
        assert node.is_provable
        assert isinstance(node, CompletedProofNode)
        assert isinstance(node.rule, TrueRight)
    
    def test_false_elimination(self):
        """⊥ ⊢ A is provable by FalseLeft."""
        seq = Sequent([PFalse()], A)
        node = build_proof_tree(seq)
        
        assert node.is_provable
        assert isinstance(node, CompletedProofNode)
        assert isinstance(node.rule, FalseLeft)
    
    def test_identity_theorem(self):
        """⊢ A → A is provable."""
        seq = Sequent([], Implies(A, A))
        node = build_proof_tree(seq)
        
        assert node.is_provable
        assert isinstance(node, InternalProofNode)
        assert node.depth == 2
        
        # Should have a branch with ImpliesRight
        branch = node.successful_branch()
        assert branch is not None
        rule, children = branch
        assert isinstance(rule, ImpliesRight)
        assert len(children) == 1
        assert isinstance(children[0], CompletedProofNode)
    
    def test_and_introduction(self):
        """A, B ⊢ A ∧ B is provable."""
        seq = Sequent([A, B], And(A, B))
        node = build_proof_tree(seq)
        
        assert node.is_provable
        assert isinstance(node, InternalProofNode)
        
        branch = node.successful_branch()
        assert branch is not None
        rule, children = branch
        assert isinstance(rule, AndRight)
        assert len(children) == 2
    
    def test_or_introduction_left(self):
        """A ⊢ A ∨ B is provable."""
        seq = Sequent([A], Or(A, B))
        node = build_proof_tree(seq)
        
        assert node.is_provable


# =============================================================================
# Test build_proof_tree - Complex Proofs
# =============================================================================

class TestBuildProofTreeComplex:
    def test_modus_ponens(self):
        """A, A → B ⊢ B is provable."""
        impl = Implies(A, B)
        seq = Sequent([A, impl], B)
        node = build_proof_tree(seq)
        
        assert node.is_provable
    
    def test_and_commutativity(self):
        """A ∧ B ⊢ B ∧ A is provable."""
        seq = Sequent([And(A, B)], And(B, A))
        node = build_proof_tree(seq)
        
        assert node.is_provable
    
    def test_or_commutativity(self):
        """A ∨ B ⊢ B ∨ A is provable."""
        seq = Sequent([Or(A, B)], Or(B, A))
        node = build_proof_tree(seq)
        
        assert node.is_provable
    
    def test_hypothetical_syllogism(self):
        """A → B, B → C ⊢ A → C is provable."""
        impl_ab = Implies(A, B)
        impl_bc = Implies(B, C)
        seq = Sequent([impl_ab, impl_bc], Implies(A, C))
        node = build_proof_tree(seq)
        
        assert node.is_provable
    
    def test_and_elimination_left(self):
        """A ∧ B ⊢ A is provable."""
        seq = Sequent([And(A, B)], A)
        node = build_proof_tree(seq)
        
        assert node.is_provable
    
    def test_and_elimination_right(self):
        """A ∧ B ⊢ B is provable."""
        seq = Sequent([And(A, B)], B)
        node = build_proof_tree(seq)
        
        assert node.is_provable
    
    def test_currying(self):
        """⊢ (A ∧ B → C) → (A → B → C) is provable."""
        seq = Sequent([], Implies(
            Implies(And(A, B), C),
            Implies(A, Implies(B, C))
        ))
        node = build_proof_tree(seq)
        
        assert node.is_provable
    
    def test_uncurrying(self):
        """⊢ (A → B → C) → (A ∧ B → C) is provable."""
        seq = Sequent([], Implies(
            Implies(A, Implies(B, C)),
            Implies(And(A, B), C)
        ))
        node = build_proof_tree(seq)
        
        assert node.is_provable
    
    def test_distribution_and_over_or(self):
        """A ∧ (B ∨ C) ⊢ (A ∧ B) ∨ (A ∧ C) is provable."""
        seq = Sequent(
            [And(A, Or(B, C))],
            Or(And(A, B), And(A, C))
        )
        node = build_proof_tree(seq)
        
        assert node.is_provable
    
    def test_distribution_or_over_and(self):
        """A ∨ (B ∧ C) ⊢ (A ∨ B) ∧ (A ∨ C) is provable."""
        seq = Sequent(
            [Or(A, And(B, C))],
            And(Or(A, B), Or(A, C))
        )
        node = build_proof_tree(seq)
        
        assert node.is_provable


# =============================================================================
# Test build_proof_tree - Unprovable Sequents
# =============================================================================

class TestUnprovableSequents:
    def test_unprovable_basic(self):
        """⊢ A is not provable (no assumptions)."""
        seq = Sequent([], A)
        node = build_proof_tree(seq)
        
        assert not node.is_provable
        assert isinstance(node, FailedProofNode)
    
    def test_unprovable_wrong_atom(self):
        """A ⊢ B is not provable."""
        seq = Sequent([A], B)
        node = build_proof_tree(seq)
        
        assert not node.is_provable
    
    def test_unprovable_implies_wrong_direction(self):
        """A → B ⊢ B → A is not provable (in general)."""
        seq = Sequent([Implies(A, B)], Implies(B, A))
        node = build_proof_tree(seq)
        
        assert not node.is_provable
    
    def test_unprovable_peirce_law(self):
        """⊢ ((A → B) → A) → A is not provable in NJ (requires classical logic)."""
        peirce = Implies(Implies(Implies(A, B), A), A)
        seq = Sequent([], peirce)
        node = build_proof_tree(seq)
        
        assert not node.is_provable
    
    def test_unprovable_excluded_middle(self):
        """⊢ A ∨ (A → ⊥) is not provable in NJ (requires classical logic)."""
        lem = Or(A, Implies(A, PFalse()))
        seq = Sequent([], lem)
        node = build_proof_tree(seq)
        
        assert not node.is_provable
    
    def test_unprovable_double_negation_elim(self):
        """⊢ ((A → ⊥) → ⊥) → A is not provable in NJ."""
        dne = Implies(Implies(Implies(A, PFalse()), PFalse()), A)
        seq = Sequent([], dne)
        node = build_proof_tree(seq)
        
        assert not node.is_provable


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    def test_empty_antecedents(self):
        """Handle empty antecedent list."""
        seq = Sequent([], PTrue())
        node = build_proof_tree(seq)
        assert node.is_provable
    
    def test_depth_limit(self):
        """Proof search respects depth limit."""
        seq = Sequent([], Implies(A, A))
        
        # With depth 1, not enough to prove A → A
        node = build_proof_tree(seq, max_depth=1)
        assert not node.is_provable
        
        # With depth 2, we can prove it
        node = build_proof_tree(seq, max_depth=2)
        assert node.is_provable
    
    def test_duplicate_antecedents(self):
        """Handle duplicate antecedents gracefully."""
        seq = Sequent([A, A, A], A)
        node = build_proof_tree(seq)
        assert node.is_provable
    
    def test_deeply_nested_proposition(self):
        """Handle deeply nested propositions."""
        nested = Implies(A, Implies(A, Implies(A, A)))
        seq = Sequent([], nested)
        node = build_proof_tree(seq)
        
        assert node.is_provable
        assert node.depth == 4  # 3 ImpliesRight + 1 Axiom
    
    def test_complex_antecedents(self):
        """Handle multiple complex antecedents."""
        seq = Sequent(
            [And(A, B), Or(C, D), Implies(A, C)],
            And(A, Or(C, D))
        )
        node = build_proof_tree(seq)
        
        assert node.is_provable


# =============================================================================
# Test Complete Proof Tree Properties
# =============================================================================

class TestCompleteProofTreeProperties:
    def test_all_branches_explored(self):
        """InternalProofNode contains all applicable rule branches."""
        seq = Sequent([A], Or(A, B))
        node = build_proof_tree(seq)
        
        assert node.is_provable
        assert isinstance(node, InternalProofNode)
        
        # Should have branches for OrRight1 and OrRight2
        rules_tried = [str(rule) for rule, _ in node.branches]
        assert "∨R₁" in rules_tried
        assert "∨R₂" in rules_tried
    
    def test_failed_branches_recorded(self):
        """Failed branches are still recorded in the tree."""
        seq = Sequent([A], Or(A, B))
        node = build_proof_tree(seq)
        
        assert isinstance(node, InternalProofNode)
        
        # OrRight2 branch (trying to prove B) should fail
        for rule, children in node.branches:
            if isinstance(rule, OrRight2):
                # Child should be trying to prove B from A
                assert len(children) == 1
                assert not children[0].is_provable
    
    def test_tree_size_reflects_exploration(self):
        """Tree size includes all explored nodes, not just successful path."""
        seq = Sequent([A], Or(A, B))
        node = build_proof_tree(seq)
        
        # Should have at least: root + OrRight1 child + OrRight2 child
        assert node.size >= 3
    
    def test_completed_nodes_are_leaves(self):
        """CompletedProofNode instances have no children (depth 1)."""
        seq = Sequent([A], A)
        node = build_proof_tree(seq)
        
        assert isinstance(node, CompletedProofNode)
        assert node.depth == 1
        assert node.size == 1
