"""Tests for dataset generation tokenization utilities."""

from __future__ import annotations

from task.prop_gen.util import tokenize as tokenize_util

from task.prop_gen.util.elem import (
    And,
    AndLeft,
    AndRight,
    Axiom,
    Atom,
    FalseLeft,
    Implies,
    ImpliesLeft,
    ImpliesRight,
    NegationRight,
    Or,
    OrLeft,
    OrRight1,
    OrRight2,
    PFalse,
    PTrue,
    Sequent,
    TrueRight,
)
def _rule_sig(rule):
    if isinstance(rule, ImpliesLeft):
        return ("ImpliesLeft", rule.implication)
    if isinstance(rule, AndLeft):
        return ("AndLeft", rule.conjunction)
    if isinstance(rule, OrLeft):
        return ("OrLeft", rule.disjunction)
    return (type(rule).__name__,)


def test_tokenize_decode_roundtrip_with_left_rules():
    p1 = Atom("p1")
    p2 = Atom("p2")
    p3 = Atom("p3")
    p4 = Atom("p4")
    p5 = Atom("p5")
    ant1 = Implies(p1, p2)
    ant2 = And(p3, p4)
    ant3 = Or(p2, PTrue())
    sequent = Sequent([ant1, ant2, ant3, p5], Implies(p1, PFalse()))
    rule = ImpliesLeft(ant1)

    tokens = tokenize_util.tokenize((sequent, rule))
    assert tokens[1] == (3, 1)

    decoded_sequent, decoded_rule = tokenize_util.decode(tokens)
    assert decoded_sequent == sequent
    assert _rule_sig(decoded_rule) == _rule_sig(rule)


def test_tokenize_decode_roundtrip_empty_ants():
    p1 = Atom("p1")
    p2 = Atom("p2")
    sequent = Sequent([], Or(p1, p2))
    rule = OrRight1()

    decoded_sequent, decoded_rule = tokenize_util.decode(
        tokenize_util.tokenize((sequent, rule))
    )
    assert decoded_sequent == sequent
    assert _rule_sig(decoded_rule) == _rule_sig(rule)
