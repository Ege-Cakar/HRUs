"""Tests for proposition sampling utilities."""

import random

import pytest

from task.prop_gen.util.elem import Atom, Implies, PFalse, PTrue, Sequent, Unprovable
from task.prop_gen.util.sample import (
    count_sequent_symbols,
    list_sequents_allow_false,
    sample_imply,
    sample_imply_no_true,
)


def _count_atoms(prop) -> int:
    if isinstance(prop, (Atom, PTrue, PFalse)):
        return 1
    if isinstance(prop, Implies):
        return _count_atoms(prop.left) + _count_atoms(prop.right)
    raise AssertionError(f"Unexpected proposition node: {type(prop)}")


def _assert_only_implies(prop) -> None:
    if isinstance(prop, (Atom, PTrue, PFalse)):
        return
    assert isinstance(prop, Implies)
    _assert_only_implies(prop.left)
    _assert_only_implies(prop.right)


def _contains_true(prop) -> bool:
    if isinstance(prop, PTrue):
        return True
    if isinstance(prop, (Atom, PFalse)):
        return False
    if isinstance(prop, Implies):
        return _contains_true(prop.left) or _contains_true(prop.right)
    raise AssertionError(f"Unexpected proposition node: {type(prop)}")


def test_sample_imply_size_one():
    random.seed(0)
    prop = sample_imply(n_vars=3, size=1)
    assert _count_atoms(prop) == 1
    assert not isinstance(prop, Implies)


def test_sample_imply_atom_count_and_shape():
    random.seed(1)
    prop = sample_imply(n_vars=2, size=5)
    assert _count_atoms(prop) == 5
    _assert_only_implies(prop)


def test_sample_imply_zero_vars():
    random.seed(2)
    prop = sample_imply(n_vars=0, size=3)
    assert _count_atoms(prop) == 3
    _assert_only_implies(prop)


def test_sample_imply_allows_constants(monkeypatch):
    monkeypatch.setattr(random, "randrange", lambda n: n - 2)
    prop = sample_imply(n_vars=4, size=1)
    assert isinstance(prop, PTrue)


def test_sample_imply_validates_args():
    with pytest.raises(ValueError):
        sample_imply(n_vars=-1, size=1)
    with pytest.raises(ValueError):
        sample_imply(n_vars=1, size=0)


def test_sample_imply_no_true_excludes_true_constant():
    random.seed(3)
    prop = sample_imply_no_true(n_vars=2, size=16)
    assert not _contains_true(prop)


def test_count_sequent_symbols_counts_entire_sequent():
    p1 = Atom("p1")
    p2 = Atom("p2")
    p3 = Atom("p3")
    seq = Sequent(
        ants=[Implies(p1, PFalse()), Implies(p2, p3)],
        cons=Implies(p2, p1),
    )
    assert count_sequent_symbols(seq) == 6


def test_list_sequents_allow_false_marks_unprovable_root():
    prop = Atom("p1")
    assert list_sequents_allow_false(prop, allow_false=False) == []

    examples = list_sequents_allow_false(prop, allow_false=True)
    assert len(examples) == 1
    sequent, rules = examples[0]
    assert sequent == Sequent([], prop)
    assert len(rules) == 1
    assert isinstance(rules[0], Unprovable)
