"""Tests for proposition sampling utilities."""

import random

import pytest

from task.prop_gen.util.elem import Atom, Implies, PFalse, PTrue
from task.prop_gen.util.sample import sample_imply


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
