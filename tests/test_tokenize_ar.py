"""Tests for autoregressive tokenization utilities."""

from __future__ import annotations

from task.prop_gen.util import tokenize_ar
from task.prop_gen.util.elem import (
    Atom,
    Implies,
    ImpliesLeft,
    ImpliesRight,
    PFalse,
    Sequent,
    Unprovable,
)


def test_tokenize_ar_roundtrip_targeted_rule():
    p1 = Atom("p1")
    p2 = Atom("p2")
    ant = Implies(p1, p2)
    sequent = Sequent([ant], p2)
    rule = ImpliesLeft(ant)

    prompt, completions = tokenize_ar.tokenize((sequent, [rule]))
    assert prompt[-1] == tokenize_ar.start_token_id
    assert len(completions) == 1
    assert completions[0][-1] == tokenize_ar.eot_token_id

    decoded_seq, decoded_rule = tokenize_ar.decode((prompt, completions[0]))
    assert decoded_seq == sequent
    assert isinstance(decoded_rule, ImpliesLeft)
    assert decoded_rule.implication == ant


def test_tokenize_ar_roundtrip_non_targeted_rule():
    p1 = Atom("p1")
    sequent = Sequent([], Implies(p1, p1))
    rule = ImpliesRight()

    prompt, completions = tokenize_ar.tokenize((sequent, [rule]))
    decoded_seq, decoded_rule = tokenize_ar.decode((prompt, completions[0]))

    assert decoded_seq == sequent
    assert isinstance(decoded_rule, ImpliesRight)


def test_tokenize_ar_unprovable_completion():
    p1 = Atom("p1")
    sequent = Sequent([], p1)

    completion = tokenize_ar.encode_completion(sequent, Unprovable())
    decoded = tokenize_ar.decode_completion(sequent, completion)

    assert completion[0] == tokenize_ar.rule_type_to_id[Unprovable]
    assert completion[-1] == tokenize_ar.eot_token_id
    assert isinstance(decoded, Unprovable)


def test_decode_prompt_requires_start():
    sequent = Sequent([], PFalse())
    prompt = tokenize_ar.tokenize_prompt(sequent)

    bad_prompt = prompt[:-1]
    try:
        tokenize_ar.decode_prompt(bad_prompt)
    except ValueError as exc:
        assert "START" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing START")
