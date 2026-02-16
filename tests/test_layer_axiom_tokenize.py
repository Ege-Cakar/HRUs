from __future__ import annotations

from task.layer_gen.util import tokenize_layer_axiom as tok
from task.prop_gen.util.elem import And, Atom, Implies, Sequent


def test_layer_axiom_tokenize_roundtrip_prompt_and_completion() -> None:
    lhs = And(Atom("p1_2"), Atom("p1_4"))
    rhs = And(Atom("p2_1"), Atom("p2_3"))
    statement = str(Implies(lhs, rhs))

    sequent = Sequent([Atom("p1_2"), Atom("p1_4")], Atom("p3_1"))
    prompt, completion = tok.tokenize_example(sequent, statement)

    decoded_seq = tok.decode_prompt(prompt)
    decoded_statement = tok.decode_completion_text(completion)

    assert decoded_seq == sequent
    assert decoded_statement == statement
    assert prompt[-1] == tok.sep_token_id
    assert completion[-1] == tok.eot_token_id


def test_decode_prompt_requires_sep() -> None:
    sequent = Sequent([Atom("p0_1")], Atom("p1_1"))
    prompt = tok.tokenize_prompt(sequent)
    bad_prompt = prompt[:-1]

    try:
        tok.decode_prompt(bad_prompt)
    except ValueError as exc:
        assert "SEP" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing SEP")
