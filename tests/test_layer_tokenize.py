from __future__ import annotations

import numpy as np

from task.layer_gen.util import tokenize_layer as tok
from task.prop_gen.util.elem import Atom, Sequent


def test_layer_tokenize_roundtrip_prompt_and_completion() -> None:
    tokenizer = tok.build_tokenizer_from_atoms(
        ["p1_2", "p1_4", "p2_1", "p2_3", "p3_1"]
    )
    statement = "p1_2 ∧ p1_4 → p2_1 ∧ p2_3"

    sequent = Sequent([Atom("p1_2"), Atom("p1_4")], Atom("p3_1"))
    prompt, completion = tokenizer.tokenize_example(sequent, statement)

    decoded_seq = tokenizer.decode_prompt(prompt)
    decoded_statement = tokenizer.decode_completion_text(completion)

    assert decoded_seq == sequent
    assert decoded_statement == statement
    assert prompt[-1] == tokenizer.sep_token_id
    assert completion[-1] == tokenizer.eot_token_id
    assert tokenizer.vocab_size == tokenizer.eot_token_id + 1 + 5


def test_decode_prompt_requires_sep() -> None:
    tokenizer = tok.build_tokenizer_from_atoms(["p0_1", "p1_1"])
    sequent = Sequent([Atom("p0_1")], Atom("p1_1"))
    prompt = tokenizer.tokenize_prompt(sequent)
    bad_prompt = prompt[:-1]

    try:
        tokenizer.decode_prompt(bad_prompt)
    except ValueError as exc:
        assert "SEP" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing SEP")


def test_tokenizer_compact_atom_ids_are_contiguous() -> None:
    tokenizer = tok.build_tokenizer_from_atoms(["p3_2", "p0_1", "p1_4"])
    expected = {
        "p0_1": tokenizer.eot_token_id + 1,
        "p1_4": tokenizer.eot_token_id + 2,
        "p3_2": tokenizer.eot_token_id + 3,
    }
    assert tokenizer.atom_to_token == expected


def test_decode_completion_rejects_parenthesized_legacy_form() -> None:
    tokenizer = tok.build_tokenizer_from_atoms(["p1_2", "p1_4", "p2_1", "p2_3"])
    legacy = "((p1_2 ∧ p1_4) → (p2_1 ∧ p2_3))"
    completion = tokenizer.encode_completion(legacy)

    try:
        tokenizer.decode_completion_text(completion)
    except ValueError as exc:
        assert "does not allow parentheses" in str(exc)
    else:
        raise AssertionError("Expected ValueError for legacy parenthesized completion format")


def test_decode_batch_ids_from_task_like_arrays() -> None:
    tokenizer = tok.build_tokenizer_from_atoms(["p0_1", "p1_1", "p1_2"])
    sequent = Sequent([Atom("p0_1")], Atom("p1_1"))
    prompt = tokenizer.tokenize_prompt(sequent)
    completion = tokenizer.encode_completion("p0_1 ∧ p1_2 → p1_1")

    batch = np.zeros((2, max(len(prompt), len(completion))), dtype=np.int32)
    batch[0, : len(prompt)] = np.asarray(prompt, dtype=np.int32)
    batch[1, : len(completion)] = np.asarray(completion, dtype=np.int32)

    decoded = tokenizer.decode_batch_ids(batch)
    assert decoded[0] == "p0_1⊢p1_1<SEP>"
    assert decoded[1] == "p0_1∧p1_2→p1_1<EOT>"
