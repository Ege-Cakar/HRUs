from __future__ import annotations

import numpy as np

from task.layer_gen.util.fol_rule_bank import FOLAtom, FOLSequent
from task.layer_gen.util import tokenize_layer_fol as tok


def _decode_single_completion(tokenizer, completion_tokens) -> str:
    statements = tokenizer.decode_completion_texts(completion_tokens)
    assert len(statements) == 1
    return statements[0]


def test_layer_fol_tokenize_roundtrip_prompt_and_completion() -> None:
    tokenizer = tok.build_tokenizer_from_identifiers(
        ["r0_1", "r0_2", "r1_1", "a", "b", "c", "x1", "x2"]
    )
    sequent = FOLSequent(
        ants=(FOLAtom("r0_1", ("a", "b")), FOLAtom("r0_2", ("b", "c"))),
        cons=FOLAtom("r1_1", ("a", "c")),
    )
    statement = "r0_1(a,b) ∧ r0_2(b,c) → r1_1(a,c)"

    prompt, completion = tokenizer.tokenize_example(sequent, statement)
    decoded_seq = tokenizer.decode_prompt(prompt)
    decoded_statement = _decode_single_completion(tokenizer, completion)

    assert decoded_seq == sequent
    assert decoded_statement == statement
    assert prompt[-1] == tokenizer.start_token_id
    assert completion[-1] == tokenizer.eot_token_id


def test_layer_fol_tokenize_roundtrip_completion_sequence() -> None:
    tokenizer = tok.build_tokenizer_from_identifiers(
        ["r0_1", "r1_1", "r2_1", "a", "b", "x1", "x2"]
    )
    statements = [
        "r0_1(a,b) → r1_1(a,b)",
        "r1_1(a,b) → r2_1(a,b)",
    ]

    completion = tokenizer.encode_completion_texts(statements)
    decoded = tokenizer.decode_completion_texts(completion)

    assert decoded == statements
    assert completion.count(tokenizer.sep_token_id) == 1
    assert completion[-1] == tokenizer.eot_token_id


def test_layered_predicates_are_char_tokenized_and_arg_commas_are_omitted() -> None:
    tokenizer = tok.build_tokenizer_from_identifiers(
        ["r3_20", "r4_1", "alice", "bob", "x1"]
    )
    completion = tokenizer.encode_completion_texts(["r3_20(alice,bob) → r4_1(alice,bob)"])
    decoded = tokenizer.decode_batch_ids([completion], include_special_tokens=False)[0]

    assert "r3_20" not in tokenizer.token_to_id
    for ch in "r3_20":
        assert ch in tokenizer.token_to_id
    assert "," not in decoded


def test_decode_prompt_requires_start() -> None:
    tokenizer = tok.build_tokenizer_from_identifiers(["r0_1", "r1_1", "a", "b"])
    sequent = FOLSequent(
        ants=(FOLAtom("r0_1", ("a",)),),
        cons=FOLAtom("r1_1", ("b",)),
    )
    prompt = tokenizer.tokenize_prompt(sequent)

    try:
        tokenizer.decode_prompt(prompt[:-1])
    except ValueError as exc:
        assert "START" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing START")


def test_try_decode_prompt_reports_error_without_raising_fol() -> None:
    tokenizer = tok.build_tokenizer_from_identifiers(["r0_1", "r1_1", "a", "b"])
    sequent = FOLSequent(
        ants=(FOLAtom("r0_1", ("a",)),),
        cons=FOLAtom("r1_1", ("b",)),
    )
    prompt = tokenizer.tokenize_prompt(sequent)

    decoded = tokenizer.try_decode_prompt(prompt)
    assert decoded.ok
    assert decoded.value == sequent
    assert decoded.error is None

    bad = tokenizer.try_decode_prompt(prompt[:-1])
    assert not bad.ok
    assert bad.value is None
    assert bad.error is not None


def test_decode_prompt_accepts_demo_prefixed_row_and_completion_suffix() -> None:
    tokenizer = tok.build_tokenizer_from_identifiers(
        ["r0_1", "r1_1", "r2_1", "a", "b", "x1"],
    )
    sequent = FOLSequent(
        ants=(FOLAtom("r0_1", ("a",)),),
        cons=FOLAtom("r2_1", ("b",)),
    )
    prompt = tokenizer.tokenize_prompt(sequent)
    demo = tokenizer.encode_completion_texts(["r0_1(a) → r1_1(a)"])
    completion = tokenizer.encode_completion_texts(["r1_1(a) → r2_1(b)"])
    row = np.array(
        demo[:-1] + [int(tokenizer.sep_token_id)] + prompt + completion + [0, 0],
        dtype=np.int32,
    )

    decoded = tokenizer.decode_prompt(row.tolist())

    assert decoded == sequent


def test_decode_batch_ids_from_task_like_arrays() -> None:
    tokenizer = tok.build_tokenizer_from_identifiers(["r0_1", "r1_1", "a", "b"])
    sequent = FOLSequent(
        ants=(FOLAtom("r0_1", ("a",)),),
        cons=FOLAtom("r1_1", ("b",)),
    )
    completion = tokenizer.encode_completion_texts(["r0_1(a) → r1_1(b)"])
    prompt = tokenizer.tokenize_prompt(sequent)

    batch = np.zeros((2, max(len(prompt), len(completion))), dtype=np.int32)
    batch[0, : len(prompt)] = np.asarray(prompt, dtype=np.int32)
    batch[1, : len(completion)] = np.asarray(completion, dtype=np.int32)

    decoded = tokenizer.decode_batch_ids(batch)
    assert decoded[0].endswith("<START>")
    assert decoded[1].endswith("<EOT>")


def test_special_tokens_are_in_unified_maps() -> None:
    tokenizer = tok.build_tokenizer_from_identifiers(["r0_1", "a"])
    assert tokenizer.char_to_id("<PAD>") == 0
    assert tokenizer.id_to_char(0) == "<PAD>"
    assert tokenizer.char_to_id("<SEP>") == tokenizer.sep_token_id
    assert tokenizer.id_to_char(tokenizer.sep_token_id) == "<SEP>"
    assert tokenizer.char_to_id("<START>") == tokenizer.start_token_id
    assert tokenizer.id_to_char(tokenizer.start_token_id) == "<START>"
    assert tokenizer.char_to_id("<EOT>") == tokenizer.eot_token_id
    assert tokenizer.id_to_char(tokenizer.eot_token_id) == "<EOT>"


def test_try_decode_completion_helpers_report_error_without_raising_fol() -> None:
    tokenizer = tok.build_tokenizer_from_identifiers(
        ["r0_1", "r1_1", "r2_1", "a", "b", "x1"],
    )
    statement = "r0_1(a,b) → r1_1(a,b)"
    completion = tokenizer.encode_completion_texts([statement])

    decoded_texts = tokenizer.try_decode_completion_texts(completion)
    assert decoded_texts.ok
    assert decoded_texts.value == [statement]
    assert decoded_texts.error is None

    decoded_clause = tokenizer.try_decode_completion_clause(completion)
    assert decoded_clause.ok
    assert decoded_clause.value is not None
    assert decoded_clause.error is None

    bad = tokenizer.try_decode_completion_texts(completion[:-1])
    assert not bad.ok
    assert bad.value is None
    assert bad.error is not None


def test_from_dict_rejects_v1_payload() -> None:
    payload_v1 = {
        "version": "layer_fol_v1_compact",
        "pad_idx": 0,
        "logic_tokens": ["⊢", "∧", "→", "(", ")", ","],
        "sep_token_id": 7,
        "eot_token_id": 8,
        "identifier_to_id": {"a": 9, "r0_1": 10},
        "vocab_size": 11,
    }

    try:
        tok.FOLLayerTokenizer.from_dict(payload_v1)
    except ValueError as exc:
        assert "Strict migration enabled" in str(exc)
    else:
        raise AssertionError("Expected ValueError for v1 tokenizer payload")


def test_fresh_predicate_r_abcd_is_char_tokenized() -> None:
    tokenizer = tok.build_tokenizer_from_identifiers(
        ["r_a1b2", "r1_1", "a", "b", "x1"],
        predicate_identifiers=["r_a1b2", "r1_1"],
    )
    # r_a1b2 should not be a single token.
    assert "r_a1b2" not in tokenizer.token_to_id
    # Individual chars should be tokens.
    for ch in "r_a1b2":
        assert ch in tokenizer.token_to_id

    # Roundtrip encode/decode should work.
    statement = "r_a1b2(a,b) → r1_1(a,b)"
    completion = tokenizer.encode_completion_texts([statement])
    decoded = _decode_single_completion(tokenizer, completion)
    assert decoded == statement


def test_standard_predicates_still_char_tokenized_alongside_fresh() -> None:
    tokenizer = tok.build_tokenizer_from_identifiers(
        ["r0_1", "r_x9z3", "a", "x1"],
        predicate_identifiers=["r0_1", "r_x9z3"],
    )
    # Both forms should be char-tokenized.
    assert "r0_1" not in tokenizer.token_to_id
    assert "r_x9z3" not in tokenizer.token_to_id
    for ch in "r0_1":
        assert ch in tokenizer.token_to_id
    for ch in "r_x9z3":
        assert ch in tokenizer.token_to_id


def test_arity_zero_atom_roundtrip_tokenize() -> None:
    """Encode/decode arity-0 atoms (no parentheses)."""
    tokenizer = tok.build_tokenizer_from_identifiers(
        ["r0_1", "r0_2", "r1_1", "a", "b", "x1"],
        predicate_identifiers=["r0_1", "r0_2", "r1_1"],
    )
    # Arity-0 atoms appear without parentheses.
    # Test a clause: r0_1 ∧ r0_2 → r1_1
    statement = "r0_1 ∧ r0_2 → r1_1"
    completion = tokenizer.encode_completion_texts([statement])
    decoded = _decode_single_completion(tokenizer, completion)
    assert decoded == statement

    # Test a prompt with arity-0 atoms.
    sequent = FOLSequent(
        ants=(FOLAtom("r0_1", ()), FOLAtom("r0_2", ())),
        cons=FOLAtom("r1_1", ()),
    )
    prompt = tokenizer.tokenize_prompt(sequent)
    decoded_seq = tokenizer.decode_prompt(prompt)
    assert decoded_seq == sequent

    # Mixed: some arity-0 and some arity-1.
    statement_mixed = "r0_1 ∧ r0_2(a) → r1_1(b)"
    completion_mixed = tokenizer.encode_completion_texts([statement_mixed])
    decoded_mixed = _decode_single_completion(tokenizer, completion_mixed)
    assert decoded_mixed == statement_mixed
