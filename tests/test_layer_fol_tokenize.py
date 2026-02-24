from __future__ import annotations

import numpy as np

from task.layer_gen.util.fol_rule_bank import FOLAtom, FOLSequent
from task.layer_gen.util import tokenize_layer_fol as tok


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
    decoded_statement = tokenizer.decode_completion_text(completion)

    assert decoded_seq == sequent
    assert decoded_statement == statement
    assert prompt[-1] == tokenizer.sep_token_id
    assert completion[-1] == tokenizer.eot_token_id


def test_decode_prompt_requires_sep() -> None:
    tokenizer = tok.build_tokenizer_from_identifiers(["r0_1", "r1_1", "a", "b"])
    sequent = FOLSequent(
        ants=(FOLAtom("r0_1", ("a",)),),
        cons=FOLAtom("r1_1", ("b",)),
    )
    prompt = tokenizer.tokenize_prompt(sequent)

    try:
        tokenizer.decode_prompt(prompt[:-1])
    except ValueError as exc:
        assert "SEP" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing SEP")


def test_decode_batch_ids_from_task_like_arrays() -> None:
    tokenizer = tok.build_tokenizer_from_identifiers(["r0_1", "r1_1", "a", "b"])
    sequent = FOLSequent(
        ants=(FOLAtom("r0_1", ("a",)),),
        cons=FOLAtom("r1_1", ("b",)),
    )
    completion = tokenizer.encode_completion("r0_1(a) → r1_1(b)")
    prompt = tokenizer.tokenize_prompt(sequent)

    batch = np.zeros((2, max(len(prompt), len(completion))), dtype=np.int32)
    batch[0, : len(prompt)] = np.asarray(prompt, dtype=np.int32)
    batch[1, : len(completion)] = np.asarray(completion, dtype=np.int32)

    decoded = tokenizer.decode_batch_ids(batch)
    assert decoded[0].endswith("<SEP>")
    assert decoded[1].endswith("<EOT>")


def test_special_tokens_are_in_unified_maps() -> None:
    tokenizer = tok.build_tokenizer_from_identifiers(["r0_1", "a"])
    assert tokenizer.char_to_id("<PAD>") == 0
    assert tokenizer.id_to_char(0) == "<PAD>"
    assert tokenizer.char_to_id("<SEP>") == tokenizer.sep_token_id
    assert tokenizer.id_to_char(tokenizer.sep_token_id) == "<SEP>"
    assert tokenizer.char_to_id("<EOT>") == tokenizer.eot_token_id
    assert tokenizer.id_to_char(tokenizer.eot_token_id) == "<EOT>"


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
