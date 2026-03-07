from __future__ import annotations

import pytest

from task.lean_gen.util import tokenize_lean_ar


def test_tokenize_lean_ar_round_trip_unicode() -> None:
    proof_state = "h : Nat\n⊢ ∀ n, n + 0 = n"
    next_tactic = "intro n\nsimp"

    prompt_tokens = tokenize_lean_ar.tokenize_prompt(proof_state)
    completion_tokens = tokenize_lean_ar.encode_completion(next_tactic)

    assert prompt_tokens[-1] == tokenize_lean_ar.start_token_id
    assert completion_tokens[-1] == tokenize_lean_ar.eot_token_id

    decoded_state = tokenize_lean_ar.decode_prompt(prompt_tokens)
    decoded_tactic = tokenize_lean_ar.decode_completion(completion_tokens)

    assert decoded_state == proof_state
    assert decoded_tactic == next_tactic


def test_tokenize_lean_ar_multi_completion_example() -> None:
    proof_state = "⊢ True"
    tactics = ["trivial", "exact True.intro"]

    prompt, completions = tokenize_lean_ar.tokenize_example(proof_state, tactics)

    assert prompt[-1] == tokenize_lean_ar.start_token_id
    assert len(completions) == 2
    assert all(comp[-1] == tokenize_lean_ar.eot_token_id for comp in completions)


def test_tokenize_lean_ar_decode_rejects_malformed_sequences() -> None:
    with pytest.raises(tokenize_lean_ar.TokenizationError):
        tokenize_lean_ar.decode_prompt([3, 4, 5])

    with pytest.raises(tokenize_lean_ar.TokenizationError):
        tokenize_lean_ar.decode_completion([3, 4, 5])


def test_tokenize_lean_ar_vocab_size() -> None:
    assert tokenize_lean_ar.vocab_size() == 260
