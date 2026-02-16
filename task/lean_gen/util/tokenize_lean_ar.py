"""Tokenization utilities for Lean proof-state to tactic autoregressive tasks."""

from __future__ import annotations

from typing import Iterable

pad_idx = 0
sep_token_id = 1
eot_token_id = 2

_BYTE_BASE = 3
_BYTE_MAX = _BYTE_BASE + 255


class TokenizationError(ValueError):
    """Raised when encoded token sequences are malformed."""


def _encode_text(text: str) -> list[int]:
    return [byte + _BYTE_BASE for byte in text.encode("utf-8")]


def _decode_text(tokens: Iterable[int]) -> str:
    raw = []
    for token in tokens:
        if token < _BYTE_BASE or token > _BYTE_MAX:
            raise TokenizationError(f"Invalid text token id: {token}")
        raw.append(token - _BYTE_BASE)
    return bytes(raw).decode("utf-8")


def tokenize_prompt(proof_state: str) -> list[int]:
    """Encode a proof state into prompt tokens, ending with SEP."""
    return _encode_text(proof_state) + [sep_token_id]


def encode_completion(next_tactic: str) -> list[int]:
    """Encode a tactic into completion tokens, ending with EOT."""
    return _encode_text(next_tactic) + [eot_token_id]


def tokenize_example(
    proof_state: str,
    next_tactic: str | list[str],
) -> tuple[list[int], list[list[int]]]:
    """Encode one training example.

    The completion side supports one or more valid tactics so callers can represent
    ambiguity when multiple next tactics are acceptable.
    """
    prompt = tokenize_prompt(proof_state)
    if isinstance(next_tactic, str):
        tactics = [next_tactic]
    else:
        tactics = list(next_tactic)

    completions = [encode_completion(tactic) for tactic in tactics]
    return prompt, completions


def decode_prompt(prompt_tokens: list[int]) -> str:
    if not prompt_tokens or prompt_tokens[-1] != sep_token_id:
        raise TokenizationError("Prompt must terminate with SEP token.")
    return _decode_text(prompt_tokens[:-1])


def decode_completion(completion_tokens: list[int]) -> str:
    if not completion_tokens or completion_tokens[-1] != eot_token_id:
        raise TokenizationError("Completion must terminate with EOT token.")
    return _decode_text(completion_tokens[:-1])


def decode(tokens: tuple[list[int], list[int]]) -> tuple[str, str]:
    """Decode `(prompt_tokens, completion_tokens)` into text."""
    prompt_tokens, completion_tokens = tokens
    return decode_prompt(prompt_tokens), decode_completion(completion_tokens)


def vocab_size() -> int:
    """Small fixed vocabulary: PAD, SEP, EOT, plus 256 byte values."""
    return _BYTE_MAX + 1
