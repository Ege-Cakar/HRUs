"""Tokenization utilities for Lean proof-state to tactic autoregressive tasks."""

from __future__ import annotations

from typing import Iterable

pad_idx = 0
sep_token_id = 1
start_token_id = 2
eot_token_id = 3

_BYTE_BASE = 4
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
    """Encode a proof state into prompt tokens, ending with START."""
    return _encode_text(proof_state) + [start_token_id]


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
    if not prompt_tokens:
        raise TokenizationError("Prompt cannot be empty.")
    nonpad = [int(tok) for tok in prompt_tokens if int(tok) != pad_idx]
    start_positions = [idx for idx, tok in enumerate(nonpad) if tok == start_token_id]
    if len(start_positions) != 1:
        raise TokenizationError("Prompt must contain exactly one START token.")
    start_idx = int(start_positions[0])
    body = nonpad[:start_idx]
    sep_positions = [idx for idx, tok in enumerate(body) if tok == sep_token_id]
    body_start = int(sep_positions[-1]) + 1 if sep_positions else 0
    body = body[body_start:]
    if not body:
        raise TokenizationError("Prompt body cannot be empty.")
    return _decode_text(body)


def decode_completion(completion_tokens: list[int]) -> str:
    if not completion_tokens or completion_tokens[-1] != eot_token_id:
        raise TokenizationError("Completion must terminate with EOT token.")
    return _decode_text(completion_tokens[:-1])


def decode(tokens: tuple[list[int], list[int]]) -> tuple[str, str]:
    """Decode `(prompt_tokens, completion_tokens)` into text."""
    prompt_tokens, completion_tokens = tokens
    return decode_prompt(prompt_tokens), decode_completion(completion_tokens)


def vocab_size() -> int:
    """Small fixed vocabulary: PAD, SEP, START, EOT, plus 256 byte values."""
    return _BYTE_MAX + 1
