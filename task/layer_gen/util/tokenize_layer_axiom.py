"""Tokenization utilities for layered axiom-sequence tasks."""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    from task.prop_gen.util.elem import (
        And,
        Atom,
        Implies,
        Proposition,
        Sequent,
    )
except ModuleNotFoundError:
    ROOT = Path(__file__).resolve().parents[3]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    from task.prop_gen.util.elem import (  # type: ignore
        And,
        Atom,
        Implies,
        Proposition,
        Sequent,
    )

pad_idx = 0

logic_char_to_id = {
    "⊢": 1,
    "∧": 2,
    "→": 3,
    "(": 4,
    ")": 5,
    ",": 6,
}

id_to_logic_char = {v: k for k, v in logic_char_to_id.items()}

_RULE_BASE = 64
sep_token_id = _RULE_BASE
eot_token_id = _RULE_BASE + 1

var_base = 256
_atom_slot = 4096
_atom_re = re.compile(r"p(\d+)_(\d+)$")


def atom_to_id(atom_name: str) -> int:
    match = _atom_re.fullmatch(atom_name)
    if match is None:
        raise ValueError(f"Unsupported atom name: {atom_name}")
    layer = int(match.group(1))
    idx = int(match.group(2))
    if layer < 0 or idx < 0:
        raise ValueError(f"Atom indices must be non-negative: {atom_name}")
    if idx >= _atom_slot:
        raise ValueError(f"Atom index too large for tokenizer slot {_atom_slot}: {atom_name}")
    return var_base + layer * _atom_slot + idx


def id_to_atom(token_id: int) -> str:
    if token_id < var_base:
        raise ValueError(f"Token {token_id} is not an atom token")
    offset = token_id - var_base
    layer, idx = divmod(offset, _atom_slot)
    return f"p{layer}_{idx}"


def char_to_id(token: str) -> int:
    if token in logic_char_to_id:
        return logic_char_to_id[token]
    if token.startswith("p"):
        return atom_to_id(token)
    raise ValueError(f"Unknown token symbol: {token}")


def id_to_char(token_id: int) -> str:
    if token_id in id_to_logic_char:
        return id_to_logic_char[token_id]
    if token_id >= var_base:
        return id_to_atom(token_id)
    raise ValueError(f"Unknown token id: {token_id}")


def _tokenize_prop_text(text: str) -> list[int]:
    tokens: list[int] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "p":
            j = i + 1
            while j < len(text) and text[j].isdigit():
                j += 1
            if j >= len(text) or text[j] != "_":
                raise ValueError(f"Invalid atom token at index {i} in {text!r}")
            j += 1
            k = j
            while k < len(text) and text[k].isdigit():
                k += 1
            if k == j:
                raise ValueError(f"Invalid atom suffix at index {i} in {text!r}")
            tokens.append(char_to_id(text[i:k]))
            i = k
            continue

        tokens.append(char_to_id(ch))
        i += 1

    return tokens


def tokenize_prop(prop: Proposition) -> list[int]:
    return _tokenize_prop_text(str(prop))


def tokenize_prompt(sequent: Sequent) -> list[int]:
    return _tokenize_prop_text(str(sequent)) + [sep_token_id]


def encode_completion(statement_text: str) -> list[int]:
    return _tokenize_prop_text(statement_text) + [eot_token_id]


def tokenize_example(sequent: Sequent, statement_text: str) -> tuple[list[int], list[int]]:
    return tokenize_prompt(sequent), encode_completion(statement_text)


def _parse_prop(tokens: list[str], start: int) -> tuple[Proposition, int]:
    if start >= len(tokens):
        raise ValueError("Unexpected end of tokens while parsing proposition.")

    tok = tokens[start]
    if tok == "(":
        left, idx = _parse_prop(tokens, start + 1)
        if idx >= len(tokens):
            raise ValueError("Unexpected end of tokens after left operand.")
        op = tokens[idx]
        idx += 1
        right, idx = _parse_prop(tokens, idx)
        if idx >= len(tokens) or tokens[idx] != ")":
            raise ValueError("Expected ')' after binary proposition.")
        idx += 1

        if op == "∧":
            return And(left, right), idx
        if op == "→":
            return Implies(left, right), idx
        raise ValueError(f"Unknown binary operator: {op}")

    if tok.startswith("p"):
        return Atom(tok), start + 1

    raise ValueError(f"Unknown proposition token: {tok}")


def _parse_prop_list(tokens: list[str]) -> list[Proposition]:
    props: list[Proposition] = []
    idx = 0
    while idx < len(tokens):
        prop, idx = _parse_prop(tokens, idx)
        props.append(prop)
        if idx < len(tokens):
            if tokens[idx] != ",":
                raise ValueError(f"Expected ',' between antecedents, found {tokens[idx]}")
            idx += 1
    return props


def decode_prompt(prompt_tokens: list[int]) -> Sequent:
    if not prompt_tokens:
        raise ValueError("Prompt cannot be empty.")
    if prompt_tokens[-1] != sep_token_id:
        raise ValueError("Prompt must terminate with SEP token.")

    sequent_tokens = prompt_tokens[:-1]
    symbols = [id_to_char(tok) for tok in sequent_tokens]
    if "⊢" not in symbols:
        raise ValueError("Prompt sequent missing turnstile '⊢'.")

    turnstile_idx = symbols.index("⊢")
    ants_tokens = symbols[:turnstile_idx]
    cons_tokens = symbols[turnstile_idx + 1 :]
    if not cons_tokens:
        raise ValueError("Prompt sequent missing consequent.")

    ants = _parse_prop_list(ants_tokens) if ants_tokens else []
    cons, end_idx = _parse_prop(cons_tokens, 0)
    if end_idx != len(cons_tokens):
        raise ValueError("Extra tokens after parsing consequent.")

    return Sequent(ants, cons)


def decode_completion_prop(completion_tokens: list[int]) -> Proposition:
    if len(completion_tokens) < 2:
        raise ValueError("Completion must include proposition tokens and EOT.")
    if completion_tokens[-1] != eot_token_id:
        raise ValueError("Completion must terminate with EOT token.")

    symbols = [id_to_char(tok) for tok in completion_tokens[:-1]]
    prop, idx = _parse_prop(symbols, 0)
    if idx != len(symbols):
        raise ValueError("Extra symbols in completion.")
    return prop


def decode_completion_text(completion_tokens: list[int]) -> str:
    return str(decode_completion_prop(completion_tokens))
