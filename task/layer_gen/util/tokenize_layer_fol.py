"""Tokenization utilities for layered first-order tasks."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from .fol_rule_bank import (
    FOLAtom,
    FOLRuleBank,
    FOLSequent,
    parse_clause_text,
    parse_conjunction_text,
    parse_sequent_text,
)

pad_idx = 0
TOKENIZER_VERSION = "layer_fol_v2_unified"

PAD_TOKEN = "<PAD>"
SEP_TOKEN = "<SEP>"
EOT_TOKEN = "<EOT>"

_LOGIC_TOKENS = ("⊢", "∧", "→", "(", ")", ",")
_RESERVED_TOKENS = (PAD_TOKEN, SEP_TOKEN, EOT_TOKEN)
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _format_conjunction(atoms: tuple[FOLAtom, ...]) -> str:
    return " ∧ ".join(atom.text for atom in atoms)


def _build_vocab(
    identifiers: Iterable[str],
) -> tuple[dict[str, int], dict[int, str], int, int]:
    unique_sorted = sorted(set(str(tok) for tok in identifiers))
    for token in unique_sorted:
        if token in _RESERVED_TOKENS:
            raise ValueError(f"Identifier token is reserved for special use: {token!r}")
        if _IDENTIFIER_RE.fullmatch(token) is None:
            raise ValueError(f"Invalid identifier token: {token!r}")

    token_to_id: dict[str, int] = {PAD_TOKEN: pad_idx}
    next_id = pad_idx + 1

    for token in _LOGIC_TOKENS:
        token_to_id[token] = next_id
        next_id += 1

    sep_token_id = next_id
    token_to_id[SEP_TOKEN] = sep_token_id
    next_id += 1

    eot_token_id = next_id
    token_to_id[EOT_TOKEN] = eot_token_id
    next_id += 1

    for token in unique_sorted:
        token_to_id[token] = next_id
        next_id += 1

    id_to_token = {idx: token for token, idx in token_to_id.items()}
    return token_to_id, id_to_token, sep_token_id, eot_token_id


@dataclass(frozen=True)
class FOLLayerTokenizer:
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]
    sep_token_id: int
    eot_token_id: int
    version: str = TOKENIZER_VERSION

    @classmethod
    def from_identifiers(cls, identifiers: Iterable[str]) -> "FOLLayerTokenizer":
        token_to_id, id_to_token, sep_token_id, eot_token_id = _build_vocab(identifiers)
        return cls(
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            sep_token_id=sep_token_id,
            eot_token_id=eot_token_id,
        )

    @classmethod
    def from_rule_bank(cls, rule_bank: FOLRuleBank) -> "FOLLayerTokenizer":
        identifiers: set[str] = set(rule_bank.constants)
        identifiers.update(rule_bank.predicate_arities)
        identifiers.update(f"x{idx}" for idx in range(1, int(rule_bank.vars_per_rule_max) + 1))
        return cls.from_identifiers(sorted(identifiers))

    @classmethod
    def from_dict(cls, payload: dict) -> "FOLLayerTokenizer":
        version = str(payload.get("version", ""))
        if version != TOKENIZER_VERSION:
            raise ValueError(
                f"Unsupported tokenizer version {version!r}; expected {TOKENIZER_VERSION!r}. "
                "Strict migration enabled: regenerate FOL datasets/metadata."
            )

        token_to_id_raw = payload.get("token_to_id")
        if not isinstance(token_to_id_raw, dict) or not token_to_id_raw:
            raise ValueError("Unified tokenizer metadata must include non-empty 'token_to_id'.")

        token_to_id = {str(token): int(tok_id) for token, tok_id in token_to_id_raw.items()}
        id_to_token = {tok_id: token for token, tok_id in token_to_id.items()}
        if len(id_to_token) != len(token_to_id):
            raise ValueError("token_to_id must be one-to-one.")

        sorted_ids = sorted(id_to_token)
        if sorted_ids != list(range(len(sorted_ids))):
            raise ValueError("Tokenizer token ids must be contiguous and start at 0.")

        if token_to_id.get(PAD_TOKEN) != pad_idx:
            raise ValueError(f"Expected {PAD_TOKEN} token id to be {pad_idx}.")

        for idx, token in enumerate(_LOGIC_TOKENS, start=1):
            got = token_to_id.get(token)
            if got != idx:
                raise ValueError(
                    f"Expected logic token {token!r} to have id {idx}, got {got}."
                )

        expected_sep = len(_LOGIC_TOKENS) + 1
        expected_eot = expected_sep + 1
        if token_to_id.get(SEP_TOKEN) != expected_sep:
            raise ValueError(
                f"Expected {SEP_TOKEN} token id {expected_sep}, got {token_to_id.get(SEP_TOKEN)}."
            )
        if token_to_id.get(EOT_TOKEN) != expected_eot:
            raise ValueError(
                f"Expected {EOT_TOKEN} token id {expected_eot}, got {token_to_id.get(EOT_TOKEN)}."
            )

        for tok_id in range(expected_eot + 1, len(sorted_ids)):
            token = id_to_token[tok_id]
            if token in _RESERVED_TOKENS or token in _LOGIC_TOKENS:
                raise ValueError(f"Unexpected reserved token in identifier range: {token!r}")
            if _IDENTIFIER_RE.fullmatch(token) is None:
                raise ValueError(f"Invalid identifier token in metadata: {token!r}")

        vocab_size = int(payload.get("vocab_size", len(sorted_ids)))
        if vocab_size != len(sorted_ids):
            raise ValueError(
                f"Inconsistent vocab_size {vocab_size}; expected {len(sorted_ids)}."
            )

        return cls(
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            sep_token_id=expected_sep,
            eot_token_id=expected_eot,
            version=version,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def to_dict(self) -> dict:
        token_to_id = {
            self.id_to_token[idx]: int(idx)
            for idx in range(self.vocab_size)
        }
        return {
            "version": self.version,
            "pad_idx": pad_idx,
            "logic_tokens": list(_LOGIC_TOKENS),
            "special_tokens": [PAD_TOKEN, SEP_TOKEN, EOT_TOKEN],
            "sep_token_id": int(self.sep_token_id),
            "eot_token_id": int(self.eot_token_id),
            "token_to_id": token_to_id,
            "vocab_size": int(self.vocab_size),
        }

    def char_to_id(self, token: str) -> int:
        if token in self.token_to_id:
            return self.token_to_id[token]
        raise ValueError(f"Unknown token symbol: {token}")

    def id_to_char(self, token_id: int) -> str:
        if token_id in self.id_to_token:
            return self.id_to_token[token_id]
        raise ValueError(f"Unknown token id: {token_id}")

    def _tokenize_text(self, text: str) -> list[int]:
        tokens: list[int] = []
        idx = 0
        while idx < len(text):
            ch = text[idx]
            if ch.isspace():
                idx += 1
                continue

            if ch in _LOGIC_TOKENS:
                tokens.append(self.char_to_id(ch))
                idx += 1
                continue

            match = _IDENTIFIER_RE.match(text, idx)
            if match is None:
                raise ValueError(f"Invalid symbol at index {idx} in {text!r}")
            ident = match.group(0)
            tokens.append(self.char_to_id(ident))
            idx = match.end()

        return tokens

    def tokenize_prompt(self, sequent: FOLSequent) -> list[int]:
        return self._tokenize_text(sequent.text) + [self.sep_token_id]

    def encode_completion(self, statement_text: str) -> list[int]:
        try:
            lhs, rhs = parse_clause_text(statement_text)
            canonical = f"{_format_conjunction(lhs)} → {_format_conjunction(rhs)}"
        except ValueError:
            atoms = parse_conjunction_text(statement_text)
            canonical = _format_conjunction(atoms)
        return self._tokenize_text(canonical) + [self.eot_token_id]

    def tokenize_example(self, sequent: FOLSequent, statement_text: str) -> tuple[list[int], list[int]]:
        return self.tokenize_prompt(sequent), self.encode_completion(statement_text)

    def decode_prompt(self, prompt_tokens: list[int]) -> FOLSequent:
        if not prompt_tokens:
            raise ValueError("Prompt cannot be empty.")
        if int(prompt_tokens[-1]) != int(self.sep_token_id):
            raise ValueError("Prompt must terminate with SEP token.")

        symbols = [self.id_to_char(int(tok)) for tok in prompt_tokens[:-1]]
        return parse_sequent_text("".join(symbols))

    def decode_completion_clause(self, completion_tokens: list[int]) -> tuple[tuple[FOLAtom, ...], tuple[FOLAtom, ...]]:
        if len(completion_tokens) < 2:
            raise ValueError("Completion must include clause tokens and EOT.")
        if int(completion_tokens[-1]) != int(self.eot_token_id):
            raise ValueError("Completion must terminate with EOT token.")

        symbols = [self.id_to_char(int(tok)) for tok in completion_tokens[:-1]]
        return parse_clause_text("".join(symbols))

    def decode_completion_text(self, completion_tokens: list[int]) -> str:
        if len(completion_tokens) < 2:
            raise ValueError("Completion must include clause tokens and EOT.")
        if int(completion_tokens[-1]) != int(self.eot_token_id):
            raise ValueError("Completion must terminate with EOT token.")

        symbols = [self.id_to_char(int(tok)) for tok in completion_tokens[:-1]]
        text = "".join(symbols)
        try:
            lhs, rhs = parse_clause_text(text)
            return f"{_format_conjunction(lhs)} → {_format_conjunction(rhs)}"
        except ValueError:
            atoms = parse_conjunction_text(text)
            return _format_conjunction(atoms)

    def decode_batch_ids(
        self,
        batch_ids,
        *,
        skip_pad: bool = True,
        include_special_tokens: bool = True,
    ) -> list[str]:
        rows = _normalize_batch_rows(batch_ids)
        decoded: list[str] = []
        for row in rows:
            parts: list[str] = []
            for token in row:
                sym = self.id_to_char(int(token))
                if skip_pad and sym == PAD_TOKEN:
                    continue
                if sym in _RESERVED_TOKENS:
                    if include_special_tokens:
                        parts.append(sym)
                    continue
                parts.append(sym)
            decoded.append("".join(parts))
        return decoded


def _normalize_batch_rows(batch_ids) -> list[list[int]]:
    if hasattr(batch_ids, "ndim"):
        ndim = int(batch_ids.ndim)
        if ndim == 1:
            return [list(int(tok) for tok in batch_ids.tolist())]
        if ndim == 2:
            return [list(int(tok) for tok in row.tolist()) for row in batch_ids]
        raise ValueError(f"Expected 1D or 2D array, got ndim={ndim}")

    if isinstance(batch_ids, list) and batch_ids and isinstance(batch_ids[0], list):
        return [list(int(tok) for tok in row) for row in batch_ids]
    return [list(int(tok) for tok in batch_ids)]


def build_tokenizer_from_identifiers(identifiers: Iterable[str]) -> FOLLayerTokenizer:
    return FOLLayerTokenizer.from_identifiers(identifiers)


def build_tokenizer_from_rule_bank(rule_bank: FOLRuleBank) -> FOLLayerTokenizer:
    return FOLLayerTokenizer.from_rule_bank(rule_bank)


def tokenizer_from_metadata(metadata: dict) -> FOLLayerTokenizer:
    payload = metadata.get("tokenizer")
    if payload is None:
        raise ValueError("Missing tokenizer metadata. Regenerate the dataset.")
    return FOLLayerTokenizer.from_dict(payload)
