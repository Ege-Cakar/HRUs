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
TOKENIZER_VERSION = "layer_fol_v1_compact"

_LOGIC_TOKENS = ("⊢", "∧", "→", "(", ")", ",")
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _format_conjunction(atoms: tuple[FOLAtom, ...]) -> str:
    return " ∧ ".join(atom.text for atom in atoms)


def _build_logic_maps() -> tuple[dict[str, int], dict[int, str]]:
    logic_to_id = {token: idx + 1 for idx, token in enumerate(_LOGIC_TOKENS)}
    id_to_logic = {idx: token for token, idx in logic_to_id.items()}
    return logic_to_id, id_to_logic


@dataclass(frozen=True)
class FOLLayerTokenizer:
    logic_char_to_id: dict[str, int]
    id_to_logic_char: dict[int, str]
    identifier_to_token: dict[str, int]
    token_to_identifier: dict[int, str]
    sep_token_id: int
    eot_token_id: int
    version: str = TOKENIZER_VERSION

    @classmethod
    def from_identifiers(cls, identifiers: Iterable[str]) -> "FOLLayerTokenizer":
        logic_to_id, id_to_logic = _build_logic_maps()
        sep = len(logic_to_id) + 1
        eot = sep + 1

        unique_sorted = sorted(set(str(tok) for tok in identifiers))
        for token in unique_sorted:
            if _IDENTIFIER_RE.fullmatch(token) is None:
                raise ValueError(f"Invalid identifier token: {token!r}")

        identifier_to_token: dict[str, int] = {}
        token_to_identifier: dict[int, str] = {}
        next_id = eot + 1
        for token in unique_sorted:
            identifier_to_token[token] = next_id
            token_to_identifier[next_id] = token
            next_id += 1

        return cls(
            logic_char_to_id=logic_to_id,
            id_to_logic_char=id_to_logic,
            identifier_to_token=identifier_to_token,
            token_to_identifier=token_to_identifier,
            sep_token_id=sep,
            eot_token_id=eot,
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
                f"Unsupported tokenizer version {version!r}; expected {TOKENIZER_VERSION!r}."
            )

        logic_tokens = tuple(payload.get("logic_tokens", _LOGIC_TOKENS))
        if logic_tokens != _LOGIC_TOKENS:
            raise ValueError(f"Unsupported logic token set: {logic_tokens!r}")

        logic_to_id, id_to_logic = _build_logic_maps()
        sep = int(payload.get("sep_token_id", len(logic_to_id) + 1))
        eot = int(payload.get("eot_token_id", sep + 1))
        if sep != len(logic_to_id) + 1 or eot != sep + 1:
            raise ValueError(
                "Expected contiguous special ids with SEP immediately after logic tokens."
            )

        identifier_to_id_raw = payload.get("identifier_to_id", {})
        identifier_to_token = {
            str(identifier): int(tok)
            for identifier, tok in identifier_to_id_raw.items()
        }

        next_expected = eot + 1
        for identifier in sorted(identifier_to_token):
            got = identifier_to_token[identifier]
            if got != next_expected:
                raise ValueError(
                    f"Identifier id mapping must be contiguous from {eot + 1}, got {got} "
                    f"for {identifier}."
                )
            next_expected += 1

        token_to_identifier = {tok: identifier for identifier, tok in identifier_to_token.items()}
        vocab_size = int(payload.get("vocab_size", next_expected))
        if vocab_size != next_expected:
            raise ValueError(
                f"Inconsistent vocab_size {vocab_size}; expected {next_expected}."
            )

        return cls(
            logic_char_to_id=logic_to_id,
            id_to_logic_char=id_to_logic,
            identifier_to_token=identifier_to_token,
            token_to_identifier=token_to_identifier,
            sep_token_id=sep,
            eot_token_id=eot,
            version=version,
        )

    @property
    def vocab_size(self) -> int:
        return self.eot_token_id + 1 + len(self.identifier_to_token)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "pad_idx": pad_idx,
            "logic_tokens": list(_LOGIC_TOKENS),
            "sep_token_id": int(self.sep_token_id),
            "eot_token_id": int(self.eot_token_id),
            "identifier_to_id": {
                token: int(tok)
                for token, tok in sorted(self.identifier_to_token.items())
            },
            "vocab_size": int(self.vocab_size),
        }

    def char_to_id(self, token: str) -> int:
        if token in self.logic_char_to_id:
            return self.logic_char_to_id[token]
        if token in self.identifier_to_token:
            return self.identifier_to_token[token]
        raise ValueError(f"Unknown token symbol: {token}")

    def id_to_char(self, token_id: int) -> str:
        if token_id in self.id_to_logic_char:
            return self.id_to_logic_char[token_id]
        if token_id in self.token_to_identifier:
            return self.token_to_identifier[token_id]
        raise ValueError(f"Unknown token id: {token_id}")

    def _tokenize_text(self, text: str) -> list[int]:
        tokens: list[int] = []
        idx = 0
        while idx < len(text):
            ch = text[idx]
            if ch.isspace():
                idx += 1
                continue

            if ch in self.logic_char_to_id:
                tokens.append(self.logic_char_to_id[ch])
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
                tok = int(token)
                if skip_pad and tok == pad_idx:
                    continue
                if tok == self.sep_token_id:
                    if include_special_tokens:
                        parts.append("<SEP>")
                    continue
                if tok == self.eot_token_id:
                    if include_special_tokens:
                        parts.append("<EOT>")
                    continue
                parts.append(self.id_to_char(tok))
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
