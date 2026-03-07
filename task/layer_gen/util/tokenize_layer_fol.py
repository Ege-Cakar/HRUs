"""Tokenization utilities for layered first-order tasks."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from .decode_result import DecodeAttempt
from .fol_rule_bank import (
    FOLAtom,
    FOLRuleBank,
    FOLSequent,
    parse_clause_text,
    parse_conjunction_text,
    parse_sequent_text,
)

pad_idx = 0
TOKENIZER_VERSION = "layer_fol_v4_prompt_start"

PAD_TOKEN = "<PAD>"
SEP_TOKEN = "<SEP>"
START_TOKEN = "<START>"
EOT_TOKEN = "<EOT>"

_LOGIC_TOKENS = ("⊢", "∧", "→", "(", ")", ",")
_RESERVED_TOKENS = (PAD_TOKEN, SEP_TOKEN, START_TOKEN, EOT_TOKEN)
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PREDICATE_RE = re.compile(r"r\d+_\d+$|r_[a-z0-9]+$")
_PREDICATE_CHAR_RE = re.compile(r"[A-Za-z0-9_]")


def _format_conjunction(atoms: tuple[FOLAtom, ...]) -> str:
    return " ∧ ".join(atom.text for atom in atoms)


def _build_vocab(
    identifiers: Iterable[str],
    *,
    predicate_identifiers: Iterable[str] | None = None,
) -> tuple[dict[str, int], dict[int, str], int, int, int]:
    all_identifiers = {str(tok) for tok in identifiers}
    if predicate_identifiers is None:
        predicate_tokens = {
            token for token in all_identifiers if _PREDICATE_RE.fullmatch(token) is not None
        }
    else:
        predicate_tokens = {str(tok) for tok in predicate_identifiers}
        all_identifiers.update(predicate_tokens)

    unique_sorted = sorted(all_identifiers)
    for token in unique_sorted:
        if token in _RESERVED_TOKENS:
            raise ValueError(f"Identifier token is reserved for special use: {token!r}")
        if _IDENTIFIER_RE.fullmatch(token) is None:
            raise ValueError(f"Invalid identifier token: {token!r}")
    for predicate in sorted(predicate_tokens):
        if _IDENTIFIER_RE.fullmatch(predicate) is None:
            raise ValueError(f"Invalid predicate token: {predicate!r}")

    token_to_id: dict[str, int] = {PAD_TOKEN: pad_idx}
    next_id = pad_idx + 1

    for token in _LOGIC_TOKENS:
        token_to_id[token] = next_id
        next_id += 1

    sep_token_id = next_id
    token_to_id[SEP_TOKEN] = sep_token_id
    next_id += 1

    start_token_id = next_id
    token_to_id[START_TOKEN] = start_token_id
    next_id += 1

    eot_token_id = next_id
    token_to_id[EOT_TOKEN] = eot_token_id
    next_id += 1

    # Non-predicate identifiers are represented as single tokens.
    for token in (tok for tok in unique_sorted if tok not in predicate_tokens):
        if token in token_to_id:
            continue
        token_to_id[token] = next_id
        next_id += 1

    # Layered predicates are represented by character tokens.
    predicate_chars = sorted({ch for token in predicate_tokens for ch in token})
    for ch in predicate_chars:
        if _PREDICATE_CHAR_RE.fullmatch(ch) is None:
            raise ValueError(f"Unsupported predicate character token: {ch!r}")
        if ch in token_to_id:
            continue
        if ch in _RESERVED_TOKENS or ch in _LOGIC_TOKENS:
            raise ValueError(f"Predicate character collides with reserved/logic token: {ch!r}")
        token_to_id[ch] = next_id
        next_id += 1

    id_to_token = {idx: token for token, idx in token_to_id.items()}
    return token_to_id, id_to_token, sep_token_id, start_token_id, eot_token_id


def _predicate_is_char_tokenized(predicate: str) -> bool:
    return _PREDICATE_RE.fullmatch(str(predicate)) is not None


def _is_identifier_symbol(symbol: str) -> bool:
    return _IDENTIFIER_RE.fullmatch(symbol) is not None


@dataclass(frozen=True)
class FOLLayerTokenizer:
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]
    sep_token_id: int
    start_token_id: int
    eot_token_id: int
    version: str = TOKENIZER_VERSION

    @classmethod
    def from_identifiers(
        cls,
        identifiers: Iterable[str],
        *,
        predicate_identifiers: Iterable[str] | None = None,
    ) -> "FOLLayerTokenizer":
        token_to_id, id_to_token, sep_token_id, start_token_id, eot_token_id = _build_vocab(
            identifiers,
            predicate_identifiers=predicate_identifiers,
        )
        return cls(
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            sep_token_id=sep_token_id,
            start_token_id=start_token_id,
            eot_token_id=eot_token_id,
        )

    @classmethod
    def from_rule_bank(cls, rule_bank: FOLRuleBank) -> "FOLLayerTokenizer":
        identifiers: set[str] = set(rule_bank.constants)
        identifiers.update(rule_bank.predicate_arities)
        identifiers.update(f"x{idx}" for idx in range(1, int(rule_bank.vars_per_rule_max) + 1))
        return cls.from_identifiers(
            sorted(identifiers),
            predicate_identifiers=sorted(rule_bank.predicate_arities),
        )

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
        expected_start = expected_sep + 1
        expected_eot = expected_start + 1
        if token_to_id.get(SEP_TOKEN) != expected_sep:
            raise ValueError(
                f"Expected {SEP_TOKEN} token id {expected_sep}, got {token_to_id.get(SEP_TOKEN)}."
            )
        if token_to_id.get(START_TOKEN) != expected_start:
            raise ValueError(
                f"Expected {START_TOKEN} token id {expected_start}, got {token_to_id.get(START_TOKEN)}."
            )
        if token_to_id.get(EOT_TOKEN) != expected_eot:
            raise ValueError(
                f"Expected {EOT_TOKEN} token id {expected_eot}, got {token_to_id.get(EOT_TOKEN)}."
            )

        for tok_id in range(expected_eot + 1, len(sorted_ids)):
            token = id_to_token[tok_id]
            if token in _RESERVED_TOKENS or token in _LOGIC_TOKENS:
                raise ValueError(f"Unexpected reserved token in identifier range: {token!r}")
            if _IDENTIFIER_RE.fullmatch(token) is None and not (
                len(token) == 1 and _PREDICATE_CHAR_RE.fullmatch(token) is not None
            ):
                raise ValueError(f"Invalid identifier/predicate-char token in metadata: {token!r}")

        vocab_size = int(payload.get("vocab_size", len(sorted_ids)))
        if vocab_size != len(sorted_ids):
            raise ValueError(
                f"Inconsistent vocab_size {vocab_size}; expected {len(sorted_ids)}."
            )

        return cls(
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            sep_token_id=expected_sep,
            start_token_id=expected_start,
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
            "special_tokens": [PAD_TOKEN, SEP_TOKEN, START_TOKEN, EOT_TOKEN],
            "sep_token_id": int(self.sep_token_id),
            "start_token_id": int(self.start_token_id),
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

    def _encode_predicate(self, predicate: str) -> list[int]:
        if _predicate_is_char_tokenized(predicate):
            return [self.char_to_id(ch) for ch in predicate]
        return [self.char_to_id(predicate)]

    def _encode_atom(self, atom: FOLAtom) -> list[int]:
        tokens: list[int] = []
        tokens.extend(self._encode_predicate(atom.predicate))
        if atom.args:
            tokens.append(self.char_to_id("("))
            for arg in atom.args:
                tokens.append(self.char_to_id(arg))
            tokens.append(self.char_to_id(")"))
        return tokens

    def tokenize_prompt(self, sequent: FOLSequent) -> list[int]:
        tokens: list[int] = []
        for idx, atom in enumerate(sequent.ants):
            if idx > 0:
                tokens.append(self.char_to_id(","))
            tokens.extend(self._encode_atom(atom))
        tokens.append(self.char_to_id("⊢"))
        tokens.extend(self._encode_atom(sequent.cons))
        tokens.append(self.start_token_id)
        return tokens

    def _encode_conjunction(self, atoms: tuple[FOLAtom, ...]) -> list[int]:
        tokens: list[int] = []
        for idx, atom in enumerate(atoms):
            if idx > 0:
                tokens.append(self.char_to_id("∧"))
            tokens.extend(self._encode_atom(atom))
        return tokens

    def _encode_completion_statement(self, statement_text: str) -> list[int]:
        try:
            lhs, rhs = parse_clause_text(statement_text)
            tokens = self._encode_conjunction(lhs)
            tokens.append(self.char_to_id("→"))
            tokens.extend(self._encode_conjunction(rhs))
        except ValueError:
            atoms = parse_conjunction_text(statement_text)
            tokens = self._encode_conjunction(atoms)
        return tokens

    def encode_completion_texts(self, statement_texts: Iterable[str]) -> list[int]:
        statement_texts = [str(text) for text in statement_texts]
        if not statement_texts:
            raise ValueError("Completion must contain at least one statement.")

        tokens: list[int] = []
        for idx, statement_text in enumerate(statement_texts):
            if idx > 0:
                tokens.append(self.sep_token_id)
            tokens.extend(self._encode_completion_statement(statement_text))
        tokens.append(self.eot_token_id)
        return tokens

    def tokenize_example(self, sequent: FOLSequent, statement_text: str) -> tuple[list[int], list[int]]:
        return self.tokenize_prompt(sequent), self.encode_completion_texts([statement_text])

    def decode_prompt(self, prompt_tokens) -> FOLSequent:
        try:
            prompt = [int(tok) for tok in prompt_tokens]
        except TypeError as err:
            raise ValueError("Prompt tokens must be a 1D integer sequence.") from err
        if not prompt:
            raise ValueError("Prompt cannot be empty.")
        nonpad = [tok for tok in prompt if tok != pad_idx]
        start_positions = [
            idx for idx, tok in enumerate(nonpad)
            if int(tok) == int(self.start_token_id)
        ]
        if not start_positions:
            raise ValueError("Prompt must contain exactly one START token.")
        if len(start_positions) != 1:
            raise ValueError("Prompt must contain exactly one START token.")

        start_idx = int(start_positions[0])
        body = nonpad[:start_idx]
        sep_positions = [
            idx for idx, tok in enumerate(body)
            if int(tok) == int(self.sep_token_id)
        ]
        body_start = int(sep_positions[-1]) + 1 if sep_positions else 0
        body_tokens = body[body_start:]
        if not body_tokens:
            raise ValueError("Prompt body cannot be empty.")

        symbols = [self.id_to_char(int(tok)) for tok in body_tokens]
        if any(sym in _RESERVED_TOKENS for sym in symbols):
            raise ValueError("Prompt body contains reserved special tokens.")
        sequent = _decode_prompt_symbols_to_sequent(symbols)
        return parse_sequent_text(sequent.text)

    def try_decode_prompt(self, prompt_tokens) -> DecodeAttempt[FOLSequent]:
        try:
            return DecodeAttempt.success(self.decode_prompt(prompt_tokens))
        except (TypeError, ValueError) as err:
            return DecodeAttempt.failure(str(err))

    def _decode_completion_statement_text(self, statement_tokens: list[int]) -> str:
        if not statement_tokens:
            raise ValueError("Completion statement cannot be empty.")
        symbols = [self.id_to_char(int(tok)) for tok in statement_tokens]
        if any(sym in _RESERVED_TOKENS for sym in symbols):
            raise ValueError("Completion body contains reserved special tokens.")
        try:
            lhs, rhs = _decode_completion_symbols_to_clause(symbols)
            canonical = f"{_format_conjunction(lhs)} → {_format_conjunction(rhs)}"
            lhs, rhs = parse_clause_text(canonical)
            return f"{_format_conjunction(lhs)} → {_format_conjunction(rhs)}"
        except ValueError:
            atoms = _decode_completion_symbols_to_conjunction(symbols)
            atoms = parse_conjunction_text(_format_conjunction(atoms))
            return _format_conjunction(atoms)

    def decode_completion_clause(self, completion_tokens: list[int]) -> tuple[tuple[FOLAtom, ...], tuple[FOLAtom, ...]]:
        statements = self.decode_completion_texts(completion_tokens)
        if len(statements) != 1:
            raise ValueError("Clause decoding requires a single completion statement.")
        return parse_clause_text(statements[0])

    def try_decode_completion_clause(
        self,
        completion_tokens: list[int],
    ) -> DecodeAttempt[tuple[tuple[FOLAtom, ...], tuple[FOLAtom, ...]]]:
        try:
            return DecodeAttempt.success(self.decode_completion_clause(completion_tokens))
        except (TypeError, ValueError) as err:
            return DecodeAttempt.failure(str(err))

    def decode_completion_texts(self, completion_tokens: list[int]) -> list[str]:
        if len(completion_tokens) < 2:
            raise ValueError("Completion must include content and EOT.")
        if int(completion_tokens[-1]) != int(self.eot_token_id):
            raise ValueError("Completion must terminate with EOT token.")

        body = [int(tok) for tok in completion_tokens[:-1]]
        if not body:
            raise ValueError("Completion must contain at least one statement.")

        out: list[str] = []
        current: list[int] = []
        for tok in body:
            if tok == int(self.sep_token_id):
                if not current:
                    raise ValueError("Completion contains an empty segment.")
                out.append(self._decode_completion_statement_text(current))
                current = []
                continue
            current.append(int(tok))

        if not current:
            raise ValueError("Completion cannot end with SEP before EOT.")
        out.append(self._decode_completion_statement_text(current))
        return out

    def try_decode_completion_texts(
        self,
        completion_tokens: list[int],
    ) -> DecodeAttempt[list[str]]:
        try:
            return DecodeAttempt.success(self.decode_completion_texts(completion_tokens))
        except (TypeError, ValueError) as err:
            return DecodeAttempt.failure(str(err))

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


def _parse_atom_from_symbols(
    symbols: list[str],
    idx: int,
) -> tuple[FOLAtom, int]:
    start = int(idx)
    while idx < len(symbols) and symbols[idx] != "(":
        sym = symbols[idx]
        if sym in {"⊢", "∧", "→", ")", ","}:
            break
        idx += 1

    if idx <= start:
        raise ValueError("Malformed atom: missing predicate in atom symbol stream.")
    predicate = "".join(symbols[start:idx])
    if _IDENTIFIER_RE.fullmatch(predicate) is None:
        raise ValueError(f"Invalid predicate {predicate!r} in atom symbol stream.")

    # Arity-0: no opening parenthesis follows the predicate.
    if idx >= len(symbols) or symbols[idx] != "(":
        return FOLAtom(predicate=predicate, args=()), idx

    idx += 1

    args: list[str] = []
    while idx < len(symbols):
        sym = symbols[idx]
        if sym == ")":
            idx += 1
            return FOLAtom(predicate=predicate, args=tuple(args)), idx
        if sym == ",":
            idx += 1
            continue
        if not _is_identifier_symbol(sym):
            raise ValueError(f"Invalid atom argument token {sym!r}.")
        args.append(sym)
        idx += 1

    raise ValueError("Malformed atom: missing closing parenthesis.")


def _parse_conjunction_from_symbols(
    symbols: list[str],
    idx: int,
) -> tuple[tuple[FOLAtom, ...], int]:
    atoms: list[FOLAtom] = []
    atom, idx = _parse_atom_from_symbols(symbols, idx)
    atoms.append(atom)
    while idx < len(symbols) and symbols[idx] == "∧":
        idx += 1
        atom, idx = _parse_atom_from_symbols(symbols, idx)
        atoms.append(atom)
    return tuple(atoms), idx


def _decode_prompt_symbols_to_sequent(symbols: list[str]) -> FOLSequent:
    turnstile_idxs = [idx for idx, sym in enumerate(symbols) if sym == "⊢"]
    if len(turnstile_idxs) != 1:
        raise ValueError("Prompt must contain exactly one top-level turnstile token.")
    turnstile_idx = int(turnstile_idxs[0])
    ant_symbols = symbols[:turnstile_idx]
    cons_symbols = symbols[turnstile_idx + 1 :]
    if not cons_symbols:
        raise ValueError("Prompt must contain consequent atom symbols.")

    ants: list[FOLAtom] = []
    idx = 0
    while idx < len(ant_symbols):
        atom, idx = _parse_atom_from_symbols(ant_symbols, idx)
        ants.append(atom)
        if idx >= len(ant_symbols):
            break
        if ant_symbols[idx] != ",":
            raise ValueError("Antecedent atoms must be separated by top-level commas.")
        idx += 1

    cons, cons_end = _parse_atom_from_symbols(cons_symbols, 0)
    if cons_end != len(cons_symbols):
        raise ValueError("Consequent atom must consume all prompt consequent tokens.")
    return FOLSequent(ants=tuple(ants), cons=cons)


def _decode_completion_symbols_to_clause(
    symbols: list[str],
) -> tuple[tuple[FOLAtom, ...], tuple[FOLAtom, ...]]:
    lhs, idx = _parse_conjunction_from_symbols(symbols, 0)
    if idx >= len(symbols) or symbols[idx] != "→":
        raise ValueError("Completion clause must include top-level implication arrow.")
    rhs, end = _parse_conjunction_from_symbols(symbols, idx + 1)
    if end != len(symbols):
        raise ValueError("Trailing symbols after completion clause.")
    return lhs, rhs


def _decode_completion_symbols_to_conjunction(
    symbols: list[str],
) -> tuple[FOLAtom, ...]:
    atoms, end = _parse_conjunction_from_symbols(symbols, 0)
    if end != len(symbols):
        raise ValueError("Trailing symbols after conjunction.")
    return atoms


def build_tokenizer_from_identifiers(
    identifiers: Iterable[str],
    *,
    predicate_identifiers: Iterable[str] | None = None,
) -> FOLLayerTokenizer:
    return FOLLayerTokenizer.from_identifiers(
        identifiers,
        predicate_identifiers=predicate_identifiers,
    )


def build_tokenizer_from_rule_bank(rule_bank: FOLRuleBank) -> FOLLayerTokenizer:
    return FOLLayerTokenizer.from_rule_bank(rule_bank)


def tokenizer_from_metadata(metadata: dict) -> FOLLayerTokenizer:
    payload = metadata.get("tokenizer")
    if payload is None:
        raise ValueError("Missing tokenizer metadata. Regenerate the dataset.")
    return FOLLayerTokenizer.from_dict(payload)
