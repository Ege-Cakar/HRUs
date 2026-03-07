"""Tokenization utilities for layered tasks."""

from __future__ import annotations

from dataclasses import dataclass
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from .decode_result import DecodeAttempt

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

if TYPE_CHECKING:
    from task.layer_gen.util.rule_bank import RuleBank

pad_idx = 0
TOKENIZER_VERSION = "layer_v2_prompt_start"

_LOGIC_TOKENS = ("⊢", "∧", "→", "(", ")", ",")
_atom_re = re.compile(r"p(\d+)_(\d+)$")


def _atom_sort_key(atom_name: str) -> tuple[int, int]:
    match = _atom_re.fullmatch(atom_name)
    if match is None:
        raise ValueError(f"Unsupported atom name: {atom_name}")
    layer = int(match.group(1))
    idx = int(match.group(2))
    if layer < 0 or idx < 0:
        raise ValueError(f"Atom indices must be non-negative: {atom_name}")
    return layer, idx


def _build_logic_maps() -> tuple[dict[str, int], dict[int, str]]:
    logic_to_id = {token: idx + 1 for idx, token in enumerate(_LOGIC_TOKENS)}
    id_to_logic = {idx: token for token, idx in logic_to_id.items()}
    return logic_to_id, id_to_logic


@dataclass(frozen=True)
class LayerTokenizer:
    logic_char_to_id: dict[str, int]
    id_to_logic_char: dict[int, str]
    atom_to_token: dict[str, int]
    token_to_atom: dict[int, str]
    sep_token_id: int
    start_token_id: int
    eot_token_id: int
    version: str = TOKENIZER_VERSION

    @classmethod
    def from_atoms(cls, atoms: Iterable[str]) -> "LayerTokenizer":
        logic_to_id, id_to_logic = _build_logic_maps()
        sep = len(logic_to_id) + 1
        start = sep + 1
        eot = start + 1

        unique_sorted = sorted(set(str(atom) for atom in atoms), key=_atom_sort_key)
        atom_to_token: dict[str, int] = {}
        token_to_atom: dict[int, str] = {}
        next_id = eot + 1
        for atom in unique_sorted:
            atom_to_token[atom] = next_id
            token_to_atom[next_id] = atom
            next_id += 1

        return cls(
            logic_char_to_id=logic_to_id,
            id_to_logic_char=id_to_logic,
            atom_to_token=atom_to_token,
            token_to_atom=token_to_atom,
            sep_token_id=sep,
            start_token_id=start,
            eot_token_id=eot,
        )

    @classmethod
    def from_rule_bank(cls, rule_bank: "RuleBank") -> "LayerTokenizer":
        atoms = [
            f"p{layer}_{idx}"
            for layer in range(int(rule_bank.n_layers))
            for idx in range(1, int(rule_bank.props_per_layer) + 1)
        ]
        return cls.from_atoms(atoms)

    @classmethod
    def from_dict(cls, payload: dict) -> "LayerTokenizer":
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
        start = int(payload.get("start_token_id", sep + 1))
        eot = int(payload.get("eot_token_id", start + 1))
        if sep != len(logic_to_id) + 1 or start != sep + 1 or eot != start + 1:
            raise ValueError(
                "Expected contiguous special ids with SEP, START, and EOT immediately after logic tokens."
            )

        atom_to_id_raw = payload.get("atom_to_id", {})
        atom_to_token = {str(atom): int(tok) for atom, tok in atom_to_id_raw.items()}

        next_expected = eot + 1
        for atom in sorted(atom_to_token, key=_atom_sort_key):
            got = atom_to_token[atom]
            if got != next_expected:
                raise ValueError(
                    f"Atom id mapping must be contiguous from {eot + 1}, got {got} for {atom}."
                )
            next_expected += 1

        token_to_atom = {tok: atom for atom, tok in atom_to_token.items()}
        vocab_size = int(payload.get("vocab_size", next_expected))
        if vocab_size != next_expected:
            raise ValueError(
                f"Inconsistent vocab_size {vocab_size}; expected {next_expected}."
            )

        return cls(
            logic_char_to_id=logic_to_id,
            id_to_logic_char=id_to_logic,
            atom_to_token=atom_to_token,
            token_to_atom=token_to_atom,
            sep_token_id=sep,
            start_token_id=start,
            eot_token_id=eot,
            version=version,
        )

    @property
    def vocab_size(self) -> int:
        return self.eot_token_id + 1 + len(self.atom_to_token)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "pad_idx": pad_idx,
            "logic_tokens": list(_LOGIC_TOKENS),
            "sep_token_id": int(self.sep_token_id),
            "start_token_id": int(self.start_token_id),
            "eot_token_id": int(self.eot_token_id),
            "atom_to_id": {atom: int(tok) for atom, tok in sorted(self.atom_to_token.items())},
            "vocab_size": int(self.vocab_size),
        }

    def char_to_id(self, token: str) -> int:
        if token == "<PAD>":
            return pad_idx
        if token == "<SEP>":
            return self.sep_token_id
        if token == "<START>":
            return self.start_token_id
        if token == "<EOT>":
            return self.eot_token_id
        if token in self.logic_char_to_id:
            return self.logic_char_to_id[token]
        if token in self.atom_to_token:
            return self.atom_to_token[token]
        raise ValueError(f"Unknown token symbol: {token}")

    def id_to_char(self, token_id: int) -> str:
        if token_id == pad_idx:
            return "<PAD>"
        if token_id == self.sep_token_id:
            return "<SEP>"
        if token_id == self.start_token_id:
            return "<START>"
        if token_id == self.eot_token_id:
            return "<EOT>"
        if token_id in self.id_to_logic_char:
            return self.id_to_logic_char[token_id]
        if token_id in self.token_to_atom:
            return self.token_to_atom[token_id]
        raise ValueError(f"Unknown token id: {token_id}")

    def _tokenize_prop_text(self, text: str) -> list[int]:
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
                tokens.append(self.char_to_id(text[i:k]))
                i = k
                continue

            tokens.append(self.char_to_id(ch))
            i += 1

        return tokens

    def tokenize_prop(self, prop: Proposition) -> list[int]:
        return self._tokenize_prop_text(str(prop))

    def tokenize_prompt(self, sequent: Sequent) -> list[int]:
        return self._tokenize_prop_text(str(sequent)) + [self.start_token_id]

    def encode_completion(self, statement_text: str) -> list[int]:
        return self._tokenize_prop_text(statement_text) + [self.eot_token_id]

    def tokenize_example(self, sequent: Sequent, statement_text: str) -> tuple[list[int], list[int]]:
        return self.tokenize_prompt(sequent), self.encode_completion(statement_text)

    def decode_prompt(self, prompt_tokens) -> Sequent:
        try:
            prompt = [int(tok) for tok in prompt_tokens]
        except TypeError as err:
            raise ValueError("Prompt tokens must be a 1D integer sequence.") from err
        if not prompt:
            raise ValueError("Prompt cannot be empty.")
        nonpad = [tok for tok in prompt if tok != pad_idx]
        start_positions = [
            idx for idx, tok in enumerate(nonpad)
            if tok == int(self.start_token_id)
        ]
        if len(start_positions) != 1:
            raise ValueError("Prompt must contain exactly one START token.")

        start_idx = int(start_positions[0])
        sequent_tokens = nonpad[:start_idx]
        sep_positions = [
            idx for idx, tok in enumerate(sequent_tokens)
            if tok == int(self.sep_token_id)
        ]
        body_start = int(sep_positions[-1]) + 1 if sep_positions else 0
        sequent_tokens = sequent_tokens[body_start:]
        if not sequent_tokens:
            raise ValueError("Prompt body cannot be empty.")
        symbols = [self.id_to_char(tok) for tok in sequent_tokens]
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

    def try_decode_prompt(self, prompt_tokens) -> DecodeAttempt[Sequent]:
        try:
            return DecodeAttempt.success(self.decode_prompt(prompt_tokens))
        except (TypeError, ValueError) as err:
            return DecodeAttempt.failure(str(err))

    def decode_completion_prop(self, completion_tokens: list[int]) -> Proposition:
        if len(completion_tokens) < 2:
            raise ValueError("Completion must include proposition tokens and EOT.")
        if completion_tokens[-1] != self.eot_token_id:
            raise ValueError("Completion must terminate with EOT token.")

        symbols = [self.id_to_char(tok) for tok in completion_tokens[:-1]]
        return _parse_layer_completion_symbols(symbols)

    def try_decode_completion_prop(
        self,
        completion_tokens: list[int],
    ) -> DecodeAttempt[Proposition]:
        try:
            return DecodeAttempt.success(self.decode_completion_prop(completion_tokens))
        except (TypeError, ValueError) as err:
            return DecodeAttempt.failure(str(err))

    def decode_completion_text(self, completion_tokens: list[int]) -> str:
        return _format_layer_completion_prop(self.decode_completion_prop(completion_tokens))

    def try_decode_completion_text(
        self,
        completion_tokens: list[int],
    ) -> DecodeAttempt[str]:
        decoded = self.try_decode_completion_prop(completion_tokens)
        if not decoded.ok or decoded.value is None:
            return DecodeAttempt.failure(decoded.error or "Unknown decode error.")
        return DecodeAttempt.success(_format_layer_completion_prop(decoded.value))

    def decode_batch_ids(
        self,
        batch_ids,
        *,
        skip_pad: bool = True,
        include_special_tokens: bool = True,
    ) -> list[str]:
        """Decode batched token ids from task outputs into readable strings."""
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
                if tok == self.start_token_id:
                    if include_special_tokens:
                        parts.append("<START>")
                    continue
                if tok == self.eot_token_id:
                    if include_special_tokens:
                        parts.append("<EOT>")
                    continue
                parts.append(self.id_to_char(tok))
            decoded.append("".join(parts))
        return decoded


def build_tokenizer_from_atoms(atoms: Iterable[str]) -> LayerTokenizer:
    return LayerTokenizer.from_atoms(atoms)


def build_tokenizer_from_rule_bank(rule_bank: "RuleBank") -> LayerTokenizer:
    return LayerTokenizer.from_rule_bank(rule_bank)


def tokenizer_from_metadata(metadata: dict) -> LayerTokenizer:
    payload = metadata.get("tokenizer")
    if payload is None:
        raise ValueError("Missing tokenizer metadata. Regenerate the dataset.")
    return LayerTokenizer.from_dict(payload)


_default_tokenizer = LayerTokenizer.from_atoms(())

logic_char_to_id = dict(_default_tokenizer.logic_char_to_id)
id_to_logic_char = dict(_default_tokenizer.id_to_logic_char)
sep_token_id = _default_tokenizer.sep_token_id
start_token_id = _default_tokenizer.start_token_id
eot_token_id = _default_tokenizer.eot_token_id


def char_to_id(token: str) -> int:
    return _default_tokenizer.char_to_id(token)


def id_to_char(token_id: int) -> str:
    return _default_tokenizer.id_to_char(token_id)


def tokenize_prop(prop: Proposition) -> list[int]:
    return _default_tokenizer.tokenize_prop(prop)


def tokenize_prompt(sequent: Sequent) -> list[int]:
    return _default_tokenizer.tokenize_prompt(sequent)


def encode_completion(statement_text: str) -> list[int]:
    return _default_tokenizer.encode_completion(statement_text)


def tokenize_example(sequent: Sequent, statement_text: str) -> tuple[list[int], list[int]]:
    return _default_tokenizer.tokenize_example(sequent, statement_text)


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


def _flatten_and(prop: Proposition) -> list[str]:
    if isinstance(prop, Atom):
        return [prop.name]
    if isinstance(prop, And):
        return _flatten_and(prop.left) + _flatten_and(prop.right)
    raise ValueError(f"Expected conjunction of atoms, got {type(prop).__name__}.")


def _format_layer_completion_prop(prop: Proposition) -> str:
    if not isinstance(prop, Implies):
        raise ValueError("Layer completion must be an implication.")
    lhs = " ∧ ".join(_flatten_and(prop.left))
    rhs = " ∧ ".join(_flatten_and(prop.right))
    return f"{lhs} → {rhs}"


def _parse_flat_conjunction(tokens: list[str]) -> Proposition:
    if not tokens:
        raise ValueError("Conjunction side cannot be empty.")
    if len(tokens) % 2 == 0:
        raise ValueError("Malformed conjunction; expected atoms separated by '∧'.")

    atoms: list[str] = []
    for idx, tok in enumerate(tokens):
        if idx % 2 == 0:
            if _atom_re.fullmatch(tok) is None:
                raise ValueError(f"Expected atom token, found {tok!r}.")
            atoms.append(tok)
            continue
        if tok != "∧":
            raise ValueError(f"Expected '∧' between atoms, found {tok!r}.")

    out: Proposition = Atom(atoms[0])
    for atom in atoms[1:]:
        out = And(out, Atom(atom))
    return out


def _parse_layer_completion_symbols(symbols: list[str]) -> Proposition:
    if "(" in symbols or ")" in symbols or "," in symbols:
        raise ValueError("Layer completion format does not allow parentheses or commas.")
    arrow_idxs = [idx for idx, tok in enumerate(symbols) if tok == "→"]
    if len(arrow_idxs) != 1:
        raise ValueError("Layer completion must contain exactly one implication arrow.")

    arrow_idx = arrow_idxs[0]
    lhs_tokens = symbols[:arrow_idx]
    rhs_tokens = symbols[arrow_idx + 1 :]
    lhs = _parse_flat_conjunction(lhs_tokens)
    rhs = _parse_flat_conjunction(rhs_tokens)
    return Implies(lhs, rhs)


def decode_prompt(prompt_tokens: list[int]) -> Sequent:
    return _default_tokenizer.decode_prompt(prompt_tokens)


def decode_completion_prop(completion_tokens: list[int]) -> Proposition:
    return _default_tokenizer.decode_completion_prop(completion_tokens)


def decode_completion_text(completion_tokens: list[int]) -> str:
    return _default_tokenizer.decode_completion_text(completion_tokens)


def decode_batch_ids(
    tokenizer: LayerTokenizer,
    batch_ids,
    *,
    skip_pad: bool = True,
    include_special_tokens: bool = True,
) -> list[str]:
    return tokenizer.decode_batch_ids(
        batch_ids,
        skip_pad=skip_pad,
        include_special_tokens=include_special_tokens,
    )


def _normalize_batch_rows(batch_ids) -> list[list[int]]:
    # Supports numpy arrays as well as nested Python lists/tuples.
    if hasattr(batch_ids, "ndim"):
        if batch_ids.ndim == 1:
            return [[int(tok) for tok in batch_ids]]
        if batch_ids.ndim == 2:
            return [[int(tok) for tok in row] for row in batch_ids]
        raise ValueError(f"Expected 1D or 2D batch ids, got ndim={batch_ids.ndim}")

    seq = list(batch_ids)
    if not seq:
        return []
    first = seq[0]
    if isinstance(first, (list, tuple)):
        return [[int(tok) for tok in row] for row in seq]
    return [[int(tok) for tok in seq]]
