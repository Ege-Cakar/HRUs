"""Core data types, parsing, validation, and I/O for FOL rule banks."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Iterable, Sequence

import numpy as np

_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_LAYERED_PREDICATE_RE = re.compile(r"r(\d+)_(\d+)$")


def _count_spec_error(name: str, message: str) -> ValueError:
    return ValueError(f"{name} {message}")


def _normalize_count_spec(
    spec: int | Sequence[int],
    *,
    expected_len: int,
    name: str,
) -> tuple[int, ...]:
    expected_len = int(expected_len)
    if expected_len < 1:
        raise _count_spec_error(name, f"expected_len must be >= 1, got {expected_len}")

    if isinstance(spec, (int, np.integer)) and not isinstance(spec, bool):
        values = (int(spec),) * expected_len
    else:
        if isinstance(spec, (str, bytes)):
            raise _count_spec_error(name, f"must be an int or sequence of ints, got {spec!r}")
        try:
            values = tuple(int(value) for value in spec)
        except TypeError as err:
            raise _count_spec_error(
                name,
                f"must be an int or sequence of ints, got {type(spec).__name__!r}",
            ) from err
        if len(values) != expected_len:
            raise _count_spec_error(
                name,
                f"must have length {expected_len}, got {len(values)}",
            )

    for value in values:
        if int(value) < 1:
            raise _count_spec_error(name, f"must contain only values >= 1, got {values}")
    return tuple(int(value) for value in values)


def _scalarize_count_spec(values: Sequence[int]) -> int | tuple[int, ...]:
    resolved = tuple(int(value) for value in values)
    if not resolved:
        raise ValueError("count specs must not be empty")
    if all(value == resolved[0] for value in resolved):
        return int(resolved[0])
    return resolved


def _count_spec_to_payload(values: Sequence[int]) -> int | list[int]:
    scalarized = _scalarize_count_spec(values)
    if isinstance(scalarized, tuple):
        return [int(value) for value in scalarized]
    return int(scalarized)


def _is_identifier(token: str) -> bool:
    return _IDENTIFIER_RE.fullmatch(token) is not None


def _is_variable(token: str) -> bool:
    return token.startswith("x") and _is_identifier(token)


def _split_top_level(text: str, delimiter: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    idx = 0
    while idx < len(text):
        ch = text[idx]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                raise ValueError(f"Unbalanced parentheses in {text!r}")
        elif ch == delimiter and depth == 0:
            parts.append(text[start:idx].strip())
            start = idx + 1
        idx += 1

    if depth != 0:
        raise ValueError(f"Unbalanced parentheses in {text!r}")
    parts.append(text[start:].strip())
    return parts


def _split_top_level_arrow(text: str) -> tuple[str, str]:
    depth = 0
    arrow_pos = -1
    idx = 0
    while idx < len(text):
        ch = text[idx]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                raise ValueError(f"Unbalanced parentheses in {text!r}")
        elif ch == "→" and depth == 0:
            if arrow_pos >= 0:
                raise ValueError(f"Expected a single top-level arrow in {text!r}")
            arrow_pos = idx
        idx += 1

    if depth != 0:
        raise ValueError(f"Unbalanced parentheses in {text!r}")
    if arrow_pos < 0:
        raise ValueError(f"Missing top-level arrow in {text!r}")

    lhs = text[:arrow_pos].strip()
    rhs = text[arrow_pos + 1 :].strip()
    if not lhs or not rhs:
        raise ValueError(f"Malformed clause around arrow in {text!r}")
    return lhs, rhs


def _parse_ident(text: str, idx: int) -> tuple[str, int]:
    match = _IDENTIFIER_RE.match(text, idx)
    if match is None:
        raise ValueError(f"Expected identifier at index {idx} in {text!r}")
    return match.group(0), match.end()


def _skip_ws(text: str, idx: int) -> int:
    while idx < len(text) and text[idx].isspace():
        idx += 1
    return idx


@dataclass(frozen=True)
class FOLAtom:
    predicate: str
    args: tuple[str, ...]

    def __post_init__(self) -> None:
        if not _is_identifier(self.predicate):
            raise ValueError(f"Invalid predicate name: {self.predicate!r}")
        for arg in self.args:
            if not _is_identifier(arg):
                raise ValueError(f"Invalid atom argument: {arg!r}")

    @property
    def arity(self) -> int:
        return len(self.args)

    @property
    def text(self) -> str:
        if not self.args:
            return self.predicate
        return f"{self.predicate}({','.join(self.args)})"

    def to_dict(self) -> dict:
        return {
            "predicate": self.predicate,
            "args": list(self.args),
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "FOLAtom":
        return cls(
            predicate=str(payload["predicate"]),
            args=tuple(str(arg) for arg in payload["args"]),
        )


@dataclass(frozen=True)
class FOLSequent:
    ants: tuple[FOLAtom, ...]
    cons: FOLAtom

    @property
    def text(self) -> str:
        ants_text = ",".join(atom.text for atom in self.ants)
        return f"{ants_text}⊢{self.cons.text}"


@dataclass(frozen=True)
class FOLLayerRule:
    src_layer: int
    dst_layer: int
    lhs: tuple[FOLAtom, ...]
    rhs: tuple[FOLAtom, ...]

    def __post_init__(self) -> None:
        if int(self.dst_layer) != int(self.src_layer) + 1:
            raise ValueError(
                f"Layer rule must connect adjacent layers, got {self.src_layer}->{self.dst_layer}"
            )
        if not self.lhs or not self.rhs:
            raise ValueError("FOLLayerRule requires non-empty lhs and rhs.")

    @property
    def statement_text(self) -> str:
        return f"{_format_conjunction(self.lhs)} → {_format_conjunction(self.rhs)}"

    def instantiate(self, subst: dict[str, str]) -> "FOLLayerRule":
        def _inst(atom: FOLAtom) -> FOLAtom:
            args = tuple(subst.get(term, term) for term in atom.args)
            return FOLAtom(atom.predicate, args)

        return FOLLayerRule(
            src_layer=int(self.src_layer),
            dst_layer=int(self.dst_layer),
            lhs=_sorted_atoms(_inst(atom) for atom in self.lhs),
            rhs=_sorted_atoms(_inst(atom) for atom in self.rhs),
        )

    def variables(self) -> set[str]:
        out: set[str] = set()
        for atom in self.lhs + self.rhs:
            for arg in atom.args:
                if _is_variable(arg):
                    out.add(arg)
        return out

    def to_dict(self) -> dict:
        return {
            "src_layer": int(self.src_layer),
            "dst_layer": int(self.dst_layer),
            "lhs": [atom.to_dict() for atom in self.lhs],
            "rhs": [atom.to_dict() for atom in self.rhs],
            "statement": self.statement_text,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "FOLLayerRule":
        return cls(
            src_layer=int(payload["src_layer"]),
            dst_layer=int(payload["dst_layer"]),
            lhs=_sorted_atoms(FOLAtom.from_dict(atom) for atom in payload["lhs"]),
            rhs=_sorted_atoms(FOLAtom.from_dict(atom) for atom in payload["rhs"]),
        )


@dataclass(frozen=True)
class FOLRuleBank:
    n_layers: int
    predicates_per_layer: int | tuple[int, ...]
    arity_max: int
    constants: tuple[str, ...]
    vars_per_rule_max: int
    predicate_arities: dict[str, int]
    transitions: dict[int, tuple[FOLLayerRule, ...]]
    layer_predicates: dict[int, tuple[str, ...]] | None = None
    arity_min: int = 1

    def transition_rules(self, src_layer: int) -> tuple[FOLLayerRule, ...]:
        return self.transitions.get(src_layer, ())

    def statement_set(self, src_layer: int) -> set[str]:
        return {rule.statement_text for rule in self.transition_rules(src_layer)}

    def predicates_per_layer_counts(self) -> tuple[int, ...]:
        return _normalize_count_spec(
            self.predicates_per_layer,
            expected_len=int(self.n_layers),
            name="predicates_per_layer",
        )

    def predicates_for_layer(self, layer: int) -> tuple[str, ...]:
        if self.layer_predicates is not None and int(layer) in self.layer_predicates:
            return tuple(self.layer_predicates[int(layer)])
        count = self.predicates_per_layer_counts()[int(layer)]
        return tuple(f"r{int(layer)}_{idx}" for idx in range(1, int(count) + 1))

    def to_dict(self) -> dict:
        transitions = {
            str(layer): [rule.to_dict() for rule in rules]
            for layer, rules in self.transitions.items()
        }
        return {
            "n_layers": int(self.n_layers),
            "predicates_per_layer": _count_spec_to_payload(self.predicates_per_layer_counts()),
            "arity_max": int(self.arity_max),
            "arity_min": int(self.arity_min),
            "constants": list(self.constants),
            "vars_per_rule_max": int(self.vars_per_rule_max),
            "predicate_arities": {
                str(pred): int(arity)
                for pred, arity in sorted(self.predicate_arities.items())
            },
            "transitions": transitions,
            **(
                {
                    "layer_predicates": {
                        str(layer): [str(pred) for pred in preds]
                        for layer, preds in sorted(self.layer_predicates.items())
                    }
                }
                if self.layer_predicates is not None
                else {}
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "FOLRuleBank":
        transitions: dict[int, tuple[FOLLayerRule, ...]] = {}
        for src_key, rules in payload["transitions"].items():
            src = int(src_key)
            transitions[src] = tuple(
                sorted(
                    (FOLLayerRule.from_dict(rule) for rule in rules),
                    key=_rule_sort_key,
                )
            )

        layer_predicates_payload = payload.get("layer_predicates")
        layer_predicates = None
        if layer_predicates_payload is not None:
            layer_predicates = {
                int(layer): tuple(str(pred) for pred in preds)
                for layer, preds in layer_predicates_payload.items()
            }

        return cls(
            n_layers=int(payload["n_layers"]),
            predicates_per_layer=_scalarize_count_spec(
                _normalize_count_spec(
                    payload["predicates_per_layer"],
                    expected_len=int(payload["n_layers"]),
                    name="predicates_per_layer",
                )
            ),
            arity_max=int(payload["arity_max"]),
            arity_min=int(payload.get("arity_min", 1)),
            constants=tuple(str(tok) for tok in payload["constants"]),
            vars_per_rule_max=int(payload["vars_per_rule_max"]),
            predicate_arities={
                str(pred): int(arity)
                for pred, arity in payload["predicate_arities"].items()
            },
            transitions=transitions,
            layer_predicates=layer_predicates,
        )


@dataclass(frozen=True)
class FOLDepth3ICLSplitBundle:
    train_bank: FOLRuleBank
    eval_bank: FOLRuleBank
    train_layer0_indices: tuple[int, ...]
    eval_layer0_indices: tuple[int, ...]
    train_layer0_predicates: tuple[str, ...]
    eval_layer0_predicates: tuple[str, ...]
    shared_layer1_predicates: tuple[str, ...]
    shared_layer2_predicates: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_depth3_icl_split_bundle(self)

    def to_dict(self) -> dict:
        return {
            "version": "fol_depth3_icl_split_v2",
            "train_bank": self.train_bank.to_dict(),
            "eval_bank": self.eval_bank.to_dict(),
            "train_layer0_indices": [int(idx) for idx in self.train_layer0_indices],
            "eval_layer0_indices": [int(idx) for idx in self.eval_layer0_indices],
            "train_layer0_predicates": [str(pred) for pred in self.train_layer0_predicates],
            "eval_layer0_predicates": [str(pred) for pred in self.eval_layer0_predicates],
            "shared_layer1_predicates": [str(pred) for pred in self.shared_layer1_predicates],
            "shared_layer2_predicates": [str(pred) for pred in self.shared_layer2_predicates],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "FOLDepth3ICLSplitBundle":
        version = str(payload.get("version", ""))
        if version not in {"fol_depth3_icl_split_v1", "fol_depth3_icl_split_v2"}:
            raise ValueError(
                "Unsupported split bundle version "
                f"{version!r}; expected one of ('fol_depth3_icl_split_v1', 'fol_depth3_icl_split_v2')."
            )
        train_layer0_predicates = tuple(str(pred) for pred in payload["train_layer0_predicates"])
        eval_layer0_predicates = tuple(str(pred) for pred in payload["eval_layer0_predicates"])
        if version == "fol_depth3_icl_split_v2":
            train_layer0_indices = tuple(int(idx) for idx in payload["train_layer0_indices"])
            eval_layer0_indices = tuple(int(idx) for idx in payload["eval_layer0_indices"])
        else:
            train_layer0_indices = tuple(
                _parse_layered_predicate_index(pred, expected_layer=0)
                for pred in train_layer0_predicates
            )
            eval_layer0_indices = tuple(
                _parse_layered_predicate_index(pred, expected_layer=0)
                for pred in eval_layer0_predicates
            )
        return cls(
            train_bank=FOLRuleBank.from_dict(payload["train_bank"]),
            eval_bank=FOLRuleBank.from_dict(payload["eval_bank"]),
            train_layer0_indices=train_layer0_indices,
            eval_layer0_indices=eval_layer0_indices,
            train_layer0_predicates=train_layer0_predicates,
            eval_layer0_predicates=eval_layer0_predicates,
            shared_layer1_predicates=tuple(
                str(pred) for pred in payload["shared_layer1_predicates"]
            ),
            shared_layer2_predicates=tuple(
                str(pred) for pred in payload["shared_layer2_predicates"]
            ),
        )


@dataclass(frozen=True)
class FOLSampledProblem:
    distance: int
    start_layer: int
    goal_atom: FOLAtom
    step_layers: tuple[int, ...]
    step_ants: tuple[tuple[FOLAtom, ...], ...]
    step_rule_templates: tuple[FOLLayerRule, ...]
    step_rules: tuple[FOLLayerRule, ...]
    step_substitutions: tuple[dict[str, str], ...]


def _atom_sort_key(atom: FOLAtom) -> tuple[str, tuple[str, ...]]:
    return atom.predicate, tuple(atom.args)


def _rule_sort_key(rule: FOLLayerRule) -> tuple:
    return (
        len(rule.lhs),
        tuple(atom.text for atom in rule.lhs),
        len(rule.rhs),
        tuple(atom.text for atom in rule.rhs),
    )


def _sorted_atoms(atoms: Iterable[FOLAtom]) -> tuple[FOLAtom, ...]:
    return tuple(sorted((atom for atom in atoms), key=_atom_sort_key))


def _format_conjunction(atoms: tuple[FOLAtom, ...]) -> str:
    return " ∧ ".join(atom.text for atom in atoms)


def _parse_layered_predicate_index(
    predicate: str,
    *,
    expected_layer: int | None = None,
) -> int:
    match = _LAYERED_PREDICATE_RE.fullmatch(str(predicate))
    if match is None:
        raise ValueError(f"Unsupported layered predicate name: {predicate!r}")
    layer = int(match.group(1))
    index = int(match.group(2))
    if expected_layer is not None and layer != int(expected_layer):
        raise ValueError(
            f"Expected predicate from layer {int(expected_layer)}, got {predicate!r}."
        )
    return int(index)


def _validate_depth3_icl_split_bundle(bundle: FOLDepth3ICLSplitBundle) -> None:
    train_bank = bundle.train_bank
    eval_bank = bundle.eval_bank

    if int(train_bank.n_layers) != 3 or int(eval_bank.n_layers) != 3:
        raise ValueError("Depth-3 ICL split requires train/eval banks with n_layers=3.")

    train_l0 = tuple(str(pred) for pred in bundle.train_layer0_predicates)
    eval_l0 = tuple(str(pred) for pred in bundle.eval_layer0_predicates)
    train_l0_indices = tuple(int(idx) for idx in bundle.train_layer0_indices)
    eval_l0_indices = tuple(int(idx) for idx in bundle.eval_layer0_indices)
    shared_l1 = tuple(str(pred) for pred in bundle.shared_layer1_predicates)
    shared_l2 = tuple(str(pred) for pred in bundle.shared_layer2_predicates)

    if (
        not train_l0
        or not eval_l0
        or not train_l0_indices
        or not eval_l0_indices
        or not shared_l1
        or not shared_l2
    ):
        raise ValueError("Split bundle predicate pools must all be non-empty.")
    if len(train_l0_indices) != len(train_l0):
        raise ValueError(
            "train_layer0_indices length does not match train_layer0_predicates length."
        )
    if len(eval_l0_indices) != len(eval_l0):
        raise ValueError(
            "eval_layer0_indices length does not match eval_layer0_predicates length."
        )
    if any(int(idx) < 1 for idx in train_l0_indices + eval_l0_indices):
        raise ValueError("Layer-0 predicate indices must be >= 1.")
    if len(set(train_l0_indices)) != len(train_l0_indices):
        raise ValueError("train_layer0_indices must be unique.")
    if len(set(eval_l0_indices)) != len(eval_l0_indices):
        raise ValueError("eval_layer0_indices must be unique.")
    if set(train_l0_indices) & set(eval_l0_indices):
        raise ValueError("train/eval layer-0 index pools must be disjoint.")
    if set(train_l0) & set(eval_l0):
        raise ValueError("Train/eval layer-0 predicate pools must be disjoint.")
    parsed_train_indices = tuple(
        _parse_layered_predicate_index(pred, expected_layer=0) for pred in train_l0
    )
    parsed_eval_indices = tuple(
        _parse_layered_predicate_index(pred, expected_layer=0) for pred in eval_l0
    )
    if set(parsed_train_indices) != set(train_l0_indices):
        raise ValueError(
            "train_layer0_indices must match layer-0 predicate indices in train_layer0_predicates."
        )
    if set(parsed_eval_indices) != set(eval_l0_indices):
        raise ValueError(
            "eval_layer0_indices must match layer-0 predicate indices in eval_layer0_predicates."
        )

    if tuple(train_bank.predicates_for_layer(0)) != train_l0:
        raise ValueError("train_bank layer-0 predicates do not match train_layer0_predicates.")
    if tuple(eval_bank.predicates_for_layer(0)) != eval_l0:
        raise ValueError("eval_bank layer-0 predicates do not match eval_layer0_predicates.")
    if tuple(train_bank.predicates_for_layer(1)) != shared_l1:
        raise ValueError("train_bank layer-1 predicates do not match shared_layer1_predicates.")
    if tuple(eval_bank.predicates_for_layer(1)) != shared_l1:
        raise ValueError("eval_bank layer-1 predicates do not match shared_layer1_predicates.")
    if tuple(train_bank.predicates_for_layer(2)) != shared_l2:
        raise ValueError("train_bank layer-2 predicates do not match shared_layer2_predicates.")
    if tuple(eval_bank.predicates_for_layer(2)) != shared_l2:
        raise ValueError("eval_bank layer-2 predicates do not match shared_layer2_predicates.")

    if train_bank.statement_set(1) != eval_bank.statement_set(1):
        raise ValueError("Depth-3 split requires identical layer-1->2 transitions in train/eval.")

    for rule in train_bank.transition_rules(0):
        if not all(atom.predicate in set(train_l0) for atom in rule.lhs):
            raise ValueError("train_bank 0->1 rule uses non-train layer-0 predicate.")
        if not all(atom.predicate in set(shared_l1) for atom in rule.rhs):
            raise ValueError("train_bank 0->1 rule uses non-shared layer-1 predicate.")
    for rule in eval_bank.transition_rules(0):
        if not all(atom.predicate in set(eval_l0) for atom in rule.lhs):
            raise ValueError("eval_bank 0->1 rule uses non-eval layer-0 predicate.")
        if not all(atom.predicate in set(shared_l1) for atom in rule.rhs):
            raise ValueError("eval_bank 0->1 rule uses non-shared layer-1 predicate.")

    shared_predicates = set(shared_l1) | set(shared_l2)
    for pred in shared_predicates:
        train_arity = train_bank.predicate_arities.get(pred)
        eval_arity = eval_bank.predicate_arities.get(pred)
        if train_arity is None or eval_arity is None or int(train_arity) != int(eval_arity):
            raise ValueError("Shared predicates must keep identical arities in train/eval banks.")

    if tuple(train_bank.constants) != tuple(eval_bank.constants):
        raise ValueError("Train/eval split banks must use identical constants.")


# -- Text parsing ----------------------------------------------------------

def parse_atom_text(text: str) -> FOLAtom:
    text = text.strip()
    idx = _skip_ws(text, 0)
    predicate, idx = _parse_ident(text, idx)
    idx = _skip_ws(text, idx)
    if idx >= len(text) or text[idx] != "(":
        # Arity-0 atom: no parentheses.
        idx = _skip_ws(text, idx)
        if idx != len(text):
            raise ValueError(f"Unexpected trailing content in {text!r}")
        return FOLAtom(predicate=predicate, args=())
    idx += 1

    args: list[str] = []
    while True:
        idx = _skip_ws(text, idx)
        if idx < len(text) and text[idx] == ")":
            idx += 1
            break

        arg, idx = _parse_ident(text, idx)
        args.append(arg)
        idx = _skip_ws(text, idx)

        if idx >= len(text):
            raise ValueError(f"Unterminated atom arguments in {text!r}")
        if text[idx] == ",":
            idx += 1
            continue
        if text[idx] == ")":
            idx += 1
            break
        raise ValueError(f"Expected ',' or ')' in {text!r}")

    idx = _skip_ws(text, idx)
    if idx != len(text):
        raise ValueError(f"Unexpected trailing content in {text!r}")

    return FOLAtom(predicate=predicate, args=tuple(args))


def parse_conjunction_text(text: str) -> tuple[FOLAtom, ...]:
    parts = [part.strip() for part in _split_top_level(text, "∧")]
    if not parts or any(not part for part in parts):
        raise ValueError(f"Malformed conjunction in {text!r}")
    return _sorted_atoms(parse_atom_text(part) for part in parts)


def parse_clause_text(text: str) -> tuple[tuple[FOLAtom, ...], tuple[FOLAtom, ...]]:
    lhs_text, rhs_text = _split_top_level_arrow(text)
    lhs = parse_conjunction_text(lhs_text)
    rhs = parse_conjunction_text(rhs_text)
    return lhs, rhs


def parse_sequent_text(text: str) -> FOLSequent:
    text = text.strip()
    depth = 0
    turnstile = -1
    for idx, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                raise ValueError(f"Unbalanced parentheses in {text!r}")
        elif ch == "⊢" and depth == 0:
            if turnstile >= 0:
                raise ValueError(f"Expected a single top-level turnstile in {text!r}")
            turnstile = idx

    if depth != 0:
        raise ValueError(f"Unbalanced parentheses in {text!r}")
    if turnstile < 0:
        raise ValueError(f"Missing top-level turnstile in {text!r}")

    ants_text = text[:turnstile].strip()
    cons_text = text[turnstile + 1 :].strip()
    if not cons_text:
        raise ValueError(f"Missing consequent in {text!r}")

    ants_parts: list[str] = []
    if ants_text:
        ants_parts = [part.strip() for part in _split_top_level(ants_text, ",")]
        if any(not part for part in ants_parts):
            raise ValueError(f"Malformed antecedent list in {text!r}")

    ants = _sorted_atoms(parse_atom_text(part) for part in ants_parts)
    cons = parse_atom_text(cons_text)
    return FOLSequent(ants=ants, cons=cons)


# -- I/O -------------------------------------------------------------------

def save_fol_rule_bank(path: Path, bank: FOLRuleBank) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bank.to_dict(), indent=2))


def load_fol_rule_bank(path: Path) -> FOLRuleBank:
    payload = json.loads(path.read_text())
    return FOLRuleBank.from_dict(payload)


def save_fol_depth3_icl_split_bundle(
    path: Path,
    bundle: FOLDepth3ICLSplitBundle,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle.to_dict(), indent=2))


def load_fol_depth3_icl_split_bundle(path: Path) -> FOLDepth3ICLSplitBundle:
    payload = json.loads(path.read_text())
    return FOLDepth3ICLSplitBundle.from_dict(payload)
