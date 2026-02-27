"""Rule-bank utilities for layered first-order reasoning tasks."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Iterable

import numpy as np

_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_LAYERED_PREDICATE_RE = re.compile(r"r(\d+)_(\d+)$")


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
        if len(self.args) < 1:
            raise ValueError("FOL atoms must have at least one argument.")
        for arg in self.args:
            if not _is_identifier(arg):
                raise ValueError(f"Invalid atom argument: {arg!r}")

    @property
    def arity(self) -> int:
        return len(self.args)

    @property
    def text(self) -> str:
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
    predicates_per_layer: int
    arity_max: int
    constants: tuple[str, ...]
    vars_per_rule_max: int
    predicate_arities: dict[str, int]
    transitions: dict[int, tuple[FOLLayerRule, ...]]
    layer_predicates: dict[int, tuple[str, ...]] | None = None

    def transition_rules(self, src_layer: int) -> tuple[FOLLayerRule, ...]:
        return self.transitions.get(src_layer, ())

    def statement_set(self, src_layer: int) -> set[str]:
        return {rule.statement_text for rule in self.transition_rules(src_layer)}

    def predicates_for_layer(self, layer: int) -> tuple[str, ...]:
        if self.layer_predicates is not None and int(layer) in self.layer_predicates:
            return tuple(self.layer_predicates[int(layer)])
        return tuple(f"r{int(layer)}_{idx}" for idx in range(1, int(self.predicates_per_layer) + 1))

    def to_dict(self) -> dict:
        transitions = {
            str(layer): [rule.to_dict() for rule in rules]
            for layer, rules in self.transitions.items()
        }
        return {
            "n_layers": int(self.n_layers),
            "predicates_per_layer": int(self.predicates_per_layer),
            "arity_max": int(self.arity_max),
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
            predicates_per_layer=int(payload["predicates_per_layer"]),
            arity_max=int(payload["arity_max"]),
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


def parse_atom_text(text: str) -> FOLAtom:
    text = text.strip()
    idx = _skip_ws(text, 0)
    predicate, idx = _parse_ident(text, idx)
    idx = _skip_ws(text, idx)
    if idx >= len(text) or text[idx] != "(":
        raise ValueError(f"Expected '(' after predicate in {text!r}")
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
    if not args:
        raise ValueError(f"Atom must contain at least one argument in {text!r}")

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


def _canonicalize_rule(
    *,
    src_layer: int,
    lhs: Iterable[FOLAtom],
    rhs: Iterable[FOLAtom],
) -> FOLLayerRule:
    return FOLLayerRule(
        src_layer=int(src_layer),
        dst_layer=int(src_layer) + 1,
        lhs=_sorted_atoms(lhs),
        rhs=_sorted_atoms(rhs),
    )


def _layer_predicates(layer: int, predicates_per_layer: int) -> tuple[str, ...]:
    return tuple(f"r{int(layer)}_{idx}" for idx in range(1, predicates_per_layer + 1))


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


def _build_random_atom(
    *,
    predicate: str,
    predicate_arities: dict[str, int],
    term_pool: tuple[str, ...],
    rng: np.random.Generator,
) -> FOLAtom:
    arity = int(predicate_arities[predicate])
    args = tuple(
        str(term_pool[int(rng.integers(0, len(term_pool)))]) for _ in range(arity)
    )
    return FOLAtom(predicate=predicate, args=args)


def _sample_transition_rules(
    *,
    src_layer: int,
    lhs_predicates: tuple[str, ...],
    rhs_predicates: tuple[str, ...],
    rules_per_transition: int,
    k_in_min: int,
    k_in_max: int,
    k_out_min: int,
    k_out_max: int,
    predicate_arities: dict[str, int],
    var_pool: tuple[str, ...],
    rng: np.random.Generator,
) -> tuple[FOLLayerRule, ...]:
    lhs_predicates_arr = np.asarray(lhs_predicates, dtype=object)
    rhs_predicates_arr = np.asarray(rhs_predicates, dtype=object)
    if lhs_predicates_arr.size < 1 or rhs_predicates_arr.size < 1:
        raise ValueError(
            f"Transition {src_layer}->{src_layer + 1} requires non-empty predicate pools."
        )

    min_lhs = int(k_in_min)
    max_lhs = min(int(k_in_max), int(lhs_predicates_arr.size))
    min_rhs = int(k_out_min)
    max_rhs = min(int(k_out_max), int(rhs_predicates_arr.size))
    if min_lhs > max_lhs or min_rhs > max_rhs:
        raise ValueError(
            "k_in_min/k_out_min exceed available predicates or configured maxima for "
            f"transition {src_layer}->{src_layer + 1}: "
            f"k_in_min={min_lhs}, k_in_max={int(k_in_max)}, lhs_pool={int(lhs_predicates_arr.size)}, "
            f"k_out_min={min_rhs}, k_out_max={int(k_out_max)}, rhs_pool={int(rhs_predicates_arr.size)}."
        )

    seen: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    rules: list[FOLLayerRule] = []
    attempts = 0
    max_attempts = max(1_000, int(rules_per_transition) * 250)

    while len(rules) < int(rules_per_transition) and attempts < max_attempts:
        attempts += 1
        lhs_size = int(rng.integers(min_lhs, max_lhs + 1))
        rhs_size = int(rng.integers(min_rhs, max_rhs + 1))

        lhs_chosen = [
            str(tok)
            for tok in rng.choice(lhs_predicates_arr, size=lhs_size, replace=False)
        ]
        rhs_chosen = [
            str(tok)
            for tok in rng.choice(rhs_predicates_arr, size=rhs_size, replace=False)
        ]

        lhs_atoms = tuple(
            _build_random_atom(
                predicate=pred,
                predicate_arities=predicate_arities,
                term_pool=var_pool,
                rng=rng,
            )
            for pred in lhs_chosen
        )
        lhs_vars = {
            term
            for atom in lhs_atoms
            for term in atom.args
            if _is_variable(term)
        }
        if not lhs_vars:
            continue

        rhs_atoms = tuple(
            _build_random_atom(
                predicate=pred,
                predicate_arities=predicate_arities,
                term_pool=tuple(sorted(lhs_vars)),
                rng=rng,
            )
            for pred in rhs_chosen
        )

        lhs = _sorted_atoms(lhs_atoms)
        rhs = _sorted_atoms(rhs_atoms)

        rhs_vars = {
            term
            for atom in rhs
            for term in atom.args
            if _is_variable(term)
        }
        if not rhs_vars.issubset(lhs_vars):
            continue

        sig = (
            tuple(atom.text for atom in lhs),
            tuple(atom.text for atom in rhs),
        )
        if sig in seen:
            continue
        seen.add(sig)

        rules.append(
            _canonicalize_rule(
                src_layer=src_layer,
                lhs=lhs,
                rhs=rhs,
            )
        )

    if len(rules) < int(rules_per_transition):
        raise RuntimeError(
            "Could not sample enough unique FOL rules for transition "
            f"{src_layer}->{src_layer + 1}."
        )

    return tuple(sorted(rules, key=_rule_sort_key))


def build_random_fol_rule_bank(
    *,
    n_layers: int,
    predicates_per_layer: int,
    rules_per_transition: int,
    arity_max: int,
    vars_per_rule_max: int,
    k_in_max: int,
    k_out_max: int,
    constants: Iterable[str],
    rng: np.random.Generator,
    k_in_min: int = 1,
    k_out_min: int = 1,
) -> FOLRuleBank:
    if n_layers < 2:
        raise ValueError(f"n_layers must be >= 2, got {n_layers}")
    if predicates_per_layer < 1:
        raise ValueError(
            f"predicates_per_layer must be >= 1, got {predicates_per_layer}"
        )
    if rules_per_transition < 1:
        raise ValueError(
            f"rules_per_transition must be >= 1, got {rules_per_transition}"
        )
    if arity_max < 1:
        raise ValueError(f"arity_max must be >= 1, got {arity_max}")
    if vars_per_rule_max < 1:
        raise ValueError(f"vars_per_rule_max must be >= 1, got {vars_per_rule_max}")
    if k_in_min < 1 or k_out_min < 1:
        raise ValueError("k_in_min and k_out_min must be >= 1")
    if k_in_max < 1 or k_out_max < 1:
        raise ValueError("k_in_max and k_out_max must be >= 1")
    if k_in_min > k_in_max:
        raise ValueError(
            f"k_in_min must be <= k_in_max, got k_in_min={k_in_min}, k_in_max={k_in_max}"
        )
    if k_out_min > k_out_max:
        raise ValueError(
            f"k_out_min must be <= k_out_max, got k_out_min={k_out_min}, k_out_max={k_out_max}"
        )
    if k_in_min > int(predicates_per_layer) or k_out_min > int(predicates_per_layer):
        raise ValueError(
            "k_in_min and k_out_min cannot exceed predicates_per_layer for random bank "
            f"generation; got predicates_per_layer={predicates_per_layer}, "
            f"k_in_min={k_in_min}, k_out_min={k_out_min}."
        )

    constants = tuple(str(tok) for tok in constants)
    if not constants:
        raise ValueError("constants must contain at least one symbol")
    for token in constants:
        if not _is_identifier(token):
            raise ValueError(f"Invalid constant symbol: {token!r}")

    predicate_arities: dict[str, int] = {}
    transitions: dict[int, tuple[FOLLayerRule, ...]] = {}
    layer_predicates = {
        int(layer): _layer_predicates(layer, int(predicates_per_layer))
        for layer in range(int(n_layers))
    }
    var_pool = tuple(f"x{idx}" for idx in range(1, vars_per_rule_max + 1))

    for layer in range(n_layers):
        for predicate in layer_predicates[int(layer)]:
            predicate_arities[predicate] = int(rng.integers(1, arity_max + 1))

    for src_layer in range(n_layers - 1):
        transitions[src_layer] = _sample_transition_rules(
            src_layer=int(src_layer),
            lhs_predicates=tuple(layer_predicates[int(src_layer)]),
            rhs_predicates=tuple(layer_predicates[int(src_layer) + 1]),
            rules_per_transition=int(rules_per_transition),
            k_in_min=int(k_in_min),
            k_in_max=int(k_in_max),
            k_out_min=int(k_out_min),
            k_out_max=int(k_out_max),
            predicate_arities=predicate_arities,
            var_pool=var_pool,
            rng=rng,
        )

    return FOLRuleBank(
        n_layers=int(n_layers),
        predicates_per_layer=int(predicates_per_layer),
        arity_max=int(arity_max),
        constants=constants,
        vars_per_rule_max=int(vars_per_rule_max),
        predicate_arities=predicate_arities,
        transitions=transitions,
        layer_predicates=layer_predicates,
    )


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


def build_depth3_icl_split_bundle(
    *,
    predicates_per_layer: int,
    rules_01_train: int,
    rules_01_eval: int,
    rules_12_shared: int,
    arity_max: int,
    vars_per_rule_max: int,
    k_in_max: int,
    k_out_max: int,
    constants: Iterable[str],
    rng: np.random.Generator,
    k_in_min: int = 1,
    k_out_min: int = 1,
) -> FOLDepth3ICLSplitBundle:
    if int(predicates_per_layer) < 1:
        raise ValueError(
            f"predicates_per_layer must be >= 1, got {predicates_per_layer}"
        )
    if int(rules_01_train) < 1 or int(rules_01_eval) < 1 or int(rules_12_shared) < 1:
        raise ValueError("All split transition rule counts must be >= 1.")
    if int(arity_max) < 1:
        raise ValueError(f"arity_max must be >= 1, got {arity_max}")
    if int(vars_per_rule_max) < 1:
        raise ValueError(f"vars_per_rule_max must be >= 1, got {vars_per_rule_max}")
    if int(k_in_min) < 1 or int(k_out_min) < 1:
        raise ValueError("k_in_min and k_out_min must be >= 1")
    if int(k_in_max) < 1 or int(k_out_max) < 1:
        raise ValueError("k_in_max and k_out_max must be >= 1")
    if int(k_in_min) > int(k_in_max):
        raise ValueError(
            f"k_in_min must be <= k_in_max, got k_in_min={k_in_min}, k_in_max={k_in_max}"
        )
    if int(k_out_min) > int(k_out_max):
        raise ValueError(
            f"k_out_min must be <= k_out_max, got k_out_min={k_out_min}, k_out_max={k_out_max}"
        )
    if int(k_in_min) > int(predicates_per_layer) or int(k_out_min) > int(predicates_per_layer):
        raise ValueError(
            "k_in_min and k_out_min cannot exceed predicates_per_layer for depth-3 split "
            f"generation; got predicates_per_layer={predicates_per_layer}, "
            f"k_in_min={k_in_min}, k_out_min={k_out_min}."
        )

    constants = tuple(str(tok) for tok in constants)
    if not constants:
        raise ValueError("constants must contain at least one symbol")
    for token in constants:
        if not _is_identifier(token):
            raise ValueError(f"Invalid constant symbol: {token!r}")

    predicates_per_layer = int(predicates_per_layer)
    arity_max = int(arity_max)
    vars_per_rule_max = int(vars_per_rule_max)
    k_in_min = int(k_in_min)
    k_in_max = int(k_in_max)
    k_out_min = int(k_out_min)
    k_out_max = int(k_out_max)

    all_layer0_indices = np.arange(1, 2 * predicates_per_layer + 1, dtype=np.int32)
    shuffled_layer0_indices = rng.permutation(all_layer0_indices)
    train_layer0_indices = tuple(
        sorted(int(idx) for idx in shuffled_layer0_indices[:predicates_per_layer].tolist())
    )
    eval_layer0_indices = tuple(
        sorted(int(idx) for idx in shuffled_layer0_indices[predicates_per_layer:].tolist())
    )
    train_layer0 = tuple(f"r0_{idx}" for idx in train_layer0_indices)
    eval_layer0 = tuple(f"r0_{idx}" for idx in eval_layer0_indices)
    shared_layer1 = _layer_predicates(1, predicates_per_layer)
    shared_layer2 = _layer_predicates(2, predicates_per_layer)

    predicate_arities_all: dict[str, int] = {}
    for predicate in train_layer0 + eval_layer0 + shared_layer1 + shared_layer2:
        if predicate in predicate_arities_all:
            continue
        predicate_arities_all[predicate] = int(rng.integers(1, arity_max + 1))

    train_predicate_arities = {
        predicate: int(predicate_arities_all[predicate])
        for predicate in train_layer0 + shared_layer1 + shared_layer2
    }
    eval_predicate_arities = {
        predicate: int(predicate_arities_all[predicate])
        for predicate in eval_layer0 + shared_layer1 + shared_layer2
    }

    var_pool = tuple(f"x{idx}" for idx in range(1, vars_per_rule_max + 1))
    shared_rules_12 = _sample_transition_rules(
        src_layer=1,
        lhs_predicates=tuple(shared_layer1),
        rhs_predicates=tuple(shared_layer2),
        rules_per_transition=int(rules_12_shared),
        k_in_min=int(k_in_min),
        k_in_max=int(k_in_max),
        k_out_min=int(k_out_min),
        k_out_max=int(k_out_max),
        predicate_arities=train_predicate_arities,
        var_pool=var_pool,
        rng=rng,
    )
    train_rules_01 = _sample_transition_rules(
        src_layer=0,
        lhs_predicates=tuple(train_layer0),
        rhs_predicates=tuple(shared_layer1),
        rules_per_transition=int(rules_01_train),
        k_in_min=int(k_in_min),
        k_in_max=int(k_in_max),
        k_out_min=int(k_out_min),
        k_out_max=int(k_out_max),
        predicate_arities=train_predicate_arities,
        var_pool=var_pool,
        rng=rng,
    )
    eval_rules_01 = _sample_transition_rules(
        src_layer=0,
        lhs_predicates=tuple(eval_layer0),
        rhs_predicates=tuple(shared_layer1),
        rules_per_transition=int(rules_01_eval),
        k_in_min=int(k_in_min),
        k_in_max=int(k_in_max),
        k_out_min=int(k_out_min),
        k_out_max=int(k_out_max),
        predicate_arities=eval_predicate_arities,
        var_pool=var_pool,
        rng=rng,
    )

    train_bank = FOLRuleBank(
        n_layers=3,
        predicates_per_layer=predicates_per_layer,
        arity_max=arity_max,
        constants=constants,
        vars_per_rule_max=vars_per_rule_max,
        predicate_arities=train_predicate_arities,
        transitions={
            0: tuple(train_rules_01),
            1: tuple(shared_rules_12),
        },
        layer_predicates={
            0: tuple(train_layer0),
            1: tuple(shared_layer1),
            2: tuple(shared_layer2),
        },
    )
    eval_bank = FOLRuleBank(
        n_layers=3,
        predicates_per_layer=predicates_per_layer,
        arity_max=arity_max,
        constants=constants,
        vars_per_rule_max=vars_per_rule_max,
        predicate_arities=eval_predicate_arities,
        transitions={
            0: tuple(eval_rules_01),
            1: tuple(shared_rules_12),
        },
        layer_predicates={
            0: tuple(eval_layer0),
            1: tuple(shared_layer1),
            2: tuple(shared_layer2),
        },
    )
    return FOLDepth3ICLSplitBundle(
        train_bank=train_bank,
        eval_bank=eval_bank,
        train_layer0_indices=tuple(train_layer0_indices),
        eval_layer0_indices=tuple(eval_layer0_indices),
        train_layer0_predicates=tuple(train_layer0),
        eval_layer0_predicates=tuple(eval_layer0),
        shared_layer1_predicates=tuple(shared_layer1),
        shared_layer2_predicates=tuple(shared_layer2),
    )


def _unify_template_atom(
    template: FOLAtom,
    ground: FOLAtom,
    subst: dict[str, str],
) -> dict[str, str] | None:
    # Unify one template atom with one ground fact under the current substitution.
    if template.predicate != ground.predicate:
        return None
    if len(template.args) != len(ground.args):
        return None

    out = dict(subst)
    for templ_term, ground_term in zip(template.args, ground.args):
        if _is_variable(templ_term):
            bound = out.get(templ_term)
            if bound is None:
                out[templ_term] = ground_term
            elif bound != ground_term:
                return None
        elif templ_term != ground_term:
            return None
    return out


def _find_lhs_substitutions(
    *,
    lhs: tuple[FOLAtom, ...],
    facts: tuple[FOLAtom, ...],
    max_solutions: int,
) -> list[dict[str, str]]:
    # Backtracking search for substitutions that make every LHS atom match facts.
    solutions: list[dict[str, str]] = []

    def _search(idx: int, subst: dict[str, str]) -> None:
        if len(solutions) >= max_solutions:
            return
        if idx >= len(lhs):
            solutions.append(dict(subst))
            return

        templ = lhs[idx]
        # Try matching this LHS atom against every available fact.
        for fact in facts:
            maybe = _unify_template_atom(templ, fact, subst)
            if maybe is None:
                continue
            _search(idx + 1, maybe)
            if len(solutions) >= max_solutions:
                return

    _search(0, {})
    return solutions


def _has_rhs_support(
    *,
    rule: FOLLayerRule,
    subst: dict[str, str],
) -> bool:
    # Keep only substitutions that bind all variables used on the RHS.
    rhs_vars = {
        term
        for atom in rule.rhs
        for term in atom.args
        if _is_variable(term)
    }
    return rhs_vars.issubset(set(subst))


def sample_fol_problem(
    *,
    bank: FOLRuleBank,
    distance: int,
    initial_ant_max: int,
    rng: np.random.Generator,
    max_attempts: int = 1024,
    max_unify_solutions: int = 128,
) -> FOLSampledProblem:
    if distance < 1:
        raise ValueError(f"distance must be >= 1, got {distance}")
    if distance >= bank.n_layers:
        raise ValueError(
            f"distance {distance} is too large for n_layers={bank.n_layers}."
        )
    if max_unify_solutions < 1:
        raise ValueError("max_unify_solutions must be >= 1")

    max_start = bank.n_layers - distance - 1
    if max_start < 0:
        raise ValueError(
            f"No valid start layer for distance {distance} with n_layers={bank.n_layers}."
        )

    constants = np.asarray(bank.constants, dtype=object)

    for _ in range(max_attempts):
        start_layer = int(rng.integers(0, max_start + 1))
        predicates = np.asarray(bank.predicates_for_layer(start_layer), dtype=object)
        if predicates.size < 1:
            continue
        max_initial = max(1, min(int(initial_ant_max), int(predicates.size)))
        initial_size = int(rng.integers(1, max_initial + 1))
        picked = [
            str(tok)
            for tok in rng.choice(predicates, size=initial_size, replace=False)
        ]

        initial_facts = {
            FOLAtom(
                predicate=pred,
                args=tuple(
                    str(constants[int(rng.integers(0, len(constants)))])
                    for _ in range(bank.predicate_arities[pred])
                ),
            )
            for pred in picked
        }

        facts_by_layer: dict[int, set[FOLAtom]] = {start_layer: set(initial_facts)}
        step_layers: list[int] = []
        step_ants: list[tuple[FOLAtom, ...]] = []
        step_templates: list[FOLLayerRule] = []
        step_rules: list[FOLLayerRule] = []
        step_substitutions: list[dict[str, str]] = []

        feasible = True
        for step in range(distance):
            src_layer = start_layer + step
            dst_layer = src_layer + 1
            src_facts = _sorted_atoms(facts_by_layer.get(src_layer, set()))
            if not src_facts:
                feasible = False
                break

            candidates: list[tuple[FOLLayerRule, dict[str, str]]] = []
            for rule in bank.transition_rules(src_layer):
                subs = _find_lhs_substitutions(
                    lhs=rule.lhs,
                    facts=src_facts,
                    max_solutions=max_unify_solutions,
                )
                if not subs:
                    continue

                valid_subs = [
                    sub
                    for sub in subs
                    if _has_rhs_support(rule=rule, subst=sub)
                ]
                if not valid_subs:
                    continue

                pick_sub = valid_subs[int(rng.integers(0, len(valid_subs)))]
                candidates.append((rule, pick_sub))

            if not candidates:
                feasible = False
                break

            pick_idx = int(rng.integers(0, len(candidates)))
            template, subst = candidates[pick_idx]
            instantiated = template.instantiate(subst)

            step_layers.append(src_layer)
            step_ants.append(src_facts)
            step_templates.append(template)
            step_rules.append(instantiated)
            step_substitutions.append(dict(subst))

            facts_by_layer.setdefault(dst_layer, set()).update(instantiated.rhs)

        final_layer = start_layer + distance
        final_facts = _sorted_atoms(facts_by_layer.get(final_layer, set()))
        if not feasible or not final_facts:
            continue

        goal_atom = final_facts[int(rng.integers(0, len(final_facts)))]
        return FOLSampledProblem(
            distance=int(distance),
            start_layer=int(start_layer),
            goal_atom=goal_atom,
            step_layers=tuple(step_layers),
            step_ants=tuple(step_ants),
            step_rule_templates=tuple(step_templates),
            step_rules=tuple(step_rules),
            step_substitutions=tuple(step_substitutions),
        )

    raise RuntimeError(
        "Failed to sample feasible FOL problem after "
        f"{max_attempts} attempts for distance={distance}."
    )


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
