"""Rule-bank utilities for layered first-order reasoning tasks."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Iterable

import numpy as np

_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


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

    def transition_rules(self, src_layer: int) -> tuple[FOLLayerRule, ...]:
        return self.transitions.get(src_layer, ())

    def statement_set(self, src_layer: int) -> set[str]:
        return {rule.statement_text for rule in self.transition_rules(src_layer)}

    def predicates_for_layer(self, layer: int) -> tuple[str, ...]:
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
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "FOLRuleBank":
        transitions: dict[int, tuple[FOLLayerRule, ...]] = {}
        for src_key, rules in payload["transitions"].items():
            src = int(src_key)
            transitions[src] = tuple(
                sorted(
                    (FOLLayerRule.from_dict(rule) for rule in rules),
                    key=lambda rule: (
                        len(rule.lhs),
                        tuple(atom.text for atom in rule.lhs),
                        len(rule.rhs),
                        tuple(atom.text for atom in rule.rhs),
                    ),
                )
            )

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
    if k_in_max < 1 or k_out_max < 1:
        raise ValueError("k_in_max and k_out_max must be >= 1")

    constants = tuple(str(tok) for tok in constants)
    if not constants:
        raise ValueError("constants must contain at least one symbol")
    for token in constants:
        if not _is_identifier(token):
            raise ValueError(f"Invalid constant symbol: {token!r}")

    predicate_arities: dict[str, int] = {}
    transitions: dict[int, tuple[FOLLayerRule, ...]] = {}
    var_pool = tuple(f"x{idx}" for idx in range(1, vars_per_rule_max + 1))

    for layer in range(n_layers):
        for predicate in _layer_predicates(layer, predicates_per_layer):
            predicate_arities[predicate] = int(rng.integers(1, arity_max + 1))

    for src_layer in range(n_layers - 1):
        lhs_predicates = np.asarray(_layer_predicates(src_layer, predicates_per_layer), dtype=object)
        rhs_predicates = np.asarray(_layer_predicates(src_layer + 1, predicates_per_layer), dtype=object)

        max_lhs = min(k_in_max, len(lhs_predicates))
        max_rhs = min(k_out_max, len(rhs_predicates))

        seen: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
        rules: list[FOLLayerRule] = []
        attempts = 0
        max_attempts = max(1_000, rules_per_transition * 250)

        while len(rules) < rules_per_transition and attempts < max_attempts:
            attempts += 1
            lhs_size = int(rng.integers(1, max_lhs + 1))
            rhs_size = int(rng.integers(1, max_rhs + 1))

            lhs_chosen = [
                str(tok)
                for tok in rng.choice(lhs_predicates, size=lhs_size, replace=False)
            ]
            rhs_chosen = [
                str(tok)
                for tok in rng.choice(rhs_predicates, size=rhs_size, replace=False)
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

        if len(rules) < rules_per_transition:
            raise RuntimeError(
                "Could not sample enough unique FOL rules for transition "
                f"{src_layer}->{src_layer + 1}."
            )

        transitions[src_layer] = tuple(
            sorted(
                rules,
                key=lambda rule: (
                    len(rule.lhs),
                    tuple(atom.text for atom in rule.lhs),
                    len(rule.rhs),
                    tuple(atom.text for atom in rule.rhs),
                ),
            )
        )

    return FOLRuleBank(
        n_layers=int(n_layers),
        predicates_per_layer=int(predicates_per_layer),
        arity_max=int(arity_max),
        constants=constants,
        vars_per_rule_max=int(vars_per_rule_max),
        predicate_arities=predicate_arities,
        transitions=transitions,
    )


def _unify_template_atom(
    template: FOLAtom,
    ground: FOLAtom,
    subst: dict[str, str],
) -> dict[str, str] | None:
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
    solutions: list[dict[str, str]] = []

    def _search(idx: int, subst: dict[str, str]) -> None:
        if len(solutions) >= max_solutions:
            return
        if idx >= len(lhs):
            solutions.append(dict(subst))
            return

        templ = lhs[idx]
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

    initial_ant_max = max(1, min(int(initial_ant_max), int(bank.predicates_per_layer)))
    constants = np.asarray(bank.constants, dtype=object)

    for _ in range(max_attempts):
        start_layer = int(rng.integers(0, max_start + 1))
        initial_size = int(rng.integers(1, initial_ant_max + 1))
        predicates = np.asarray(bank.predicates_for_layer(start_layer), dtype=object)
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
