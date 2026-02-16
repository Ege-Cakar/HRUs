"""Rule-bank utilities for layered modus-ponens tasks."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

try:
    from task.prop_gen.util.elem import And, Atom, Implies, Proposition
except ModuleNotFoundError:
    ROOT = Path(__file__).resolve().parents[3]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    from task.prop_gen.util.elem import And, Atom, Implies, Proposition  # type: ignore


def _atom_sort_key(name: str) -> tuple[int, int]:
    if not name.startswith("p") or "_" not in name:
        raise ValueError(f"Invalid layered atom name: {name}")
    layer_part, idx_part = name[1:].split("_", maxsplit=1)
    return int(layer_part), int(idx_part)


def _atoms_to_prop(atoms: tuple[str, ...]) -> Proposition:
    if not atoms:
        raise ValueError("Conjunction must contain at least one atom.")
    out: Proposition = Atom(atoms[0])
    for atom in atoms[1:]:
        out = And(out, Atom(atom))
    return out


@dataclass(frozen=True)
class LayerRule:
    src_layer: int
    dst_layer: int
    lhs: tuple[str, ...]
    rhs: tuple[str, ...]

    def statement_prop(self) -> Implies:
        return Implies(_atoms_to_prop(self.lhs), _atoms_to_prop(self.rhs))

    @property
    def statement_text(self) -> str:
        return str(self.statement_prop())

    def to_dict(self) -> dict:
        return {
            "src_layer": int(self.src_layer),
            "dst_layer": int(self.dst_layer),
            "lhs": list(self.lhs),
            "rhs": list(self.rhs),
            "statement": self.statement_text,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "LayerRule":
        return cls(
            src_layer=int(payload["src_layer"]),
            dst_layer=int(payload["dst_layer"]),
            lhs=tuple(str(tok) for tok in payload["lhs"]),
            rhs=tuple(str(tok) for tok in payload["rhs"]),
        )


@dataclass(frozen=True)
class RuleBank:
    n_layers: int
    props_per_layer: int
    transitions: dict[int, tuple[LayerRule, ...]]

    def transition_rules(self, src_layer: int) -> tuple[LayerRule, ...]:
        return self.transitions.get(src_layer, ())

    def statement_set(self, src_layer: int) -> set[str]:
        return {rule.statement_text for rule in self.transition_rules(src_layer)}

    def to_dict(self) -> dict:
        serialized = {}
        for src_layer, rules in self.transitions.items():
            serialized[str(src_layer)] = [rule.to_dict() for rule in rules]

        return {
            "n_layers": int(self.n_layers),
            "props_per_layer": int(self.props_per_layer),
            "transitions": serialized,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "RuleBank":
        transitions = {}
        for src_key, rules in payload["transitions"].items():
            transitions[int(src_key)] = tuple(
                sorted(
                    (LayerRule.from_dict(rule) for rule in rules),
                    key=lambda r: (len(r.lhs), r.lhs, len(r.rhs), r.rhs),
                )
            )

        return cls(
            n_layers=int(payload["n_layers"]),
            props_per_layer=int(payload["props_per_layer"]),
            transitions=transitions,
        )


@dataclass(frozen=True)
class SampledProblem:
    distance: int
    start_layer: int
    goal_atom: str
    step_layers: tuple[int, ...]
    step_ants: tuple[tuple[str, ...], ...]
    step_rules: tuple[LayerRule, ...]



def _layer_atoms(layer: int, props_per_layer: int) -> list[str]:
    return [f"p{layer}_{idx}" for idx in range(1, props_per_layer + 1)]


def build_random_rule_bank(
    *,
    n_layers: int,
    props_per_layer: int,
    rules_per_transition: int,
    k_in_max: int,
    k_out_max: int,
    rng: np.random.Generator,
) -> RuleBank:
    if n_layers < 2:
        raise ValueError(f"n_layers must be >= 2, got {n_layers}")
    if props_per_layer < 1:
        raise ValueError(f"props_per_layer must be >= 1, got {props_per_layer}")
    if rules_per_transition < 1:
        raise ValueError(f"rules_per_transition must be >= 1, got {rules_per_transition}")
    if k_in_max < 1 or k_out_max < 1:
        raise ValueError("k_in_max and k_out_max must be >= 1")

    transitions: dict[int, tuple[LayerRule, ...]] = {}

    for src_layer in range(n_layers - 1):
        dst_layer = src_layer + 1
        src_atoms = np.array(_layer_atoms(src_layer, props_per_layer), dtype=object)
        dst_atoms = np.array(_layer_atoms(dst_layer, props_per_layer), dtype=object)

        seen: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
        rules: list[LayerRule] = []

        max_lhs = min(k_in_max, props_per_layer)
        max_rhs = min(k_out_max, props_per_layer)

        attempts = 0
        max_attempts = max(1_000, rules_per_transition * 100)
        while len(rules) < rules_per_transition and attempts < max_attempts:
            attempts += 1
            lhs_size = int(rng.integers(1, max_lhs + 1))
            rhs_size = int(rng.integers(1, max_rhs + 1))

            lhs_raw = tuple(str(tok) for tok in rng.choice(src_atoms, size=lhs_size, replace=False))
            rhs_raw = tuple(str(tok) for tok in rng.choice(dst_atoms, size=rhs_size, replace=False))

            lhs = tuple(sorted(lhs_raw, key=_atom_sort_key))
            rhs = tuple(sorted(rhs_raw, key=_atom_sort_key))

            sig = (lhs, rhs)
            if sig in seen:
                continue
            seen.add(sig)
            rules.append(
                LayerRule(
                    src_layer=src_layer,
                    dst_layer=dst_layer,
                    lhs=lhs,
                    rhs=rhs,
                )
            )

        if len(rules) < rules_per_transition:
            raise RuntimeError(
                f"Could not sample enough unique rules for transition {src_layer}->{dst_layer}."
            )

        transitions[src_layer] = tuple(
            sorted(
                rules,
                key=lambda r: (len(r.lhs), r.lhs, len(r.rhs), r.rhs),
            )
        )

    return RuleBank(
        n_layers=n_layers,
        props_per_layer=props_per_layer,
        transitions=transitions,
    )


def _sorted_atoms(atoms: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted((str(atom) for atom in atoms), key=_atom_sort_key))


def sample_problem(
    *,
    bank: RuleBank,
    distance: int,
    initial_ant_max: int,
    rng: np.random.Generator,
    max_attempts: int = 256,
) -> SampledProblem:
    if distance < 1:
        raise ValueError(f"distance must be >= 1, got {distance}")
    if distance >= bank.n_layers:
        raise ValueError(
            f"distance {distance} is too large for n_layers={bank.n_layers}."
        )

    max_start = bank.n_layers - distance - 1
    if max_start < 0:
        raise ValueError(
            f"No valid start layer for distance {distance} with n_layers={bank.n_layers}."
        )

    initial_ant_max = max(1, min(initial_ant_max, bank.props_per_layer))

    for _ in range(max_attempts):
        start_layer = int(rng.integers(0, max_start + 1))
        initial_size = int(rng.integers(1, initial_ant_max + 1))
        initial_pool = np.array(_layer_atoms(start_layer, bank.props_per_layer), dtype=object)
        initial_ants = _sorted_atoms(rng.choice(initial_pool, size=initial_size, replace=False))

        facts: dict[int, set[str]] = {start_layer: set(initial_ants)}
        step_layers: list[int] = []
        step_ants: list[tuple[str, ...]] = []
        step_rules: list[LayerRule] = []

        feasible = True
        for step in range(distance):
            src_layer = start_layer + step
            dst_layer = src_layer + 1
            src_facts = facts.get(src_layer, set())
            if not src_facts:
                feasible = False
                break

            candidates = [
                rule for rule in bank.transition_rules(src_layer)
                if set(rule.lhs).issubset(src_facts)
            ]
            if not candidates:
                feasible = False
                break

            pick_idx = int(rng.integers(0, len(candidates)))
            picked = candidates[pick_idx]

            step_layers.append(src_layer)
            step_ants.append(_sorted_atoms(src_facts))
            step_rules.append(picked)

            facts.setdefault(dst_layer, set()).update(picked.rhs)

        final_layer = start_layer + distance
        final_facts = _sorted_atoms(facts.get(final_layer, set()))
        if not feasible or not final_facts:
            continue

        goal_atom = str(final_facts[int(rng.integers(0, len(final_facts)))])
        return SampledProblem(
            distance=distance,
            start_layer=start_layer,
            goal_atom=goal_atom,
            step_layers=tuple(step_layers),
            step_ants=tuple(step_ants),
            step_rules=tuple(step_rules),
        )

    raise RuntimeError(
        f"Failed to sample feasible problem after {max_attempts} attempts for distance={distance}."
    )


def save_rule_bank(path: Path, bank: RuleBank) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bank.to_dict(), indent=2))


def load_rule_bank(path: Path) -> RuleBank:
    payload = json.loads(path.read_text())
    return RuleBank.from_dict(payload)
