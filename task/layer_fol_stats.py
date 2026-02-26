# <codecell>
"""Interactive diagnostics for layered first-order tasks.

This script estimates:
1) P(exists valid path of exactly distance d) under independent start/goal draws.
2) Distribution of shortest path lengths under a separate random start/goal protocol.
3) Sensitivity of long-distance sampling feasibility to key rule-bank parameters.

Usage:
- Edit CONFIG directly in this file.
- Run the file as a script or execute codecells interactively.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from task.layer_gen.util.fol_rule_bank import (  # noqa: E402
    FOLAtom,
    FOLLayerRule,
    FOLRuleBank,
    build_random_fol_rule_bank,
    sample_fol_problem,
)


# <codecell>
# Edit all settings here (no argparse by design).
CONFIG = {
    "seed": np.random.randint(0, np.iinfo(np.int32).max),
    "out_dir": ROOT / "task" / "layer_fol_stats" / "set",
    # one of: "reachability", "sensitivity", "both"
    "study_mode": "both",
    "base_rule_bank": {
        "n_layers": 30,
        "predicates_per_layer": 16,
        "rules_per_transition": 24,
        "arity_max": 3,
        "vars_per_rule_max": 4,
        "constants": ("a", "b", "c", "d"),
        "k_in_max": 3,
        "k_out_max": 3,
    },
    # Main metric: exact-distance reachability for these d values.
    "distance_values": np.arange(1, 21),
    "initial_ant_max": 3,
    "n_banks_per_setting": 2,
    "n_trials_per_distance": 80,
    # Optional per-trial CSVs can be large.
    "save_trial_records": False,
    # Separate shortest-path diagnostic protocol.
    "path_length_trials_per_setting": 300,
    "path_length_max_distance": 20,
    # Feasibility / compute caps.
    "max_unify_solutions": 128,
    "max_states_per_depth": 2048,
    "max_branch_expansions_per_state": 512,
    # Sweep one or more parameters. Each entry is independent.
    "sweeps": [
        {
            "name": "rules_per_transition",
            "values": [4, 8, 16, 24, 32, 64],
        },
    ],
    # Reproduction study for long-distance sampling feasibility.
    "sensitivity_repro": {
        "base_rule_bank": {
            "n_layers": 30,
            "predicates_per_layer": 16,
            "rules_per_transition": 32,
            "arity_max": 3,
            "vars_per_rule_max": 4,
            "constants": tuple(f"c{i}" for i in range(64)),
            "k_in_max": 3,
            "k_out_max": 3,
        },
        "distance": 12,
        "seed_values": [0, 1, 2, 3, 4],
        "trials_per_seed": 12,
        "initial_ant_max": 3,
        "sample_max_attempts": 2048,
        "max_unify_solutions": 128,
        "sweeps": [
            {"name": "rules_per_transition", "values": [8, 12, 16, 24, 32, 48]},
            {"name": "k_in_max", "values": [1, 2, 3, 4, 5]},
            {"name": "k_out_max", "values": [1, 2, 3, 4]},
            {"name": "predicates_per_layer", "values": [8, 12, 16, 24, 32]},
            {"name": "constants_count", "values": [4, 16, 64, 256]},
        ],
    },
}

_study_mode_env = os.environ.get("LAYER_FOL_STATS_STUDY_MODE")
if _study_mode_env is not None and str(_study_mode_env).strip():
    CONFIG["study_mode"] = str(_study_mode_env).strip()

_out_dir_env = os.environ.get("LAYER_FOL_STATS_OUT_DIR")
if _out_dir_env is not None and str(_out_dir_env).strip():
    CONFIG["out_dir"] = Path(str(_out_dir_env)).expanduser()


# <codecell>
def _sorted_fol_atoms(atoms) -> tuple[FOLAtom, ...]:
    return tuple(sorted(atoms, key=lambda atom: (atom.predicate, tuple(atom.args))))


def _is_variable(token: str) -> bool:
    return str(token).startswith("x")


def _facts_signature(facts: tuple[FOLAtom, ...]) -> tuple[str, ...]:
    return tuple(atom.text for atom in facts)


def _sample_ground_atom_for_layer(
    *,
    bank: FOLRuleBank,
    layer: int,
    rng: np.random.Generator,
) -> FOLAtom:
    predicates = bank.predicates_for_layer(int(layer))
    predicate = str(predicates[int(rng.integers(0, len(predicates)))])
    arity = int(bank.predicate_arities[predicate])
    constants = bank.constants
    args = tuple(str(constants[int(rng.integers(0, len(constants)))]) for _ in range(arity))
    return FOLAtom(predicate=predicate, args=args)


def _sample_start_facts_for_layer(
    *,
    bank: FOLRuleBank,
    layer: int,
    initial_ant_max: int,
    rng: np.random.Generator,
) -> tuple[FOLAtom, ...]:
    predicates = np.asarray(bank.predicates_for_layer(int(layer)), dtype=object)
    n_facts = int(rng.integers(1, min(int(initial_ant_max), len(predicates)) + 1))
    chosen = [str(tok) for tok in rng.choice(predicates, size=n_facts, replace=False)]
    constants = bank.constants
    facts = {
        FOLAtom(
            predicate=predicate,
            args=tuple(
                str(constants[int(rng.integers(0, len(constants)))])
                for _ in range(int(bank.predicate_arities[predicate]))
            ),
        )
        for predicate in chosen
    }
    return _sorted_fol_atoms(facts)


def _sample_exact_distance_trial(
    *,
    bank: FOLRuleBank,
    distance: int,
    initial_ant_max: int,
    rng: np.random.Generator,
) -> tuple[int, tuple[FOLAtom, ...], FOLAtom]:
    max_start = int(bank.n_layers) - int(distance) - 1
    if max_start < 0:
        raise ValueError(
            f"distance={distance} incompatible with n_layers={bank.n_layers}"
        )
    start_layer = int(rng.integers(0, max_start + 1))
    goal_layer = start_layer + int(distance)
    start_facts = _sample_start_facts_for_layer(
        bank=bank,
        layer=start_layer,
        initial_ant_max=initial_ant_max,
        rng=rng,
    )
    goal_atom = _sample_ground_atom_for_layer(
        bank=bank,
        layer=goal_layer,
        rng=rng,
    )
    return start_layer, start_facts, goal_atom


def _sample_path_length_trial(
    *,
    bank: FOLRuleBank,
    initial_ant_max: int,
    max_distance: int,
    rng: np.random.Generator,
) -> tuple[int, tuple[FOLAtom, ...], int, FOLAtom]:
    max_start = int(bank.n_layers) - 2
    if max_start < 0:
        raise ValueError(f"n_layers={bank.n_layers} must be >= 2")
    start_layer = int(rng.integers(0, max_start + 1))
    furthest_goal = min(int(bank.n_layers) - 1, start_layer + int(max_distance))
    if furthest_goal <= start_layer:
        goal_layer = start_layer + 1
    else:
        goal_layer = int(rng.integers(start_layer + 1, furthest_goal + 1))
    start_facts = _sample_start_facts_for_layer(
        bank=bank,
        layer=start_layer,
        initial_ant_max=initial_ant_max,
        rng=rng,
    )
    goal_atom = _sample_ground_atom_for_layer(
        bank=bank,
        layer=goal_layer,
        rng=rng,
    )
    return start_layer, start_facts, goal_layer, goal_atom


def _unify_template_atom_with_ground(
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


def _find_lhs_substitutions_for_facts(
    *,
    lhs: tuple[FOLAtom, ...],
    facts: tuple[FOLAtom, ...],
    max_solutions: int,
) -> list[dict[str, str]]:
    solutions: list[dict[str, str]] = []

    def _search(idx: int, subst: dict[str, str]) -> None:
        if len(solutions) >= int(max_solutions):
            return
        if idx >= len(lhs):
            solutions.append(dict(subst))
            return

        templ = lhs[idx]
        for fact in facts:
            maybe = _unify_template_atom_with_ground(templ, fact, subst)
            if maybe is None:
                continue
            _search(idx + 1, maybe)
            if len(solutions) >= int(max_solutions):
                return

    _search(0, {})
    return solutions


def _subst_binds_rhs_variables(*, rule: FOLLayerRule, subst: dict[str, str]) -> bool:
    rhs_vars = {
        term
        for atom in rule.rhs
        for term in atom.args
        if _is_variable(term)
    }
    return rhs_vars.issubset(set(subst))


@dataclass(frozen=True)
class TransitionExpandResult:
    next_states: tuple[tuple[FOLAtom, ...], ...]
    truncated: bool


def _enumerate_next_states(
    *,
    bank: FOLRuleBank,
    src_layer: int,
    facts: tuple[FOLAtom, ...],
    max_unify_solutions: int,
    max_branch_expansions_per_state: int,
) -> TransitionExpandResult:
    dedup: dict[tuple[str, ...], tuple[FOLAtom, ...]] = {}
    truncated = False
    cap = int(max_branch_expansions_per_state)
    for rule in bank.transition_rules(int(src_layer)):
        subs = _find_lhs_substitutions_for_facts(
            lhs=rule.lhs,
            facts=facts,
            max_solutions=int(max_unify_solutions),
        )
        for subst in subs:
            if not _subst_binds_rhs_variables(rule=rule, subst=subst):
                continue
            rhs = _sorted_fol_atoms(rule.instantiate(subst).rhs)
            sig = _facts_signature(rhs)
            dedup[sig] = rhs
            if len(dedup) >= cap:
                truncated = True
                break
        if truncated:
            break

    return TransitionExpandResult(
        next_states=tuple(dedup[key] for key in sorted(dedup)),
        truncated=truncated,
    )


@dataclass(frozen=True)
class ReachabilityResult:
    reachable: bool
    truncated: bool
    n_states_by_depth: tuple[int, ...]


def exists_path_exact_distance(
    *,
    bank: FOLRuleBank,
    start_layer: int,
    start_facts: tuple[FOLAtom, ...],
    goal_atom: FOLAtom,
    distance: int,
    max_unify_solutions: int,
    max_states_per_depth: int,
    max_branch_expansions_per_state: int,
) -> ReachabilityResult:
    distance = int(distance)
    start_layer = int(start_layer)
    if distance < 0:
        raise ValueError(f"distance must be >= 0, got {distance}")

    frontier: dict[tuple[str, ...], tuple[FOLAtom, ...]] = {
        _facts_signature(_sorted_fol_atoms(start_facts)): _sorted_fol_atoms(start_facts)
    }
    n_states_by_depth = [len(frontier)]
    truncated = False

    for step in range(distance):
        src_layer = start_layer + step
        next_frontier: dict[tuple[str, ...], tuple[FOLAtom, ...]] = {}

        for facts in frontier.values():
            expanded = _enumerate_next_states(
                bank=bank,
                src_layer=src_layer,
                facts=facts,
                max_unify_solutions=int(max_unify_solutions),
                max_branch_expansions_per_state=int(max_branch_expansions_per_state),
            )
            if expanded.truncated:
                truncated = True
            for next_state in expanded.next_states:
                next_frontier[_facts_signature(next_state)] = next_state

        if len(next_frontier) > int(max_states_per_depth):
            truncated = True
            keep_keys = sorted(next_frontier)[: int(max_states_per_depth)]
            next_frontier = {key: next_frontier[key] for key in keep_keys}

        frontier = next_frontier
        n_states_by_depth.append(len(frontier))
        if not frontier:
            break

    reachable = any(goal_atom in facts for facts in frontier.values())
    return ReachabilityResult(
        reachable=bool(reachable),
        truncated=bool(truncated),
        n_states_by_depth=tuple(int(x) for x in n_states_by_depth),
    )


def shortest_path_length(
    *,
    bank: FOLRuleBank,
    start_layer: int,
    start_facts: tuple[FOLAtom, ...],
    goal_layer: int,
    goal_atom: FOLAtom,
    max_unify_solutions: int,
    max_states_per_depth: int,
    max_branch_expansions_per_state: int,
) -> tuple[int | None, bool]:
    delta = int(goal_layer) - int(start_layer)
    if delta < 0:
        return None, False

    result = exists_path_exact_distance(
        bank=bank,
        start_layer=int(start_layer),
        start_facts=start_facts,
        goal_atom=goal_atom,
        distance=int(delta),
        max_unify_solutions=int(max_unify_solutions),
        max_states_per_depth=int(max_states_per_depth),
        max_branch_expansions_per_state=int(max_branch_expansions_per_state),
    )
    if result.reachable:
        return int(delta), bool(result.truncated)
    return None, bool(result.truncated)


def _wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials <= 0:
        return 0.0, 0.0
    n = float(trials)
    p = float(successes) / n
    z2 = float(z) ** 2
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = (float(z) / denom) * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return lo, hi


def _make_bank_config(base: dict, override_name: str, override_value: int) -> dict:
    out = dict(base)
    out[str(override_name)] = int(override_value)
    return out


def _build_constants_by_count(count: int) -> tuple[str, ...]:
    count = int(count)
    if count < 1:
        raise ValueError(f"constants_count must be >= 1, got {count}")
    return tuple(f"c{idx}" for idx in range(count))


def _make_sensitivity_bank_config(base: dict, override_name: str, override_value: int) -> dict:
    out = dict(base)
    name = str(override_name)
    value = int(override_value)
    if name == "constants_count":
        out["constants"] = _build_constants_by_count(value)
    else:
        out[name] = value
    return out


def _build_bank(bank_cfg: dict, seed: int) -> FOLRuleBank:
    return build_random_fol_rule_bank(
        n_layers=int(bank_cfg["n_layers"]),
        predicates_per_layer=int(bank_cfg["predicates_per_layer"]),
        rules_per_transition=int(bank_cfg["rules_per_transition"]),
        arity_max=int(bank_cfg["arity_max"]),
        vars_per_rule_max=int(bank_cfg["vars_per_rule_max"]),
        constants=tuple(str(tok) for tok in bank_cfg["constants"]),
        k_in_max=int(bank_cfg["k_in_max"]),
        k_out_max=int(bank_cfg["k_out_max"]),
        rng=np.random.default_rng(int(seed)),
    )


def run_sampling_sensitivity_study(
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sweep_rows: list[dict] = []
    seed_rows: list[dict] = []

    repro_cfg = dict(cfg["sensitivity_repro"])
    base_bank_cfg = dict(repro_cfg["base_rule_bank"])
    distance = int(repro_cfg["distance"])
    seed_values = [int(seed) for seed in repro_cfg["seed_values"]]
    trials_per_seed = int(repro_cfg["trials_per_seed"])
    initial_ant_max = int(repro_cfg["initial_ant_max"])
    sample_max_attempts = int(repro_cfg["sample_max_attempts"])
    max_unify_solutions = int(repro_cfg["max_unify_solutions"])

    for sweep in repro_cfg["sweeps"]:
        sweep_name = str(sweep["name"])
        sweep_values = [int(v) for v in sweep["values"]]

        for sweep_value in sweep_values:
            bank_cfg = _make_sensitivity_bank_config(
                base=base_bank_cfg,
                override_name=sweep_name,
                override_value=sweep_value,
            )
            seed_rates: list[float] = []
            total_trials = 0
            total_success = 0

            for seed in seed_values:
                rng = np.random.default_rng(int(seed))
                bank = _build_bank(bank_cfg, seed=int(seed))
                n_success = 0
                for _ in range(trials_per_seed):
                    try:
                        sample_fol_problem(
                            bank=bank,
                            distance=distance,
                            initial_ant_max=initial_ant_max,
                            rng=rng,
                            max_attempts=sample_max_attempts,
                            max_unify_solutions=max_unify_solutions,
                        )
                        n_success += 1
                    except RuntimeError:
                        pass

                n_trials = int(trials_per_seed)
                feasible_rate = float(n_success / n_trials) if n_trials else 0.0
                seed_rates.append(feasible_rate)
                total_trials += n_trials
                total_success += int(n_success)
                seed_rows.append(
                    {
                        "sweep_name": sweep_name,
                        "sweep_value": int(sweep_value),
                        "seed": int(seed),
                        "distance": int(distance),
                        "n_trials": int(n_trials),
                        "n_success": int(n_success),
                        "feasible_rate": float(feasible_rate),
                    }
                )

            seed_arr = np.asarray(seed_rates, dtype=np.float64)
            sweep_rows.append(
                {
                    "sweep_name": sweep_name,
                    "sweep_value": int(sweep_value),
                    "distance": int(distance),
                    "n_seeds": int(len(seed_values)),
                    "trials_per_seed": int(trials_per_seed),
                    "total_trials": int(total_trials),
                    "total_success": int(total_success),
                    "pooled_feasible_rate": float(total_success / total_trials)
                    if total_trials
                    else 0.0,
                    "mean_feasible_rate": float(np.mean(seed_arr)) if seed_rates else 0.0,
                    "std_feasible_rate": float(np.std(seed_arr)) if seed_rates else 0.0,
                    "min_seed_rate": float(np.min(seed_arr)) if seed_rates else 0.0,
                    "max_seed_rate": float(np.max(seed_arr)) if seed_rates else 0.0,
                    "seed_rates": json.dumps([float(x) for x in seed_rates]),
                }
            )

    summary_df = pd.DataFrame(sweep_rows)
    seed_df = pd.DataFrame(seed_rows)
    return summary_df, seed_df


def run_reachability_study(
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reach_rows: list[dict] = []
    reach_trial_rows: list[dict] = []
    path_summary_rows: list[dict] = []
    path_hist_rows: list[dict] = []
    path_trial_rows: list[dict] = []

    base_seed = int(cfg["seed"])
    base_bank_cfg = dict(cfg["base_rule_bank"])

    for sweep_idx, sweep in enumerate(cfg["sweeps"]):
        sweep_name = str(sweep["name"])
        sweep_values = [int(v) for v in sweep["values"]]

        for value_idx, sweep_value in enumerate(sweep_values):
            bank_cfg = _make_bank_config(base_bank_cfg, sweep_name, sweep_value)

            # exact-distance reachability
            for distance in [int(d) for d in cfg["distance_values"]]:
                n_success = 0
                n_trials = 0
                n_truncated = 0
                states_per_trial: list[float] = []

                for bank_idx in range(int(cfg["n_banks_per_setting"])):
                    bank_seed = (
                        base_seed
                        + 10_000_019 * sweep_idx
                        + 1_000_003 * value_idx
                        + 100_003 * bank_idx
                        + 9_973 * distance
                    )
                    trial_seed = bank_seed + 17
                    bank = _build_bank(bank_cfg, bank_seed)
                    rng = np.random.default_rng(trial_seed)

                    for trial_idx in range(int(cfg["n_trials_per_distance"])):
                        start_layer, start_facts, goal_atom = _sample_exact_distance_trial(
                            bank=bank,
                            distance=int(distance),
                            initial_ant_max=int(cfg["initial_ant_max"]),
                            rng=rng,
                        )
                        result = exists_path_exact_distance(
                            bank=bank,
                            start_layer=int(start_layer),
                            start_facts=start_facts,
                            goal_atom=goal_atom,
                            distance=int(distance),
                            max_unify_solutions=int(cfg["max_unify_solutions"]),
                            max_states_per_depth=int(cfg["max_states_per_depth"]),
                            max_branch_expansions_per_state=int(
                                cfg["max_branch_expansions_per_state"]
                            ),
                        )

                        n_trials += 1
                        n_success += int(result.reachable)
                        n_truncated += int(result.truncated)
                        states_per_trial.append(float(np.mean(result.n_states_by_depth)))

                        if cfg["save_trial_records"]:
                            reach_trial_rows.append(
                                {
                                    "sweep_name": sweep_name,
                                    "sweep_value": int(sweep_value),
                                    "distance": int(distance),
                                    "bank_idx": int(bank_idx),
                                    "trial_idx": int(trial_idx),
                                    "start_layer": int(start_layer),
                                    "goal_atom": goal_atom.text,
                                    "reachable": int(result.reachable),
                                    "truncated": int(result.truncated),
                                    "avg_states_per_depth": float(
                                        np.mean(result.n_states_by_depth)
                                    ),
                                }
                            )

                ci_low, ci_high = _wilson_interval(n_success, n_trials)
                reach_rows.append(
                    {
                        "sweep_name": sweep_name,
                        "sweep_value": int(sweep_value),
                        "distance": int(distance),
                        "n_trials": int(n_trials),
                        "n_reachable": int(n_success),
                        "p_reachable_exact_d": float(n_success / n_trials) if n_trials else 0.0,
                        "ci_low": float(ci_low),
                        "ci_high": float(ci_high),
                        "truncation_rate": float(n_truncated / n_trials) if n_trials else 0.0,
                        "avg_states_per_depth": float(np.mean(states_per_trial))
                        if states_per_trial
                        else 0.0,
                        "bank_n_layers": int(bank_cfg["n_layers"]),
                    }
                )

            # shortest-path diagnostic
            lengths: list[int] = []
            n_trials = 0
            n_reachable = 0
            n_truncated = 0
            for bank_idx in range(int(cfg["n_banks_per_setting"])):
                bank_seed = (
                    base_seed
                    + 20_000_033 * sweep_idx
                    + 2_000_003 * value_idx
                    + 200_003 * bank_idx
                )
                trial_seed = bank_seed + 29
                bank = _build_bank(bank_cfg, bank_seed)
                rng = np.random.default_rng(trial_seed)

                for trial_idx in range(int(cfg["path_length_trials_per_setting"])):
                    start_layer, start_facts, goal_layer, goal_atom = _sample_path_length_trial(
                        bank=bank,
                        initial_ant_max=int(cfg["initial_ant_max"]),
                        max_distance=int(cfg["path_length_max_distance"]),
                        rng=rng,
                    )
                    length, truncated = shortest_path_length(
                        bank=bank,
                        start_layer=int(start_layer),
                        start_facts=start_facts,
                        goal_layer=int(goal_layer),
                        goal_atom=goal_atom,
                        max_unify_solutions=int(cfg["max_unify_solutions"]),
                        max_states_per_depth=int(cfg["max_states_per_depth"]),
                        max_branch_expansions_per_state=int(
                            cfg["max_branch_expansions_per_state"]
                        ),
                    )
                    n_trials += 1
                    n_reachable += int(length is not None)
                    n_truncated += int(truncated)
                    if length is not None:
                        lengths.append(int(length))

                    if cfg["save_trial_records"]:
                        path_trial_rows.append(
                            {
                                "sweep_name": sweep_name,
                                "sweep_value": int(sweep_value),
                                "bank_idx": int(bank_idx),
                                "trial_idx": int(trial_idx),
                                "start_layer": int(start_layer),
                                "goal_layer": int(goal_layer),
                                "goal_atom": goal_atom.text,
                                "shortest_length": None if length is None else int(length),
                                "reachable": int(length is not None),
                                "truncated": int(truncated),
                            }
                        )

            unreachable_rate = float(1.0 - n_reachable / n_trials) if n_trials else 0.0
            path_summary_rows.append(
                {
                    "sweep_name": sweep_name,
                    "sweep_value": int(sweep_value),
                    "n_trials": int(n_trials),
                    "n_reachable": int(n_reachable),
                    "reachable_rate": float(n_reachable / n_trials) if n_trials else 0.0,
                    "truncation_rate": float(n_truncated / n_trials) if n_trials else 0.0,
                    "unreachable_rate": float(unreachable_rate),
                    "mean_shortest_length_reachable_only": float(np.mean(lengths))
                    if lengths
                    else np.nan,
                    "median_shortest_length_reachable_only": float(np.median(lengths))
                    if lengths
                    else np.nan,
                }
            )

            if lengths:
                for length in sorted(set(lengths)):
                    count = int(sum(1 for x in lengths if x == length))
                    path_hist_rows.append(
                        {
                            "sweep_name": sweep_name,
                            "sweep_value": int(sweep_value),
                            "length_bin": int(length),
                            "count": int(count),
                            "fraction_over_reachable": float(count / len(lengths)),
                            "unreachable_rate": float(unreachable_rate),
                            "n_trials": int(n_trials),
                        }
                    )

    return (
        pd.DataFrame(reach_rows),
        pd.DataFrame(reach_trial_rows),
        pd.DataFrame(path_summary_rows),
        pd.DataFrame(path_hist_rows),
        pd.DataFrame(path_trial_rows),
    )


# <codecell>
def plot_reachability(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return

    sns.set_theme(style="whitegrid")
    out_dir.mkdir(parents=True, exist_ok=True)

    needed = {"distance", "p_reachable_exact_d", "sweep_name", "sweep_value", "ci_low", "ci_high"}
    if not needed.issubset(set(df.columns)):
        return

    for sweep_name in sorted(df["sweep_name"].dropna().unique()):
        part = df[df["sweep_name"] == sweep_name].copy()
        part = part.sort_values(["distance", "sweep_value"])
        heat = part.pivot(index="distance", columns="sweep_value", values="p_reachable_exact_d")
        if heat.empty:
            continue

        plt.figure(figsize=(8, 4))
        sns.heatmap(
            heat,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            cbar_kws={"label": "P(reachable at exact d)"},
        )
        plt.title(f"Exact-distance reachability vs {sweep_name}")
        plt.xlabel(sweep_name)
        plt.ylabel("distance")
        plt.tight_layout()
        plt.savefig(out_dir / f"reachability_heatmap_{sweep_name}.png", dpi=180)
        plt.close()

        plt.figure(figsize=(8, 4))
        sns.lineplot(
            data=part,
            x="distance",
            y="p_reachable_exact_d",
            hue="sweep_value",
            marker="o",
            palette="viridis",
        )
        for _, row in part.iterrows():
            plt.vlines(
                x=float(row["distance"]),
                ymin=float(row["ci_low"]),
                ymax=float(row["ci_high"]),
                color="gray",
                alpha=0.35,
                linewidth=1.0,
            )
        plt.ylim(0.0, 1.0)
        plt.title(f"Exact-distance reachability curves ({sweep_name})")
        plt.ylabel("P(reachable)")
        plt.tight_layout()
        plt.savefig(out_dir / f"reachability_lines_{sweep_name}.png", dpi=180)
        plt.close()


def plot_path_length_histograms(
    *,
    path_hist_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    if path_hist_df.empty:
        return

    sns.set_theme(style="whitegrid")
    out_dir.mkdir(parents=True, exist_ok=True)

    for sweep_name in sorted(path_hist_df["sweep_name"].dropna().unique()):
        part = path_hist_df[path_hist_df["sweep_name"] == sweep_name].copy()
        for sweep_value in sorted(part["sweep_value"].dropna().unique()):
            sub = part[part["sweep_value"] == sweep_value].copy()
            if sub.empty:
                continue

            sub = sub.sort_values("length_bin")
            unreachable_rate = float(sub["unreachable_rate"].iloc[0])
            plt.figure(figsize=(8, 4))
            plt.bar(
                sub["length_bin"].astype(int).to_numpy(),
                sub["count"].astype(int).to_numpy(),
                color="#2E6F95",
                edgecolor="white",
            )
            plt.title(
                f"Shortest path lengths: {sweep_name}={int(sweep_value)} "
                f"(unreachable={unreachable_rate:.2%})"
            )
            plt.xlabel("shortest path length")
            plt.ylabel("count")
            plt.tight_layout()
            plt.savefig(
                out_dir / f"path_length_hist_{sweep_name}_{int(sweep_value):03d}.png",
                dpi=180,
            )
            plt.close()


# <codecell>
def _jsonable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def save_reachability_outputs(
    *,
    cfg: dict,
    reach_df: pd.DataFrame,
    reach_trial_df: pd.DataFrame,
    path_summary_df: pd.DataFrame,
    path_hist_df: pd.DataFrame,
    path_trial_df: pd.DataFrame,
) -> Path:
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    reach_summary_path = out_dir / "reachability_summary.csv"
    path_summary_path = out_dir / "path_length_summary.csv"
    path_hist_path = out_dir / "path_length_histogram.csv"
    reach_df.to_csv(reach_summary_path, index=False)
    path_summary_df.to_csv(path_summary_path, index=False)
    path_hist_df.to_csv(path_hist_path, index=False)

    if cfg["save_trial_records"]:
        reach_trial_df.to_csv(out_dir / "reachability_trials.csv", index=False)
        path_trial_df.to_csv(out_dir / "path_length_trials.csv", index=False)

    plot_reachability(reach_df, out_dir=out_dir)
    plot_path_length_histograms(path_hist_df=path_hist_df, out_dir=out_dir)

    return out_dir


def plot_sampling_sensitivity(
    *,
    summary_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    if summary_df.empty:
        return

    sns.set_theme(style="whitegrid")
    out_dir.mkdir(parents=True, exist_ok=True)

    for sweep_name in sorted(summary_df["sweep_name"].dropna().unique()):
        part = summary_df[summary_df["sweep_name"] == sweep_name].copy()
        if part.empty:
            continue
        part = part.sort_values("sweep_value")

        x = part["sweep_value"].astype(int).to_numpy()
        y = part["mean_feasible_rate"].astype(float).to_numpy()
        y_min = part["min_seed_rate"].astype(float).to_numpy()
        y_max = part["max_seed_rate"].astype(float).to_numpy()
        yerr_low = y - y_min
        yerr_high = y_max - y

        plt.figure(figsize=(7, 4))
        plt.errorbar(
            x=x,
            y=y,
            yerr=np.vstack([yerr_low, yerr_high]),
            fmt="-o",
            linewidth=1.8,
            capsize=4,
            color="#2E6F95",
        )
        plt.ylim(0.0, 1.0)
        plt.xlabel(sweep_name)
        plt.ylabel("mean feasible sampling rate")
        plt.title(f"Sampling sensitivity at distance={int(part['distance'].iloc[0])}")
        plt.tight_layout()
        plt.savefig(out_dir / f"sampling_sensitivity_{sweep_name}.png", dpi=180)
        plt.close()


def save_sampling_sensitivity_outputs(
    *,
    cfg: dict,
    summary_df: pd.DataFrame,
    seed_df: pd.DataFrame,
) -> Path:
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "sampling_sensitivity_summary.csv"
    seed_path = out_dir / "sampling_sensitivity_seed_rates.csv"
    cfg_path = out_dir / "sampling_sensitivity_config.json"
    summary_df.to_csv(summary_path, index=False)
    seed_df.to_csv(seed_path, index=False)
    cfg_path.write_text(
        json.dumps(_jsonable(cfg["sensitivity_repro"]), indent=2)
    )
    plot_sampling_sensitivity(summary_df=summary_df, out_dir=out_dir)
    return out_dir


# <codecell>
def print_reachability_console_summary(
    *,
    reach_df: pd.DataFrame,
    path_df: pd.DataFrame,
) -> None:
    print("\n=== Exact-distance reachability summary ===")
    if reach_df.empty:
        print("No rows.")
    else:
        cols = [
            "sweep_name",
            "sweep_value",
            "distance",
            "n_trials",
            "p_reachable_exact_d",
            "ci_low",
            "ci_high",
            "truncation_rate",
        ]
        print(reach_df[cols].sort_values(["sweep_name", "sweep_value", "distance"]).to_string(index=False))

    print("\n=== Path-length diagnostic summary ===")
    if path_df.empty:
        print("No rows.")
    else:
        summary_cols = [
            "sweep_name",
            "sweep_value",
            "n_trials",
            "reachable_rate",
            "mean_shortest_length_reachable_only",
            "median_shortest_length_reachable_only",
            "truncation_rate",
        ]
        summary = path_df[path_df.columns.intersection(summary_cols)].copy()
        if not summary.empty:
            print(
                summary.sort_values(["sweep_name", "sweep_value"]).to_string(index=False)
            )


def print_sampling_sensitivity_console_summary(
    *,
    summary_df: pd.DataFrame,
) -> None:
    print("\n=== Sampling sensitivity summary ===")
    if summary_df.empty:
        print("No rows.")
        return

    cols = [
        "sweep_name",
        "sweep_value",
        "distance",
        "n_seeds",
        "trials_per_seed",
        "mean_feasible_rate",
        "std_feasible_rate",
        "min_seed_rate",
        "max_seed_rate",
    ]
    present_cols = summary_df.columns.intersection(cols)
    print(
        summary_df[present_cols]
        .sort_values(["sweep_name", "sweep_value"])
        .to_string(index=False)
    )


def _resolve_study_mode(mode_raw: str) -> tuple[bool, bool]:
    mode = str(mode_raw).strip().lower()
    if mode == "reachability":
        return True, False
    if mode == "sensitivity":
        return False, True
    if mode == "both":
        return True, True
    raise ValueError(f"Unsupported study_mode={mode_raw!r}; expected reachability|sensitivity|both")


# <codecell>
# Execute study at top level for interactive workflows.
RUN_REACHABILITY, RUN_SENSITIVITY = _resolve_study_mode(CONFIG.get("study_mode", "both"))
OUT_DIR = Path(CONFIG["out_dir"])

REACH_DF = pd.DataFrame()
REACH_TRIAL_DF = pd.DataFrame()
PATH_SUMMARY_DF = pd.DataFrame()
PATH_HIST_DF = pd.DataFrame()
PATH_TRIAL_DF = pd.DataFrame()
SENSITIVITY_SUMMARY_DF = pd.DataFrame()
SENSITIVITY_SEED_DF = pd.DataFrame()

if RUN_REACHABILITY:
    REACH_DF, REACH_TRIAL_DF, PATH_SUMMARY_DF, PATH_HIST_DF, PATH_TRIAL_DF = run_reachability_study(
        CONFIG
    )
    OUT_DIR = save_reachability_outputs(
        cfg=CONFIG,
        reach_df=REACH_DF,
        reach_trial_df=REACH_TRIAL_DF,
        path_summary_df=PATH_SUMMARY_DF,
        path_hist_df=PATH_HIST_DF,
        path_trial_df=PATH_TRIAL_DF,
    )
    print_reachability_console_summary(reach_df=REACH_DF, path_df=PATH_SUMMARY_DF)

if RUN_SENSITIVITY:
    SENSITIVITY_SUMMARY_DF, SENSITIVITY_SEED_DF = run_sampling_sensitivity_study(CONFIG)
    OUT_DIR = save_sampling_sensitivity_outputs(
        cfg=CONFIG,
        summary_df=SENSITIVITY_SUMMARY_DF,
        seed_df=SENSITIVITY_SEED_DF,
    )
    print_sampling_sensitivity_console_summary(summary_df=SENSITIVITY_SUMMARY_DF)

print(f"\nSaved outputs to: {OUT_DIR}")

# %%
