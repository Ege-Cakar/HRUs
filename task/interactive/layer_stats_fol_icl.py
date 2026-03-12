# <codecell>
"""Interactive stats for schema-based demo reasoning in depth3_icl_transfer FOL.

Estimates the probability that a random inferred demo schema is "bad", where:
1) demonstration slots are sampled as in task generation,
2) sampled slots are deduplicated to unique schemas,
3) one schema is chosen uniformly from that unique set,
4) the schema is grounded using substitutions induced by current facts,
5) "bad" means no such grounding preserves downstream goal reachability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm is unavailable
    tqdm = None

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from task.layer_fol import (
    _collect_applicable_demo_schemas,
    _find_lhs_substitutions_for_facts,
    _sample_demo_schemas,
    _subst_binds_rhs_variables,
)
from task.layer_gen.util.fol_rule_bank import (
    FOLAtom,
    FOLLayerRule,
    FOLRuleBank,
    build_depth3_icl_split_bundle,
    sample_fol_problem,
)


# <codecell>
# Edit settings here (no argparse by design).
CONFIG = {
    "seed": 0,
    "out_dir": ROOT / "task" / "interactive" / "set" / "layer_stats_fol_icl",
    "base_split": {
        "predicates_per_layer": 64,
        "rules_01_train": 64,
        "rules_01_eval": 64,
        "rules_12_shared": 64,
        "arity_max": 3,
        "vars_per_rule_max": 6,
        "k_in_max": 3,
        "k_out_max": 5,
        "constants_count": 64,
    },
    "roles": ("train", "eval"),
    "exact_n_demos_values": [1, 2, 4, 8, 16, 32],
    "n_bundles_per_setting": 2,
    "n_prompts_per_bundle_per_role": 200,
    "initial_ant_max": 5,
    "sample_max_attempts": 4096,
    "max_unify_solutions": 128,
    "ci_method": "bootstrap",
    "n_bootstrap": 1000,
    "bootstrap_seed_offset": 100_003,
    "save_trial_rows": False,
    "show_progress": True,
    "sweeps": [
        {"name": "predicates_per_layer", "values": [32, 64, 128, 256]},
        {"name": "rules_01_train", "values": [32, 64, 128, 256]},
        {"name": "rules_01_eval", "values": [32, 64, 128, 256]},
        {"name": "rules_12_shared", "values": [32, 64, 128, 256]},
        {"name": "constants_count", "values": [32, 64, 128, 512]},
    ],
}

_out_dir_env = os.environ.get("LAYER_STATS_FOL_ICL_OUT_DIR")
if _out_dir_env is not None and str(_out_dir_env).strip():
    CONFIG["out_dir"] = Path(str(_out_dir_env).strip()).expanduser()

if os.environ.get("LAYER_STATS_FOL_ICL_SMOKE", "").strip():
    CONFIG["n_bundles_per_setting"] = 1
    CONFIG["n_prompts_per_bundle_per_role"] = 24
    CONFIG["exact_n_demos_values"] = [1, 4, 8]
    CONFIG["sweeps"] = [
        {"name": "predicates_per_layer", "values": [8, 16]},
        {"name": "rules_01_eval", "values": [8, 32]},
    ]


_LAYERED_PRED_RE = re.compile(r"r(\d+)_(\d+)$")


# <codecell>
@dataclass
class DrawAccumulator:
    n_prompts: int = 0
    n_prompts_with_schema: int = 0
    n_no_schema: int = 0
    schema_count_sum: int = 0
    unique_schema_count_sum: int = 0
    context_constants_count_sum: int = 0
    n_schema_no_valid_grounding: int = 0
    n_bad: int = 0
    n_schema_choices: int = 0
    p_bad_prompt_values: list[float] = field(default_factory=list)


def _layer_from_predicate(predicate: str) -> int:
    match = _LAYERED_PRED_RE.fullmatch(str(predicate))
    if match is None:
        raise ValueError(f"Unsupported layered predicate name: {predicate!r}")
    return int(match.group(1))


def _facts_key(facts: tuple[FOLAtom, ...]) -> tuple[str, ...]:
    return tuple(sorted(atom.text for atom in facts))


def _constants_from_facts(facts: tuple[FOLAtom, ...]) -> tuple[str, ...]:
    constants = sorted({str(term) for atom in facts for term in atom.args})
    return tuple(constants)


def _unique_schemas_in_order(schemas: list[FOLLayerRule]) -> list[FOLLayerRule]:
    out: list[FOLLayerRule] = []
    seen: set[str] = set()
    for schema in schemas:
        key = str(schema.statement_text)
        if key in seen:
            continue
        seen.add(key)
        out.append(schema)
    return out


def _reachable_goal_exact_steps(
    *,
    rule_bank: FOLRuleBank,
    layer: int,
    facts: tuple[FOLAtom, ...],
    goal: FOLAtom,
    steps_remaining: int,
    max_unify_solutions: int,
    memo: dict[tuple[int, int, tuple[str, ...]], bool],
) -> bool:
    key = (int(layer), int(steps_remaining), _facts_key(facts))
    cached = memo.get(key)
    if cached is not None:
        return bool(cached)

    if int(steps_remaining) == 0:
        out = bool(goal in set(facts))
        memo[key] = out
        return out

    if int(layer) >= int(rule_bank.n_layers) - 1:
        memo[key] = False
        return False

    facts_tuple = tuple(facts)
    for rule in rule_bank.transition_rules(int(layer)):
        substitutions = _find_lhs_substitutions_for_facts(
            lhs=rule.lhs,
            facts=facts_tuple,
            max_solutions=int(max_unify_solutions),
        )
        if not substitutions:
            continue

        for subst in substitutions:
            if not _subst_binds_rhs_variables(rule=rule, subst=subst):
                continue
            next_rule = rule.instantiate(subst)
            if _reachable_goal_exact_steps(
                rule_bank=rule_bank,
                layer=int(layer) + 1,
                facts=tuple(next_rule.rhs),
                goal=goal,
                steps_remaining=int(steps_remaining) - 1,
                max_unify_solutions=int(max_unify_solutions),
                memo=memo,
            ):
                memo[key] = True
                return True

    memo[key] = False
    return False


def _predicted_rule_reaches_goal(
    *,
    rule_bank: FOLRuleBank,
    matched_rule: FOLLayerRule,
    goal: FOLAtom,
    goal_layer: int,
    max_unify_solutions: int,
) -> bool:
    dst_layer = int(matched_rule.dst_layer)
    remaining = int(goal_layer) - dst_layer
    if remaining < 0:
        return False
    memo: dict[tuple[int, int, tuple[str, ...]], bool] = {}
    return _reachable_goal_exact_steps(
        rule_bank=rule_bank,
        layer=dst_layer,
        facts=tuple(matched_rule.rhs),
        goal=goal,
        steps_remaining=remaining,
        max_unify_solutions=int(max_unify_solutions),
        memo=memo,
    )


def _schema_reaches_goal_via_context_groundings(
    *,
    rule_bank: FOLRuleBank,
    schema: FOLLayerRule,
    ants: tuple[FOLAtom, ...],
    goal: FOLAtom,
    goal_layer: int,
    max_unify_solutions: int,
) -> tuple[bool, bool]:
    substitutions = _find_lhs_substitutions_for_facts(
        lhs=schema.lhs,
        facts=tuple(ants),
        max_solutions=int(max_unify_solutions),
    )
    valid_subs = [
        subst
        for subst in substitutions
        if _subst_binds_rhs_variables(rule=schema, subst=subst)
    ]
    if not valid_subs:
        return False, False

    for subst in valid_subs:
        grounded_rule = schema.instantiate(subst)
        if _predicted_rule_reaches_goal(
            rule_bank=rule_bank,
            matched_rule=grounded_rule,
            goal=goal,
            goal_layer=int(goal_layer),
            max_unify_solutions=int(max_unify_solutions),
        ):
            return True, True
    return False, True


def _bootstrap_mean_ci(
    values: list[float],
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 1:
        val = float(arr[0])
        return val, val
    n_bootstrap = int(n_bootstrap)
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}")
    idx = rng.integers(0, arr.size, size=(n_bootstrap, arr.size))
    means = arr[idx].mean(axis=1)
    lo_q = float(alpha / 2.0)
    hi_q = float(1.0 - alpha / 2.0)
    lo, hi = np.quantile(means, [lo_q, hi_q])
    return float(lo), float(hi)


def _constants_from_count(count: int) -> tuple[str, ...]:
    count = int(count)
    if count < 1:
        raise ValueError(f"constants_count must be >= 1, got {count}")
    return tuple(f"c{idx}" for idx in range(count))


def _apply_sweep_to_split_config(base: dict[str, Any], sweep_name: str, sweep_value: int) -> dict[str, Any]:
    out = dict(base)
    out[str(sweep_name)] = int(sweep_value)
    return out


def _build_split_bundle_from_cfg(split_cfg: dict[str, Any], seed: int):
    return build_depth3_icl_split_bundle(
        predicates_per_layer=int(split_cfg["predicates_per_layer"]),
        rules_01_train=int(split_cfg["rules_01_train"]),
        rules_01_eval=int(split_cfg["rules_01_eval"]),
        rules_12_shared=int(split_cfg["rules_12_shared"]),
        arity_max=int(split_cfg["arity_max"]),
        vars_per_rule_max=int(split_cfg["vars_per_rule_max"]),
        k_in_max=int(split_cfg["k_in_max"]),
        k_out_max=int(split_cfg["k_out_max"]),
        constants=_constants_from_count(int(split_cfg["constants_count"])),
        rng=np.random.default_rng(int(seed)),
    )


def _sample_prompt_state(
    *,
    bank: FOLRuleBank,
    role: str,
    initial_ant_max: int,
    sample_max_attempts: int,
    max_unify_solutions: int,
    rng: np.random.Generator,
) -> tuple[int, tuple[FOLAtom, ...], FOLAtom, int]:
    sampled = sample_fol_problem(
        bank=bank,
        distance=2,
        initial_ant_max=int(initial_ant_max),
        rng=rng,
        max_attempts=int(sample_max_attempts),
        max_unify_solutions=int(max_unify_solutions),
    )
    if str(role) == "eval":
        step_idx = 0
    else:
        step_idx = int(rng.integers(0, len(sampled.step_rules)))
    src_layer = int(sampled.step_layers[step_idx])
    ants = tuple(sampled.step_ants[step_idx])
    goal_atom = sampled.goal_atom
    goal_layer = _layer_from_predicate(goal_atom.predicate)
    return src_layer, ants, goal_atom, goal_layer


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def run_study(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed = int(cfg["seed"])
    roles = tuple(str(role) for role in cfg["roles"])
    exact_n_demos_values = [int(v) for v in cfg["exact_n_demos_values"]]
    n_bundles = int(cfg["n_bundles_per_setting"])
    n_prompts = int(cfg["n_prompts_per_bundle_per_role"])
    initial_ant_max = int(cfg["initial_ant_max"])
    sample_max_attempts = int(cfg["sample_max_attempts"])
    max_unify_solutions = int(cfg["max_unify_solutions"])
    ci_method = str(cfg.get("ci_method", "bootstrap")).strip().lower()
    n_bootstrap = int(cfg.get("n_bootstrap", 1000))
    bootstrap_seed_offset = int(cfg.get("bootstrap_seed_offset", 100_003))
    save_trial_rows = bool(cfg["save_trial_rows"])
    show_progress = bool(cfg.get("show_progress", True)) and tqdm is not None
    base_split = dict(cfg["base_split"])
    if ci_method != "bootstrap":
        raise ValueError(
            f"Unsupported ci_method={ci_method!r}; expected 'bootstrap'."
        )
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}")

    master_rng = np.random.default_rng(seed)
    bootstrap_rng = np.random.default_rng(int(seed) + int(bootstrap_seed_offset))
    summary_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []

    settings_total = int(sum(len(sweep["values"]) for sweep in cfg["sweeps"]))
    settings_bar = tqdm(
        total=settings_total,
        desc="sweep settings",
        disable=not show_progress,
    )

    for sweep in cfg["sweeps"]:
        sweep_name = str(sweep["name"])
        sweep_values = [int(v) for v in sweep["values"]]

        for sweep_value in sweep_values:
            split_cfg = _apply_sweep_to_split_config(base_split, sweep_name, sweep_value)
            accumulators: dict[tuple[str, int, int], DrawAccumulator] = {}
            prompt_total = int(n_bundles) * int(len(roles)) * int(n_prompts)
            prompt_bar = tqdm(
                total=prompt_total,
                desc=f"{sweep_name}={int(sweep_value)}",
                disable=not show_progress,
                leave=False,
            )

            for bundle_idx in range(n_bundles):
                bundle_seed = int(master_rng.integers(0, np.iinfo(np.int32).max))
                prompt_seed = int(master_rng.integers(0, np.iinfo(np.int32).max))
                bundle = _build_split_bundle_from_cfg(split_cfg, seed=bundle_seed)
                prompt_rng = np.random.default_rng(prompt_seed)

                for role in roles:
                    role_bank = bundle.train_bank if role == "train" else bundle.eval_bank
                    for prompt_idx in range(n_prompts):
                        src_layer, ants, goal_atom, goal_layer = _sample_prompt_state(
                            bank=role_bank,
                            role=role,
                            initial_ant_max=initial_ant_max,
                            sample_max_attempts=sample_max_attempts,
                            max_unify_solutions=max_unify_solutions,
                            rng=prompt_rng,
                        )
                        prompt_bar.update(1)
                        schemas = _collect_applicable_demo_schemas(
                            rule_bank=role_bank,
                            src_layer=int(src_layer),
                            ants=ants,
                            max_unify_solutions=max_unify_solutions,
                        )
                        schema_count = int(len(schemas))
                        context_constants = _constants_from_facts(ants)
                        context_constants_count = int(len(context_constants))

                        for n_demos in exact_n_demos_values:
                            key = (role, int(src_layer), int(n_demos))
                            acc = accumulators.setdefault(key, DrawAccumulator())
                            acc.n_prompts += 1
                            acc.context_constants_count_sum += context_constants_count

                            if schema_count <= 0:
                                acc.n_no_schema += 1
                                if save_trial_rows:
                                    trial_rows.append(
                                        {
                                            "sweep_name": sweep_name,
                                            "sweep_value": int(sweep_value),
                                            "bundle_idx": int(bundle_idx),
                                            "bundle_seed": int(bundle_seed),
                                            "prompt_seed": int(prompt_seed),
                                            "role": role,
                                            "src_layer": int(src_layer),
                                            "prompt_idx": int(prompt_idx),
                                            "exact_n_demos": int(n_demos),
                                            "schema_count": 0,
                                            "unique_schema_count": 0,
                                            "n_bad": 0,
                                            "n_schema_choices": 0,
                                            "p_bad_prompt": np.nan,
                                            "goal_atom": goal_atom.text,
                                            "goal_layer": int(goal_layer),
                                            "context_constants_count": int(context_constants_count),
                                            "no_demo_schema": 1,
                                            "n_schema_no_valid_grounding": 0,
                                        }
                                    )
                                continue

                            acc.n_prompts_with_schema += 1
                            acc.schema_count_sum += schema_count

                            sampled_schemas = _sample_demo_schemas(
                                rng=prompt_rng,
                                schemas=schemas,
                                n_demos=int(n_demos),
                            )
                            unique_sampled_schemas = _unique_schemas_in_order(sampled_schemas)
                            n_unique = int(len(unique_sampled_schemas))
                            if n_unique <= 0:
                                continue

                            acc.unique_schema_count_sum += n_unique
                            n_bad = 0
                            n_schema_no_valid_grounding = 0
                            for schema in unique_sampled_schemas:
                                reaches_goal, has_valid_grounding = _schema_reaches_goal_via_context_groundings(
                                    rule_bank=role_bank,
                                    schema=schema,
                                    ants=ants,
                                    goal=goal_atom,
                                    goal_layer=int(goal_layer),
                                    max_unify_solutions=max_unify_solutions,
                                )
                                if not has_valid_grounding:
                                    n_schema_no_valid_grounding += 1
                                n_bad += int(not reaches_goal)

                            acc.n_bad += int(n_bad)
                            acc.n_schema_choices += int(n_unique)
                            acc.n_schema_no_valid_grounding += int(n_schema_no_valid_grounding)
                            p_bad_prompt = float(n_bad / n_unique)
                            acc.p_bad_prompt_values.append(p_bad_prompt)

                            if save_trial_rows:
                                trial_rows.append(
                                    {
                                        "sweep_name": sweep_name,
                                        "sweep_value": int(sweep_value),
                                        "bundle_idx": int(bundle_idx),
                                        "bundle_seed": int(bundle_seed),
                                        "prompt_seed": int(prompt_seed),
                                        "role": role,
                                        "src_layer": int(src_layer),
                                        "prompt_idx": int(prompt_idx),
                                        "exact_n_demos": int(n_demos),
                                        "schema_count": int(schema_count),
                                        "unique_schema_count": int(n_unique),
                                        "n_bad": int(n_bad),
                                        "n_schema_choices": int(n_unique),
                                        "p_bad_prompt": p_bad_prompt,
                                        "goal_atom": goal_atom.text,
                                        "goal_layer": int(goal_layer),
                                        "context_constants_count": int(context_constants_count),
                                        "no_demo_schema": 0,
                                        "n_schema_no_valid_grounding": int(n_schema_no_valid_grounding),
                                    }
                                )

            for (role, src_layer, n_demos), acc in sorted(accumulators.items()):
                p_bad = (
                    float(np.mean(acc.p_bad_prompt_values))
                    if acc.p_bad_prompt_values
                    else float("nan")
                )
                ci_low, ci_high = _bootstrap_mean_ci(
                    acc.p_bad_prompt_values,
                    rng=bootstrap_rng,
                    n_bootstrap=n_bootstrap,
                )
                no_schema_rate = float(acc.n_no_schema / acc.n_prompts) if acc.n_prompts > 0 else 0.0
                mean_schema_count = (
                    float(acc.schema_count_sum / acc.n_prompts_with_schema)
                    if acc.n_prompts_with_schema > 0
                    else 0.0
                )
                mean_unique_schema_count = (
                    float(acc.unique_schema_count_sum / acc.n_prompts_with_schema)
                    if acc.n_prompts_with_schema > 0
                    else 0.0
                )
                mean_context_constants = (
                    float(acc.context_constants_count_sum / acc.n_prompts)
                    if acc.n_prompts > 0
                    else 0.0
                )
                summary_rows.append(
                    {
                        "sweep_name": sweep_name,
                        "sweep_value": int(sweep_value),
                        "role": role,
                        "src_layer": int(src_layer),
                        "exact_n_demos": int(n_demos),
                        "n_bundles": int(n_bundles),
                        "n_prompts": int(acc.n_prompts),
                        "n_prompts_with_schema": int(acc.n_prompts_with_schema),
                        "n_no_schema": int(acc.n_no_schema),
                        "no_demo_schema_rate": no_schema_rate,
                        "ci_method": ci_method,
                        "mean_applicable_schema_count": mean_schema_count,
                        "mean_unique_schema_count": mean_unique_schema_count,
                        "mean_context_constants_count": mean_context_constants,
                        "n_bad": int(acc.n_bad),
                        "n_schema_choices": int(acc.n_schema_choices),
                        "n_schema_no_valid_grounding": int(acc.n_schema_no_valid_grounding),
                        "n_prompts_scored": int(len(acc.p_bad_prompt_values)),
                        "p_bad": p_bad,
                        "ci_low": float(ci_low),
                        "ci_high": float(ci_high),
                    }
                )
            prompt_bar.close()
            settings_bar.update(1)
    settings_bar.close()

    return pd.DataFrame(summary_rows), pd.DataFrame(trial_rows)


def _plot_heatmaps(summary_df: pd.DataFrame, out_dir: Path) -> None:
    if summary_df.empty:
        return
    required = {"sweep_name", "sweep_value", "role", "src_layer", "exact_n_demos", "p_bad"}
    if not required.issubset(set(summary_df.columns)):
        return

    for (sweep_name, role, src_layer), part in summary_df.groupby(
        ["sweep_name", "role", "src_layer"], sort=True
    ):
        heat = (
            part.pivot(index="exact_n_demos", columns="sweep_value", values="p_bad")
            .sort_index()
            .sort_index(axis=1)
        )
        if heat.empty:
            continue
        plt.figure(figsize=(8, 5))
        sns.heatmap(
            heat,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            cbar_kws={"label": "P(bad random schema choice)"},
        )
        plt.title(
            f"Bad schema-choice probability heatmap ({sweep_name}, role={role}, src={src_layer})"
        )
        plt.xlabel(sweep_name)
        plt.ylabel("exact_n_demos")
        plt.tight_layout()
        plt.savefig(
            out_dir / f"p_bad_heatmap_{sweep_name}_role_{role}_src_{int(src_layer)}.png",
            dpi=180,
        )
        plt.close()


def _plot_lines(summary_df: pd.DataFrame, out_dir: Path) -> None:
    if summary_df.empty:
        return
    required = {
        "sweep_name",
        "sweep_value",
        "role",
        "src_layer",
        "exact_n_demos",
        "p_bad",
        "ci_low",
        "ci_high",
    }
    if not required.issubset(set(summary_df.columns)):
        return

    for (sweep_name, role, src_layer), part in summary_df.groupby(
        ["sweep_name", "role", "src_layer"], sort=True
    ):
        ordered = part.sort_values(["exact_n_demos", "sweep_value"])
        if ordered.empty:
            continue
        plt.figure(figsize=(8, 5))
        sns.lineplot(
            data=ordered,
            x="sweep_value",
            y="p_bad",
            hue="exact_n_demos",
            marker="o",
        )
        for _, row in ordered.iterrows():
            if pd.isna(row["p_bad"]):
                continue
            plt.vlines(
                x=float(row["sweep_value"]),
                ymin=float(row["ci_low"]),
                ymax=float(row["ci_high"]),
                color="gray",
                linewidth=1.0,
                alpha=0.4,
            )
        plt.ylim(0.0, 1.0)
        plt.title(f"Bad schema-choice probability ({sweep_name}, role={role}, src={src_layer})")
        plt.xlabel(sweep_name)
        plt.ylabel("P(bad random schema choice)")
        plt.tight_layout()
        plt.savefig(
            out_dir / f"p_bad_lines_{sweep_name}_role_{role}_src_{int(src_layer)}.png",
            dpi=180,
        )
        plt.close()


def save_outputs(
    *,
    cfg: dict[str, Any],
    summary_df: pd.DataFrame,
    trial_df: pd.DataFrame,
) -> Path:
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    if bool(cfg["save_trial_rows"]) and not trial_df.empty:
        trial_df.to_csv(out_dir / "trials.csv", index=False)

    (out_dir / "config.json").write_text(json.dumps(_json_safe(cfg), indent=2))

    _plot_heatmaps(summary_df, out_dir=out_dir)
    _plot_lines(summary_df, out_dir=out_dir)
    return out_dir


def print_console_summary(summary_df: pd.DataFrame) -> None:
    print("\n=== Random schema-choice bad-probability summary ===")
    if summary_df.empty:
        print("(no rows)")
        return
    cols = [
        "sweep_name",
        "sweep_value",
        "role",
        "src_layer",
        "exact_n_demos",
        "p_bad",
        "ci_low",
        "ci_high",
        "n_schema_choices",
        "no_demo_schema_rate",
        "mean_applicable_schema_count",
        "mean_unique_schema_count",
    ]
    ordered = summary_df[cols].sort_values(
        ["sweep_name", "role", "src_layer", "exact_n_demos", "sweep_value"]
    )
    print(ordered.to_string(index=False))


# <codecell>
SUMMARY_DF, TRIAL_DF = run_study(CONFIG)
OUT_DIR = save_outputs(cfg=CONFIG, summary_df=SUMMARY_DF, trial_df=TRIAL_DF)
print(f"\nSaved outputs to: {OUT_DIR}")
print_console_summary(SUMMARY_DF)
