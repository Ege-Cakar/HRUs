# <codecell>
"""Interactive stats for mid-layer predicate count vs Zipf alpha difficulty.

Computes two probabilities across a grid of (mid_pred × alpha × n_demos):

1. **P(demo good)**: P(at least one Zipf-sampled in-context demo is rank-1,
   i.e. applicable to current facts AND leads to a goal-reachable state).
2. **P(random good)**: P(a uniformly random rulebank rule at the source layer
   is rank-1) — baseline difficulty of blind guessing.

Uses the depth-3 fresh-ICL setup from experiment 13 where all rules have
arity 0, k_in=1, k_out=1.  Layer 0 has 1 predicate, layer 1 has ``mid_pred``
predicates, and layer 2 has ``base_num_pred`` (default 16) predicates.
"""

from __future__ import annotations

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

from task.layer_fol.demos import (
    _classify_rules_by_rank,
    _sample_demo_schemas_zipf,
)
from task.layer_gen.util.fol_rule_bank import (
    FOLAtom,
    FOLLayerRule,
    FOLRuleBank,
    build_fresh_layer0_bank,
    build_random_fol_rule_bank,
    generate_fresh_predicate_names,
    sample_fol_problem,
)


# <codecell>
# Edit settings here (no argparse by design).
CONFIG = {
    "seed": 0,
    "out_dir": ROOT / "task" / "interactive" / "set" / "layer_stats_mid_pred",
    "base_num_pred": 16,
    "sweep_mid_pred": [16, 32, 64, 128, 256],
    "sweep_alpha": [0, 0.5, 1, 10],
    "exact_n_demos_values": [1, 2, 4, 8, 16, 32],
    "step_indices": (0, 1),
    "n_banks_per_setting": 2,
    "n_prompts_per_bank": 200,
    "bank": {
        "arity_max": 0,
        "vars_per_rule_max": 6,
        "k_in_min": 1,
        "k_in_max": 1,
        "k_out_min": 1,
        "k_out_max": 1,
        "constants_count": 1,
    },
    "initial_ant_max": 1,
    "sample_max_attempts": 4096,
    "max_unify_solutions": 128,
    "ci_method": "bootstrap",
    "n_bootstrap": 1000,
    "bootstrap_seed_offset": 100_003,
    "show_progress": True,
    "save_trial_rows": False,
}

_out_dir_env = os.environ.get("LAYER_STATS_MID_PRED_OUT_DIR")
if _out_dir_env is not None and str(_out_dir_env).strip():
    CONFIG["out_dir"] = Path(str(_out_dir_env).strip()).expanduser()

if os.environ.get("LAYER_STATS_MID_PRED_SMOKE", "").strip():
    CONFIG["n_banks_per_setting"] = 1
    CONFIG["n_prompts_per_bank"] = 24
    CONFIG["sweep_mid_pred"] = [16, 32]
    CONFIG["sweep_alpha"] = [0, 1]
    CONFIG["exact_n_demos_values"] = [1, 4]

### START TEST CONFIGS
# CONFIG["n_banks_per_setting"] = 1
# CONFIG["n_prompts_per_bank"] = 24
# CONFIG["sweep_mid_pred"] = [16, 32]
# CONFIG["sweep_alpha"] = [0, 1]
# CONFIG["exact_n_demos_values"] = [1, 4]
# CONFIG["step_indices"] = (0,)
### END TEST CONFIGS


_LAYERED_PRED_RE = re.compile(r"r(\d+)_(\d+)$")


# <codecell>
def _layer_from_predicate(predicate: str) -> int:
    match = _LAYERED_PRED_RE.fullmatch(str(predicate))
    if match is None:
        raise ValueError(f"Unsupported layered predicate name: {predicate!r}")
    return int(match.group(1))


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


# <codecell>
def _build_base_bank(
    mid_pred: int,
    base_num_pred: int,
    bank_cfg: dict[str, Any],
    seed: int,
) -> FOLRuleBank:
    """Build a 3-layer rule bank with (1, mid_pred, base_num_pred) predicates."""
    arity_max = int(bank_cfg["arity_max"])
    return build_random_fol_rule_bank(
        n_layers=3,
        predicates_per_layer=(1, int(mid_pred), int(base_num_pred)),
        rules_per_transition=(int(base_num_pred), int(base_num_pred) ** 2),
        arity_min=min(1, arity_max),
        arity_max=arity_max,
        vars_per_rule_max=int(bank_cfg["vars_per_rule_max"]),
        k_in_min=int(bank_cfg["k_in_min"]),
        k_in_max=int(bank_cfg["k_in_max"]),
        k_out_min=int(bank_cfg["k_out_min"]),
        k_out_max=int(bank_cfg["k_out_max"]),
        constants=_constants_from_count(int(bank_cfg["constants_count"])),
        rng=np.random.default_rng(int(seed)),
    )


def _sample_prompt_state_fresh(
    *,
    base_bank: FOLRuleBank,
    bank_cfg: dict[str, Any],
    forced_step_idx: int,
    initial_ant_max: int,
    sample_max_attempts: int,
    max_unify_solutions: int,
    rng: np.random.Generator,
) -> tuple[FOLRuleBank, int, tuple[FOLAtom, ...], FOLAtom, int] | None:
    """Sample a single prompt state with fresh layer-0 predicates.

    Returns ``(temp_bank, src_layer, ants, goal_atom, goal_layer)`` or
    ``None`` if no feasible problem could be sampled.
    """
    n_fresh = len(base_bank.predicates_for_layer(0))
    fresh_preds = generate_fresh_predicate_names(n_fresh, rng)
    temp_bank = build_fresh_layer0_bank(
        base_bank=base_bank,
        fresh_predicates=fresh_preds,
        rules_per_transition=len(base_bank.transition_rules(0)),
        k_in_min=int(bank_cfg["k_in_min"]),
        k_in_max=int(bank_cfg["k_in_max"]),
        k_out_min=int(bank_cfg["k_out_min"]),
        k_out_max=int(bank_cfg["k_out_max"]),
        rng=rng,
    )

    try:
        sampled = sample_fol_problem(
            bank=temp_bank,
            distance=2,
            initial_ant_max=int(initial_ant_max),
            rng=rng,
            max_attempts=int(sample_max_attempts),
            max_unify_solutions=int(max_unify_solutions),
        )
    except RuntimeError:
        return None

    step_idx = int(forced_step_idx)
    src_layer = int(sampled.step_layers[step_idx])
    ants = tuple(sampled.step_ants[step_idx])
    goal_atom = sampled.goal_atom
    goal_layer = _layer_from_predicate(goal_atom.predicate)
    return temp_bank, src_layer, ants, goal_atom, goal_layer


# <codecell>
def run_study(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed = int(cfg["seed"])
    step_indices = tuple(int(s) for s in cfg["step_indices"])
    sweep_mid_pred = [int(v) for v in cfg["sweep_mid_pred"]]
    sweep_alpha = [float(v) for v in cfg["sweep_alpha"]]
    exact_n_demos_values = [int(v) for v in cfg["exact_n_demos_values"]]
    n_banks = int(cfg["n_banks_per_setting"])
    n_prompts = int(cfg["n_prompts_per_bank"])
    base_num_pred = int(cfg["base_num_pred"])
    initial_ant_max = int(cfg["initial_ant_max"])
    sample_max_attempts = int(cfg["sample_max_attempts"])
    max_unify_solutions = int(cfg["max_unify_solutions"])
    ci_method = str(cfg.get("ci_method", "bootstrap")).strip().lower()
    n_bootstrap = int(cfg.get("n_bootstrap", 1000))
    bootstrap_seed_offset = int(cfg.get("bootstrap_seed_offset", 100_003))
    save_trial_rows = bool(cfg["save_trial_rows"])
    show_progress = bool(cfg.get("show_progress", True)) and tqdm is not None
    bank_cfg = dict(cfg["bank"])
    if ci_method != "bootstrap":
        raise ValueError(
            f"Unsupported ci_method={ci_method!r}; expected 'bootstrap'."
        )

    master_rng = np.random.default_rng(seed)
    bootstrap_rng = np.random.default_rng(int(seed) + int(bootstrap_seed_offset))
    summary_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []

    # Accumulator key: (mid_pred, alpha, n_demos, step_idx)
    AccKey = tuple[int, float, int, int]
    p_demo_good_acc: dict[AccKey, list[float]] = {}
    p_random_good_acc: dict[AccKey, list[float]] = {}
    n_prompts_acc: dict[AccKey, int] = {}
    n_rank1_sum_acc: dict[AccKey, int] = {}
    n_total_rules_sum_acc: dict[AccKey, int] = {}
    n_sample_failures_acc: dict[AccKey, int] = {}

    settings_total = len(sweep_mid_pred)
    settings_bar = tqdm(
        total=settings_total,
        desc="mid_pred settings",
        disable=not show_progress,
    )

    for mid_pred in sweep_mid_pred:
        prompt_total = (
            int(n_banks) * int(len(step_indices)) * int(n_prompts)
        )
        prompt_bar = tqdm(
            total=prompt_total,
            desc=f"mid_pred={int(mid_pred)}",
            disable=not show_progress,
            leave=False,
        )

        for bank_idx in range(n_banks):
            bank_seed = int(master_rng.integers(0, np.iinfo(np.int32).max))
            prompt_seed = int(master_rng.integers(0, np.iinfo(np.int32).max))
            base_bank = _build_base_bank(mid_pred, base_num_pred, bank_cfg, seed=bank_seed)
            prompt_rng = np.random.default_rng(prompt_seed)

            for step_idx in step_indices:
                for prompt_idx in range(n_prompts):
                    result = _sample_prompt_state_fresh(
                        base_bank=base_bank,
                        bank_cfg=bank_cfg,
                        forced_step_idx=step_idx,
                        initial_ant_max=initial_ant_max,
                        sample_max_attempts=sample_max_attempts,
                        max_unify_solutions=max_unify_solutions,
                        rng=prompt_rng,
                    )
                    prompt_bar.update(1)
                    if result is None:
                        for alpha in sweep_alpha:
                            for n_demos in exact_n_demos_values:
                                key: AccKey = (mid_pred, alpha, n_demos, step_idx)
                                n_sample_failures_acc[key] = (
                                    n_sample_failures_acc.get(key, 0) + 1
                                )
                        continue

                    temp_bank, src_layer, ants, goal_atom, goal_layer = result

                    # Classify ALL rules at src_layer into ranks 1-4
                    rules = list(temp_bank.transition_rules(src_layer))
                    ranked = _classify_rules_by_rank(
                        rules=rules,
                        ants=ants,
                        goal_atom=goal_atom,
                        rule_bank=temp_bank,
                        max_unify_solutions=max_unify_solutions,
                    )
                    n_rank1 = len(ranked.get(1, []))
                    n_total = len(rules)

                    # P(random good) for this prompt
                    p_random_good = n_rank1 / n_total if n_total > 0 else 0.0

                    for alpha in sweep_alpha:
                        # Use a separate rng for demo sampling so alpha
                        # variations don't interfere with the prompt rng.
                        demo_rng = np.random.default_rng(
                            int(prompt_rng.integers(0, np.iinfo(np.int64).max))
                        )
                        for n_demos in exact_n_demos_values:
                            key = (mid_pred, alpha, n_demos, step_idx)

                            # Initialise accumulators on first encounter
                            p_demo_good_acc.setdefault(key, [])
                            p_random_good_acc.setdefault(key, [])
                            n_prompts_acc[key] = n_prompts_acc.get(key, 0) + 1
                            n_rank1_sum_acc[key] = (
                                n_rank1_sum_acc.get(key, 0) + n_rank1
                            )
                            n_total_rules_sum_acc[key] = (
                                n_total_rules_sum_acc.get(key, 0) + n_total
                            )

                            # Sample demos with Zipf(alpha)
                            sampled, sampled_ranks = _sample_demo_schemas_zipf(
                                rng=demo_rng,
                                ranked_rules=ranked,
                                n_demos=n_demos,
                                alpha=alpha,
                                include_oracle=False,
                                oracle_rule=None,
                                demo_ranked=False,
                            )

                            # P(demo good) indicator: any rank-1 rule sampled?
                            has_good_demo = float(
                                any(r == 1 for r in sampled_ranks)
                            )
                            p_demo_good_acc[key].append(has_good_demo)
                            p_random_good_acc[key].append(p_random_good)

                            if save_trial_rows:
                                trial_rows.append(
                                    {
                                        "mid_pred": int(mid_pred),
                                        "alpha": float(alpha),
                                        "n_demos": int(n_demos),
                                        "step_idx": int(step_idx),
                                        "src_layer": int(src_layer),
                                        "bank_idx": int(bank_idx),
                                        "prompt_idx": int(prompt_idx),
                                        "n_rank1": int(n_rank1),
                                        "n_total_rules": int(n_total),
                                        "p_random_good": p_random_good,
                                        "has_good_demo": has_good_demo,
                                    }
                                )

        prompt_bar.close()
        settings_bar.update(1)
    settings_bar.close()

    # Build summary rows
    for key in sorted(p_demo_good_acc.keys()):
        mid_pred, alpha, n_demos, step_idx = key
        demo_vals = p_demo_good_acc[key]
        random_vals = p_random_good_acc[key]
        n_total_prompts = n_prompts_acc.get(key, 0)

        p_demo_good = float(np.mean(demo_vals)) if demo_vals else float("nan")
        p_demo_ci_low, p_demo_ci_high = _bootstrap_mean_ci(
            demo_vals, rng=bootstrap_rng, n_bootstrap=n_bootstrap,
        )
        p_random = float(np.mean(random_vals)) if random_vals else float("nan")
        p_random_ci_low, p_random_ci_high = _bootstrap_mean_ci(
            random_vals, rng=bootstrap_rng, n_bootstrap=n_bootstrap,
        )

        mean_n_rank1 = (
            float(n_rank1_sum_acc.get(key, 0) / n_total_prompts)
            if n_total_prompts > 0
            else 0.0
        )
        mean_n_total_rules = (
            float(n_total_rules_sum_acc.get(key, 0) / n_total_prompts)
            if n_total_prompts > 0
            else 0.0
        )

        # For src_layer, derive from step_idx: step 0 → layer 0, step 1 → layer 1
        src_layer_val = int(step_idx)

        summary_rows.append(
            {
                "mid_pred": int(mid_pred),
                "alpha": float(alpha),
                "n_demos": int(n_demos),
                "step_idx": int(step_idx),
                "src_layer": src_layer_val,
                "n_prompts": int(n_total_prompts),
                "p_demo_good": p_demo_good,
                "p_demo_good_ci_low": float(p_demo_ci_low),
                "p_demo_good_ci_high": float(p_demo_ci_high),
                "p_random_good": p_random,
                "p_random_good_ci_low": float(p_random_ci_low),
                "p_random_good_ci_high": float(p_random_ci_high),
                "mean_n_rank1": mean_n_rank1,
                "mean_n_total_rules": mean_n_total_rules,
                "n_sample_failures": n_sample_failures_acc.get(key, 0),
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(trial_rows)


# <codecell>
def _plot_demo_good_heatmaps(summary_df: pd.DataFrame, out_dir: Path) -> None:
    """Heatmaps of p_demo_good: alpha (columns) × n_demos (rows), per (mid_pred, step_idx)."""
    if summary_df.empty:
        return
    required = {"mid_pred", "alpha", "n_demos", "step_idx", "p_demo_good"}
    if not required.issubset(set(summary_df.columns)):
        return

    for (mid_pred, step_idx), part in summary_df.groupby(
        ["mid_pred", "step_idx"], sort=True
    ):
        heat = (
            part.pivot(index="n_demos", columns="alpha", values="p_demo_good")
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
            cbar_kws={"label": "P(demo good)"},
        )
        plt.title(
            f"P(demo good) heatmap (mid_pred={mid_pred}, step={step_idx})"
        )
        plt.xlabel("alpha")
        plt.ylabel("n_demos")
        plt.tight_layout()
        plt.savefig(
            out_dir
            / f"p_demo_good_heatmap_mid{int(mid_pred)}_step{int(step_idx)}.png",
            dpi=180,
        )
        plt.close()


def _plot_demo_good_lines(summary_df: pd.DataFrame, out_dir: Path) -> None:
    """Line plots of p_demo_good: alpha on x-axis, lines by n_demos, per (mid_pred, step_idx).

    Includes p_random_good as a horizontal dashed reference line.
    """
    if summary_df.empty:
        return
    required = {
        "mid_pred", "alpha", "n_demos", "step_idx",
        "p_demo_good", "p_demo_good_ci_low", "p_demo_good_ci_high",
        "p_random_good",
    }
    if not required.issubset(set(summary_df.columns)):
        return

    for (mid_pred, step_idx), part in summary_df.groupby(
        ["mid_pred", "step_idx"], sort=True
    ):
        ordered = part.sort_values(["n_demos", "alpha"])
        if ordered.empty:
            continue
        plt.figure(figsize=(8, 5))
        sns.lineplot(
            data=ordered,
            x="alpha",
            y="p_demo_good",
            hue="n_demos",
            marker="o",
        )
        for _, row in ordered.iterrows():
            if pd.isna(row["p_demo_good"]):
                continue
            plt.vlines(
                x=float(row["alpha"]),
                ymin=float(row["p_demo_good_ci_low"]),
                ymax=float(row["p_demo_good_ci_high"]),
                color="gray",
                linewidth=1.0,
                alpha=0.4,
            )

        # p_random_good reference line (constant across alpha/n_demos)
        p_random_ref = ordered["p_random_good"].mean()
        if not pd.isna(p_random_ref):
            plt.axhline(
                y=p_random_ref,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label=f"P(random good)={p_random_ref:.3f}",
            )
        plt.ylim(0.0, 1.05)
        plt.title(
            f"P(demo good) (mid_pred={mid_pred}, step={step_idx})"
        )
        plt.xlabel("alpha")
        plt.ylabel("P(demo good)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            out_dir
            / f"p_demo_good_lines_mid{int(mid_pred)}_step{int(step_idx)}.png",
            dpi=180,
        )
        plt.close()


def _plot_random_good_heatmap(summary_df: pd.DataFrame, out_dir: Path) -> None:
    """Heatmap of p_random_good: mid_pred (columns) × step_idx (rows)."""
    if summary_df.empty:
        return
    required = {"mid_pred", "step_idx", "p_random_good"}
    if not required.issubset(set(summary_df.columns)):
        return

    # p_random_good is constant across alpha/n_demos for a given (mid_pred, step_idx).
    # Average to collapse those dimensions.
    grouped = (
        summary_df.groupby(["step_idx", "mid_pred"], sort=True)["p_random_good"]
        .mean()
        .reset_index()
    )
    heat = (
        grouped.pivot(
            index="step_idx", columns="mid_pred", values="p_random_good"
        )
        .sort_index()
        .sort_index(axis=1)
    )
    if heat.empty:
        return
    plt.figure(figsize=(8, 4))
    sns.heatmap(
        heat,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        vmin=0.0,
        vmax=max(0.1, float(heat.max().max())),
        cbar_kws={"label": "P(random good)"},
    )
    plt.title("P(random good) by mid_pred and step_idx")
    plt.xlabel("mid_pred")
    plt.ylabel("step_idx")
    plt.tight_layout()
    plt.savefig(out_dir / "p_random_good_heatmap.png", dpi=180)
    plt.close()


# <codecell>
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

    _plot_demo_good_heatmaps(summary_df, out_dir=out_dir)
    _plot_demo_good_lines(summary_df, out_dir=out_dir)
    _plot_random_good_heatmap(summary_df, out_dir=out_dir)
    return out_dir


def print_console_summary(summary_df: pd.DataFrame) -> None:
    print("\n=== Mid-pred × Zipf alpha difficulty summary ===")
    if summary_df.empty:
        print("(no rows)")
        return
    cols = [
        "mid_pred",
        "alpha",
        "n_demos",
        "step_idx",
        "p_demo_good",
        "p_demo_good_ci_low",
        "p_demo_good_ci_high",
        "p_random_good",
        "p_random_good_ci_low",
        "p_random_good_ci_high",
        "mean_n_rank1",
        "mean_n_total_rules",
        "n_sample_failures",
    ]
    present = [c for c in cols if c in summary_df.columns]
    ordered = summary_df[present].sort_values(
        ["mid_pred", "step_idx", "alpha", "n_demos"]
    )
    print(ordered.to_string(index=False))


# <codecell>
SUMMARY_DF, TRIAL_DF = run_study(CONFIG)
OUT_DIR = save_outputs(cfg=CONFIG, summary_df=SUMMARY_DF, trial_df=TRIAL_DF)
print(f"\nSaved outputs to: {OUT_DIR}")
print_console_summary(SUMMARY_DF)
