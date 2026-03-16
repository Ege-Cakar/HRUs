# <codecell>
"""Interactive stats for cluster demo distribution parameters.

Computes two probabilities across a grid of (cluster_k x cluster_n_samples x alpha x n_demos):

1. **P(demo good)**: P(at least one cluster-sampled in-context demo is rank-1,
   i.e. applicable to current facts AND leads to a goal-reachable state).
2. **P(random good)**: P(a uniformly random rulebank rule at the source layer
   is rank-1) -- baseline difficulty of blind guessing.

Uses the depth-3 fresh-ICL setup from experiment 13 where all rules have
arity 0, k_in=1, k_out=1.  Layer 0 has 1 predicate, layer 1 has ``mid_pred``
predicates, and layer 2 has ``base_num_pred`` (default 16) predicates.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
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

from task.layer_fol import FOLLayerTask
from task.layer_fol.cluster_precompute import ClusterPrecomputeClient
from task.layer_fol.demos import (
    _batch_cluster_select,
    _classify_rules_by_rank,
    _precompute_cluster_candidate_rankings,
)
from task.layer_gen.util.fol_rule_bank import (
    FOLAtom,
    FOLRuleBank,
    sample_fol_problem,
)


# <codecell>
# Edit settings here (no argparse by design).
CONFIG = {
    "seed": 0,
    "out_dir": ROOT / "task" / "interactive" / "set" / "layer_stats_cluster",
    "base_num_pred": 16,
    "sweep_mid_pred": [64],
    "sweep_alpha": [0, 2, 4, 10],
    "exact_n_demos_values": [1, 8, 16, 32, 64],
    "sweep_cluster_k": [1, 25, 100],
    "sweep_cluster_n_samples": [100],
    "cluster_base_dist": "zipf_per_rule",
    "cluster_unselected_rank": None,
    "step_indices": (0, 1),
    "n_banks_per_setting": 1,
    "n_prompts_per_bank": 100,
    "task_kwargs": {
        "arity_max": 0,
        "arity_min": 0,
        "vars_per_rule_max": 6,
        "k_in_min": 1,
        "k_in_max": 1,
        "k_out_min": 1,
        "k_out_max": 1,
        "constants": ("c0",),
        "initial_ant_max": 1,
        "sample_max_attempts": 4096,
        "max_unify_solutions": 128,
    },
    "n_inspect_samples": 3,
    "ci_method": "bootstrap",
    "n_bootstrap": 1000,
    "bootstrap_seed_offset": 100_003,
    "show_progress": True,
    "save_trial_rows": False,
}

_out_dir_env = os.environ.get("LAYER_STATS_CLUSTER_OUT_DIR")
if _out_dir_env is not None and str(_out_dir_env).strip():
    CONFIG["out_dir"] = Path(str(_out_dir_env).strip()).expanduser()

if os.environ.get("LAYER_STATS_CLUSTER_SMOKE", "").strip():
    CONFIG["n_banks_per_setting"] = 1
    CONFIG["n_prompts_per_bank"] = 24
    CONFIG["sweep_mid_pred"] = [32, 64]
    CONFIG["sweep_cluster_k"] = [1, 3]
    CONFIG["sweep_cluster_n_samples"] = [8, 20]
    CONFIG["sweep_alpha"] = [0, 1]
    CONFIG["exact_n_demos_values"] = [1, 4]

### START TEST CONFIGS
# CONFIG["n_banks_per_setting"] = 1
# CONFIG["n_prompts_per_bank"] = 24
# CONFIG["sweep_mid_pred"] = [32, 64]
# CONFIG["sweep_cluster_k"] = [1, 3]
# CONFIG["sweep_cluster_n_samples"] = [8, 20]
# CONFIG["sweep_alpha"] = [0, 1]
# CONFIG["exact_n_demos_values"] = [1, 4]
# CONFIG["step_indices"] = (0,)
### END TEST CONFIGS


# <codecell>
def _sample_from_task(
    task: FOLLayerTask,
    rng: np.random.Generator,
    step_idx: int,
) -> tuple[FOLRuleBank, int, tuple[FOLAtom, ...], FOLAtom] | None:
    """Sample a single prompt state using a FOLLayerTask's bank config.

    Returns ``(temp_bank, src_layer, ants, goal_atom)`` or ``None`` if no
    feasible problem could be sampled.
    """
    temp_bank = task.build_fresh_temp_bank(rng)
    try:
        sampled = sample_fol_problem(
            bank=temp_bank,
            distance=2,
            initial_ant_max=task.initial_ant_max,
            rng=rng,
            max_attempts=task.sample_max_attempts,
            max_unify_solutions=task.max_unify_solutions,
        )
    except RuntimeError:
        return None
    src_layer = int(sampled.step_layers[step_idx])
    ants = tuple(sampled.step_ants[step_idx])
    return temp_bank, src_layer, ants, sampled.goal_atom


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
def run_study(
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[int, dict]]]:
    seed = int(cfg["seed"])
    step_indices = tuple(int(s) for s in cfg["step_indices"])
    sweep_mid_pred = [int(v) for v in cfg["sweep_mid_pred"]]
    sweep_alpha = [float(v) for v in cfg["sweep_alpha"]]
    exact_n_demos_values = [int(v) for v in cfg["exact_n_demos_values"]]
    sweep_cluster_k = [int(v) for v in cfg["sweep_cluster_k"]]
    sweep_cluster_n_samples = [int(v) for v in cfg["sweep_cluster_n_samples"]]
    cluster_base_dist = str(cfg["cluster_base_dist"])
    cluster_unselected_rank = cfg["cluster_unselected_rank"]
    n_banks = int(cfg["n_banks_per_setting"])
    n_prompts = int(cfg["n_prompts_per_bank"])
    base_num_pred = int(cfg["base_num_pred"])
    task_kwargs = dict(cfg["task_kwargs"])
    max_unify_solutions = int(task_kwargs["max_unify_solutions"])
    n_inspect_samples = int(cfg.get("n_inspect_samples", 3))
    ci_method = str(cfg.get("ci_method", "bootstrap")).strip().lower()
    n_bootstrap = int(cfg.get("n_bootstrap", 1000))
    bootstrap_seed_offset = int(cfg.get("bootstrap_seed_offset", 100_003))
    save_trial_rows = bool(cfg["save_trial_rows"])
    show_progress = bool(cfg.get("show_progress", True)) and tqdm is not None
    if ci_method != "bootstrap":
        raise ValueError(
            f"Unsupported ci_method={ci_method!r}; expected 'bootstrap'."
        )

    master_rng = np.random.default_rng(seed)
    bootstrap_rng = np.random.default_rng(int(seed) + int(bootstrap_seed_offset))
    summary_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []

    # Launch parallel precompute server for cluster candidate rankings.
    max_cns = max(sweep_cluster_n_samples) if sweep_cluster_n_samples else 0
    n_server_workers = min(os.cpu_count() or 1, max(1, max_cns // 10), 16)
    server: ClusterPrecomputeClient | None = None
    if n_server_workers > 1 and max_cns >= 20:
        try:
            server = ClusterPrecomputeClient(n_workers=n_server_workers, cwd=ROOT)
        except Exception:
            server = None

    # Accumulator key: (cluster_k, cluster_n_samples, mid_pred, alpha, n_demos, step_idx)
    AccKey = tuple[int, int, int, float, int, int]
    p_demo_good_acc: dict[AccKey, list[float]] = {}
    p_random_good_acc: dict[AccKey, list[float]] = {}
    n_prompts_acc: dict[AccKey, int] = {}
    n_rank1_sum_acc: dict[AccKey, int] = {}
    n_rank2_sum_acc: dict[AccKey, int] = {}
    n_rank3_sum_acc: dict[AccKey, int] = {}
    n_rank4_sum_acc: dict[AccKey, int] = {}
    n_total_rules_sum_acc: dict[AccKey, int] = {}
    n_sample_failures_acc: dict[AccKey, int] = {}
    p_good_is_valid_acc: dict[AccKey, list[float]] = {}
    inspect_records: list[tuple[int, dict]] = []

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
            task = FOLLayerTask(
                mode="online",
                task_split="depth3_fresh_icl",
                distance_range=2,
                batch_size=1,
                seed=bank_seed,
                predicates_per_layer=(base_num_pred, mid_pred, base_num_pred),
                rules_per_transition=(base_num_pred ** 2, base_num_pred ** 2),
                fresh_icl_base_bank_seed=bank_seed,
                max_n_demos=0,
                online_prefetch=True,
                **task_kwargs,
            )
            prompt_rng = np.random.default_rng(prompt_seed)

            for step_idx in step_indices:
                for prompt_idx in range(n_prompts):
                    result = _sample_from_task(task, prompt_rng, step_idx)
                    prompt_bar.update(1)
                    if result is None:
                        for ck in sweep_cluster_k:
                            for cns in sweep_cluster_n_samples:
                                for alpha in sweep_alpha:
                                    for n_demos in exact_n_demos_values:
                                        key: AccKey = (ck, cns, mid_pred, alpha, n_demos, step_idx)
                                        n_sample_failures_acc[key] = (
                                            n_sample_failures_acc.get(key, 0) + 1
                                        )
                        continue

                    temp_bank, src_layer, ants, goal_atom = result

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
                    n_rank2 = len(ranked.get(2, []))
                    n_rank3 = len(ranked.get(3, []))
                    n_rank4 = len(ranked.get(4, []))
                    n_total = len(rules)

                    # P(random good) for this prompt
                    p_random_good = n_rank1 / n_total if n_total > 0 else 0.0

                    # P(good is valid): among goal-reachable rules (rank 1 + rank 3),
                    # what fraction are also applicable (rank 1)?
                    n_goal_reachable = n_rank1 + n_rank3
                    p_good_is_valid = (
                        n_rank1 / n_goal_reachable if n_goal_reachable > 0 else float("nan")
                    )

                    # Pre-draw one RNG seed per cluster_n_samples value
                    # so that expanding sweeps does not change prompt_rng state.
                    cns_precomp_seeds = {
                        cns: int(prompt_rng.integers(0, np.iinfo(np.int64).max))
                        for cns in sweep_cluster_n_samples
                    }
                    cns_select_seeds = {
                        cns: int(prompt_rng.integers(0, np.iinfo(np.int64).max))
                        for cns in sweep_cluster_n_samples
                    }

                    rule_to_idx = {rule: i for i, rule in enumerate(rules)}

                    for cns in sweep_cluster_n_samples:
                        # Precompute candidate rankings (expensive) once per cns
                        precomp_rng = np.random.default_rng(cns_precomp_seeds[cns])
                        candidates = _precompute_cluster_candidate_rankings(
                            rule_bank=temp_bank,
                            src_layer=src_layer,
                            rules=rules,
                            actual_ranked=ranked,
                            rng=precomp_rng,
                            cluster_n_samples=cns,
                            max_unify_solutions=max_unify_solutions,
                            distance=2,
                            initial_ant_max=int(task_kwargs["initial_ant_max"]),
                            server=server,
                        )

                        # Batch select: vectorised weights, shared distance
                        # matrix across ck values.
                        select_rng = np.random.default_rng(cns_select_seeds[cns])
                        batch_results = _batch_cluster_select(
                            candidate_rankings=candidates,
                            actual_ranked=ranked,
                            rules=rules,
                            rule_to_idx=rule_to_idx,
                            rng=select_rng,
                            alpha_values=sweep_alpha,
                            n_demos_values=exact_n_demos_values,
                            cluster_k_values=sweep_cluster_k,
                            cluster_base_dist=cluster_base_dist,
                            cluster_unselected_rank=cluster_unselected_rank,
                            demo_unique=task.demo_unique,
                        )

                        # Record results for all (ck, alpha, n_demos) combos
                        for (ck, alpha, n_demos), (sampled, sampled_ranks) in batch_results.items():
                            key = (ck, cns, mid_pred, alpha, n_demos, step_idx)

                            p_demo_good_acc.setdefault(key, [])
                            p_random_good_acc.setdefault(key, [])
                            n_prompts_acc[key] = n_prompts_acc.get(key, 0) + 1
                            n_rank1_sum_acc[key] = (
                                n_rank1_sum_acc.get(key, 0) + n_rank1
                            )
                            n_rank2_sum_acc[key] = (
                                n_rank2_sum_acc.get(key, 0) + n_rank2
                            )
                            n_rank3_sum_acc[key] = (
                                n_rank3_sum_acc.get(key, 0) + n_rank3
                            )
                            n_rank4_sum_acc[key] = (
                                n_rank4_sum_acc.get(key, 0) + n_rank4
                            )
                            n_total_rules_sum_acc[key] = (
                                n_total_rules_sum_acc.get(key, 0) + n_total
                            )

                            has_good_demo = float(
                                any(r == 1 for r in sampled_ranks)
                            )
                            p_demo_good_acc[key].append(has_good_demo)
                            p_random_good_acc[key].append(p_random_good)
                            p_good_is_valid_acc.setdefault(key, [])
                            p_good_is_valid_acc[key].append(p_good_is_valid)

                            if save_trial_rows:
                                trial_rows.append(
                                    {
                                        "cluster_k": int(ck),
                                        "cluster_n_samples": int(cns),
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
                                        "p_good_is_valid": p_good_is_valid,
                                        "has_good_demo": has_good_demo,
                                    }
                                )

            task.close()

        # Collect inspection samples for this mid_pred value
        if n_inspect_samples > 0:
            inspect_seed = int(master_rng.integers(0, np.iinfo(np.int32).max))
            inspect_task = FOLLayerTask(
                mode="online",
                task_split="depth3_fresh_icl",
                distance_range=2,
                batch_size=1,
                seed=inspect_seed,
                predicates_per_layer=(base_num_pred, mid_pred, base_num_pred),
                rules_per_transition=(base_num_pred ** 2, base_num_pred ** 2),
                fresh_icl_base_bank_seed=inspect_seed,
                max_n_demos=8,
                min_n_demos=8,
                demo_distribution="cluster",
                demo_distribution_alpha=1.0,
                cluster_k=3,
                cluster_n_samples=20,
                cluster_base_dist=cluster_base_dist,
                cluster_unselected_rank=cluster_unselected_rank,
                online_prefetch=False,
                **task_kwargs,
            )
            inspect_rng = np.random.default_rng(inspect_seed)
            for _ in range(n_inspect_samples):
                record = inspect_task._strategy.sample_record(rng=inspect_rng)
                inspect_records.append((mid_pred, record))
            inspect_task.close()

        prompt_bar.close()
        settings_bar.update(1)
    settings_bar.close()

    if server is not None:
        server.close()

    # Build summary rows
    for key in sorted(p_demo_good_acc.keys()):
        ck, cns, mid_pred, alpha, n_demos, step_idx = key
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
        good_valid_vals = p_good_is_valid_acc.get(key, [])
        # Filter NaN values (prompts with no goal-reachable rules)
        good_valid_finite = [v for v in good_valid_vals if not np.isnan(v)]
        p_good_valid = float(np.mean(good_valid_finite)) if good_valid_finite else float("nan")
        p_good_valid_ci_low, p_good_valid_ci_high = _bootstrap_mean_ci(
            good_valid_finite, rng=bootstrap_rng, n_bootstrap=n_bootstrap,
        )

        mean_n_rank1 = (
            float(n_rank1_sum_acc.get(key, 0) / n_total_prompts)
            if n_total_prompts > 0
            else 0.0
        )
        mean_n_rank2 = (
            float(n_rank2_sum_acc.get(key, 0) / n_total_prompts)
            if n_total_prompts > 0
            else 0.0
        )
        mean_n_rank3 = (
            float(n_rank3_sum_acc.get(key, 0) / n_total_prompts)
            if n_total_prompts > 0
            else 0.0
        )
        mean_n_rank4 = (
            float(n_rank4_sum_acc.get(key, 0) / n_total_prompts)
            if n_total_prompts > 0
            else 0.0
        )
        mean_n_total_rules = (
            float(n_total_rules_sum_acc.get(key, 0) / n_total_prompts)
            if n_total_prompts > 0
            else 0.0
        )

        # For src_layer, derive from step_idx: step 0 -> layer 0, step 1 -> layer 1
        src_layer_val = int(step_idx)

        summary_rows.append(
            {
                "cluster_k": int(ck),
                "cluster_n_samples": int(cns),
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
                "p_good_is_valid": p_good_valid,
                "p_good_is_valid_ci_low": float(p_good_valid_ci_low),
                "p_good_is_valid_ci_high": float(p_good_valid_ci_high),
                "mean_n_rank1": mean_n_rank1,
                "mean_n_rank2": mean_n_rank2,
                "mean_n_rank3": mean_n_rank3,
                "mean_n_rank4": mean_n_rank4,
                "mean_n_total_rules": mean_n_total_rules,
                "n_sample_failures": n_sample_failures_acc.get(key, 0),
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(trial_rows), inspect_records


# <codecell>
def _plot_demo_good_heatmaps(summary_df: pd.DataFrame, out_dir: Path) -> None:
    """Heatmaps of p_demo_good: alpha (columns) x n_demos (rows), per (cluster_k, cluster_n_samples, mid_pred, step_idx)."""
    if summary_df.empty:
        return
    required = {"cluster_k", "cluster_n_samples", "mid_pred", "alpha", "n_demos", "step_idx", "p_demo_good"}
    if not required.issubset(set(summary_df.columns)):
        return

    group_cols = ["cluster_k", "cluster_n_samples", "mid_pred", "step_idx"]

    for group_key, part in summary_df.groupby(group_cols, sort=True):
        ck, cns, mid_pred, step_idx = group_key
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
            f"P(demo good) heatmap (k={ck}, ns={cns}, mid_pred={mid_pred}, step={step_idx})"
        )
        plt.xlabel("alpha")
        plt.ylabel("n_demos")
        plt.tight_layout()
        plt.savefig(
            out_dir
            / f"p_demo_good_heatmap_k{int(ck)}_ns{int(cns)}_mid{int(mid_pred)}_step{int(step_idx)}.png",
            dpi=180,
        )
        plt.close()


def _plot_demo_good_lines(summary_df: pd.DataFrame, out_dir: Path) -> None:
    """Line plots of p_demo_good: alpha on x-axis, lines by n_demos, per (cluster_k, cluster_n_samples, mid_pred, step_idx).

    Includes p_random_good as a horizontal dashed reference line.
    """
    if summary_df.empty:
        return
    required = {
        "cluster_k", "cluster_n_samples", "mid_pred", "alpha", "n_demos", "step_idx",
        "p_demo_good", "p_demo_good_ci_low", "p_demo_good_ci_high",
        "p_random_good",
    }
    if not required.issubset(set(summary_df.columns)):
        return

    group_cols = ["cluster_k", "cluster_n_samples", "mid_pred", "step_idx"]

    for group_key, part in summary_df.groupby(group_cols, sort=True):
        ck, cns, mid_pred, step_idx = group_key
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
            legend="full"
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
        # p_good_is_valid reference line (constant across alpha/n_demos)
        if "p_good_is_valid" in ordered.columns:
            p_valid_ref = ordered["p_good_is_valid"].mean()
            if not pd.isna(p_valid_ref):
                plt.axhline(
                    y=p_valid_ref,
                    color="magenta",
                    linestyle="--",
                    linewidth=1.5,
                    label=f"P(good is valid)={p_valid_ref:.3f}",
                )
        plt.ylim(0.0, 1.05)
        plt.title(
            f"P(demo good) (k={ck}, ns={cns}, mid_pred={mid_pred}, step={step_idx})"
        )
        plt.xlabel("alpha")
        plt.ylabel("P(demo good)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            out_dir
            / f"p_demo_good_lines_k{int(ck)}_ns{int(cns)}_mid{int(mid_pred)}_step{int(step_idx)}.png",
            dpi=180,
        )
        plt.close()


def _plot_demo_good_vs_cluster_params(summary_df: pd.DataFrame, out_dir: Path) -> None:
    """Heatmap of p_demo_good: cluster_k (rows) x cluster_n_samples (columns), per (mid_pred, step_idx, alpha, n_demos)."""
    if summary_df.empty:
        return
    required = {"cluster_k", "cluster_n_samples", "mid_pred", "alpha", "n_demos", "step_idx", "p_demo_good"}
    if not required.issubset(set(summary_df.columns)):
        return

    group_cols = ["mid_pred", "step_idx", "alpha", "n_demos"]

    for group_key, part in summary_df.groupby(group_cols, sort=True):
        mid_pred, step_idx, alpha_val, n_demos = group_key
        heat = (
            part.pivot(index="cluster_k", columns="cluster_n_samples", values="p_demo_good")
            .sort_index()
            .sort_index(axis=1)
        )
        if heat.empty:
            continue
        plt.figure(figsize=(7, 5))
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
            f"P(demo good) vs cluster params\n(mid={mid_pred}, step={step_idx}, alpha={alpha_val}, n_demos={n_demos})"
        )
        plt.xlabel("cluster_n_samples")
        plt.ylabel("cluster_k")
        plt.tight_layout()
        plt.savefig(
            out_dir
            / f"p_demo_good_cluster_mid{int(mid_pred)}_step{int(step_idx)}_a{float(alpha_val)}_nd{int(n_demos)}.png",
            dpi=180,
        )
        plt.close()


def _plot_random_good_heatmap(summary_df: pd.DataFrame, out_dir: Path) -> None:
    """Heatmap of p_random_good: mid_pred (columns) x step_idx (rows)."""
    if summary_df.empty:
        return
    required = {"mid_pred", "step_idx", "p_random_good"}
    if not required.issubset(set(summary_df.columns)):
        return

    # p_random_good is constant across alpha/n_demos/cluster params for a given (mid_pred, step_idx).
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


def _plot_rank_sizes(summary_df: pd.DataFrame, out_dir: Path) -> None:
    """Line plot of mean rank sizes vs mid_pred, faceted by step_idx."""
    if summary_df.empty:
        return
    rank_cols = ["mean_n_rank1", "mean_n_rank2", "mean_n_rank3", "mean_n_rank4"]
    required = {"step_idx", "mid_pred"} | set(rank_cols)
    if not required.issubset(set(summary_df.columns)):
        return

    # Collapse alpha/n_demos/cluster dimensions by averaging
    grouped = (
        summary_df.groupby(["step_idx", "mid_pred"], sort=True)[rank_cols]
        .mean()
        .reset_index()
    )
    step_vals = sorted(grouped["step_idx"].unique())
    n_steps = len(step_vals)
    if n_steps == 0:
        return

    rank_labels = {
        "mean_n_rank1": "Rank 1 (applicable + goal-reachable)",
        "mean_n_rank2": "Rank 2 (applicable only)",
        "mean_n_rank3": "Rank 3 (goal-reachable only)",
        "mean_n_rank4": "Rank 4 (irrelevant)",
    }
    rank_colors = {
        "mean_n_rank1": "tab:green",
        "mean_n_rank2": "tab:orange",
        "mean_n_rank3": "tab:blue",
        "mean_n_rank4": "tab:red",
    }

    for use_log in (False, True):
        fig, axes = plt.subplots(1, n_steps, figsize=(6 * n_steps, 5), squeeze=False)
        for col_idx, step in enumerate(step_vals):
            ax = axes[0, col_idx]
            sub = grouped[grouped["step_idx"] == step].sort_values("mid_pred")
            for rc in rank_cols:
                ax.plot(
                    sub["mid_pred"], sub[rc],
                    marker="o", label=rank_labels[rc], color=rank_colors[rc],
                )
            ax.set_xlabel("mid_pred")
            ax.set_ylabel("mean count")
            ax.set_title(f"step_idx={step}")
            if use_log:
                ax.set_yscale("log")
            ax.legend(fontsize="small")
        fig.suptitle(
            "Mean rank sizes vs mid_pred" + (" (log scale)" if use_log else ""),
            fontsize=14,
        )
        fig.tight_layout()
        suffix = "_log" if use_log else ""
        fig.savefig(out_dir / f"rank_sizes_vs_mid_pred{suffix}.png", dpi=180)
        plt.close(fig)


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
    _plot_demo_good_vs_cluster_params(summary_df, out_dir=out_dir)
    _plot_random_good_heatmap(summary_df, out_dir=out_dir)
    _plot_rank_sizes(summary_df, out_dir=out_dir)
    return out_dir


def print_console_summary(summary_df: pd.DataFrame) -> None:
    print("\n=== Cluster demo distribution × parameter sweep summary ===")
    if summary_df.empty:
        print("(no rows)")
        return
    cols = [
        "cluster_k",
        "cluster_n_samples",
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
        "p_good_is_valid",
        "p_good_is_valid_ci_low",
        "p_good_is_valid_ci_high",
        "mean_n_rank1",
        "mean_n_rank2",
        "mean_n_rank3",
        "mean_n_rank4",
        "mean_n_total_rules",
        "n_sample_failures",
    ]
    present = [c for c in cols if c in summary_df.columns]
    sort_cols = [c for c in ["cluster_k", "cluster_n_samples", "mid_pred", "step_idx", "alpha", "n_demos"] if c in present]
    ordered = summary_df[present].sort_values(sort_cols)
    print(ordered.to_string(index=False))


def print_inspect_samples(
    inspect_records: list[tuple[int, dict]],
) -> None:
    """Print a few full task records for visual inspection."""
    if not inspect_records:
        return
    print("\n=== Inspection samples ===")
    total_per_mid = {}
    for mid_pred, _ in inspect_records:
        total_per_mid[mid_pred] = total_per_mid.get(mid_pred, 0) + 1
    counter: dict[int, int] = {}
    for mid_pred, record in inspect_records:
        counter[mid_pred] = counter.get(mid_pred, 0) + 1
        idx = counter[mid_pred]
        total = total_per_mid[mid_pred]
        print(f"--- mid_pred={mid_pred}, sample {idx}/{total} ---")
        print(f"  src_layer: {record.get('src_layer', '?')}")
        ctx = record.get("rule_context", {})
        active_rules = ctx.get("active_rule_texts", [])
        print(f"  n_active_rules: {len(active_rules)}")
        statements = record.get("statement_texts", [])
        if statements:
            print(f"  answer: {statements[0]}")
        demo_texts = ctx.get("demo_schema_texts", [])
        demo_ranks = ctx.get("demo_ranks", [])
        if demo_texts:
            print(f"  demos ({len(demo_texts)}):")
            for d_idx, text in enumerate(demo_texts):
                rank_str = f"[rank {demo_ranks[d_idx]}] " if d_idx < len(demo_ranks) else ""
                print(f"    {rank_str}{text}")
    print()


# <codecell>
SUMMARY_DF, TRIAL_DF, INSPECT_RECORDS = run_study(CONFIG)
OUT_DIR = save_outputs(cfg=CONFIG, summary_df=SUMMARY_DF, trial_df=TRIAL_DF)
print(f"\nSaved outputs to: {OUT_DIR}")
print_console_summary(SUMMARY_DF)
print_inspect_samples(INSPECT_RECORDS)
