"""Analysis script for 8_fol_icl demo-sweep architecture comparison."""

# <codecell>
from __future__ import annotations

import ast
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from common import collate_dfs, set_theme


set_theme()
OUT_DIR = Path("fig/8_fol_icl")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DISTANCE_METRIC_COLS = [
    "loss",
    "token_acc",
    "final_token_acc",
    "seq_exact_acc",
    "n_rule_examples",
    "n_valid_rule",
    "n_invalid_rule",
    "n_correct_rule",
    "n_decode_error",
    "n_unknown_rule_error",
    "n_wrong_rule_error",
    "valid_rule_rate",
    "invalid_rule_rate",
    "correct_rule_rate",
    "correct_given_valid_rate",
    "decode_error_rate",
    "unknown_rule_error_rate",
    "wrong_rule_error_rate",
    "free_run_rule_valid_rate",
    "free_run_rule_reachable_rate",
    "free_run_decode_error_rate",
    "free_run_unknown_rule_error_rate",
    "rollout_n_examples",
    "rollout_success_rate",
    "rollout_decode_error_rate",
    "rollout_unknown_rule_error_rate",
    "rollout_inapplicable_rule_error_rate",
    "rollout_goal_not_reached_rate",
    "rollout_avg_steps",
]

SWEEP_METRICS = [
    "rollout_success_rate",
    "final_token_acc",
    "free_run_rule_reachable_rate",
]


def _as_int_list(value) -> list[int]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return [int(v) for v in value]
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []
        if isinstance(parsed, (list, tuple, np.ndarray)):
            return [int(v) for v in parsed]
    return []


def _infer_train_max_distance(info: dict) -> float:
    val = info.get("train_max_distance")
    if val is not None and not pd.isna(val):
        return float(int(val))
    train_distances = _as_int_list(info.get("train_distances"))
    if not train_distances:
        return np.nan
    return float(max(train_distances))


def _extract_final_row(row):
    info = row.get("info", {}) or {}
    final = row.get("metrics_final", {}) or {}

    train_distances = _as_int_list(info.get("train_distances"))
    eval_distances = _as_int_list(info.get("eval_distances"))
    ood_distances = _as_int_list(info.get("ood_distances"))

    return pd.Series(
        {
            "run_row_id": row.get("__row_id"),
            "run_id": row.get("run_id"),
            "model_family": row.get("model_family"),
            "train_max_distance": _infer_train_max_distance(info),
            "selection_metric_name": row.get("selection_metric_name"),
            "selection_metric_value": row.get("selection_metric_value", np.nan),
            "selection_eval_max_n_demos": row.get("selection_eval_max_n_demos", np.nan),
            "ood_rollout_success_avg": row.get("ood_rollout_success_avg", np.nan),
            "ood_final_token_acc_avg": row.get("ood_final_token_acc_avg", np.nan),
            "ood_valid_rule_rate_avg": row.get("ood_valid_rule_rate_avg", np.nan),
            "ood_correct_rule_rate_avg": row.get("ood_correct_rule_rate_avg", np.nan),
            "ood_free_run_rule_reachable_rate_avg": row.get(
                "ood_free_run_rule_reachable_rate_avg", np.nan
            ),
            "train_distances": str(train_distances),
            "eval_distances": str(eval_distances),
            "ood_distances": str(ood_distances),
            "train_max_n_demos": info.get("train_max_n_demos", np.nan),
            "eval_max_n_demos_sweep": str(_as_int_list(info.get("eval_max_n_demos_sweep"))),
            "lr": info.get("lr", np.nan),
            "n_layers": info.get("n_layers", np.nan),
            "n_hidden": info.get("n_hidden", np.nan),
            "n_heads": info.get("n_heads", np.nan),
            "d_state": info.get("d_state", np.nan),
            "d_conv": info.get("d_conv", np.nan),
            "scan_chunk_len": info.get("scan_chunk_len", np.nan),
            "n_seq": info.get("n_seq", np.nan),
            "n_vocab": info.get("n_vocab", np.nan),
            "loss": final.get("loss", np.nan),
            "final_token_acc": final.get("final_token_acc", np.nan),
            "seq_exact_acc": final.get("seq_exact_acc", np.nan),
            "token_acc": final.get("token_acc", np.nan),
            "valid_rule_rate": final.get("valid_rule_rate", np.nan),
            "correct_rule_rate": final.get("correct_rule_rate", np.nan),
            "free_run_rule_reachable_rate": final.get("free_run_rule_reachable_rate", np.nan),
        }
    )


def _explode_distance_eval_demo_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        info = row.get("info", {}) or {}
        by_eval_demo = row.get("metrics_by_eval_demo", {}) or {}
        train_max_distance = _infer_train_max_distance(info)

        for eval_demo, by_distance in by_eval_demo.items():
            eval_demo_int = int(eval_demo)
            by_distance = by_distance or {}
            for distance, metrics in by_distance.items():
                distance_int = int(distance)
                metrics = metrics or {}
                out = {
                    "run_row_id": row.get("__row_id"),
                    "run_id": row.get("run_id"),
                    "model_family": row.get("model_family"),
                    "train_max_distance": train_max_distance,
                    "eval_max_n_demos": eval_demo_int,
                    "distance": distance_int,
                    "lr": info.get("lr", np.nan),
                    "n_layers": info.get("n_layers", np.nan),
                    "n_hidden": info.get("n_hidden", np.nan),
                }
                for metric_name in DISTANCE_METRIC_COLS:
                    out[metric_name] = metrics.get(metric_name, np.nan)
                rows.append(out)

    return pd.DataFrame(rows)


def select_best_by_family_and_train_max(final_df: pd.DataFrame) -> pd.DataFrame:
    ranked = final_df.sort_values(
        ["train_max_distance", "model_family", "selection_metric_value", "run_row_id"],
        ascending=[True, True, False, True],
    )
    return (
        ranked.groupby(["train_max_distance", "model_family"], as_index=False, sort=True)
        .head(1)
        .reset_index(drop=True)
    )


def filter_distance_eval_df_to_best(
    distance_eval_demo_df: pd.DataFrame,
    best_df: pd.DataFrame,
    train_max_distance: int,
) -> pd.DataFrame:
    best_ids = best_df.loc[
        best_df["train_max_distance"] == float(train_max_distance),
        "run_row_id",
    ]
    mask = (distance_eval_demo_df["train_max_distance"] == float(train_max_distance)) & (
        distance_eval_demo_df["run_row_id"].isin(best_ids)
    )
    return distance_eval_demo_df.loc[mask].copy()


def _compute_ood_by_demo(
    *,
    best_distance_eval_df: pd.DataFrame,
    train_max_distance: int,
) -> pd.DataFrame:
    if best_distance_eval_df.empty:
        return pd.DataFrame()
    ood_df = best_distance_eval_df.loc[
        best_distance_eval_df["distance"] > float(train_max_distance)
    ].copy()
    if ood_df.empty:
        return pd.DataFrame()

    agg = (
        ood_df.groupby(["model_family", "train_max_distance", "eval_max_n_demos"], as_index=False)
        .agg({metric: "mean" for metric in SWEEP_METRICS})
        .sort_values(["model_family", "eval_max_n_demos"])
        .reset_index(drop=True)
    )
    return agg


def _plot_ood_demo_sweep(
    *,
    ood_by_demo: pd.DataFrame,
    metric: str,
    out_path: Path,
    train_max_distance: int,
) -> None:
    if ood_by_demo.empty or metric not in ood_by_demo.columns:
        return

    plt.figure(figsize=(6.5, 4.0))
    sns.lineplot(
        data=ood_by_demo,
        x="eval_max_n_demos",
        y=metric,
        hue="model_family",
        marker="o",
    )
    plt.xscale("symlog", linthresh=1)
    plt.xlabel("Eval max_n_demos")
    plt.ylabel(metric)
    plt.title(f"OOD {metric} vs eval demos (train <= {int(train_max_distance)})")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def _plot_distance_by_demo(
    *,
    best_distance_eval_df: pd.DataFrame,
    metric: str,
    out_path: Path,
    train_max_distance: int,
) -> None:
    if best_distance_eval_df.empty or metric not in best_distance_eval_df.columns:
        return

    plot_df = best_distance_eval_df.dropna(subset=[metric]).copy()
    if plot_df.empty:
        return

    g = sns.relplot(
        data=plot_df,
        kind="line",
        x="distance",
        y=metric,
        hue="model_family",
        col="eval_max_n_demos",
        marker="o",
        col_wrap=4,
        facet_kws={"sharex": True, "sharey": False},
        height=3.0,
        aspect=1.15,
    )
    for ax in np.ravel(g.axes):
        ax.axvline(float(train_max_distance), color="0.35", linestyle="--", linewidth=1.0)
    g.set_axis_labels("Layer distance", metric)
    g.set_titles("eval_max_n_demos={col_name}")
    g.fig.suptitle(f"{metric} by distance and eval demos (train <= {int(train_max_distance)})")
    g.fig.subplots_adjust(top=0.9)
    g.savefig(out_path, bbox_inches="tight")
    plt.close(g.fig)


def _plot_reachable_heatmaps(
    *,
    best_distance_eval_df: pd.DataFrame,
    out_dir: Path,
    train_max_distance: int,
) -> None:
    if best_distance_eval_df.empty:
        return

    metric = "free_run_rule_reachable_rate"
    for family in sorted(best_distance_eval_df["model_family"].unique()):
        fam_df = best_distance_eval_df.loc[best_distance_eval_df["model_family"] == family].copy()
        if fam_df.empty:
            continue

        pivot = fam_df.pivot_table(
            index="distance",
            columns="eval_max_n_demos",
            values=metric,
            aggfunc="mean",
        ).sort_index(axis=0).sort_index(axis=1)
        if pivot.empty:
            continue

        plt.figure(figsize=(6.8, 4.6))
        sns.heatmap(
            pivot,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            cbar_kws={"label": metric},
        )
        plt.xlabel("Eval max_n_demos")
        plt.ylabel("Distance")
        plt.title(f"{family} reachable-rate heatmap (train <= {int(train_max_distance)})")
        plt.savefig(out_dir / f"reachable_heatmap_{family}.svg", bbox_inches="tight")
        plt.close()


def _save_plots_for_train_max(
    *,
    train_max_distance: int,
    best_df: pd.DataFrame,
    distance_eval_demo_df: pd.DataFrame,
    out_root: Path,
) -> None:
    out_dir = out_root / f"train_max_{int(train_max_distance):02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_distance_eval_df = filter_distance_eval_df_to_best(
        distance_eval_demo_df,
        best_df,
        train_max_distance,
    )
    if best_distance_eval_df.empty:
        return

    ood_by_demo = _compute_ood_by_demo(
        best_distance_eval_df=best_distance_eval_df,
        train_max_distance=train_max_distance,
    )
    ood_by_demo.to_csv(out_dir / "ood_by_eval_demo_best.csv", index=False)

    for metric in SWEEP_METRICS:
        _plot_ood_demo_sweep(
            ood_by_demo=ood_by_demo,
            metric=metric,
            out_path=out_dir / f"ood_{metric}_vs_eval_demo.svg",
            train_max_distance=train_max_distance,
        )
        _plot_distance_by_demo(
            best_distance_eval_df=best_distance_eval_df,
            metric=metric,
            out_path=out_dir / f"distance_{metric}_by_eval_demo.svg",
            train_max_distance=train_max_distance,
        )

    _plot_reachable_heatmaps(
        best_distance_eval_df=best_distance_eval_df,
        out_dir=out_dir,
        train_max_distance=train_max_distance,
    )


def _plot_sweep_vs_train_distance(
    *,
    best_df: pd.DataFrame,
    distance_eval_demo_df: pd.DataFrame,
    out_root: Path,
) -> None:
    frames = []
    for train_max_distance in sorted(best_df["train_max_distance"].dropna().astype(int).unique()):
        best_slice = filter_distance_eval_df_to_best(
            distance_eval_demo_df,
            best_df,
            int(train_max_distance),
        )
        ood_by_demo = _compute_ood_by_demo(
            best_distance_eval_df=best_slice,
            train_max_distance=int(train_max_distance),
        )
        if ood_by_demo.empty:
            continue
        frames.append(ood_by_demo)

    if not frames:
        return

    ood_all = pd.concat(frames, ignore_index=True)
    ood_all.to_csv(out_root / "ood_by_eval_demo_all_train_max.csv", index=False)

    for metric in SWEEP_METRICS:
        if metric not in ood_all.columns or ood_all[metric].notna().sum() == 0:
            continue
        g = sns.relplot(
            data=ood_all,
            kind="line",
            x="eval_max_n_demos",
            y=metric,
            hue="train_max_distance",
            col="model_family",
            marker="o",
            facet_kws={"sharex": True, "sharey": False},
            height=3.3,
            aspect=1.15,
        )
        for ax in np.ravel(g.axes):
            ax.set_xscale("symlog", linthresh=1)
        g.set_axis_labels("Eval max_n_demos", metric)
        g.set_titles("{col_name}")
        g.fig.suptitle(f"OOD {metric} vs eval demos across train distances")
        g.fig.subplots_adjust(top=0.84)
        g.savefig(out_root / f"ood_{metric}_vs_eval_demo_across_train_max.svg", bbox_inches="tight")
        plt.close(g.fig)


df = collate_dfs("remote/8_fol_icl/set", show_progress=True)
if len(df) == 0:
    raise ValueError("No results found in remote/8_fol_icl/set")

df = df.reset_index(drop=True).copy()
df["__row_id"] = np.arange(len(df), dtype=np.int64)

final_df = df.apply(_extract_final_row, axis=1).reset_index(drop=True)
distance_eval_demo_df = _explode_distance_eval_demo_rows(df)
best_df = select_best_by_family_and_train_max(final_df)

final_df.to_csv(OUT_DIR / "summary_final.csv", index=False)
distance_eval_demo_df.to_csv(OUT_DIR / "summary_by_distance_eval_demo.csv", index=False)
best_df.to_csv(OUT_DIR / "best_by_family_and_train_max.csv", index=False)

train_max_values = sorted(int(v) for v in best_df["train_max_distance"].dropna().astype(int).unique())
for train_max_distance in train_max_values:
    _save_plots_for_train_max(
        train_max_distance=train_max_distance,
        best_df=best_df,
        distance_eval_demo_df=distance_eval_demo_df,
        out_root=OUT_DIR,
    )

_plot_sweep_vs_train_distance(
    best_df=best_df,
    distance_eval_demo_df=distance_eval_demo_df,
    out_root=OUT_DIR,
)

print("Saved:", OUT_DIR)
