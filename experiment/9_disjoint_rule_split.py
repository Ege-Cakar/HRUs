"""Analysis script for 9_disjoint_rule_split."""

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
OUT_DIR = Path("fig/9_disjoint_rule_split")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROLE_EVAL_DEMO_METRIC_COLS = [
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
    "first_transition_n_examples",
    "first_transition_n_valid_rule",
    "first_transition_n_invalid_rule",
    "first_transition_n_reachable_rule",
    "first_transition_n_decode_error",
    "first_transition_n_unknown_rule_error",
    "first_transition_n_wrong_rule_error",
    "first_transition_n_correct_rule",
    "first_transition_rule_valid_rate",
    "first_transition_rule_reachable_rate",
    "first_transition_rule_reachable_given_valid_rate",
    "first_transition_correct_rule_rate",
    "first_transition_decode_error_rate",
    "first_transition_unknown_rule_error_rate",
    "first_transition_wrong_rule_error_rate",
    "rollout_n_examples",
    "rollout_success_rate",
    "rollout_decode_error_rate",
    "rollout_unknown_rule_error_rate",
    "rollout_inapplicable_rule_error_rate",
    "rollout_goal_not_reached_rate",
    "rollout_avg_steps",
]

SWEEP_METRICS = [
    "first_transition_rule_reachable_rate",
    "rollout_success_rate",
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


def _extract_final_row(row):
    info = row.get("info", {}) or {}
    final = row.get("metrics_final", {}) or {}

    return pd.Series(
        {
            "run_row_id": row.get("__row_id"),
            "run_id": row.get("run_id"),
            "name": row.get("name"),
            "model_family": row.get("model_family"),
            "selection_role": row.get("selection_role"),
            "selection_eval_max_n_demos": row.get("selection_eval_max_n_demos", np.nan),
            "selection_metric_name": row.get("selection_metric_name"),
            "selection_metric_value": row.get("selection_metric_value", np.nan),
            "eval_first_transition_rule_reachable_rate": row.get(
                "eval_first_transition_rule_reachable_rate", np.nan
            ),
            "eval_first_transition_rule_valid_rate": row.get(
                "eval_first_transition_rule_valid_rate", np.nan
            ),
            "eval_rollout_success_rate": row.get("eval_rollout_success_rate", np.nan),
            "train_first_transition_rule_reachable_rate": row.get(
                "train_first_transition_rule_reachable_rate", np.nan
            ),
            "train_rollout_success_rate": row.get("train_rollout_success_rate", np.nan),
            "task_split": info.get("task_split"),
            "eval_roles": str(info.get("eval_roles")),
            "distance_range": str(_as_int_list(info.get("distance_range"))),
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
        }
    )


def _explode_role_eval_demo_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        info = row.get("info", {}) or {}
        by_role = row.get("metrics_by_role_eval_demo", {}) or {}
        for role, by_demo in by_role.items():
            by_demo = by_demo or {}
            for eval_demo, metrics in by_demo.items():
                metrics = metrics or {}
                out = {
                    "run_row_id": row.get("__row_id"),
                    "run_id": row.get("run_id"),
                    "name": row.get("name"),
                    "model_family": row.get("model_family"),
                    "eval_role": str(role),
                    "eval_max_n_demos": int(eval_demo),
                    "lr": info.get("lr", np.nan),
                    "n_layers": info.get("n_layers", np.nan),
                    "n_hidden": info.get("n_hidden", np.nan),
                }
                for metric_name in ROLE_EVAL_DEMO_METRIC_COLS:
                    out[metric_name] = metrics.get(metric_name, np.nan)
                rows.append(out)

    return pd.DataFrame(rows)


def select_best_by_family(final_df: pd.DataFrame) -> pd.DataFrame:
    ranked = final_df.sort_values(
        ["model_family", "selection_metric_value", "run_row_id"],
        ascending=[True, False, True],
    )
    return (
        ranked.groupby(["model_family"], as_index=False, sort=True)
        .head(1)
        .reset_index(drop=True)
    )


def filter_role_eval_demo_df_to_best(
    role_eval_demo_df: pd.DataFrame,
    best_df: pd.DataFrame,
) -> pd.DataFrame:
    best_ids = best_df["run_row_id"]
    return role_eval_demo_df.loc[role_eval_demo_df["run_row_id"].isin(best_ids)].copy()


def _plot_eval_demo_lines(
    *,
    best_role_eval_demo_df: pd.DataFrame,
    metric: str,
    out_path: Path,
) -> None:
    plot_df = best_role_eval_demo_df.loc[
        best_role_eval_demo_df["eval_role"] == "eval"
    ].copy()
    if plot_df.empty or metric not in plot_df.columns:
        return

    plt.figure(figsize=(6.5, 4.0))
    sns.lineplot(
        data=plot_df,
        x="eval_max_n_demos",
        y=metric,
        hue="model_family",
        marker="o",
    )
    plt.xscale("symlog", linthresh=1)
    plt.xlabel("Eval max_n_demos")
    plt.ylabel(metric)
    plt.title(f"Eval role: {metric} vs demos")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def _plot_transfer_gap(
    *,
    best_role_eval_demo_df: pd.DataFrame,
    metric: str,
    out_path: Path,
) -> None:
    if best_role_eval_demo_df.empty or metric not in best_role_eval_demo_df.columns:
        return

    pivot = best_role_eval_demo_df.pivot_table(
        index=["model_family", "eval_max_n_demos"],
        columns="eval_role",
        values=metric,
        aggfunc="mean",
    ).reset_index()
    if "train" not in pivot.columns or "eval" not in pivot.columns:
        return

    pivot[f"{metric}_gap_eval_minus_train"] = pivot["eval"] - pivot["train"]

    plt.figure(figsize=(6.5, 4.0))
    sns.lineplot(
        data=pivot,
        x="eval_max_n_demos",
        y=f"{metric}_gap_eval_minus_train",
        hue="model_family",
        marker="o",
    )
    plt.xscale("symlog", linthresh=1)
    plt.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0)
    plt.xlabel("Eval max_n_demos")
    plt.ylabel("eval - train")
    plt.title(f"Transfer gap for {metric}")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def _plot_eval_heatmap(
    *,
    best_role_eval_demo_df: pd.DataFrame,
    metric: str,
    out_path: Path,
) -> None:
    plot_df = best_role_eval_demo_df.loc[
        best_role_eval_demo_df["eval_role"] == "eval"
    ].copy()
    if plot_df.empty or metric not in plot_df.columns:
        return

    pivot = plot_df.pivot_table(
        index="model_family",
        columns="eval_max_n_demos",
        values=metric,
        aggfunc="mean",
    ).sort_index(axis=0).sort_index(axis=1)
    if pivot.empty:
        return

    plt.figure(figsize=(7.2, 3.8))
    sns.heatmap(
        pivot,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": metric},
    )
    plt.xlabel("Eval max_n_demos")
    plt.ylabel("Model family")
    plt.title(f"Eval role heatmap: {metric}")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


df = collate_dfs("remote/9_disjoint_rule_split/set", show_progress=True)
if len(df) == 0:
    raise ValueError("No results found in remote/9_disjoint_rule_split/set")

df = df.reset_index(drop=True).copy()
df["__row_id"] = np.arange(len(df), dtype=np.int64)

final_df = df.apply(_extract_final_row, axis=1).reset_index(drop=True)
role_eval_demo_df = _explode_role_eval_demo_rows(df)
best_df = select_best_by_family(final_df)
best_role_eval_demo_df = filter_role_eval_demo_df_to_best(role_eval_demo_df, best_df)

final_df.to_csv(OUT_DIR / "summary_final.csv", index=False)
role_eval_demo_df.to_csv(OUT_DIR / "summary_by_role_eval_demo.csv", index=False)
best_df.to_csv(OUT_DIR / "best_by_family.csv", index=False)
best_role_eval_demo_df.to_csv(OUT_DIR / "best_by_role_eval_demo.csv", index=False)

for metric in SWEEP_METRICS:
    _plot_eval_demo_lines(
        best_role_eval_demo_df=best_role_eval_demo_df,
        metric=metric,
        out_path=OUT_DIR / f"eval_{metric}_vs_demo.svg",
    )
    _plot_transfer_gap(
        best_role_eval_demo_df=best_role_eval_demo_df,
        metric=metric,
        out_path=OUT_DIR / f"gap_{metric}_eval_minus_train.svg",
    )

_plot_eval_heatmap(
    best_role_eval_demo_df=best_role_eval_demo_df,
    metric="first_transition_rule_reachable_rate",
    out_path=OUT_DIR / "eval_reachable_rate_heatmap.svg",
)

print("Saved:", OUT_DIR)
