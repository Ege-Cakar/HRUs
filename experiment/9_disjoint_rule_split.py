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

ROLE_METRIC_PLOT_GROUPS = [
    {
        "filename": "role_demo_metrics_loss.svg",
        "title": "Loss by eval demos",
        "metrics": ["loss"],
        "sharey": False,
    },
    {
        "filename": "role_demo_metrics_accuracy.svg",
        "title": "Accuracy metrics by eval demos",
        "metrics": ["token_acc", "final_token_acc", "seq_exact_acc"],
        "sharey": False,
    },
    {
        "filename": "role_demo_metrics_rule_match_rates.svg",
        "title": "Rule-match rates by eval demos",
        "metrics": [
            "valid_rule_rate",
            "invalid_rule_rate",
            "correct_rule_rate",
            "correct_given_valid_rate",
            "decode_error_rate",
            "unknown_rule_error_rate",
            "wrong_rule_error_rate",
        ],
        "sharey": False,
    },
    {
        "filename": "role_demo_metrics_first_transition_rates.svg",
        "title": "First-transition rates by eval demos",
        "metrics": [
            "first_transition_rule_valid_rate",
            "first_transition_rule_reachable_rate",
            "first_transition_rule_reachable_given_valid_rate",
            "first_transition_correct_rule_rate",
            "first_transition_decode_error_rate",
            "first_transition_unknown_rule_error_rate",
            "first_transition_wrong_rule_error_rate",
        ],
        "sharey": False,
    },
    {
        "filename": "role_demo_metrics_rollout_rates.svg",
        "title": "Rollout rates and steps by eval demos",
        "metrics": [
            "rollout_success_rate",
            "rollout_decode_error_rate",
            "rollout_unknown_rule_error_rate",
            "rollout_inapplicable_rule_error_rate",
            "rollout_goal_not_reached_rate",
            "rollout_avg_steps",
        ],
        "sharey": False,
    },
    {
        "filename": "role_demo_metrics_rule_match_counts.svg",
        "title": "Rule-match counts by eval demos",
        "metrics": [
            "n_rule_examples",
            "n_valid_rule",
            "n_invalid_rule",
            "n_correct_rule",
            "n_decode_error",
            "n_unknown_rule_error",
            "n_wrong_rule_error",
        ],
        "sharey": False,
    },
    {
        "filename": "role_demo_metrics_first_transition_counts.svg",
        "title": "First-transition counts by eval demos",
        "metrics": [
            "first_transition_n_examples",
            "first_transition_n_valid_rule",
            "first_transition_n_invalid_rule",
            "first_transition_n_reachable_rule",
            "first_transition_n_decode_error",
            "first_transition_n_unknown_rule_error",
            "first_transition_n_wrong_rule_error",
            "first_transition_n_correct_rule",
        ],
        "sharey": False,
    },
    {
        "filename": "role_demo_metrics_rollout_counts.svg",
        "title": "Rollout counts by eval demos",
        "metrics": ["rollout_n_examples"],
        "sharey": False,
    },
]

COMMON_CONFIG_COLS = [
    "target_format",
    "task_split",
    "eval_roles",
    "distance_range",
    "train_max_n_demos",
    "eval_max_n_demos_sweep",
    "train_iters",
    "lr",
    "n_layers",
    "n_hidden",
    "n_seq",
    "n_vocab",
]

FAMILY_CONFIG_COLS = {
    "transformer": ["n_heads", "pos_encoding", "use_swiglu"],
    "mamba1": ["n_heads", "d_state", "d_conv", "scan_chunk_len"],
    "mamba2": ["n_heads", "d_state", "d_conv", "scan_chunk_len"],
    "mamba2_bonsai": ["n_heads", "d_state", "d_conv", "scan_chunk_len"],
}


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


def _extract_train_iters(row, *, info: dict | None = None) -> float:
    info = (row.get("info", {}) or {}) if info is None else info
    train_args = row.get("train_args", {}) or {}
    val = train_args.get("train_iters", info.get("train_iters"))
    if val is None:
        return np.nan
    try:
        return float(int(val))
    except (TypeError, ValueError):
        return np.nan


def _extract_final_row(row):
    info = row.get("info", {}) or {}
    final = row.get("metrics_final", {}) or {}

    return pd.Series(
        {
            "run_row_id": row.get("__row_id"),
            "run_id": row.get("run_id"),
            "name": row.get("name"),
            "model_family": row.get("model_family"),
            "train_iters": _extract_train_iters(row, info=info),
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
            "target_format": info.get("target_format"),
            "task_split": info.get("task_split"),
            "eval_roles": str(info.get("eval_roles")),
            "distance_range": str(_as_int_list(info.get("distance_range"))),
            "train_max_n_demos": info.get("train_max_n_demos", np.nan),
            "eval_max_n_demos_sweep": str(_as_int_list(info.get("eval_max_n_demos_sweep"))),
            "lr": info.get("lr", np.nan),
            "n_layers": info.get("n_layers", np.nan),
            "n_hidden": info.get("n_hidden", np.nan),
            "n_heads": info.get("n_heads", np.nan),
            "pos_encoding": info.get("pos_encoding"),
            "use_swiglu": info.get("use_swiglu"),
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
        train_iters = _extract_train_iters(row, info=info)
        for role, by_demo in by_role.items():
            by_demo = by_demo or {}
            for eval_demo, metrics in by_demo.items():
                metrics = metrics or {}
                out = {
                    "run_row_id": row.get("__row_id"),
                    "run_id": row.get("run_id"),
                    "name": row.get("name"),
                    "model_family": row.get("model_family"),
                    "train_iters": train_iters,
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


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict, np.ndarray)):
        return False
    try:
        return bool(pd.isna(value))
    except TypeError:
        return False


def _format_md_value(value) -> str:
    if _is_missing(value):
        return "-"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (np.floating, float)):
        return f"{float(value):.6g}"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    return str(value)


def _write_best_config_markdown(
    *,
    best_rows: pd.DataFrame,
    out_path: Path,
) -> None:
    lines = [
        "# Best configurations by model family",
        "",
        "Selection metric: `eval_first_transition_rule_reachable_rate` (higher is better).",
        "",
    ]

    id_metric_cols = [
        "run_id",
        "run_row_id",
        "name",
        "model_family",
        "selection_role",
        "selection_eval_max_n_demos",
        "selection_metric_name",
        "selection_metric_value",
        "eval_first_transition_rule_reachable_rate",
        "eval_first_transition_rule_valid_rate",
        "eval_rollout_success_rate",
        "train_first_transition_rule_reachable_rate",
        "train_rollout_success_rate",
    ]

    for _, row in best_rows.iterrows():
        family = str(row.get("model_family"))
        lines.extend(
            [
                f"## {family}",
                "",
                "| Key | Value |",
                "| --- | --- |",
            ]
        )

        cols = list(id_metric_cols)
        cols.extend(COMMON_CONFIG_COLS)
        cols.extend(FAMILY_CONFIG_COLS.get(family, []))

        for col in cols:
            val = row.get(col)
            if _is_missing(val):
                continue
            lines.append(f"| `{col}` | `{_format_md_value(val)}` |")

        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def _plot_role_metric_group(
    *,
    best_role_eval_demo_df: pd.DataFrame,
    metric_names: list[str],
    out_path: Path,
    title: str,
    sharey: bool = False,
) -> None:
    available = [m for m in metric_names if m in best_role_eval_demo_df.columns]
    if not available:
        return

    long_df = best_role_eval_demo_df[
        ["model_family", "eval_role", "eval_max_n_demos", *available]
    ].melt(
        id_vars=["model_family", "eval_role", "eval_max_n_demos"],
        value_vars=available,
        var_name="metric",
        value_name="value",
    )
    long_df = long_df.dropna(subset=["value"])
    if long_df.empty:
        return

    g = sns.relplot(
        data=long_df,
        kind="line",
        x="eval_max_n_demos",
        y="value",
        hue="model_family",
        style="eval_role",
        col="metric",
        marker="o",
        col_wrap=min(3, max(1, len(available))),
        facet_kws={"sharex": True, "sharey": bool(sharey)},
        height=3.0,
        aspect=1.15,
    )
    for ax in np.ravel(g.axes):
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlim(left=0.0)
    g.set_axis_labels("Eval max_n_demos", "Metric value")
    g.set_titles("{col_name}")
    g.fig.suptitle(title)
    g.fig.subplots_adjust(top=0.88)
    g.savefig(out_path, bbox_inches="tight")
    plt.close(g.fig)


def _plot_role_demo_lines(
    *,
    best_role_eval_demo_df: pd.DataFrame,
    metric: str,
    out_path: Path,
    eval_role: str = "eval",
) -> None:
    plot_df = best_role_eval_demo_df.loc[
        best_role_eval_demo_df["eval_role"] == str(eval_role)
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
    plt.xlim(left=0.0)
    plt.xlabel("Eval max_n_demos")
    plt.ylabel(metric)
    plt.title(f"{str(eval_role).capitalize()} role: {metric} vs demos")
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
    plt.xlim(left=0.0)
    plt.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0)
    plt.xlabel("Eval max_n_demos")
    plt.ylabel("eval - train")
    plt.title(f"Transfer gap for {metric}")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def _plot_role_heatmap(
    *,
    best_role_eval_demo_df: pd.DataFrame,
    metric: str,
    out_path: Path,
    eval_role: str = "eval",
) -> None:
    plot_df = best_role_eval_demo_df.loc[
        best_role_eval_demo_df["eval_role"] == str(eval_role)
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
    plt.title(f"{str(eval_role).capitalize()} role heatmap: {metric}")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def _save_aggregates_for_train_iters(
    *,
    train_iters: int,
    final_df: pd.DataFrame,
    role_eval_demo_df: pd.DataFrame,
    out_root: Path,
) -> None:
    out_dir = out_root / f"train_iters_{int(train_iters)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    final_slice = final_df.loc[final_df["train_iters"] == float(train_iters)].copy()
    role_slice = role_eval_demo_df.loc[
        role_eval_demo_df["train_iters"] == float(train_iters)
    ].copy()
    if final_slice.empty or role_slice.empty:
        return

    final_agg_metrics = [
        "selection_metric_value",
        "eval_first_transition_rule_reachable_rate",
        "eval_first_transition_rule_valid_rate",
        "eval_rollout_success_rate",
        "train_first_transition_rule_reachable_rate",
        "train_rollout_success_rate",
        "loss",
        "final_token_acc",
        "seq_exact_acc",
        "token_acc",
    ]
    final_agg_metrics = [metric for metric in final_agg_metrics if metric in final_slice.columns]
    final_agg = (
        final_slice.groupby(["train_iters", "model_family"], as_index=False)
        .agg({metric: "mean" for metric in final_agg_metrics})
        .sort_values(["model_family"])
        .reset_index(drop=True)
    )
    final_agg.to_csv(out_dir / "summary_final_aggregated_by_family.csv", index=False)

    role_metric_cols = [metric for metric in ROLE_EVAL_DEMO_METRIC_COLS if metric in role_slice.columns]
    role_agg = (
        role_slice.groupby(
            ["train_iters", "model_family", "eval_role", "eval_max_n_demos"],
            as_index=False,
        )
        .agg({metric: "mean" for metric in role_metric_cols})
        .sort_values(["model_family", "eval_role", "eval_max_n_demos"])
        .reset_index(drop=True)
    )
    role_agg.to_csv(out_dir / "summary_by_role_eval_demo_aggregated.csv", index=False)

    for group in ROLE_METRIC_PLOT_GROUPS:
        _plot_role_metric_group(
            best_role_eval_demo_df=role_agg,
            metric_names=list(group["metrics"]),
            out_path=out_dir / str(group["filename"]),
            title=f"{str(group['title'])} (train_iters={int(train_iters)})",
            sharey=bool(group["sharey"]),
        )

    for metric in SWEEP_METRICS:
        for role in ("train", "eval"):
            _plot_role_demo_lines(
                best_role_eval_demo_df=role_agg,
                metric=metric,
                out_path=out_dir / f"{role}_{metric}_vs_demo.svg",
                eval_role=role,
            )
        _plot_transfer_gap(
            best_role_eval_demo_df=role_agg,
            metric=metric,
            out_path=out_dir / f"gap_{metric}_eval_minus_train.svg",
        )

    for role in ("train", "eval"):
        _plot_role_heatmap(
            best_role_eval_demo_df=role_agg,
            metric="first_transition_rule_reachable_rate",
            out_path=out_dir / f"{role}_reachable_rate_heatmap.svg",
            eval_role=role,
        )

    _plot_role_heatmap(
        best_role_eval_demo_df=role_agg,
        metric="rollout_success_rate",
        out_path=out_dir / "eval_rollout_success_heatmap.svg",
        eval_role="eval",
    )


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
_write_best_config_markdown(
    best_rows=best_df.sort_values("model_family").reset_index(drop=True),
    out_path=OUT_DIR / "best_configs_by_family.md",
)

for group in ROLE_METRIC_PLOT_GROUPS:
    _plot_role_metric_group(
        best_role_eval_demo_df=best_role_eval_demo_df,
        metric_names=list(group["metrics"]),
        out_path=OUT_DIR / str(group["filename"]),
        title=str(group["title"]),
        sharey=bool(group["sharey"]),
    )

for metric in SWEEP_METRICS:
    for role in ("train", "eval"):
        _plot_role_demo_lines(
            best_role_eval_demo_df=best_role_eval_demo_df,
            metric=metric,
            out_path=OUT_DIR / f"{role}_{metric}_vs_demo.svg",
            eval_role=role,
        )
    _plot_transfer_gap(
        best_role_eval_demo_df=best_role_eval_demo_df,
        metric=metric,
        out_path=OUT_DIR / f"gap_{metric}_eval_minus_train.svg",
    )

for role in ("train", "eval"):
    _plot_role_heatmap(
        best_role_eval_demo_df=best_role_eval_demo_df,
        metric="first_transition_rule_reachable_rate",
        out_path=OUT_DIR / f"{role}_reachable_rate_heatmap.svg",
        eval_role=role,
    )

_plot_role_heatmap(
    best_role_eval_demo_df=best_role_eval_demo_df,
    metric="rollout_success_rate",
    out_path=OUT_DIR / "eval_rollout_success_heatmap.svg",
    eval_role="eval",
)

train_iters_values = sorted(int(v) for v in final_df["train_iters"].dropna().astype(int).unique())
for train_iters in train_iters_values:
    _save_aggregates_for_train_iters(
        train_iters=train_iters,
        final_df=final_df,
        role_eval_demo_df=role_eval_demo_df,
        out_root=OUT_DIR,
    )

print("Saved:", OUT_DIR)
