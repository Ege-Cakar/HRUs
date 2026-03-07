"""Analysis script for 11_fresh_rule_split_full_completion."""

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
OUT_DIR = Path("fig/11_fresh_rule_split_full_completion")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROLE_EVAL_DEMO_METRIC_COLS = [
    "loss",
    "token_acc",
    "final_token_acc",
    "seq_exact_acc",
    "n_completion_examples",
    "n_completion_success",
    "n_completion_failure_decode_error",
    "n_completion_failure_unknown_rule_error",
    "n_completion_failure_inapplicable_rule_error",
    "n_completion_failure_goal_not_reached",
    "completion_success_rate",
    "completion_decode_error_rate",
    "completion_unknown_rule_error_rate",
    "completion_inapplicable_rule_error_rate",
    "completion_goal_not_reached_rate",
    "completion_avg_steps",
    "rollout_n_examples",
    "rollout_success_rate",
    "rollout_decode_error_rate",
    "rollout_unknown_rule_error_rate",
    "rollout_inapplicable_rule_error_rate",
    "rollout_goal_not_reached_rate",
    "rollout_avg_steps",
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
        "title": "Token metrics by eval demos",
        "metrics": ["token_acc", "final_token_acc", "seq_exact_acc"],
        "sharey": False,
    },
    {
        "filename": "role_demo_metrics_completion_rates.svg",
        "title": "Completion-path rates by eval demos",
        "metrics": [
            "completion_success_rate",
            "completion_decode_error_rate",
            "completion_unknown_rule_error_rate",
            "completion_inapplicable_rule_error_rate",
            "completion_goal_not_reached_rate",
            "completion_avg_steps",
        ],
        "sharey": False,
    },
    {
        "filename": "role_demo_metrics_rollout_rates.svg",
        "title": "Rollout rates by eval demos",
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


def _extract_mid_pred(row, *, info: dict | None = None) -> float:
    info = (row.get("info", {}) or {}) if info is None else info
    fresh_cfg = row.get("fresh_icl_config", {}) or {}
    for value in (row.get("mid_pred"), info.get("mid_pred"), fresh_cfg.get("mid_pred")):
        if value is None:
            continue
        try:
            return float(int(value))
        except (TypeError, ValueError):
            continue
    return np.nan


def _extract_task_shape_idx(row, *, info: dict | None = None) -> float:
    info = (row.get("info", {}) or {}) if info is None else info
    for value in (row.get("task_shape_idx"), info.get("task_shape_idx")):
        if value is None:
            continue
        try:
            return float(int(value))
        except (TypeError, ValueError):
            continue
    return np.nan


def _extract_task_shape_tag(row, *, info: dict | None = None) -> str | None:
    info = (row.get("info", {}) or {}) if info is None else info
    for value in (row.get("task_shape_tag"), info.get("task_shape_tag")):
        if value is not None:
            return str(value)
    return None


def _extract_final_row(row):
    info = row.get("info", {}) or {}
    final = row.get("metrics_final", {}) or {}
    return pd.Series(
        {
            "run_row_id": row.get("__row_id"),
            "run_id": row.get("run_id"),
            "name": row.get("name"),
            "model_family": row.get("model_family"),
            "task_shape_idx": _extract_task_shape_idx(row, info=info),
            "task_shape_tag": _extract_task_shape_tag(row, info=info),
            "mid_pred": _extract_mid_pred(row, info=info),
            "train_iters": _extract_train_iters(row, info=info),
            "selection_role": row.get("selection_role"),
            "selection_eval_max_n_demos": row.get("selection_eval_max_n_demos", np.nan),
            "selection_metric_name": row.get("selection_metric_name"),
            "selection_metric_value": row.get("selection_metric_value", np.nan),
            "eval_rollout_success_rate": row.get("eval_rollout_success_rate", np.nan),
            "train_rollout_success_rate": row.get("train_rollout_success_rate", np.nan),
            "target_format": info.get("target_format"),
            "task_split": info.get("task_split"),
            "completion_format": info.get("completion_format"),
            "eval_roles": str(info.get("eval_roles")),
            "distance_range": str(_as_int_list(info.get("distance_range"))),
            "predicates_per_layer": str(info.get("predicates_per_layer")),
            "rules_per_transition": str(info.get("rules_per_transition")),
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
            "grad_accum_steps": info.get("grad_accum_steps", np.nan),
            "microbatch_size": info.get("microbatch_size", np.nan),
            "effective_batch_size": info.get("effective_batch_size", np.nan),
            "loss": final.get("loss", np.nan),
            "token_acc": final.get("token_acc", np.nan),
            "final_token_acc": final.get("final_token_acc", np.nan),
            "seq_exact_acc": final.get("seq_exact_acc", np.nan),
        }
    )


def _explode_role_eval_demo_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        info = row.get("info", {}) or {}
        by_role = row.get("metrics_by_role_eval_demo", {}) or {}
        train_iters = _extract_train_iters(row, info=info)
        mid_pred = _extract_mid_pred(row, info=info)
        task_shape_idx = _extract_task_shape_idx(row, info=info)
        task_shape_tag = _extract_task_shape_tag(row, info=info)
        for role, by_demo in by_role.items():
            for eval_demo, metrics in (by_demo or {}).items():
                metrics = metrics or {}
                out = {
                    "run_row_id": row.get("__row_id"),
                    "run_id": row.get("run_id"),
                    "name": row.get("name"),
                    "model_family": row.get("model_family"),
                    "task_shape_idx": task_shape_idx,
                    "task_shape_tag": task_shape_tag,
                    "mid_pred": mid_pred,
                    "eval_role": str(role),
                    "eval_max_n_demos": int(eval_demo),
                    "train_iters": train_iters,
                    "predicates_per_layer": str(info.get("predicates_per_layer")),
                    "rules_per_transition": str(info.get("rules_per_transition")),
                    "lr": info.get("lr", np.nan),
                    "n_layers": info.get("n_layers", np.nan),
                    "n_hidden": info.get("n_hidden", np.nan),
                }
                for metric in ROLE_EVAL_DEMO_METRIC_COLS:
                    out[metric] = metrics.get(metric, np.nan)
                rows.append(out)
    return pd.DataFrame(rows)


def _remove_top_level_figures(out_dir: Path) -> None:
    for path in out_dir.iterdir():
        if path.is_file() and path.suffix.lower() in {".svg", ".png", ".pdf"}:
            path.unlink()


def _plot_role_metric_group(
    *,
    role_eval_demo_df: pd.DataFrame,
    eval_role: str,
    metric_names: list[str],
    out_path: Path,
    title: str,
    sharey: bool = False,
) -> None:
    plot_df = role_eval_demo_df.loc[role_eval_demo_df["eval_role"] == str(eval_role)].copy()
    available = [metric for metric in metric_names if metric in plot_df.columns]
    if plot_df.empty or not available:
        return

    long_df = plot_df[
        ["model_family", "mid_pred", "eval_max_n_demos", *available]
    ].melt(
        id_vars=["model_family", "mid_pred", "eval_max_n_demos"],
        value_vars=available,
        var_name="metric",
        value_name="value",
    )
    long_df = long_df.dropna(subset=["value", "mid_pred"])
    if long_df.empty:
        return

    long_df["mid_pred"] = long_df["mid_pred"].astype(int)
    long_df["mid_pred_label"] = long_df["mid_pred"].astype(str)
    hue_order = [str(v) for v in sorted(long_df["mid_pred"].unique())]

    g = sns.relplot(
        data=long_df,
        kind="line",
        x="eval_max_n_demos",
        y="value",
        hue="mid_pred_label",
        style="model_family",
        markers=True,
        dashes=True,
        estimator="mean",
        errorbar=None,
        col="metric",
        col_wrap=min(3, max(1, len(available))),
        facet_kws={"sharex": True, "sharey": bool(sharey)},
        height=3.0,
        aspect=1.15,
        hue_order=hue_order,
    )
    for ax in np.ravel(g.axes):
        ax.set_xlim(left=0.0)
    g.set_axis_labels("Eval max_n_demos", "Metric value")
    g.set_titles("{col_name}")
    legend = g._legend
    if legend is not None:
        legend.set_title("mid_pred / family")
    g.fig.suptitle(title)
    g.fig.subplots_adjust(top=0.88)
    g.savefig(out_path, bbox_inches="tight")
    plt.close(g.fig)


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
        "eval_rollout_success_rate",
        "train_rollout_success_rate",
        "loss",
        "token_acc",
        "final_token_acc",
        "seq_exact_acc",
    ]
    final_agg = (
        final_slice.groupby(["train_iters", "model_family", "mid_pred"], as_index=False)
        .agg({metric: "mean" for metric in final_agg_metrics if metric in final_slice.columns})
        .sort_values(["model_family", "mid_pred"])
        .reset_index(drop=True)
    )
    final_agg.to_csv(out_dir / "summary_final_aggregated_by_family_mid_pred.csv", index=False)

    role_metric_cols = [metric for metric in ROLE_EVAL_DEMO_METRIC_COLS if metric in role_slice.columns]
    role_agg = (
        role_slice.groupby(
            ["train_iters", "model_family", "mid_pred", "eval_role", "eval_max_n_demos"],
            as_index=False,
        )
        .agg({metric: "mean" for metric in role_metric_cols})
        .sort_values(["eval_role", "model_family", "mid_pred", "eval_max_n_demos"])
        .reset_index(drop=True)
    )
    role_agg.to_csv(out_dir / "summary_by_role_eval_demo_aggregated.csv", index=False)

    for group in ROLE_METRIC_PLOT_GROUPS:
        for eval_role in ("train", "eval"):
            _plot_role_metric_group(
                role_eval_demo_df=role_agg,
                eval_role=eval_role,
                metric_names=list(group["metrics"]),
                out_path=out_dir / f"{eval_role}_{group['filename']}",
                title=f"{str(eval_role).capitalize()}: {str(group['title'])} (train_iters={int(train_iters)})",
                sharey=bool(group["sharey"]),
            )


df = collate_dfs("remote/11_fresh_rule_split_full_completion/set", show_progress=True)
if df.empty:
    raise ValueError("No results found in remote/11_fresh_rule_split_full_completion/set")

_remove_top_level_figures(OUT_DIR)

df = df.reset_index(drop=True)
df["__row_id"] = np.arange(len(df))

final_df = df.apply(_extract_final_row, axis=1).reset_index(drop=True)
role_eval_demo_df = _explode_role_eval_demo_rows(df)
selection_df = final_df.sort_values(
    ["model_family", "mid_pred", "selection_metric_value", "train_iters"],
    ascending=[True, True, False, True],
).reset_index(drop=True)

final_df.to_csv(OUT_DIR / "summary_final.csv", index=False)
role_eval_demo_df.to_csv(OUT_DIR / "summary_by_role_eval_demo.csv", index=False)
selection_df.to_csv(OUT_DIR / "selection_ranked.csv", index=False)

train_iters_values = sorted(int(v) for v in final_df["train_iters"].dropna().astype(int).unique())
for train_iters in train_iters_values:
    _save_aggregates_for_train_iters(
        train_iters=train_iters,
        final_df=final_df,
        role_eval_demo_df=role_eval_demo_df,
        out_root=OUT_DIR,
    )

print("Saved:", OUT_DIR)
