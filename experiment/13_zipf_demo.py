"""Analysis script for 13_zipf_demo."""

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
from model.compute import (
    compute_metrics_from_info,
    memory_bytes_estimate,
    training_flops_total,
)


set_theme()
OUT_DIR = Path("fig/13_zipf_demo")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SELECTION_EVAL_MAX_N_DEMOS = 8

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
    "rollout_n_examples",
    "rollout_success_rate",
    "rollout_decode_error_rate",
    "rollout_unknown_rule_error_rate",
    "rollout_wrong_rule_error_rate",
    "rollout_inapplicable_rule_error_rate",
    "rollout_goal_not_reached_rate",
    "rollout_avg_steps",
    "rollout_step0_reachable_rate",
    "rollout_step1_reachable_rate",
    "rollout_step0_valid_count",
    "rollout_step1_valid_count",
]

ROLE_METRIC_PLOT_GROUPS = [
    {
        "filename": "role_demo_metrics_loss.png",
        "title": "Loss by eval demos",
        "metrics": ["loss"],
        "sharey": False,
    },
    {
        "filename": "role_demo_metrics_accuracy.png",
        "title": "Accuracy metrics by eval demos",
        "metrics": ["token_acc", "final_token_acc", "seq_exact_acc"],
        "sharey": False,
    },
    {
        "filename": "role_demo_metrics_rule_match_rates.png",
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
        "filename": "role_demo_metrics_rollout_rates.png",
        "title": "Rollout rates and steps by eval demos",
        "metrics": [
            "rollout_success_rate",
            "rollout_decode_error_rate",
            "rollout_unknown_rule_error_rate",
            "rollout_wrong_rule_error_rate",
            "rollout_inapplicable_rule_error_rate",
            "rollout_goal_not_reached_rate",
            "rollout_avg_steps",
        ],
        "sharey": False,
    },
    {
        "filename": "role_demo_metrics_reachability.png",
        "title": "Per-step reachability by eval demos",
        "metrics": ["rollout_step0_reachable_rate", "rollout_step1_reachable_rate"],
        "sharey": True,
    },
    {
        "filename": "role_demo_metrics_rule_match_counts.png",
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
        "filename": "role_demo_metrics_rollout_counts.png",
        "title": "Rollout counts by eval demos",
        "metrics": ["rollout_n_examples", "rollout_step0_valid_count", "rollout_step1_valid_count"],
        "sharey": False,
    },
]

MODEL_FAMILY_DASHES = {
    "mamba2_bonsai": "",
    "transformer": (2, 2),
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


def _extract_train_alpha(row, *, info: dict | None = None) -> float:
    info = (row.get("info", {}) or {}) if info is None else info
    for value in (row.get("train_alpha"), info.get("train_alpha")):
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return np.nan


def _extract_train_demo_ranked(row, *, info: dict | None = None) -> bool | None:
    info = (row.get("info", {}) or {}) if info is None else info
    for value in (row.get("train_demo_ranked"), info.get("train_demo_ranked")):
        if value is not None:
            return bool(value)
    return None


def _extract_train_max_n_demos(row, *, info: dict | None = None) -> float:
    info = (row.get("info", {}) or {}) if info is None else info
    for value in (row.get("train_max_n_demos"), info.get("train_max_n_demos")):
        if value is not None:
            try:
                return float(int(value))
            except (TypeError, ValueError):
                continue
    return 8.0  # backward compat default


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


def _compute_row_n_params(info):
    try:
        return compute_metrics_from_info(info)["n_params"]
    except (KeyError, ValueError, TypeError):
        return np.nan


def _compute_row_training_flops(info, *, train_iters_val):
    try:
        train_n_seq = info.get("train_fixed_length_n_seq", info.get("n_seq"))
        metrics = compute_metrics_from_info(info, n_seq_override=train_n_seq)
        ti = int(train_iters_val) if not np.isnan(train_iters_val) else 0
        return training_flops_total(
            metrics["forward_flops"],
            train_iters=ti,
            batch_size=int(info.get("microbatch_size", 1)),
            grad_accum_steps=int(info.get("grad_accum_steps", 1)),
        )
    except (KeyError, ValueError, TypeError):
        return np.nan


def _extract_final_row(row):
    info = row.get("info", {}) or {}
    final = row.get("metrics_final", {}) or {}
    train_iters_val = _extract_train_iters(row, info=info)
    n_params = _compute_row_n_params(info)
    row_training_flops = _compute_row_training_flops(info, train_iters_val=train_iters_val)

    return pd.Series(
        {
            "run_row_id": row.get("__row_id"),
            "run_id": row.get("run_id"),
            "name": row.get("name"),
            "model_family": row.get("model_family"),
            "task_shape_idx": _extract_task_shape_idx(row, info=info),
            "task_shape_tag": _extract_task_shape_tag(row, info=info),
            "mid_pred": _extract_mid_pred(row, info=info),
            "train_iters": train_iters_val,
            "train_alpha": _extract_train_alpha(row, info=info),
            "train_demo_ranked": _extract_train_demo_ranked(row, info=info),
            "selection_role": row.get("selection_role"),
            "selection_eval_max_n_demos": row.get("selection_eval_max_n_demos", np.nan),
            "selection_metric_name": row.get("selection_metric_name"),
            "selection_metric_value": row.get("selection_metric_value", np.nan),
            "eval_rollout_success_rate": row.get("eval_rollout_success_rate", np.nan),
            "train_rollout_success_rate": row.get("train_rollout_success_rate", np.nan),
            "target_format": info.get("target_format"),
            "task_split": info.get("task_split"),
            "eval_roles": str(info.get("eval_roles")),
            "distance_range": str(_as_int_list(info.get("distance_range"))),
            "predicates_per_layer": str(info.get("predicates_per_layer")),
            "rules_per_transition": str(info.get("rules_per_transition")),
            "train_max_n_demos": _extract_train_max_n_demos(row, info=info),
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
            "n_params": n_params,
            "training_flops_total": row_training_flops,
            "loss": final.get("loss", np.nan),
            "final_token_acc": final.get("final_token_acc", np.nan),
            "seq_exact_acc": final.get("seq_exact_acc", np.nan),
            "token_acc": final.get("token_acc", np.nan),
        }
    )


def _common_row_fields(row, *, info, train_iters, mid_pred, train_alpha, task_shape_idx, task_shape_tag, train_max_n_demos):
    train_demo_ranked = _extract_train_demo_ranked(row, info=info)
    n_params = _compute_row_n_params(info)
    row_training_flops = _compute_row_training_flops(info, train_iters_val=train_iters)
    return {
        "run_row_id": row.get("__row_id"),
        "run_id": row.get("run_id"),
        "name": row.get("name"),
        "model_family": row.get("model_family"),
        "task_shape_idx": task_shape_idx,
        "task_shape_tag": task_shape_tag,
        "mid_pred": mid_pred,
        "train_iters": train_iters,
        "train_alpha": float(train_alpha),
        "train_demo_ranked": train_demo_ranked,
        "train_max_n_demos": float(train_max_n_demos),
        "lr": info.get("lr", np.nan),
        "n_layers": info.get("n_layers", np.nan),
        "n_hidden": info.get("n_hidden", np.nan),
        "predicates_per_layer": str(info.get("predicates_per_layer")),
        "rules_per_transition": str(info.get("rules_per_transition")),
        "n_params": n_params,
        "training_flops_total": row_training_flops,
    }


def _explode_4level_metrics(
    df: pd.DataFrame,
    *,
    metrics_col: str,
    eval_type: str | None = None,
) -> pd.DataFrame:
    """Explode role -> n_demos -> eval_alpha -> eval_ranked -> metrics."""
    rows = []
    for _, row in df.iterrows():
        info = row.get("info", {}) or {}
        by_role = row.get(metrics_col, {}) or {}
        train_iters = _extract_train_iters(row, info=info)
        mid_pred = _extract_mid_pred(row, info=info)
        train_alpha = _extract_train_alpha(row, info=info)
        task_shape_idx = _extract_task_shape_idx(row, info=info)
        task_shape_tag = _extract_task_shape_tag(row, info=info)
        train_max_n_demos = _extract_train_max_n_demos(row, info=info)
        base = _common_row_fields(
            row, info=info, train_iters=train_iters, mid_pred=mid_pred,
            train_alpha=train_alpha, task_shape_idx=task_shape_idx,
            task_shape_tag=task_shape_tag, train_max_n_demos=train_max_n_demos,
        )
        for role, by_demo in by_role.items():
            for eval_demo, by_alpha in (by_demo or {}).items():
                for eval_alpha, by_ranked in (by_alpha or {}).items():
                    if isinstance(by_ranked, dict) and not any(
                        isinstance(k, bool) for k in by_ranked
                    ):
                        # Legacy 3-level format: by_ranked IS the metrics dict
                        metrics = by_ranked or {}
                        eval_top = metrics.get("eval_top_seq_lens", [])
                        out = {
                            **base,
                            "eval_role": str(role),
                            "eval_max_n_demos": int(eval_demo),
                            "eval_alpha": float(eval_alpha),
                            "eval_demo_ranked": None,
                            "eval_dominant_seq_len": int(eval_top[0]["seq_len"]) if eval_top else np.nan,
                        }
                        if eval_type is not None:
                            out["eval_type"] = eval_type
                        for metric_name in ROLE_EVAL_DEMO_METRIC_COLS:
                            out[metric_name] = metrics.get(metric_name, np.nan)
                        rows.append(out)
                    else:
                        for eval_ranked, metrics in (by_ranked or {}).items():
                            metrics = metrics or {}
                            eval_top = metrics.get("eval_top_seq_lens", [])
                            out = {
                                **base,
                                "eval_role": str(role),
                                "eval_max_n_demos": int(eval_demo),
                                "eval_alpha": float(eval_alpha),
                                "eval_demo_ranked": bool(eval_ranked),
                                "eval_dominant_seq_len": int(eval_top[0]["seq_len"]) if eval_top else np.nan,
                            }
                            if eval_type is not None:
                                out["eval_type"] = eval_type
                            for metric_name in ROLE_EVAL_DEMO_METRIC_COLS:
                                out[metric_name] = metrics.get(metric_name, np.nan)
                            rows.append(out)
    return pd.DataFrame(rows)


def _explode_role_eval_demo_rows(df: pd.DataFrame) -> pd.DataFrame:
    return _explode_4level_metrics(df, metrics_col="metrics_by_role_eval_demo")


def _explode_role_eval_needle_rows(df: pd.DataFrame) -> pd.DataFrame:
    return _explode_4level_metrics(
        df, metrics_col="metrics_by_role_eval_needle", eval_type="needle",
    )


def _explode_role_eval_demo_all_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        info = row.get("info", {}) or {}
        by_role = row.get("metrics_by_role_eval_demo_all", {}) or {}
        train_iters = _extract_train_iters(row, info=info)
        mid_pred = _extract_mid_pred(row, info=info)
        train_alpha = _extract_train_alpha(row, info=info)
        task_shape_idx = _extract_task_shape_idx(row, info=info)
        task_shape_tag = _extract_task_shape_tag(row, info=info)
        train_max_n_demos = _extract_train_max_n_demos(row, info=info)
        base = _common_row_fields(
            row, info=info, train_iters=train_iters, mid_pred=mid_pred,
            train_alpha=train_alpha, task_shape_idx=task_shape_idx,
            task_shape_tag=task_shape_tag, train_max_n_demos=train_max_n_demos,
        )
        for role, metrics in by_role.items():
            metrics = metrics or {}
            out = {
                **base,
                "eval_role": str(role),
                "eval_type": "demo_all",
            }
            for metric_name in ROLE_EVAL_DEMO_METRIC_COLS:
                out[metric_name] = metrics.get(metric_name, np.nan)
            rows.append(out)
    return pd.DataFrame(rows)


def _remove_top_level_figures(out_dir: Path) -> None:
    for path in out_dir.iterdir():
        if path.is_file() and path.suffix.lower() in {".png", ".pdf"}:
            path.unlink()


def _plot_role_metric_group(
    *,
    role_eval_demo_df: pd.DataFrame,
    eval_role: str,
    metric_names: list[str],
    out_path: Path,
    title: str,
    sharey: bool = False,
    train_max_n_demos: int | None = None,
) -> None:
    plot_df = role_eval_demo_df.loc[role_eval_demo_df["eval_role"] == str(eval_role)].copy()
    available = [metric for metric in metric_names if metric in plot_df.columns]
    if plot_df.empty or not available:
        return

    plot_df["facet_label"] = (
        "train_a=" + plot_df["train_alpha"].astype(str)
        + " " + plot_df["train_demo_ranked"].map(
            {True: "ranked", False: "unranked", None: "n/a"}
        ).fillna("n/a")
    )
    plot_df["eval_alpha_label"] = plot_df["eval_alpha"].astype(str)

    long_df = plot_df[
        ["model_family", "train_alpha", "facet_label", "eval_alpha",
         "eval_alpha_label", "eval_max_n_demos", *available]
    ].melt(
        id_vars=["model_family", "train_alpha", "facet_label", "eval_alpha",
                 "eval_alpha_label", "eval_max_n_demos"],
        value_vars=available,
        var_name="metric",
        value_name="value",
    )
    long_df = long_df.dropna(subset=["value"])
    if long_df.empty:
        return

    eval_alpha_order = sorted(long_df["eval_alpha_label"].unique(), key=lambda s: float(s))

    g = sns.relplot(
        data=long_df,
        kind="line",
        x="eval_max_n_demos",
        y="value",
        hue="eval_alpha_label",
        style="model_family",
        row="facet_label",
        col="metric",
        dashes=MODEL_FAMILY_DASHES,
        markers=True,
        estimator="mean",
        errorbar=None,
        facet_kws={"sharex": True, "sharey": bool(sharey)},
        height=3.5,
        aspect=1.3,
        hue_order=eval_alpha_order,
    )
    for ax in np.ravel(g.axes):
        ax.set_xlim(left=0.0)
        if train_max_n_demos is not None:
            ax.axvline(train_max_n_demos, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    g.set_axis_labels("Eval max_n_demos", "Metric value")
    g.set_titles("{row_name} | {col_name}")
    legend = g._legend
    if legend is not None:
        legend.set_title("eval_alpha / family")
    g.fig.suptitle(title, y=1.02)
    g.savefig(out_path, bbox_inches="tight")
    plt.close(g.fig)


def _plot_heatmap(
    *,
    role_eval_demo_df: pd.DataFrame,
    model_family: str,
    train_iters: int,
    mid_pred: int,
    out_dir: Path,
    eval_max_n_demos: int,
    suffix: str = "",
) -> None:
    df = role_eval_demo_df.loc[
        (role_eval_demo_df["eval_role"] == "eval")
        & (role_eval_demo_df["model_family"] == str(model_family))
        & (role_eval_demo_df["eval_max_n_demos"] == int(eval_max_n_demos))
    ].copy()
    if df.empty:
        return

    pivot = df.pivot_table(
        index="train_alpha",
        columns="eval_alpha",
        values="rollout_success_rate",
        aggfunc="mean",
    )
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        ax=ax,
    )
    ax.set_title(f"rollout_success_rate — {model_family}\n(demos={eval_max_n_demos}){suffix}")
    ax.set_xlabel("eval_alpha")
    ax.set_ylabel("train_alpha")
    fig.tight_layout()
    fname = f"heatmap_rollout_success_rate_{model_family}"
    if suffix:
        fname += f"_{suffix.strip().replace(' ', '_')}"
    fig.savefig(
        out_dir / f"{fname}.png",
        bbox_inches="tight",
    )
    plt.close(fig)


def _plot_demo_all_comparison(
    *,
    demo_all_df: pd.DataFrame,
    out_path: Path,
    title: str,
) -> None:
    plot_df = demo_all_df.loc[demo_all_df["eval_role"] == "eval"].copy()
    if plot_df.empty:
        return
    if "rollout_success_rate" not in plot_df.columns:
        return

    plot_df["facet_label"] = (
        plot_df["model_family"] + " train_a=" + plot_df["train_alpha"].astype(str)
        + " " + plot_df["train_demo_ranked"].map(
            {True: "ranked", False: "unranked", None: "n/a"}
        ).fillna("n/a")
    )
    plot_df = plot_df.dropna(subset=["rollout_success_rate"])
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(plot_df["facet_label"].unique()) * 0.8), 4))
    sns.barplot(
        data=plot_df,
        x="facet_label",
        y="rollout_success_rate",
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("rollout_success_rate")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_accuracy_vs_compute_metric(
    *, eval_df: pd.DataFrame, x_col: str, x_label: str, title: str, out_path: Path,
) -> None:
    """Scatter plot of rollout_success_rate vs a compute metric (FLOPs or memory)."""
    plot_df = eval_df.loc[
        (eval_df["eval_role"] == "eval")
        & eval_df[x_col].notna()
        & eval_df["rollout_success_rate"].notna()
    ].copy()
    if plot_df.empty:
        return

    plot_df["facet_label"] = (
        "train_a=" + plot_df["train_alpha"].astype(str)
        + " " + plot_df["train_demo_ranked"].map(
            {True: "ranked", False: "unranked", None: "n/a"}
        ).fillna("n/a")
    )
    g = sns.relplot(
        data=plot_df,
        kind="scatter",
        x=x_col,
        y="rollout_success_rate",
        hue="model_family",
        style="model_family",
        col="facet_label",
        col_wrap=min(3, plot_df["facet_label"].nunique()),
        height=3.5,
        aspect=1.3,
    )
    for ax in np.ravel(g.axes):
        ax.set_xscale("log")
    g.set_axis_labels(x_label, "Rollout success rate")
    g.set_titles("{col_name}")
    g.fig.suptitle(title, y=1.02)
    g.savefig(out_path, bbox_inches="tight")
    plt.close(g.fig)


def _save_aggregates_for_mid_pred_and_train_iters(
    *,
    mid_pred: int,
    train_iters: int,
    train_max_n_demos: int,
    final_df: pd.DataFrame,
    role_eval_demo_df: pd.DataFrame,
    role_eval_needle_df: pd.DataFrame,
    role_eval_demo_all_df: pd.DataFrame,
    out_root: Path,
) -> None:
    out_dir = out_root / f"mid{int(mid_pred)}" / f"train_iters_{int(train_iters)}" / f"train_max_demos_{int(train_max_n_demos)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    final_slice = final_df.loc[
        (final_df["train_iters"] == float(train_iters))
        & (final_df["mid_pred"] == float(mid_pred))
        & (final_df["train_max_n_demos"] == float(train_max_n_demos))
    ].copy()
    role_slice = role_eval_demo_df.loc[
        (role_eval_demo_df["train_iters"] == float(train_iters))
        & (role_eval_demo_df["mid_pred"] == float(mid_pred))
        & (role_eval_demo_df["train_max_n_demos"] == float(train_max_n_demos))
    ].copy()
    needle_slice = role_eval_needle_df.loc[
        (role_eval_needle_df["train_iters"] == float(train_iters))
        & (role_eval_needle_df["mid_pred"] == float(mid_pred))
        & (role_eval_needle_df["train_max_n_demos"] == float(train_max_n_demos))
    ].copy() if not role_eval_needle_df.empty else pd.DataFrame()
    demo_all_slice = role_eval_demo_all_df.loc[
        (role_eval_demo_all_df["train_iters"] == float(train_iters))
        & (role_eval_demo_all_df["mid_pred"] == float(mid_pred))
        & (role_eval_demo_all_df["train_max_n_demos"] == float(train_max_n_demos))
    ].copy() if not role_eval_demo_all_df.empty else pd.DataFrame()

    if final_slice.empty:
        return

    final_agg_metrics = [
        "selection_metric_value",
        "eval_rollout_success_rate",
        "train_rollout_success_rate",
        "loss",
        "final_token_acc",
        "seq_exact_acc",
        "token_acc",
    ]
    groupby_cols = ["train_iters", "model_family", "mid_pred", "train_alpha", "train_max_n_demos"]
    if "train_demo_ranked" in final_slice.columns:
        groupby_cols.append("train_demo_ranked")
    final_agg = (
        final_slice.groupby(
            groupby_cols,
            as_index=False,
        )
        .agg({metric: "mean" for metric in final_agg_metrics if metric in final_slice.columns})
        .sort_values(["model_family", "mid_pred", "train_alpha"])
        .reset_index(drop=True)
    )
    final_agg.to_csv(out_dir / "summary_final_aggregated.csv", index=False)

    # --- Standard eval plots (separate files per eval_demo_ranked) ---
    if not role_slice.empty:
        role_metric_cols = [
            metric for metric in ROLE_EVAL_DEMO_METRIC_COLS if metric in role_slice.columns
        ]
        role_groupby = [
            "train_iters", "model_family", "mid_pred", "train_alpha",
            "train_max_n_demos", "train_demo_ranked", "eval_role", "eval_max_n_demos", "eval_alpha",
        ]
        if "eval_demo_ranked" in role_slice.columns:
            role_groupby.append("eval_demo_ranked")
        role_groupby = [c for c in role_groupby if c in role_slice.columns]
        role_agg = (
            role_slice.groupby(role_groupby, as_index=False, dropna=False)
            .agg({metric: "mean" for metric in role_metric_cols})
            .sort_values(
                ["eval_role", "model_family", "train_alpha", "eval_alpha", "eval_max_n_demos"]
            )
            .reset_index(drop=True)
        )
        role_agg.to_csv(out_dir / "summary_by_role_eval_demo_aggregated.csv", index=False)

        eval_ranked_values = (
            role_agg["eval_demo_ranked"].dropna().unique().tolist()
            if "eval_demo_ranked" in role_agg.columns
            else [None]
        )
        for eval_ranked_val in eval_ranked_values:
            if eval_ranked_val is not None:
                sub = role_agg.loc[role_agg["eval_demo_ranked"] == eval_ranked_val].copy()
                tag = "ranked" if eval_ranked_val else "unranked"
            else:
                sub = role_agg.copy()
                tag = "all"

            for group in ROLE_METRIC_PLOT_GROUPS:
                for eval_role in ("train", "eval"):
                    _plot_role_metric_group(
                        role_eval_demo_df=sub,
                        eval_role=eval_role,
                        metric_names=list(group["metrics"]),
                        out_path=out_dir / f"eval_{tag}_{eval_role}_{group['filename']}",
                        title=(
                            f"{str(eval_role).capitalize()}: {str(group['title'])} "
                            f"(mid{int(mid_pred)}, ti={int(train_iters)}, eval_{tag})"
                        ),
                        sharey=bool(group["sharey"]),
                        train_max_n_demos=int(train_max_n_demos),
                    )

            for model_family in sub["model_family"].unique():
                _plot_heatmap(
                    role_eval_demo_df=sub,
                    model_family=str(model_family),
                    train_iters=int(train_iters),
                    mid_pred=int(mid_pred),
                    out_dir=out_dir,
                    eval_max_n_demos=SELECTION_EVAL_MAX_N_DEMOS,
                    suffix=f"eval_{tag}",
                )

            # --- Compute metric scatter plots (unaggregated) ---
            if eval_ranked_val is not None and "eval_demo_ranked" in role_slice.columns:
                compute_sub = role_slice.loc[role_slice["eval_demo_ranked"] == eval_ranked_val]
            else:
                compute_sub = role_slice
            if "forward_flops_at_eval_seq" in compute_sub.columns:
                _plot_accuracy_vs_compute_metric(
                    eval_df=compute_sub,
                    x_col="forward_flops_at_eval_seq",
                    x_label="Forward FLOPs at eval seq len (log)",
                    title=f"Accuracy vs Inference FLOPs (mid{int(mid_pred)}, ti={int(train_iters)}, eval_{tag})",
                    out_path=out_dir / f"eval_{tag}_accuracy_vs_inference_flops.png",
                )
            if "activation_memory_at_eval_seq" in compute_sub.columns:
                _plot_accuracy_vs_compute_metric(
                    eval_df=compute_sub,
                    x_col="activation_memory_at_eval_seq",
                    x_label="Activation memory bytes at eval seq len (log)",
                    title=f"Accuracy vs Activation Memory (mid{int(mid_pred)}, ti={int(train_iters)}, eval_{tag})",
                    out_path=out_dir / f"eval_{tag}_accuracy_vs_activation_memory.png",
                )

    # --- Needle eval plots ---
    if not needle_slice.empty:
        needle_metric_cols = [
            metric for metric in ROLE_EVAL_DEMO_METRIC_COLS if metric in needle_slice.columns
        ]
        needle_groupby = [
            "train_iters", "model_family", "mid_pred", "train_alpha",
            "train_max_n_demos", "train_demo_ranked", "eval_role", "eval_max_n_demos", "eval_alpha",
        ]
        if "eval_demo_ranked" in needle_slice.columns:
            needle_groupby.append("eval_demo_ranked")
        needle_groupby = [c for c in needle_groupby if c in needle_slice.columns]
        needle_agg = (
            needle_slice.groupby(needle_groupby, as_index=False, dropna=False)
            .agg({metric: "mean" for metric in needle_metric_cols})
            .sort_values(
                ["eval_role", "model_family", "train_alpha", "eval_alpha", "eval_max_n_demos"]
            )
            .reset_index(drop=True)
        )
        needle_agg.to_csv(out_dir / "summary_by_role_eval_needle_aggregated.csv", index=False)

        needle_ranked_values = (
            needle_agg["eval_demo_ranked"].dropna().unique().tolist()
            if "eval_demo_ranked" in needle_agg.columns
            else [None]
        )
        for eval_ranked_val in needle_ranked_values:
            if eval_ranked_val is not None:
                sub = needle_agg.loc[needle_agg["eval_demo_ranked"] == eval_ranked_val].copy()
                tag = "ranked" if eval_ranked_val else "unranked"
            else:
                sub = needle_agg.copy()
                tag = "all"

            for group in ROLE_METRIC_PLOT_GROUPS:
                for eval_role in ("train", "eval"):
                    _plot_role_metric_group(
                        role_eval_demo_df=sub,
                        eval_role=eval_role,
                        metric_names=list(group["metrics"]),
                        out_path=out_dir / f"needle_{tag}_{eval_role}_{group['filename']}",
                        title=(
                            f"Needle {str(eval_role).capitalize()}: {str(group['title'])} "
                            f"(mid{int(mid_pred)}, ti={int(train_iters)}, eval_{tag})"
                        ),
                        sharey=bool(group["sharey"]),
                        train_max_n_demos=int(train_max_n_demos),
                    )

            for model_family in sub["model_family"].unique():
                _plot_heatmap(
                    role_eval_demo_df=sub,
                    model_family=str(model_family),
                    train_iters=int(train_iters),
                    mid_pred=int(mid_pred),
                    out_dir=out_dir,
                    eval_max_n_demos=SELECTION_EVAL_MAX_N_DEMOS,
                    suffix=f"needle_{tag}",
                )

            # --- Compute metric scatter plots (unaggregated) ---
            if eval_ranked_val is not None and "eval_demo_ranked" in needle_slice.columns:
                compute_sub = needle_slice.loc[needle_slice["eval_demo_ranked"] == eval_ranked_val]
            else:
                compute_sub = needle_slice
            if "forward_flops_at_eval_seq" in compute_sub.columns:
                _plot_accuracy_vs_compute_metric(
                    eval_df=compute_sub,
                    x_col="forward_flops_at_eval_seq",
                    x_label="Forward FLOPs at eval seq len (log)",
                    title=f"Needle: Accuracy vs Inference FLOPs (mid{int(mid_pred)}, ti={int(train_iters)}, needle_{tag})",
                    out_path=out_dir / f"needle_{tag}_accuracy_vs_inference_flops.png",
                )
            if "activation_memory_at_eval_seq" in compute_sub.columns:
                _plot_accuracy_vs_compute_metric(
                    eval_df=compute_sub,
                    x_col="activation_memory_at_eval_seq",
                    x_label="Activation memory bytes at eval seq len (log)",
                    title=f"Needle: Accuracy vs Activation Memory (mid{int(mid_pred)}, ti={int(train_iters)}, needle_{tag})",
                    out_path=out_dir / f"needle_{tag}_accuracy_vs_activation_memory.png",
                )

    # --- demo_all comparison ---
    if not demo_all_slice.empty:
        demo_all_slice.to_csv(out_dir / "summary_by_role_eval_demo_all_aggregated.csv", index=False)
        _plot_demo_all_comparison(
            demo_all_df=demo_all_slice,
            out_path=out_dir / "demo_all_rollout_comparison.png",
            title=f"demo_all rollout_success_rate (mid{int(mid_pred)}, ti={int(train_iters)})",
        )


# <codecell>
df = collate_dfs("remote/13_zipf_demo/set", show_progress=True)
if len(df) == 0:
    raise ValueError("No results found in remote/13_zipf_demo/set")

if OUT_DIR.exists():
    _remove_top_level_figures(OUT_DIR)

df = df.reset_index(drop=True).copy()
df["__row_id"] = np.arange(len(df), dtype=np.int64)

final_df = df.apply(_extract_final_row, axis=1).reset_index(drop=True)
role_eval_demo_df = _explode_role_eval_demo_rows(df)
role_eval_needle_df = _explode_role_eval_needle_rows(df)
role_eval_demo_all_df = _explode_role_eval_demo_all_rows(df)

# <codecell>
# --- Compute forward FLOPs at eval sequence length ---
_info_by_row_id = {
    int(row["__row_id"]): (row.get("info", {}) or {})
    for _, row in df.iterrows()
}


def _compute_eval_forward_flops(row):
    info = _info_by_row_id.get(int(row["run_row_id"]), {})
    seq_len = row.get("eval_dominant_seq_len")
    if pd.isna(seq_len):
        return np.nan
    try:
        return compute_metrics_from_info(info, n_seq_override=int(seq_len))["forward_flops"]
    except (KeyError, ValueError, TypeError):
        return np.nan


def _compute_eval_activation_memory(row):
    info = _info_by_row_id.get(int(row["run_row_id"]), {})
    seq_len = row.get("eval_dominant_seq_len")
    if pd.isna(seq_len):
        return np.nan
    try:
        metrics = compute_metrics_from_info(info, n_seq_override=int(seq_len))
        return memory_bytes_estimate(
            metrics["n_params"],
            batch_size=1,
            n_seq=int(seq_len),
            n_hidden=int(info.get("n_hidden", 128)),
            n_layers=int(info.get("n_layers", 2)),
            n_heads=int(info.get("n_heads", 4)),
            model_family=info.get("model_family", "transformer"),
        )["activations_bytes"]
    except (KeyError, ValueError, TypeError):
        return np.nan


for _eval_df in [role_eval_demo_df, role_eval_needle_df]:
    if _eval_df.empty or "eval_dominant_seq_len" not in _eval_df.columns:
        continue
    _eval_df["forward_flops_at_eval_seq"] = _eval_df.apply(
        _compute_eval_forward_flops, axis=1,
    )
    _eval_df["activation_memory_at_eval_seq"] = _eval_df.apply(
        _compute_eval_activation_memory, axis=1,
    )

sort_cols = ["model_family", "mid_pred", "train_alpha", "train_max_n_demos"]
if "train_demo_ranked" in final_df.columns:
    sort_cols.append("train_demo_ranked")
selection_df = final_df.sort_values(
    [*sort_cols, "selection_metric_value", "train_iters"],
    ascending=[True] * len(sort_cols) + [False, True],
).reset_index(drop=True)

final_df.to_csv(OUT_DIR / "summary_final.csv", index=False)
role_eval_demo_df.to_csv(OUT_DIR / "summary_by_role_eval_demo.csv", index=False)
if not role_eval_needle_df.empty:
    role_eval_needle_df.to_csv(OUT_DIR / "summary_by_role_eval_needle.csv", index=False)
if not role_eval_demo_all_df.empty:
    role_eval_demo_all_df.to_csv(OUT_DIR / "summary_by_role_eval_demo_all.csv", index=False)
selection_df.to_csv(OUT_DIR / "selection_ranked.csv", index=False)

mid_pred_values = sorted(int(v) for v in final_df["mid_pred"].dropna().astype(int).unique())
train_iters_values = sorted(int(v) for v in final_df["train_iters"].dropna().astype(int).unique())
train_max_n_demos_values = sorted(int(v) for v in final_df["train_max_n_demos"].dropna().astype(int).unique())
if not train_max_n_demos_values:
    train_max_n_demos_values = [8]

for mid_pred in mid_pred_values:
    for train_iters in train_iters_values:
        for train_max_n_demos in train_max_n_demos_values:
            _save_aggregates_for_mid_pred_and_train_iters(
                mid_pred=mid_pred,
                train_iters=train_iters,
                train_max_n_demos=train_max_n_demos,
                final_df=final_df,
                role_eval_demo_df=role_eval_demo_df,
                role_eval_needle_df=role_eval_needle_df,
                role_eval_demo_all_df=role_eval_demo_all_df,
                out_root=OUT_DIR,
            )

print("Saved:", OUT_DIR)


# <codecell>
# ====== Compute-efficiency plots ======


def _build_info_from_final_row(row) -> dict:
    """Reconstruct a minimal info dict from final_df columns."""
    info = {
        "model_family": row.get("model_family"),
        "n_vocab": row.get("n_vocab"),
        "n_layers": row.get("n_layers"),
        "n_hidden": row.get("n_hidden"),
        "n_heads": row.get("n_heads"),
        "use_swiglu": row.get("use_swiglu"),
        "d_state": row.get("d_state"),
        "d_conv": row.get("d_conv"),
        "scan_chunk_len": row.get("scan_chunk_len"),
        "n_seq": row.get("n_seq"),
    }
    return {k: v for k, v in info.items() if v is not None and not (isinstance(v, float) and np.isnan(v))}


def _plot_param_count_comparison(*, final_df, out_dir):
    """Plot 1: Parameter count by model family, broken down by component."""
    records = []
    seen = set()
    for _, row in final_df.iterrows():
        family = row.get("model_family")
        if family in seen or family is None:
            continue
        seen.add(family)
        info = _build_info_from_final_row(row)
        try:
            params = compute_metrics_from_info(info)["params"]
        except (KeyError, ValueError, TypeError):
            continue
        records.append({
            "model_family": family,
            "embedding": params["embedding"],
            "blocks": params["blocks_total"],
            "output_head": params["output_head"],
        })
    if not records:
        return

    plot_df = pd.DataFrame(records).melt(
        id_vars=["model_family"],
        value_vars=["embedding", "blocks", "output_head"],
        var_name="component",
        value_name="count",
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=plot_df, x="model_family", y="count", hue="component", ax=ax)
    ax.set_title("Parameter Count by Component")
    ax.set_ylabel("Parameters")
    fig.tight_layout()
    fig.savefig(out_dir / "compute_param_count_comparison.png", bbox_inches="tight")
    plt.close(fig)


def _plot_flops_vs_seq_len(*, final_df, out_dir):
    """Plot 2: Forward FLOPs vs sequence length — quadratic vs linear scaling.

    Two panels: (a) total forward FLOPs per layer (excluding the shared output
    head so the comparison is fair), (b) attention/SSD component only, which
    isolates the O(S²) vs O(S) scaling story.
    """
    seq_lens = np.array([16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16_000, 32_000, 64_000, 128_000])
    records = []
    seen = set()
    for _, row in final_df.iterrows():
        family = row.get("model_family")
        if family in seen or family is None:
            continue
        seen.add(family)
        info = _build_info_from_final_row(row)
        for s in seq_lens:
            try:
                metrics = compute_metrics_from_info(info, n_seq_override=int(s))
            except (KeyError, ValueError, TypeError):
                continue
            flops = metrics["flops"]
            rec = {
                "model_family": family,
                "n_seq": int(s),
                "total": flops["total"],
                "all_layers": flops["all_layers"],
                "per_layer": flops["per_layer"],
            }
            if family == "transformer":
                rec["seq_dependent"] = flops["attn_per_layer"]
                rec["seq_dep_label"] = "attention"
            elif family == "mamba2_bonsai":
                rec["seq_dependent"] = flops["ssd_per_layer"]
                rec["seq_dep_label"] = "SSD scan"
            records.append(rec)
    if not records:
        return

    plot_df = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel (a): per-layer FLOPs (apples-to-apples, no output head)
    ax = axes[0]
    for family, grp in plot_df.groupby("model_family"):
        grp_sorted = grp.sort_values("n_seq")
        dash = MODEL_FAMILY_DASHES.get(family, "")
        ax.plot(grp_sorted["n_seq"], grp_sorted["per_layer"],
                label=family, marker="o",
                dashes=dash if dash else (None, None))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("FLOPs per layer (log)")
    ax.set_title("(a) Per-layer FLOPs")
    ax.legend()

    # Panel (b): attention vs SSD component only
    ax = axes[1]
    for family, grp in plot_df.groupby("model_family"):
        grp_sorted = grp.sort_values("n_seq")
        label_suffix = grp_sorted["seq_dep_label"].iloc[0]
        dash = MODEL_FAMILY_DASHES.get(family, "")
        ax.plot(grp_sorted["n_seq"], grp_sorted["seq_dependent"],
                label=f"{family} ({label_suffix})", marker="o",
                dashes=dash if dash else (None, None))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("FLOPs (log)")
    ax.set_title("(b) Attention vs SSD scan FLOPs")
    ax.legend()

    fig.suptitle("FLOPs vs Sequence Length", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "compute_flops_vs_seq_len.png", bbox_inches="tight")
    plt.close(fig)


def _plot_training_compute_summary(*, final_df, out_dir):
    """Plot 4: Total training FLOPs by model config."""
    plot_df = final_df.dropna(subset=["training_flops_total"]).copy()
    if plot_df.empty:
        return

    plot_df["config_label"] = plot_df["model_family"]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=plot_df, x="config_label", y="training_flops_total",
                estimator="mean", errorbar=None, ax=ax)
    ax.set_title("Training Compute (total FLOPs)")
    ax.set_ylabel("Training FLOPs")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(out_dir / "compute_training_summary.png", bbox_inches="tight")
    plt.close(fig)


def _plot_memory_comparison(*, final_df, out_dir):
    """Plot 5: Estimated peak training memory by model config."""
    records = []
    for _, row in final_df.iterrows():
        family = row.get("model_family")
        n_params = row.get("n_params")
        if pd.isna(n_params) or family is None:
            continue
        n_seq = row.get("n_seq", 512)
        n_hidden = row.get("n_hidden", 256)
        n_layers = row.get("n_layers", 4)
        n_heads = row.get("n_heads", 8) if family == "transformer" else None
        mem = memory_bytes_estimate(
            int(n_params), batch_size=8, n_seq=int(n_seq),
            n_hidden=int(n_hidden), n_layers=int(n_layers), n_heads=n_heads,
            model_family=family,
        )
        records.append({
            "model_family": family,
            "params_GB": mem["params_bytes"] / 1e9,
            "optimizer_GB": mem["optimizer_bytes"] / 1e9,
            "activations_GB": mem["activations_bytes"] / 1e9,
        })
    if not records:
        return

    plot_df = pd.DataFrame(records).groupby("model_family", as_index=False).first()
    long_df = plot_df.melt(
        id_vars=["model_family"],
        value_vars=["params_GB", "optimizer_GB", "activations_GB"],
        var_name="component",
        value_name="GB",
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=long_df, x="model_family", y="GB", hue="component", ax=ax)
    ax.set_title("Estimated Peak Training Memory")
    ax.set_ylabel("GB")
    fig.tight_layout()
    fig.savefig(out_dir / "compute_memory_comparison.png", bbox_inches="tight")
    plt.close(fig)


def _plot_compute_efficiency(*, role_eval_demo_df, out_dir):
    """Plot 6: rollout_success_rate vs n_params scatter."""
    plot_df = role_eval_demo_df.loc[
        (role_eval_demo_df["eval_role"] == "eval")
        & role_eval_demo_df["n_params"].notna()
        & role_eval_demo_df["rollout_success_rate"].notna()
    ].copy()
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    for family, grp in plot_df.groupby("model_family"):
        ax.scatter(
            grp["n_params"], grp["rollout_success_rate"],
            label=family, alpha=0.5, s=20,
        )
    ax.set_xlabel("Parameter count")
    ax.set_ylabel("Rollout success rate")
    ax.set_title("Compute Efficiency: Accuracy vs Parameters")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "compute_efficiency_params.png", bbox_inches="tight")
    plt.close(fig)


# --- Generate compute plots ---
_plot_param_count_comparison(final_df=final_df, out_dir=OUT_DIR)
_plot_flops_vs_seq_len(final_df=final_df, out_dir=OUT_DIR)
_plot_training_compute_summary(final_df=final_df, out_dir=OUT_DIR)
_plot_memory_comparison(final_df=final_df, out_dir=OUT_DIR)
_plot_compute_efficiency(role_eval_demo_df=role_eval_demo_df, out_dir=OUT_DIR)

print("Compute plots saved to:", OUT_DIR)

# %%
