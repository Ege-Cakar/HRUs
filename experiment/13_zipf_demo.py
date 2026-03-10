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
            "train_alpha": _extract_train_alpha(row, info=info),
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
        mid_pred = _extract_mid_pred(row, info=info)
        train_alpha = _extract_train_alpha(row, info=info)
        task_shape_idx = _extract_task_shape_idx(row, info=info)
        task_shape_tag = _extract_task_shape_tag(row, info=info)
        for role, by_demo in by_role.items():
            for eval_demo, by_alpha in (by_demo or {}).items():
                for eval_alpha, metrics in (by_alpha or {}).items():
                    metrics = metrics or {}
                    out = {
                        "run_row_id": row.get("__row_id"),
                        "run_id": row.get("run_id"),
                        "name": row.get("name"),
                        "model_family": row.get("model_family"),
                        "task_shape_idx": task_shape_idx,
                        "task_shape_tag": task_shape_tag,
                        "mid_pred": mid_pred,
                        "train_iters": train_iters,
                        "train_alpha": float(train_alpha),
                        "eval_role": str(role),
                        "eval_max_n_demos": int(eval_demo),
                        "eval_alpha": float(eval_alpha),
                        "lr": info.get("lr", np.nan),
                        "n_layers": info.get("n_layers", np.nan),
                        "n_hidden": info.get("n_hidden", np.nan),
                        "predicates_per_layer": str(info.get("predicates_per_layer")),
                        "rules_per_transition": str(info.get("rules_per_transition")),
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
) -> None:
    plot_df = role_eval_demo_df.loc[role_eval_demo_df["eval_role"] == str(eval_role)].copy()
    available = [metric for metric in metric_names if metric in plot_df.columns]
    if plot_df.empty or not available:
        return

    plot_df["train_alpha_label"] = "train_a=" + plot_df["train_alpha"].astype(str)
    plot_df["eval_alpha_label"] = plot_df["eval_alpha"].astype(str)

    long_df = plot_df[
        ["model_family", "train_alpha", "train_alpha_label", "eval_alpha",
         "eval_alpha_label", "eval_max_n_demos", *available]
    ].melt(
        id_vars=["model_family", "train_alpha", "train_alpha_label", "eval_alpha",
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
        row="train_alpha_label",
        col="metric",
        dashes=MODEL_FAMILY_DASHES,
        markers=True,
        estimator="mean",
        errorbar=None,
        facet_kws={"sharex": True, "sharey": bool(sharey)},
        height=3.0,
        aspect=1.15,
        hue_order=eval_alpha_order,
    )
    for ax in np.ravel(g.axes):
        ax.set_xlim(left=0.0)
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
    ax.set_title(f"rollout_success_rate — {model_family}\n(demos={eval_max_n_demos})")
    ax.set_xlabel("eval_alpha")
    ax.set_ylabel("train_alpha")
    fig.tight_layout()
    fig.savefig(
        out_dir / f"heatmap_rollout_success_rate_{model_family}.png",
        bbox_inches="tight",
    )
    plt.close(fig)


def _save_aggregates_for_mid_pred_and_train_iters(
    *,
    mid_pred: int,
    train_iters: int,
    final_df: pd.DataFrame,
    role_eval_demo_df: pd.DataFrame,
    out_root: Path,
) -> None:
    out_dir = out_root / f"mid{int(mid_pred)}" / f"train_iters_{int(train_iters)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    final_slice = final_df.loc[
        (final_df["train_iters"] == float(train_iters))
        & (final_df["mid_pred"] == float(mid_pred))
    ].copy()
    role_slice = role_eval_demo_df.loc[
        (role_eval_demo_df["train_iters"] == float(train_iters))
        & (role_eval_demo_df["mid_pred"] == float(mid_pred))
    ].copy()
    if final_slice.empty or role_slice.empty:
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
    final_agg = (
        final_slice.groupby(
            ["train_iters", "model_family", "mid_pred", "train_alpha"],
            as_index=False,
        )
        .agg({metric: "mean" for metric in final_agg_metrics if metric in final_slice.columns})
        .sort_values(["model_family", "mid_pred", "train_alpha"])
        .reset_index(drop=True)
    )
    final_agg.to_csv(out_dir / "summary_final_aggregated.csv", index=False)

    role_metric_cols = [
        metric for metric in ROLE_EVAL_DEMO_METRIC_COLS if metric in role_slice.columns
    ]
    role_agg = (
        role_slice.groupby(
            ["train_iters", "model_family", "mid_pred", "train_alpha",
             "eval_role", "eval_max_n_demos", "eval_alpha"],
            as_index=False,
        )
        .agg({metric: "mean" for metric in role_metric_cols})
        .sort_values(["eval_role", "model_family", "train_alpha", "eval_alpha", "eval_max_n_demos"])
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
                title=(
                    f"{str(eval_role).capitalize()}: {str(group['title'])} "
                    f"(mid{int(mid_pred)}, train_iters={int(train_iters)})"
                ),
                sharey=bool(group["sharey"]),
            )

    for model_family in role_agg["model_family"].unique():
        _plot_heatmap(
            role_eval_demo_df=role_agg,
            model_family=str(model_family),
            train_iters=int(train_iters),
            mid_pred=int(mid_pred),
            out_dir=out_dir,
            eval_max_n_demos=SELECTION_EVAL_MAX_N_DEMOS,
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
selection_df = final_df.sort_values(
    ["model_family", "mid_pred", "train_alpha", "selection_metric_value", "train_iters"],
    ascending=[True, True, True, False, True],
).reset_index(drop=True)

final_df.to_csv(OUT_DIR / "summary_final.csv", index=False)
role_eval_demo_df.to_csv(OUT_DIR / "summary_by_role_eval_demo.csv", index=False)
selection_df.to_csv(OUT_DIR / "selection_ranked.csv", index=False)

mid_pred_values = sorted(int(v) for v in final_df["mid_pred"].dropna().astype(int).unique())
train_iters_values = sorted(int(v) for v in final_df["train_iters"].dropna().astype(int).unique())

for mid_pred in mid_pred_values:
    for train_iters in train_iters_values:
        _save_aggregates_for_mid_pred_and_train_iters(
            mid_pred=mid_pred,
            train_iters=train_iters,
            final_df=final_df,
            role_eval_demo_df=role_eval_demo_df,
            out_root=OUT_DIR,
        )

print("Saved:", OUT_DIR)

# %%
