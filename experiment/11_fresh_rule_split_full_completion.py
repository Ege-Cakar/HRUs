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
    },
    {
        "filename": "role_demo_metrics_accuracy.svg",
        "title": "Token metrics by eval demos",
        "metrics": ["token_acc", "final_token_acc", "seq_exact_acc"],
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
        ],
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
        ],
    },
]

COMMON_CONFIG_COLS = [
    "target_format",
    "task_split",
    "completion_format",
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
    "grad_accum_steps",
    "microbatch_size",
    "effective_batch_size",
]

FAMILY_CONFIG_COLS = {
    "transformer": ["n_heads", "pos_encoding", "use_swiglu"],
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
            "eval_rollout_success_rate": row.get("eval_rollout_success_rate", np.nan),
            "train_rollout_success_rate": row.get("train_rollout_success_rate", np.nan),
            "target_format": info.get("target_format"),
            "task_split": info.get("task_split"),
            "completion_format": info.get("completion_format"),
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
        for role, by_demo in by_role.items():
            for eval_demo, metrics in (by_demo or {}).items():
                metrics = metrics or {}
                out = {
                    "run_row_id": row.get("__row_id"),
                    "run_id": row.get("run_id"),
                    "name": row.get("name"),
                    "model_family": row.get("model_family"),
                    "role": role,
                    "eval_max_n_demos": int(eval_demo),
                    "train_iters": train_iters,
                }
                for col in COMMON_CONFIG_COLS + FAMILY_CONFIG_COLS.get(str(row.get("model_family")), []):
                    out[col] = info.get(col)
                for metric in ROLE_EVAL_DEMO_METRIC_COLS:
                    out[metric] = metrics.get(metric, np.nan)
                rows.append(out)
    return pd.DataFrame(rows)


def _plot_role_metric_group(df_role_demo: pd.DataFrame, *, filename: str, title: str, metrics: list[str]):
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4.5 * n_metrics, 3.5), squeeze=False)
    axes = axes.ravel()

    for ax, metric in zip(axes, metrics):
        sns.lineplot(
            data=df_role_demo,
            x="eval_max_n_demos",
            y=metric,
            hue="model_family",
            style="role",
            markers=True,
            dashes=False,
            estimator="mean",
            errorbar=None,
            ax=ax,
        )
        ax.set_title(metric)
        ax.set_xlabel("eval_max_n_demos")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, bbox_inches="tight")
    plt.close(fig)


df = collate_dfs("remote/11_fresh_rule_split_full_completion/set", show_progress=True)
if df.empty:
    raise ValueError("No results found in remote/11_fresh_rule_split_full_completion/set")

df = df.reset_index(drop=True)
df["__row_id"] = np.arange(len(df))

df_final = df.apply(_extract_final_row, axis=1)
df_final.to_csv(OUT_DIR / "summary_final.csv", index=False)

df_role_demo = _explode_role_eval_demo_rows(df)
df_role_demo.to_csv(OUT_DIR / "summary_role_demo.csv", index=False)

for group in ROLE_METRIC_PLOT_GROUPS:
    _plot_role_metric_group(df_role_demo, **group)

selection_df = df_final.sort_values(
    ["model_family", "selection_metric_value", "train_iters"],
    ascending=[True, False, True],
)
selection_df.to_csv(OUT_DIR / "selection_ranked.csv", index=False)
