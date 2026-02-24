"""Analysis script for 6_layer_sweep architecture comparison."""

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
OUT_DIR = Path("fig/6_layer_sweep")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROLLOUT_ERROR_RATE_COLS = [
    ("rollout_decode_error_rate", "decode_error"),
    ("rollout_unknown_rule_error_rate", "unknown_rule_error"),
    ("rollout_inapplicable_rule_error_rate", "inapplicable_rule_error"),
    ("rollout_goal_not_reached_rate", "goal_not_reached"),
]

COMMON_CONFIG_COLS = [
    "lr",
    "n_layers",
    "n_hidden",
    "n_seq",
    "n_vocab",
    "target_format",
    "train_distances",
    "eval_distances",
    "ood_distances",
]

FAMILY_CONFIG_COLS = {
    "transformer": ["n_heads", "pos_encoding", "use_swiglu"],
    "mamba1": ["n_heads", "d_state", "d_conv", "scan_chunk_len"],
    "mamba2": ["n_heads", "d_state", "d_conv", "scan_chunk_len"],
    "mixer_completion": ["n_channels", "max_out_len"],
}

DISTANCE_METRIC_COLS = [
    "loss",
    "token_acc",
    "final_token_acc",
    "seq_exact_acc",
    "token_acc_full",
    "token_acc_unpadded",
    "seq_exact_acc_unpadded",
    "eot_pos_acc",
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
    "rollout_inapplicable_rule_error_rate",
    "rollout_goal_not_reached_rate",
    "rollout_avg_steps",
]

METRIC_PLOT_GROUPS = [
    {
        "filename": "distance_metrics_loss.svg",
        "title": "Loss by distance",
        "metrics": ["loss"],
        "sharey": False,
    },
    {
        "filename": "distance_metrics_accuracy.svg",
        "title": "Accuracy metrics by distance",
        "metrics": [
            "token_acc",
            "final_token_acc",
            "seq_exact_acc",
            "token_acc_full",
            "token_acc_unpadded",
            "seq_exact_acc_unpadded",
            "eot_pos_acc",
        ],
        "sharey": False,
    },
    {
        "filename": "distance_metrics_rule_match_rates.svg",
        "title": "Rule-match rates by distance",
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
        "filename": "distance_metrics_rollout_rates.svg",
        "title": "Rollout rates and steps by distance",
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
        "filename": "distance_metrics_rule_match_counts.svg",
        "title": "Rule-match counts by distance",
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
        "filename": "distance_metrics_rollout_counts.svg",
        "title": "Rollout counts by distance",
        "metrics": ["rollout_n_examples"],
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
    family = row.get("model_family")

    train_distances = _as_int_list(info.get("train_distances"))
    eval_distances = _as_int_list(info.get("eval_distances"))
    ood_distances = _as_int_list(info.get("ood_distances"))
    train_max_distance = _infer_train_max_distance(info)

    out = {
        "run_row_id": row.get("__row_id"),
        "run_id": row.get("run_id"),
        "model_family": family,
        "train_max_distance": train_max_distance,
        "selection_metric_name": row.get("selection_metric_name"),
        "selection_metric_value": row.get("selection_metric_value", np.nan),
        "ood_rollout_success_avg": row.get("ood_rollout_success_avg", np.nan),
        "ood_final_token_acc_avg": row.get("ood_final_token_acc_avg", np.nan),
        "ood_valid_rule_rate_avg": row.get("ood_valid_rule_rate_avg", np.nan),
        "ood_correct_rule_rate_avg": row.get("ood_correct_rule_rate_avg", np.nan),
        "train_distances": str(train_distances) if train_distances else str(info.get("train_distances")),
        "eval_distances": str(eval_distances) if eval_distances else str(info.get("eval_distances")),
        "ood_distances": str(ood_distances) if ood_distances else str(info.get("ood_distances")),
        "lr": info.get("lr", np.nan),
        "n_layers": info.get("n_layers", np.nan),
        "n_hidden": info.get("n_hidden", np.nan),
        "n_seq": info.get("n_seq", np.nan),
        "n_vocab": info.get("n_vocab", np.nan),
        "target_format": info.get("target_format"),
        "loss": final.get("loss", np.nan),
        "final_token_acc": final.get("final_token_acc", np.nan),
        "seq_exact_acc": final.get("seq_exact_acc", np.nan),
        "token_acc": final.get("token_acc", np.nan),
        "token_acc_full": final.get("token_acc_full", np.nan),
        "token_acc_unpadded": final.get("token_acc_unpadded", np.nan),
        "seq_exact_acc_unpadded": final.get("seq_exact_acc_unpadded", np.nan),
        "eot_pos_acc": final.get("eot_pos_acc", np.nan),
        "valid_rule_rate": final.get("valid_rule_rate", np.nan),
        "correct_rule_rate": final.get("correct_rule_rate", np.nan),
    }

    if family == "transformer":
        out["pos_encoding"] = info.get("pos_encoding")
        out["n_heads"] = info.get("n_heads", np.nan)
        out["use_swiglu"] = info.get("use_swiglu")
    elif family in {"mamba1", "mamba2"}:
        out["n_heads"] = info.get("n_heads", np.nan)
        out["d_state"] = info.get("d_state", np.nan)
        out["d_conv"] = info.get("d_conv", np.nan)
        out["scan_chunk_len"] = info.get("scan_chunk_len", np.nan)
    elif family == "mixer_completion":
        out["n_channels"] = info.get("n_channels", np.nan)
        out["max_out_len"] = info.get("max_out_len", np.nan)

    return pd.Series(out)


def _explode_distance_rows(df):
    rows = []
    for _, row in df.iterrows():
        family = row.get("model_family")
        info = row.get("info", {}) or {}
        by_distance = row.get("metrics_by_distance", {}) or {}
        train_max_distance = _infer_train_max_distance(info)

        for distance, metrics in by_distance.items():
            distance_int = int(distance)
            metrics = metrics or {}
            out = {
                "run_row_id": row.get("__row_id"),
                "run_id": row.get("run_id"),
                "train_max_distance": train_max_distance,
                "model_family": family,
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
        ["train_max_distance", "model_family", "ood_rollout_success_avg", "run_row_id"],
        ascending=[True, True, False, True],
    )
    return (
        ranked.groupby(["train_max_distance", "model_family"], as_index=False, sort=True)
        .head(1)
        .reset_index(drop=True)
    )


def filter_distance_df_to_best(
    distance_df: pd.DataFrame,
    best_df: pd.DataFrame,
    train_max_distance: int,
) -> pd.DataFrame:
    best_ids = best_df.loc[
        best_df["train_max_distance"] == float(train_max_distance),
        "run_row_id",
    ]
    mask = (distance_df["train_max_distance"] == float(train_max_distance)) & (
        distance_df["run_row_id"].isin(best_ids)
    )
    return distance_df.loc[mask].copy()


def _failed_error_breakdown_by_distance(by_distance_best: pd.DataFrame) -> pd.DataFrame:
    if by_distance_best.empty:
        return pd.DataFrame()

    df = by_distance_best.copy()
    failure_rate = (1.0 - df["rollout_success_rate"].astype(float)).clip(lower=0.0)
    out = df[["model_family", "distance"]].copy()

    for rate_col, error_type in ROLLOUT_ERROR_RATE_COLS:
        rate = df[rate_col].astype(float).clip(lower=0.0)
        out[error_type] = np.where(failure_rate > 0.0, rate / failure_rate, 0.0)

    return out.melt(
        id_vars=["model_family", "distance"],
        var_name="error_type",
        value_name="error_share_failed_only",
    )


def _plot_metrics_by_distance_group(
    *,
    by_distance_best: pd.DataFrame,
    metric_names: list[str],
    out_path: Path,
    title: str,
    train_max_distance: int,
    sharey: bool,
) -> None:
    available = [
        metric_name
        for metric_name in metric_names
        if metric_name in by_distance_best.columns and by_distance_best[metric_name].notna().any()
    ]
    if not available:
        return

    long_df = by_distance_best.melt(
        id_vars=["model_family", "distance"],
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
        x="distance",
        y="value",
        hue="model_family",
        col="metric",
        marker="o",
        col_wrap=min(3, max(1, len(available))),
        facet_kws={"sharex": True, "sharey": bool(sharey)},
        height=3.0,
        aspect=1.15,
    )
    for ax in np.ravel(g.axes):
        ax.axvline(float(train_max_distance), color="0.35", linestyle="--", linewidth=1.1)
    g.set_axis_labels("Layer distance", "Metric value")
    g.set_titles("{col_name}")
    g.fig.suptitle(f"{title} (train <= {int(train_max_distance)})")
    g.fig.subplots_adjust(top=0.86)
    g.savefig(out_path, bbox_inches="tight")
    plt.close(g.fig)


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
    train_max_distance: int,
    out_path: Path,
) -> None:
    lines = [
        f"# Best configurations for train_max_distance={int(train_max_distance)}",
        "",
        "Selection metric: `ood_rollout_success_avg` (higher is better).",
        "",
        "Metric notes:",
        "- `final_token_acc` is final non-padding token accuracy (mixer uses EOT-position accuracy in this sweep).",
        "- `valid_rule_rate` is the fraction of decoded completions that match any valid source-layer rule.",
        "- `correct_rule_rate` is exact match to the expected rule statement.",
        "",
    ]

    id_metric_cols = [
        "run_id",
        "run_row_id",
        "model_family",
        "train_max_distance",
        "ood_rollout_success_avg",
        "ood_final_token_acc_avg",
        "ood_valid_rule_rate_avg",
        "ood_correct_rule_rate_avg",
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


def _save_plots_for_train_max(
    *,
    train_max_distance: int,
    best_df: pd.DataFrame,
    distance_df: pd.DataFrame,
    out_root: Path,
) -> None:
    out_dir = out_root / f"train_max_{int(train_max_distance):02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    legacy_validity_bar = out_dir / "ood_rule_validity_bar.svg"
    if legacy_validity_bar.exists():
        legacy_validity_bar.unlink()

    by_distance_best = filter_distance_df_to_best(distance_df, best_df, train_max_distance)
    by_distance_best = by_distance_best.sort_values(["model_family", "distance"])

    for group in METRIC_PLOT_GROUPS:
        _plot_metrics_by_distance_group(
            by_distance_best=by_distance_best,
            metric_names=list(group["metrics"]),
            out_path=out_dir / str(group["filename"]),
            title=str(group["title"]),
            train_max_distance=train_max_distance,
            sharey=bool(group["sharey"]),
        )

    plt.figure(figsize=(6.3, 3.8))
    sns.lineplot(
        data=by_distance_best,
        x="distance",
        y="rollout_success_rate",
        hue="model_family",
        marker="o",
    )
    plt.axvline(float(train_max_distance), color="0.35", linestyle="--", linewidth=1.1)
    plt.xlabel("Layer distance")
    plt.ylabel("Rollout success")
    plt.title(f"Best per family (train distance <= {int(train_max_distance)})")
    plt.savefig(out_dir / "rollout_success_best.svg", bbox_inches="tight")

    plt.figure(figsize=(6.3, 3.8))
    sns.lineplot(
        data=by_distance_best,
        x="distance",
        y="final_token_acc",
        hue="model_family",
        marker="o",
    )
    plt.axvline(float(train_max_distance), color="0.35", linestyle="--", linewidth=1.1)
    plt.xlabel("Layer distance")
    plt.ylabel("Final token accuracy")
    plt.title(f"Best per family token accuracy (train <= {int(train_max_distance)})")
    plt.savefig(out_dir / "token_final_acc_best.svg", bbox_inches="tight")

    plt.figure(figsize=(6.3, 3.8))
    sns.lineplot(
        data=by_distance_best,
        x="distance",
        y="valid_rule_rate",
        hue="model_family",
        marker="o",
    )
    plt.axvline(float(train_max_distance), color="0.35", linestyle="--", linewidth=1.1)
    plt.xlabel("Layer distance")
    plt.ylabel("Valid-rule rate")
    plt.title(f"Best per family valid-rule rate (train <= {int(train_max_distance)})")
    plt.savefig(out_dir / "valid_rule_rate_by_distance_best.svg", bbox_inches="tight")

    best_rows = best_df.loc[best_df["train_max_distance"] == float(train_max_distance)].copy()
    best_rows = best_rows.sort_values("model_family")

    plt.figure(figsize=(5.6, 3.6))
    sns.barplot(data=best_rows, x="model_family", y="ood_rollout_success_avg")
    plt.xlabel("")
    plt.ylabel("OOD rollout success")
    plt.xticks(rotation=15)
    plt.title(f"OOD (> {int(train_max_distance)}) best config per family")
    plt.savefig(out_dir / "ood_rollout_bar.svg", bbox_inches="tight")

    error_breakdown = _failed_error_breakdown_by_distance(by_distance_best)
    if not error_breakdown.empty:
        g = sns.relplot(
            data=error_breakdown,
            kind="line",
            x="distance",
            y="error_share_failed_only",
            hue="error_type",
            col="model_family",
            marker="o",
            facet_kws={"sharex": True, "sharey": True},
            height=3.2,
            aspect=1.05,
        )
        for ax in np.ravel(g.axes):
            ax.axvline(float(train_max_distance), color="0.35", linestyle="--", linewidth=1.1)
        g.set_axis_labels("Layer distance", "Error share | failed rollouts")
        g.set_titles("{col_name}")
        g.fig.suptitle(
            f"Rollout error breakdown by distance (conditioned on failure, train <= {int(train_max_distance)})"
        )
        g.fig.subplots_adjust(top=0.8)
        g.savefig(
            out_dir / "rollout_error_breakdown_failed_only_by_distance_best.svg",
            bbox_inches="tight",
        )
        plt.close(g.fig)

    _write_best_config_markdown(
        best_rows=best_rows,
        train_max_distance=train_max_distance,
        out_path=out_dir / f"best_configs_train_max_{int(train_max_distance):02d}.md",
    )


df = collate_dfs("remote/6_layer_sweep/set", show_progress=True)
if len(df) == 0:
    raise ValueError("No results found in remote/6_layer_sweep/set")

df = df.reset_index(drop=True).copy()
df["__row_id"] = np.arange(len(df), dtype=np.int64)

final_df = df.apply(_extract_final_row, axis=1).reset_index(drop=True)
distance_df = _explode_distance_rows(df)
best_df = select_best_by_family_and_train_max(final_df)

final_df.to_csv(OUT_DIR / "summary_final.csv", index=False)
distance_df.to_csv(OUT_DIR / "summary_by_distance.csv", index=False)
best_df.to_csv(OUT_DIR / "best_by_family_and_train_max.csv", index=False)

train_max_values = sorted(int(v) for v in best_df["train_max_distance"].dropna().astype(int).unique())
for train_max_distance in train_max_values:
    _save_plots_for_train_max(
        train_max_distance=train_max_distance,
        best_df=best_df,
        distance_df=distance_df,
        out_root=OUT_DIR,
    )

print("Saved:", OUT_DIR)
