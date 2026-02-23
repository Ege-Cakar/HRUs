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
    elif family == "mamba2":
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
            rows.append(
                {
                    "run_row_id": row.get("__row_id"),
                    "run_id": row.get("run_id"),
                    "train_max_distance": train_max_distance,
                    "model_family": family,
                    "distance": distance_int,
                    "rollout_success_rate": metrics.get("rollout_success_rate", np.nan),
                    "rollout_decode_error_rate": metrics.get("rollout_decode_error_rate", np.nan),
                    "rollout_unknown_rule_error_rate": metrics.get("rollout_unknown_rule_error_rate", np.nan),
                    "rollout_inapplicable_rule_error_rate": metrics.get(
                        "rollout_inapplicable_rule_error_rate", np.nan
                    ),
                    "rollout_goal_not_reached_rate": metrics.get(
                        "rollout_goal_not_reached_rate", np.nan
                    ),
                    "rollout_avg_steps": metrics.get("rollout_avg_steps", np.nan),
                    "final_token_acc": metrics.get("final_token_acc", np.nan),
                    "token_acc": metrics.get("token_acc", np.nan),
                    "token_acc_full": metrics.get("token_acc_full", np.nan),
                    "token_acc_unpadded": metrics.get("token_acc_unpadded", np.nan),
                    "eot_pos_acc": metrics.get("eot_pos_acc", np.nan),
                    "valid_rule_rate": metrics.get("valid_rule_rate", np.nan),
                    "correct_rule_rate": metrics.get("correct_rule_rate", np.nan),
                    "lr": info.get("lr", np.nan),
                    "n_layers": info.get("n_layers", np.nan),
                    "n_hidden": info.get("n_hidden", np.nan),
                }
            )

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


def _save_plots_for_train_max(
    *,
    train_max_distance: int,
    best_df: pd.DataFrame,
    distance_df: pd.DataFrame,
    out_root: Path,
) -> None:
    out_dir = out_root / f"train_max_{int(train_max_distance):02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    by_distance_best = filter_distance_df_to_best(distance_df, best_df, train_max_distance)
    by_distance_best = by_distance_best.sort_values(["model_family", "distance"])

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

    best_rows = best_df.loc[best_df["train_max_distance"] == float(train_max_distance)].copy()
    best_rows = best_rows.sort_values("model_family")

    plt.figure(figsize=(5.6, 3.6))
    sns.barplot(data=best_rows, x="model_family", y="ood_rollout_success_avg")
    plt.xlabel("")
    plt.ylabel("OOD rollout success")
    plt.xticks(rotation=15)
    plt.title(f"OOD (> {int(train_max_distance)}) best config per family")
    plt.savefig(out_dir / "ood_rollout_bar.svg", bbox_inches="tight")

    plt.figure(figsize=(5.6, 3.6))
    sns.barplot(data=best_rows, x="model_family", y="ood_valid_rule_rate_avg")
    plt.xlabel("")
    plt.ylabel("OOD valid-rule rate")
    plt.xticks(rotation=15)
    plt.title(f"OOD valid-rule rate (> {int(train_max_distance)})")
    plt.savefig(out_dir / "ood_rule_validity_bar.svg", bbox_inches="tight")


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
