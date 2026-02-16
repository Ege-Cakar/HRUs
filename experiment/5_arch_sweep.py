"""Analysis script for the 5_arch_sweep architecture comparison."""

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
OUT_DIR = Path("fig/5_arch_sweep")
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


def _infer_train_max_size(info: dict) -> float:
    val = info.get("train_max_size")
    if val is not None and not pd.isna(val):
        return float(int(val))
    train_sizes = _as_int_list(info.get("train_sizes"))
    if not train_sizes:
        return np.nan
    return float(max(train_sizes))


def _extract_final_row(row):
    info = row.get("info", {}) or {}
    final = row.get("metrics_final", {}) or {}
    family = row.get("model_family")

    train_sizes = _as_int_list(info.get("train_sizes"))
    eval_sizes = _as_int_list(info.get("eval_sizes"))
    ood_sizes = _as_int_list(info.get("ood_sizes"))
    train_max_size = _infer_train_max_size(info)

    out = {
        "run_row_id": row.get("__row_id"),
        "run_id": row.get("run_id"),
        "model_family": family,
        "train_max_size": train_max_size,
        "ood_metric_name": row.get("ood_metric_name"),
        "ood_metric_avg": row.get("ood_metric_avg", np.nan),
        "train_sizes": str(train_sizes) if train_sizes else str(info.get("train_sizes")),
        "eval_sizes": str(eval_sizes) if eval_sizes else str(info.get("eval_sizes")),
        "ood_sizes": str(ood_sizes) if ood_sizes else str(info.get("ood_sizes")),
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
    }

    if family == "transformer":
        out["pos_encoding"] = info.get("pos_encoding")
        out["n_heads"] = info.get("n_heads", np.nan)
        out["use_swiglu"] = info.get("use_swiglu")
    elif family == "mamba1":
        out["d_state"] = info.get("d_state", np.nan)
        out["d_conv"] = info.get("d_conv", np.nan)
    elif family == "mixer_completion":
        out["n_channels"] = info.get("n_channels", np.nan)
        out["max_out_len"] = info.get("max_out_len", np.nan)

    return pd.Series(out)


def _explode_size_rows(df):
    rows = []
    for _, row in df.iterrows():
        family = row.get("model_family")
        info = row.get("info", {}) or {}
        by_size = row.get("metrics_by_size", {}) or {}
        train_max_size = _infer_train_max_size(info)

        for size, metrics in by_size.items():
            size_int = int(size)
            metrics = metrics or {}
            if family == "mixer_completion":
                comparison_acc = metrics.get("eot_pos_acc", np.nan)
            else:
                comparison_acc = metrics.get("final_token_acc", np.nan)

            rows.append(
                {
                    "run_row_id": row.get("__row_id"),
                    "run_id": row.get("run_id"),
                    "train_max_size": train_max_size,
                    "model_family": family,
                    "size": size_int,
                    "comparison_acc": comparison_acc,
                    "loss": metrics.get("loss", np.nan),
                    "final_token_acc": metrics.get("final_token_acc", np.nan),
                    "seq_exact_acc": metrics.get("seq_exact_acc", np.nan),
                    "token_acc": metrics.get("token_acc", np.nan),
                    "token_acc_full": metrics.get("token_acc_full", np.nan),
                    "token_acc_unpadded": metrics.get("token_acc_unpadded", np.nan),
                    "seq_exact_acc_unpadded": metrics.get("seq_exact_acc_unpadded", np.nan),
                    "eot_pos_acc": metrics.get("eot_pos_acc", np.nan),
                    "lr": info.get("lr", np.nan),
                    "n_layers": info.get("n_layers", np.nan),
                    "n_hidden": info.get("n_hidden", np.nan),
                }
            )

    return pd.DataFrame(rows)


def select_best_by_family_and_train_max(final_df: pd.DataFrame) -> pd.DataFrame:
    rank_cols = ["train_max_size", "model_family", "ood_metric_avg", "run_row_id"]
    ranked = final_df.sort_values(
        rank_cols,
        ascending=[True, True, False, True],
    )
    return (
        ranked.groupby(["train_max_size", "model_family"], as_index=False, sort=True)
        .head(1)
        .reset_index(drop=True)
    )


def filter_size_df_to_best(
    size_df: pd.DataFrame,
    best_df: pd.DataFrame,
    train_max_size: int,
) -> pd.DataFrame:
    best_ids = best_df.loc[
        best_df["train_max_size"] == float(train_max_size),
        "run_row_id",
    ]
    mask = (size_df["train_max_size"] == float(train_max_size)) & (
        size_df["run_row_id"].isin(best_ids)
    )
    return size_df.loc[mask].copy()


def _save_plots_for_train_max(
    *,
    train_max_size: int,
    best_df: pd.DataFrame,
    size_df: pd.DataFrame,
    out_root: Path,
) -> None:
    out_dir = out_root / f"train_max_{int(train_max_size):02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    by_size_best = filter_size_df_to_best(size_df, best_df, train_max_size)
    by_size_best = by_size_best.sort_values(["model_family", "size"])

    plt.figure(figsize=(6.2, 3.7))
    sns.lineplot(
        data=by_size_best,
        x="size",
        y="comparison_acc",
        hue="model_family",
        marker="o",
    )
    plt.axvline(float(train_max_size), color="0.35", linestyle="--", linewidth=1.1)
    plt.xlabel("Sequent size")
    plt.ylabel("Comparison accuracy")
    plt.title(f"Best per family (train sizes <= {int(train_max_size)})")
    plt.savefig(out_dir / "length_generalization_best.svg", bbox_inches="tight")

    best_rows = best_df.loc[best_df["train_max_size"] == float(train_max_size)].copy()
    best_rows = best_rows.sort_values("model_family")

    plt.figure(figsize=(5.4, 3.5))
    sns.barplot(data=best_rows, x="model_family", y="ood_metric_avg")
    plt.xlabel("")
    plt.ylabel("OOD comparison accuracy")
    plt.xticks(rotation=15)
    plt.title(f"OOD (> {int(train_max_size)}) best config per family")
    plt.savefig(out_dir / "ood_best_bar.svg", bbox_inches="tight")


df = collate_dfs("remote/5_arch_sweep/set", show_progress=True)
if len(df) == 0:
    raise ValueError("No results found in remote/5_arch_sweep/set")

df = df.reset_index(drop=True).copy()
df["__row_id"] = np.arange(len(df), dtype=np.int64)

final_df = df.apply(_extract_final_row, axis=1).reset_index(drop=True)
size_df = _explode_size_rows(df)
best_df = select_best_by_family_and_train_max(final_df)

final_df.to_csv(OUT_DIR / "summary_final.csv", index=False)
size_df.to_csv(OUT_DIR / "summary_by_size.csv", index=False)
best_df.to_csv(OUT_DIR / "best_by_family_and_train_max.csv", index=False)

train_max_values = sorted(
    int(v) for v in best_df["train_max_size"].dropna().astype(int).unique()
)
for train_max_size in train_max_values:
    _save_plots_for_train_max(
        train_max_size=train_max_size,
        best_df=best_df,
        size_df=size_df,
        out_root=OUT_DIR,
    )

print("Saved:", OUT_DIR)

