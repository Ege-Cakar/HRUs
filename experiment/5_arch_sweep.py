"""Analysis script for the 5_arch_sweep architecture comparison."""

from __future__ import annotations

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


def _extract_final_row(row):
    info = row.get("info", {}) or {}
    final = row.get("metrics_final", {}) or {}
    family = row.get("model_family")

    out = {
        "run_id": row.get("run_id"),
        "model_family": family,
        "ood_metric_name": row.get("ood_metric_name"),
        "ood_metric_avg": row.get("ood_metric_avg", np.nan),
        "train_sizes": str(info.get("train_sizes")),
        "eval_sizes": str(info.get("eval_sizes")),
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

        for size, metrics in by_size.items():
            size_int = int(size)
            metrics = metrics or {}
            if family == "mixer_completion":
                comparison_acc = metrics.get("eot_pos_acc", np.nan)
            else:
                comparison_acc = metrics.get("final_token_acc", np.nan)

            rows.append(
                {
                    "run_id": row.get("run_id"),
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


df = collate_dfs("remote/5_arch_sweep/set", show_progress=True)
final_df = df.apply(_extract_final_row, axis=1).reset_index(drop=True)
size_df = _explode_size_rows(df)

final_df.to_csv(OUT_DIR / "summary_final.csv", index=False)
size_df.to_csv(OUT_DIR / "summary_by_size.csv", index=False)

# Length generalization curves.
agg = (
    size_df.groupby(["model_family", "size"], as_index=False)["comparison_acc"]
    .mean()
    .sort_values(["model_family", "size"])
)

plt.figure(figsize=(6, 3.6))
sns.lineplot(data=agg, x="size", y="comparison_acc", hue="model_family", marker="o")
plt.xlabel("Sequent size")
plt.ylabel("Comparison accuracy")
plt.title("Length generalization (size 5..25)")
plt.savefig(OUT_DIR / "length_generalization_curve.svg", bbox_inches="tight")

# OOD summary over size >= 16.
ood = size_df[size_df["size"] >= 16]
plt.figure(figsize=(5.2, 3.4))
sns.boxplot(data=ood, x="model_family", y="comparison_acc")
plt.xlabel("")
plt.ylabel("OOD comparison accuracy")
plt.xticks(rotation=15)
plt.savefig(OUT_DIR / "ood_comparison_box.svg", bbox_inches="tight")

# Best run per family across OOD metric.
best = (
    final_df.sort_values("ood_metric_avg", ascending=False)
    .groupby("model_family", as_index=False)
    .head(1)
)
best.to_csv(OUT_DIR / "best_by_family.csv", index=False)

print("Saved:", OUT_DIR)
