"""Plotting script for the 2_sweep ImplySizeTask transformer sweep."""

# <codecell>
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

OUT_DIR = Path("fig/2_sweep")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# <codecell>
df = collate_dfs("remote/2_sweep", show_progress=True)
df


def extract_plot_vals(row):
    info = row.get("info", {}) or {}
    hist = row.get("hist", {}) or {}
    test_hist = hist.get("test", []) or []
    test_metrics = test_hist[-1] if test_hist else {}

    return pd.Series(
        {
            "train_max": info.get("train_max"),
            "test_max": info.get("test_max"),
            "pos_encoding": info.get("pos_encoding"),
            "n_layers": info.get("n_layers"),
            "head_dim": info.get("head_dim"),
            "n_heads": info.get("n_heads"),
            "n_hidden": info.get("n_hidden"),
            "lr": info.get("lr"),
            "nonlinearity": info.get("nonlinearity"),
            "loss": test_metrics.get("loss", np.nan),
            "rule_acc": test_metrics.get("rule_acc", np.nan),
            "pos_acc": test_metrics.get("pos_acc", np.nan),
            "joint_acc": test_metrics.get("joint_acc", np.nan),
            "rule_membership_acc": test_metrics.get("rule_membership_acc", np.nan),
        }
    )


plot_df = df.apply(extract_plot_vals, axis=1).reset_index(drop=True)
plot_df.to_csv(OUT_DIR / "summary.csv", index=False)
plot_df

# <codecell>
group_cols = [
    "train_max",
    "test_max",
    "pos_encoding",
    "n_layers",
    "head_dim",
    "n_heads",
    "n_hidden",
    "nonlinearity",
]

best_df = (
    plot_df.groupby(group_cols, as_index=False)["joint_acc"]
    .max()
    .rename(columns={"joint_acc": "best_joint_acc"})
)

best_df["n_layers"] = best_df["n_layers"].astype(str)

g = sns.relplot(
    data=best_df,
    x="n_hidden",
    y="best_joint_acc",
    col="train_max",
    row="head_dim",
    hue="pos_encoding",
    style="n_layers",
    kind="scatter",
    height=3.0,
    aspect=1.2,
    facet_kws=dict(sharex=False, sharey=True),
)
g.set_axis_labels("Hidden (n_hidden)", "Best joint acc (over lr)")
plt.tight_layout()
plt.savefig(OUT_DIR / "best_over_lr.svg")

# <codecell>
baseline = plot_df[(plot_df["n_layers"] == 4) & (plot_df["n_heads"] == 8)].copy()
if baseline.empty:
    print("warn: baseline config (n_layers=4, n_heads=8) not found; skipping lr plot.")
else:
    g = sns.relplot(
        data=baseline,
        x="lr",
        y="joint_acc",
        col="train_max",
        row="head_dim",
        hue="pos_encoding",
        style="nonlinearity",
        kind="line",
        marker="o",
        height=3.0,
        aspect=1.2,
        facet_kws=dict(sharex=False, sharey=True),
    )
    g.set_axis_labels("Learning rate", "Joint acc")
    for ax in g.axes.flatten():
        ax.set_xscale("log")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "lr_sweep.svg")
