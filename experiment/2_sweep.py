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
df = collate_dfs("remote/2_sweep/set", show_progress=True)
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
def extract_final_joint_acc(row):
    hist = row.get("hist", {}) or {}
    test_hist = hist.get("test", []) or []
    if not test_hist:
        return np.nan
    return test_hist[-1].get("joint_acc", np.nan)


def extract_test_history(row):
    info = row.get("info", {}) or {}
    hist = row.get("hist", {}) or {}
    test_hist = hist.get("test", []) or []
    train_args = row.get("train_args", {}) or {}

    train_iters = train_args.get("train_iters")
    test_every = train_args.get("test_every")

    if isinstance(train_iters, (int, np.integer)) and isinstance(
        test_every, (int, np.integer)
    ):
        steps = []
        for step in range(train_iters):
            if ((step + 1) % test_every == 0) or ((step + 1) == train_iters):
                steps.append(step + 1)
        if len(steps) != len(test_hist):
            steps = list(range(1, len(test_hist) + 1))
    else:
        steps = list(range(1, len(test_hist) + 1))

    rows = []
    for step, metrics in zip(steps, test_hist):
        rows.append(
            {
                "train_max": info.get("train_max"),
                "step": step,
                "rule_acc": metrics.get("rule_acc", np.nan),
                "pos_acc": metrics.get("pos_acc", np.nan),
                "joint_acc": metrics.get("joint_acc", np.nan),
                "rule_membership_acc": metrics.get("rule_membership_acc", np.nan),
            }
        )
    return pd.DataFrame(rows)

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

# --- Main plot: facet by n_layers instead of using style ---
g = sns.relplot(
    data=best_df,
    x="n_hidden",
    y="best_joint_acc",
    col="n_layers",
    row="train_max",
    hue="pos_encoding",
    style="head_dim",
    markers=True,
    kind="scatter",
    height=2.8,
    aspect=1.0,
    facet_kws=dict(sharex=True, sharey=True, margin_titles=True),
)
g.set_axis_labels("Hidden dim", "Best joint acc")
g.set_titles(col_template="{col_name} layers", row_template="train ≤ {row_name}")
sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0), frameon=True)
g.figure.subplots_adjust(right=0.82)
plt.savefig(OUT_DIR / "best_over_lr.png", bbox_inches="tight")

# <codecell>
# --- Supplementary: n_layers effect line plot ---
layers_agg = (
    best_df.groupby(["train_max", "n_layers", "pos_encoding"], as_index=False)
    ["best_joint_acc"].mean()
)
g2 = sns.relplot(
    data=layers_agg,
    x="n_layers",
    y="best_joint_acc",
    col="train_max",
    hue="pos_encoding",
    marker="o",
    kind="line",
    height=3.0,
    aspect=1.1,
    facet_kws=dict(sharey=True),
)
g2.set_axis_labels("Number of layers", "Mean best joint acc")
g2.set_titles(col_template="train ≤ {col_name}")
sns.move_legend(g2, "upper left", bbox_to_anchor=(1.0, 1.0), frameon=True)
g2.figure.subplots_adjust(right=0.85)
plt.savefig(OUT_DIR / "layers_effect.png", bbox_inches="tight")

# <codecell>
baseline = plot_df[(plot_df["n_layers"] == 2) & (plot_df["n_heads"] == 8)].copy()
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
    sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0), frameon=True)
    g.figure.subplots_adjust(right=0.82)
    plt.savefig(OUT_DIR / "lr_sweep.png", bbox_inches="tight")

# <codecell>
# --- Best model accuracy through time (per train_max) ---
best_hist_df = (
    df.assign(
        train_max=lambda x: x["info"].map(lambda info: (info or {}).get("train_max")),
        final_joint_acc=lambda x: x.apply(extract_final_joint_acc, axis=1),
    )
    .dropna(subset=["train_max"])
    .sort_values("final_joint_acc", ascending=False)
    .groupby("train_max", as_index=False)
    .head(1)
)

if best_hist_df.empty:
    print("warn: no runs found for best-through-time plot; skipping.")
else:
    best_hist_df = pd.concat(
        [extract_test_history(row) for _, row in best_hist_df.iterrows()],
        ignore_index=True,
    )

    if best_hist_df.empty:
        print("warn: best-through-time history empty; skipping plot.")
    else:
        acc_cols = [
            col
            for col in ["rule_acc", "pos_acc", "joint_acc", "rule_membership_acc"]
            if col in best_hist_df.columns and not best_hist_df[col].isna().all()
        ]
        long_df = best_hist_df.melt(
            id_vars=["train_max", "step"],
            value_vars=acc_cols,
            var_name="metric",
            value_name="acc",
        )

        g3 = sns.relplot(
            data=long_df,
            x="step",
            y="acc",
            col="train_max",
            hue="metric",
            kind="line",
            marker="o",
            height=3.0,
            aspect=1.1,
            facet_kws=dict(sharey=True),
        )
        g3.set_axis_labels("Training step", "Accuracy")
        g3.set_titles(col_template="train ≤ {col_name}")
        sns.move_legend(g3, "upper left", bbox_to_anchor=(1.0, 1.0), frameon=True)
        g3.figure.subplots_adjust(right=0.85)
        plt.savefig(OUT_DIR / "best_through_time.png", bbox_inches="tight")
