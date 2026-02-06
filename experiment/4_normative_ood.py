"""Analysis script for Transformer + normative OOD sweep."""

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
OUT_DIR = Path("fig/4_normative_ood")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _flatten_row(row):
    info = row.get("info", {}) or {}
    metrics = row.get("metrics", {}) or {}
    return pd.Series(
        {
            "run_id": row.get("run_id"),
            "model_family": row.get("model_family"),
            "train_max": info.get("train_max"),
            "test_max": info.get("test_max"),
            "pos_encoding": info.get("pos_encoding"),
            "n_layers": info.get("n_layers"),
            "head_dim": info.get("head_dim"),
            "n_heads": info.get("n_heads"),
            "n_hidden": info.get("n_hidden"),
            "lr": info.get("lr"),
            "nonlinearity": info.get("nonlinearity"),
            "rule_acc": metrics.get("rule_acc", np.nan),
            "pos_acc": metrics.get("pos_acc", np.nan),
            "joint_acc": metrics.get("joint_acc", np.nan),
            "rule_membership_acc": metrics.get("rule_membership_acc", np.nan),
            "nll": metrics.get("nll", np.nan),
            "calibration_ece": metrics.get("calibration_ece", np.nan),
            "count": metrics.get("count", np.nan),
        }
    )


def _fit_ridge_predictor(df: pd.DataFrame, target: str, alpha: float = 1e-3):
    x_df = pd.get_dummies(
        df[
            [
                "train_max",
                "n_layers",
                "n_heads",
                "n_hidden",
                "head_dim",
                "lr",
                "nonlinearity",
                "pos_encoding",
                "loglinear_joint",
                "kernel_joint",
                "bounded_joint",
            ]
        ],
        columns=["nonlinearity", "pos_encoding"],
        dtype=float,
    )
    x = x_df.to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)
    if x.shape[0] < 4:
        raise ValueError("Need at least four rows for train/test predictor split.")
    n_train = max(1, int(0.8 * x.shape[0]))
    rng = np.random.default_rng(0)
    order = rng.permutation(x.shape[0])
    train_idx = order[:n_train]
    test_idx = order[n_train:]
    if test_idx.size == 0:
        test_idx = train_idx[-1:]
        train_idx = train_idx[:-1]

    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]
    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True)
    x_std = np.where(x_std < 1e-8, 1.0, x_std)
    y_mean = float(y_train.mean())
    x_train_n = (x_train - x_mean) / x_std
    x_test_n = (x_test - x_mean) / x_std
    y_train_c = y_train - y_mean

    gram = x_train_n.T @ x_train_n + alpha * np.eye(x_train_n.shape[1])
    w = np.linalg.solve(gram, x_train_n.T @ y_train_c)
    y_hat = x_test_n @ w + y_mean
    mse = float(np.mean((y_test - y_hat) ** 2))
    denom = float(np.mean((y_test - y_test.mean()) ** 2))
    r2 = 1.0 - mse / denom if denom > 0 else float("nan")

    out = pd.DataFrame({"y_true": y_test, "y_pred": y_hat})
    return out, mse, r2


df = collate_dfs("remote/4_normative_ood/set", show_progress=True)
flat = df.apply(_flatten_row, axis=1).reset_index(drop=True)
flat.to_csv(OUT_DIR / "summary.csv", index=False)
flat

# Main OOD comparison across model families.
g = sns.relplot(
    data=flat,
    x="n_hidden",
    y="joint_acc",
    row="train_max",
    col="model_family",
    hue="pos_encoding",
    style="nonlinearity",
    kind="scatter",
    height=2.8,
    aspect=1.1,
    facet_kws=dict(sharex=True, sharey=True, margin_titles=True),
)
g.set_axis_labels("Hidden dim", "OOD joint acc")
g.set_titles(col_template="{col_name}", row_template="train <= {row_name}")
sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0), frameon=True)
g.figure.subplots_adjust(right=0.84)
plt.savefig(OUT_DIR / "ood_joint_by_family.svg", bbox_inches="tight")

# Calibration and NLL comparison.
melt = flat.melt(
    id_vars=["model_family", "train_max"],
    value_vars=["nll", "calibration_ece"],
    var_name="metric",
    value_name="value",
)
g2 = sns.catplot(
    data=melt,
    x="model_family",
    y="value",
    hue="metric",
    col="train_max",
    kind="box",
    height=3.0,
    aspect=1.1,
    sharey=False,
)
g2.set_axis_labels("", "Value")
g2.set_titles(col_template="train <= {col_name}")
for ax in g2.axes.flat:
    ax.tick_params(axis="x", rotation=20)
plt.savefig(OUT_DIR / "calibration_nll_by_family.svg", bbox_inches="tight")

# Build predictive dataset: one row per architecture/split with normative + transformer metrics.
pivot = (
    flat.pivot_table(
        index=[
            "run_id",
            "train_max",
            "test_max",
            "pos_encoding",
            "n_layers",
            "head_dim",
            "n_heads",
            "n_hidden",
            "lr",
            "nonlinearity",
        ],
        columns="model_family",
        values="joint_acc",
        aggfunc="mean",
    )
    .reset_index()
    .rename_axis(None, axis=1)
)
needed = {"transformer", "loglinear", "kernel", "bounded_prover"}
if needed.issubset(set(pivot.columns)):
    pred_df = pivot.rename(
        columns={
            "transformer": "transformer_joint",
            "loglinear": "loglinear_joint",
            "kernel": "kernel_joint",
            "bounded_prover": "bounded_joint",
        }
    ).dropna()
    if pred_df.shape[0] >= 4:
        pred_points, mse, r2 = _fit_ridge_predictor(pred_df, target="transformer_joint", alpha=1e-2)
        fig, ax = plt.subplots()
        ax.scatter(pred_points["y_true"], pred_points["y_pred"], alpha=0.8)
        lims = [
            min(pred_points["y_true"].min(), pred_points["y_pred"].min()),
            max(pred_points["y_true"].max(), pred_points["y_pred"].max()),
        ]
        ax.plot(lims, lims, linestyle="--", color="black", linewidth=1)
        ax.set_xlabel("Transformer OOD joint acc (true)")
        ax.set_ylabel("Predicted")
        ax.set_title(f"Ridge predictor: MSE={mse:.4f}, R2={r2:.3f}")
        plt.savefig(OUT_DIR / "predict_transformer_joint.svg", bbox_inches="tight")
        pred_points.to_csv(OUT_DIR / "predictor_holdout_points.csv", index=False)
        print(f"Predictor metrics: MSE={mse:.6f}, R2={r2:.4f}")
    else:
        print("warn: not enough complete rows for predictor fit.")
else:
    print("warn: missing required families for predictor analysis.")
