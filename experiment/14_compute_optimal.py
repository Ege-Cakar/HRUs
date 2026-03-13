"""Analysis script for 14_compute_optimal — compute-optimal frontier."""

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
from model.compute import (
    compute_metrics_from_info,
    training_flops_total,
)


set_theme()
OUT_DIR = Path("fig/14_compute_optimal")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FAMILY_PALETTE = {
    "transformer": "C0",
    "mamba2_bonsai": "C1",
}

MODEL_FAMILY_LABELS = {
    "transformer": "Transformer",
    "mamba2_bonsai": "Mamba2 Bonsai",
}


# <codecell>
# --- Data loading ---

def _extract_info_field(row, field, default=np.nan):
    info = row.get("info", {}) or {}
    for value in (row.get(field), info.get(field)):
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return default


def _build_summary_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Extract flat summary DataFrame from collated results."""
    records = []
    for _, row in df_raw.iterrows():
        info = row.get("info", {}) or {}
        train_args = row.get("train_args", {}) or {}

        n_layers = int(info.get("n_layers", 0))
        n_hidden = int(info.get("n_hidden", 0))
        n_heads = int(info.get("n_heads", 0))
        model_family = str(info.get("model_family", ""))
        lr = float(info.get("lr", np.nan))
        train_iters = int(info.get("train_iters", 0))

        # Compute metrics
        try:
            cm = compute_metrics_from_info(info)
            n_params = cm["n_params"]
            forward_flops = cm["forward_flops"]
        except (KeyError, ValueError, TypeError):
            n_params = row.get("n_params", np.nan)
            forward_flops = row.get("forward_flops", np.nan)

        total_training_flops = row.get("total_training_flops", np.nan)
        if pd.isna(total_training_flops) or total_training_flops == 0:
            try:
                train_n_seq = info.get("train_fixed_length_n_seq", info.get("n_seq"))
                cm_train = compute_metrics_from_info(info, n_seq_override=train_n_seq)
                total_training_flops = training_flops_total(
                    cm_train["forward_flops"],
                    train_iters=train_iters,
                    batch_size=int(info.get("microbatch_size", 1)),
                    grad_accum_steps=int(info.get("grad_accum_steps", 1)),
                )
            except (KeyError, ValueError, TypeError):
                pass

        # Eval metrics
        eval_metrics = row.get("eval_metrics", {}) or {}
        rollout_success_rate = row.get(
            "rollout_success_rate",
            eval_metrics.get("rollout_success_rate", np.nan),
        )

        metrics_final = row.get("metrics_final", {}) or {}

        model_tag = f"{model_family}_l{n_layers}_h{n_hidden}"

        records.append({
            "run_id": row.get("run_id"),
            "name": row.get("name"),
            "model_family": model_family,
            "n_layers": n_layers,
            "n_hidden": n_hidden,
            "n_heads": n_heads,
            "lr": lr,
            "train_iters": train_iters,
            "n_params": n_params,
            "forward_flops": forward_flops,
            "total_training_flops": total_training_flops,
            "log_training_flops": np.log10(total_training_flops) if total_training_flops > 0 else np.nan,
            "rollout_success_rate": float(rollout_success_rate) if not pd.isna(rollout_success_rate) else np.nan,
            "final_loss": float(metrics_final.get("loss", np.nan)),
            "final_token_acc": float(metrics_final.get("token_acc", np.nan)),
            "model_tag": model_tag,
            "train_wall_s": row.get("train_wall_s", np.nan),
        })

    return pd.DataFrame(records)


def _select_best_lr(summary: pd.DataFrame) -> pd.DataFrame:
    """For each (model config, train_iters), keep only the best LR by rollout_success_rate."""
    group_cols = ["model_family", "n_layers", "n_hidden", "n_heads", "train_iters"]
    best_idx = summary.groupby(group_cols)["rollout_success_rate"].idxmax()
    return summary.loc[best_idx.dropna()].reset_index(drop=True)


# <codecell>

df_raw = collate_dfs("remote/14_compute_optimal/set")
print(f"Loaded {len(df_raw)} rows")

summary = _build_summary_df(df_raw)
summary.to_csv(OUT_DIR / "summary_all_runs.csv", index=False)
print(f"Summary: {len(summary)} rows, {summary['model_family'].nunique()} families")
print(summary.groupby("model_family")[["n_params", "rollout_success_rate"]].describe())

best = _select_best_lr(summary)
print(f"Best LR selection: {len(best)} rows")


# <codecell>
# --- Plot 1: Training curves ---

def _extract_training_curves(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Extract per-checkpoint loss/performance from training history."""
    records = []
    for _, row in df_raw.iterrows():
        info = row.get("info", {}) or {}
        hist = row.get("hist")
        if hist is None:
            continue

        model_family = str(info.get("model_family", ""))
        n_layers = int(info.get("n_layers", 0))
        n_hidden = int(info.get("n_hidden", 0))
        lr = float(info.get("lr", np.nan))
        train_iters = int(info.get("train_iters", 0))
        test_every = max(1, train_iters // 16)
        model_tag = f"{model_family}_l{n_layers}_h{n_hidden}"

        test_list = hist.get("test", [])
        for ckpt_idx, metrics in enumerate(test_list):
            step = (ckpt_idx + 1) * test_every
            records.append({
                "model_family": model_family,
                "model_tag": model_tag,
                "n_layers": n_layers,
                "n_hidden": n_hidden,
                "lr": lr,
                "train_iters": train_iters,
                "step": step,
                "loss": float(metrics.get("loss", np.nan)),
                "token_acc": float(metrics.get("token_acc", np.nan)),
                "final_token_acc": float(metrics.get("final_token_acc", np.nan)),
            })
    return pd.DataFrame(records)


curves = _extract_training_curves(df_raw)
if not curves.empty:
    for family in curves["model_family"].unique():
        fam_curves = curves[curves["model_family"] == family]
        model_tags = sorted(fam_curves["model_tag"].unique())

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for tag in model_tags:
            tag_data = fam_curves[fam_curves["model_tag"] == tag]
            # Pick best LR (lowest final loss)
            best_lr = tag_data.groupby("lr")["loss"].last().idxmin()
            tag_best = tag_data[tag_data["lr"] == best_lr]
            # Average across train_iters (longest run)
            longest = tag_best[tag_best["train_iters"] == tag_best["train_iters"].max()]

            axes[0].plot(longest["step"], longest["loss"], label=tag)
            axes[1].plot(longest["step"], longest["final_token_acc"], label=tag)

        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Loss")
        axes[0].set_title(f"{MODEL_FAMILY_LABELS.get(family, family)} — Training Loss")
        axes[0].legend(fontsize=7, ncol=2)

        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Final Token Acc")
        axes[1].set_title(f"{MODEL_FAMILY_LABELS.get(family, family)} — Final Token Acc")
        axes[1].legend(fontsize=7, ncol=2)

        plt.tight_layout()
        plt.savefig(OUT_DIR / f"training_curves_{family}.png", dpi=150, bbox_inches="tight")
        plt.close()

    print("Saved training curve plots")


# <codecell>
# --- Plot 2: Frontier scatter ---

fig, ax = plt.subplots(figsize=(8, 5))
for family in best["model_family"].unique():
    fam = best[best["model_family"] == family]
    sizes = np.clip(np.log10(fam["n_params"].values + 1) * 15, 10, 200)
    ax.scatter(
        fam["total_training_flops"],
        fam["rollout_success_rate"],
        s=sizes,
        alpha=0.7,
        label=MODEL_FAMILY_LABELS.get(family, family),
        color=MODEL_FAMILY_PALETTE.get(family),
    )

ax.set_xscale("log")
ax.set_xlabel("Total Training FLOPs")
ax.set_ylabel("Rollout Success Rate")
ax.set_title("Compute-Optimal Frontier")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "frontier_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved frontier scatter")


# <codecell>
# --- Plot 3: Compute-optimal envelope ---

def _monotone_envelope(x, y):
    """Compute monotone increasing envelope: for each x, best y so far."""
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    y_env = np.maximum.accumulate(y_sorted)
    return x_sorted, y_env


fig, ax = plt.subplots(figsize=(8, 5))
for family in best["model_family"].unique():
    fam = best[best["model_family"] == family].dropna(subset=["total_training_flops", "rollout_success_rate"])
    if fam.empty:
        continue

    color = MODEL_FAMILY_PALETTE.get(family)
    label = MODEL_FAMILY_LABELS.get(family, family)
    sizes = np.clip(np.log10(fam["n_params"].values + 1) * 15, 10, 200)

    ax.scatter(
        fam["total_training_flops"],
        fam["rollout_success_rate"],
        s=sizes,
        alpha=0.3,
        color=color,
    )

    x_env, y_env = _monotone_envelope(
        fam["total_training_flops"].values,
        fam["rollout_success_rate"].values,
    )
    ax.plot(x_env, y_env, color=color, linewidth=2, label=f"{label} envelope")

ax.set_xscale("log")
ax.set_xlabel("Total Training FLOPs")
ax.set_ylabel("Rollout Success Rate")
ax.set_title("Compute-Optimal Envelope")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "frontier_envelope.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved frontier envelope")


# <codecell>
# --- Plot 4: Iso-FLOPs analysis ---

def _assign_flops_bin(flops, n_bins=6):
    """Assign each run to a log-scale FLOPs bin."""
    log_flops = np.log10(flops)
    edges = np.linspace(log_flops.min(), log_flops.max(), n_bins + 1)
    labels = [f"1e{e:.1f}" for e in (edges[:-1] + edges[1:]) / 2]
    bins = pd.cut(log_flops, bins=edges, labels=labels, include_lowest=True)
    return bins


if not best.empty and best["total_training_flops"].notna().any():
    plot_data = best.dropna(subset=["total_training_flops", "rollout_success_rate"]).copy()
    if len(plot_data) > 5:
        plot_data["flops_bin"] = _assign_flops_bin(plot_data["total_training_flops"])
        plot_data["log_n_params"] = np.log10(plot_data["n_params"] + 1)

        # For each FLOPs bin, find the best model size
        iso_best = plot_data.groupby(["flops_bin", "model_family", "model_tag"]).agg(
            rollout_success_rate=("rollout_success_rate", "max"),
            n_params=("n_params", "first"),
        ).reset_index()

        fig, ax = plt.subplots(figsize=(10, 5))
        x_vals = sorted(iso_best["flops_bin"].unique())
        width = 0.35
        families = sorted(iso_best["model_family"].unique())

        for i, family in enumerate(families):
            fam_data = iso_best[iso_best["model_family"] == family]
            fam_best = fam_data.groupby("flops_bin").apply(
                lambda g: g.loc[g["rollout_success_rate"].idxmax()]
            ).reset_index(drop=True)

            x_pos = [x_vals.index(b) + i * width for b in fam_best["flops_bin"]]
            ax.bar(
                x_pos,
                fam_best["rollout_success_rate"],
                width=width,
                label=MODEL_FAMILY_LABELS.get(family, family),
                color=MODEL_FAMILY_PALETTE.get(family),
                alpha=0.8,
            )
            # Annotate with model size
            for xp, row_vals in zip(x_pos, fam_best.itertuples()):
                ax.text(
                    xp, row_vals.rollout_success_rate + 0.01,
                    f"{row_vals.n_params / 1e6:.1f}M",
                    ha="center", va="bottom", fontsize=7, rotation=45,
                )

        ax.set_xticks([j + width / 2 for j in range(len(x_vals))])
        ax.set_xticklabels(x_vals, rotation=30, fontsize=8)
        ax.set_xlabel("FLOPs Bin (log10)")
        ax.set_ylabel("Best Rollout Success Rate")
        ax.set_title("Iso-FLOPs: Best Model Size per Compute Budget")
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / "iso_flops_analysis.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved iso-FLOPs analysis")


# <codecell>
# --- Plot 5: Scaling law fit ---

def _fit_scaling_law(best_df: pd.DataFrame):
    """Fit N_opt = a * C^b on log-log scale for each model family."""
    results = {}

    for family in best_df["model_family"].unique():
        fam = best_df[best_df["model_family"] == family].copy()
        fam = fam.dropna(subset=["total_training_flops", "rollout_success_rate"])
        if len(fam) < 3:
            continue

        # For each FLOPs level, find the best-performing model
        fam["log_flops"] = np.log10(fam["total_training_flops"])
        fam["flops_bin_fine"] = pd.qcut(fam["log_flops"], q=min(8, len(fam)), duplicates="drop")

        optimal_points = []
        for bin_label, group in fam.groupby("flops_bin_fine"):
            best_row = group.loc[group["rollout_success_rate"].idxmax()]
            optimal_points.append({
                "log_C": np.log10(best_row["total_training_flops"]),
                "log_N": np.log10(best_row["n_params"]),
                "C": best_row["total_training_flops"],
                "N": best_row["n_params"],
                "rollout_success_rate": best_row["rollout_success_rate"],
            })

        opt_df = pd.DataFrame(optimal_points)
        if len(opt_df) < 2:
            continue

        # Fit log(N_opt) = b * log(C) + log(a)
        coeffs = np.polyfit(opt_df["log_C"], opt_df["log_N"], 1)
        b, log_a = coeffs
        a = 10 ** log_a

        results[family] = {
            "a": a,
            "b": b,
            "opt_df": opt_df,
        }

    return results


if not best.empty and len(best) > 5:
    scaling_results = _fit_scaling_law(best)

    if scaling_results:
        fig, ax = plt.subplots(figsize=(8, 5))

        for family, res in scaling_results.items():
            opt_df = res["opt_df"]
            color = MODEL_FAMILY_PALETTE.get(family)
            label = MODEL_FAMILY_LABELS.get(family, family)

            ax.scatter(opt_df["C"], opt_df["N"], color=color, s=60, zorder=5, label=f"{label} (optimal)")

            # Fit line
            C_range = np.logspace(opt_df["log_C"].min() - 0.3, opt_df["log_C"].max() + 0.3, 100)
            N_fit = res["a"] * C_range ** res["b"]
            ax.plot(C_range, N_fit, color=color, linestyle="--", alpha=0.7,
                    label=f"{label}: N = {res['a']:.2e} * C^{res['b']:.3f}")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Total Training FLOPs (C)")
        ax.set_ylabel("Optimal Model Size N (params)")
        ax.set_title("Compute-Optimal Scaling: N_opt vs C")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "scaling_law_fit.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved scaling law fit")

        for family, res in scaling_results.items():
            print(f"  {family}: N_opt = {res['a']:.4e} * C^{res['b']:.4f}")


# <codecell>
# --- Plot 6: Loss vs FLOPs (alternative frontier view) ---

if not best.empty:
    fig, ax = plt.subplots(figsize=(8, 5))
    for family in best["model_family"].unique():
        fam = best[best["model_family"] == family].dropna(subset=["total_training_flops", "final_loss"])
        if fam.empty:
            continue

        color = MODEL_FAMILY_PALETTE.get(family)
        label = MODEL_FAMILY_LABELS.get(family, family)
        sizes = np.clip(np.log10(fam["n_params"].values + 1) * 15, 10, 200)

        ax.scatter(
            fam["total_training_flops"],
            fam["final_loss"],
            s=sizes,
            alpha=0.5,
            color=color,
        )

        x_env, y_env = _monotone_envelope(
            fam["total_training_flops"].values,
            -fam["final_loss"].values,  # negate for decreasing envelope
        )
        ax.plot(x_env, -y_env, color=color, linewidth=2, label=f"{label} envelope")

    ax.set_xscale("log")
    ax.set_xlabel("Total Training FLOPs")
    ax.set_ylabel("Final Loss")
    ax.set_title("Loss vs Compute")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "loss_vs_flops.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved loss vs FLOPs")


# <codecell>

print(f"\nAll plots saved to {OUT_DIR}")
print(f"Summary CSV: {OUT_DIR / 'summary_all_runs.csv'}")
