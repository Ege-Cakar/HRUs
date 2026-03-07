"""Analysis script for 12_hf_fol."""

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
OUT_DIR = Path("fig/12_hf_fol")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _remove_top_level_figures(out_dir: Path) -> None:
    for path in out_dir.iterdir():
        if path.is_file() and path.suffix.lower() in {".svg", ".png", ".pdf"}:
            path.unlink()


def _infer_model_family(row) -> str:
    for value in (row.get("model_config_name"), row.get("model_name")):
        text = "" if value is None else str(value).lower()
        if "mamba" in text:
            return "mamba2"
        if text:
            return "transformer"
    return "unknown"


def _extract_mid_pred(row) -> float:
    fresh_cfg = row.get("fresh_icl_config", {}) or {}
    for value in (row.get("mid_pred"), fresh_cfg.get("mid_pred")):
        if value is None:
            continue
        try:
            return float(int(value))
        except (TypeError, ValueError):
            continue
    return np.nan


def _extract_task_shape_idx(row) -> float:
    value = row.get("task_shape_idx")
    if value is None:
        return np.nan
    try:
        return float(int(value))
    except (TypeError, ValueError):
        return np.nan


def _extract_task_shape_tag(row) -> str | None:
    value = row.get("task_shape_tag")
    if value is None:
        return None
    return str(value)


def _extract_train_iters(row) -> float:
    value = row.get("train_iters")
    if value is None:
        return np.nan
    try:
        return float(int(value))
    except (TypeError, ValueError):
        return np.nan


def _extract_test_every(row) -> int | None:
    value = row.get("test_every")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _eval_steps(*, train_iters: int, test_every: int, n_records: int) -> list[int]:
    steps = [
        int(step)
        for step in range(1, int(train_iters) + 1)
        if step % int(test_every) == 0 or step == int(train_iters)
    ]
    if len(steps) == int(n_records):
        return steps
    if int(n_records) == 0:
        return []
    if int(train_iters) < 1:
        return list(range(1, int(n_records) + 1))
    return [int(v) for v in np.linspace(1, int(train_iters), int(n_records), dtype=int)]


def _extract_final_row(row):
    hist = row.get("hist", {}) or {}
    train_hist = hist.get("train", []) if isinstance(hist, dict) else []
    eval_hist = hist.get("test", []) if isinstance(hist, dict) else []
    final_train = train_hist[-1] if train_hist else {}
    final_eval = eval_hist[-1] if eval_hist else {}
    fresh_cfg = row.get("fresh_icl_config", {}) or {}

    return pd.Series(
        {
            "run_id": row.get("run_id"),
            "run_split": row.get("run_split", np.nan),
            "combo_idx": row.get("combo_idx", np.nan),
            "model_config_name": row.get("model_config_name"),
            "model_name": row.get("model_name"),
            "model_family": _infer_model_family(row),
            "task_shape_idx": _extract_task_shape_idx(row),
            "task_shape_tag": _extract_task_shape_tag(row),
            "mid_pred": _extract_mid_pred(row),
            "train_iters": _extract_train_iters(row),
            "test_every": _extract_test_every(row),
            "lr": row.get("lr", np.nan),
            "batch_size": row.get("batch_size", np.nan),
            "grad_accum_steps": row.get("grad_accum_steps", np.nan),
            "n_gpus": row.get("n_gpus", np.nan),
            "n_vocab": row.get("n_vocab", np.nan),
            "n_seq_ar": row.get("n_seq_ar", np.nan),
            "train_loss": final_train.get("loss", np.nan),
            "train_acc": final_train.get("acc", np.nan),
            "eval_loss": final_eval.get("loss", np.nan),
            "eval_acc": final_eval.get("acc", np.nan),
            "predicates_per_layer": str(fresh_cfg.get("predicates_per_layer")),
            "rules_per_transition": str(fresh_cfg.get("rules_per_transition")),
        }
    )


def _explode_history_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        hist = row.get("hist", {}) or {}
        if not isinstance(hist, dict):
            continue

        train_iters = _extract_train_iters(row)
        test_every = _extract_test_every(row)
        if pd.isna(train_iters) or test_every is None:
            continue

        base = {
            "run_id": row.get("run_id"),
            "model_config_name": row.get("model_config_name"),
            "model_name": row.get("model_name"),
            "model_family": _infer_model_family(row),
            "task_shape_idx": _extract_task_shape_idx(row),
            "task_shape_tag": _extract_task_shape_tag(row),
            "mid_pred": _extract_mid_pred(row),
            "train_iters": train_iters,
            "test_every": float(test_every),
        }

        for hist_key, split_name in (("train", "train"), ("test", "eval")):
            entries = hist.get(hist_key, []) or []
            steps = _eval_steps(
                train_iters=int(train_iters),
                test_every=int(test_every),
                n_records=len(entries),
            )
            for step, metrics in zip(steps, entries):
                metrics = metrics or {}
                rows.append(
                    {
                        **base,
                        "split": split_name,
                        "step": int(step),
                        "loss": metrics.get("loss", np.nan),
                        "acc": metrics.get("acc", np.nan),
                    }
                )

    return pd.DataFrame(rows)


def _plot_history_metric(
    *,
    history_df: pd.DataFrame,
    split_name: str,
    metric: str,
    out_path: Path,
    title: str,
) -> None:
    plot_df = history_df.loc[history_df["split"] == str(split_name)].copy()
    if plot_df.empty or metric not in plot_df.columns:
        return

    plot_df = plot_df.dropna(subset=[metric, "mid_pred", "step"])
    if plot_df.empty:
        return

    plot_df["mid_pred"] = plot_df["mid_pred"].astype(int)
    plot_df["mid_pred_label"] = plot_df["mid_pred"].astype(str)
    hue_order = [str(v) for v in sorted(plot_df["mid_pred"].unique())]

    plt.figure(figsize=(7.0, 4.2))
    ax = sns.lineplot(
        data=plot_df,
        x="step",
        y=metric,
        hue="mid_pred_label",
        style="model_family",
        markers=True,
        dashes=True,
        estimator="mean",
        errorbar=None,
        hue_order=hue_order,
    )
    ax.set_xlim(left=0.0)
    ax.set_xlabel("Step")
    ax.set_ylabel(metric)
    ax.set_title(title)
    legend = ax.get_legend()
    if legend is not None:
        legend.set_title("mid_pred / family")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def _save_aggregates_for_train_iters(
    *,
    train_iters: int,
    final_df: pd.DataFrame,
    history_df: pd.DataFrame,
    out_root: Path,
) -> None:
    out_dir = out_root / f"train_iters_{int(train_iters)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    final_slice = final_df.loc[final_df["train_iters"] == float(train_iters)].copy()
    history_slice = history_df.loc[history_df["train_iters"] == float(train_iters)].copy()
    if final_slice.empty or history_slice.empty:
        return

    final_agg = (
        final_slice.groupby(["train_iters", "model_family", "mid_pred"], as_index=False)
        .agg(
            {
                "train_loss": "mean",
                "train_acc": "mean",
                "eval_loss": "mean",
                "eval_acc": "mean",
            }
        )
        .sort_values(["model_family", "mid_pred"])
        .reset_index(drop=True)
    )
    final_agg.to_csv(out_dir / "summary_final_aggregated_by_family_mid_pred.csv", index=False)

    history_agg = (
        history_slice.groupby(
            ["train_iters", "model_family", "mid_pred", "split", "step"],
            as_index=False,
        )
        .agg({"loss": "mean", "acc": "mean"})
        .sort_values(["split", "model_family", "mid_pred", "step"])
        .reset_index(drop=True)
    )
    history_agg.to_csv(out_dir / "summary_history_aggregated.csv", index=False)

    for split_name in ("train", "eval"):
        for metric in ("loss", "acc"):
            _plot_history_metric(
                history_df=history_agg,
                split_name=split_name,
                metric=metric,
                out_path=out_dir / f"{split_name}_{metric}_by_step.svg",
                title=f"{split_name.capitalize()} {metric} by step (train_iters={int(train_iters)})",
            )


df = collate_dfs("remote/12_hf_fol/set", show_progress=True)
if df.empty:
    raise ValueError("No results found in remote/12_hf_fol/set")

_remove_top_level_figures(OUT_DIR)

df = df.reset_index(drop=True).copy()
final_df = df.apply(_extract_final_row, axis=1).reset_index(drop=True)
history_df = _explode_history_rows(df)

final_df.to_csv(OUT_DIR / "summary_final.csv", index=False)
history_df.to_csv(OUT_DIR / "summary_history_long.csv", index=False)

train_iters_values = sorted(int(v) for v in final_df["train_iters"].dropna().astype(int).unique())
for train_iters in train_iters_values:
    _save_aggregates_for_train_iters(
        train_iters=train_iters,
        final_df=final_df,
        history_df=history_df,
        out_root=OUT_DIR,
    )

print("Saved:", OUT_DIR)
