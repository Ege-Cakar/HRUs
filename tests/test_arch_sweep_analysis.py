"""Tests for experiment/5_arch_sweep analysis helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "experiment" / "5_arch_sweep.py"

_SPEC = importlib.util.spec_from_file_location("arch_sweep_analysis", SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

select_best_by_family_and_train_max = _MODULE.select_best_by_family_and_train_max
filter_size_df_to_best = _MODULE.filter_size_df_to_best
_infer_train_max_size = _MODULE._infer_train_max_size


def test_infer_train_max_size_prefers_explicit_field() -> None:
    info = {"train_max_size": 10, "train_sizes": [5, 6, 7]}
    out = _infer_train_max_size(info)
    assert out == 10.0


def test_select_best_by_family_and_train_max_groups_correctly() -> None:
    final_df = pd.DataFrame(
        [
            {"run_row_id": 1, "train_max_size": 5.0, "model_family": "transformer", "ood_metric_avg": 0.6},
            {"run_row_id": 2, "train_max_size": 5.0, "model_family": "transformer", "ood_metric_avg": 0.8},
            {"run_row_id": 3, "train_max_size": 5.0, "model_family": "mamba2", "ood_metric_avg": 0.7},
            {"run_row_id": 4, "train_max_size": 10.0, "model_family": "transformer", "ood_metric_avg": 0.5},
            {"run_row_id": 5, "train_max_size": 10.0, "model_family": "transformer", "ood_metric_avg": 0.4},
            {"run_row_id": 6, "train_max_size": 10.0, "model_family": "mamba2", "ood_metric_avg": 0.9},
        ]
    )

    best = select_best_by_family_and_train_max(final_df)
    got_ids = set(best["run_row_id"].tolist())
    assert got_ids == {2, 3, 4, 6}


def test_filter_size_df_to_best_keeps_only_best_run_rows() -> None:
    best_df = pd.DataFrame(
        [
            {"run_row_id": 2, "train_max_size": 5.0, "model_family": "transformer"},
            {"run_row_id": 3, "train_max_size": 5.0, "model_family": "mamba2"},
            {"run_row_id": 8, "train_max_size": 10.0, "model_family": "transformer"},
        ]
    )
    size_df = pd.DataFrame(
        [
            {"run_row_id": 2, "train_max_size": 5.0, "model_family": "transformer", "size": 5, "comparison_acc": 0.1},
            {"run_row_id": 2, "train_max_size": 5.0, "model_family": "transformer", "size": 6, "comparison_acc": 0.2},
            {"run_row_id": 9, "train_max_size": 5.0, "model_family": "transformer", "size": 5, "comparison_acc": 0.9},
            {"run_row_id": 3, "train_max_size": 5.0, "model_family": "mamba2", "size": 5, "comparison_acc": 0.3},
            {"run_row_id": 8, "train_max_size": 10.0, "model_family": "transformer", "size": 10, "comparison_acc": 0.8},
        ]
    )

    out = filter_size_df_to_best(size_df, best_df, train_max_size=5)
    assert set(out["run_row_id"].tolist()) == {2, 3}
    assert np.all(out["train_max_size"].to_numpy() == 5.0)
