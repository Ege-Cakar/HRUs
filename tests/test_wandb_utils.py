"""Tests for wandb helper utilities."""

from dataclasses import dataclass
from pathlib import Path

import pytest

from wandb_utils import make_experiment_wandb_config, sanitize_for_wandb, wandb_run_context


@dataclass
class _Payload:
    width: int
    path: Path


def test_make_experiment_wandb_config_sanitizes_non_serializable_fields(tmp_path):
    payload = {
        "keep_scalar": 3,
        "drop_callable": lambda: None,
        "nested": {"path": tmp_path / "data", "flag": True},
        "items": [1, lambda: None, 2],
    }

    cfg = make_experiment_wandb_config(
        enabled=True,
        project="demo",
        run_id=7,
        run_name="demo-7",
        api_key_path=tmp_path / "wandb.txt",
        model_config=_Payload(width=8, path=tmp_path / "model"),
        train_args=payload,
    )

    assert cfg.config["model_config"]["width"] == 8
    assert cfg.config["model_config"]["path"] == str(tmp_path / "model")
    assert cfg.config["train_args"]["keep_scalar"] == 3
    assert "drop_callable" not in cfg.config["train_args"]
    assert cfg.config["train_args"]["nested"]["path"] == str(tmp_path / "data")
    assert cfg.config["train_args"]["items"] == [1, 2]


def test_sanitize_for_wandb_skips_unknown_objects():
    class _Unknown:
        pass

    assert sanitize_for_wandb({"bad": _Unknown()}) == {}


def test_wandb_run_context_requires_key_file_when_enabled(tmp_path):
    with pytest.raises(FileNotFoundError, match="wandb API key file"):
        with wandb_run_context(
            make_experiment_wandb_config(
                enabled=True,
                project="demo",
                run_id=1,
                run_name="demo-1",
                api_key_path=tmp_path / "missing.txt",
            )
        ):
            pass
