"""Shared wandb helpers for experiment-controlled training instrumentation."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, is_dataclass
import importlib
import math
from pathlib import Path
from typing import Any, Iterator


_SKIP = object()


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool = False
    project: str | None = None
    name: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    api_key_path: str | Path = "key/wandb.txt"
    entity: str | None = None


def sanitize_for_wandb(value: Any) -> Any:
    """Recursively drop non-serializable objects from wandb config payloads."""
    if callable(value):
        return _SKIP
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return sanitize_for_wandb(asdict(value))
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            clean_item = sanitize_for_wandb(item)
            if clean_item is not _SKIP:
                out[str(key)] = clean_item
        return out
    if isinstance(value, (list, tuple, set)):
        out = []
        for item in value:
            clean_item = sanitize_for_wandb(item)
            if clean_item is not _SKIP:
                out.append(clean_item)
        return out
    if hasattr(value, "item") and callable(value.item):
        try:
            return sanitize_for_wandb(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return sanitize_for_wandb(value.to_dict())
        except TypeError:
            pass
    return _SKIP


def build_wandb_config(
    *,
    enabled: bool,
    project: str,
    name: str | None,
    api_key_path: str | Path,
    config: Mapping[str, Any] | None = None,
    entity: str | None = None,
) -> WandbConfig:
    clean_config = sanitize_for_wandb(config or {})
    if clean_config is _SKIP:
        clean_config = {}
    return WandbConfig(
        enabled=bool(enabled),
        project=str(project),
        name=name,
        config=clean_config,
        api_key_path=str(api_key_path),
        entity=entity,
    )


def make_experiment_wandb_config(
    *,
    enabled: bool,
    project: str,
    run_id: int | str,
    run_name: str,
    api_key_path: str | Path,
    model_config: Any = None,
    train_args: Mapping[str, Any] | None = None,
    info: Mapping[str, Any] | None = None,
    extra_config: Mapping[str, Any] | None = None,
    entity: str | None = None,
) -> WandbConfig:
    payload: dict[str, Any] = {
        "run_id": run_id,
        "experiment": project,
    }
    if model_config is not None:
        payload["model_config"] = model_config
    if train_args is not None:
        payload["train_args"] = train_args
    if info is not None:
        payload["info"] = info
    if extra_config is not None:
        payload["extra"] = extra_config
    return build_wandb_config(
        enabled=enabled,
        project=project,
        name=run_name,
        api_key_path=api_key_path,
        config=payload,
        entity=entity,
    )


def flatten_wandb_metrics(prefix: str, metrics: Mapping[str, Any] | None) -> dict[str, Any]:
    if not metrics:
        return {}

    out: dict[str, Any] = {}
    for key, value in metrics.items():
        metric_key = f"{prefix}/{key}"
        if isinstance(value, Mapping):
            out.update(flatten_wandb_metrics(metric_key, value))
            continue
        clean_value = sanitize_for_wandb(value)
        if clean_value is _SKIP:
            continue
        if isinstance(clean_value, (bool, int, float, str)):
            out[metric_key] = clean_value
    return out


@contextmanager
def wandb_run_context(cfg: WandbConfig | None) -> Iterator[Any | None]:
    """Initialize wandb lazily and always finish the run if enabled."""
    if cfg is None or not cfg.enabled:
        yield None
        return
    if not cfg.project:
        raise ValueError("wandb project is required when wandb is enabled")

    api_key_path = Path(cfg.api_key_path)
    if not api_key_path.exists():
        raise FileNotFoundError(f"wandb API key file not found: {api_key_path}")
    api_key = api_key_path.read_text(encoding="utf-8").strip()
    if not api_key:
        raise ValueError(f"wandb API key file is empty: {api_key_path}")

    wandb = importlib.import_module("wandb")
    wandb.login(key=api_key, relogin=True)
    wandb.init(
        project=cfg.project,
        name=cfg.name,
        config=cfg.config,
        entity=cfg.entity,
    )
    try:
        yield wandb
    finally:
        wandb.finish()


def log_wandb_metrics(
    wandb: Any | None,
    *,
    step: int,
    train: Mapping[str, Any] | None = None,
    test: Mapping[str, Any] | None = None,
    summary: Mapping[str, Any] | None = None,
) -> None:
    if wandb is None:
        return
    payload = {}
    payload.update(flatten_wandb_metrics("train", train))
    payload.update(flatten_wandb_metrics("test", test))
    payload.update(flatten_wandb_metrics("summary", summary))
    if payload:
        wandb.log(payload, step=int(step))
