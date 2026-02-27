"""Shared iterator adapters for experiment batching pipelines."""

from __future__ import annotations

from typing import Callable


class MixerBatchAdapter:
    """Adapt (xs, ys) tasks to prompt-only inputs with transformed targets."""

    def __init__(
        self,
        base_task,
        *,
        prompt_builder: Callable,
        target_builder: Callable,
    ):
        self.base_task = base_task
        self.prompt_builder = prompt_builder
        self.target_builder = target_builder

    def __iter__(self):
        return self

    def __next__(self):
        xs, ys = next(self.base_task)
        return self.prompt_builder(xs), self.target_builder(ys)
