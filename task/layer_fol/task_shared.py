"""Shared types for layered FOL task loading and sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from task.layer_gen.util.fol_rule_bank import FOLDepth3ICLSplitBundle, FOLRuleBank
from task.layer_gen.util.tokenize_layer_fol import FOLLayerTokenizer


@dataclass(frozen=True)
class OnlineSampleConfig:
    seed_base: int
    distances: tuple[int, ...]
    initial_ant_max: int
    sample_max_attempts: int
    max_unify_solutions: int
    max_n_demos: int
    min_n_demos: int
    include_oracle: bool
    forced_step_idx: int | None
    completion_format: str
    demo_distribution: str
    demo_distribution_alpha: float
    demo_ranked: bool
    demo_all: bool


@dataclass(frozen=True)
class FreshOnlineSampleConfig(OnlineSampleConfig):
    fresh_layer0_predicates: int
    fresh_rules_per_transition: int
    k_in_min: int
    k_in_max: int
    k_out_min: int
    k_out_max: int
    predicate_name_len: int


@dataclass(frozen=True)
class OnlineWorkerSpec:
    init_fn: Callable[..., None]
    sample_records_fn: Callable[[int], list[dict]]
    initargs: tuple


class FOLTaskSplitStrategy(Protocol):
    task_split: str
    rule_bank: FOLRuleBank | None
    tokenizer: FOLLayerTokenizer | None
    split_bundle: FOLDepth3ICLSplitBundle | None
    base_bank: FOLRuleBank | None
    online_forced_step_idx: int | None

    def sample_record(self, *, rng: np.random.Generator) -> dict:
        ...

    def make_worker_spec(self) -> OnlineWorkerSpec:
        ...

    def make_server_config(
        self,
        *,
        workers: int,
        buffer_size: int,
        batch_size: int,
    ) -> dict | None:
        ...
