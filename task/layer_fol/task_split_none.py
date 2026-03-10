"""Default layered FOL task split strategy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from task.layer_gen.util import tokenize_layer_fol
from task.layer_gen.util.fol_rule_bank import (
    FOLDepth3ICLSplitBundle,
    FOLRuleBank,
    build_random_fol_rule_bank,
    load_fol_rule_bank,
)
from .task_sampling import (
    _init_fol_online_worker,
    _sample_fol_online_worker_records,
    sample_online_record,
)
from .task_shared import FOLTaskSplitStrategy, OnlineSampleConfig, OnlineWorkerSpec


@dataclass
class NoSplitStrategy(FOLTaskSplitStrategy):
    rule_bank: FOLRuleBank
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer
    sample_config: OnlineSampleConfig
    split_bundle: FOLDepth3ICLSplitBundle | None = None
    base_bank: FOLRuleBank | None = None
    online_forced_step_idx: int | None = None
    task_split: str = "none"

    @classmethod
    def build(
        cls,
        *,
        seed: int,
        distances: tuple[int, ...],
        rule_bank_path,
        n_layers: int,
        predicates_per_layer,
        rules_per_transition,
        arity_max: int,
        arity_min: int,
        vars_per_rule_max: int,
        constants,
        k_in_min: int,
        k_in_max: int,
        k_out_min: int,
        k_out_max: int,
        initial_ant_max: int,
        sample_max_attempts: int,
        max_unify_solutions: int,
        max_n_demos: int,
        min_n_demos: int,
        include_oracle: bool,
        completion_format: str,
        rng: np.random.Generator,
        demo_all: bool = False,
    ) -> "NoSplitStrategy":
        if rule_bank_path is not None:
            rule_bank = load_fol_rule_bank(Path(rule_bank_path))
        else:
            rule_bank = build_random_fol_rule_bank(
                n_layers=int(n_layers),
                predicates_per_layer=predicates_per_layer,
                rules_per_transition=rules_per_transition,
                arity_max=int(arity_max),
                arity_min=int(arity_min),
                vars_per_rule_max=int(vars_per_rule_max),
                constants=tuple(str(tok) for tok in constants),
                k_in_min=int(k_in_min),
                k_in_max=int(k_in_max),
                k_out_min=int(k_out_min),
                k_out_max=int(k_out_max),
                rng=rng,
            )
        tokenizer = tokenize_layer_fol.build_tokenizer_from_rule_bank(rule_bank)
        sample_config = OnlineSampleConfig(
            seed_base=int(seed),
            distances=tuple(int(distance) for distance in distances),
            initial_ant_max=int(initial_ant_max),
            sample_max_attempts=int(sample_max_attempts),
            max_unify_solutions=int(max_unify_solutions),
            max_n_demos=int(max_n_demos),
            min_n_demos=int(min_n_demos),
            include_oracle=bool(include_oracle),
            forced_step_idx=None,
            completion_format=str(completion_format),
            demo_distribution="uniform",
            demo_distribution_alpha=1.0,
            demo_ranked=True,
            demo_all=bool(demo_all),
        )
        return cls(
            rule_bank=rule_bank,
            tokenizer=tokenizer,
            sample_config=sample_config,
        )

    def sample_record(self, *, rng: np.random.Generator) -> dict:
        return sample_online_record(
            bank=self.rule_bank,
            tokenizer=self.tokenizer,
            rng=rng,
            config=self.sample_config,
        )

    def make_worker_spec(self) -> OnlineWorkerSpec:
        return OnlineWorkerSpec(
            init_fn=_init_fol_online_worker,
            sample_records_fn=_sample_fol_online_worker_records,
            initargs=(
                int(self.sample_config.seed_base),
                self.rule_bank.to_dict(),
                None,
                tuple(int(distance) for distance in self.sample_config.distances),
                int(self.sample_config.initial_ant_max),
                int(self.sample_config.sample_max_attempts),
                int(self.sample_config.max_unify_solutions),
                int(self.sample_config.max_n_demos),
                int(self.sample_config.min_n_demos),
                bool(self.sample_config.include_oracle),
                None,
                str(self.sample_config.completion_format),
                bool(self.sample_config.demo_all),
            ),
        )

    def make_server_config(
        self,
        *,
        workers: int,
        buffer_size: int,
        batch_size: int,
    ) -> dict | None:
        return {
            "seed": int(self.sample_config.seed_base),
            "bank_payload": self.rule_bank.to_dict(),
            "tokenizer_payload": None,
            "distances": tuple(int(distance) for distance in self.sample_config.distances),
            "initial_ant_max": int(self.sample_config.initial_ant_max),
            "sample_max_attempts": int(self.sample_config.sample_max_attempts),
            "max_unify_solutions": int(self.sample_config.max_unify_solutions),
            "max_n_demos": int(self.sample_config.max_n_demos),
            "min_n_demos": int(self.sample_config.min_n_demos),
            "include_oracle": bool(self.sample_config.include_oracle),
            "forced_step_idx": None,
            "completion_format": str(self.sample_config.completion_format),
            "demo_all": bool(self.sample_config.demo_all),
            "workers": int(workers),
            "buffer_size": int(buffer_size),
            "batch_size": int(batch_size),
        }
