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
        demo_distribution: str = "uniform",
        demo_distribution_alpha: float = 1.0,
        demo_ranked: bool = True,
        demo_all: bool = False,
        demo_unique: bool = True,
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
            demo_distribution=str(demo_distribution),
            demo_distribution_alpha=float(demo_distribution_alpha),
            demo_ranked=bool(demo_ranked),
            demo_all=bool(demo_all),
            demo_unique=bool(demo_unique),
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
                self.sample_config.to_dict(),
                self.rule_bank.to_dict(),
                None,
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
            "config_payload": self.sample_config.to_dict(),
            "bank_payload": self.rule_bank.to_dict(),
            "tokenizer_payload": None,
            "workers": int(workers),
            "buffer_size": int(buffer_size),
            "batch_size": int(batch_size),
        }
