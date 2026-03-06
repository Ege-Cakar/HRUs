"""Fresh-ICL split strategy for layered FOL tasks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from task.layer_gen.util import tokenize_layer_fol
from task.layer_gen.util.fol_rule_bank import (
    FOLDepth3ICLSplitBundle,
    FOLRuleBank,
    build_random_fol_rule_bank,
)
from .common import _build_tokenizer_for_fresh_icl
from .task_sampling import (
    _init_fol_online_fresh_worker,
    _sample_fol_online_fresh_worker_records,
    sample_online_fresh_record,
)
from .task_shared import (
    FOLTaskSplitStrategy,
    FreshOnlineSampleConfig,
    OnlineWorkerSpec,
)


@dataclass
class Depth3FreshICLSplitStrategy(FOLTaskSplitStrategy):
    rule_bank: FOLRuleBank
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer
    sample_config: FreshOnlineSampleConfig
    base_bank: FOLRuleBank | None
    fresh_icl_n_predicates: int | None
    online_forced_step_idx: int | None
    split_bundle: FOLDepth3ICLSplitBundle | None = None
    task_split: str = "depth3_fresh_icl"

    @classmethod
    def build(
        cls,
        *,
        mode: str,
        split_role: str,
        distances: tuple[int, ...],
        seed: int,
        predicates_per_layer: int,
        rules_per_transition: int,
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
        completion_format: str,
        fresh_icl_n_predicates,
        predicate_name_len: int,
        rule_bank_path,
        split_rule_bundle_path,
        rng: np.random.Generator,
    ) -> "Depth3FreshICLSplitStrategy":
        if str(mode) != "online":
            raise ValueError("task_split='depth3_fresh_icl' requires mode='online'.")
        if rule_bank_path is not None:
            raise ValueError(
                "rule_bank_path cannot be combined with task_split='depth3_fresh_icl'."
            )
        if split_rule_bundle_path is not None:
            raise ValueError(
                "split_rule_bundle_path cannot be combined with task_split='depth3_fresh_icl'."
            )
        if tuple(int(distance) for distance in distances) != (2,):
            raise ValueError(
                "task_split='depth3_fresh_icl' requires distance_range to resolve "
                f"to [2], got {list(distances)}."
            )

        fresh_n_predicates = (
            int(fresh_icl_n_predicates)
            if fresh_icl_n_predicates is not None
            else int(predicates_per_layer)
        )
        base_bank = build_random_fol_rule_bank(
            n_layers=3,
            predicates_per_layer=int(predicates_per_layer),
            rules_per_transition=int(rules_per_transition),
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
        tokenizer = _build_tokenizer_for_fresh_icl(
            base_bank=base_bank,
            predicate_name_len=int(predicate_name_len),
        )
        online_forced_step_idx = 0 if str(split_role) == "eval" else None
        sample_config = FreshOnlineSampleConfig(
            seed_base=int(seed),
            distances=(2,),
            initial_ant_max=int(initial_ant_max),
            sample_max_attempts=int(sample_max_attempts),
            max_unify_solutions=int(max_unify_solutions),
            max_n_demos=int(max_n_demos),
            min_n_demos=int(min_n_demos),
            forced_step_idx=online_forced_step_idx,
            completion_format=str(completion_format),
            fresh_icl_n_predicates=int(fresh_n_predicates),
            rules_per_transition=int(rules_per_transition),
            k_in_min=int(k_in_min),
            k_in_max=int(k_in_max),
            k_out_min=int(k_out_min),
            k_out_max=int(k_out_max),
            predicate_name_len=int(predicate_name_len),
        )
        return cls(
            rule_bank=base_bank,
            tokenizer=tokenizer,
            sample_config=sample_config,
            base_bank=base_bank,
            fresh_icl_n_predicates=int(fresh_n_predicates),
            online_forced_step_idx=online_forced_step_idx,
        )

    def sample_record(self, *, rng: np.random.Generator) -> dict:
        if self.base_bank is None:
            raise RuntimeError("depth3_fresh_icl requires a base bank.")
        return sample_online_fresh_record(
            base_bank=self.base_bank,
            tokenizer=self.tokenizer,
            rng=rng,
            config=self.sample_config,
        )

    def make_worker_spec(self) -> OnlineWorkerSpec:
        if self.base_bank is None or self.fresh_icl_n_predicates is None:
            raise RuntimeError("Fresh-ICL prefetch requires base bank metadata.")
        return OnlineWorkerSpec(
            init_fn=_init_fol_online_fresh_worker,
            sample_records_fn=_sample_fol_online_fresh_worker_records,
            initargs=(
                int(self.sample_config.seed_base),
                self.base_bank.to_dict(),
                self.tokenizer.to_dict(),
                int(self.fresh_icl_n_predicates),
                int(self.sample_config.rules_per_transition),
                int(self.sample_config.k_in_min),
                int(self.sample_config.k_in_max),
                int(self.sample_config.k_out_min),
                int(self.sample_config.k_out_max),
                int(self.sample_config.initial_ant_max),
                int(self.sample_config.sample_max_attempts),
                int(self.sample_config.max_unify_solutions),
                int(self.sample_config.max_n_demos),
                int(self.sample_config.min_n_demos),
                (
                    None
                    if self.online_forced_step_idx is None
                    else int(self.online_forced_step_idx)
                ),
                str(self.sample_config.completion_format),
                int(self.sample_config.predicate_name_len),
            ),
        )

    def make_server_config(
        self,
        *,
        workers: int,
        buffer_size: int,
        batch_size: int,
    ) -> dict | None:
        return None
