"""Depth-3 transfer split strategy for layered FOL tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from task.layer_gen.util import tokenize_layer_fol
from task.layer_gen.util.fol_rule_bank import (
    FOLDepth3ICLSplitBundle,
    FOLRuleBank,
    load_fol_depth3_icl_split_bundle,
)
from .common import _build_tokenizer_for_split_bundle
from .task_sampling import (
    _init_fol_online_worker,
    _sample_fol_online_worker_records,
    sample_online_record,
)
from .task_shared import FOLTaskSplitStrategy, OnlineSampleConfig, OnlineWorkerSpec


@dataclass
class Depth3ICLTransferSplitStrategy(FOLTaskSplitStrategy):
    rule_bank: FOLRuleBank
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer
    sample_config: OnlineSampleConfig
    split_bundle: FOLDepth3ICLSplitBundle | None
    online_forced_step_idx: int | None
    base_bank: FOLRuleBank | None = None
    task_split: str = "depth3_icl_transfer"

    @classmethod
    def build(
        cls,
        *,
        mode: str,
        split_role: str,
        split_rule_bundle_path,
        rule_bank_path,
        distances: tuple[int, ...],
        seed: int,
        initial_ant_max: int,
        sample_max_attempts: int,
        max_unify_solutions: int,
        max_n_demos: int,
        min_n_demos: int,
        include_oracle: bool,
        completion_format: str,
        demo_distribution: str = "uniform",
        demo_distribution_alpha: float = 1.0,
        demo_ranked: bool = True,
        demo_all: bool = False,
        demo_unique: bool = True,
    ) -> "Depth3ICLTransferSplitStrategy":
        if str(mode) != "online":
            raise ValueError("task_split='depth3_icl_transfer' requires mode='online'.")
        if split_rule_bundle_path is None:
            raise ValueError(
                "split_rule_bundle_path is required when task_split='depth3_icl_transfer'."
            )
        if rule_bank_path is not None:
            raise ValueError(
                "rule_bank_path cannot be combined with task_split='depth3_icl_transfer'; "
                "use split_rule_bundle_path."
            )
        if tuple(int(distance) for distance in distances) != (2,):
            raise ValueError(
                "task_split='depth3_icl_transfer' requires distance_range to resolve "
                f"to [2], got {list(distances)}."
            )

        split_bundle = load_fol_depth3_icl_split_bundle(Path(split_rule_bundle_path))
        rule_bank = (
            split_bundle.train_bank
            if str(split_role) == "train"
            else split_bundle.eval_bank
        )
        tokenizer = _build_tokenizer_for_split_bundle(split_bundle)
        online_forced_step_idx = 0 if str(split_role) == "eval" else None
        sample_config = OnlineSampleConfig(
            seed_base=int(seed),
            distances=(2,),
            initial_ant_max=int(initial_ant_max),
            sample_max_attempts=int(sample_max_attempts),
            max_unify_solutions=int(max_unify_solutions),
            max_n_demos=int(max_n_demos),
            min_n_demos=int(min_n_demos),
            include_oracle=bool(include_oracle),
            forced_step_idx=online_forced_step_idx,
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
            split_bundle=split_bundle,
            online_forced_step_idx=online_forced_step_idx,
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
                self.tokenizer.to_dict(),
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
            "tokenizer_payload": self.tokenizer.to_dict(),
            "workers": int(workers),
            "buffer_size": int(buffer_size),
            "batch_size": int(batch_size),
        }
