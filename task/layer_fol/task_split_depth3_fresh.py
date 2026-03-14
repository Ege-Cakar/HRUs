"""Fresh-ICL split strategy for layered FOL tasks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from task.layer_gen.util import tokenize_layer_fol
from task.layer_gen.util.fol_rule_bank import (
    FOLDepth3ICLSplitBundle,
    FOLRuleBank,
    _normalize_count_spec,
    build_random_fol_rule_bank,
    sample_fol_problem,
)
from .common import _build_tokenizer_for_fresh_icl
from .demos import _classify_rules_by_rank, _precompute_cluster_candidate_rankings
from .task_sampling import (
    _init_fol_online_worker,
    _sample_fol_online_worker_records,
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
    online_forced_step_idx: int | None
    split_bundle: FOLDepth3ICLSplitBundle | None = None
    task_split: str = "depth3_fresh_icl"
    _precomputed_cluster_candidates_by_layer: dict | None = None

    @classmethod
    def build(
        cls,
        *,
        mode: str,
        split_role: str,
        distances: tuple[int, ...],
        seed: int,
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
        fresh_icl_base_bank_seed,
        predicate_name_len: int,
        rule_bank_path,
        split_rule_bundle_path,
        rng: np.random.Generator,
        demo_distribution: str = "uniform",
        demo_distribution_alpha: float = 1.0,
        demo_ranked: bool = True,
        demo_all: bool = False,
        demo_unique: bool = True,
        cluster_n_samples: int = 100,
        cluster_k: int = 5,
        cluster_base_dist: str = "zipf_per_rule",
        cluster_unselected_rank: int | None = None,
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
        _ = rng

        predicates_per_layer_counts = _normalize_count_spec(
            predicates_per_layer,
            expected_len=3,
            name="predicates_per_layer",
        )
        rules_per_transition_counts = _normalize_count_spec(
            rules_per_transition,
            expected_len=2,
            name="rules_per_transition",
        )
        fresh_layer0_predicates = int(predicates_per_layer_counts[0])
        fresh_rules_per_transition = int(rules_per_transition_counts[0])
        base_bank_seed = (
            int(seed)
            if fresh_icl_base_bank_seed is None
            else int(fresh_icl_base_bank_seed)
        )
        base_bank = build_random_fol_rule_bank(
            n_layers=3,
            predicates_per_layer=predicates_per_layer_counts,
            rules_per_transition=rules_per_transition_counts,
            arity_max=int(arity_max),
            arity_min=int(arity_min),
            vars_per_rule_max=int(vars_per_rule_max),
            constants=tuple(str(tok) for tok in constants),
            k_in_min=int(k_in_min),
            k_in_max=int(k_in_max),
            k_out_min=int(k_out_min),
            k_out_max=int(k_out_max),
            rng=np.random.default_rng(base_bank_seed),
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
            include_oracle=bool(include_oracle),
            forced_step_idx=online_forced_step_idx,
            completion_format=str(completion_format),
            fresh_layer0_predicates=int(fresh_layer0_predicates),
            fresh_rules_per_transition=int(fresh_rules_per_transition),
            k_in_min=int(k_in_min),
            k_in_max=int(k_in_max),
            k_out_min=int(k_out_min),
            k_out_max=int(k_out_max),
            predicate_name_len=int(predicate_name_len),
            demo_distribution=str(demo_distribution),
            demo_distribution_alpha=float(demo_distribution_alpha),
            demo_ranked=bool(demo_ranked),
            demo_all=bool(demo_all),
            demo_unique=bool(demo_unique),
            cluster_n_samples=int(cluster_n_samples),
            cluster_k=int(cluster_k),
            cluster_base_dist=str(cluster_base_dist),
            cluster_unselected_rank=cluster_unselected_rank,
        )
        strategy = cls(
            rule_bank=base_bank,
            tokenizer=tokenizer,
            sample_config=sample_config,
            base_bank=base_bank,
            online_forced_step_idx=online_forced_step_idx,
        )
        if str(demo_distribution) == "cluster" and int(cluster_n_samples) > 0:
            strategy._init_cluster_precomputation(
                seed=int(seed),
                cluster_n_samples=int(cluster_n_samples),
                max_unify_solutions=int(max_unify_solutions),
                initial_ant_max=int(initial_ant_max),
            )
        return strategy

    def _init_cluster_precomputation(
        self,
        *,
        seed: int,
        cluster_n_samples: int,
        max_unify_solutions: int,
        initial_ant_max: int,
    ) -> None:
        """Precompute cluster candidate rankings for base-bank layers.

        For fresh-ICL, layers >= 1 use the base bank's transition rules
        (shared across all temp banks).  We precompute candidate rankings
        for these layers once and reuse them for every training sample,
        avoiding the expensive per-sample ``_precompute_cluster_candidate_rankings``
        call for ~50% of samples (those at src_layer >= 1).
        """
        if self.base_bank is None:
            return
        precomp_rng = np.random.default_rng(int(seed) + 7_919)
        by_layer: dict[int, list] = {}
        distance = int(self.sample_config.distances[0])
        for src_layer in range(1, self.base_bank.n_layers - 1):
            rules = list(self.base_bank.transition_rules(src_layer))
            if not rules:
                continue
            # Sample a dummy problem to get a valid fallback ranked dict.
            dummy_ranked = {1: [], 2: [], 3: [], 4: list(rules)}
            candidates = _precompute_cluster_candidate_rankings(
                rule_bank=self.base_bank,
                src_layer=src_layer,
                rules=rules,
                actual_ranked=dummy_ranked,
                rng=precomp_rng,
                cluster_n_samples=cluster_n_samples,
                max_unify_solutions=max_unify_solutions,
                distance=distance,
                initial_ant_max=initial_ant_max,
            )
            by_layer[src_layer] = candidates
        self._precomputed_cluster_candidates_by_layer = by_layer or None

    def sample_record(self, *, rng: np.random.Generator) -> dict:
        if self.base_bank is None:
            raise RuntimeError("depth3_fresh_icl requires a base bank.")
        return sample_online_fresh_record(
            base_bank=self.base_bank,
            tokenizer=self.tokenizer,
            rng=rng,
            config=self.sample_config,
            precomputed_cluster_candidates_by_layer=self._precomputed_cluster_candidates_by_layer,
        )

    def make_worker_spec(self) -> OnlineWorkerSpec:
        if self.base_bank is None:
            raise RuntimeError("Fresh-ICL prefetch requires base bank metadata.")
        return OnlineWorkerSpec(
            init_fn=_init_fol_online_worker,
            sample_records_fn=_sample_fol_online_worker_records,
            initargs=(
                self.sample_config.to_dict(),
                self.base_bank.to_dict(),
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
        if self.base_bank is None:
            raise RuntimeError("Fresh-ICL server prefetch requires base bank metadata.")
        return {
            "config_payload": self.sample_config.to_dict(),
            "bank_payload": self.base_bank.to_dict(),
            "tokenizer_payload": self.tokenizer.to_dict(),
            "workers": int(workers),
            "buffer_size": int(buffer_size),
            "batch_size": int(batch_size),
        }
