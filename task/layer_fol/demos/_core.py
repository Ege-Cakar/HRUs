"""Main entry points, adapter, and instantiation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from task.layer_gen.util import tokenize_layer_fol
from task.layer_gen.util.fol_rule_bank import FOLAtom, FOLLayerRule, FOLRuleBank

from ..eval_inputs import extract_prompt_info_from_row_tokens
from ._clustering import _sample_cluster
from ._ranking import _classify_rules_by_rank
from ._sampling import (
    _resolve_demo_ranking_beta,
    _sample_demo_schemas,
    _sample_uniform,
    _sample_zipf_ranked,
    sample_ranked_demos,
)
from ._unify import _collect_applicable_demo_schemas


@dataclass(frozen=True)
class FOLDemoAugmentationResult:
    prompt_tokens: list[int]
    demo_schemas: tuple[FOLLayerRule, ...]
    demo_instances: tuple[str, ...]
    demo_ranks: tuple[int, ...] = ()


def _instantiate_demo_schema_with_random_constants(
    *,
    rule: FOLLayerRule,
    constants: tuple[str, ...],
    rng: np.random.Generator,
) -> str:
    if not constants:
        raise ValueError("Cannot instantiate demo rule schema without constants.")

    variables = tuple(sorted(rule.variables()))
    if not variables:
        return str(rule.statement_text)

    substitution = {
        var: str(constants[int(rng.integers(0, len(constants)))])
        for var in variables
    }
    return str(rule.instantiate(substitution).statement_text)


def _prepend_demo_statements_to_prompt(
    *,
    prompt_tokens: list[int],
    demo_statements: list[str],
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer,
) -> list[int]:
    out: list[int] = []
    for statement in demo_statements:
        demo_completion = tokenizer.encode_completion_texts([statement])
        out.extend(int(tok) for tok in demo_completion[:-1])
        out.append(int(tokenizer.sep_token_id))
    out.extend(int(tok) for tok in prompt_tokens)
    return out


def augment_prompt_with_demos(
    *,
    prompt_tokens: list[int],
    rule_bank: FOLRuleBank,
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer,
    rng: np.random.Generator,
    src_layer: int,
    ants: tuple[FOLAtom, ...],
    max_n_demos: int,
    min_n_demos: int,
    max_unify_solutions: int,
    include_oracle: bool = False,
    oracle_rule: FOLLayerRule | None = None,
    demo_distribution: str = "uniform",
    demo_distribution_alpha: float = 1.0,
    goal_atom: FOLAtom | None = None,
    demo_ranked: bool = True,
    demo_all: bool = False,
    demo_unique: bool = True,
    cluster_n_samples: int = 100,
    cluster_k: int = 5,
    cluster_base_dist: str = "zipf_per_rule",
    cluster_unselected_rank: int | None = None,
    cluster_distance: int | None = None,
    cluster_initial_ant_max: int | None = None,
    precomputed_cluster_candidates: list | None = None,
    demo_ranking_beta: float | None = None,
) -> FOLDemoAugmentationResult:
    max_n_demos = int(max_n_demos)
    min_n_demos = int(min_n_demos)
    include_oracle = bool(include_oracle)
    demo_distribution = str(demo_distribution)
    demo_distribution_alpha = float(demo_distribution_alpha)
    demo_ranked = bool(demo_ranked)
    demo_all = bool(demo_all)
    demo_unique = bool(demo_unique)
    demo_ranking_beta = _resolve_demo_ranking_beta(demo_ranking_beta, demo_ranked)

    _empty = FOLDemoAugmentationResult(
        prompt_tokens=list(int(tok) for tok in prompt_tokens),
        demo_schemas=(),
        demo_instances=(),
    )

    if demo_all:
        all_rules = list(rule_bank.transition_rules(int(src_layer)))
        if not all_rules:
            return _empty
        demo_statements = [
            _instantiate_demo_schema_with_random_constants(
                rule=rule,
                constants=rule_bank.constants,
                rng=rng,
            )
            for rule in all_rules
        ]
        augmented_prompt = _prepend_demo_statements_to_prompt(
            prompt_tokens=prompt_tokens,
            demo_statements=demo_statements,
            tokenizer=tokenizer,
        )
        return FOLDemoAugmentationResult(
            prompt_tokens=augmented_prompt,
            demo_schemas=tuple(all_rules),
            demo_instances=tuple(str(s) for s in demo_statements),
            demo_ranks=(),
        )

    if demo_distribution not in {"uniform", "zipf", "zipf_headless", "zipf_per_rule", "zipf_per_rule_headless", "cluster"}:
        raise ValueError(
            f"demo_distribution must be 'uniform', 'zipf', 'zipf_headless', "
            f"'zipf_per_rule', 'zipf_per_rule_headless', or 'cluster', "
            f"got {demo_distribution!r}"
        )
    if demo_distribution in {"zipf", "zipf_headless", "zipf_per_rule", "zipf_per_rule_headless", "cluster"} and goal_atom is None:
        raise ValueError(f"demo_distribution={demo_distribution!r} requires goal_atom.")
    if min_n_demos > max_n_demos:
        raise ValueError(
            f"min_n_demos must be <= max_n_demos, got {min_n_demos} > {max_n_demos}"
        )
    if include_oracle and max_n_demos < 1:
        raise ValueError("include_oracle=True requires max_n_demos >= 1.")
    if include_oracle and oracle_rule is None:
        raise ValueError("include_oracle=True requires oracle_rule.")

    if max_n_demos <= 0:
        return _empty

    n_demos = int(rng.integers(min_n_demos, max_n_demos + 1))
    if include_oracle and n_demos < 1:
        raise ValueError("include_oracle=True requires sampling at least one demo.")
    if n_demos <= 0:
        return _empty

    if demo_distribution == "cluster":
        sampled_schemas, sampled_ranks = _sample_cluster(
            rule_bank=rule_bank,
            src_layer=int(src_layer),
            ants=ants,
            max_unify_solutions=int(max_unify_solutions),
            rng=rng,
            n_demos=n_demos,
            include_oracle=include_oracle,
            oracle_rule=oracle_rule,
            alpha=demo_distribution_alpha,
            goal_atom=goal_atom,
            demo_ranked=demo_ranked,
            demo_unique=demo_unique,
            cluster_n_samples=int(cluster_n_samples),
            cluster_k=int(cluster_k),
            cluster_base_dist=str(cluster_base_dist),
            cluster_unselected_rank=cluster_unselected_rank,
            distance=cluster_distance,
            initial_ant_max=cluster_initial_ant_max,
            precomputed_cluster_candidates=precomputed_cluster_candidates,
            demo_ranking_beta=demo_ranking_beta,
        )
    elif demo_distribution == "uniform":
        sampled_schemas, sampled_ranks = _sample_uniform(
            rule_bank=rule_bank,
            src_layer=int(src_layer),
            ants=ants,
            max_unify_solutions=int(max_unify_solutions),
            rng=rng,
            n_demos=n_demos,
            include_oracle=include_oracle,
            oracle_rule=oracle_rule,
            demo_unique=demo_unique,
        )
    else:
        per_rule = demo_distribution in {"zipf_per_rule", "zipf_per_rule_headless"}
        headless = demo_distribution in {"zipf_headless", "zipf_per_rule_headless"}
        sampled_schemas, sampled_ranks = _sample_zipf_ranked(
            rule_bank=rule_bank,
            src_layer=int(src_layer),
            ants=ants,
            max_unify_solutions=int(max_unify_solutions),
            rng=rng,
            n_demos=n_demos,
            include_oracle=include_oracle,
            oracle_rule=oracle_rule,
            alpha=demo_distribution_alpha,
            goal_atom=goal_atom,
            headless=headless,
            demo_ranked=demo_ranked,
            demo_unique=demo_unique,
            demo_ranking_beta=demo_ranking_beta,
            per_rule=per_rule,
        )

    if not sampled_schemas:
        return _empty

    demo_statements = [
        _instantiate_demo_schema_with_random_constants(
            rule=schema,
            constants=rule_bank.constants,
            rng=rng,
        )
        for schema in sampled_schemas
    ]

    augmented_prompt = _prepend_demo_statements_to_prompt(
        prompt_tokens=prompt_tokens,
        demo_statements=demo_statements,
        tokenizer=tokenizer,
    )
    return FOLDemoAugmentationResult(
        prompt_tokens=augmented_prompt,
        demo_schemas=tuple(sampled_schemas),
        demo_instances=tuple(str(statement) for statement in demo_statements),
        demo_ranks=tuple(sampled_ranks),
    )


def _augment_prompt_with_demos(
    *,
    prompt_tokens: list[int],
    rule_bank: FOLRuleBank,
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer,
    rng: np.random.Generator,
    src_layer: int,
    ants: tuple[FOLAtom, ...],
    max_n_demos: int,
    min_n_demos: int,
    max_unify_solutions: int,
    include_oracle: bool = False,
    oracle_rule: FOLLayerRule | None = None,
    demo_distribution: str = "uniform",
    demo_distribution_alpha: float = 1.0,
    goal_atom: FOLAtom | None = None,
    demo_ranked: bool = True,
    demo_all: bool = False,
    demo_unique: bool = True,
    cluster_n_samples: int = 100,
    cluster_k: int = 5,
    cluster_base_dist: str = "zipf_per_rule",
    cluster_unselected_rank: int | None = None,
    cluster_distance: int | None = None,
    cluster_initial_ant_max: int | None = None,
    demo_ranking_beta: float | None = None,
) -> list[int]:
    return augment_prompt_with_demos(
        prompt_tokens=prompt_tokens,
        rule_bank=rule_bank,
        tokenizer=tokenizer,
        rng=rng,
        src_layer=src_layer,
        ants=ants,
        max_n_demos=max_n_demos,
        min_n_demos=min_n_demos,
        max_unify_solutions=max_unify_solutions,
        include_oracle=include_oracle,
        oracle_rule=oracle_rule,
        demo_distribution=demo_distribution,
        demo_distribution_alpha=demo_distribution_alpha,
        goal_atom=goal_atom,
        demo_ranked=demo_ranked,
        demo_all=demo_all,
        demo_unique=demo_unique,
        cluster_n_samples=cluster_n_samples,
        cluster_k=cluster_k,
        cluster_base_dist=cluster_base_dist,
        cluster_unselected_rank=cluster_unselected_rank,
        cluster_distance=cluster_distance,
        cluster_initial_ant_max=cluster_initial_ant_max,
        demo_ranking_beta=demo_ranking_beta,
    ).prompt_tokens


class FOLDemoAugmentedAdapter:
    """Adapter wrapper that prepends sampled demonstration rules to prompts."""

    def __init__(
        self,
        *,
        base_adapter,
        rule_bank: FOLRuleBank,
        tokenizer,
        min_n_demos: int,
        max_n_demos: int,
        max_unify_solutions: int,
        include_oracle: bool = False,
        demo_distribution: str = "uniform",
        demo_distribution_alpha: float = 1.0,
        demo_ranked: bool = True,
        demo_all: bool = False,
        demo_unique: bool = True,
        cluster_n_samples: int = 100,
        cluster_k: int = 5,
        cluster_base_dist: str = "zipf_per_rule",
        cluster_unselected_rank: int | None = None,
        cluster_distance: int | None = None,
        cluster_initial_ant_max: int | None = None,
        demo_ranking_beta: float | None = None,
    ) -> None:
        self.base_adapter = base_adapter
        self.rule_bank = rule_bank
        self.tokenizer = tokenizer
        self.min_n_demos = int(min_n_demos)
        self.max_n_demos = int(max_n_demos)
        self.max_unify_solutions = int(max_unify_solutions)
        self.include_oracle = bool(include_oracle)
        self.demo_distribution = str(demo_distribution)
        self.demo_distribution_alpha = float(demo_distribution_alpha)
        self.demo_ranked = bool(demo_ranked)
        self.demo_all = bool(demo_all)
        self.demo_unique = bool(demo_unique)
        self.demo_ranking_beta = _resolve_demo_ranking_beta(demo_ranking_beta, demo_ranked)
        self.cluster_n_samples = int(cluster_n_samples)
        self.cluster_k = int(cluster_k)
        self.cluster_base_dist = str(cluster_base_dist)
        self.cluster_unselected_rank = (
            None if cluster_unselected_rank is None
            else int(cluster_unselected_rank)
        )
        self.cluster_distance = (
            None if cluster_distance is None
            else int(cluster_distance)
        )
        self.cluster_initial_ant_max = (
            None if cluster_initial_ant_max is None
            else int(cluster_initial_ant_max)
        )
        self._last_demo_rules: list[FOLLayerRule] = []
        self._oracle_rule: FOLLayerRule | None = None

    def get_last_demo_rules(self) -> list[FOLLayerRule]:
        return list(self._last_demo_rules)

    def set_oracle_rule(self, oracle_rule: FOLLayerRule | None) -> None:
        self._oracle_rule = oracle_rule

    def predict_completion(
        self,
        *,
        model: Callable[[np.ndarray], Any],
        prompt_tokens,
        tokenizer,
        temperature: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        self._last_demo_rules = []
        if self.max_n_demos <= 0 and not self.demo_all:
            return self.base_adapter.predict_completion(
                model=model,
                prompt_tokens=prompt_tokens,
                tokenizer=tokenizer,
                temperature=temperature,
                rng=rng,
            )

        if rng is None:
            rng = np.random.default_rng()

        prompt = np.asarray(prompt_tokens, dtype=np.int32).tolist()
        oracle_rule = self._oracle_rule
        self._oracle_rule = None
        try:
            _, sequent, src_layer, _ = extract_prompt_info_from_row_tokens(
                np.asarray(prompt, dtype=np.int32),
                tokenizer=self.tokenizer,
            )
        except ValueError:
            self._last_demo_rules = []
        else:
            augmented = augment_prompt_with_demos(
                prompt_tokens=prompt,
                rule_bank=self.rule_bank,
                tokenizer=self.tokenizer,
                rng=rng,
                src_layer=int(src_layer),
                ants=tuple(sequent.ants),
                min_n_demos=self.min_n_demos,
                max_n_demos=self.max_n_demos,
                max_unify_solutions=self.max_unify_solutions,
                include_oracle=self.include_oracle,
                oracle_rule=oracle_rule,
                demo_distribution=self.demo_distribution,
                demo_distribution_alpha=self.demo_distribution_alpha,
                goal_atom=sequent.cons,
                demo_ranked=self.demo_ranked,
                demo_all=self.demo_all,
                demo_unique=self.demo_unique,
                cluster_n_samples=self.cluster_n_samples,
                cluster_k=self.cluster_k,
                cluster_base_dist=self.cluster_base_dist,
                cluster_unselected_rank=self.cluster_unselected_rank,
                cluster_distance=self.cluster_distance,
                cluster_initial_ant_max=self.cluster_initial_ant_max,
                demo_ranking_beta=self.demo_ranking_beta,
            )
            prompt = augmented.prompt_tokens
            self._last_demo_rules = list(augmented.demo_schemas)

        return self.base_adapter.predict_completion(
            model=model,
            prompt_tokens=prompt,
            tokenizer=tokenizer,
            temperature=temperature,
            rng=rng,
        )
