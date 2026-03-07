"""Online sampling helpers for layered FOL tasks."""

from __future__ import annotations

import os
import threading

import numpy as np

from task.layer_gen.util.fol_completion import sampled_completion_texts
from task.layer_gen.util import tokenize_layer_fol
from task.layer_gen.util.fol_rule_bank import (
    FOLRuleBank,
    FOLLayerRule,
    FOLSequent,
    build_fresh_layer0_bank,
    generate_fresh_predicate_names,
    sample_fol_problem,
)
from .demos import augment_prompt_with_demos
from .task_shared import FreshOnlineSampleConfig, OnlineSampleConfig


_FOL_ONLINE_WORKER_LOCAL = threading.local()


def pick_sampled_step_index(
    *,
    rng: np.random.Generator,
    n_steps: int,
    forced_step_idx: int | None,
) -> int:
    if int(n_steps) < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if forced_step_idx is None:
        return int(rng.integers(0, int(n_steps)))
    forced = int(forced_step_idx)
    if forced < 0 or forced >= int(n_steps):
        raise ValueError(
            f"forced_step_idx={forced} is out of range for sampled problem with {n_steps} steps."
        )
    return forced


def _rule_texts(rules: tuple[FOLLayerRule, ...] | list[FOLLayerRule]) -> list[str]:
    return [str(rule.statement_text) for rule in rules]


def tokenize_sampled_completion(
    *,
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer,
    sequent: FOLSequent,
    sampled,
    step_idx: int,
    completion_format: str,
) -> tuple[list[int], list[int], list[str]]:
    prompt = tokenizer.tokenize_prompt(sequent)
    statements = sampled_completion_texts(
        sampled=sampled,
        step_idx=int(step_idx),
        completion_format=completion_format,
    )
    completion = tokenizer.encode_completion_texts(statements)
    return prompt, completion, statements


def _sample_problem_with_retry(
    *,
    bank: FOLRuleBank,
    distance: int,
    initial_ant_max: int,
    rng: np.random.Generator,
    sample_max_attempts: int,
    max_unify_solutions: int,
    failure_label: str,
):
    sampled = None
    last_err = None
    for _ in range(3):
        try:
            sampled = sample_fol_problem(
                bank=bank,
                distance=int(distance),
                initial_ant_max=int(initial_ant_max),
                rng=rng,
                max_attempts=int(sample_max_attempts),
                max_unify_solutions=int(max_unify_solutions),
            )
            break
        except RuntimeError as err:
            last_err = err

    if sampled is None:
        raise RuntimeError(
            f"Failed to sample {failure_label} for distance={int(distance)} "
            f"after 3 retries with max_attempts={int(sample_max_attempts)}."
        ) from last_err
    return sampled


def _base_record(
    *,
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer,
    rule_bank: FOLRuleBank,
    rng: np.random.Generator,
    sampled,
    distance: int,
    config: OnlineSampleConfig,
) -> tuple[dict, int, tuple]:
    step_idx = pick_sampled_step_index(
        rng=rng,
        n_steps=len(sampled.step_rules),
        forced_step_idx=config.forced_step_idx,
    )
    src_layer = sampled.step_layers[step_idx]
    ants = sampled.step_ants[step_idx]
    sequent = FOLSequent(ants=ants, cons=sampled.goal_atom)
    prompt, completion, completion_statements = tokenize_sampled_completion(
        tokenizer=tokenizer,
        sequent=sequent,
        sampled=sampled,
        step_idx=step_idx,
        completion_format=config.completion_format,
    )
    augmented = augment_prompt_with_demos(
        prompt_tokens=prompt,
        rule_bank=rule_bank,
        tokenizer=tokenizer,
        rng=rng,
        src_layer=int(src_layer),
        ants=ants,
        max_n_demos=int(config.max_n_demos),
        min_n_demos=int(config.min_n_demos),
        max_unify_solutions=int(config.max_unify_solutions),
    )
    record = {
        "distance": int(distance),
        "src_layer": int(src_layer),
        "completion_format": str(config.completion_format),
        "prompt": np.asarray(augmented.prompt_tokens, dtype=np.int32),
        "completions": [np.asarray(completion, dtype=np.int32)],
        "statement_texts": list(completion_statements),
    }
    return record, int(src_layer), tuple(augmented.demo_schemas), tuple(augmented.demo_instances)


def sample_online_record(
    *,
    bank: FOLRuleBank,
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer,
    rng: np.random.Generator,
    config: OnlineSampleConfig,
) -> dict:
    distance = int(rng.choice(config.distances))
    sampled = _sample_problem_with_retry(
        bank=bank,
        distance=distance,
        initial_ant_max=config.initial_ant_max,
        rng=rng,
        sample_max_attempts=config.sample_max_attempts,
        max_unify_solutions=config.max_unify_solutions,
        failure_label="online FOLLayerTask record",
    )
    record, _, _, _ = _base_record(
        tokenizer=tokenizer,
        rule_bank=bank,
        rng=rng,
        sampled=sampled,
        distance=distance,
        config=config,
    )
    return record


def fresh_rule_context(
    *,
    base_bank: FOLRuleBank,
    temp_bank: FOLRuleBank,
    src_layer: int,
    demo_schemas: tuple[FOLLayerRule, ...],
    demo_instances: tuple[str, ...],
) -> dict:
    src_layer = int(src_layer)
    fixed_rules = tuple(base_bank.transition_rules(src_layer)) if src_layer == 1 else ()
    return {
        "src_layer": src_layer,
        "active_rule_texts": _rule_texts(tuple(temp_bank.transition_rules(src_layer))),
        "fixed_rule_texts": _rule_texts(tuple(fixed_rules)),
        "demo_schema_texts": _rule_texts(tuple(demo_schemas)),
        "demo_instance_texts": [str(text) for text in demo_instances],
    }


def sample_online_fresh_record(
    *,
    base_bank: FOLRuleBank,
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer,
    rng: np.random.Generator,
    config: FreshOnlineSampleConfig,
) -> dict:
    fresh_preds = generate_fresh_predicate_names(
        config.fresh_icl_n_predicates,
        rng,
        name_len=config.predicate_name_len,
    )
    temp_bank = build_fresh_layer0_bank(
        base_bank=base_bank,
        fresh_predicates=fresh_preds,
        rules_per_transition=config.rules_per_transition,
        k_in_min=config.k_in_min,
        k_in_max=config.k_in_max,
        k_out_min=config.k_out_min,
        k_out_max=config.k_out_max,
        rng=rng,
    )
    distance = 2
    sampled = _sample_problem_with_retry(
        bank=temp_bank,
        distance=distance,
        initial_ant_max=config.initial_ant_max,
        rng=rng,
        sample_max_attempts=config.sample_max_attempts,
        max_unify_solutions=config.max_unify_solutions,
        failure_label="fresh-ICL FOLLayerTask record",
    )
    record, src_layer, demo_schemas, demo_instances = _base_record(
        tokenizer=tokenizer,
        rule_bank=temp_bank,
        rng=rng,
        sampled=sampled,
        distance=distance,
        config=config,
    )
    record["rule_context"] = fresh_rule_context(
        base_bank=base_bank,
        temp_bank=temp_bank,
        src_layer=int(src_layer),
        demo_schemas=demo_schemas,
        demo_instances=demo_instances,
    )
    return record


def _worker_seed(seed_base: int) -> int:
    return (
        int(seed_base)
        + int(os.getpid()) * 1_000_003
        + int(threading.get_ident() % 1_000_003)
    )


def _init_fol_online_worker(
    seed_base: int,
    bank_payload: dict,
    tokenizer_payload: dict | None,
    distances: tuple[int, ...],
    initial_ant_max: int,
    sample_max_attempts: int,
    max_unify_solutions: int,
    max_n_demos: int,
    min_n_demos: int,
    forced_step_idx: int | None,
    completion_format: str = "single",
) -> None:
    bank = FOLRuleBank.from_dict(bank_payload)
    if tokenizer_payload is None:
        tokenizer = tokenize_layer_fol.build_tokenizer_from_rule_bank(bank)
    else:
        tokenizer = tokenize_layer_fol.FOLLayerTokenizer.from_dict(tokenizer_payload)
    _FOL_ONLINE_WORKER_LOCAL.state = {
        "bank": bank,
        "tokenizer": tokenizer,
        "config": OnlineSampleConfig(
            seed_base=int(seed_base),
            distances=tuple(int(distance) for distance in distances),
            initial_ant_max=int(initial_ant_max),
            sample_max_attempts=int(sample_max_attempts),
            max_unify_solutions=int(max_unify_solutions),
            max_n_demos=int(max_n_demos),
            min_n_demos=int(min_n_demos),
            forced_step_idx=(
                None if forced_step_idx is None else int(forced_step_idx)
            ),
            completion_format=str(completion_format),
        ),
        "rng": np.random.default_rng(_worker_seed(seed_base)),
    }


def _sample_fol_online_worker_record() -> dict:
    state = getattr(_FOL_ONLINE_WORKER_LOCAL, "state", None)
    if state is None:
        raise RuntimeError("FOL online worker state was not initialized.")
    return sample_online_record(
        bank=state["bank"],
        tokenizer=state["tokenizer"],
        rng=state["rng"],
        config=state["config"],
    )


def _sample_fol_online_worker_records(n_records: int) -> list[dict]:
    n_records = int(n_records)
    if n_records < 1:
        raise ValueError(f"n_records must be >= 1, got {n_records}")
    return [_sample_fol_online_worker_record() for _ in range(n_records)]


def _init_fol_online_fresh_worker(
    seed_base: int,
    base_bank_payload: dict,
    tokenizer_payload: dict,
    fresh_icl_n_predicates: int,
    rules_per_transition: int,
    k_in_min: int,
    k_in_max: int,
    k_out_min: int,
    k_out_max: int,
    initial_ant_max: int,
    sample_max_attempts: int,
    max_unify_solutions: int,
    max_n_demos: int,
    min_n_demos: int,
    forced_step_idx: int | None,
    completion_format: str,
    predicate_name_len: int = 1,
) -> None:
    _FOL_ONLINE_WORKER_LOCAL.state = {
        "base_bank": FOLRuleBank.from_dict(base_bank_payload),
        "tokenizer": tokenize_layer_fol.FOLLayerTokenizer.from_dict(tokenizer_payload),
        "config": FreshOnlineSampleConfig(
            seed_base=int(seed_base),
            distances=(2,),
            initial_ant_max=int(initial_ant_max),
            sample_max_attempts=int(sample_max_attempts),
            max_unify_solutions=int(max_unify_solutions),
            max_n_demos=int(max_n_demos),
            min_n_demos=int(min_n_demos),
            forced_step_idx=(
                None if forced_step_idx is None else int(forced_step_idx)
            ),
            completion_format=str(completion_format),
            fresh_icl_n_predicates=int(fresh_icl_n_predicates),
            rules_per_transition=int(rules_per_transition),
            k_in_min=int(k_in_min),
            k_in_max=int(k_in_max),
            k_out_min=int(k_out_min),
            k_out_max=int(k_out_max),
            predicate_name_len=int(predicate_name_len),
        ),
        "rng": np.random.default_rng(_worker_seed(seed_base)),
    }


def _sample_fol_online_fresh_worker_record() -> dict:
    state = getattr(_FOL_ONLINE_WORKER_LOCAL, "state", None)
    if state is None:
        raise RuntimeError("FOL online fresh worker state was not initialized.")
    return sample_online_fresh_record(
        base_bank=state["base_bank"],
        tokenizer=state["tokenizer"],
        rng=state["rng"],
        config=state["config"],
    )


def _sample_fol_online_fresh_worker_records(n_records: int) -> list[dict]:
    n_records = int(n_records)
    if n_records < 1:
        raise ValueError(f"n_records must be >= 1, got {n_records}")
    return [_sample_fol_online_fresh_worker_record() for _ in range(n_records)]
