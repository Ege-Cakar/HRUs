"""Generate ArrayRecord datasets for both FOL task conditions.

Produces offline shard datasets for the **internalized** (no demos) and
**ICL** (demos in prompt) conditions from a single shared rule bank.

Usage
-----
::

    python -m task.layer_gen.generate_fol_dataset \\
        --out-dir data/fol_seed42 \\
        --seed 42 \\
        --conditions internalized,icl \\
        --min-distance 1 --max-distance 8 \\
        --examples-per-distance 50000 \\
        --workers 32 \\
        --max-n-demos 16 \\
        --demo-distribution zipf_per_rule \\
        --demo-alpha 1.0

Output structure::

    data/fol_seed42/
      rule_bank.json
      internalized/
        distance_001/ ... distance_008/
        metadata.json
      icl/
        distance_001/ ... distance_008/
        metadata.json
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import pickle
import time

import numpy as np
from tqdm import tqdm

# Reuse infrastructure from the existing generation module.
from task.layer_gen.generate_layer_fol import (
    ArrayRecordShardWriter,
    _AutoregStats,
    _DistanceResult,
    _tokenize_sampled_completion,
    _update_autoreg_stats,
    _write_metadata,
)
from task.layer_gen.util.fol_rule_bank import (
    FOLRuleBank,
    FOLSequent,
    build_random_fol_rule_bank,
    load_fol_rule_bank,
    sample_fol_problem,
    save_fol_rule_bank,
)
from task.layer_gen.util import tokenize_layer_fol
from task.layer_fol.demos._core import augment_prompt_with_demos


# ── Argument parsing ─────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate offline FOL datasets for internalized and ICL conditions.",
    )
    # Output
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)

    # Conditions
    p.add_argument(
        "--conditions",
        type=str,
        default="internalized,icl",
        help="Comma-separated list of conditions to generate (internalized, icl).",
    )

    # Rule bank
    p.add_argument("--rule-bank-path", type=Path, default=None)
    p.add_argument("--n-layers", type=int, default=16)
    p.add_argument("--predicates-per-layer", type=int, default=8)
    p.add_argument("--rules-per-transition", type=int, default=32)
    p.add_argument("--arity-min", type=int, default=1)
    p.add_argument("--arity-max", type=int, default=3)
    p.add_argument("--vars-per-rule-max", type=int, default=4)
    p.add_argument("--constants", type=str, default="a,b,c,d")
    p.add_argument("--k-in-min", type=int, default=1)
    p.add_argument("--k-in-max", type=int, default=3)
    p.add_argument("--k-out-min", type=int, default=1)
    p.add_argument("--k-out-max", type=int, default=3)

    # Sampling
    p.add_argument("--min-distance", type=int, default=1)
    p.add_argument("--max-distance", type=int, default=4)
    p.add_argument("--examples-per-distance", type=int, default=10_000)
    p.add_argument("--examples-per-shard", type=int, default=50_000)
    p.add_argument("--initial-ant-max", type=int, default=3)
    p.add_argument("--sample-max-attempts", type=int, default=4096)
    p.add_argument("--max-unify-solutions", type=int, default=128)
    p.add_argument("--completion-format", type=str, default="single")
    p.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 4))

    # ICL-specific
    p.add_argument("--max-n-demos", type=int, default=16)
    p.add_argument("--min-n-demos", type=int, default=1)
    p.add_argument("--demo-distribution", type=str, default="zipf_per_rule")
    p.add_argument("--demo-alpha", type=float, default=1.0)
    p.add_argument("--demo-ranked", action="store_true", default=True)
    p.add_argument("--no-demo-ranked", action="store_false", dest="demo_ranked")
    p.add_argument("--demo-ranking-beta", type=float, default=None)
    p.add_argument("--demo-unique", action="store_true", default=True)
    p.add_argument("--include-oracle", action="store_true", default=False)

    # Writer
    p.add_argument(
        "--arrayrecord-options",
        type=str,
        default="group_size:1",
    )

    return p.parse_args()


def _parse_constants(raw: str) -> tuple[str, ...]:
    return tuple(tok.strip() for tok in str(raw).split(",") if tok.strip())


def _parse_conditions(raw: str) -> list[str]:
    conditions = [c.strip() for c in raw.split(",") if c.strip()]
    for c in conditions:
        if c not in ("internalized", "icl"):
            raise ValueError(f"Unknown condition: {c!r}. Must be 'internalized' or 'icl'.")
    return conditions


# ── Distance seed derivation ─────────────────────────────────────────────────

def _distance_seed(base: int, distance: int, condition_salt: int = 0) -> int:
    return int(base) + int(distance) * 1_000_003 + int(condition_salt)


# ── Internalized generation (no demos) ───────────────────────────────────────

def _generate_internalized_distance(
    *,
    out_dir: str,
    distance: int,
    examples_per_distance: int,
    examples_per_shard: int,
    writer_options: str,
    initial_ant_max: int,
    sample_max_attempts: int,
    max_unify_solutions: int,
    seed: int,
    bank_payload: dict,
    completion_format: str,
) -> _DistanceResult:
    """Generate shards for one distance — internalized condition (no demos)."""
    rng = np.random.default_rng(seed)
    bank = FOLRuleBank.from_dict(bank_payload)
    tokenizer = tokenize_layer_fol.build_tokenizer_from_rule_bank(bank)
    writer = ArrayRecordShardWriter(
        out_dir=Path(out_dir) / f"distance_{distance:03d}",
        examples_per_shard=examples_per_shard,
        writer_options=writer_options,
    )
    stats = _AutoregStats()
    examples = 0

    try:
        for _ in range(examples_per_distance):
            sampled = sample_fol_problem(
                bank=bank,
                distance=distance,
                initial_ant_max=initial_ant_max,
                rng=rng,
                max_attempts=sample_max_attempts,
                max_unify_solutions=max_unify_solutions,
            )
            for step_idx, (src_layer, ants, inst_rule) in enumerate(
                zip(sampled.step_layers, sampled.step_ants, sampled.step_rules)
            ):
                sequent = FOLSequent(ants=ants, cons=sampled.goal_atom)
                prompt, completion, statement_texts = _tokenize_sampled_completion(
                    tokenizer=tokenizer,
                    sequent=sequent,
                    sampled=sampled,
                    step_idx=step_idx,
                    completion_format=completion_format,
                )
                payload = pickle.dumps(
                    {
                        "condition": "internalized",
                        "distance": int(sampled.distance),
                        "start_layer": int(sampled.start_layer),
                        "src_layer": int(src_layer),
                        "step_idx": int(step_idx),
                        "goal_atom": sampled.goal_atom.text,
                        "prompt": np.asarray(prompt, dtype=np.int32),
                        "completions": [np.asarray(completion, dtype=np.int32)],
                        "statement_text": inst_rule.statement_text,
                        "statement_texts": list(statement_texts),
                        "completion_format": str(completion_format),
                    },
                    protocol=5,
                )
                writer.write(payload)
                _update_autoreg_stats(stats, prompt, completion)
            examples += 1
    finally:
        writer.close()

    return _DistanceResult(
        distance=distance,
        examples=examples,
        records=writer.total_records,
        shards=writer.shard_count,
        stats=stats,
    )


# ── ICL generation (with demos) ──────────────────────────────────────────────

def _generate_icl_distance(
    *,
    out_dir: str,
    distance: int,
    examples_per_distance: int,
    examples_per_shard: int,
    writer_options: str,
    initial_ant_max: int,
    sample_max_attempts: int,
    max_unify_solutions: int,
    seed: int,
    bank_payload: dict,
    completion_format: str,
    # ICL params
    max_n_demos: int,
    min_n_demos: int,
    demo_distribution: str,
    demo_distribution_alpha: float,
    demo_ranked: bool,
    demo_ranking_beta: float | None,
    demo_unique: bool,
    include_oracle: bool,
) -> _DistanceResult:
    """Generate shards for one distance — ICL condition (demos in prompt)."""
    rng = np.random.default_rng(seed)
    bank = FOLRuleBank.from_dict(bank_payload)
    tokenizer = tokenize_layer_fol.build_tokenizer_from_rule_bank(bank)
    writer = ArrayRecordShardWriter(
        out_dir=Path(out_dir) / f"distance_{distance:03d}",
        examples_per_shard=examples_per_shard,
        writer_options=writer_options,
    )
    stats = _AutoregStats()
    examples = 0

    try:
        for _ in range(examples_per_distance):
            sampled = sample_fol_problem(
                bank=bank,
                distance=distance,
                initial_ant_max=initial_ant_max,
                rng=rng,
                max_attempts=sample_max_attempts,
                max_unify_solutions=max_unify_solutions,
            )
            for step_idx, (src_layer, ants, inst_rule) in enumerate(
                zip(sampled.step_layers, sampled.step_ants, sampled.step_rules)
            ):
                sequent = FOLSequent(ants=ants, cons=sampled.goal_atom)
                prompt, completion, statement_texts = _tokenize_sampled_completion(
                    tokenizer=tokenizer,
                    sequent=sequent,
                    sampled=sampled,
                    step_idx=step_idx,
                    completion_format=completion_format,
                )

                # Augment prompt with demonstrations.
                augmented = augment_prompt_with_demos(
                    prompt_tokens=prompt,
                    rule_bank=bank,
                    tokenizer=tokenizer,
                    rng=rng,
                    src_layer=int(src_layer),
                    ants=ants,
                    max_n_demos=max_n_demos,
                    min_n_demos=min_n_demos,
                    max_unify_solutions=max_unify_solutions,
                    include_oracle=include_oracle,
                    oracle_rule=inst_rule if include_oracle else None,
                    demo_distribution=demo_distribution,
                    demo_distribution_alpha=demo_distribution_alpha,
                    goal_atom=sampled.goal_atom,
                    demo_ranked=demo_ranked,
                    demo_ranking_beta=demo_ranking_beta,
                    demo_unique=demo_unique,
                )
                augmented_prompt = augmented.prompt_tokens

                payload = pickle.dumps(
                    {
                        "condition": "icl",
                        "distance": int(sampled.distance),
                        "start_layer": int(sampled.start_layer),
                        "src_layer": int(src_layer),
                        "step_idx": int(step_idx),
                        "goal_atom": sampled.goal_atom.text,
                        "prompt": np.asarray(augmented_prompt, dtype=np.int32),
                        "completions": [np.asarray(completion, dtype=np.int32)],
                        "statement_text": inst_rule.statement_text,
                        "statement_texts": list(statement_texts),
                        "completion_format": str(completion_format),
                        "demo_count": len(augmented.demo_schemas),
                        "demo_ranks": list(int(r) for r in augmented.demo_ranks),
                    },
                    protocol=5,
                )
                writer.write(payload)
                _update_autoreg_stats(stats, augmented_prompt, completion)
            examples += 1
    finally:
        writer.close()

    return _DistanceResult(
        distance=distance,
        examples=examples,
        records=writer.total_records,
        shards=writer.shard_count,
        stats=stats,
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def _build_or_load_bank(args, rng) -> FOLRuleBank:
    if args.rule_bank_path is not None:
        return load_fol_rule_bank(args.rule_bank_path)
    constants = _parse_constants(args.constants)
    return build_random_fol_rule_bank(
        n_layers=args.n_layers,
        predicates_per_layer=args.predicates_per_layer,
        rules_per_transition=args.rules_per_transition,
        arity_min=args.arity_min,
        arity_max=args.arity_max,
        vars_per_rule_max=args.vars_per_rule_max,
        constants=constants,
        k_in_min=args.k_in_min,
        k_in_max=args.k_in_max,
        k_out_min=args.k_out_min,
        k_out_max=args.k_out_max,
        rng=rng,
    )


def _run_condition(
    *,
    condition: str,
    out_dir: Path,
    distances: list[int],
    args: argparse.Namespace,
    bank_payload: dict,
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer,
    max_workers: int,
) -> dict[int, _DistanceResult]:
    """Generate all distances for one condition."""
    condition_dir = out_dir / condition
    condition_dir.mkdir(parents=True, exist_ok=True)

    condition_salt = 0 if condition == "internalized" else 7_000_000
    total = len(distances) * args.examples_per_distance

    # Choose executor.
    executor = None
    backend = "single"
    if max_workers == 1:
        executor = ThreadPoolExecutor(max_workers=1)
        backend = "thread"
    else:
        try:
            executor = ProcessPoolExecutor(max_workers=max_workers)
            backend = "process"
        except (PermissionError, OSError):
            executor = ThreadPoolExecutor(max_workers=max_workers)
            backend = "thread"

    results: dict[int, _DistanceResult] = {}
    with tqdm(total=total, desc=f"generate {condition}", leave=True) as bar:
        with executor as pool:
            futures = {}
            for d in distances:
                d_seed = _distance_seed(args.seed, d, condition_salt)
                common_kwargs = dict(
                    out_dir=str(condition_dir),
                    distance=d,
                    examples_per_distance=args.examples_per_distance,
                    examples_per_shard=args.examples_per_shard,
                    writer_options=args.arrayrecord_options,
                    initial_ant_max=args.initial_ant_max,
                    sample_max_attempts=args.sample_max_attempts,
                    max_unify_solutions=args.max_unify_solutions,
                    seed=d_seed,
                    bank_payload=bank_payload,
                    completion_format=args.completion_format,
                )
                if condition == "internalized":
                    future = pool.submit(
                        _generate_internalized_distance,
                        **common_kwargs,
                    )
                else:
                    future = pool.submit(
                        _generate_icl_distance,
                        **common_kwargs,
                        max_n_demos=args.max_n_demos,
                        min_n_demos=args.min_n_demos,
                        demo_distribution=args.demo_distribution,
                        demo_distribution_alpha=args.demo_alpha,
                        demo_ranked=args.demo_ranked,
                        demo_ranking_beta=args.demo_ranking_beta,
                        demo_unique=args.demo_unique,
                        include_oracle=args.include_oracle,
                    )
                futures[future] = d

            for future in as_completed(futures):
                result = future.result()
                results[result.distance] = result
                bar.update(result.examples)

    # Write metadata for this condition.
    config = {
        "condition": condition,
        "n_layers": args.n_layers,
        "predicates_per_layer": args.predicates_per_layer,
        "rules_per_transition": args.rules_per_transition,
        "arity_min": args.arity_min,
        "arity_max": args.arity_max,
        "completion_format": args.completion_format,
    }
    if condition == "icl":
        config.update({
            "max_n_demos": args.max_n_demos,
            "min_n_demos": args.min_n_demos,
            "demo_distribution": args.demo_distribution,
            "demo_distribution_alpha": args.demo_alpha,
            "demo_ranked": args.demo_ranked,
            "demo_unique": args.demo_unique,
            "include_oracle": args.include_oracle,
        })

    _write_metadata(
        out_dir=condition_dir,
        results=results,
        tokenizer=tokenizer,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        seed=args.seed,
        examples_per_distance=args.examples_per_distance,
        min_distance=args.min_distance,
        max_distance=args.max_distance,
        rule_bank_path=str((out_dir / "rule_bank.json").resolve()),
        workers=max_workers,
        parallel_backend=backend,
        config=config,
    )
    return results


def main() -> None:
    args = _parse_args()
    conditions = _parse_conditions(args.conditions)

    if args.min_distance < 1:
        raise ValueError("--min-distance must be >= 1")
    if args.max_distance < args.min_distance:
        raise ValueError("--max-distance must be >= --min-distance")

    distances = list(range(args.min_distance, args.max_distance + 1))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    bank = _build_or_load_bank(args, rng)
    tokenizer = tokenize_layer_fol.build_tokenizer_from_rule_bank(bank)

    # Save the shared rule bank.
    rule_bank_path = args.out_dir / "rule_bank.json"
    save_fol_rule_bank(rule_bank_path, bank)
    print(f"Rule bank saved to {rule_bank_path}")

    bank_payload = bank.to_dict()
    max_workers = max(
        1,
        min(args.workers, len(distances), os.cpu_count() or args.workers),
    )

    for condition in conditions:
        print(f"\n{'='*60}")
        print(f"Generating condition: {condition}")
        print(f"{'='*60}")
        results = _run_condition(
            condition=condition,
            out_dir=args.out_dir,
            distances=distances,
            args=args,
            bank_payload=bank_payload,
            tokenizer=tokenizer,
            max_workers=max_workers,
        )
        total_records = sum(r.records for r in results.values())
        total_shards = sum(r.shards for r in results.values())
        print(f"  {condition}: {total_records} records in {total_shards} shards")

    print(f"\nDone. Output: {args.out_dir}")


if __name__ == "__main__":
    main()
