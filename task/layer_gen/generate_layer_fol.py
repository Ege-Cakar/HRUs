"""Generate ArrayRecord datasets for layered first-order tasks."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import os
from pathlib import Path
import pickle
import time

import numpy as np
from tqdm import tqdm

from array_record.python import array_record_module

if __package__ in (None, ""):
    from util.fol_rule_bank import (  # type: ignore
        FOLRuleBank,
        FOLSequent,
        build_random_fol_rule_bank,
        load_fol_rule_bank,
        sample_fol_problem,
        save_fol_rule_bank,
    )
    from util import tokenize_layer_fol  # type: ignore
else:
    from .util.fol_rule_bank import (
        FOLRuleBank,
        FOLSequent,
        build_random_fol_rule_bank,
        load_fol_rule_bank,
        sample_fol_problem,
        save_fol_rule_bank,
    )
    from .util import tokenize_layer_fol


@dataclass
class _AutoregStats:
    max_token: int = 0
    max_seq: int = 0
    max_prompt_seq: int = 0
    max_completion_seq: int = 0


@dataclass(frozen=True)
class _DistanceResult:
    distance: int
    examples: int
    records: int
    shards: int
    stats: _AutoregStats


class ArrayRecordShardWriter:
    def __init__(
        self,
        out_dir: Path,
        examples_per_shard: int,
        writer_options: str,
    ) -> None:
        self.out_dir = out_dir
        self.examples_per_shard = max(1, examples_per_shard)
        self.writer_options = writer_options
        self.shard_idx = 0
        self.total_records = 0
        self._records_in_shard = 0
        self._writer = None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._open_writer()

    def _shard_path(self) -> Path:
        return self.out_dir / f"shard_{self.shard_idx:05d}.array_record"

    def _open_writer(self) -> None:
        self._writer = array_record_module.ArrayRecordWriter(
            str(self._shard_path()),
            self.writer_options,
        )

    def _close_writer(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def _rollover(self) -> None:
        self._close_writer()
        self.shard_idx += 1
        self._records_in_shard = 0
        self._open_writer()

    def write(self, payload: bytes) -> None:
        if self._records_in_shard >= self.examples_per_shard:
            self._rollover()
        self._writer.write(payload)
        self._records_in_shard += 1
        self.total_records += 1

    def close(self) -> None:
        self._close_writer()

    @property
    def shard_count(self) -> int:
        if self.total_records == 0:
            return 0
        return self.shard_idx + 1


def _default_seed() -> int:
    return int(np.random.randint(0, np.iinfo(np.int32).max))


def _parse_constants(raw: str) -> tuple[str, ...]:
    constants = tuple(tok.strip() for tok in str(raw).split(",") if tok.strip())
    if not constants:
        raise ValueError("--constants must contain at least one symbol")
    return constants


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate layered first-order datasets.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)

    parser.add_argument("--n-layers", type=int, default=16)
    parser.add_argument("--predicates-per-layer", type=int, default=8)
    parser.add_argument("--rules-per-transition", type=int, default=32)
    parser.add_argument("--k-in-max", type=int, default=3)
    parser.add_argument("--k-out-max", type=int, default=3)
    parser.add_argument("--arity-max", type=int, default=3)
    parser.add_argument("--vars-per-rule-max", type=int, default=4)
    parser.add_argument("--constants", type=str, default="a,b,c,d")
    parser.add_argument("--initial-ant-max", type=int, default=3)
    parser.add_argument("--sample-max-attempts", type=int, default=4096)
    parser.add_argument("--max-unify-solutions", type=int, default=128)

    parser.add_argument("--min-distance", type=int, default=1)
    parser.add_argument("--max-distance", type=int, default=4)
    parser.add_argument("--examples-per-distance", type=int, default=10_000)
    parser.add_argument("--examples-per-shard", type=int, default=50_000)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 4))
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--rule-bank-path", type=Path, default=None)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--arrayrecord-options",
        type=str,
        default="group_size:1",
        help="ArrayRecord writer options string.",
    )
    return parser.parse_args()


def _update_autoreg_stats(stats: _AutoregStats, prompt: list[int], completion: list[int]) -> None:
    if prompt:
        stats.max_token = max(stats.max_token, max(prompt))
    if completion:
        stats.max_token = max(stats.max_token, max(completion))

    stats.max_prompt_seq = max(stats.max_prompt_seq, len(prompt))
    stats.max_completion_seq = max(stats.max_completion_seq, len(completion))
    stats.max_seq = max(stats.max_seq, len(prompt) + len(completion) - 1)


def _build_or_load_rule_bank(args: argparse.Namespace, rng: np.random.Generator) -> FOLRuleBank:
    if args.rule_bank_path is not None:
        return load_fol_rule_bank(args.rule_bank_path)

    constants = _parse_constants(args.constants)
    return build_random_fol_rule_bank(
        n_layers=int(args.n_layers),
        predicates_per_layer=int(args.predicates_per_layer),
        rules_per_transition=int(args.rules_per_transition),
        arity_max=int(args.arity_max),
        vars_per_rule_max=int(args.vars_per_rule_max),
        k_in_max=int(args.k_in_max),
        k_out_max=int(args.k_out_max),
        constants=constants,
        rng=rng,
    )


def _distance_seed(base_seed: int, distance: int) -> int:
    return int(base_seed + distance * 1_000_003)


def _generate_distance_data(
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
) -> _DistanceResult:
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
                distance=int(distance),
                initial_ant_max=int(initial_ant_max),
                rng=rng,
                max_attempts=int(sample_max_attempts),
                max_unify_solutions=int(max_unify_solutions),
            )

            for step_idx, (src_layer, ants, instantiated_rule) in enumerate(
                zip(sampled.step_layers, sampled.step_ants, sampled.step_rules)
            ):
                sequent = FOLSequent(ants=ants, cons=sampled.goal_atom)
                prompt, completion = tokenizer.tokenize_example(
                    sequent,
                    instantiated_rule.statement_text,
                )
                payload = pickle.dumps(
                    {
                        "distance": int(sampled.distance),
                        "start_layer": int(sampled.start_layer),
                        "src_layer": int(src_layer),
                        "step_idx": int(step_idx),
                        "goal_atom": sampled.goal_atom.text,
                        "prompt": np.asarray(prompt, dtype=np.int32),
                        "completions": [np.asarray(completion, dtype=np.int32)],
                        "statement_text": instantiated_rule.statement_text,
                    },
                    protocol=5,
                )
                writer.write(payload)
                _update_autoreg_stats(stats, prompt, completion)

            examples += 1
    finally:
        writer.close()

    return _DistanceResult(
        distance=int(distance),
        examples=int(examples),
        records=int(writer.total_records),
        shards=int(writer.shard_count),
        stats=stats,
    )


def _write_metadata(
    *,
    out_dir: Path,
    results: dict[int, _DistanceResult],
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer,
    created_at: str,
    seed: int,
    examples_per_distance: int,
    min_distance: int,
    max_distance: int,
    rule_bank_path: str,
    workers: int,
    parallel_backend: str,
    config: dict,
) -> None:
    distances = {}
    stats_list = []
    for distance in sorted(results):
        result = results[distance]
        st = result.stats
        stats_list.append(st)
        distances[str(distance)] = {
            "examples": int(result.examples),
            "records": int(result.records),
            "shards": int(result.shards),
            "stats": {
                "max_token": int(st.max_token),
                "max_seq": int(st.max_seq),
                "max_prompt_seq": int(st.max_prompt_seq),
                "max_completion_seq": int(st.max_completion_seq),
            },
        }

    overall = {
        "max_token": int(max(st.max_token for st in stats_list)),
        "max_seq": int(max(st.max_seq for st in stats_list)),
        "max_prompt_seq": int(max(st.max_prompt_seq for st in stats_list)),
        "max_completion_seq": int(max(st.max_completion_seq for st in stats_list)),
    }

    metadata = {
        "created_at": created_at,
        "format": "arrayrecord_layer_fol",
        "seed": int(seed),
        "examples_per_distance": int(examples_per_distance),
        "distance_range": [int(min_distance), int(max_distance)],
        "rule_bank": rule_bank_path,
        "workers": int(workers),
        "parallel_backend": parallel_backend,
        "config": config,
        "tokenizer": tokenizer.to_dict(),
        "stats": overall,
        "distances": distances,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def main() -> None:
    args = _parse_args()
    if args.seed is None:
        args.seed = _default_seed()

    if args.min_distance < 1:
        raise ValueError("--min-distance must be >= 1")
    if args.max_distance < args.min_distance:
        raise ValueError("--max-distance must be >= --min-distance")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    distances = list(range(args.min_distance, args.max_distance + 1))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    bank = _build_or_load_rule_bank(args, rng)
    tokenizer = tokenize_layer_fol.build_tokenizer_from_rule_bank(bank)
    rule_bank_out = args.out_dir / "rule_bank.json"
    save_fol_rule_bank(rule_bank_out, bank)

    config = {
        "n_layers": int(bank.n_layers),
        "predicates_per_layer": int(bank.predicates_per_layer),
        "rules_per_transition": int(args.rules_per_transition),
        "k_in_max": int(args.k_in_max),
        "k_out_max": int(args.k_out_max),
        "arity_max": int(bank.arity_max),
        "vars_per_rule_max": int(bank.vars_per_rule_max),
        "constants": list(bank.constants),
        "initial_ant_max": int(args.initial_ant_max),
        "sample_max_attempts": int(args.sample_max_attempts),
        "max_unify_solutions": int(args.max_unify_solutions),
    }

    created_at = time.strftime("%Y-%m-%d %H:%M:%S")
    total_target = len(distances) * int(args.examples_per_distance)
    max_workers = max(
        1,
        min(int(args.workers), len(distances), os.cpu_count() or int(args.workers)),
    )
    bank_payload = bank.to_dict()
    results: dict[int, _DistanceResult] = {}

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

    with tqdm(total=total_target, desc="generate layered fol", leave=True) as bar:
        with executor as pool:
            future_to_distance = {
                pool.submit(
                    _generate_distance_data,
                    out_dir=str(args.out_dir),
                    distance=distance,
                    examples_per_distance=int(args.examples_per_distance),
                    examples_per_shard=int(args.examples_per_shard),
                    writer_options=args.arrayrecord_options,
                    initial_ant_max=int(args.initial_ant_max),
                    sample_max_attempts=int(args.sample_max_attempts),
                    max_unify_solutions=int(args.max_unify_solutions),
                    seed=_distance_seed(int(args.seed), distance),
                    bank_payload=bank_payload,
                ): distance
                for distance in distances
            }
            for future in as_completed(future_to_distance):
                result = future.result()
                results[result.distance] = result
                bar.update(int(result.examples))

    _write_metadata(
        out_dir=args.out_dir,
        results=results,
        tokenizer=tokenizer,
        created_at=created_at,
        seed=args.seed,
        examples_per_distance=args.examples_per_distance,
        min_distance=args.min_distance,
        max_distance=args.max_distance,
        rule_bank_path=str(rule_bank_out.resolve()),
        workers=max_workers,
        parallel_backend=backend,
        config=config,
    )


if __name__ == "__main__":
    main()
