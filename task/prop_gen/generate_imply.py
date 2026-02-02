"""
Generate propositional implicational logic dataset, based on depth
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from array_record.python import array_record_module

from util.sample import sample_imply, list_sequents
from util.tokenize import tokenize, TokenizedExample


@dataclass(frozen=True)
class GenerationStats:
    size: int
    examples: int
    shards: int
    seconds: float


class ArrayRecordShardWriter:
    def __init__(
        self,
        out_dir: Path,
        size: int,
        n_vars: int,
        examples_per_shard: int,
        writer_options: str,
    ) -> None:
        self.out_dir = out_dir
        self.size = size
        self.n_vars = n_vars
        self.examples_per_shard = max(1, examples_per_shard)
        self.writer_options = writer_options
        self.shard_idx = 0
        self.total_examples = 0
        self._examples_in_shard = 0
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
        self._examples_in_shard = 0
        self._open_writer()

    def write(self, example: TokenizedExample) -> None:
        if self._examples_in_shard >= self.examples_per_shard:
            self._rollover()
        sequent_tokens, rule_tokens = example
        sequent = np.asarray(sequent_tokens, dtype=np.int32)
        if rule_tokens:
            rules = np.asarray(rule_tokens, dtype=np.int32)
        else:
            rules = np.zeros((0, 2), dtype=np.int32)
        payload = pickle.dumps(
            {
                "sequent": sequent,
                "rules": rules,
            },
            protocol=5,
        )
        self._writer.write(payload)
        self._examples_in_shard += 1
        self.total_examples += 1

    def close(self) -> None:
        self._close_writer()

    @property
    def shard_count(self) -> int:
        if self.total_examples == 0:
            return 0
        return self.shard_idx + 1


def _size_seed(base_seed: int, size: int) -> int:
    return base_seed + size * 1_000_003


def _generate_for_size(
    size: int,
    n_vars: int,
    n_exs: int,
    out_dir: Path,
    examples_per_shard: int,
    arrayrecord_options: str,
    seed: int,
    log_every: int,
) -> GenerationStats:
    start = time.time()
    rng = random.Random(_size_seed(seed, size))
    writer = ArrayRecordShardWriter(
        out_dir=out_dir / f"size_{size:02d}",
        size=size,
        n_vars=n_vars,
        examples_per_shard=examples_per_shard,
        writer_options=arrayrecord_options,
    )
    total = 0
    last_log = 0
    while total < n_exs:
        prop = sample_imply(n_vars, size, rng=rng)
        for ex in list_sequents(prop):
            writer.write(tokenize(ex))
            total += 1
            if log_every > 0 and total - last_log >= log_every:
                print(f"[size {size}] generated {total}/{n_exs}")
                last_log = total
            if total >= n_exs:
                break
    writer.close()
    elapsed = time.time() - start
    return GenerationStats(
        size=size,
        examples=writer.total_examples,
        shards=writer.shard_count,
        seconds=elapsed,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate implicational propositional logic dataset.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-vars", type=int, default=3)
    parser.add_argument("--min-size", type=int, default=2)
    parser.add_argument("--max-size", type=int, default=25)
    parser.add_argument("--examples-per-size", type=int, default=10_000)
    parser.add_argument("--examples-per-shard", type=int, default=50_000)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 4))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--arrayrecord-options",
        type=str,
        default="group_size:1024",
        help="ArrayRecord writer options string.",
    )
    parser.add_argument("--log-every", type=int, default=5_000)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    sizes = list(range(args.min_size, args.max_size + 1))
    if not sizes:
        raise ValueError("No sizes to generate. Check --min-size/--max-size.")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    max_workers = max(1, min(args.workers, len(sizes)))
    stats: list[GenerationStats] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _generate_for_size,
                size,
                args.n_vars,
                args.examples_per_size,
                args.out_dir,
                args.examples_per_shard,
                args.arrayrecord_options,
                args.seed,
                args.log_every,
            ): size
            for size in sizes
        }
        for future in as_completed(futures):
            stats.append(future.result())

    stats_sorted = sorted(stats, key=lambda s: s.size)
    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "format": "arrayrecord",
        "n_vars": args.n_vars,
        "min_size": args.min_size,
        "max_size": args.max_size,
        "examples_per_size": args.examples_per_size,
        "examples_per_shard": args.examples_per_shard,
        "arrayrecord_options": args.arrayrecord_options,
        "sizes": {
            str(stat.size): {
                "examples": stat.examples,
                "shards": stat.shards,
                "seconds": round(stat.seconds, 3),
            }
            for stat in stats_sorted
        },
    }
    metadata_path = args.out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Done. Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
