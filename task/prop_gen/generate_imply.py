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

import numpy as np
from tqdm import tqdm

from array_record.python import array_record_module

from util.sample import sample_imply, list_sequents
from util.tokenize import tokenize
from util.elem import TokenizedExample

@dataclass(frozen=True)
class GenerationStats:
    size: int
    examples: int
    shards: int
    seconds: float
    max_token: int
    max_pos: int
    max_seq: int


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
            rule = np.asarray(rule_tokens, dtype=np.int32)
        else:
            rule = np.zeros((0, 2), dtype=np.int32)
        payload = pickle.dumps(
            {
                "sequent": sequent,
                "rule": rule,
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


def _default_seed() -> int:
    return int(np.random.randint(0, np.iinfo(np.int32).max))


def _generate_for_size(
    size: int,
    n_vars: int,
    n_exs: int,
    out_dir: Path,
    examples_per_shard: int,
    arrayrecord_options: str,
    seed: int,
    progress_bar: tqdm | None,
    overall_bar: tqdm | None,
    progress_step: int,
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
    pending = 0
    max_token = 0
    max_pos = 0
    max_seq = 0
    while total < n_exs:
        prop = sample_imply(n_vars, size, rng=rng)
        for example in list_sequents(prop, rng=rng):
            tokens = tokenize(example)
            sequent_tokens, rule_tokens = tokens
            if sequent_tokens:
                max_token = max(max_token, max(sequent_tokens))
                max_seq = max(max_seq, len(sequent_tokens))
            if rule_tokens:
                max_pos = max(max_pos, max(pos for _, pos in rule_tokens))
            writer.write(tokens)
            total += 1
            pending += 1
            if pending >= progress_step:
                if progress_bar is not None:
                    progress_bar.update(pending)
                if overall_bar is not None:
                    overall_bar.update(pending)
                pending = 0
            if total >= n_exs:
                break
    if pending:
        if progress_bar is not None:
            progress_bar.update(pending)
        if overall_bar is not None:
            overall_bar.update(pending)
    writer.close()
    elapsed = time.time() - start
    return GenerationStats(
        size=size,
        examples=writer.total_examples,
        shards=writer.shard_count,
        seconds=elapsed,
        max_token=max_token,
        max_pos=max_pos,
        max_seq=max_seq,
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
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--arrayrecord-options",
        type=str,
        default="group_size:1",
        help="ArrayRecord writer options string.",
    )
    parser.add_argument("--log-every", type=int, default=5_000)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.seed is None:
        args.seed = _default_seed()
    sizes = list(range(args.min_size, args.max_size + 1))
    if not sizes:
        raise ValueError("No sizes to generate. Check --min-size/--max-size.")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    max_workers = max(1, min(args.workers, len(sizes)))
    stats: list[GenerationStats] = []

    progress_step = max(1, args.log_every) if args.log_every > 0 else 1
    total_examples = args.examples_per_size * len(sizes)

    bars: dict[int, tqdm] = {}
    for idx, size in enumerate(sizes):
        bars[size] = tqdm(
            total=args.examples_per_size,
            position=idx,
            desc=f"size {size:02d}",
            leave=True,
        )
    overall_bar = tqdm(
        total=total_examples,
        position=len(sizes),
        desc="overall",
        leave=True,
    )

    try:
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
                    bars[size],
                    overall_bar,
                    progress_step,
                ): size
                for size in sizes
            }
            for future in as_completed(futures):
                stats.append(future.result())
    finally:
        for bar in bars.values():
            bar.close()
        overall_bar.close()

    stats_sorted = sorted(stats, key=lambda s: s.size)
    overall_stats = {
        "max_token": max(stat.max_token for stat in stats_sorted),
        "max_pos": max(stat.max_pos for stat in stats_sorted),
        "max_seq": max(stat.max_seq for stat in stats_sorted),
    }
    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "format": "arrayrecord",
        "seed": args.seed,
        "n_vars": args.n_vars,
        "min_size": args.min_size,
        "max_size": args.max_size,
        "examples_per_size": args.examples_per_size,
        "examples_per_shard": args.examples_per_shard,
        "arrayrecord_options": args.arrayrecord_options,
        "stats": overall_stats,
        "sizes": {
            str(stat.size): {
                "examples": stat.examples,
                "shards": stat.shards,
                "seconds": round(stat.seconds, 3),
                "stats": {
                    "max_token": stat.max_token,
                    "max_pos": stat.max_pos,
                    "max_seq": stat.max_seq,
                },
            }
            for stat in stats_sorted
        },
    }
    metadata_path = args.out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Done. Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
