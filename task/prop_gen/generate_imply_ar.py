"""Generate autoregressive implicational dataset with optional unprovable labels."""

from __future__ import annotations

import argparse
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
import json
import os
import pickle
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tqdm import tqdm

from array_record.python import array_record_module

from util.sample import (
    count_sequent_symbols,
    list_sequents_allow_false,
    sample_imply_no_true,
)
from util.tokenize_ar import tokenize


@dataclass(frozen=True)
class GenerationStats:
    size: int
    examples: int
    shards: int
    seconds: float
    max_token: int
    max_seq: int
    max_prompt_seq: int
    max_completion_seq: int


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

    def write(self, prompt_tokens: list[int], completion_tokens: list[list[int]]) -> None:
        if self._examples_in_shard >= self.examples_per_shard:
            self._rollover()

        prompt = np.asarray(prompt_tokens, dtype=np.int32)
        completions = [np.asarray(tokens, dtype=np.int32) for tokens in completion_tokens]
        payload = pickle.dumps(
            {
                "prompt": prompt,
                "completions": completions,
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


@dataclass
class _SizeAccumulator:
    max_token: int = 0
    max_seq: int = 0
    max_prompt_seq: int = 0
    max_completion_seq: int = 0


@dataclass
class _BucketState:
    writer: ArrayRecordShardWriter
    target_examples: int
    examples: int = 0
    acc: _SizeAccumulator = field(default_factory=_SizeAccumulator)

    @property
    def is_full(self) -> bool:
        return self.examples >= self.target_examples


def _default_seed() -> int:
    return int(np.random.randint(0, np.iinfo(np.int32).max))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate autoregressive implicational propositional dataset.",
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
    parser.add_argument("--allow-false", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def _update_stats(
    acc: _SizeAccumulator,
    prompt_tokens: list[int],
    completion_tokens: list[list[int]],
) -> None:
    full_lengths = [len(prompt_tokens) + len(comp) - 1 for comp in completion_tokens]
    if prompt_tokens:
        acc.max_token = max(acc.max_token, max(prompt_tokens))
    for comp in completion_tokens:
        if comp:
            acc.max_token = max(acc.max_token, max(comp))
    if full_lengths:
        acc.max_seq = max(acc.max_seq, max(full_lengths))
    acc.max_prompt_seq = max(acc.max_prompt_seq, len(prompt_tokens))
    if completion_tokens:
        acc.max_completion_seq = max(
            acc.max_completion_seq,
            max(len(comp) for comp in completion_tokens),
        )


def _flush_progress(
    pending_updates: dict[int, int],
    bars: dict[int, tqdm],
    overall_bar: tqdm,
) -> None:
    for size, pending in pending_updates.items():
        if pending <= 0:
            continue
        bars[size].update(pending)
        overall_bar.update(pending)
        pending_updates[size] = 0


def _worker_generate_root(
    root_size: int,
    n_vars: int,
    allow_false: bool,
    min_size: int,
    max_size: int,
    worker_seed: int,
) -> dict[int, list[tuple[list[int], list[list[int]]]]]:
    rng = random.Random(worker_seed)
    prop = sample_imply_no_true(n_vars, root_size, rng=rng)
    examples = list_sequents_allow_false(prop, allow_false=allow_false, rng=rng)

    bucketed: dict[int, list[tuple[list[int], list[list[int]]]]] = {}
    for sequent, rules in examples:
        bucket_size = count_sequent_symbols(sequent)
        if bucket_size < min_size or bucket_size > max_size:
            continue
        prompt_tokens, completion_tokens = tokenize((sequent, rules))
        if not completion_tokens:
            continue
        bucketed.setdefault(bucket_size, []).append((prompt_tokens, completion_tokens))
    return bucketed


def main() -> None:
    args = _parse_args()
    if args.seed is None:
        args.seed = _default_seed()

    sizes = list(range(args.min_size, args.max_size + 1))
    if not sizes:
        raise ValueError("No sizes to generate. Check --min-size/--max-size.")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    states: dict[int, _BucketState] = {}
    for size in sizes:
        writer = ArrayRecordShardWriter(
            out_dir=args.out_dir / f"size_{size:02d}",
            examples_per_shard=args.examples_per_shard,
            writer_options=args.arrayrecord_options,
        )
        states[size] = _BucketState(
            writer=writer,
            target_examples=args.examples_per_size,
        )

    progress_step = max(1, args.log_every) if args.log_every > 0 else 1
    total_target = args.examples_per_size * len(sizes)
    pending_updates: dict[int, int] = {size: 0 for size in sizes}

    bars = {
        size: tqdm(total=args.examples_per_size, desc=f"size {size:02d}", position=i, leave=True)
        for i, size in enumerate(sizes)
    }
    overall_bar = tqdm(total=total_target, desc="overall", position=len(sizes), leave=True)

    start = time.time()
    max_workers = max(1, min(args.workers, os.cpu_count() or args.workers))
    inflight_limit = max_workers * 2
    scheduler_rng = random.Random(args.seed)
    active_sizes = list(sizes)

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

    inflight = {}

    def _submit_one() -> bool:
        if not active_sizes:
            return False
        root_size = scheduler_rng.choice(active_sizes)
        worker_seed = scheduler_rng.randrange(0, np.iinfo(np.int64).max)
        fut = executor.submit(
            _worker_generate_root,
            root_size,
            args.n_vars,
            bool(args.allow_false),
            args.min_size,
            args.max_size,
            worker_seed,
        )
        inflight[fut] = root_size
        return True

    try:
        while len(inflight) < inflight_limit and _submit_one():
            pass

        while inflight:
            done, _ = wait(tuple(inflight.keys()), return_when=FIRST_COMPLETED)
            for fut in done:
                inflight.pop(fut, None)
                bucketed = fut.result()
                for bucket_size, examples in bucketed.items():
                    state = states.get(bucket_size)
                    if state is None or state.is_full:
                        continue

                    for prompt_tokens, completion_tokens in examples:
                        if state.is_full:
                            break
                        state.writer.write(prompt_tokens, completion_tokens)
                        state.examples += 1
                        _update_stats(state.acc, prompt_tokens, completion_tokens)

                        pending_updates[bucket_size] += 1
                        if pending_updates[bucket_size] >= progress_step:
                            bars[bucket_size].update(pending_updates[bucket_size])
                            overall_bar.update(pending_updates[bucket_size])
                            pending_updates[bucket_size] = 0

                    if state.is_full and bucket_size in active_sizes:
                        active_sizes.remove(bucket_size)

            if not active_sizes and all(state.is_full for state in states.values()):
                for fut in inflight:
                    fut.cancel()
                inflight.clear()
                break

            while len(inflight) < inflight_limit and _submit_one():
                pass

        _flush_progress(pending_updates, bars, overall_bar)

    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        for bar in bars.values():
            bar.close()
        overall_bar.close()
        for state in states.values():
            state.writer.close()

    elapsed = time.time() - start

    stats_sorted = []
    for size in sizes:
        state = states[size]
        stats_sorted.append(
            GenerationStats(
                size=size,
                examples=state.examples,
                shards=state.writer.shard_count,
                seconds=elapsed,
                max_token=state.acc.max_token,
                max_seq=state.acc.max_seq,
                max_prompt_seq=state.acc.max_prompt_seq,
                max_completion_seq=state.acc.max_completion_seq,
            )
        )

    overall_stats = {
        "max_token": max(stat.max_token for stat in stats_sorted),
        "max_seq": max(stat.max_seq for stat in stats_sorted),
        "max_prompt_seq": max(stat.max_prompt_seq for stat in stats_sorted),
        "max_completion_seq": max(stat.max_completion_seq for stat in stats_sorted),
    }

    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "format": "arrayrecord_autoreg",
        "seed": args.seed,
        "n_vars": args.n_vars,
        "min_size": args.min_size,
        "max_size": args.max_size,
        "examples_per_size": args.examples_per_size,
        "examples_per_shard": args.examples_per_shard,
        "workers": max_workers,
        "parallel_backend": backend,
        "arrayrecord_options": args.arrayrecord_options,
        "allow_false": bool(args.allow_false),
        "stats": overall_stats,
        "sizes": {
            str(stat.size): {
                "examples": stat.examples,
                "shards": stat.shards,
                "seconds": round(stat.seconds, 3),
                "stats": {
                    "max_token": stat.max_token,
                    "max_seq": stat.max_seq,
                    "max_prompt_seq": stat.max_prompt_seq,
                    "max_completion_seq": stat.max_completion_seq,
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
