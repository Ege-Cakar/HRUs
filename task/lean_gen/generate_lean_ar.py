"""Generate ArrayRecord Lean proof-state -> next-tactic autoregressive datasets."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import pickle
import time

import numpy as np
from tqdm import tqdm

from array_record.python import array_record_module

if __package__ in (None, ""):
    from extract import (  # type: ignore
        DepthSplit,
        LeanTacticStep,
        extract_lean_tactic_steps,
        hashed_theorem_ids,
        split_by_depth_quantile,
    )
    from util.tokenize_lean_ar import tokenize_example  # type: ignore
else:
    from .extract import (
        DepthSplit,
        LeanTacticStep,
        extract_lean_tactic_steps,
        hashed_theorem_ids,
        split_by_depth_quantile,
    )
    from .util.tokenize_lean_ar import tokenize_example


@dataclass(frozen=True)
class GenerationStats:
    depth: int
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

    def write(self, payload: bytes) -> None:
        if self._examples_in_shard >= self.examples_per_shard:
            self._rollover()
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


def _default_seed() -> int:
    return int(np.random.randint(0, np.iinfo(np.int32).max))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LeanDojo-based autoregressive proof-state datasets.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        help="Path to LeanDojo-exported JSON/JSONL data.",
    )
    parser.add_argument(
        "--split-quantile",
        type=float,
        default=0.7,
        help="Train split theorem quantile by proof length (0, 1).",
    )
    parser.add_argument("--examples-per-shard", type=int, default=50_000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-theorems", type=int, default=None)
    parser.add_argument("--max-steps-per-theorem", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=1_000)
    parser.add_argument(
        "--arrayrecord-options",
        type=str,
        default="group_size:1",
        help="ArrayRecord writer options string.",
    )
    return parser.parse_args()


def _update_stats(
    *,
    max_token: int,
    max_seq: int,
    max_prompt_seq: int,
    max_completion_seq: int,
    prompt: list[int],
    completions: list[list[int]],
) -> tuple[int, int, int, int]:
    if prompt:
        max_token = max(max_token, max(prompt))
        max_prompt_seq = max(max_prompt_seq, len(prompt))
    for completion in completions:
        if completion:
            max_token = max(max_token, max(completion))
            max_completion_seq = max(max_completion_seq, len(completion))
            max_seq = max(max_seq, len(prompt) + len(completion) - 1)
    return max_token, max_seq, max_prompt_seq, max_completion_seq


def _write_split(
    *,
    split_name: str,
    split_examples: tuple[LeanTacticStep, ...],
    split_theorems: tuple[str, ...],
    out_dir: Path,
    benchmark: str,
    split_quantile: float,
    depth_cutoff: int,
    seed: int,
    examples_per_shard: int,
    log_every: int,
    arrayrecord_options: str,
) -> dict:
    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    by_depth: dict[int, list[LeanTacticStep]] = {}
    for ex in split_examples:
        by_depth.setdefault(ex.proof_len, []).append(ex)

    stats: list[GenerationStats] = []
    pending = 0
    progress_step = max(1, log_every) if log_every > 0 else 1

    bar = tqdm(total=len(split_examples), desc=f"{split_name}", leave=True)
    try:
        for depth in sorted(by_depth):
            depth_start = time.time()
            writer = ArrayRecordShardWriter(
                out_dir=split_dir / f"depth_{depth:03d}",
                examples_per_shard=examples_per_shard,
                writer_options=arrayrecord_options,
            )

            max_token = 0
            max_seq = 0
            max_prompt_seq = 0
            max_completion_seq = 0

            for ex in by_depth[depth]:
                prompt, completions = tokenize_example(ex.proof_state, ex.next_tactic)

                payload = pickle.dumps(
                    {
                        "prompt": np.asarray(prompt, dtype=np.int32),
                        "completions": [np.asarray(comp, dtype=np.int32) for comp in completions],
                        "theorem_id": ex.theorem_id,
                        "proof_len": int(ex.proof_len),
                        "step_idx": int(ex.step_idx),
                    },
                    protocol=5,
                )
                writer.write(payload)

                max_token, max_seq, max_prompt_seq, max_completion_seq = _update_stats(
                    max_token=max_token,
                    max_seq=max_seq,
                    max_prompt_seq=max_prompt_seq,
                    max_completion_seq=max_completion_seq,
                    prompt=prompt,
                    completions=completions,
                )

                pending += 1
                if pending >= progress_step:
                    bar.update(pending)
                    pending = 0

            if pending:
                bar.update(pending)
                pending = 0

            writer.close()
            stats.append(
                GenerationStats(
                    depth=depth,
                    examples=writer.total_examples,
                    shards=writer.shard_count,
                    seconds=time.time() - depth_start,
                    max_token=max_token,
                    max_seq=max_seq,
                    max_prompt_seq=max_prompt_seq,
                    max_completion_seq=max_completion_seq,
                )
            )
    finally:
        bar.close()

    if not stats:
        raise ValueError(f"Split {split_name!r} is empty.")

    overall_stats = {
        "max_token": max(stat.max_token for stat in stats),
        "max_seq": max(stat.max_seq for stat in stats),
        "max_prompt_seq": max(stat.max_prompt_seq for stat in stats),
        "max_completion_seq": max(stat.max_completion_seq for stat in stats),
    }

    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "format": "arrayrecord_lean_autoreg",
        "benchmark": benchmark,
        "split": split_name,
        "seed": seed,
        "split_quantile": split_quantile,
        "depth_cutoff": depth_cutoff,
        "examples_per_shard": examples_per_shard,
        "arrayrecord_options": arrayrecord_options,
        "stats": overall_stats,
        "theorem_count": len(split_theorems),
        "theorem_ids": list(split_theorems),
        "theorem_id_hashes": hashed_theorem_ids(split_theorems),
        "depths": {
            str(stat.depth): {
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
            for stat in stats
        },
    }

    (split_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def _split_summary(split_name: str, split_meta: dict) -> dict:
    depths = [int(depth) for depth in split_meta["depths"].keys()]
    return {
        "split": split_name,
        "theorems": int(split_meta["theorem_count"]),
        "examples": int(sum(depth["examples"] for depth in split_meta["depths"].values())),
        "min_depth": min(depths),
        "max_depth": max(depths),
        "stats": split_meta["stats"],
    }


def _write_root_metadata(
    *,
    out_dir: Path,
    benchmark: str,
    split: DepthSplit,
    seed: int,
    max_theorems: int | None,
    max_steps_per_theorem: int | None,
    workers: int,
    split_quantile: float,
    examples_per_shard: int,
    arrayrecord_options: str,
    train_meta: dict,
    test_meta: dict,
) -> None:
    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "format": "arrayrecord_lean_autoreg_split",
        "benchmark": benchmark,
        "seed": seed,
        "workers": workers,
        "split_quantile": split_quantile,
        "depth_cutoff": split.depth_cutoff,
        "max_theorems": max_theorems,
        "max_steps_per_theorem": max_steps_per_theorem,
        "examples_per_shard": examples_per_shard,
        "arrayrecord_options": arrayrecord_options,
        "train": _split_summary("train", train_meta),
        "test": _split_summary("test", test_meta),
        "train_theorem_ids": list(split.train_theorems),
        "test_theorem_ids": list(split.test_theorems),
        "train_theorem_hashes": hashed_theorem_ids(split.train_theorems),
        "test_theorem_hashes": hashed_theorem_ids(split.test_theorems),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def main() -> None:
    args = _parse_args()
    if args.seed is None:
        args.seed = _default_seed()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    examples = extract_lean_tactic_steps(
        args.benchmark,
        max_theorems=args.max_theorems,
        max_steps_per_theorem=args.max_steps_per_theorem,
        seed=args.seed,
    )

    split = split_by_depth_quantile(examples, train_quantile=args.split_quantile)

    train_meta = _write_split(
        split_name="train",
        split_examples=split.train_examples,
        split_theorems=split.train_theorems,
        out_dir=args.out_dir,
        benchmark=args.benchmark,
        split_quantile=args.split_quantile,
        depth_cutoff=split.depth_cutoff,
        seed=args.seed,
        examples_per_shard=args.examples_per_shard,
        log_every=args.log_every,
        arrayrecord_options=args.arrayrecord_options,
    )
    test_meta = _write_split(
        split_name="test",
        split_examples=split.test_examples,
        split_theorems=split.test_theorems,
        out_dir=args.out_dir,
        benchmark=args.benchmark,
        split_quantile=args.split_quantile,
        depth_cutoff=split.depth_cutoff,
        seed=args.seed,
        examples_per_shard=args.examples_per_shard,
        log_every=args.log_every,
        arrayrecord_options=args.arrayrecord_options,
    )

    _write_root_metadata(
        out_dir=args.out_dir,
        benchmark=args.benchmark,
        split=split,
        seed=args.seed,
        max_theorems=args.max_theorems,
        max_steps_per_theorem=args.max_steps_per_theorem,
        workers=args.workers,
        split_quantile=args.split_quantile,
        examples_per_shard=args.examples_per_shard,
        arrayrecord_options=args.arrayrecord_options,
        train_meta=train_meta,
        test_meta=test_meta,
    )
    print(f"Done. Wrote dataset to {args.out_dir}")


if __name__ == "__main__":
    main()
