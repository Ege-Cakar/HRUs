"""Autoregressive task loader for Lean proof-state -> next-tactic data."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import pickle
from typing import Iterable

import grain
from grain._src.core import sharding, transforms
from grain._src.python import data_sources, samplers
import numpy as np


class LeanDojoAutoregDepthTask:
    STATS_KEYS = ("max_token", "max_seq", "max_prompt_seq", "max_completion_seq")

    def __init__(
        self,
        ds_path,
        depth_range=(1, 32),
        batch_size=128,
        *,
        shuffle=True,
        seed=None,
        worker_count=0,
        reader_options=None,
        drop_remainder=False,
    ) -> None:
        self.ds_path = Path(ds_path)
        self.depth_range = depth_range
        self._depths = self._normalize_depths(depth_range)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed if seed is not None else int(
            np.random.randint(0, np.iinfo(np.int32).max)
        )
        self.worker_count = worker_count
        self.reader_options = reader_options
        self.drop_remainder = drop_remainder
        self.stats = self._stats_from_metadata(self.ds_path, self._depths)

        self._epoch = 0
        self._data_source = self._build_data_source()
        self._dataloader = self._build_dataloader()
        self._iterator = iter(self._dataloader)

    def __next__(self):
        try:
            return next(self._iterator)
        except StopIteration:
            self._epoch += 1
            self._dataloader = self._build_dataloader()
            self._iterator = iter(self._dataloader)
            return next(self._iterator)

    def __iter__(self):
        return self

    @staticmethod
    def _normalize_depths(depth_range) -> list[int]:
        if isinstance(depth_range, int):
            return [depth_range]
        if isinstance(depth_range, tuple) and len(depth_range) == 2:
            start, end = depth_range
            if start > end:
                start, end = end, start
            return list(range(start, end + 1))
        return [int(depth) for depth in depth_range]

    @classmethod
    def stats_from_metadata(cls, ds_path, depth_range) -> dict:
        depths = cls._normalize_depths(depth_range)
        return cls._stats_from_metadata(Path(ds_path), depths)

    @classmethod
    def _stats_from_metadata(cls, ds_path: Path, depths: Iterable[int]) -> dict:
        metadata_path = ds_path / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Missing metadata at {metadata_path}. Regenerate the dataset."
            )
        metadata = json.loads(metadata_path.read_text())
        depth_meta = metadata.get("depths", {})

        stats_list = []
        missing = []
        for depth in depths:
            stats = depth_meta.get(str(depth), {}).get("stats")
            if stats is None:
                missing.append(depth)
                continue
            stats_list.append(stats)

        if missing:
            raise ValueError(f"Missing stats for depths {missing} in {metadata_path}.")

        return {
            key: max(int(stats.get(key, 0)) for stats in stats_list)
            for key in cls.STATS_KEYS
        }

    def _collect_shards(self, depths: Iterable[int]) -> list[str]:
        shards: list[str] = []
        for depth in depths:
            depth_dir = self.ds_path / f"depth_{depth:03d}"
            if not depth_dir.exists():
                raise FileNotFoundError(f"Missing depth directory: {depth_dir}")
            depth_shards = sorted(depth_dir.glob("shard_*.array_record"))
            if not depth_shards:
                raise FileNotFoundError(f"No shards found in {depth_dir}")
            shards.extend(str(path) for path in depth_shards)
        return shards

    def _build_data_source(self):
        shards = self._collect_shards(self._depths)
        return data_sources.ArrayRecordDataSource(
            shards,
            reader_options=self.reader_options,
        )

    def _build_dataloader(self) -> grain.DataLoader:
        shard_opts = sharding.NoSharding()
        sampler = samplers.IndexSampler(
            num_records=len(self._data_source),
            shard_options=shard_opts,
            shuffle=self.shuffle,
            num_epochs=1,
            seed=self.seed + self._epoch if self.shuffle else None,
        )

        operations = [
            _DecodeRecord(),
            transforms.Batch(
                batch_size=self.batch_size,
                drop_remainder=self.drop_remainder,
                batch_fn=_batch_records_autoreg,
            ),
        ]

        return grain.DataLoader(
            data_source=self._data_source,
            sampler=sampler,
            operations=operations,
            worker_count=self.worker_count,
            shard_options=shard_opts,
        )


@dataclass(frozen=True)
class _DecodeRecord(transforms.MapTransform):
    def map(self, element):
        return pickle.loads(element)


def _normalize_completions(completions) -> list[np.ndarray]:
    if isinstance(completions, np.ndarray):
        arr = np.asarray(completions, dtype=np.int32)
        if arr.ndim == 1:
            return [arr]
        if arr.ndim == 2:
            return [arr[idx] for idx in range(arr.shape[0])]
        raise ValueError(f"Completions array must be 1D or 2D, got {arr.shape}")

    out: list[np.ndarray] = []
    for completion in completions:
        arr = np.asarray(completion, dtype=np.int32)
        if arr.ndim != 1:
            raise ValueError(f"Completion must be 1D, got {arr.shape}")
        out.append(arr)
    return out


def _pad_sequences(arrays: list[np.ndarray]) -> np.ndarray:
    max_len = max(arr.shape[0] for arr in arrays)
    out = np.zeros((len(arrays), max_len), dtype=np.int32)
    for idx, arr in enumerate(arrays):
        out[idx, : arr.shape[0]] = arr
    return out


def _batch_records_autoreg(records):
    if not records:
        return np.zeros((0, 0), dtype=np.int32), np.zeros((0, 0), dtype=np.int32)

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    for rec in records:
        prompt = np.asarray(rec["prompt"], dtype=np.int32)
        if prompt.ndim != 1:
            raise ValueError(f"Prompt must be 1D, got {prompt.shape}")

        completions = _normalize_completions(rec["completions"])
        if not completions:
            raise ValueError("Cannot sample from empty completion list.")

        pick = np.random.randint(len(completions))
        completion = completions[pick]

        full = np.concatenate([prompt, completion], axis=0)
        if full.shape[0] < 2:
            raise ValueError("Prompt + completion must contain at least 2 tokens.")

        x = full[:-1].copy()
        y = full[1:].copy()

        if prompt.shape[0] > 1:
            y[: prompt.shape[0] - 1] = 0

        xs.append(x)
        ys.append(y)

    return _pad_sequences(xs), _pad_sequences(ys)
