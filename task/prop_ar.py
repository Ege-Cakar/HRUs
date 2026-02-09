"""Autoregressive task loader for implicational proposition data."""

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


class ImplyAutoregSizeTask:
    STATS_KEYS = ("max_token", "max_seq", "max_prompt_seq", "max_completion_seq")

    def __init__(
        self,
        ds_path,
        size_range=(2, 5),
        batch_size=128,
        *,
        shuffle=True,
        seed=None,
        worker_count=0,
        reader_options=None,
        drop_remainder=False,
    ) -> None:
        self.ds_path = Path(ds_path)
        self.size_range = size_range
        self._sizes = self._normalize_sizes(size_range)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed if seed is not None else int(
            np.random.randint(0, np.iinfo(np.int32).max)
        )
        self.worker_count = worker_count
        self.reader_options = reader_options
        self.drop_remainder = drop_remainder
        self.stats = self._stats_from_metadata(self.ds_path, self._sizes)

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
    def _normalize_sizes(size_range) -> list[int]:
        if isinstance(size_range, int):
            return [size_range]
        if isinstance(size_range, tuple) and len(size_range) == 2:
            start, end = size_range
            if start > end:
                start, end = end, start
            return list(range(start, end + 1))
        return list(size_range)

    @classmethod
    def stats_from_metadata(cls, ds_path, size_range) -> dict:
        sizes = cls._normalize_sizes(size_range)
        return cls._stats_from_metadata(Path(ds_path), sizes)

    @classmethod
    def _stats_from_metadata(cls, ds_path: Path, sizes: Iterable[int]) -> dict:
        metadata_path = ds_path / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Missing metadata at {metadata_path}. Regenerate the dataset."
            )
        metadata = json.loads(metadata_path.read_text())
        sizes_meta = metadata.get("sizes", {})

        stats_list = []
        missing = []
        for size in sizes:
            size_meta = sizes_meta.get(str(size))
            if not size_meta or "stats" not in size_meta:
                missing.append(size)
                continue
            stats_list.append(size_meta["stats"])
        if missing:
            raise ValueError(f"Missing stats for sizes {missing} in {metadata_path}.")

        return {
            key: max(int(stats.get(key, 0)) for stats in stats_list)
            for key in cls.STATS_KEYS
        }

    def _collect_shards(self, sizes: Iterable[int]) -> list[str]:
        shards: list[str] = []
        for size in sizes:
            size_dir = self.ds_path / f"size_{size:02d}"
            if not size_dir.exists():
                raise FileNotFoundError(f"Missing size directory: {size_dir}")
            size_shards = sorted(size_dir.glob("shard_*.array_record"))
            if not size_shards:
                raise FileNotFoundError(f"No shards found in {size_dir}")
            shards.extend(str(p) for p in size_shards)
        return shards

    def _build_data_source(self):
        shards = self._collect_shards(self._sizes)
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
    for comp in completions:
        arr = np.asarray(comp, dtype=np.int32)
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
