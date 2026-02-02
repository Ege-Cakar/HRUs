"""Task based on learning inference rule application in propositional logic"""

# <codecell>

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Iterable

import numpy as np
import grain
from grain._src.core import sharding, transforms
from grain._src.python import data_sources, samplers


class ImplySizeTask:

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
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed if seed is not None else int(
            np.random.randint(0, np.iinfo(np.int32).max)
        )
        self.worker_count = worker_count
        self.reader_options = reader_options
        self.drop_remainder = drop_remainder

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

    def _normalize_sizes(self) -> list[int]:
        if isinstance(self.size_range, int):
            return [self.size_range]
        if isinstance(self.size_range, tuple) and len(self.size_range) == 2:
            start, end = self.size_range
            if start > end:
                start, end = end, start
            return list(range(start, end + 1))
        return list(self.size_range)

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
        sizes = self._normalize_sizes()
        shards = self._collect_shards(sizes)
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
                batch_fn=_batch_records,
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


def _batch_records(records):
    if not records:
        return (
            np.zeros((0, 0), dtype=np.int32),
            np.zeros((0, 2), dtype=np.int32),
        )
    batch_size = len(records)
    max_seq = max(rec["sequent"].shape[0] for rec in records)
    seq_batch = np.zeros((batch_size, max_seq), dtype=np.int32)
    rules_batch = np.zeros((batch_size, 2), dtype=np.int32)
    for idx, rec in enumerate(records):
        sequent = rec["sequent"]
        seq_batch[idx, : sequent.shape[0]] = sequent
        rule = rec["rule"]
        rules_batch[idx, :] = rule
    return seq_batch, rules_batch

