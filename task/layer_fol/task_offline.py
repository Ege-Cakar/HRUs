"""Offline dataset helpers for layered FOL tasks."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import grain
from grain._src.core import sharding, transforms
from grain._src.python import data_sources, samplers
import pickle

from task.layer_gen.util import tokenize_layer_fol


def normalize_distances(distance_range) -> list[int]:
    if isinstance(distance_range, int):
        return [distance_range]
    if isinstance(distance_range, tuple) and len(distance_range) == 2:
        start, end = distance_range
        if start > end:
            start, end = end, start
        return list(range(int(start), int(end) + 1))
    return [int(distance) for distance in distance_range]


def load_metadata(ds_path: Path) -> dict:
    metadata_path = ds_path / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing metadata at {metadata_path}. Regenerate the dataset."
        )
    return json.loads(metadata_path.read_text())


def load_tokenizer_and_stats(
    *,
    ds_path: Path,
    distances: Iterable[int],
    expected_completion_format: str,
    stats_keys: tuple[str, ...],
):
    metadata = load_metadata(ds_path)
    metadata_completion_format = str(
        metadata.get("config", {}).get("completion_format", "single")
    )
    if metadata_completion_format != str(expected_completion_format):
        raise ValueError(
            "Dataset completion_format mismatch: "
            f"task requested {expected_completion_format!r}, "
            f"but metadata declares {metadata_completion_format!r}."
        )

    tokenizer = tokenize_layer_fol.tokenizer_from_metadata(metadata)
    stats = stats_from_metadata(
        ds_path=ds_path,
        distances=distances,
        stats_keys=stats_keys,
        metadata=metadata,
    )
    return tokenizer, stats


def stats_from_metadata(
    *,
    ds_path: Path,
    distances: Iterable[int],
    stats_keys: tuple[str, ...],
    metadata: dict | None = None,
) -> dict:
    if metadata is None:
        metadata = load_metadata(ds_path)

    distance_meta = metadata.get("distances", {})
    stats_list = []
    missing = []
    for distance in distances:
        stats = distance_meta.get(str(distance), {}).get("stats")
        if stats is None:
            missing.append(distance)
            continue
        stats_list.append(stats)

    if missing:
        raise ValueError(
            f"Missing stats for distances {missing} in {ds_path / 'metadata.json'}."
        )

    return {
        key: max(int(stats.get(key, 0)) for stats in stats_list)
        for key in stats_keys
    }


def collect_shards(*, ds_path: Path, distances: Iterable[int]) -> list[str]:
    shards: list[str] = []
    for distance in distances:
        distance_dir = ds_path / f"distance_{distance:03d}"
        if not distance_dir.exists():
            raise FileNotFoundError(f"Missing distance directory: {distance_dir}")
        distance_shards = sorted(distance_dir.glob("shard_*.array_record"))
        if not distance_shards:
            raise FileNotFoundError(f"No shards found in {distance_dir}")
        shards.extend(str(path) for path in distance_shards)
    return shards


def build_data_source(*, ds_path: Path, distances: Iterable[int], reader_options):
    shards = collect_shards(ds_path=ds_path, distances=distances)
    return data_sources.ArrayRecordDataSource(
        shards,
        reader_options=reader_options,
    )


def build_dataloader(
    *,
    data_source,
    batch_size: int,
    drop_remainder: bool,
    batch_fn,
    shuffle: bool,
    seed: int,
    epoch: int,
    worker_count: int,
) -> grain.DataLoader:
    shard_opts = sharding.NoSharding()
    sampler = samplers.IndexSampler(
        num_records=len(data_source),
        shard_options=shard_opts,
        shuffle=bool(shuffle),
        num_epochs=1,
        seed=int(seed) + int(epoch) if shuffle else None,
    )

    operations = [
        DecodeRecord(),
        transforms.Batch(
            batch_size=int(batch_size),
            drop_remainder=bool(drop_remainder),
            batch_fn=batch_fn,
        ),
    ]

    return grain.DataLoader(
        data_source=data_source,
        sampler=sampler,
        operations=operations,
        worker_count=worker_count,
        shard_options=shard_opts,
    )


@dataclass(frozen=True)
class DecodeRecord(transforms.MapTransform):
    def map(self, element):
        return pickle.loads(element)
