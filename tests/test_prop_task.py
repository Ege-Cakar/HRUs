from __future__ import annotations

import pickle
import json
from pathlib import Path

import numpy as np
from array_record.python import array_record_module

from task.prop import ImplySizeTask


def _write_array_record(path: Path, records) -> None:
    writer = array_record_module.ArrayRecordWriter(str(path), "group_size:1")
    for record in records:
        writer.write(pickle.dumps(record, protocol=5))
    writer.close()


def test_imply_size_task_batching(tmp_path: Path) -> None:
    size_dir = tmp_path / "size_02"
    size_dir.mkdir(parents=True)
    shard_path = size_dir / "shard_00000.array_record"

    rec1 = {
        "sequent": np.array([1, 2, 3], dtype=np.int32),
        "rule": np.array([[1, 0]], dtype=np.int32),
    }
    rec2 = {
        "sequent": np.array([4, 5], dtype=np.int32),
        "rule": np.array([[3, 0]], dtype=np.int32),
    }
    _write_array_record(shard_path, [rec1, rec2])

    max_token = int(max(rec1["sequent"].max(), rec2["sequent"].max()))
    max_seq = int(max(rec1["sequent"].shape[0], rec2["sequent"].shape[0]))
    max_pos = int(max(rec1["rule"][:, 1].max(), rec2["rule"][:, 1].max()))
    metadata = {
        "sizes": {
            "2": {
                "examples": 2,
                "shards": 1,
                "seconds": 0.0,
                "stats": {
                    "max_token": max_token,
                    "max_pos": max_pos,
                    "max_seq": max_seq,
                },
            }
        }
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))

    task = ImplySizeTask(
        tmp_path,
        size_range=(2, 2),
        batch_size=2,
        shuffle=False,
        worker_count=0,
        drop_remainder=False,
    )

    seq_batch, rule_batch = next(task)

    assert seq_batch.shape == (2, 3)
    assert rule_batch.shape == (2, 2)
    assert seq_batch.dtype == np.int32
    assert rule_batch.dtype == np.int32
    assert np.array_equal(seq_batch[0], np.array([1, 2, 3], dtype=np.int32))
    assert np.array_equal(seq_batch[1], np.array([4, 5, 0], dtype=np.int32))
    assert np.array_equal(rule_batch[0], np.array([1, 0], dtype=np.int32))
    assert np.array_equal(rule_batch[1], np.array([3, 0], dtype=np.int32))
    assert task.stats == {"max_token": max_token, "max_pos": max_pos, "max_seq": max_seq}
