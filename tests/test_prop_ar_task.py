from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from array_record.python import array_record_module

from task.prop_ar import ImplyAutoregSizeTask, _batch_records_autoreg
from task.prop_gen.util import tokenize_ar


def _write_array_record(path: Path, records) -> None:
    writer = array_record_module.ArrayRecordWriter(str(path), "group_size:1")
    for record in records:
        writer.write(pickle.dumps(record, protocol=5))
    writer.close()


def test_batch_records_autoreg_masks_prompt_labels() -> None:
    start = tokenize_ar.start_token_id
    eot = tokenize_ar.eot_token_id
    records = [
        {
            "prompt": np.array([11, 12, 13, start], dtype=np.int32),
            "completions": [np.array([40, 41, eot], dtype=np.int32)],
        }
    ]

    xs, ys = _batch_records_autoreg(records)
    assert xs.shape == (1, 6)
    assert ys.shape == (1, 6)

    # prompt length is 4, so first 3 targets are prompt-internal and must be masked.
    assert np.array_equal(ys[0, :3], np.array([0, 0, 0], dtype=np.int32))
    assert np.array_equal(ys[0, 3:], np.array([40, 41, eot], dtype=np.int32))


def test_batch_records_autoreg_samples_among_candidates() -> None:
    start = tokenize_ar.start_token_id
    eot = tokenize_ar.eot_token_id
    records = [
        {
            "prompt": np.array([1, start], dtype=np.int32),
            "completions": [
                np.array([50, eot], dtype=np.int32),
                np.array([51, eot], dtype=np.int32),
            ],
        }
    ]

    seen = set()
    for _ in range(128):
        _, ys = _batch_records_autoreg(records)
        seen.add(int(ys[0, 1]))
    assert seen == {50, 51}


def test_imply_autoreg_size_task_batching(tmp_path: Path) -> None:
    start = tokenize_ar.start_token_id
    eot = tokenize_ar.eot_token_id

    size_dir = tmp_path / "size_02"
    size_dir.mkdir(parents=True)
    shard_path = size_dir / "shard_00000.array_record"

    rec1 = {
        "prompt": np.array([1, 2, start], dtype=np.int32),
        "completions": [np.array([40, eot], dtype=np.int32)],
    }
    rec2 = {
        "prompt": np.array([3, start], dtype=np.int32),
        "completions": [
            np.array([41, eot], dtype=np.int32),
            np.array([42, eot], dtype=np.int32),
        ],
    }
    _write_array_record(shard_path, [rec1, rec2])

    metadata = {
        "sizes": {
            "2": {
                "examples": 2,
                "shards": 1,
                "seconds": 0.0,
                "stats": {
                    "max_token": 42,
                    "max_seq": 4,
                    "max_prompt_seq": 3,
                    "max_completion_seq": 2,
                },
            }
        }
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))

    task = ImplyAutoregSizeTask(
        tmp_path,
        size_range=(2, 2),
        batch_size=2,
        shuffle=False,
        worker_count=0,
        drop_remainder=False,
    )

    xs, ys = next(task)

    assert xs.shape == (2, 4)
    assert ys.shape == (2, 4)
    assert xs.dtype == np.int32
    assert ys.dtype == np.int32

    assert ys[0, 0] == 0  # masked prompt-internal label
    assert ys[0, 1] == 0  # masked prompt-internal label

    assert task.stats == {
        "max_token": 42,
        "max_seq": 4,
        "max_prompt_seq": 3,
        "max_completion_seq": 2,
    }
