from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from array_record.python import array_record_module

from task.lean_ar import LeanDojoAutoregDepthTask, _batch_records_autoreg
from task.lean_gen.util import tokenize_lean_ar


def _write_array_record(path: Path, records) -> None:
    writer = array_record_module.ArrayRecordWriter(str(path), "group_size:1")
    for record in records:
        writer.write(pickle.dumps(record, protocol=5))
    writer.close()


def test_batch_records_autoreg_masks_prompt_labels() -> None:
    prompt = tokenize_lean_ar.tokenize_prompt("h : True\n⊢ True")
    completion = tokenize_lean_ar.encode_completion("exact h")

    records = [
        {
            "prompt": np.array(prompt, dtype=np.int32),
            "completions": [np.array(completion, dtype=np.int32)],
        }
    ]

    xs, ys = _batch_records_autoreg(records)

    assert xs.shape == ys.shape
    assert np.all(ys[0, : len(prompt) - 1] == 0)


def test_batch_records_autoreg_samples_completion_candidates() -> None:
    prompt = tokenize_lean_ar.tokenize_prompt("⊢ True")
    completion_a = tokenize_lean_ar.encode_completion("trivial")
    completion_b = tokenize_lean_ar.encode_completion("exact True.intro")

    records = [
        {
            "prompt": np.array(prompt, dtype=np.int32),
            "completions": [
                np.array(completion_a, dtype=np.int32),
                np.array(completion_b, dtype=np.int32),
            ],
        }
    ]

    seen = set()
    for _ in range(128):
        _, ys = _batch_records_autoreg(records)
        first_completion_token = int(ys[0, len(prompt) - 1])
        seen.add(first_completion_token)

    assert seen == {completion_a[0], completion_b[0]}


def test_lean_depth_task_batching(tmp_path: Path) -> None:
    depth_dir = tmp_path / "depth_002"
    depth_dir.mkdir(parents=True)
    shard_path = depth_dir / "shard_00000.array_record"

    prompt1, completions1 = tokenize_lean_ar.tokenize_example("⊢ True", "trivial")
    prompt2, completions2 = tokenize_lean_ar.tokenize_example(
        "h : True\n⊢ True",
        ["exact h", "assumption"],
    )

    rec1 = {
        "prompt": np.array(prompt1, dtype=np.int32),
        "completions": [np.array(c, dtype=np.int32) for c in completions1],
    }
    rec2 = {
        "prompt": np.array(prompt2, dtype=np.int32),
        "completions": [np.array(c, dtype=np.int32) for c in completions2],
    }
    _write_array_record(shard_path, [rec1, rec2])

    max_token = int(
        max(
            max(prompt1 + completions1[0]),
            max(prompt2 + completions2[0]),
            max(prompt2 + completions2[1]),
        )
    )
    max_seq = int(
        max(
            len(prompt1) + len(completions1[0]) - 1,
            len(prompt2) + len(completions2[0]) - 1,
            len(prompt2) + len(completions2[1]) - 1,
        )
    )
    max_prompt_seq = int(max(len(prompt1), len(prompt2)))
    max_completion_seq = int(max(len(completions1[0]), len(completions2[0]), len(completions2[1])))

    metadata = {
        "depths": {
            "2": {
                "examples": 2,
                "shards": 1,
                "seconds": 0.0,
                "stats": {
                    "max_token": max_token,
                    "max_seq": max_seq,
                    "max_prompt_seq": max_prompt_seq,
                    "max_completion_seq": max_completion_seq,
                },
            }
        }
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))

    task = LeanDojoAutoregDepthTask(
        tmp_path,
        depth_range=(2, 2),
        batch_size=2,
        shuffle=False,
        worker_count=0,
        drop_remainder=False,
    )

    xs, ys = next(task)

    assert xs.shape == ys.shape
    assert xs.shape[0] == 2
    assert xs.dtype == np.int32
    assert ys.dtype == np.int32

    # Prompt-internal targets are masked.
    assert np.all(ys[0, : len(prompt1) - 1] == 0)
    assert np.all(ys[1, : len(prompt2) - 1] == 0)

    assert task.stats == {
        "max_token": max_token,
        "max_seq": max_seq,
        "max_prompt_seq": max_prompt_seq,
        "max_completion_seq": max_completion_seq,
    }
