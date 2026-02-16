from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from array_record.python import array_record_module

from task.layer_axiom import LayerAxiomTask
from task.layer_gen.util import tokenize_layer_axiom as tok


def _write_array_record(path: Path, records) -> None:
    writer = array_record_module.ArrayRecordWriter(str(path), "group_size:1")
    for record in records:
        writer.write(pickle.dumps(record, protocol=5))
    writer.close()


def test_layer_axiom_task_offline_autoreg(tmp_path: Path) -> None:
    distance_dir = tmp_path / "distance_001"
    distance_dir.mkdir(parents=True)
    shard_path = distance_dir / "shard_00000.array_record"

    rec = {
        "prompt": np.array([1, 2, tok.sep_token_id], dtype=np.int32),
        "completions": [np.array([11, tok.eot_token_id], dtype=np.int32)],
    }
    _write_array_record(shard_path, [rec])

    metadata = {
        "distances": {
            "1": {
                "examples": 1,
                "records": 1,
                "shards": 1,
                "stats": {
                    "max_token": int(tok.eot_token_id),
                    "max_seq": 4,
                    "max_prompt_seq": 3,
                    "max_completion_seq": 2,
                },
            }
        }
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))

    task = LayerAxiomTask(
        ds_path=tmp_path,
        distance_range=(1, 1),
        batch_size=1,
        mode="offline",
        objective="autoreg",
        shuffle=False,
        worker_count=0,
    )

    xs, ys = next(task)
    assert xs.shape == (1, 4)
    assert ys.shape == (1, 4)
    assert ys[0, 0] == 0



def test_layer_axiom_task_offline_first_step(tmp_path: Path) -> None:
    distance_dir = tmp_path / "distance_002"
    distance_dir.mkdir(parents=True)
    shard_path = distance_dir / "shard_00000.array_record"

    rec = {
        "prompt": np.array([5, 6, tok.sep_token_id], dtype=np.int32),
        "target_first": np.array([21, 22, tok.eot_token_id], dtype=np.int32),
    }
    _write_array_record(shard_path, [rec])

    metadata = {
        "distances": {
            "2": {
                "examples": 1,
                "records": 1,
                "shards": 1,
                "stats": {
                    "max_token": int(tok.eot_token_id),
                    "max_seq": 3,
                    "max_prompt_seq": 3,
                    "max_target_seq": 3,
                },
            }
        }
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))

    task = LayerAxiomTask(
        ds_path=tmp_path,
        distance_range=(2, 2),
        batch_size=1,
        mode="offline",
        objective="first_step",
        shuffle=False,
        worker_count=0,
    )

    prompts, targets = next(task)
    assert prompts.shape == (1, 3)
    assert targets.shape == (1, 3)
    assert np.array_equal(targets[0], rec["target_first"])



def test_layer_axiom_task_online_sampling_autoreg() -> None:
    task = LayerAxiomTask(
        distance_range=(1, 2),
        batch_size=4,
        mode="online",
        objective="autoreg",
        seed=11,
        n_layers=6,
        props_per_layer=5,
        rules_per_transition=8,
        k_in_max=2,
        k_out_max=2,
        initial_ant_max=3,
    )

    xs, ys = next(task)
    assert xs.shape[0] == 4
    assert ys.shape[0] == 4
    assert xs.dtype == np.int32
    assert ys.dtype == np.int32
    assert np.any(ys == 0)
