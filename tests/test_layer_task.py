from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from array_record.python import array_record_module

from task.layer import LayerTask
from task.layer_gen.util import tokenize_layer as tok
from task.prop_gen.util.elem import Atom, Sequent


def _write_array_record(path: Path, records) -> None:
    writer = array_record_module.ArrayRecordWriter(str(path), "group_size:1")
    for record in records:
        writer.write(pickle.dumps(record, protocol=5))
    writer.close()


def test_layer_task_offline_autoreg(tmp_path: Path) -> None:
    distance_dir = tmp_path / "distance_001"
    distance_dir.mkdir(parents=True)
    shard_path = distance_dir / "shard_00000.array_record"

    tokenizer = tok.build_tokenizer_from_atoms(["p0_1", "p1_1"])
    prompt = tokenizer.tokenize_prompt(Sequent([Atom("p0_1")], Atom("p1_1")))
    completion = tokenizer.encode_completion("(p0_1→p1_1)")
    rec = {
        "prompt": np.array(prompt, dtype=np.int32),
        "completions": [np.array(completion, dtype=np.int32)],
    }
    _write_array_record(shard_path, [rec])

    max_token = int(max(max(prompt), max(completion)))
    metadata = {
        "tokenizer": tokenizer.to_dict(),
        "distances": {
            "1": {
                "examples": 1,
                "records": 1,
                "shards": 1,
                "stats": {
                    "max_token": max_token,
                    "max_seq": len(prompt) + len(completion) - 1,
                    "max_prompt_seq": len(prompt),
                    "max_completion_seq": len(completion),
                },
            }
        }
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))

    task = LayerTask(
        ds_path=tmp_path,
        distance_range=(1, 1),
        batch_size=1,
        mode="offline",
        shuffle=False,
        worker_count=0,
    )

    xs, ys = next(task)
    expected_len = len(prompt) + len(completion) - 1
    assert xs.shape == (1, expected_len)
    assert ys.shape == (1, expected_len)
    assert ys[0, 0] == 0


def test_layer_task_online_sampling_autoreg() -> None:
    task = LayerTask(
        distance_range=(1, 2),
        batch_size=4,
        mode="online",
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
