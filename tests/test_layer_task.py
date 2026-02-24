from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from array_record.python import array_record_module

from task.layer import LayerTask
from task.layer_gen.util import tokenize_layer as tok
from task.prop_gen.util.elem import Atom, Sequent


def _write_array_record(path: Path, records) -> None:
    writer = array_record_module.ArrayRecordWriter(str(path), "group_size:1")
    for record in records:
        writer.write(pickle.dumps(record, protocol=5))
    writer.close()


def _write_metadata_for_records(tmp_path: Path, tokenizer, records) -> None:
    max_token = 0
    max_seq = 0
    max_prompt_seq = 0
    max_completion_seq = 0

    for rec in records:
        prompt = np.asarray(rec["prompt"], dtype=np.int32)
        max_prompt_seq = max(max_prompt_seq, int(prompt.shape[0]))
        if prompt.shape[0] > 0:
            max_token = max(max_token, int(prompt.max()))

        for completion in rec["completions"]:
            comp = np.asarray(completion, dtype=np.int32)
            max_completion_seq = max(max_completion_seq, int(comp.shape[0]))
            if comp.shape[0] > 0:
                max_token = max(max_token, int(comp.max()))
            max_seq = max(max_seq, int(prompt.shape[0] + comp.shape[0] - 1))

    metadata = {
        "tokenizer": tokenizer.to_dict(),
        "distances": {
            "1": {
                "examples": len(records),
                "records": len(records),
                "shards": 1,
                "stats": {
                    "max_token": max_token,
                    "max_seq": max_seq,
                    "max_prompt_seq": max_prompt_seq,
                    "max_completion_seq": max_completion_seq,
                },
            }
        },
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))


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
    _write_metadata_for_records(tmp_path, tokenizer, [rec])

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


def test_layer_task_offline_autoreg_global_max_uses_metadata_max_seq(tmp_path: Path) -> None:
    distance_dir = tmp_path / "distance_001"
    distance_dir.mkdir(parents=True)
    shard_path = distance_dir / "shard_00000.array_record"

    tokenizer = tok.build_tokenizer_from_atoms(["p0_1", "p1_1"])
    prompt = tokenizer.tokenize_prompt(Sequent([Atom("p0_1")], Atom("p1_1")))
    completion_short = tokenizer.encode_completion("p1_1")
    completion_long = tokenizer.encode_completion("(p0_1→p1_1)")

    rec_short = {
        "prompt": np.array(prompt, dtype=np.int32),
        "completions": [np.array(completion_short, dtype=np.int32)],
    }
    rec_long = {
        "prompt": np.array(prompt, dtype=np.int32),
        "completions": [np.array(completion_long, dtype=np.int32)],
    }
    _write_array_record(shard_path, [rec_short, rec_long])
    _write_metadata_for_records(tmp_path, tokenizer, [rec_short, rec_long])

    task = LayerTask(
        ds_path=tmp_path,
        distance_range=(1, 1),
        batch_size=1,
        mode="offline",
        shuffle=False,
        worker_count=0,
        prediction_objective="autoregressive",
        fixed_length_mode="global_max",
    )

    xs, ys = next(task)
    expected_seq = len(prompt) + len(completion_long) - 1
    assert xs.shape == (1, expected_seq)
    assert ys.shape == (1, expected_seq)


def test_layer_task_offline_all_at_once_returns_prompt_and_completion(tmp_path: Path) -> None:
    distance_dir = tmp_path / "distance_001"
    distance_dir.mkdir(parents=True)
    shard_path = distance_dir / "shard_00000.array_record"

    tokenizer = tok.build_tokenizer_from_atoms(["p0_1", "p1_1"])
    prompt = tokenizer.tokenize_prompt(Sequent([Atom("p0_1")], Atom("p1_1")))
    completion = tokenizer.encode_completion("p1_1")
    rec = {
        "prompt": np.array(prompt, dtype=np.int32),
        "completions": [np.array(completion, dtype=np.int32)],
    }
    _write_array_record(shard_path, [rec])
    _write_metadata_for_records(tmp_path, tokenizer, [rec])

    task = LayerTask(
        ds_path=tmp_path,
        distance_range=(1, 1),
        batch_size=1,
        mode="offline",
        shuffle=False,
        worker_count=0,
        prediction_objective="all_at_once",
    )

    xs, ys = next(task)
    assert xs.shape == (1, len(prompt))
    assert ys.shape == (1, len(completion))
    assert np.array_equal(xs[0], np.asarray(prompt, dtype=np.int32))
    assert np.array_equal(ys[0], np.asarray(completion, dtype=np.int32))
    assert np.all(ys[0] != 0)


def test_layer_task_offline_all_at_once_eot_pads_targets(tmp_path: Path) -> None:
    distance_dir = tmp_path / "distance_001"
    distance_dir.mkdir(parents=True)
    shard_path = distance_dir / "shard_00000.array_record"

    tokenizer = tok.build_tokenizer_from_atoms(["p0_1", "p1_1"])
    prompt = tokenizer.tokenize_prompt(Sequent([Atom("p0_1")], Atom("p1_1")))
    completion_short = tokenizer.encode_completion("p1_1")
    completion_long = tokenizer.encode_completion("(p0_1→p1_1)")

    rec1 = {
        "prompt": np.array(prompt, dtype=np.int32),
        "completions": [np.array(completion_short, dtype=np.int32)],
    }
    rec2 = {
        "prompt": np.array(prompt, dtype=np.int32),
        "completions": [np.array(completion_long, dtype=np.int32)],
    }
    _write_array_record(shard_path, [rec1, rec2])
    _write_metadata_for_records(tmp_path, tokenizer, [rec1, rec2])

    task = LayerTask(
        ds_path=tmp_path,
        distance_range=(1, 1),
        batch_size=2,
        mode="offline",
        shuffle=False,
        worker_count=0,
        prediction_objective="all_at_once",
    )

    xs, ys = next(task)
    assert np.array_equal(xs[0], np.asarray(prompt, dtype=np.int32))
    assert np.array_equal(xs[1], np.asarray(prompt, dtype=np.int32))

    expected_short = np.asarray(
        completion_short + [tokenizer.eot_token_id] * (len(completion_long) - len(completion_short)),
        dtype=np.int32,
    )
    assert np.array_equal(ys[0], expected_short)
    assert np.array_equal(ys[1], np.asarray(completion_long, dtype=np.int32))


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
        online_prefetch_backend="sync",
    )

    xs, ys = next(task)
    assert xs.shape[0] == 4
    assert ys.shape[0] == 4
    assert xs.dtype == np.int32
    assert ys.dtype == np.int32
    assert np.any(ys == 0)


def test_layer_task_online_autoreg_global_max_has_stable_length() -> None:
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
        fixed_length_mode="global_max",
        online_prefetch_backend="sync",
    )

    max_completion = 1
    for rules in task.rule_bank.transitions.values():
        for rule in rules:
            max_completion = max(
                max_completion,
                len(task.tokenizer.encode_completion(rule.statement_text)),
            )
    expected_n_seq = 2 * int(task.rule_bank.props_per_layer) + 2 + max_completion - 1

    widths = set()
    for _ in range(6):
        xs, ys = next(task)
        assert xs.shape == ys.shape
        widths.add(xs.shape[1])
    assert widths == {expected_n_seq}


def test_layer_task_online_autoreg_global_max_explicit_override() -> None:
    fixed_n_seq = 64
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
        fixed_length_mode="global_max",
        fixed_length_n_seq=fixed_n_seq,
        online_prefetch_backend="sync",
    )

    for _ in range(3):
        xs, ys = next(task)
        assert xs.shape == (4, fixed_n_seq)
        assert ys.shape == (4, fixed_n_seq)


def test_layer_task_online_sampling_all_at_once() -> None:
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
        prediction_objective="all_at_once",
        online_prefetch_backend="sync",
    )

    xs, ys = next(task)
    assert xs.shape[0] == 4
    assert ys.shape[0] == 4
    assert xs.dtype == np.int32
    assert ys.dtype == np.int32
    assert np.all(np.any(ys == task.tokenizer.eot_token_id, axis=1))


def test_layer_task_global_max_does_not_change_all_at_once_shapes() -> None:
    fixed_n_seq = 64
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
        prediction_objective="all_at_once",
        fixed_length_mode="global_max",
        fixed_length_n_seq=fixed_n_seq,
        online_prefetch_backend="sync",
    )

    xs, ys = next(task)
    assert xs.shape[0] == 4
    assert ys.shape[0] == 4
    assert xs.shape[1] < fixed_n_seq
    assert ys.shape[1] < fixed_n_seq


def test_layer_task_rejects_unknown_prediction_objective() -> None:
    with pytest.raises(ValueError, match="prediction_objective"):
        LayerTask(mode="online", prediction_objective="unknown")


def test_layer_task_rejects_unknown_fixed_length_mode() -> None:
    with pytest.raises(ValueError, match="fixed_length_mode"):
        LayerTask(mode="online", fixed_length_mode="unknown")


def test_layer_task_online_prefetch_enabled_by_default() -> None:
    task = LayerTask(
        distance_range=(1, 2),
        batch_size=4,
        mode="online",
        seed=7,
        n_layers=6,
        props_per_layer=5,
        rules_per_transition=8,
        k_in_max=2,
        k_out_max=2,
        initial_ant_max=3,
        online_prefetch_workers=1,
    )
    try:
        assert task.online_prefetch_enabled
        assert task.online_prefetch_backend_resolved in {"process", "thread"}
        assert task.online_prefetch_workers_resolved == 1
        assert task.online_prefetch_buffer_size_resolved >= task.batch_size
        xs, ys = next(task)
        assert xs.shape[0] == 4
        assert ys.shape[0] == 4
    finally:
        task.close()


def test_layer_task_online_prefetch_thread_context_manager_closes() -> None:
    with LayerTask(
        distance_range=(1, 2),
        batch_size=4,
        mode="online",
        seed=8,
        n_layers=6,
        props_per_layer=5,
        rules_per_transition=8,
        k_in_max=2,
        k_out_max=2,
        initial_ant_max=3,
        online_prefetch_backend="thread",
        online_prefetch_workers=1,
        online_prefetch_buffer_size=8,
    ) as task:
        assert task.online_prefetch_enabled
        assert task.online_prefetch_backend_resolved == "thread"
        next(task)
    assert not task.online_prefetch_enabled
    assert task._online_prefetch_buffer is None


def test_layer_task_online_prefetch_rejects_invalid_config() -> None:
    with pytest.raises(ValueError, match="online_prefetch_backend"):
        LayerTask(mode="online", online_prefetch_backend="bad-backend")
    with pytest.raises(ValueError, match="online_prefetch_workers"):
        LayerTask(mode="online", online_prefetch_workers=0)
    with pytest.raises(ValueError, match="online_prefetch_buffer_size"):
        LayerTask(mode="online", online_prefetch_buffer_size=0)


def test_layer_task_online_prefetch_process_fallback(monkeypatch) -> None:
    def _raise_process(*args, **kwargs):
        raise OSError("process pool unavailable")

    monkeypatch.setattr("task.layer.ProcessPoolExecutor", _raise_process)
    task = LayerTask(
        distance_range=(1, 2),
        batch_size=2,
        mode="online",
        seed=9,
        n_layers=6,
        props_per_layer=5,
        rules_per_transition=8,
        k_in_max=2,
        k_out_max=2,
        initial_ant_max=3,
        online_prefetch_backend="process",
        online_prefetch_workers=1,
    )
    try:
        assert task.online_prefetch_backend_resolved in {"thread", "sync"}
        if task.online_prefetch_backend_resolved == "thread":
            assert task.online_prefetch_enabled
            xs, ys = next(task)
            assert xs.shape[0] == 2
            assert ys.shape[0] == 2
    finally:
        task.close()
