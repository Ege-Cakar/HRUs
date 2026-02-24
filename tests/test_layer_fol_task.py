from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from array_record.python import array_record_module

from task.layer_fol import FOLLayerTask
from task.layer_gen.util import tokenize_layer_fol as tok
from task.layer_gen.util.fol_rule_bank import FOLAtom, FOLSequent


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


def _simple_prompt_and_completion(tokenizer):
    sequent = FOLSequent(
        ants=(FOLAtom("r0_1", ("a", "b")),),
        cons=FOLAtom("r1_1", ("a", "b")),
    )
    prompt = tokenizer.tokenize_prompt(sequent)
    completion = tokenizer.encode_completion("r0_1(a,b) → r1_1(a,b)")
    return prompt, completion


def test_layer_fol_task_offline_autoreg(tmp_path: Path) -> None:
    distance_dir = tmp_path / "distance_001"
    distance_dir.mkdir(parents=True)
    shard_path = distance_dir / "shard_00000.array_record"

    tokenizer = tok.build_tokenizer_from_identifiers(["r0_1", "r1_1", "a", "b", "x1"])
    prompt, completion = _simple_prompt_and_completion(tokenizer)
    rec = {
        "prompt": np.array(prompt, dtype=np.int32),
        "completions": [np.array(completion, dtype=np.int32)],
    }
    _write_array_record(shard_path, [rec])
    _write_metadata_for_records(tmp_path, tokenizer, [rec])

    task = FOLLayerTask(
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


def test_layer_fol_task_offline_autoreg_global_max_uses_metadata_max_seq(tmp_path: Path) -> None:
    distance_dir = tmp_path / "distance_001"
    distance_dir.mkdir(parents=True)
    shard_path = distance_dir / "shard_00000.array_record"

    tokenizer = tok.build_tokenizer_from_identifiers(["r0_1", "r1_1", "a", "b", "x1"])
    prompt, completion_long = _simple_prompt_and_completion(tokenizer)
    completion_short = tokenizer.encode_completion("r1_1(a,b)")

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

    task = FOLLayerTask(
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


def test_layer_fol_task_online_sampling_autoreg() -> None:
    task = FOLLayerTask(
        distance_range=(1, 2),
        batch_size=4,
        mode="online",
        seed=11,
        n_layers=6,
        predicates_per_layer=4,
        rules_per_transition=8,
        arity_max=3,
        vars_per_rule_max=4,
        constants=("a", "b", "c"),
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


def test_layer_fol_task_online_autoreg_global_max_has_stable_length() -> None:
    task = FOLLayerTask(
        distance_range=(1, 2),
        batch_size=4,
        mode="online",
        seed=12,
        n_layers=6,
        predicates_per_layer=4,
        rules_per_transition=8,
        arity_max=3,
        vars_per_rule_max=4,
        constants=("a", "b", "c"),
        k_in_max=2,
        k_out_max=2,
        initial_ant_max=3,
        fixed_length_mode="global_max",
        online_prefetch_backend="sync",
    )

    widths = set()
    for _ in range(6):
        xs, ys = next(task)
        assert xs.shape == ys.shape
        widths.add(xs.shape[1])
    assert widths == {task._global_autoreg_seq_len}


def test_layer_fol_task_online_sampling_all_at_once() -> None:
    task = FOLLayerTask(
        distance_range=(1, 2),
        batch_size=4,
        mode="online",
        seed=13,
        n_layers=6,
        predicates_per_layer=4,
        rules_per_transition=8,
        arity_max=3,
        vars_per_rule_max=4,
        constants=("a", "b", "c"),
        k_in_max=2,
        k_out_max=2,
        initial_ant_max=3,
        prediction_objective="all_at_once",
        online_prefetch_backend="sync",
    )

    xs, ys = next(task)
    assert xs.shape[0] == 4
    assert ys.shape[0] == 4
    assert np.all(np.any(ys == task.tokenizer.eot_token_id, axis=1))


def test_layer_fol_task_rejects_unknown_prediction_objective() -> None:
    with pytest.raises(ValueError, match="prediction_objective"):
        FOLLayerTask(mode="online", prediction_objective="unknown")


def test_layer_fol_task_rejects_unknown_fixed_length_mode() -> None:
    with pytest.raises(ValueError, match="fixed_length_mode"):
        FOLLayerTask(mode="online", fixed_length_mode="unknown")


def test_layer_fol_task_online_prefetch_enabled_by_default() -> None:
    task = FOLLayerTask(
        distance_range=(1, 2),
        batch_size=4,
        mode="online",
        seed=14,
        n_layers=6,
        predicates_per_layer=4,
        rules_per_transition=8,
        arity_max=3,
        vars_per_rule_max=4,
        constants=("a", "b", "c"),
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


def test_layer_fol_task_online_prefetch_thread_context_manager_closes() -> None:
    with FOLLayerTask(
        distance_range=(1, 2),
        batch_size=4,
        mode="online",
        seed=15,
        n_layers=6,
        predicates_per_layer=4,
        rules_per_transition=8,
        arity_max=3,
        vars_per_rule_max=4,
        constants=("a", "b", "c"),
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


def test_layer_fol_task_online_prefetch_rejects_invalid_config() -> None:
    with pytest.raises(ValueError, match="online_prefetch_backend"):
        FOLLayerTask(mode="online", online_prefetch_backend="bad-backend")
    with pytest.raises(ValueError, match="online_prefetch_workers"):
        FOLLayerTask(mode="online", online_prefetch_workers=0)
    with pytest.raises(ValueError, match="online_prefetch_buffer_size"):
        FOLLayerTask(mode="online", online_prefetch_buffer_size=0)


def test_layer_fol_task_online_prefetch_process_fallback(monkeypatch) -> None:
    def _raise_process(*args, **kwargs):
        raise OSError("process pool unavailable")

    monkeypatch.setattr("task.layer_fol.ProcessPoolExecutor", _raise_process)
    task = FOLLayerTask(
        distance_range=(1, 2),
        batch_size=2,
        mode="online",
        seed=16,
        n_layers=6,
        predicates_per_layer=4,
        rules_per_transition=8,
        arity_max=3,
        vars_per_rule_max=4,
        constants=("a", "b", "c"),
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
