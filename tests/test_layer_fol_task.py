from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from array_record.python import array_record_module

from task.layer_fol import (
    FOLLayerTask,
    _collect_applicable_demo_schemas,
    _find_instantiation_for_rule,
    format_preview_record,
    match_rule_completion_fol,
    print_task_preview,
)
from task.layer_gen.util import tokenize_layer_fol as tok
from task.layer_gen.util.fol_rule_bank import (
    FOLAtom,
    FOLLayerRule,
    FOLSequent,
    build_depth3_icl_split_bundle,
    parse_clause_text,
    save_fol_depth3_icl_split_bundle,
)


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
    completion = tokenizer.encode_completion_texts(["r0_1(a,b) → r1_1(a,b)"])
    return prompt, completion


def _split_prompt_segments(prompt_tokens: np.ndarray, sep_token_id: int) -> list[list[int]]:
    segments: list[list[int]] = []
    current: list[int] = []
    for tok in prompt_tokens.tolist():
        tok = int(tok)
        if tok == int(sep_token_id):
            segments.append(current)
            current = []
        else:
            current.append(tok)
    if current:
        segments.append(current)
    return segments


def _constants_from_atoms(atoms) -> set[str]:
    return {
        str(term)
        for atom in atoms
        for term in atom.args
        if isinstance(term, str) and not term.startswith("x")
    }


def _write_depth3_split_bundle(tmp_path: Path, seed: int = 0) -> Path:
    bundle = build_depth3_icl_split_bundle(
        predicates_per_layer=4,
        rules_01_train=8,
        rules_01_eval=8,
        rules_12_shared=8,
        arity_max=3,
        vars_per_rule_max=4,
        k_in_max=2,
        k_out_max=2,
        constants=("a", "b", "c"),
        rng=np.random.default_rng(seed),
    )
    path = tmp_path / "depth3_icl_split.json"
    save_fol_depth3_icl_split_bundle(path, bundle)
    return path


def _record_target_rule(rec: dict) -> FOLLayerRule:
    src_layer = int(rec["src_layer"])
    statement_texts = list(rec["statement_texts"])
    if len(statement_texts) != 1:
        raise ValueError("Expected a single target statement for oracle matching.")
    lhs, rhs = parse_clause_text(str(statement_texts[0]))
    return FOLLayerRule(
        src_layer=src_layer,
        dst_layer=src_layer + 1,
        lhs=lhs,
        rhs=rhs,
    )


def _record_oracle_schema(task: FOLLayerTask, rec: dict) -> FOLLayerRule:
    prompt = np.asarray(rec["prompt"], dtype=np.int32)
    segments = _split_prompt_segments(prompt, task.tokenizer.sep_token_id)
    main_prompt = list(segments[-1])
    sequent = task.tokenizer.decode_prompt(main_prompt)
    applicable = _collect_applicable_demo_schemas(
        rule_bank=task.rule_bank,
        src_layer=int(rec["src_layer"]),
        ants=sequent.ants,
        max_unify_solutions=task.max_unify_solutions,
    )
    target_rule = _record_target_rule(rec)
    for schema in applicable:
        if _find_instantiation_for_rule(
            template=schema,
            lhs_ground=target_rule.lhs,
            rhs_ground=target_rule.rhs,
        ) is not None:
            return schema
    raise AssertionError("Could not resolve oracle schema for sampled record.")


def _record_has_demo_for_schema(task: FOLLayerTask, rec: dict, schema: FOLLayerRule) -> bool:
    prompt = np.asarray(rec["prompt"], dtype=np.int32)
    segments = _split_prompt_segments(prompt, task.tokenizer.sep_token_id)
    for demo in segments[:-1]:
        lhs_atoms, rhs_atoms = task.tokenizer.decode_completion_clause(
            list(demo) + [int(task.tokenizer.eot_token_id)]
        )
        if _find_instantiation_for_rule(
            template=schema,
            lhs_ground=lhs_atoms,
            rhs_ground=rhs_atoms,
        ) is not None:
            return True
    return False


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
    completion_short = tokenizer.encode_completion_texts(["r1_1(a,b)"])

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


def test_layer_fol_task_online_sampling_full_completion() -> None:
    task = FOLLayerTask(
        distance_range=(2, 2),
        batch_size=4,
        mode="online",
        seed=21,
        n_layers=6,
        predicates_per_layer=4,
        rules_per_transition=8,
        arity_max=3,
        vars_per_rule_max=4,
        constants=("a", "b", "c"),
        k_in_max=2,
        k_out_max=2,
        initial_ant_max=3,
        completion_format="full",
        online_prefetch_backend="sync",
    )

    assert task.tokenizer is not None
    record = task._sample_online_record()
    completion = np.asarray(record["completions"][0], dtype=np.int32)
    statements = task.tokenizer.decode_completion_texts(completion.tolist())
    assert len(statements) >= 1
    assert record["completion_format"] == "full"
    assert record["statement_texts"] == statements


def test_layer_fol_preview_formats_single_completion() -> None:
    task = FOLLayerTask(
        distance_range=(2, 2),
        batch_size=2,
        mode="online",
        seed=11,
        task_split="depth3_fresh_icl",
        split_role="train",
        predicates_per_layer=4,
        rules_per_transition=8,
        fresh_icl_n_predicates=4,
        arity_max=1,
        vars_per_rule_max=4,
        constants=("a", "b"),
        k_in_max=1,
        k_out_max=1,
        initial_ant_max=2,
        min_n_demos=2,
        max_n_demos=2,
        online_prefetch_backend="sync",
    )

    text = format_preview_record(task, task._sample_online_record(), role="train")

    assert "[train]" in text
    assert "sequent:" in text
    assert "completion:" in text
    assert "n_demos=2" in text
    assert "demo[0]:" in text


def test_layer_fol_preview_formats_full_completion() -> None:
    task = FOLLayerTask(
        distance_range=(2, 2),
        batch_size=2,
        mode="online",
        seed=17,
        task_split="depth3_fresh_icl",
        split_role="train",
        predicates_per_layer=4,
        rules_per_transition=8,
        fresh_icl_n_predicates=4,
        arity_max=1,
        vars_per_rule_max=4,
        constants=("a", "b"),
        k_in_max=1,
        k_out_max=1,
        initial_ant_max=2,
        min_n_demos=2,
        max_n_demos=2,
        completion_format="full",
        online_prefetch_backend="sync",
    )

    text = format_preview_record(task, task._sample_online_record(), role="eval")

    assert "[eval]" in text
    assert "prompt:" in text
    assert "n_steps=" in text
    assert "completion[0]:" in text


def test_layer_fol_print_task_preview_prints_requested_example_count(capsys) -> None:
    task = FOLLayerTask(
        distance_range=(2, 2),
        batch_size=2,
        mode="online",
        seed=23,
        task_split="depth3_fresh_icl",
        split_role="train",
        predicates_per_layer=4,
        rules_per_transition=8,
        fresh_icl_n_predicates=4,
        arity_max=1,
        vars_per_rule_max=4,
        constants=("a", "b"),
        k_in_max=1,
        k_out_max=1,
        initial_ant_max=2,
        min_n_demos=1,
        max_n_demos=2,
        online_prefetch_backend="sync",
    )

    print_task_preview(task, role="train", n_examples=3)
    out = capsys.readouterr().out

    assert "TRAIN DATA PREVIEW (3 examples)" in out
    assert out.count("example[") == 3


def test_layer_fol_preview_rejects_offline_task(tmp_path: Path) -> None:
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

    with pytest.raises(ValueError, match="online FOLLayerTask"):
        format_preview_record(task, rec, role="train")


def test_layer_fol_task_online_sampling_respects_k_in_out_min() -> None:
    task = FOLLayerTask(
        distance_range=(1, 2),
        batch_size=4,
        mode="online",
        seed=111,
        n_layers=6,
        predicates_per_layer=5,
        rules_per_transition=8,
        arity_max=3,
        vars_per_rule_max=4,
        constants=("a", "b", "c"),
        k_in_min=2,
        k_in_max=3,
        k_out_min=2,
        k_out_max=4,
        initial_ant_max=3,
        online_prefetch_backend="sync",
    )

    assert task.rule_bank is not None
    for rules in task.rule_bank.transitions.values():
        for rule in rules:
            assert len(rule.lhs) >= 2
            assert len(rule.rhs) >= 2


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


def test_layer_fol_task_offline_rejects_completion_format_mismatch(tmp_path: Path) -> None:
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

    with pytest.raises(ValueError, match="completion_format mismatch"):
        FOLLayerTask(
            ds_path=tmp_path,
            distance_range=(1, 1),
            batch_size=1,
            mode="offline",
            shuffle=False,
            worker_count=0,
            completion_format="full",
        )


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


def test_layer_fol_task_online_autoreg_next_pow2_buckets_length() -> None:
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
        fixed_length_mode="next_pow2",
        online_prefetch_backend="sync",
    )
    try:
        xs, ys = task._apply_autoreg_fixed_length(
            (
                np.ones((4, 30), dtype=np.int32),
                np.ones((4, 30), dtype=np.int32),
            )
        )
        assert xs.shape == (4, 32)
        assert ys.shape == (4, 32)

        xs2, ys2 = task._apply_autoreg_fixed_length(
            (
                np.ones((4, 100), dtype=np.int32),
                np.ones((4, 100), dtype=np.int32),
            )
        )
        assert xs2.shape == (4, 128)
        assert ys2.shape == (4, 128)
    finally:
        task.close()


def test_layer_fol_task_online_autoreg_next_pow2_respects_cap() -> None:
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
        fixed_length_mode="next_pow2",
        fixed_length_n_seq=64,
        online_prefetch_backend="sync",
    )
    try:
        with pytest.raises(ValueError, match="next_pow2"):
            task._apply_autoreg_fixed_length(
                (
                    np.ones((4, 65), dtype=np.int32),
                    np.ones((4, 65), dtype=np.int32),
                )
            )
    finally:
        task.close()


def test_layer_fol_task_rejects_unknown_prediction_objective() -> None:
    with pytest.raises(ValueError, match="prediction_objective"):
        FOLLayerTask(mode="online", prediction_objective="unknown")


def test_layer_fol_task_rejects_unknown_fixed_length_mode() -> None:
    with pytest.raises(ValueError, match="fixed_length_mode"):
        FOLLayerTask(mode="online", fixed_length_mode="unknown")


def test_layer_fol_task_rejects_negative_max_n_demos() -> None:
    with pytest.raises(ValueError, match="max_n_demos"):
        FOLLayerTask(mode="online", max_n_demos=-1)


def test_layer_fol_task_rejects_negative_min_n_demos() -> None:
    with pytest.raises(ValueError, match="min_n_demos"):
        FOLLayerTask(mode="online", max_n_demos=1, min_n_demos=-1)


def test_layer_fol_task_rejects_min_n_demos_greater_than_max_n_demos() -> None:
    with pytest.raises(ValueError, match="min_n_demos"):
        FOLLayerTask(mode="online", max_n_demos=1, min_n_demos=2)


def test_layer_fol_task_include_oracle_requires_positive_demo_bounds() -> None:
    with pytest.raises(ValueError, match="max_n_demos"):
        FOLLayerTask(mode="online", max_n_demos=0, include_oracle=True)
    with pytest.raises(ValueError, match="min_n_demos"):
        FOLLayerTask(mode="online", max_n_demos=1, min_n_demos=0, include_oracle=True)


def test_layer_fol_task_online_sampling_prepends_demos_when_enabled() -> None:
    task = FOLLayerTask(
        distance_range=(1, 2),
        batch_size=2,
        mode="online",
        seed=17,
        n_layers=6,
        predicates_per_layer=4,
        rules_per_transition=8,
        arity_max=3,
        vars_per_rule_max=4,
        constants=("a", "b", "c"),
        k_in_max=2,
        k_out_max=2,
        initial_ant_max=3,
        max_n_demos=3,
        online_prefetch_backend="sync",
    )

    for _ in range(10):
        rec = task._sample_online_record()
        prompt = np.asarray(rec["prompt"], dtype=np.int32)
        segments = _split_prompt_segments(prompt, task.tokenizer.sep_token_id)
        n_demos = len(segments) - 1
        assert 1 <= n_demos <= 3


def test_layer_fol_task_online_sampling_respects_demo_min_max_bounds() -> None:
    task = FOLLayerTask(
        distance_range=(1, 2),
        batch_size=2,
        mode="online",
        seed=170,
        n_layers=6,
        predicates_per_layer=4,
        rules_per_transition=8,
        arity_max=3,
        vars_per_rule_max=4,
        constants=("a", "b", "c"),
        k_in_max=2,
        k_out_max=2,
        initial_ant_max=3,
        min_n_demos=2,
        max_n_demos=4,
        online_prefetch_backend="sync",
    )

    for _ in range(20):
        rec = task._sample_online_record()
        prompt = np.asarray(rec["prompt"], dtype=np.int32)
        segments = _split_prompt_segments(prompt, task.tokenizer.sep_token_id)
        n_demos = len(segments) - 1
        assert 2 <= n_demos <= 4


def test_layer_fol_task_online_sampling_can_sample_zero_demos_when_min_is_zero() -> None:
    task = FOLLayerTask(
        distance_range=(1, 2),
        batch_size=2,
        mode="online",
        seed=171,
        n_layers=6,
        predicates_per_layer=4,
        rules_per_transition=8,
        arity_max=3,
        vars_per_rule_max=4,
        constants=("a", "b", "c"),
        k_in_max=2,
        k_out_max=2,
        initial_ant_max=3,
        min_n_demos=0,
        max_n_demos=2,
        online_prefetch_backend="sync",
    )

    saw_zero_demos = False
    saw_positive_demos = False
    for _ in range(120):
        rec = task._sample_online_record()
        prompt = np.asarray(rec["prompt"], dtype=np.int32)
        segments = _split_prompt_segments(prompt, task.tokenizer.sep_token_id)
        n_demos = len(segments) - 1
        saw_zero_demos = saw_zero_demos or n_demos == 0
        saw_positive_demos = saw_positive_demos or n_demos > 0
        if saw_zero_demos and saw_positive_demos:
            break

    assert saw_zero_demos
    assert saw_positive_demos


def test_layer_fol_task_online_sampling_uses_exact_demo_count_when_min_equals_max() -> None:
    task = FOLLayerTask(
        distance_range=(1, 2),
        batch_size=2,
        mode="online",
        seed=172,
        n_layers=6,
        predicates_per_layer=4,
        rules_per_transition=8,
        arity_max=3,
        vars_per_rule_max=4,
        constants=("a", "b", "c"),
        k_in_max=2,
        k_out_max=2,
        initial_ant_max=3,
        min_n_demos=2,
        max_n_demos=2,
        online_prefetch_backend="sync",
    )

    for _ in range(20):
        rec = task._sample_online_record()
        prompt = np.asarray(rec["prompt"], dtype=np.int32)
        segments = _split_prompt_segments(prompt, task.tokenizer.sep_token_id)
        n_demos = len(segments) - 1
        assert n_demos == 2


def test_layer_fol_task_online_sampling_demo_candidates_sampled_with_replacement() -> None:
    task = FOLLayerTask(
        distance_range=(1, 1),
        batch_size=1,
        mode="online",
        seed=18,
        n_layers=4,
        predicates_per_layer=1,
        rules_per_transition=1,
        arity_max=1,
        vars_per_rule_max=1,
        constants=("a", "b"),
        k_in_max=1,
        k_out_max=1,
        initial_ant_max=1,
        max_n_demos=3,
        online_prefetch_backend="sync",
    )

    saw_multi_demo = False
    saw_duplicate_demo = False
    for _ in range(80):
        rec = task._sample_online_record()
        prompt = np.asarray(rec["prompt"], dtype=np.int32)
        segments = _split_prompt_segments(prompt, task.tokenizer.sep_token_id)
        demo_segments = segments[:-1]
        if len(demo_segments) < 2:
            continue
        saw_multi_demo = True
        if demo_segments[0] == demo_segments[1]:
            saw_duplicate_demo = True
            break

    assert saw_multi_demo
    assert saw_duplicate_demo


def test_layer_fol_task_online_sampling_demo_constants_not_tied_to_antecedent() -> None:
    task = FOLLayerTask(
        distance_range=(1, 1),
        batch_size=1,
        mode="online",
        seed=181,
        n_layers=4,
        predicates_per_layer=2,
        rules_per_transition=4,
        arity_max=2,
        vars_per_rule_max=2,
        constants=("alice", "bob", "carol", "dave"),
        k_in_max=2,
        k_out_max=2,
        initial_ant_max=1,
        max_n_demos=3,
        online_prefetch_backend="sync",
    )

    saw_outside_constant = False
    for _ in range(120):
        rec = task._sample_online_record()
        prompt = np.asarray(rec["prompt"], dtype=np.int32)
        segments = _split_prompt_segments(prompt, task.tokenizer.sep_token_id)
        demo_segments = segments[:-1]
        if not demo_segments:
            continue

        main_prompt = list(segments[-1])
        sequent = task.tokenizer.decode_prompt(main_prompt)
        antecedent_constants = _constants_from_atoms(sequent.ants)

        for demo in demo_segments:
            lhs_atoms, rhs_atoms = task.tokenizer.decode_completion_clause(
                list(demo) + [int(task.tokenizer.eot_token_id)]
            )
            demo_constants = _constants_from_atoms(lhs_atoms + rhs_atoms)
            if demo_constants - antecedent_constants:
                saw_outside_constant = True
                break
        if saw_outside_constant:
            break

    assert saw_outside_constant


def test_layer_fol_task_online_sampling_demo_schemas_are_applicable() -> None:
    task = FOLLayerTask(
        distance_range=(1, 2),
        batch_size=1,
        mode="online",
        seed=182,
        n_layers=6,
        predicates_per_layer=4,
        rules_per_transition=8,
        arity_max=3,
        vars_per_rule_max=4,
        constants=("a", "b", "c", "d"),
        k_in_max=2,
        k_out_max=2,
        initial_ant_max=3,
        max_n_demos=3,
        online_prefetch_backend="sync",
    )

    for _ in range(40):
        rec = task._sample_online_record()
        prompt = np.asarray(rec["prompt"], dtype=np.int32)
        segments = _split_prompt_segments(prompt, task.tokenizer.sep_token_id)
        demo_segments = segments[:-1]
        if not demo_segments:
            continue

        src_layer = int(rec["src_layer"])
        main_prompt = list(segments[-1])
        sequent = task.tokenizer.decode_prompt(main_prompt)
        applicable = _collect_applicable_demo_schemas(
            rule_bank=task.rule_bank,
            src_layer=src_layer,
            ants=sequent.ants,
            max_unify_solutions=task.max_unify_solutions,
        )
        assert applicable

        for demo in demo_segments:
            lhs_atoms, rhs_atoms = task.tokenizer.decode_completion_clause(
                list(demo) + [int(task.tokenizer.eot_token_id)]
            )
            is_from_applicable_schema = any(
                _find_instantiation_for_rule(
                    template=schema,
                    lhs_ground=lhs_atoms,
                    rhs_ground=rhs_atoms,
                )
                is not None
                for schema in applicable
            )
            assert is_from_applicable_schema


def test_layer_fol_task_online_sampling_include_oracle_guarantees_oracle_demo() -> None:
    task = FOLLayerTask(
        distance_range=(1, 2),
        batch_size=1,
        mode="online",
        seed=183,
        n_layers=6,
        predicates_per_layer=4,
        rules_per_transition=8,
        arity_max=3,
        vars_per_rule_max=4,
        constants=("a", "b", "c", "d"),
        k_in_max=2,
        k_out_max=2,
        initial_ant_max=3,
        min_n_demos=3,
        max_n_demos=3,
        include_oracle=True,
        online_prefetch_backend="sync",
    )

    for _ in range(30):
        rec = task._sample_online_record()
        oracle_schema = _record_oracle_schema(task, rec)
        assert _record_has_demo_for_schema(task, rec, oracle_schema)


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
        assert task.online_prefetch_backend_resolved in {"server", "thread", "sync"}
        if task.online_prefetch_backend_resolved != "sync":
            assert task.online_prefetch_enabled
        assert task.online_prefetch_workers_resolved == 1
        assert task.online_prefetch_buffer_size_resolved >= task.batch_size
        xs, ys = next(task)
        assert xs.shape[0] == 4
        assert ys.shape[0] == 4
    finally:
        task.close()


def test_layer_fol_task_online_prefetch_thread_include_oracle_samples() -> None:
    with FOLLayerTask(
        distance_range=(1, 2),
        batch_size=2,
        mode="online",
        seed=142,
        n_layers=6,
        predicates_per_layer=4,
        rules_per_transition=8,
        arity_max=3,
        vars_per_rule_max=4,
        constants=("a", "b", "c"),
        k_in_max=2,
        k_out_max=2,
        initial_ant_max=3,
        min_n_demos=2,
        max_n_demos=2,
        include_oracle=True,
        online_prefetch_backend="thread",
        online_prefetch_workers=1,
        online_prefetch_buffer_size=4,
    ) as task:
        assert task.online_prefetch_enabled
        assert task._online_prefetch_buffer is not None
        records = task._online_prefetch_buffer.take(4)
        assert records
        for rec in records:
            oracle_schema = _record_oracle_schema(task, rec)
            assert _record_has_demo_for_schema(task, rec, oracle_schema)


def test_layer_fol_task_online_prefetch_server_backend_samples() -> None:
    task = FOLLayerTask(
        distance_range=(1, 2),
        batch_size=2,
        mode="online",
        seed=141,
        n_layers=6,
        predicates_per_layer=4,
        rules_per_transition=8,
        arity_max=3,
        vars_per_rule_max=4,
        constants=("a", "b", "c"),
        k_in_max=2,
        k_out_max=2,
        initial_ant_max=3,
        online_prefetch_backend="server",
        online_prefetch_workers=1,
        online_prefetch_buffer_size=4,
    )
    try:
        assert task.online_prefetch_backend_resolved in {"server", "thread", "sync"}
        xs, ys = next(task)
        assert xs.shape[0] == 2
        assert ys.shape[0] == 2
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


def test_layer_fol_task_online_prefetch_thread_supports_demos() -> None:
    with FOLLayerTask(
        distance_range=(1, 2),
        batch_size=2,
        mode="online",
        seed=151,
        n_layers=6,
        predicates_per_layer=4,
        rules_per_transition=8,
        arity_max=3,
        vars_per_rule_max=4,
        constants=("a", "b", "c"),
        k_in_max=2,
        k_out_max=2,
        initial_ant_max=3,
        max_n_demos=2,
        online_prefetch_backend="thread",
        online_prefetch_workers=1,
        online_prefetch_buffer_size=4,
    ) as task:
        xs, ys = next(task)
        assert xs.shape[0] == 2
        assert ys.shape[0] == 2


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

    monkeypatch.setattr("task.layer_fol.task_prefetch.ProcessPoolExecutor", _raise_process)
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


def test_layer_fol_task_online_prefetch_server_fallback(monkeypatch) -> None:
    class _FailServerClient:
        def __init__(self, **kwargs):
            raise RuntimeError("server unavailable")

    monkeypatch.setattr(
        "task.layer_fol.task_prefetch._FOLOnlineSamplerServerClient",
        _FailServerClient,
    )
    task = FOLLayerTask(
        distance_range=(1, 2),
        batch_size=2,
        mode="online",
        seed=166,
        n_layers=6,
        predicates_per_layer=4,
        rules_per_transition=8,
        arity_max=3,
        vars_per_rule_max=4,
        constants=("a", "b", "c"),
        k_in_max=2,
        k_out_max=2,
        initial_ant_max=3,
        online_prefetch_backend="server",
        online_prefetch_workers=1,
    )
    try:
        assert task.online_prefetch_backend_resolved in {"thread", "sync"}
        xs, ys = next(task)
        assert xs.shape[0] == 2
        assert ys.shape[0] == 2
    finally:
        task.close()


def test_layer_fol_task_online_autoreg_global_max_accounts_for_demos() -> None:
    task_no_demo = FOLLayerTask(
        distance_range=(1, 2),
        batch_size=4,
        mode="online",
        seed=19,
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
        max_n_demos=0,
        online_prefetch_backend="sync",
    )
    task_with_demo = FOLLayerTask(
        distance_range=(1, 2),
        batch_size=4,
        mode="online",
        seed=19,
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
        max_n_demos=3,
        online_prefetch_backend="sync",
    )

    assert task_with_demo._global_autoreg_seq_len > task_no_demo._global_autoreg_seq_len

    widths = set()
    for _ in range(6):
        xs, ys = next(task_with_demo)
        assert xs.shape == ys.shape
        widths.add(xs.shape[1])
    assert widths == {task_with_demo._global_autoreg_seq_len}


def test_layer_fol_task_split_mode_requires_online_and_bundle_path(tmp_path: Path) -> None:
    split_path = _write_depth3_split_bundle(tmp_path, seed=201)
    with pytest.raises(ValueError, match="requires mode='online'"):
        FOLLayerTask(
            mode="offline",
            task_split="depth3_icl_transfer",
            split_rule_bundle_path=split_path,
        )

    with pytest.raises(ValueError, match="split_rule_bundle_path"):
        FOLLayerTask(
            mode="online",
            task_split="depth3_icl_transfer",
            distance_range=(2, 2),
        )


def test_layer_fol_task_split_mode_requires_distance_2_only(tmp_path: Path) -> None:
    split_path = _write_depth3_split_bundle(tmp_path, seed=202)
    with pytest.raises(ValueError, match="resolve to \\[2\\]"):
        FOLLayerTask(
            mode="online",
            task_split="depth3_icl_transfer",
            split_role="train",
            split_rule_bundle_path=split_path,
            distance_range=(1, 2),
        )


def test_layer_fol_task_split_train_samples_both_src_layers(tmp_path: Path) -> None:
    split_path = _write_depth3_split_bundle(tmp_path, seed=203)
    task = FOLLayerTask(
        mode="online",
        task_split="depth3_icl_transfer",
        split_role="train",
        split_rule_bundle_path=split_path,
        distance_range=(2, 2),
        batch_size=1,
        seed=203,
        initial_ant_max=3,
        max_n_demos=0,
        online_prefetch_backend="sync",
    )
    seen = set()
    for _ in range(80):
        rec = task._sample_online_record()
        seen.add(int(rec["src_layer"]))
        if seen == {0, 1}:
            break
    assert seen == {0, 1}


def test_layer_fol_task_split_eval_samples_only_src_layer_0(tmp_path: Path) -> None:
    split_path = _write_depth3_split_bundle(tmp_path, seed=204)
    task = FOLLayerTask(
        mode="online",
        task_split="depth3_icl_transfer",
        split_role="eval",
        split_rule_bundle_path=split_path,
        distance_range=(2, 2),
        batch_size=1,
        seed=204,
        initial_ant_max=3,
        max_n_demos=0,
        online_prefetch_backend="sync",
    )
    for _ in range(40):
        rec = task._sample_online_record()
        assert int(rec["src_layer"]) == 0


def test_layer_fol_task_split_train_tokenizer_contains_eval_only_predicates(tmp_path: Path) -> None:
    split_path = _write_depth3_split_bundle(tmp_path, seed=205)
    train_task = FOLLayerTask(
        mode="online",
        task_split="depth3_icl_transfer",
        split_role="train",
        split_rule_bundle_path=split_path,
        distance_range=(2, 2),
        batch_size=1,
        seed=205,
        max_n_demos=0,
        online_prefetch_backend="sync",
    )
    eval_only_predicate = train_task._split_bundle.eval_layer0_predicates[0]
    assert eval_only_predicate not in set(train_task.rule_bank.predicates_for_layer(0))
    assert eval_only_predicate not in train_task.tokenizer.token_to_id
    for ch in eval_only_predicate:
        assert ch in train_task.tokenizer.token_to_id


def test_layer_fol_task_split_eval_demo_schemas_are_applicable(tmp_path: Path) -> None:
    split_path = _write_depth3_split_bundle(tmp_path, seed=206)
    task = FOLLayerTask(
        mode="online",
        task_split="depth3_icl_transfer",
        split_role="eval",
        split_rule_bundle_path=split_path,
        distance_range=(2, 2),
        batch_size=1,
        seed=206,
        initial_ant_max=3,
        max_n_demos=3,
        online_prefetch_backend="sync",
    )

    saw_demo = False
    for _ in range(40):
        rec = task._sample_online_record()
        assert int(rec["src_layer"]) == 0
        prompt = np.asarray(rec["prompt"], dtype=np.int32)
        segments = _split_prompt_segments(prompt, task.tokenizer.sep_token_id)
        demo_segments = segments[:-1]
        if not demo_segments:
            continue
        saw_demo = True

        main_prompt = list(segments[-1])
        sequent = task.tokenizer.decode_prompt(main_prompt)
        applicable = _collect_applicable_demo_schemas(
            rule_bank=task.rule_bank,
            src_layer=0,
            ants=sequent.ants,
            max_unify_solutions=task.max_unify_solutions,
        )
        assert applicable

        for demo in demo_segments:
            lhs_atoms, rhs_atoms = task.tokenizer.decode_completion_clause(
                list(demo) + [int(task.tokenizer.eot_token_id)]
            )
            is_from_applicable_schema = any(
                _find_instantiation_for_rule(
                    template=schema,
                    lhs_ground=lhs_atoms,
                    rhs_ground=rhs_atoms,
                )
                is not None
                for schema in applicable
            )
            assert is_from_applicable_schema
    assert saw_demo


def test_layer_fol_task_split_eval_include_oracle_guarantees_oracle_demo(tmp_path: Path) -> None:
    split_path = _write_depth3_split_bundle(tmp_path, seed=2061)
    task = FOLLayerTask(
        mode="online",
        task_split="depth3_icl_transfer",
        split_role="eval",
        split_rule_bundle_path=split_path,
        distance_range=(2, 2),
        batch_size=1,
        seed=2061,
        initial_ant_max=3,
        min_n_demos=2,
        max_n_demos=2,
        include_oracle=True,
        online_prefetch_backend="sync",
    )

    for _ in range(20):
        rec = task._sample_online_record()
        oracle_schema = _record_oracle_schema(task, rec)
        assert _record_has_demo_for_schema(task, rec, oracle_schema)


def test_layer_fol_task_split_eval_prefetch_thread_forces_src_layer_0(tmp_path: Path) -> None:
    split_path = _write_depth3_split_bundle(tmp_path, seed=207)
    with FOLLayerTask(
        mode="online",
        task_split="depth3_icl_transfer",
        split_role="eval",
        split_rule_bundle_path=split_path,
        distance_range=(2, 2),
        batch_size=2,
        seed=207,
        initial_ant_max=3,
        max_n_demos=1,
        online_prefetch_backend="thread",
        online_prefetch_workers=1,
        online_prefetch_buffer_size=4,
    ) as task:
        assert task.online_prefetch_enabled
        assert task._online_prefetch_buffer is not None
        records = task._online_prefetch_buffer.take(6)
        assert records
        assert all(int(rec["src_layer"]) == 0 for rec in records)
        xs, ys = next(task)
        assert xs.shape[0] == 2
        assert ys.shape[0] == 2


def test_layer_fol_task_fresh_icl_constructs_and_samples() -> None:
    task = FOLLayerTask(
        mode="online",
        task_split="depth3_fresh_icl",
        distance_range=(2, 2),
        batch_size=4,
        seed=301,
        predicates_per_layer=4,
        rules_per_transition=8,
        arity_max=3,
        vars_per_rule_max=4,
        constants=("a", "b", "c"),
        k_in_max=2,
        k_out_max=2,
        initial_ant_max=3,
        max_n_demos=3,
        online_prefetch_backend="sync",
    )
    xs, ys = next(task)
    assert xs.shape[0] == 4
    assert ys.shape[0] == 4
    assert xs.dtype == np.int32
    assert ys.dtype == np.int32


def test_layer_fol_task_fresh_icl_produces_fresh_predicates() -> None:
    import re

    task = FOLLayerTask(
        mode="online",
        task_split="depth3_fresh_icl",
        distance_range=(2, 2),
        batch_size=1,
        seed=302,
        predicates_per_layer=4,
        rules_per_transition=8,
        arity_max=3,
        vars_per_rule_max=4,
        constants=("a", "b", "c"),
        k_in_max=2,
        k_out_max=2,
        initial_ant_max=3,
        max_n_demos=0,
        online_prefetch_backend="sync",
    )
    fresh_re = re.compile(r"r_[a-z0-9]+")
    saw_fresh = False
    for _ in range(20):
        rec = task._sample_online_record()
        prompt = np.asarray(rec["prompt"], dtype=np.int32)
        decoded = task.tokenizer.decode_batch_ids(
            prompt.reshape(1, -1), include_special_tokens=False
        )[0]
        if fresh_re.search(decoded):
            saw_fresh = True
            break
    assert saw_fresh


def test_layer_fol_task_fresh_icl_eval_forces_step_idx_0() -> None:
    task = FOLLayerTask(
        mode="online",
        task_split="depth3_fresh_icl",
        split_role="eval",
        distance_range=(2, 2),
        batch_size=1,
        seed=303,
        predicates_per_layer=4,
        rules_per_transition=8,
        arity_max=3,
        vars_per_rule_max=4,
        constants=("a", "b", "c"),
        k_in_max=2,
        k_out_max=2,
        initial_ant_max=3,
        max_n_demos=0,
        online_prefetch_backend="sync",
    )
    for _ in range(20):
        rec = task._sample_online_record()
        assert int(rec["src_layer"]) == 0


def test_layer_fol_task_fresh_icl_train_samples_both_layers() -> None:
    task = FOLLayerTask(
        mode="online",
        task_split="depth3_fresh_icl",
        split_role="train",
        distance_range=(2, 2),
        batch_size=1,
        seed=304,
        predicates_per_layer=4,
        rules_per_transition=8,
        arity_max=3,
        vars_per_rule_max=4,
        constants=("a", "b", "c"),
        k_in_max=2,
        k_out_max=2,
        initial_ant_max=3,
        max_n_demos=0,
        online_prefetch_backend="sync",
    )
    seen = set()
    for _ in range(80):
        rec = task._sample_online_record()
        seen.add(int(rec["src_layer"]))
        if seen == {0, 1}:
            break
    assert seen == {0, 1}


def test_layer_fol_task_fresh_icl_requires_online_mode() -> None:
    with pytest.raises(ValueError, match="requires mode='online'"):
        FOLLayerTask(
            mode="offline",
            task_split="depth3_fresh_icl",
            distance_range=(2, 2),
        )


def test_layer_fol_task_fresh_icl_requires_distance_2() -> None:
    with pytest.raises(ValueError, match="resolve to \\[2\\]"):
        FOLLayerTask(
            mode="online",
            task_split="depth3_fresh_icl",
            distance_range=(1, 2),
        )


def test_layer_fol_task_fresh_icl_rejects_rule_bank_path(tmp_path: Path) -> None:
    fake_bank = tmp_path / "fake_bank.json"
    fake_bank.write_text("{}")
    with pytest.raises(ValueError, match="rule_bank_path cannot be combined"):
        FOLLayerTask(
            mode="online",
            task_split="depth3_fresh_icl",
            distance_range=(2, 2),
            rule_bank_path=fake_bank,
        )


def test_layer_fol_task_fresh_icl_rejects_split_rule_bundle_path(tmp_path: Path) -> None:
    split_path = _write_depth3_split_bundle(tmp_path, seed=305)
    with pytest.raises(ValueError, match="split_rule_bundle_path cannot be combined"):
        FOLLayerTask(
            mode="online",
            task_split="depth3_fresh_icl",
            distance_range=(2, 2),
            split_rule_bundle_path=split_path,
        )


def test_layer_fol_task_fresh_icl_prefetch_enabled() -> None:
    task = FOLLayerTask(
        mode="online",
        task_split="depth3_fresh_icl",
        distance_range=(2, 2),
        batch_size=2,
        seed=306,
        predicates_per_layer=4,
        rules_per_transition=8,
        arity_max=3,
        vars_per_rule_max=4,
        constants=("a", "b", "c"),
        k_in_max=2,
        k_out_max=2,
        initial_ant_max=3,
        max_n_demos=0,
    )
    assert task.online_prefetch_enabled
    assert task.online_prefetch_backend_resolved in {"server", "thread", "sync"}
    task.close()


def test_layer_fol_task_fresh_icl_prefetch_server_backend_samples() -> None:
    task = FOLLayerTask(
        mode="online",
        task_split="depth3_fresh_icl",
        split_role="train",
        distance_range=(2, 2),
        batch_size=2,
        seed=3061,
        predicates_per_layer=4,
        rules_per_transition=8,
        arity_max=3,
        vars_per_rule_max=4,
        constants=("a", "b", "c"),
        k_in_max=2,
        k_out_max=2,
        initial_ant_max=3,
        max_n_demos=2,
        online_prefetch_backend="server",
        online_prefetch_workers=1,
        online_prefetch_buffer_size=4,
    )
    try:
        assert task.online_prefetch_backend_resolved in {"server", "thread", "sync"}
        xs, ys = next(task)
        assert xs.shape[0] == 2
        assert ys.shape[0] == 2
        rec = task._sample_online_record()
        assert "rule_context" in rec
    finally:
        task.close()


def test_layer_fol_task_fresh_icl_prefetch_server_fallback(monkeypatch) -> None:
    class _FailServerClient:
        def __init__(self, **kwargs):
            raise RuntimeError("server unavailable")

    monkeypatch.setattr(
        "task.layer_fol.task_prefetch._FOLOnlineSamplerServerClient",
        _FailServerClient,
    )
    task = FOLLayerTask(
        mode="online",
        task_split="depth3_fresh_icl",
        distance_range=(2, 2),
        batch_size=2,
        seed=3062,
        predicates_per_layer=4,
        rules_per_transition=8,
        arity_max=3,
        vars_per_rule_max=4,
        constants=("a", "b", "c"),
        k_in_max=2,
        k_out_max=2,
        initial_ant_max=3,
        max_n_demos=0,
        online_prefetch_backend="server",
        online_prefetch_workers=1,
        online_prefetch_buffer_size=4,
    )
    try:
        assert task.online_prefetch_backend_resolved in {"thread", "sync"}
        xs, ys = next(task)
        assert xs.shape[0] == 2
        assert ys.shape[0] == 2
    finally:
        task.close()


def test_layer_fol_task_fresh_icl_respects_fresh_icl_n_predicates() -> None:
    task = FOLLayerTask(
        mode="online",
        task_split="depth3_fresh_icl",
        distance_range=(2, 2),
        batch_size=1,
        seed=307,
        predicates_per_layer=4,
        rules_per_transition=8,
        arity_max=3,
        vars_per_rule_max=4,
        constants=("a", "b", "c"),
        k_in_max=2,
        k_out_max=2,
        initial_ant_max=3,
        max_n_demos=0,
        fresh_icl_n_predicates=6,
        online_prefetch_backend="sync",
    )
    assert task._fresh_icl_n_predicates == 6
    xs, ys = next(task)
    assert xs.shape[0] == 1


def test_layer_fol_task_fresh_icl_demos_use_fresh_rules() -> None:
    task = FOLLayerTask(
        mode="online",
        task_split="depth3_fresh_icl",
        split_role="eval",
        distance_range=(2, 2),
        batch_size=1,
        seed=308,
        predicates_per_layer=4,
        rules_per_transition=8,
        arity_max=3,
        vars_per_rule_max=4,
        constants=("a", "b", "c"),
        k_in_max=2,
        k_out_max=2,
        initial_ant_max=3,
        max_n_demos=3,
        online_prefetch_backend="sync",
    )
    saw_demo = False
    for _ in range(40):
        rec = task._sample_online_record()
        prompt = np.asarray(rec["prompt"], dtype=np.int32)
        segments = _split_prompt_segments(prompt, task.tokenizer.sep_token_id)
        demo_segments = segments[:-1]
        if demo_segments:
            saw_demo = True
            break
    assert saw_demo


def test_layer_fol_task_fresh_icl_records_include_rule_context() -> None:
    task = FOLLayerTask(
        mode="online",
        task_split="depth3_fresh_icl",
        split_role="train",
        distance_range=(2, 2),
        batch_size=1,
        seed=309,
        predicates_per_layer=4,
        rules_per_transition=8,
        arity_max=3,
        vars_per_rule_max=4,
        constants=("a", "b", "c"),
        k_in_max=2,
        k_out_max=2,
        initial_ant_max=3,
        max_n_demos=0,
        online_prefetch_backend="sync",
    )
    seen: set[int] = set()
    for _ in range(100):
        rec = task._sample_online_record()
        src_layer = int(rec["src_layer"])
        ctx = rec["rule_context"]
        assert int(ctx["src_layer"]) == src_layer
        assert isinstance(ctx["active_rule_texts"], list)
        assert isinstance(ctx["fixed_rule_texts"], list)
        assert isinstance(ctx["demo_schema_texts"], list)
        assert isinstance(ctx["demo_instance_texts"], list)
        assert len(ctx["active_rule_texts"]) > 0
        if src_layer == 0:
            assert ctx["fixed_rule_texts"] == []
        else:
            assert len(ctx["fixed_rule_texts"]) > 0
        seen.add(src_layer)
        if seen == {0, 1}:
            break
    assert seen == {0, 1}


def test_layer_fol_task_fresh_icl_rule_context_matches_active_rules() -> None:
    task = FOLLayerTask(
        mode="online",
        task_split="depth3_fresh_icl",
        split_role="eval",
        distance_range=(2, 2),
        batch_size=1,
        seed=310,
        predicates_per_layer=4,
        rules_per_transition=8,
        arity_max=3,
        vars_per_rule_max=4,
        constants=("a", "b", "c"),
        k_in_max=2,
        k_out_max=2,
        initial_ant_max=3,
        max_n_demos=4,
        online_prefetch_backend="sync",
    )

    def _rules_from_texts(src_layer: int, texts: list[str]) -> list[FOLLayerRule]:
        out: list[FOLLayerRule] = []
        for text in texts:
            lhs, rhs = parse_clause_text(str(text))
            out.append(
                FOLLayerRule(
                    src_layer=int(src_layer),
                    dst_layer=int(src_layer) + 1,
                    lhs=lhs,
                    rhs=rhs,
                )
            )
        return out

    saw_demo_context = False
    for _ in range(80):
        rec = task._sample_online_record()
        src_layer = int(rec["src_layer"])
        assert src_layer == 0
        ctx = rec["rule_context"]
        active_rules = _rules_from_texts(src_layer, list(ctx["active_rule_texts"]))
        fixed_rules = _rules_from_texts(src_layer, list(ctx["fixed_rule_texts"]))
        demo_rules = _rules_from_texts(src_layer, list(ctx["demo_schema_texts"]))
        gt_completion = np.asarray(rec["completions"][0], dtype=np.int32)

        matched = match_rule_completion_fol(
            rule_bank=task.rule_bank,
            src_layer=src_layer,
            completion_tokens=gt_completion,
            tokenizer=task.tokenizer,
            active_rules=active_rules,
            fixed_rules=fixed_rules,
            demo_rules=demo_rules,
        )
        assert matched.is_valid_rule
        assert matched.match_source == "active"

        if ctx["demo_schema_texts"] and ctx["demo_instance_texts"]:
            saw_demo_context = True
            schemas = _rules_from_texts(src_layer, list(ctx["demo_schema_texts"]))
            for instance_text in ctx["demo_instance_texts"]:
                lhs, rhs = parse_clause_text(str(instance_text))
                assert any(
                    _find_instantiation_for_rule(
                        template=schema,
                        lhs_ground=lhs,
                        rhs_ground=rhs,
                    )
                    is not None
                    for schema in schemas
                )
            break
    assert saw_demo_context
