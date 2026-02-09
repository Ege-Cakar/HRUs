from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path

from array_record.python import array_record_module

from task.prop_gen.util.elem import Unprovable
from task.prop_gen.util.sample import count_sequent_symbols
from task.prop_gen.util import tokenize_ar


def _run_generate(
    tmp_path: Path,
    *,
    allow_false: bool,
    out_name: str,
    examples: int = 6,
    min_size: int = 2,
    max_size: int = 2,
    workers: int = 1,
) -> Path:
    script = Path(__file__).parents[1] / "task" / "prop_gen" / "generate_imply_ar.py"
    out_dir = tmp_path / out_name
    cmd = [
        sys.executable,
        str(script),
        "--out-dir",
        str(out_dir),
        "--n-vars",
        "1",
        "--min-size",
        str(min_size),
        "--max-size",
        str(max_size),
        "--examples-per-size",
        str(examples),
        "--examples-per-shard",
        "3",
        "--workers",
        str(workers),
        "--seed",
        "7",
        "--log-every",
        "0",
    ]
    if allow_false:
        cmd.append("--allow-false")
    subprocess.run(cmd, check=True, cwd=script.parent)
    return out_dir


def _scan(out_dir: Path):
    shard_dir = out_dir / "size_02"
    shards = sorted(shard_dir.glob("shard_*.array_record"))
    assert shards

    total = 0
    max_token = 0
    max_seq = 0
    max_prompt_seq = 0
    max_completion_seq = 0
    has_unprovable = False

    unprovable_id = tokenize_ar.rule_type_to_id[Unprovable]

    for shard in shards:
        reader = array_record_module.ArrayRecordReader(str(shard))
        records = reader.read_all()
        reader.close()

        for rec in records:
            payload = pickle.loads(rec)
            prompt = payload["prompt"]
            completions = payload["completions"]

            assert prompt.ndim == 1
            assert len(completions) > 0

            sequent = tokenize_ar.decode_prompt(prompt.tolist())
            assert count_sequent_symbols(sequent) == 2

            max_prompt_seq = max(max_prompt_seq, int(prompt.shape[0]))
            for completion in completions:
                assert completion.ndim == 1
                comp = completion.tolist()
                assert comp[-1] == tokenize_ar.eot_token_id
                if comp[0] == unprovable_id:
                    has_unprovable = True

                full_len = int(prompt.shape[0] + completion.shape[0] - 1)
                max_seq = max(max_seq, full_len)
                max_completion_seq = max(max_completion_seq, int(completion.shape[0]))
                max_token = max(max_token, int(max(prompt.max(), completion.max())))
            total += 1

    return {
        "total": total,
        "max_token": max_token,
        "max_seq": max_seq,
        "max_prompt_seq": max_prompt_seq,
        "max_completion_seq": max_completion_seq,
        "has_unprovable": has_unprovable,
    }


def _count_records(out_dir: Path, size: int) -> int:
    shard_dir = out_dir / f"size_{size:02d}"
    shards = sorted(shard_dir.glob("shard_*.array_record"))
    assert shards
    total = 0
    for shard in shards:
        reader = array_record_module.ArrayRecordReader(str(shard))
        total += len(reader.read_all())
        reader.close()
    return total


def test_generate_imply_ar_arrayrecord_and_metadata(tmp_path: Path) -> None:
    out_dir = _run_generate(tmp_path, allow_false=True, out_name="imply_ar_out", examples=8)
    scan = _scan(out_dir)

    assert scan["total"] == 8

    metadata = json.loads((out_dir / "metadata.json").read_text())
    assert metadata["allow_false"] is True

    size_meta = metadata["sizes"]["2"]["stats"]
    assert size_meta["max_token"] == scan["max_token"]
    assert size_meta["max_seq"] == scan["max_seq"]
    assert size_meta["max_prompt_seq"] == scan["max_prompt_seq"]
    assert size_meta["max_completion_seq"] == scan["max_completion_seq"]


def test_generate_imply_ar_allow_false_flag_controls_unprovable(tmp_path: Path) -> None:
    out_true = _run_generate(tmp_path, allow_false=True, out_name="allow_true", examples=20)
    out_false = _run_generate(tmp_path, allow_false=False, out_name="allow_false", examples=20)

    scan_true = _scan(out_true)
    scan_false = _scan(out_false)

    assert scan_true["has_unprovable"] is True
    assert scan_false["has_unprovable"] is False


def test_generate_imply_ar_multisize_multiprocess_exact_bucket_counts(tmp_path: Path) -> None:
    out_dir = _run_generate(
        tmp_path,
        allow_false=False,
        out_name="multi_workers",
        examples=6,
        min_size=2,
        max_size=3,
        workers=2,
    )

    assert _count_records(out_dir, 2) == 6
    assert _count_records(out_dir, 3) == 6

    metadata = json.loads((out_dir / "metadata.json").read_text())
    assert metadata["workers"] == 2
    assert metadata["sizes"]["2"]["examples"] == 6
    assert metadata["sizes"]["3"]["examples"] == 6
