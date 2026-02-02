from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pickle

from array_record.python import array_record_module


def _run_generate(tmp_path: Path) -> None:
    script = Path(__file__).parents[1] / "task" / "prop_gen" / "generate_imply.py"
    out_dir = tmp_path / "imply_out"
    cmd = [
        sys.executable,
        str(script),
        "--out-dir",
        str(out_dir),
        "--n-vars",
        "2",
        "--min-size",
        "2",
        "--max-size",
        "2",
        "--examples-per-size",
        "5",
        "--examples-per-shard",
        "3",
        "--workers",
        "1",
        "--log-every",
        "0",
    ]
    subprocess.run(cmd, check=True, cwd=script.parent)


def test_generate_imply_arrayrecord(tmp_path: Path) -> None:
    _run_generate(tmp_path)

    out_dir = tmp_path / "imply_out" / "size_02"
    shards = sorted(out_dir.glob("shard_*.array_record"))
    assert shards, "Expected at least one shard file"

    total = 0
    for shard in shards:
        reader = array_record_module.ArrayRecordReader(str(shard))
        records = reader.read_all()
        reader.close()
        total += len(records)
        for record in records:
            payload = pickle.loads(record)
            sequent = payload["sequent"]
            rules = payload["rules"]
            assert sequent.ndim == 1
            assert rules.ndim == 2
            assert rules.shape[1] == 2

    assert total == 5
