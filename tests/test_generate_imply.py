from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pickle
import json

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

    base_dir = tmp_path / "imply_out"
    out_dir = base_dir / "size_02"
    shards = sorted(out_dir.glob("shard_*.array_record"))
    assert shards, "Expected at least one shard file"

    total = 0
    max_token = 0
    max_pos = 0
    max_seq = 0
    for shard in shards:
        reader = array_record_module.ArrayRecordReader(str(shard))
        records = reader.read_all()
        reader.close()
        total += len(records)
        for record in records:
            payload = pickle.loads(record)
            sequent = payload["sequent"]
            rule = payload["rule"]
            assert sequent.ndim == 1
            assert rule.ndim == 2
            assert rule.shape[1] == 2
            if sequent.size:
                max_token = max(max_token, int(sequent.max()))
                max_seq = max(max_seq, int(sequent.shape[0]))
            if rule.size:
                max_pos = max(max_pos, int(rule[:, 1].max()))

    assert total == 5

    metadata_path = base_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    size_meta = metadata["sizes"]["2"]
    stats = size_meta["stats"]
    assert stats["max_token"] == max_token
    assert stats["max_pos"] == max_pos
    assert stats["max_seq"] == max_seq
