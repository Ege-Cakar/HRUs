from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path

from array_record.python import array_record_module


def _write_benchmark(path: Path) -> None:
    records = [
        {
            "theorem_id": "thm_a",
            "proof_len": 2,
            "steps": [
                {"proof_state": "A0", "next_tactic": "intro h"},
                {"proof_state": "A1", "next_tactic": "exact h"},
            ],
        },
        {
            "theorem_id": "thm_b",
            "states": ["B0", "B1", "B2", "B3"],
            "tactics": ["t0", "t1", "t2", "t3"],
        },
        {
            "theorem_id": "thm_c",
            "proof_len": 7,
            "step_idx": 0,
            "proof_state": "C0",
            "next_tactic": "have h : True := trivial",
        },
        {
            "theorem_id": "thm_c",
            "proof_len": 7,
            "step_idx": 1,
            "proof_state": "C1",
            "next_tactic": "exact h",
        },
    ]
    text = "\n".join(json.dumps(record) for record in records)
    path.write_text(text + "\n")


def _run_generate(tmp_path: Path) -> Path:
    benchmark = tmp_path / "benchmark.jsonl"
    _write_benchmark(benchmark)

    script = Path(__file__).parents[1] / "task" / "lean_gen" / "generate_lean_ar.py"
    out_dir = tmp_path / "lean_ar_out"
    cmd = [
        sys.executable,
        str(script),
        "--out-dir",
        str(out_dir),
        "--benchmark",
        str(benchmark),
        "--split-quantile",
        "0.67",
        "--examples-per-shard",
        "2",
        "--workers",
        "1",
        "--seed",
        "17",
        "--log-every",
        "0",
    ]
    subprocess.run(cmd, check=True, cwd=script.parent)
    return out_dir


def _scan_split(split_dir: Path) -> dict:
    depth_dirs = sorted(split_dir.glob("depth_*"))
    assert depth_dirs

    total = 0
    theorem_ids = set()
    max_token = 0
    max_seq = 0
    max_prompt_seq = 0
    max_completion_seq = 0
    depths = set()

    for depth_dir in depth_dirs:
        depth = int(depth_dir.name.split("_")[1])
        shards = sorted(depth_dir.glob("shard_*.array_record"))
        assert shards
        for shard in shards:
            reader = array_record_module.ArrayRecordReader(str(shard))
            records = reader.read_all()
            reader.close()
            for record in records:
                payload = pickle.loads(record)
                prompt = payload["prompt"]
                completions = payload["completions"]

                theorem_ids.add(payload["theorem_id"])
                depths.add(int(payload["proof_len"]))
                assert int(payload["proof_len"]) == depth

                max_prompt_seq = max(max_prompt_seq, int(prompt.shape[0]))
                for completion in completions:
                    max_completion_seq = max(max_completion_seq, int(completion.shape[0]))
                    max_seq = max(max_seq, int(prompt.shape[0] + completion.shape[0] - 1))
                    max_token = max(max_token, int(max(prompt.max(), completion.max())))

                total += 1

    return {
        "total": total,
        "theorem_ids": theorem_ids,
        "depths": depths,
        "max_token": max_token,
        "max_seq": max_seq,
        "max_prompt_seq": max_prompt_seq,
        "max_completion_seq": max_completion_seq,
    }


def test_generate_lean_ar_split_and_metadata(tmp_path: Path) -> None:
    out_dir = _run_generate(tmp_path)

    root_meta = json.loads((out_dir / "metadata.json").read_text())
    train_meta = json.loads((out_dir / "train" / "metadata.json").read_text())
    test_meta = json.loads((out_dir / "test" / "metadata.json").read_text())

    train_scan = _scan_split(out_dir / "train")
    test_scan = _scan_split(out_dir / "test")

    assert train_scan["total"] + test_scan["total"] == 8

    assert train_scan["theorem_ids"].isdisjoint(test_scan["theorem_ids"])
    assert set(root_meta["train_theorem_ids"]) == train_scan["theorem_ids"]
    assert set(root_meta["test_theorem_ids"]) == test_scan["theorem_ids"]

    cutoff = int(root_meta["depth_cutoff"])
    assert max(train_scan["depths"]) <= cutoff
    assert min(test_scan["depths"]) > cutoff

    assert train_meta["stats"]["max_token"] == train_scan["max_token"]
    assert train_meta["stats"]["max_seq"] == train_scan["max_seq"]
    assert train_meta["stats"]["max_prompt_seq"] == train_scan["max_prompt_seq"]
    assert train_meta["stats"]["max_completion_seq"] == train_scan["max_completion_seq"]

    assert test_meta["stats"]["max_token"] == test_scan["max_token"]
    assert test_meta["stats"]["max_seq"] == test_scan["max_seq"]
    assert test_meta["stats"]["max_prompt_seq"] == test_scan["max_prompt_seq"]
    assert test_meta["stats"]["max_completion_seq"] == test_scan["max_completion_seq"]
