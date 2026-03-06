from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path

from array_record.python import array_record_module

from task.layer_gen.util import tokenize_layer_fol as tok


def _run_generate(
    tmp_path: Path,
    *,
    examples: int = 4,
    workers: int = 1,
    completion_format: str = "single",
) -> Path:
    script = Path(__file__).parents[1] / "task" / "layer_gen" / "generate_layer_fol.py"
    out_dir = tmp_path / "layer_fol_out"

    cmd = [
        sys.executable,
        str(script),
        "--out-dir",
        str(out_dir),
        "--n-layers",
        "6",
        "--predicates-per-layer",
        "4",
        "--rules-per-transition",
        "6",
        "--arity-max",
        "3",
        "--vars-per-rule-max",
        "4",
        "--constants",
        "a,b,c",
        "--k-in-max",
        "2",
        "--k-out-max",
        "2",
        "--initial-ant-max",
        "3",
        "--min-distance",
        "1",
        "--max-distance",
        "2",
        "--examples-per-distance",
        str(examples),
        "--examples-per-shard",
        "2",
        "--workers",
        str(workers),
        "--seed",
        "13",
        "--log-every",
        "0",
        "--completion-format",
        str(completion_format),
    ]
    subprocess.run(cmd, check=True, cwd=script.parent)
    return out_dir


def _count_records(distance_dir: Path) -> int:
    shards = sorted(distance_dir.glob("shard_*.array_record"))
    assert shards

    total = 0
    for shard in shards:
        reader = array_record_module.ArrayRecordReader(str(shard))
        total += len(reader.read_all())
        reader.close()
    return total


def _read_one(distance_dir: Path):
    shard = sorted(distance_dir.glob("shard_*.array_record"))[0]
    reader = array_record_module.ArrayRecordReader(str(shard))
    records = reader.read_all()
    reader.close()
    return pickle.loads(records[0])


def test_generate_layer_fol_outputs_and_metadata(tmp_path: Path) -> None:
    examples = 4
    out_dir = _run_generate(tmp_path, examples=examples, workers=2)

    assert (out_dir / "rule_bank.json").exists()
    assert (out_dir / "metadata.json").exists()

    root_meta = json.loads((out_dir / "metadata.json").read_text())
    tokenizer = tok.FOLLayerTokenizer.from_dict(root_meta["tokenizer"])

    assert root_meta["tokenizer"]["version"] == tok.TOKENIZER_VERSION
    assert root_meta["stats"]["max_token"] + 1 <= root_meta["tokenizer"]["vocab_size"]
    assert root_meta["workers"] == 2
    assert root_meta["parallel_backend"] in {"process", "thread"}

    for distance in (1, 2):
        count = _count_records(out_dir / f"distance_{distance:03d}")
        assert count == examples * distance
        assert root_meta["distances"][str(distance)]["examples"] == examples

    rec = _read_one(out_dir / "distance_001")
    sequent = tokenizer.decode_prompt(rec["prompt"].tolist())
    statement = tokenizer.decode_completion_text(rec["completions"][0].tolist())
    assert sequent.cons.text.startswith("r")
    assert "→" in statement


def test_generate_layer_fol_single_worker_metadata(tmp_path: Path) -> None:
    out_dir = _run_generate(tmp_path, examples=2, workers=1)
    root_meta = json.loads((out_dir / "metadata.json").read_text())
    assert root_meta["workers"] == 1
    assert root_meta["parallel_backend"] == "thread"


def test_generate_layer_fol_full_completion_outputs_sequence(tmp_path: Path) -> None:
    out_dir = _run_generate(tmp_path, examples=2, workers=1, completion_format="full")
    root_meta = json.loads((out_dir / "metadata.json").read_text())
    tokenizer = tok.FOLLayerTokenizer.from_dict(root_meta["tokenizer"])
    rec = _read_one(out_dir / "distance_002")

    statements = tokenizer.decode_completion_sequence_texts(rec["completions"][0].tolist())
    assert root_meta["config"]["completion_format"] == "full"
    assert rec["completion_format"] == "full"
    assert rec["statement_texts"] == statements
    assert len(statements) >= 1
