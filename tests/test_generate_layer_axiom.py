from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path

from array_record.python import array_record_module

from task.layer_gen.util import tokenize_layer_axiom as tok


def _run_generate(tmp_path: Path, *, examples: int = 4) -> Path:
    script = Path(__file__).parents[1] / "task" / "layer_gen" / "generate_layer_axiom.py"
    out_dir = tmp_path / "layer_axiom_out"

    cmd = [
        sys.executable,
        str(script),
        "--out-dir",
        str(out_dir),
        "--n-layers",
        "6",
        "--props-per-layer",
        "4",
        "--rules-per-transition",
        "6",
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
        "--seed",
        "13",
        "--objective",
        "both",
        "--log-every",
        "0",
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


def test_generate_layer_axiom_outputs_and_metadata(tmp_path: Path) -> None:
    examples = 5
    out_dir = _run_generate(tmp_path, examples=examples)

    assert (out_dir / "rule_bank.json").exists()
    assert (out_dir / "metadata.json").exists()

    ar_root = out_dir / "autoreg"
    fs_root = out_dir / "first_step"
    root_meta = json.loads((out_dir / "metadata.json").read_text())
    ar_meta = json.loads((ar_root / "metadata.json").read_text())
    fs_meta = json.loads((fs_root / "metadata.json").read_text())
    ar_tokenizer = tok.LayerAxiomTokenizer.from_dict(ar_meta["tokenizer"])
    fs_tokenizer = tok.LayerAxiomTokenizer.from_dict(fs_meta["tokenizer"])

    assert root_meta["tokenizer"]["version"] == tok.TOKENIZER_VERSION
    assert ar_meta["stats"]["max_token"] + 1 <= ar_meta["tokenizer"]["vocab_size"]
    assert fs_meta["stats"]["max_token"] + 1 <= fs_meta["tokenizer"]["vocab_size"]
    assert ar_meta["tokenizer"]["vocab_size"] < 256

    for distance in (1, 2):
        ar_count = _count_records(ar_root / f"distance_{distance:03d}")
        fs_count = _count_records(fs_root / f"distance_{distance:03d}")

        assert ar_count == examples * distance
        assert fs_count == examples

        assert ar_meta["distances"][str(distance)]["examples"] == examples
        assert fs_meta["distances"][str(distance)]["examples"] == examples

    ar_rec = _read_one(ar_root / "distance_001")
    fs_rec = _read_one(fs_root / "distance_001")

    # Decode checks ensure tokenization is consistent and parseable.
    sequent = ar_tokenizer.decode_prompt(ar_rec["prompt"].tolist())
    statement = ar_tokenizer.decode_completion_text(ar_rec["completions"][0].tolist())
    assert str(sequent.cons).startswith("p")
    assert "→" in statement
    assert "(" not in statement
    assert ")" not in statement

    fs_seq = fs_tokenizer.decode_prompt(fs_rec["prompt"].tolist())
    fs_stmt = fs_tokenizer.decode_completion_text(fs_rec["target_first"].tolist())
    assert str(fs_seq.cons).startswith("p")
    assert "→" in fs_stmt
    assert "(" not in fs_stmt
    assert ")" not in fs_stmt
