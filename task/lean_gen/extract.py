"""Extract Lean proof-state -> next-tactic examples for autoregressive training."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Iterable

import numpy as np


_THEOREM_ID_KEYS = (
    "theorem_id",
    "theorem_uid",
    "theorem_full_name",
    "full_name",
    "theorem",
    "decl_name",
    "name",
)

_PROOF_STATE_KEYS = (
    "proof_state",
    "state",
    "state_before",
    "goal",
    "goals",
)

_NEXT_TACTIC_KEYS = (
    "next_tactic",
    "tactic",
    "action",
    "human_tactic_code",
)

_STEP_IDX_KEYS = ("step_idx", "step_id", "idx", "tactic_idx")
_PROOF_LEN_KEYS = ("proof_len", "proof_length", "num_steps", "total_steps")


@dataclass(frozen=True)
class LeanTacticStep:
    theorem_id: str
    proof_len: int
    step_idx: int
    proof_state: str
    next_tactic: str


@dataclass(frozen=True)
class DepthSplit:
    train_quantile: float
    depth_cutoff: int
    train_theorems: tuple[str, ...]
    test_theorems: tuple[str, ...]
    train_examples: tuple[LeanTacticStep, ...]
    test_examples: tuple[LeanTacticStep, ...]


class ExtractionError(ValueError):
    """Raised when records cannot be normalized into tactic-step examples."""


def extract_lean_tactic_steps(
    benchmark: str,
    *,
    max_theorems: int | None = None,
    max_steps_per_theorem: int | None = None,
    seed: int = 0,
) -> list[LeanTacticStep]:
    """Load step-level examples from a LeanDojo export.

    `benchmark` currently accepts a local JSON/JSONL file (or directory containing
    JSONL shards) exported from LeanDojo tooling.
    """
    path = Path(benchmark).expanduser()
    if path.exists():
        raw_records = _load_records_from_path(path)
    else:
        raw_records = _load_records_from_lean_dojo(benchmark)

    examples = _flatten_records(raw_records)
    examples = _normalize_proof_lengths(examples)
    examples = _limit_examples(
        examples,
        max_theorems=max_theorems,
        max_steps_per_theorem=max_steps_per_theorem,
        seed=seed,
    )

    if not examples:
        raise ExtractionError(
            f"No valid (proof_state, next_tactic) examples found in benchmark={benchmark!r}."
        )
    return examples


def split_by_depth_quantile(
    examples: Iterable[LeanTacticStep],
    train_quantile: float,
) -> DepthSplit:
    """Split examples theorem-wise: shallow theorems to train, deeper to test."""
    if train_quantile <= 0 or train_quantile >= 1:
        raise ValueError(f"train_quantile must be in (0, 1), got {train_quantile}")

    theorem_depths: dict[str, int] = {}
    for ex in examples:
        theorem_depths[ex.theorem_id] = max(theorem_depths.get(ex.theorem_id, 0), ex.proof_len)

    if len(theorem_depths) < 2:
        raise ValueError("Need at least two theorems for a train/test split.")

    depth_values = np.array(list(theorem_depths.values()), dtype=np.int32)
    depth_cutoff = int(np.quantile(depth_values, train_quantile, method="higher"))

    train_theorems = sorted(
        theorem_id
        for theorem_id, depth in theorem_depths.items()
        if depth <= depth_cutoff
    )
    test_theorems = sorted(
        theorem_id
        for theorem_id, depth in theorem_depths.items()
        if depth > depth_cutoff
    )

    if not train_theorems or not test_theorems:
        ordered = sorted(theorem_depths.items(), key=lambda item: (item[1], item[0]))
        n_theorems = len(ordered)
        train_n = int(np.floor(n_theorems * train_quantile))
        train_n = max(1, min(n_theorems - 1, train_n))
        train_theorems = sorted(theorem_id for theorem_id, _ in ordered[:train_n])
        test_theorems = sorted(theorem_id for theorem_id, _ in ordered[train_n:])
        depth_cutoff = theorem_depths[train_theorems[-1]]

    train_set = set(train_theorems)
    test_set = set(test_theorems)

    train_examples = sorted(
        (ex for ex in examples if ex.theorem_id in train_set),
        key=lambda ex: (ex.theorem_id, ex.step_idx),
    )
    test_examples = sorted(
        (ex for ex in examples if ex.theorem_id in test_set),
        key=lambda ex: (ex.theorem_id, ex.step_idx),
    )

    return DepthSplit(
        train_quantile=float(train_quantile),
        depth_cutoff=depth_cutoff,
        train_theorems=tuple(train_theorems),
        test_theorems=tuple(test_theorems),
        train_examples=tuple(train_examples),
        test_examples=tuple(test_examples),
    )


def hashed_theorem_ids(theorem_ids: Iterable[str]) -> list[str]:
    return [hashlib.sha256(theorem_id.encode("utf-8")).hexdigest()[:12] for theorem_id in theorem_ids]


def _load_records_from_path(path: Path) -> list[dict]:
    if path.is_dir():
        records: list[dict] = []
        for shard in sorted(path.glob("*.jsonl")):
            records.extend(_load_jsonl(shard))
        if records:
            return records
        for shard in sorted(path.glob("*.json")):
            records.extend(_load_json(shard))
        return records

    if path.suffix == ".jsonl":
        return _load_jsonl(path)
    if path.suffix == ".json":
        return _load_json(path)

    raise ExtractionError(f"Unsupported benchmark file format: {path}")


def _load_records_from_lean_dojo(benchmark: str) -> list[dict]:
    try:
        import lean_dojo_v2  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "lean-dojo-v2 is required. Install project dependencies and pass a "
            "LeanDojo-exported JSON/JSONL benchmark file path."
        ) from exc

    raise ExtractionError(
        "Named LeanDojo benchmark loading is not available in this script yet. "
        "Pass --benchmark as a local LeanDojo-exported JSON/JSONL path."
    )


def _load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parsed = json.loads(line)
        if isinstance(parsed, dict):
            out.append(parsed)
    return out


def _load_json(path: Path) -> list[dict]:
    parsed = json.loads(path.read_text())
    if isinstance(parsed, list):
        return [entry for entry in parsed if isinstance(entry, dict)]
    if isinstance(parsed, dict):
        if "records" in parsed and isinstance(parsed["records"], list):
            return [entry for entry in parsed["records"] if isinstance(entry, dict)]
        return [parsed]
    return []


def _flatten_records(records: Iterable[dict]) -> list[LeanTacticStep]:
    out: list[LeanTacticStep] = []
    for idx, record in enumerate(records):
        theorem_id = _normalize_text(_first_present(record, _THEOREM_ID_KEYS))
        if theorem_id is None:
            theorem_id = f"theorem_{idx:07d}"

        if "states" in record and "tactics" in record:
            states = record.get("states")
            tactics = record.get("tactics")
            if isinstance(states, list) and isinstance(tactics, list):
                proof_len = _coerce_int(record.get("proof_len"), default=len(tactics))
                for step_idx, tactic in enumerate(tactics):
                    if step_idx >= len(states):
                        break
                    step = _build_step(
                        theorem_id=theorem_id,
                        proof_len=proof_len,
                        step_idx=step_idx,
                        proof_state=states[step_idx],
                        next_tactic=tactic,
                    )
                    if step is not None:
                        out.append(step)
                continue

        maybe_steps = record.get("steps") or record.get("trace")
        if isinstance(maybe_steps, list):
            proof_len = _coerce_int(record.get("proof_len"), default=len(maybe_steps))
            for step_idx, raw_step in enumerate(maybe_steps):
                if isinstance(raw_step, dict):
                    step = _build_step(
                        theorem_id=theorem_id,
                        proof_len=_coerce_int(_first_present(raw_step, _PROOF_LEN_KEYS), default=proof_len),
                        step_idx=_coerce_int(_first_present(raw_step, _STEP_IDX_KEYS), default=step_idx),
                        proof_state=_first_present(raw_step, _PROOF_STATE_KEYS),
                        next_tactic=_first_present(raw_step, _NEXT_TACTIC_KEYS),
                    )
                elif isinstance(raw_step, (list, tuple)) and len(raw_step) >= 2:
                    step = _build_step(
                        theorem_id=theorem_id,
                        proof_len=proof_len,
                        step_idx=step_idx,
                        proof_state=raw_step[0],
                        next_tactic=raw_step[1],
                    )
                else:
                    step = None
                if step is not None:
                    out.append(step)
            continue

        step = _build_step(
            theorem_id=theorem_id,
            proof_len=_coerce_int(_first_present(record, _PROOF_LEN_KEYS), default=0),
            step_idx=_coerce_int(_first_present(record, _STEP_IDX_KEYS), default=0),
            proof_state=_first_present(record, _PROOF_STATE_KEYS),
            next_tactic=_first_present(record, _NEXT_TACTIC_KEYS),
        )
        if step is not None:
            out.append(step)

    return out


def _normalize_proof_lengths(examples: list[LeanTacticStep]) -> list[LeanTacticStep]:
    inferred: dict[str, int] = {}
    for ex in examples:
        inferred[ex.theorem_id] = max(inferred.get(ex.theorem_id, 0), ex.step_idx + 1)

    out: list[LeanTacticStep] = []
    for ex in examples:
        proof_len = ex.proof_len if ex.proof_len > 0 else inferred[ex.theorem_id]
        out.append(
            LeanTacticStep(
                theorem_id=ex.theorem_id,
                proof_len=proof_len,
                step_idx=ex.step_idx,
                proof_state=ex.proof_state,
                next_tactic=ex.next_tactic,
            )
        )
    return out


def _limit_examples(
    examples: list[LeanTacticStep],
    *,
    max_theorems: int | None,
    max_steps_per_theorem: int | None,
    seed: int,
) -> list[LeanTacticStep]:
    out = list(examples)

    if max_theorems is not None:
        if max_theorems <= 0:
            raise ValueError(f"max_theorems must be positive, got {max_theorems}")
        theorem_ids = sorted({ex.theorem_id for ex in out})
        if max_theorems < len(theorem_ids):
            rng = random.Random(seed)
            rng.shuffle(theorem_ids)
            keep = set(theorem_ids[:max_theorems])
            out = [ex for ex in out if ex.theorem_id in keep]

    if max_steps_per_theorem is not None:
        if max_steps_per_theorem <= 0:
            raise ValueError(
                f"max_steps_per_theorem must be positive, got {max_steps_per_theorem}"
            )
        kept: list[LeanTacticStep] = []
        buckets: dict[str, list[LeanTacticStep]] = {}
        for ex in out:
            buckets.setdefault(ex.theorem_id, []).append(ex)
        for theorem_id in sorted(buckets):
            steps = sorted(buckets[theorem_id], key=lambda ex: ex.step_idx)
            kept.extend(steps[:max_steps_per_theorem])
        out = kept

    return sorted(out, key=lambda ex: (ex.theorem_id, ex.step_idx))


def _first_present(record: dict, keys: Iterable[str]):
    for key in keys:
        if key in record:
            value = record[key]
            if value is not None:
                return value
    return None


def _coerce_int(value, *, default: int) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_text(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _build_step(
    *,
    theorem_id: str,
    proof_len: int,
    step_idx: int,
    proof_state,
    next_tactic,
) -> LeanTacticStep | None:
    proof_state_text = _normalize_text(proof_state)
    next_tactic_text = _normalize_text(next_tactic)
    if proof_state_text is None or next_tactic_text is None:
        return None

    return LeanTacticStep(
        theorem_id=theorem_id,
        proof_len=max(0, int(proof_len)),
        step_idx=max(0, int(step_idx)),
        proof_state=proof_state_text,
        next_tactic=next_tactic_text,
    )
