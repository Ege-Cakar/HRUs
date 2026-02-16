"""Generate ArrayRecord datasets for layered tasks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import pickle
import time

import numpy as np
from tqdm import tqdm

from array_record.python import array_record_module

if __package__ in (None, ""):
    from util.rule_bank import (  # type: ignore
        RuleBank,
        build_random_rule_bank,
        load_rule_bank,
        sample_problem,
        save_rule_bank,
    )
    from util import tokenize_layer  # type: ignore
else:
    from .util.rule_bank import (
        RuleBank,
        build_random_rule_bank,
        load_rule_bank,
        sample_problem,
        save_rule_bank,
    )
    from .util import tokenize_layer

from task.prop_gen.util.elem import Atom, Sequent


@dataclass
class _AutoregStats:
    max_token: int = 0
    max_seq: int = 0
    max_prompt_seq: int = 0
    max_completion_seq: int = 0


class ArrayRecordShardWriter:
    def __init__(
        self,
        out_dir: Path,
        examples_per_shard: int,
        writer_options: str,
    ) -> None:
        self.out_dir = out_dir
        self.examples_per_shard = max(1, examples_per_shard)
        self.writer_options = writer_options
        self.shard_idx = 0
        self.total_records = 0
        self._records_in_shard = 0
        self._writer = None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._open_writer()

    def _shard_path(self) -> Path:
        return self.out_dir / f"shard_{self.shard_idx:05d}.array_record"

    def _open_writer(self) -> None:
        self._writer = array_record_module.ArrayRecordWriter(
            str(self._shard_path()),
            self.writer_options,
        )

    def _close_writer(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def _rollover(self) -> None:
        self._close_writer()
        self.shard_idx += 1
        self._records_in_shard = 0
        self._open_writer()

    def write(self, payload: bytes) -> None:
        if self._records_in_shard >= self.examples_per_shard:
            self._rollover()
        self._writer.write(payload)
        self._records_in_shard += 1
        self.total_records += 1

    def close(self) -> None:
        self._close_writer()

    @property
    def shard_count(self) -> int:
        if self.total_records == 0:
            return 0
        return self.shard_idx + 1


@dataclass
class _DistanceStateAutoreg:
    writer: ArrayRecordShardWriter
    examples: int = 0
    stats: _AutoregStats = field(default_factory=_AutoregStats)


def _default_seed() -> int:
    return int(np.random.randint(0, np.iinfo(np.int32).max))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate layered datasets.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)

    parser.add_argument("--n-layers", type=int, default=16)
    parser.add_argument("--props-per-layer", type=int, default=8)
    parser.add_argument("--rules-per-transition", type=int, default=32)
    parser.add_argument("--k-in-max", type=int, default=3)
    parser.add_argument("--k-out-max", type=int, default=3)
    parser.add_argument("--initial-ant-max", type=int, default=3)

    parser.add_argument("--min-distance", type=int, default=1)
    parser.add_argument("--max-distance", type=int, default=4)
    parser.add_argument("--examples-per-distance", type=int, default=10_000)
    parser.add_argument("--examples-per-shard", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--rule-bank-path", type=Path, default=None)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--arrayrecord-options",
        type=str,
        default="group_size:1",
        help="ArrayRecord writer options string.",
    )
    return parser.parse_args()


def _update_autoreg_stats(stats: _AutoregStats, prompt: list[int], completion: list[int]) -> None:
    if prompt:
        stats.max_token = max(stats.max_token, max(prompt))
    if completion:
        stats.max_token = max(stats.max_token, max(completion))

    stats.max_prompt_seq = max(stats.max_prompt_seq, len(prompt))
    stats.max_completion_seq = max(stats.max_completion_seq, len(completion))
    stats.max_seq = max(stats.max_seq, len(prompt) + len(completion) - 1)


def _build_or_load_rule_bank(args: argparse.Namespace, rng: np.random.Generator) -> RuleBank:
    if args.rule_bank_path is not None:
        return load_rule_bank(args.rule_bank_path)
    return build_random_rule_bank(
        n_layers=args.n_layers,
        props_per_layer=args.props_per_layer,
        rules_per_transition=args.rules_per_transition,
        k_in_max=args.k_in_max,
        k_out_max=args.k_out_max,
        rng=rng,
    )


def _write_metadata(
    *,
    out_dir: Path,
    states: dict[int, _DistanceStateAutoreg],
    tokenizer: tokenize_layer.LayerTokenizer,
    created_at: str,
    seed: int,
    examples_per_distance: int,
    min_distance: int,
    max_distance: int,
    rule_bank_path: str,
    config: dict,
) -> None:
    distances = {}
    stats_list = []
    for distance, state in states.items():
        st = state.stats
        stats_list.append(st)
        distances[str(distance)] = {
            "examples": int(state.examples),
            "records": int(state.writer.total_records),
            "shards": int(state.writer.shard_count),
            "stats": {
                "max_token": int(st.max_token),
                "max_seq": int(st.max_seq),
                "max_prompt_seq": int(st.max_prompt_seq),
                "max_completion_seq": int(st.max_completion_seq),
            },
        }

    overall = {
        "max_token": int(max(st.max_token for st in stats_list)),
        "max_seq": int(max(st.max_seq for st in stats_list)),
        "max_prompt_seq": int(max(st.max_prompt_seq for st in stats_list)),
        "max_completion_seq": int(max(st.max_completion_seq for st in stats_list)),
    }

    metadata = {
        "created_at": created_at,
        "format": "arrayrecord_layer",
        "seed": seed,
        "examples_per_distance": int(examples_per_distance),
        "distance_range": [int(min_distance), int(max_distance)],
        "rule_bank": rule_bank_path,
        "config": config,
        "tokenizer": tokenizer.to_dict(),
        "stats": overall,
        "distances": distances,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def main() -> None:
    args = _parse_args()
    if args.seed is None:
        args.seed = _default_seed()

    if args.min_distance < 1:
        raise ValueError("--min-distance must be >= 1")
    if args.max_distance < args.min_distance:
        raise ValueError("--max-distance must be >= --min-distance")

    distances = list(range(args.min_distance, args.max_distance + 1))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    bank = _build_or_load_rule_bank(args, rng)
    tokenizer = tokenize_layer.build_tokenizer_from_rule_bank(bank)
    rule_bank_out = args.out_dir / "rule_bank.json"
    save_rule_bank(rule_bank_out, bank)

    autoreg_states: dict[int, _DistanceStateAutoreg] = {}
    for distance in distances:
        writer = ArrayRecordShardWriter(
            out_dir=args.out_dir / f"distance_{distance:03d}",
            examples_per_shard=args.examples_per_shard,
            writer_options=args.arrayrecord_options,
        )
        autoreg_states[distance] = _DistanceStateAutoreg(writer=writer)

    config = {
        "n_layers": int(bank.n_layers),
        "props_per_layer": int(bank.props_per_layer),
        "rules_per_transition": int(args.rules_per_transition),
        "k_in_max": int(args.k_in_max),
        "k_out_max": int(args.k_out_max),
        "initial_ant_max": int(args.initial_ant_max),
    }

    created_at = time.strftime("%Y-%m-%d %H:%M:%S")
    total_target = len(distances) * int(args.examples_per_distance)
    step = max(1, int(args.log_every)) if args.log_every != 0 else 1

    with tqdm(total=total_target, desc="generate layered", leave=True) as bar:
        pending = 0
        for distance in distances:
            for _ in range(args.examples_per_distance):
                sampled = sample_problem(
                    bank=bank,
                    distance=distance,
                    initial_ant_max=args.initial_ant_max,
                    rng=rng,
                )

                state = autoreg_states[distance]
                for step_idx, (src_layer, ants, rule) in enumerate(
                    zip(sampled.step_layers, sampled.step_ants, sampled.step_rules)
                ):
                    sequent = Sequent([Atom(atom) for atom in ants], Atom(sampled.goal_atom))
                    prompt, completion = tokenizer.tokenize_example(sequent, rule.statement_text)
                    payload = pickle.dumps(
                        {
                            "distance": int(sampled.distance),
                            "start_layer": int(sampled.start_layer),
                            "src_layer": int(src_layer),
                            "step_idx": int(step_idx),
                            "goal_atom": sampled.goal_atom,
                            "prompt": np.asarray(prompt, dtype=np.int32),
                            "completions": [np.asarray(completion, dtype=np.int32)],
                            "statement_text": rule.statement_text,
                        },
                        protocol=5,
                    )
                    state.writer.write(payload)
                    _update_autoreg_stats(state.stats, prompt, completion)
                state.examples += 1

                pending += 1
                if pending >= step:
                    bar.update(pending)
                    pending = 0

        if pending:
            bar.update(pending)

    for state in autoreg_states.values():
        state.writer.close()

    _write_metadata(
        out_dir=args.out_dir,
        states=autoreg_states,
        tokenizer=tokenizer,
        created_at=created_at,
        seed=args.seed,
        examples_per_distance=args.examples_per_distance,
        min_distance=args.min_distance,
        max_distance=args.max_distance,
        rule_bank_path=str(rule_bank_out.resolve()),
        config=config,
    )


if __name__ == "__main__":
    main()
