"""Task loader/sampler for layered axiom-sequence reasoning."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import pickle
from typing import Iterable

import grain
from grain._src.core import sharding, transforms
from grain._src.python import data_sources, samplers
import numpy as np

from task.layer_gen.util import tokenize_layer_axiom
from task.layer_gen.util.rule_bank import (
    RuleBank,
    build_random_rule_bank,
    load_rule_bank,
    sample_problem,
)
from task.prop_gen.util.elem import Atom, Sequent


class LayerAxiomTask:
    STATS_KEYS_BY_OBJECTIVE = {
        "autoreg": ("max_token", "max_seq", "max_prompt_seq", "max_completion_seq"),
        "first_step": ("max_token", "max_seq", "max_prompt_seq", "max_target_seq"),
    }

    def __init__(
        self,
        ds_path=None,
        distance_range=(1, 4),
        batch_size=128,
        *,
        mode="offline",
        objective="autoreg",
        shuffle=True,
        seed=None,
        worker_count=0,
        reader_options=None,
        drop_remainder=False,
        # online mode / rule bank config
        rule_bank_path=None,
        n_layers=16,
        props_per_layer=8,
        rules_per_transition=32,
        k_in_max=3,
        k_out_max=3,
        initial_ant_max=3,
    ) -> None:
        self.mode = str(mode)
        self.objective = str(objective)
        if self.mode not in {"offline", "online"}:
            raise ValueError(f"mode must be 'offline' or 'online', got {self.mode!r}")
        if self.objective not in self.STATS_KEYS_BY_OBJECTIVE:
            raise ValueError(
                f"objective must be one of {tuple(self.STATS_KEYS_BY_OBJECTIVE)}, got {self.objective!r}"
            )
        if rule_bank_path is None:
            if ds_path is not None:
                rule_bank_path = Path(ds_path) / "rule_bank.json"
                if not rule_bank_path.exists():
                    print(f'warn: rule_bank.json not found in ds_path={ds_path}')
                    rule_bank_path = None

        self.distance_range = distance_range
        self._distances = self._normalize_distances(distance_range)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = seed if seed is not None else int(
            np.random.randint(0, np.iinfo(np.int32).max)
        )
        self.worker_count = worker_count
        self.reader_options = reader_options
        self.drop_remainder = drop_remainder

        self.initial_ant_max = int(initial_ant_max)
        self._rng = np.random.default_rng(self.seed)

        self._epoch = 0
        self._data_source = None
        self._dataloader = None
        self._iterator = None

        self.ds_path = Path(ds_path) if ds_path is not None else None
        self._objective_ds_path: Path | None = None

        self._rule_bank: RuleBank | None = None
        self._tokenizer: tokenize_layer_axiom.LayerAxiomTokenizer | None = None
        if self.mode == "online" or rule_bank_path is not None:
            if rule_bank_path is not None:
                self._rule_bank = load_rule_bank(Path(rule_bank_path))
            else:
                self._rule_bank = build_random_rule_bank(
                    n_layers=int(n_layers),
                    props_per_layer=int(props_per_layer),
                    rules_per_transition=int(rules_per_transition),
                    k_in_max=int(k_in_max),
                    k_out_max=int(k_out_max),
                    rng=self._rng,
                )
            self._tokenizer = tokenize_layer_axiom.build_tokenizer_from_rule_bank(self._rule_bank)

        if self.mode == "offline":
            if self.ds_path is None:
                raise ValueError("ds_path is required when mode='offline'.")
            self._objective_ds_path = self._resolve_objective_path(self.ds_path, self.objective)
            metadata_path = self._objective_ds_path / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            self._tokenizer = tokenize_layer_axiom.tokenizer_from_metadata(metadata)
            self.stats = self._stats_from_metadata(
                self._objective_ds_path,
                self._distances,
                self.objective,
            )

            self._data_source = self._build_data_source()
            self._dataloader = self._build_dataloader()
            self._iterator = iter(self._dataloader)
        else:
            self.stats = {}

    def __next__(self):
        if self.mode == "offline":
            try:
                return next(self._iterator)
            except StopIteration:
                self._epoch += 1
                self._dataloader = self._build_dataloader()
                self._iterator = iter(self._dataloader)
                return next(self._iterator)

        records = [self._sample_online_record() for _ in range(self.batch_size)]
        if self.objective == "autoreg":
            return _batch_records_autoreg(records)
        return _batch_records_first_step(records)

    def __iter__(self):
        return self

    @staticmethod
    def _normalize_distances(distance_range) -> list[int]:
        if isinstance(distance_range, int):
            return [distance_range]
        if isinstance(distance_range, tuple) and len(distance_range) == 2:
            start, end = distance_range
            if start > end:
                start, end = end, start
            return list(range(int(start), int(end) + 1))
        return [int(distance) for distance in distance_range]

    @staticmethod
    def _resolve_objective_path(ds_path: Path, objective: str) -> Path:
        objective_path = ds_path / objective
        if (objective_path / "metadata.json").exists():
            return objective_path
        return ds_path

    @classmethod
    def stats_from_metadata(cls, ds_path, distance_range, *, objective="autoreg") -> dict:
        distances = cls._normalize_distances(distance_range)
        root = cls._resolve_objective_path(Path(ds_path), objective)
        return cls._stats_from_metadata(root, distances, objective)

    @classmethod
    def _stats_from_metadata(
        cls,
        ds_path: Path,
        distances: Iterable[int],
        objective: str,
    ) -> dict:
        metadata_path = ds_path / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Missing metadata at {metadata_path}. Regenerate the dataset."
            )

        metadata = json.loads(metadata_path.read_text())
        distance_meta = metadata.get("distances", {})

        stats_list = []
        missing = []
        for distance in distances:
            stats = distance_meta.get(str(distance), {}).get("stats")
            if stats is None:
                missing.append(distance)
                continue
            stats_list.append(stats)

        if missing:
            raise ValueError(f"Missing stats for distances {missing} in {metadata_path}.")

        keys = cls.STATS_KEYS_BY_OBJECTIVE[objective]
        return {
            key: max(int(stats.get(key, 0)) for stats in stats_list)
            for key in keys
        }

    def _collect_shards(self, distances: Iterable[int]) -> list[str]:
        if self._objective_ds_path is None:
            raise RuntimeError("Offline shard collection requires objective dataset path.")

        shards: list[str] = []
        for distance in distances:
            distance_dir = self._objective_ds_path / f"distance_{distance:03d}"
            if not distance_dir.exists():
                raise FileNotFoundError(f"Missing distance directory: {distance_dir}")
            distance_shards = sorted(distance_dir.glob("shard_*.array_record"))
            if not distance_shards:
                raise FileNotFoundError(f"No shards found in {distance_dir}")
            shards.extend(str(path) for path in distance_shards)
        return shards

    def _build_data_source(self):
        shards = self._collect_shards(self._distances)
        return data_sources.ArrayRecordDataSource(
            shards,
            reader_options=self.reader_options,
        )

    def _build_dataloader(self) -> grain.DataLoader:
        shard_opts = sharding.NoSharding()
        sampler = samplers.IndexSampler(
            num_records=len(self._data_source),
            shard_options=shard_opts,
            shuffle=self.shuffle,
            num_epochs=1,
            seed=self.seed + self._epoch if self.shuffle else None,
        )

        if self.objective == "autoreg":
            batch_fn = _batch_records_autoreg
        else:
            batch_fn = _batch_records_first_step

        operations = [
            _DecodeRecord(),
            transforms.Batch(
                batch_size=self.batch_size,
                drop_remainder=self.drop_remainder,
                batch_fn=batch_fn,
            ),
        ]

        return grain.DataLoader(
            data_source=self._data_source,
            sampler=sampler,
            operations=operations,
            worker_count=self.worker_count,
            shard_options=shard_opts,
        )

    def _sample_online_record(self) -> dict:
        if self._rule_bank is None:
            raise RuntimeError("Online mode requires a rule bank.")
        if self._tokenizer is None:
            raise RuntimeError("Online mode requires a tokenizer.")

        distance = int(self._rng.choice(self._distances))
        sampled = sample_problem(
            bank=self._rule_bank,
            distance=distance,
            initial_ant_max=self.initial_ant_max,
            rng=self._rng,
        )

        if self.objective == "autoreg":
            step_idx = int(self._rng.integers(0, len(sampled.step_rules)))
            src_layer = sampled.step_layers[step_idx]
            ants = sampled.step_ants[step_idx]
            rule = sampled.step_rules[step_idx]
            sequent = Sequent([Atom(atom) for atom in ants], Atom(sampled.goal_atom))
            prompt, completion = self._tokenizer.tokenize_example(sequent, rule.statement_text)
            return {
                "distance": distance,
                "src_layer": int(src_layer),
                "prompt": np.asarray(prompt, dtype=np.int32),
                "completions": [np.asarray(completion, dtype=np.int32)],
            }

        src_layer = sampled.step_layers[0]
        ants = sampled.step_ants[0]
        rule = sampled.step_rules[0]
        sequent = Sequent([Atom(atom) for atom in ants], Atom(sampled.goal_atom))
        prompt, target = self._tokenizer.tokenize_example(sequent, rule.statement_text)
        return {
            "distance": distance,
            "src_layer": int(src_layer),
            "prompt": np.asarray(prompt, dtype=np.int32),
            "target_first": np.asarray(target, dtype=np.int32),
        }

    @property
    def rule_bank(self) -> RuleBank | None:
        return self._rule_bank

    @property
    def tokenizer(self) -> tokenize_layer_axiom.LayerAxiomTokenizer | None:
        return self._tokenizer


@dataclass(frozen=True)
class _DecodeRecord(transforms.MapTransform):
    def map(self, element):
        return pickle.loads(element)


def _normalize_completions(completions) -> list[np.ndarray]:
    if isinstance(completions, np.ndarray):
        arr = np.asarray(completions, dtype=np.int32)
        if arr.ndim == 1:
            return [arr]
        if arr.ndim == 2:
            return [arr[idx] for idx in range(arr.shape[0])]
        raise ValueError(f"Completions array must be 1D or 2D, got {arr.shape}")

    out: list[np.ndarray] = []
    for completion in completions:
        arr = np.asarray(completion, dtype=np.int32)
        if arr.ndim != 1:
            raise ValueError(f"Completion must be 1D, got {arr.shape}")
        out.append(arr)
    return out


def _pad_sequences(arrays: list[np.ndarray]) -> np.ndarray:
    max_len = max(arr.shape[0] for arr in arrays)
    out = np.zeros((len(arrays), max_len), dtype=np.int32)
    for idx, arr in enumerate(arrays):
        out[idx, : arr.shape[0]] = arr
    return out


def _batch_records_autoreg(records):
    if not records:
        return np.zeros((0, 0), dtype=np.int32), np.zeros((0, 0), dtype=np.int32)

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    for rec in records:
        prompt = np.asarray(rec["prompt"], dtype=np.int32)
        if prompt.ndim != 1:
            raise ValueError(f"Prompt must be 1D, got {prompt.shape}")

        completions = _normalize_completions(rec["completions"])
        if not completions:
            raise ValueError("Cannot sample from empty completion list.")

        pick = np.random.randint(len(completions))
        completion = completions[pick]

        full = np.concatenate([prompt, completion], axis=0)
        if full.shape[0] < 2:
            raise ValueError("Prompt + completion must contain at least 2 tokens.")

        x = full[:-1].copy()
        y = full[1:].copy()

        if prompt.shape[0] > 1:
            y[: prompt.shape[0] - 1] = 0

        xs.append(x)
        ys.append(y)

    return _pad_sequences(xs), _pad_sequences(ys)


def _batch_records_first_step(records):
    if not records:
        return np.zeros((0, 0), dtype=np.int32), np.zeros((0, 0), dtype=np.int32)

    prompts: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for rec in records:
        prompt = np.asarray(rec["prompt"], dtype=np.int32)
        if prompt.ndim != 1:
            raise ValueError(f"Prompt must be 1D, got {prompt.shape}")

        key = "target_first" if "target_first" in rec else "target"
        target = np.asarray(rec[key], dtype=np.int32)
        if target.ndim != 1:
            raise ValueError(f"Target must be 1D, got {target.shape}")

        prompts.append(prompt)
        targets.append(target)

    return _pad_sequences(prompts), _pad_sequences(targets)


def completion_is_valid_for_layer(
    *,
    rule_bank: RuleBank,
    src_layer: int,
    completion_tokens: list[int] | np.ndarray,
    tokenizer: tokenize_layer_axiom.LayerAxiomTokenizer | None = None,
) -> bool:
    try:
        completion = [int(tok) for tok in completion_tokens]
        tokenizer = (
            tokenizer
            if tokenizer is not None
            else tokenize_layer_axiom.build_tokenizer_from_rule_bank(rule_bank)
        )
        statement = tokenizer.decode_completion_text(completion)
    except (ValueError, TypeError):
        return False

    return statement in rule_bank.statement_set(int(src_layer))
