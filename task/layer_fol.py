"""Task loader/sampler and evaluation tools for layered first-order tasks."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import json
from pathlib import Path
import pickle
from typing import Any, Callable, Iterable, Protocol

import grain
from grain._src.core import sharding, transforms
from grain._src.python import data_sources, samplers
import numpy as np

from task.layer import AutoregressiveLogitsAdapter, CompletionLogitsAdapter
from task.layer_gen.util import tokenize_layer_fol
from task.layer_gen.util.fol_rule_bank import (
    FOLAtom,
    FOLLayerRule,
    FOLRuleBank,
    FOLSequent,
    build_random_fol_rule_bank,
    load_fol_rule_bank,
    parse_atom_text,
    parse_clause_text,
    sample_fol_problem,
)


class FOLLayerTask:
    STATS_KEYS = ("max_token", "max_seq", "max_prompt_seq", "max_completion_seq")

    def __init__(
        self,
        ds_path=None,
        distance_range=(1, 4),
        batch_size=128,
        *,
        mode="offline",
        shuffle=True,
        seed=None,
        worker_count=0,
        reader_options=None,
        drop_remainder=False,
        prediction_objective="autoregressive",
        fixed_length_mode="batch_max",
        fixed_length_n_seq=None,
        # online mode / rule bank config
        rule_bank_path=None,
        n_layers=16,
        predicates_per_layer=8,
        rules_per_transition=32,
        arity_max=3,
        vars_per_rule_max=4,
        constants=("a", "b", "c", "d"),
        k_in_max=3,
        k_out_max=3,
        initial_ant_max=3,
        sample_max_attempts=4096,
        max_unify_solutions=128,
    ) -> None:
        self.mode = str(mode)
        if self.mode not in {"offline", "online"}:
            raise ValueError(f"mode must be 'offline' or 'online', got {self.mode!r}")
        self.prediction_objective = str(prediction_objective)
        if self.prediction_objective not in {"autoregressive", "all_at_once"}:
            raise ValueError(
                "prediction_objective must be 'autoregressive' or 'all_at_once', "
                f"got {self.prediction_objective!r}"
            )
        self.fixed_length_mode = str(fixed_length_mode)
        if self.fixed_length_mode not in {"batch_max", "global_max"}:
            raise ValueError(
                "fixed_length_mode must be 'batch_max' or 'global_max', "
                f"got {self.fixed_length_mode!r}"
            )
        self.fixed_length_n_seq = (
            None if fixed_length_n_seq is None else int(fixed_length_n_seq)
        )
        if self.fixed_length_n_seq is not None and self.fixed_length_n_seq < 2:
            raise ValueError(
                f"fixed_length_n_seq must be >= 2, got {self.fixed_length_n_seq}"
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
        self.sample_max_attempts = int(sample_max_attempts)
        if self.sample_max_attempts < 1:
            raise ValueError(
                f"sample_max_attempts must be >= 1, got {self.sample_max_attempts}"
            )
        self.max_unify_solutions = int(max_unify_solutions)
        if self.max_unify_solutions < 1:
            raise ValueError(
                f"max_unify_solutions must be >= 1, got {self.max_unify_solutions}"
            )

        self._rng = np.random.default_rng(self.seed)

        self._epoch = 0
        self._data_source = None
        self._dataloader = None
        self._iterator = None
        self._batch_fn = None
        self._global_autoreg_seq_len: int | None = None

        self.ds_path = Path(ds_path) if ds_path is not None else None

        self._rule_bank: FOLRuleBank | None = None
        self._tokenizer: tokenize_layer_fol.FOLLayerTokenizer | None = None
        if self.mode == "online" or rule_bank_path is not None:
            if rule_bank_path is not None:
                self._rule_bank = load_fol_rule_bank(Path(rule_bank_path))
            else:
                self._rule_bank = build_random_fol_rule_bank(
                    n_layers=int(n_layers),
                    predicates_per_layer=int(predicates_per_layer),
                    rules_per_transition=int(rules_per_transition),
                    arity_max=int(arity_max),
                    vars_per_rule_max=int(vars_per_rule_max),
                    constants=tuple(str(tok) for tok in constants),
                    k_in_max=int(k_in_max),
                    k_out_max=int(k_out_max),
                    rng=self._rng,
                )
            self._tokenizer = tokenize_layer_fol.build_tokenizer_from_rule_bank(self._rule_bank)

        if self.mode == "offline":
            if self.ds_path is None:
                raise ValueError("ds_path is required when mode='offline'.")
            metadata_path = self.ds_path / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            self._tokenizer = tokenize_layer_fol.tokenizer_from_metadata(metadata)
            self.stats = self._stats_from_metadata(
                self.ds_path,
                self._distances,
            )
            self._batch_fn = self._make_batch_fn()

            self._data_source = self._build_data_source()
            self._dataloader = self._build_dataloader()
            self._iterator = iter(self._dataloader)
        else:
            self.stats = {}
            self._batch_fn = self._make_batch_fn()

        if (
            self.prediction_objective == "autoregressive"
            and self.fixed_length_mode == "global_max"
        ):
            self._global_autoreg_seq_len = self._resolve_global_autoreg_seq_len()

    def __next__(self):
        if self.mode == "offline":
            try:
                batch = next(self._iterator)
            except StopIteration:
                self._epoch += 1
                self._dataloader = self._build_dataloader()
                self._iterator = iter(self._dataloader)
                batch = next(self._iterator)
            return self._pad_autoreg_batch_to_global_len(batch)

        records = [self._sample_online_record() for _ in range(self.batch_size)]
        batch = self._batch_fn(records)
        return self._pad_autoreg_batch_to_global_len(batch)

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

    @classmethod
    def stats_from_metadata(cls, ds_path, distance_range) -> dict:
        distances = cls._normalize_distances(distance_range)
        return cls._stats_from_metadata(Path(ds_path), distances)

    @classmethod
    def _stats_from_metadata(
        cls,
        ds_path: Path,
        distances: Iterable[int],
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

        return {
            key: max(int(stats.get(key, 0)) for stats in stats_list)
            for key in cls.STATS_KEYS
        }

    def _collect_shards(self, distances: Iterable[int]) -> list[str]:
        if self.ds_path is None:
            raise RuntimeError("Offline shard collection requires dataset path.")

        shards: list[str] = []
        for distance in distances:
            distance_dir = self.ds_path / f"distance_{distance:03d}"
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

        operations = [
            _DecodeRecord(),
            transforms.Batch(
                batch_size=self.batch_size,
                drop_remainder=self.drop_remainder,
                batch_fn=self._batch_fn,
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
        sampled = None
        last_err = None
        for _ in range(3):
            try:
                sampled = sample_fol_problem(
                    bank=self._rule_bank,
                    distance=distance,
                    initial_ant_max=self.initial_ant_max,
                    rng=self._rng,
                    max_attempts=self.sample_max_attempts,
                    max_unify_solutions=self.max_unify_solutions,
                )
                break
            except RuntimeError as err:
                last_err = err

        if sampled is None:
            raise RuntimeError(
                f"Failed to sample online FOLLayerTask record for distance={distance} "
                f"after 3 retries with max_attempts={self.sample_max_attempts}."
            ) from last_err

        step_idx = int(self._rng.integers(0, len(sampled.step_rules)))
        src_layer = sampled.step_layers[step_idx]
        ants = sampled.step_ants[step_idx]
        rule = sampled.step_rules[step_idx]
        sequent = FOLSequent(ants=ants, cons=sampled.goal_atom)
        prompt, completion = self._tokenizer.tokenize_example(sequent, rule.statement_text)
        return {
            "distance": int(distance),
            "src_layer": int(src_layer),
            "prompt": np.asarray(prompt, dtype=np.int32),
            "completions": [np.asarray(completion, dtype=np.int32)],
        }

    @property
    def rule_bank(self) -> FOLRuleBank | None:
        return self._rule_bank

    @property
    def tokenizer(self) -> tokenize_layer_fol.FOLLayerTokenizer | None:
        return self._tokenizer

    def _make_batch_fn(self):
        if self.prediction_objective == "autoregressive":
            return _batch_records_autoreg

        if self._tokenizer is None:
            raise RuntimeError("All-at-once objective requires tokenizer.")
        return partial(
            _batch_records_all_at_once,
            eot_token_id=int(self._tokenizer.eot_token_id),
        )

    def _resolve_global_autoreg_seq_len(self) -> int:
        if self.fixed_length_n_seq is not None:
            return int(self.fixed_length_n_seq)

        if self.mode == "offline":
            if "max_seq" not in self.stats:
                raise RuntimeError("Offline global fixed length requires stats['max_seq'].")
            n_seq = int(self.stats["max_seq"])
        else:
            if self._rule_bank is None:
                raise RuntimeError("Online global fixed length requires a rule bank.")
            if self._tokenizer is None:
                raise RuntimeError("Online global fixed length requires a tokenizer.")

            max_rhs_atoms = max(
                len(rule.rhs)
                for rules in self._rule_bank.transitions.values()
                for rule in rules
            )
            max_prompt_facts = max(int(self.initial_ant_max), int(max_rhs_atoms))
            max_atom_len = 2 * int(self._rule_bank.arity_max) + 2
            max_prompt_len = (
                max_prompt_facts * max_atom_len
                + max(0, max_prompt_facts - 1)
                + 1
                + max_atom_len
                + 1
            )

            max_completion_len = 1
            for rules in self._rule_bank.transitions.values():
                for rule in rules:
                    max_completion_len = max(
                        max_completion_len,
                        len(self._tokenizer.encode_completion(rule.statement_text)),
                    )
            n_seq = int(max_prompt_len + max_completion_len - 1)

        if n_seq < 2:
            raise ValueError(f"Resolved global fixed length must be >= 2, got {n_seq}")
        return n_seq

    def _pad_autoreg_batch_to_global_len(self, batch):
        if self.prediction_objective != "autoregressive":
            return batch
        if self.fixed_length_mode != "global_max":
            return batch
        if self._global_autoreg_seq_len is None:
            raise RuntimeError("Global autoregressive seq length was not initialized.")

        xs, ys = batch
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        if xs.ndim != 2 or ys.ndim != 2:
            raise ValueError(
                f"Autoregressive batches must be 2D, got xs={xs.shape}, ys={ys.shape}"
            )
        if xs.shape[0] != ys.shape[0]:
            raise ValueError(
                f"Batch size mismatch for autoregressive batch: xs={xs.shape}, ys={ys.shape}"
            )

        n_seq = int(self._global_autoreg_seq_len)
        if xs.shape[1] > n_seq or ys.shape[1] > n_seq:
            raise ValueError(
                "Autoregressive batch sequence exceeds fixed global length: "
                f"xs={xs.shape}, ys={ys.shape}, fixed_length_n_seq={n_seq}"
            )

        if xs.shape[1] == n_seq and ys.shape[1] == n_seq:
            return xs, ys

        out_x = np.full((xs.shape[0], n_seq), 0, dtype=xs.dtype)
        out_y = np.full((ys.shape[0], n_seq), 0, dtype=ys.dtype)
        if xs.shape[1] > 0:
            out_x[:, : xs.shape[1]] = xs
        if ys.shape[1] > 0:
            out_y[:, : ys.shape[1]] = ys
        return out_x, out_y


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


def _pad_sequences(arrays: list[np.ndarray], *, pad_value: int = 0) -> np.ndarray:
    max_len = max(arr.shape[0] for arr in arrays)
    out = np.full((len(arrays), max_len), int(pad_value), dtype=np.int32)
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


def _batch_records_all_at_once(records, *, eot_token_id: int):
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
        if completion.shape[0] < 1:
            raise ValueError("Completion must contain at least one token.")
        if int(completion[-1]) != int(eot_token_id):
            raise ValueError(
                "Completion must terminate with EOT token for all-at-once objective."
            )

        xs.append(prompt.copy())
        ys.append(completion.copy())

    return _pad_sequences(xs), _pad_sequences(ys, pad_value=eot_token_id)


def completion_is_valid_for_layer_fol(
    *,
    rule_bank: FOLRuleBank,
    src_layer: int,
    completion_tokens: list[int] | np.ndarray,
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer | None = None,
) -> bool:
    result = match_rule_completion_fol(
        rule_bank=rule_bank,
        src_layer=src_layer,
        completion_tokens=completion_tokens,
        tokenizer=tokenizer,
    )
    return result.is_valid_rule


completion_is_valid_for_layer = completion_is_valid_for_layer_fol


class FOLLayerPredictionAdapter(Protocol):
    def predict_completion(
        self,
        *,
        model: Callable[[np.ndarray], Any],
        prompt_tokens: list[int] | np.ndarray,
        tokenizer: tokenize_layer_fol.FOLLayerTokenizer,
        temperature: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class FOLRuleMatchResult:
    src_layer: int
    decoded_statement: str | None
    expected_statement_text: str | None
    matched_rule: FOLLayerRule | None
    decode_error: bool
    unknown_rule_error: bool
    wrong_rule_error: bool
    is_valid_rule: bool
    is_correct: bool


@dataclass(frozen=True)
class FOLRuleMatchMetrics:
    n_examples: int
    n_correct: int
    n_decode_error: int
    n_unknown_rule_error: int
    n_wrong_rule_error: int
    accuracy: float
    decode_error_rate: float
    unknown_rule_error_rate: float
    wrong_rule_error_rate: float
    results: tuple[FOLRuleMatchResult, ...]


@dataclass(frozen=True)
class FOLLayerRolloutExample:
    distance: int
    start_layer: int
    goal_atom: str
    initial_ants: tuple[str, ...]
    max_steps: int
    oracle_rule_statements: tuple[str, ...] = ()


@dataclass(frozen=True)
class FOLLayerRolloutTraceStep:
    step_idx: int
    src_layer: int
    prompt_tokens: tuple[int, ...]
    completion_tokens: tuple[int, ...]
    decoded_statement: str | None
    matched_rule_statement: str | None
    decode_error: bool
    unknown_rule_error: bool
    inapplicable_rule_error: bool
    goal_reached: bool


@dataclass(frozen=True)
class FOLLayerRolloutResult:
    success: bool
    failure_reason: str | None
    n_steps: int
    goal_reached: bool
    final_layer: int
    final_facts: tuple[str, ...]
    steps: tuple[FOLLayerRolloutTraceStep, ...]
    example: FOLLayerRolloutExample


@dataclass(frozen=True)
class FOLLayerRolloutMetrics:
    n_examples: int
    n_success: int
    n_failure_decode_error: int
    n_failure_unknown_rule_error: int
    n_failure_inapplicable_rule_error: int
    n_failure_goal_not_reached: int
    success_rate: float
    decode_error_rate: float
    unknown_rule_error_rate: float
    inapplicable_rule_error_rate: float
    goal_not_reached_rate: float
    avg_steps: float
    results: tuple[FOLLayerRolloutResult, ...]


FAILURE_DECODE_ERROR = "decode_error"
FAILURE_UNKNOWN_RULE_ERROR = "unknown_rule_error"
FAILURE_INAPPLICABLE_RULE_ERROR = "inapplicable_rule_error"
FAILURE_GOAL_NOT_REACHED = "goal_not_reached"


def _is_variable(token: str) -> bool:
    return token.startswith("x")


def _unify_template_atom_with_ground(
    template: FOLAtom,
    ground: FOLAtom,
    subst: dict[str, str],
) -> dict[str, str] | None:
    if template.predicate != ground.predicate:
        return None
    if len(template.args) != len(ground.args):
        return None

    out = dict(subst)
    for templ_term, ground_term in zip(template.args, ground.args):
        if _is_variable(templ_term):
            bound = out.get(templ_term)
            if bound is None:
                out[templ_term] = ground_term
            elif bound != ground_term:
                return None
        elif templ_term != ground_term:
            return None
    return out


def _find_multiset_matches(
    *,
    templates: tuple[FOLAtom, ...],
    grounds: tuple[FOLAtom, ...],
    seed_subst: dict[str, str],
    max_solutions: int,
) -> list[dict[str, str]]:
    if len(templates) != len(grounds):
        return []

    solutions: list[dict[str, str]] = []
    used = [False] * len(grounds)

    def _search(idx: int, subst: dict[str, str]) -> None:
        if len(solutions) >= max_solutions:
            return
        if idx >= len(templates):
            solutions.append(dict(subst))
            return

        template = templates[idx]
        for ground_idx, ground in enumerate(grounds):
            if used[ground_idx]:
                continue
            maybe = _unify_template_atom_with_ground(template, ground, subst)
            if maybe is None:
                continue
            used[ground_idx] = True
            _search(idx + 1, maybe)
            used[ground_idx] = False
            if len(solutions) >= max_solutions:
                return

    _search(0, dict(seed_subst))
    return solutions


def _find_instantiation_for_rule(
    *,
    template: FOLLayerRule,
    lhs_ground: tuple[FOLAtom, ...],
    rhs_ground: tuple[FOLAtom, ...],
    max_solutions: int = 64,
) -> dict[str, str] | None:
    lhs_subs = _find_multiset_matches(
        templates=template.lhs,
        grounds=lhs_ground,
        seed_subst={},
        max_solutions=max_solutions,
    )
    if not lhs_subs:
        return None

    for lhs_sub in lhs_subs:
        rhs_subs = _find_multiset_matches(
            templates=template.rhs,
            grounds=rhs_ground,
            seed_subst=lhs_sub,
            max_solutions=1,
        )
        if rhs_subs:
            return rhs_subs[0]
    return None


def _match_instantiated_rule(
    *,
    rule_bank: FOLRuleBank,
    src_layer: int,
    lhs_ground: tuple[FOLAtom, ...],
    rhs_ground: tuple[FOLAtom, ...],
) -> FOLLayerRule | None:
    for rule in rule_bank.transition_rules(src_layer):
        subst = _find_instantiation_for_rule(
            template=rule,
            lhs_ground=lhs_ground,
            rhs_ground=rhs_ground,
        )
        if subst is not None:
            return rule.instantiate(subst)
    return None


def match_rule_completion_fol(
    *,
    rule_bank: FOLRuleBank,
    src_layer: int,
    completion_tokens: list[int] | np.ndarray,
    expected_statement_text: str | None = None,
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer | None = None,
) -> FOLRuleMatchResult:
    tokenizer = _resolve_fol_tokenizer(rule_bank=rule_bank, tokenizer=tokenizer)
    src_layer = int(src_layer)

    try:
        completion = [int(tok) for tok in np.asarray(completion_tokens, dtype=np.int32).tolist()]
    except (TypeError, ValueError):
        completion = []

    try:
        decoded_statement = tokenizer.decode_completion_text(completion)
        lhs_ground, rhs_ground = parse_clause_text(decoded_statement)
    except (ValueError, TypeError):
        return FOLRuleMatchResult(
            src_layer=src_layer,
            decoded_statement=None,
            expected_statement_text=expected_statement_text,
            matched_rule=None,
            decode_error=True,
            unknown_rule_error=False,
            wrong_rule_error=False,
            is_valid_rule=False,
            is_correct=False,
        )

    matched_rule = _match_instantiated_rule(
        rule_bank=rule_bank,
        src_layer=src_layer,
        lhs_ground=lhs_ground,
        rhs_ground=rhs_ground,
    )
    if matched_rule is None:
        return FOLRuleMatchResult(
            src_layer=src_layer,
            decoded_statement=decoded_statement,
            expected_statement_text=expected_statement_text,
            matched_rule=None,
            decode_error=False,
            unknown_rule_error=True,
            wrong_rule_error=False,
            is_valid_rule=False,
            is_correct=False,
        )

    wrong_rule_error = (
        expected_statement_text is not None
        and decoded_statement != str(expected_statement_text)
    )
    is_correct = not wrong_rule_error
    return FOLRuleMatchResult(
        src_layer=src_layer,
        decoded_statement=decoded_statement,
        expected_statement_text=expected_statement_text,
        matched_rule=matched_rule,
        decode_error=False,
        unknown_rule_error=False,
        wrong_rule_error=bool(wrong_rule_error),
        is_valid_rule=True,
        is_correct=is_correct,
    )


match_rule_completion = match_rule_completion_fol


def evaluate_rule_matches_fol(
    *,
    rule_bank: FOLRuleBank,
    src_layers: Iterable[int],
    completion_tokens: Iterable[list[int] | np.ndarray],
    expected_statement_texts: Iterable[str | None] | None = None,
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer | None = None,
) -> FOLRuleMatchMetrics:
    src_layers = [int(layer) for layer in src_layers]
    completion_tokens = list(completion_tokens)
    if len(src_layers) != len(completion_tokens):
        raise ValueError(
            f"src_layers and completion_tokens must have same length, got "
            f"{len(src_layers)} and {len(completion_tokens)}"
        )

    if expected_statement_texts is None:
        expected_statement_texts = [None] * len(src_layers)
    else:
        expected_statement_texts = list(expected_statement_texts)
        if len(expected_statement_texts) != len(src_layers):
            raise ValueError(
                "expected_statement_texts must match src_layers length, got "
                f"{len(expected_statement_texts)} and {len(src_layers)}"
            )

    results = tuple(
        match_rule_completion_fol(
            rule_bank=rule_bank,
            src_layer=src_layer,
            completion_tokens=completion,
            expected_statement_text=expected_statement,
            tokenizer=tokenizer,
        )
        for src_layer, completion, expected_statement in zip(
            src_layers, completion_tokens, expected_statement_texts
        )
    )

    n_examples = len(results)
    n_correct = sum(int(result.is_correct) for result in results)
    n_decode_error = sum(int(result.decode_error) for result in results)
    n_unknown_rule_error = sum(int(result.unknown_rule_error) for result in results)
    n_wrong_rule_error = sum(int(result.wrong_rule_error) for result in results)

    return FOLRuleMatchMetrics(
        n_examples=n_examples,
        n_correct=n_correct,
        n_decode_error=n_decode_error,
        n_unknown_rule_error=n_unknown_rule_error,
        n_wrong_rule_error=n_wrong_rule_error,
        accuracy=_safe_rate(n_correct, n_examples),
        decode_error_rate=_safe_rate(n_decode_error, n_examples),
        unknown_rule_error_rate=_safe_rate(n_unknown_rule_error, n_examples),
        wrong_rule_error_rate=_safe_rate(n_wrong_rule_error, n_examples),
        results=results,
    )


def sample_rollout_examples_fol(
    *,
    rule_bank: FOLRuleBank,
    distance: int,
    n_examples: int,
    initial_ant_max: int,
    max_steps: int | None = None,
    max_unify_solutions: int = 128,
    rng: np.random.Generator | None = None,
) -> list[FOLLayerRolloutExample]:
    if n_examples < 1:
        raise ValueError(f"n_examples must be >= 1, got {n_examples}")
    if rng is None:
        rng = np.random.default_rng()

    out: list[FOLLayerRolloutExample] = []
    for _ in range(int(n_examples)):
        sampled = sample_fol_problem(
            bank=rule_bank,
            distance=int(distance),
            initial_ant_max=int(initial_ant_max),
            rng=rng,
            max_unify_solutions=int(max_unify_solutions),
        )
        if not sampled.step_ants:
            raise RuntimeError("Sampled problem contained no steps.")
        out.append(
            FOLLayerRolloutExample(
                distance=int(sampled.distance),
                start_layer=int(sampled.start_layer),
                goal_atom=str(sampled.goal_atom.text),
                initial_ants=tuple(atom.text for atom in sampled.step_ants[0]),
                max_steps=int(sampled.distance if max_steps is None else max_steps),
                oracle_rule_statements=tuple(
                    str(rule.statement_text) for rule in sampled.step_rules
                ),
            )
        )
    return out


def run_layer_rollout_fol(
    *,
    rule_bank: FOLRuleBank,
    example: FOLLayerRolloutExample,
    model: Callable[[np.ndarray], Any],
    adapter: FOLLayerPredictionAdapter,
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer | None = None,
    temperature: float = 0.0,
    rng: np.random.Generator | None = None,
) -> FOLLayerRolloutResult:
    tokenizer = _resolve_fol_tokenizer(rule_bank=rule_bank, tokenizer=tokenizer)
    if rng is None:
        rng = np.random.default_rng()

    facts = {parse_atom_text(atom_text) for atom_text in example.initial_ants}
    goal = parse_atom_text(example.goal_atom)
    traces: list[FOLLayerRolloutTraceStep] = []

    for step_idx in range(int(example.max_steps)):
        src_layer = int(example.start_layer) + step_idx
        prompt = tokenizer.tokenize_prompt(
            FOLSequent(
                ants=_sorted_fol_atoms(facts),
                cons=goal,
            )
        )
        completion = adapter.predict_completion(
            model=model,
            prompt_tokens=prompt,
            tokenizer=tokenizer,
            temperature=temperature,
            rng=rng,
        )

        matched = match_rule_completion_fol(
            rule_bank=rule_bank,
            src_layer=src_layer,
            completion_tokens=completion,
            tokenizer=tokenizer,
        )

        if matched.decode_error:
            traces.append(
                FOLLayerRolloutTraceStep(
                    step_idx=step_idx,
                    src_layer=src_layer,
                    prompt_tokens=tuple(int(tok) for tok in prompt),
                    completion_tokens=tuple(int(tok) for tok in completion),
                    decoded_statement=None,
                    matched_rule_statement=None,
                    decode_error=True,
                    unknown_rule_error=False,
                    inapplicable_rule_error=False,
                    goal_reached=False,
                )
            )
            return FOLLayerRolloutResult(
                success=False,
                failure_reason=FAILURE_DECODE_ERROR,
                n_steps=len(traces),
                goal_reached=False,
                final_layer=src_layer,
                final_facts=tuple(atom.text for atom in _sorted_fol_atoms(facts)),
                steps=tuple(traces),
                example=example,
            )

        if matched.unknown_rule_error or matched.matched_rule is None:
            traces.append(
                FOLLayerRolloutTraceStep(
                    step_idx=step_idx,
                    src_layer=src_layer,
                    prompt_tokens=tuple(int(tok) for tok in prompt),
                    completion_tokens=tuple(int(tok) for tok in completion),
                    decoded_statement=matched.decoded_statement,
                    matched_rule_statement=None,
                    decode_error=False,
                    unknown_rule_error=True,
                    inapplicable_rule_error=False,
                    goal_reached=False,
                )
            )
            return FOLLayerRolloutResult(
                success=False,
                failure_reason=FAILURE_UNKNOWN_RULE_ERROR,
                n_steps=len(traces),
                goal_reached=False,
                final_layer=src_layer,
                final_facts=tuple(atom.text for atom in _sorted_fol_atoms(facts)),
                steps=tuple(traces),
                example=example,
            )

        rule = matched.matched_rule
        if not set(rule.lhs).issubset(facts):
            traces.append(
                FOLLayerRolloutTraceStep(
                    step_idx=step_idx,
                    src_layer=src_layer,
                    prompt_tokens=tuple(int(tok) for tok in prompt),
                    completion_tokens=tuple(int(tok) for tok in completion),
                    decoded_statement=matched.decoded_statement,
                    matched_rule_statement=rule.statement_text,
                    decode_error=False,
                    unknown_rule_error=False,
                    inapplicable_rule_error=True,
                    goal_reached=False,
                )
            )
            return FOLLayerRolloutResult(
                success=False,
                failure_reason=FAILURE_INAPPLICABLE_RULE_ERROR,
                n_steps=len(traces),
                goal_reached=False,
                final_layer=src_layer,
                final_facts=tuple(atom.text for atom in _sorted_fol_atoms(facts)),
                steps=tuple(traces),
                example=example,
            )

        facts = set(rule.rhs)
        goal_reached = goal in facts
        traces.append(
            FOLLayerRolloutTraceStep(
                step_idx=step_idx,
                src_layer=src_layer,
                prompt_tokens=tuple(int(tok) for tok in prompt),
                completion_tokens=tuple(int(tok) for tok in completion),
                decoded_statement=matched.decoded_statement,
                matched_rule_statement=rule.statement_text,
                decode_error=False,
                unknown_rule_error=False,
                inapplicable_rule_error=False,
                goal_reached=goal_reached,
            )
        )

        if goal_reached:
            return FOLLayerRolloutResult(
                success=True,
                failure_reason=None,
                n_steps=len(traces),
                goal_reached=True,
                final_layer=int(rule.dst_layer),
                final_facts=tuple(atom.text for atom in _sorted_fol_atoms(facts)),
                steps=tuple(traces),
                example=example,
            )

    return FOLLayerRolloutResult(
        success=False,
        failure_reason=FAILURE_GOAL_NOT_REACHED,
        n_steps=len(traces),
        goal_reached=False,
        final_layer=int(example.start_layer) + len(traces),
        final_facts=tuple(atom.text for atom in _sorted_fol_atoms(facts)),
        steps=tuple(traces),
        example=example,
    )


def evaluate_layer_rollouts_fol(
    *,
    rule_bank: FOLRuleBank,
    examples: Iterable[FOLLayerRolloutExample],
    model: Callable[[np.ndarray], Any],
    adapter: FOLLayerPredictionAdapter,
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer | None = None,
    temperature: float = 0.0,
    rng: np.random.Generator | None = None,
) -> FOLLayerRolloutMetrics:
    if rng is None:
        rng = np.random.default_rng()

    results = tuple(
        run_layer_rollout_fol(
            rule_bank=rule_bank,
            example=example,
            model=model,
            adapter=adapter,
            tokenizer=tokenizer,
            temperature=temperature,
            rng=rng,
        )
        for example in examples
    )

    n_examples = len(results)
    n_success = sum(int(result.success) for result in results)
    n_failure_decode_error = sum(
        int(result.failure_reason == FAILURE_DECODE_ERROR)
        for result in results
    )
    n_failure_unknown_rule_error = sum(
        int(result.failure_reason == FAILURE_UNKNOWN_RULE_ERROR)
        for result in results
    )
    n_failure_inapplicable_rule_error = sum(
        int(result.failure_reason == FAILURE_INAPPLICABLE_RULE_ERROR)
        for result in results
    )
    n_failure_goal_not_reached = sum(
        int(result.failure_reason == FAILURE_GOAL_NOT_REACHED)
        for result in results
    )
    avg_steps = float(np.mean([result.n_steps for result in results])) if results else 0.0

    return FOLLayerRolloutMetrics(
        n_examples=n_examples,
        n_success=n_success,
        n_failure_decode_error=n_failure_decode_error,
        n_failure_unknown_rule_error=n_failure_unknown_rule_error,
        n_failure_inapplicable_rule_error=n_failure_inapplicable_rule_error,
        n_failure_goal_not_reached=n_failure_goal_not_reached,
        success_rate=_safe_rate(n_success, n_examples),
        decode_error_rate=_safe_rate(n_failure_decode_error, n_examples),
        unknown_rule_error_rate=_safe_rate(n_failure_unknown_rule_error, n_examples),
        inapplicable_rule_error_rate=_safe_rate(n_failure_inapplicable_rule_error, n_examples),
        goal_not_reached_rate=_safe_rate(n_failure_goal_not_reached, n_examples),
        avg_steps=avg_steps,
        results=results,
    )


def _resolve_fol_tokenizer(
    *,
    rule_bank: FOLRuleBank,
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer | None,
) -> tokenize_layer_fol.FOLLayerTokenizer:
    if tokenizer is not None:
        return tokenizer
    return tokenize_layer_fol.build_tokenizer_from_rule_bank(rule_bank)


def _safe_rate(num: int, den: int) -> float:
    if den <= 0:
        return 0.0
    return float(num) / float(den)


def _sorted_fol_atoms(atoms: Iterable[FOLAtom]) -> tuple[FOLAtom, ...]:
    return tuple(sorted((atom for atom in atoms), key=lambda atom: atom.text))


evaluate_rule_matches = evaluate_rule_matches_fol
sample_rollout_examples = sample_rollout_examples_fol
run_layer_rollout = run_layer_rollout_fol
evaluate_layer_rollouts = evaluate_layer_rollouts_fol
