"""Task loader and sampler façade for layered first-order tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from task.layer_gen.util import online_prefetch as online_prefetch_util
from task.layer_gen.util import tokenize_layer_fol
from task.layer_gen.util.fol_rule_bank import FOLDepth3ICLSplitBundle, FOLRuleBank
from .common import compute_fol_dims
from .task_batching import (
    ceil_pow2,
    coerce_autoreg_batch,
    make_batch_fn,
    pad_autoreg_batch_to_length,
)
from .task_offline import (
    build_data_source,
    build_dataloader,
    load_tokenizer_and_stats,
    normalize_distances,
    stats_from_metadata,
)
from .task_prefetch import (
    PrefetchSetup,
    _FOLOnlineSamplerServerClient,
    init_executor_prefetch,
    init_server_prefetch,
)
from .task_sampling import (
    _init_fol_online_fresh_worker,
    _init_fol_online_worker,
    _sample_fol_online_fresh_worker_records,
    _sample_fol_online_worker_records,
)
from .task_shared import FOLTaskSplitStrategy
from .task_split_depth3_fresh import Depth3FreshICLSplitStrategy
from .task_split_depth3_transfer import Depth3ICLTransferSplitStrategy
from .task_split_none import NoSplitStrategy


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
        completion_format="single",
        fixed_length_mode="batch_max",
        fixed_length_n_seq=None,
        task_split="none",
        split_role="train",
        split_rule_bundle_path=None,
        rule_bank_path=None,
        n_layers=16,
        predicates_per_layer=8,
        rules_per_transition=32,
        arity_max=3,
        vars_per_rule_max=4,
        constants=("a", "b", "c", "d"),
        k_in_min=1,
        k_in_max=3,
        k_out_min=1,
        k_out_max=3,
        initial_ant_max=3,
        max_n_demos=0,
        min_n_demos=None,
        sample_max_attempts=4096,
        max_unify_solutions=128,
        online_prefetch=True,
        online_prefetch_backend="server",
        online_prefetch_workers=None,
        online_prefetch_buffer_size=None,
        fresh_icl_n_predicates=None,
        arity_min=1,
        predicate_name_len=1,
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
        self.completion_format = str(completion_format)
        if self.completion_format not in {"single", "full"}:
            raise ValueError(
                "completion_format must be 'single' or 'full', "
                f"got {self.completion_format!r}"
            )
        self.fixed_length_mode = str(fixed_length_mode)
        if self.fixed_length_mode not in {"batch_max", "global_max", "next_pow2"}:
            raise ValueError(
                "fixed_length_mode must be 'batch_max', 'global_max', or 'next_pow2', "
                f"got {self.fixed_length_mode!r}"
            )
        self.fixed_length_n_seq = (
            None if fixed_length_n_seq is None else int(fixed_length_n_seq)
        )
        if self.fixed_length_n_seq is not None and self.fixed_length_n_seq < 2:
            raise ValueError(
                f"fixed_length_n_seq must be >= 2, got {self.fixed_length_n_seq}"
            )
        self.task_split = str(task_split)
        if self.task_split not in {"none", "depth3_icl_transfer", "depth3_fresh_icl"}:
            raise ValueError(
                "task_split must be 'none', 'depth3_icl_transfer', or 'depth3_fresh_icl', "
                f"got {self.task_split!r}"
            )
        self.split_role = str(split_role)
        if self.split_role not in {"train", "eval"}:
            raise ValueError(
                "split_role must be 'train' or 'eval', "
                f"got {self.split_role!r}"
            )

        self.split_rule_bundle_path = (
            None
            if split_rule_bundle_path is None
            else Path(split_rule_bundle_path)
        )
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
        self.rules_per_transition = int(rules_per_transition)
        self._k_in_min = int(k_in_min)
        self._k_in_max = int(k_in_max)
        self._k_out_min = int(k_out_min)
        self._k_out_max = int(k_out_max)
        self._arity_min = int(arity_min)
        self._predicate_name_len = int(predicate_name_len)
        self.max_n_demos = int(max_n_demos)
        if self.max_n_demos < 0:
            raise ValueError(f"max_n_demos must be >= 0, got {self.max_n_demos}")
        if min_n_demos is None:
            self.min_n_demos = 0 if self.max_n_demos == 0 else 1
        else:
            self.min_n_demos = int(min_n_demos)
        if self.min_n_demos < 0:
            raise ValueError(f"min_n_demos must be >= 0, got {self.min_n_demos}")
        if self.min_n_demos > self.max_n_demos:
            raise ValueError(
                "min_n_demos must be <= max_n_demos, "
                f"got min_n_demos={self.min_n_demos}, max_n_demos={self.max_n_demos}"
            )
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
        self._online_prefetch_requested = bool(online_prefetch)
        self._online_prefetch_backend_requested = str(online_prefetch_backend)
        self._online_prefetch_workers_requested = (
            None if online_prefetch_workers is None else int(online_prefetch_workers)
        )
        self._online_prefetch_buffer_size_requested = (
            None
            if online_prefetch_buffer_size is None
            else int(online_prefetch_buffer_size)
        )

        self._rng = np.random.default_rng(self.seed)
        self._epoch = 0
        self._data_source = None
        self._dataloader = None
        self._iterator = None
        self._batch_fn = None
        self._global_autoreg_seq_len: int | None = None
        self._online_executor = None
        self._online_server_client: _FOLOnlineSamplerServerClient | None = None
        self._online_prefetch_buffer = None
        self._online_prefetch_enabled = False
        self._online_prefetch_backend_resolved = "sync"
        self._online_prefetch_workers_resolved = 1
        self._online_prefetch_buffer_size_resolved = max(1, self.batch_size)

        self.ds_path = Path(ds_path) if ds_path is not None else None
        self._rule_bank: FOLRuleBank | None = None
        self._tokenizer: tokenize_layer_fol.FOLLayerTokenizer | None = None
        self._split_bundle: FOLDepth3ICLSplitBundle | None = None
        self._online_forced_step_idx: int | None = None
        self._base_bank: FOLRuleBank | None = None
        self._fresh_icl_n_predicates: int | None = None
        self._strategy: FOLTaskSplitStrategy | None = None

        resolved_rule_bank_path = rule_bank_path
        if self.task_split == "none" and resolved_rule_bank_path is None and self.ds_path is not None:
            candidate = self.ds_path / "rule_bank.json"
            if candidate.exists():
                resolved_rule_bank_path = candidate
            else:
                print(f"warn: rule_bank.json not found in ds_path={self.ds_path}")

        if (
            self.task_split != "none"
            or self.mode == "online"
            or resolved_rule_bank_path is not None
        ):
            self._strategy = self._build_strategy(
                rule_bank_path=resolved_rule_bank_path,
                split_rule_bundle_path=self.split_rule_bundle_path,
                n_layers=int(n_layers),
                predicates_per_layer=int(predicates_per_layer),
                rules_per_transition=int(rules_per_transition),
                arity_max=int(arity_max),
                arity_min=int(arity_min),
                vars_per_rule_max=int(vars_per_rule_max),
                constants=tuple(str(tok) for tok in constants),
                k_in_min=int(k_in_min),
                k_in_max=int(k_in_max),
                k_out_min=int(k_out_min),
                k_out_max=int(k_out_max),
                fresh_icl_n_predicates=fresh_icl_n_predicates,
            )
            self._adopt_strategy_state(self._strategy)

        if self.mode == "offline":
            if self.ds_path is None:
                raise ValueError("ds_path is required when mode='offline'.")
            self._tokenizer, self.stats = load_tokenizer_and_stats(
                ds_path=self.ds_path,
                distances=self._distances,
                expected_completion_format=self.completion_format,
                stats_keys=self.STATS_KEYS,
            )
            self._batch_fn = self._make_batch_fn()
            self._data_source = self._build_data_source()
            self._dataloader = self._build_dataloader()
            self._iterator = iter(self._dataloader)
        else:
            self.stats = {}
            self._batch_fn = self._make_batch_fn()
            self._init_online_prefetch()

        if (
            self.prediction_objective == "autoregressive"
            and self.fixed_length_mode == "global_max"
        ):
            self._global_autoreg_seq_len = self._resolve_global_autoreg_seq_len()

    def _build_strategy(self, **kwargs) -> FOLTaskSplitStrategy:
        distances = tuple(int(distance) for distance in self._distances)
        if self.task_split == "none":
            return NoSplitStrategy.build(
                seed=int(self.seed),
                distances=distances,
                rule_bank_path=kwargs["rule_bank_path"],
                n_layers=kwargs["n_layers"],
                predicates_per_layer=kwargs["predicates_per_layer"],
                rules_per_transition=kwargs["rules_per_transition"],
                arity_max=kwargs["arity_max"],
                arity_min=kwargs["arity_min"],
                vars_per_rule_max=kwargs["vars_per_rule_max"],
                constants=kwargs["constants"],
                k_in_min=kwargs["k_in_min"],
                k_in_max=kwargs["k_in_max"],
                k_out_min=kwargs["k_out_min"],
                k_out_max=kwargs["k_out_max"],
                initial_ant_max=int(self.initial_ant_max),
                sample_max_attempts=int(self.sample_max_attempts),
                max_unify_solutions=int(self.max_unify_solutions),
                max_n_demos=int(self.max_n_demos),
                min_n_demos=int(self.min_n_demos),
                completion_format=str(self.completion_format),
                rng=self._rng,
            )
        if self.task_split == "depth3_icl_transfer":
            return Depth3ICLTransferSplitStrategy.build(
                mode=self.mode,
                split_role=self.split_role,
                split_rule_bundle_path=kwargs["split_rule_bundle_path"],
                rule_bank_path=kwargs["rule_bank_path"],
                distances=distances,
                seed=int(self.seed),
                initial_ant_max=int(self.initial_ant_max),
                sample_max_attempts=int(self.sample_max_attempts),
                max_unify_solutions=int(self.max_unify_solutions),
                max_n_demos=int(self.max_n_demos),
                min_n_demos=int(self.min_n_demos),
                completion_format=str(self.completion_format),
            )
        return Depth3FreshICLSplitStrategy.build(
            mode=self.mode,
            split_role=self.split_role,
            distances=distances,
            seed=int(self.seed),
            predicates_per_layer=kwargs["predicates_per_layer"],
            rules_per_transition=kwargs["rules_per_transition"],
            arity_max=kwargs["arity_max"],
            arity_min=kwargs["arity_min"],
            vars_per_rule_max=kwargs["vars_per_rule_max"],
            constants=kwargs["constants"],
            k_in_min=kwargs["k_in_min"],
            k_in_max=kwargs["k_in_max"],
            k_out_min=kwargs["k_out_min"],
            k_out_max=kwargs["k_out_max"],
            initial_ant_max=int(self.initial_ant_max),
            sample_max_attempts=int(self.sample_max_attempts),
            max_unify_solutions=int(self.max_unify_solutions),
            max_n_demos=int(self.max_n_demos),
            min_n_demos=int(self.min_n_demos),
            completion_format=str(self.completion_format),
            fresh_icl_n_predicates=kwargs["fresh_icl_n_predicates"],
            predicate_name_len=int(self._predicate_name_len),
            rule_bank_path=kwargs["rule_bank_path"],
            split_rule_bundle_path=kwargs["split_rule_bundle_path"],
            rng=self._rng,
        )

    def _adopt_strategy_state(self, strategy: FOLTaskSplitStrategy) -> None:
        self._rule_bank = strategy.rule_bank
        self._tokenizer = strategy.tokenizer
        self._split_bundle = strategy.split_bundle
        self._online_forced_step_idx = strategy.online_forced_step_idx
        self._base_bank = strategy.base_bank
        self._fresh_icl_n_predicates = strategy.fresh_icl_n_predicates

    def __next__(self):
        if self.mode == "offline":
            try:
                batch = next(self._iterator)
            except StopIteration:
                self._epoch += 1
                self._dataloader = self._build_dataloader()
                self._iterator = iter(self._dataloader)
                batch = next(self._iterator)
            return self._apply_autoreg_fixed_length(batch)

        if self._online_server_client is not None:
            records = self._online_server_client.take(self.batch_size)
        elif self._online_prefetch_buffer is None:
            records = [self._sample_online_record() for _ in range(self.batch_size)]
        else:
            records = self._online_prefetch_buffer.take(self.batch_size)
        batch = self._batch_fn(records)
        return self._apply_autoreg_fixed_length(batch)

    def __iter__(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _normalize_distances(distance_range) -> list[int]:
        return normalize_distances(distance_range)

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
        return stats_from_metadata(
            ds_path=ds_path,
            distances=distances,
            stats_keys=cls.STATS_KEYS,
        )

    def _build_data_source(self):
        if self.ds_path is None:
            raise RuntimeError("Offline shard collection requires dataset path.")
        return build_data_source(
            ds_path=self.ds_path,
            distances=self._distances,
            reader_options=self.reader_options,
        )

    def _build_dataloader(self):
        return build_dataloader(
            data_source=self._data_source,
            batch_size=self.batch_size,
            drop_remainder=self.drop_remainder,
            batch_fn=self._batch_fn,
            shuffle=self.shuffle,
            seed=self.seed,
            epoch=self._epoch,
            worker_count=self.worker_count,
        )

    def _init_online_prefetch(self) -> None:
        if self._strategy is None:
            raise RuntimeError("Online mode requires an initialized split strategy.")

        requested_backend = str(self._online_prefetch_backend_requested)
        if requested_backend == "server":
            server_config = self._strategy.make_server_config(
                workers=1,
                buffer_size=max(1, self.batch_size),
                batch_size=int(self.batch_size),
            )
            if server_config is not None:
                enabled, _, workers, buffer_size = (
                    online_prefetch_util.resolve_online_prefetch_config(
                        enable=self._online_prefetch_requested,
                        backend="process",
                        workers=self._online_prefetch_workers_requested,
                        buffer_size=self._online_prefetch_buffer_size_requested,
                        batch_size=self.batch_size,
                    )
                )
                self._online_prefetch_workers_resolved = int(workers)
                self._online_prefetch_buffer_size_resolved = int(buffer_size)
                self._online_prefetch_backend_resolved = "server" if enabled else "sync"
                self._online_prefetch_enabled = bool(enabled)
                if not enabled:
                    return

                try:
                    server_config = self._strategy.make_server_config(
                        workers=int(workers),
                        buffer_size=int(buffer_size),
                        batch_size=int(self.batch_size),
                    )
                    if server_config is None:
                        raise RuntimeError("Server config unexpectedly unavailable.")
                    setup = init_server_prefetch(
                        server_config=server_config,
                        repo_root=Path(__file__).resolve().parents[1],
                    )
                    self._apply_prefetch_setup(setup)
                    return
                except Exception:
                    setup = init_executor_prefetch(
                        worker_spec=self._strategy.make_worker_spec(),
                        backend="thread",
                        workers=int(workers),
                        buffer_size=int(buffer_size),
                        batch_size=int(self.batch_size),
                    )
                    self._apply_prefetch_setup(setup)
                    return

            requested_backend = "process"

        enabled, backend, workers, buffer_size = (
            online_prefetch_util.resolve_online_prefetch_config(
                enable=self._online_prefetch_requested,
                backend=requested_backend,
                workers=self._online_prefetch_workers_requested,
                buffer_size=self._online_prefetch_buffer_size_requested,
                batch_size=self.batch_size,
            )
        )
        self._online_prefetch_workers_resolved = int(workers)
        self._online_prefetch_buffer_size_resolved = int(buffer_size)
        self._online_prefetch_backend_resolved = str(backend)
        self._online_prefetch_enabled = bool(enabled)
        if not enabled:
            return

        setup = init_executor_prefetch(
            worker_spec=self._strategy.make_worker_spec(),
            backend=str(backend),
            workers=int(workers),
            buffer_size=int(buffer_size),
            batch_size=int(self.batch_size),
        )
        self._apply_prefetch_setup(setup)

    def _apply_prefetch_setup(self, setup: PrefetchSetup) -> None:
        self._online_executor = setup.executor
        self._online_prefetch_buffer = setup.buffer
        self._online_server_client = setup.server_client
        self._online_prefetch_enabled = bool(setup.enabled)
        self._online_prefetch_backend_resolved = str(setup.backend_resolved)

    @property
    def online_prefetch_enabled(self) -> bool:
        return bool(self._online_prefetch_enabled)

    @property
    def online_prefetch_backend_resolved(self) -> str:
        return str(self._online_prefetch_backend_resolved)

    @property
    def online_prefetch_workers_resolved(self) -> int:
        return int(self._online_prefetch_workers_resolved)

    @property
    def online_prefetch_buffer_size_resolved(self) -> int:
        return int(self._online_prefetch_buffer_size_resolved)

    def close(self) -> None:
        if self._online_server_client is not None:
            self._online_server_client.close()
            self._online_server_client = None

        if self._online_prefetch_buffer is not None:
            self._online_prefetch_buffer.close()
            self._online_prefetch_buffer = None
            self._online_executor = None
        elif self._online_executor is not None:
            self._online_executor.shutdown(wait=True, cancel_futures=True)
            self._online_executor = None
        self._online_prefetch_enabled = False

    def _sample_online_record(self) -> dict:
        if self._strategy is None:
            raise RuntimeError("Online mode requires a split strategy.")
        return self._strategy.sample_record(rng=self._rng)

    @property
    def rule_bank(self) -> FOLRuleBank | None:
        return self._rule_bank

    @property
    def tokenizer(self) -> tokenize_layer_fol.FOLLayerTokenizer | None:
        return self._tokenizer

    def _make_batch_fn(self):
        return make_batch_fn(
            prediction_objective=self.prediction_objective,
            tokenizer=self._tokenizer,
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

            dims = compute_fol_dims(
                rule_banks=[self._rule_bank],
                tokenizer=self._tokenizer,
                initial_ant_max=self.initial_ant_max,
                max_n_demos=self.max_n_demos,
                completion_format=self.completion_format,
                completion_steps_max=max(self._distances),
            )
            n_seq = int(dims["n_seq_ar"])

        if n_seq < 2:
            raise ValueError(f"Resolved global fixed length must be >= 2, got {n_seq}")
        return n_seq

    def _apply_autoreg_fixed_length(self, batch):
        if self.prediction_objective != "autoregressive":
            return batch
        if self.fixed_length_mode == "batch_max":
            return batch
        if self.fixed_length_mode == "global_max":
            if self._global_autoreg_seq_len is None:
                raise RuntimeError("Global autoregressive seq length was not initialized.")
            return pad_autoreg_batch_to_length(
                batch,
                n_seq=int(self._global_autoreg_seq_len),
            )
        if self.fixed_length_mode == "next_pow2":
            xs, _ = coerce_autoreg_batch(batch)
            n_seq = max(2, int(ceil_pow2(xs.shape[1])))
            if self.fixed_length_n_seq is not None:
                cap = int(self.fixed_length_n_seq)
                if n_seq > cap:
                    raise ValueError(
                        "Autoregressive next_pow2 sequence exceeds fixed_length_n_seq cap: "
                        f"batch_n_seq={xs.shape[1]}, bucket_n_seq={n_seq}, fixed_length_n_seq={cap}"
                    )
            return pad_autoreg_batch_to_length(batch, n_seq=n_seq)
        raise ValueError(f"Unsupported fixed_length_mode={self.fixed_length_mode!r}")


__all__ = [
    "FOLLayerTask",
    "_init_fol_online_worker",
    "_sample_fol_online_worker_records",
    "_init_fol_online_fresh_worker",
    "_sample_fol_online_fresh_worker_records",
]
