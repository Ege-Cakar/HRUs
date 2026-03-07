"""Generate ArrayRecord datasets for layered first-order tasks."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import os
from pathlib import Path
import pickle
import sys
import time
import traceback

import numpy as np
from tqdm import tqdm

from array_record.python import array_record_module

if __package__ in (None, ""):
    from util.fol_completion import sampled_completion_texts  # type: ignore
    from util.fol_rule_bank import (  # type: ignore
        FOLRuleBank,
        FOLSequent,
        build_random_fol_rule_bank,
        load_fol_rule_bank,
        sample_fol_problem,
        save_fol_rule_bank,
    )
    from util import online_prefetch as online_prefetch_util  # type: ignore
    from util import tokenize_layer_fol  # type: ignore
else:
    from .util import online_prefetch as online_prefetch_util
    from .util.fol_completion import sampled_completion_texts
    from .util.fol_rule_bank import (
        FOLRuleBank,
        FOLSequent,
        build_random_fol_rule_bank,
        load_fol_rule_bank,
        sample_fol_problem,
        save_fol_rule_bank,
    )
    from .util import tokenize_layer_fol


@dataclass
class _AutoregStats:
    max_token: int = 0
    max_seq: int = 0
    max_prompt_seq: int = 0
    max_completion_seq: int = 0


@dataclass(frozen=True)
class _DistanceResult:
    distance: int
    examples: int
    records: int
    shards: int
    stats: _AutoregStats


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


def _default_seed() -> int:
    return int(np.random.randint(0, np.iinfo(np.int32).max))


def _parse_constants(raw: str) -> tuple[str, ...]:
    constants = tuple(tok.strip() for tok in str(raw).split(",") if tok.strip())
    if not constants:
        raise ValueError("--constants must contain at least one symbol")
    return constants


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate layered first-order datasets.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--server-mode",
        action="store_true",
        help="Run as an online sampling server over framed stdio IPC.",
    )

    parser.add_argument("--n-layers", type=int, default=16)
    parser.add_argument("--predicates-per-layer", type=int, default=8)
    parser.add_argument("--rules-per-transition", type=int, default=32)
    parser.add_argument("--k-in-max", type=int, default=3)
    parser.add_argument("--k-out-max", type=int, default=3)
    parser.add_argument("--arity-max", type=int, default=3)
    parser.add_argument("--vars-per-rule-max", type=int, default=4)
    parser.add_argument("--constants", type=str, default="a,b,c,d")
    parser.add_argument("--initial-ant-max", type=int, default=3)
    parser.add_argument("--sample-max-attempts", type=int, default=4096)
    parser.add_argument("--max-unify-solutions", type=int, default=128)
    parser.add_argument("--completion-format", type=str, default="single")

    parser.add_argument("--min-distance", type=int, default=1)
    parser.add_argument("--max-distance", type=int, default=4)
    parser.add_argument("--examples-per-distance", type=int, default=10_000)
    parser.add_argument("--examples-per-shard", type=int, default=50_000)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 4))
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


def _tokenize_sampled_completion(
    *,
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer,
    sequent: FOLSequent,
    sampled,
    step_idx: int,
    completion_format: str,
) -> tuple[list[int], list[int], list[str]]:
    prompt = tokenizer.tokenize_prompt(sequent)
    statement_texts = sampled_completion_texts(
        sampled=sampled,
        step_idx=int(step_idx),
        completion_format=completion_format,
    )
    completion = tokenizer.encode_completion_texts(statement_texts)
    return prompt, completion, statement_texts


def _build_or_load_rule_bank(args: argparse.Namespace, rng: np.random.Generator) -> FOLRuleBank:
    if args.rule_bank_path is not None:
        return load_fol_rule_bank(args.rule_bank_path)

    constants = _parse_constants(args.constants)
    return build_random_fol_rule_bank(
        n_layers=int(args.n_layers),
        predicates_per_layer=int(args.predicates_per_layer),
        rules_per_transition=int(args.rules_per_transition),
        arity_max=int(args.arity_max),
        vars_per_rule_max=int(args.vars_per_rule_max),
        k_in_max=int(args.k_in_max),
        k_out_max=int(args.k_out_max),
        constants=constants,
        rng=rng,
    )


def _distance_seed(base_seed: int, distance: int) -> int:
    return int(base_seed + distance * 1_000_003)


def _generate_distance_data(
    *,
    out_dir: str,
    distance: int,
    examples_per_distance: int,
    examples_per_shard: int,
    writer_options: str,
    initial_ant_max: int,
    sample_max_attempts: int,
    max_unify_solutions: int,
    seed: int,
    bank_payload: dict,
    completion_format: str,
) -> _DistanceResult:
    rng = np.random.default_rng(seed)
    bank = FOLRuleBank.from_dict(bank_payload)
    tokenizer = tokenize_layer_fol.build_tokenizer_from_rule_bank(bank)
    writer = ArrayRecordShardWriter(
        out_dir=Path(out_dir) / f"distance_{distance:03d}",
        examples_per_shard=examples_per_shard,
        writer_options=writer_options,
    )
    stats = _AutoregStats()
    examples = 0

    try:
        for _ in range(examples_per_distance):
            sampled = sample_fol_problem(
                bank=bank,
                distance=int(distance),
                initial_ant_max=int(initial_ant_max),
                rng=rng,
                max_attempts=int(sample_max_attempts),
                max_unify_solutions=int(max_unify_solutions),
            )

            for step_idx, (src_layer, ants, instantiated_rule) in enumerate(
                zip(sampled.step_layers, sampled.step_ants, sampled.step_rules)
            ):
                sequent = FOLSequent(ants=ants, cons=sampled.goal_atom)
                prompt, completion, statement_texts = _tokenize_sampled_completion(
                    tokenizer=tokenizer,
                    sequent=sequent,
                    sampled=sampled,
                    step_idx=step_idx,
                    completion_format=completion_format,
                )
                payload = pickle.dumps(
                    {
                        "distance": int(sampled.distance),
                        "start_layer": int(sampled.start_layer),
                        "src_layer": int(src_layer),
                        "step_idx": int(step_idx),
                        "goal_atom": sampled.goal_atom.text,
                        "prompt": np.asarray(prompt, dtype=np.int32),
                        "completions": [np.asarray(completion, dtype=np.int32)],
                        "statement_text": instantiated_rule.statement_text,
                        "statement_texts": list(statement_texts),
                        "completion_format": str(completion_format),
                    },
                    protocol=5,
                )
                writer.write(payload)
                _update_autoreg_stats(stats, prompt, completion)

            examples += 1
    finally:
        writer.close()

    return _DistanceResult(
        distance=int(distance),
        examples=int(examples),
        records=int(writer.total_records),
        shards=int(writer.shard_count),
        stats=stats,
    )


def _write_metadata(
    *,
    out_dir: Path,
    results: dict[int, _DistanceResult],
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer,
    created_at: str,
    seed: int,
    examples_per_distance: int,
    min_distance: int,
    max_distance: int,
    rule_bank_path: str,
    workers: int,
    parallel_backend: str,
    config: dict,
) -> None:
    distances = {}
    stats_list = []
    for distance in sorted(results):
        result = results[distance]
        st = result.stats
        stats_list.append(st)
        distances[str(distance)] = {
            "examples": int(result.examples),
            "records": int(result.records),
            "shards": int(result.shards),
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
        "format": "arrayrecord_layer_fol",
        "seed": int(seed),
        "examples_per_distance": int(examples_per_distance),
        "distance_range": [int(min_distance), int(max_distance)],
        "rule_bank": rule_bank_path,
        "workers": int(workers),
        "parallel_backend": parallel_backend,
        "config": config,
        "tokenizer": tokenizer.to_dict(),
        "stats": overall,
        "distances": distances,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


_IPC_LEN_BYTES = 8
_IPC_MAX_FRAME_BYTES = 256 * 1024 * 1024


def _read_exact(stream, n_bytes: int) -> bytes:
    chunks: list[bytes] = []
    remaining = int(n_bytes)
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("Unexpected EOF while reading framed message.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _ipc_read_pickle_frame(stream):
    header = stream.read(_IPC_LEN_BYTES)
    if not header:
        raise EOFError("Peer closed stream.")
    if len(header) != _IPC_LEN_BYTES:
        raise RuntimeError("Received truncated IPC frame header.")
    payload_len = int.from_bytes(header, "big")
    if payload_len < 0 or payload_len > _IPC_MAX_FRAME_BYTES:
        raise RuntimeError(f"Invalid IPC frame length: {payload_len}")
    payload = _read_exact(stream, payload_len)
    return pickle.loads(payload)


def _ipc_write_pickle_frame(stream, payload_obj) -> None:
    payload = pickle.dumps(payload_obj, protocol=5)
    frame_len = len(payload)
    if frame_len > _IPC_MAX_FRAME_BYTES:
        raise RuntimeError(
            f"IPC payload exceeds max frame size: {frame_len} > {_IPC_MAX_FRAME_BYTES}"
        )
    stream.write(frame_len.to_bytes(_IPC_LEN_BYTES, "big"))
    stream.write(payload)
    stream.flush()


def _load_fol_online_worker_fns():
    try:
        from task.layer_fol import (  # pylint: disable=import-outside-toplevel
            _init_fol_online_worker,
            _init_fol_online_fresh_worker,
            _sample_fol_online_worker_records,
            _sample_fol_online_fresh_worker_records,
        )
    except Exception as err:
        raise RuntimeError(
            "Could not import FOL online worker functions. "
            "Run server mode as module: python -m task.layer_gen.generate_layer_fol --server-mode"
        ) from err
    return {
        "standard": (
            _init_fol_online_worker,
            _sample_fol_online_worker_records,
        ),
        "fresh_icl": (
            _init_fol_online_fresh_worker,
            _sample_fol_online_fresh_worker_records,
        ),
    }


def _server_worker_spec_from_config(config: dict) -> tuple[object, object, tuple]:
    worker_fns = _load_fol_online_worker_fns()
    sampler_kind = str(config.get("sampler_kind", "standard"))
    if sampler_kind not in worker_fns:
        raise ValueError(f"Unsupported sampler_kind: {sampler_kind!r}")

    if sampler_kind == "fresh_icl":
        initargs = (
            int(config["seed"]),
            dict(config["base_bank_payload"]),
            dict(config["tokenizer_payload"]),
            int(config["fresh_icl_n_predicates"]),
            int(config["rules_per_transition"]),
            int(config["k_in_min"]),
            int(config["k_in_max"]),
            int(config["k_out_min"]),
            int(config["k_out_max"]),
            int(config["initial_ant_max"]),
            int(config["sample_max_attempts"]),
            int(config["max_unify_solutions"]),
            int(config["max_n_demos"]),
            int(config.get("min_n_demos", 0)),
            bool(config.get("include_oracle", False)),
            (
                None
                if config.get("forced_step_idx", None) is None
                else int(config["forced_step_idx"])
            ),
            str(config.get("completion_format", "single")),
            int(config.get("predicate_name_len", 1)),
        )
    else:
        max_n_demos = int(config["max_n_demos"])
        min_n_demos = int(
            config.get(
                "min_n_demos",
                0 if max_n_demos == 0 else 1,
            )
        )
        initargs = (
            int(config["seed"]),
            dict(config["bank_payload"]),
            config.get("tokenizer_payload", None),
            tuple(int(distance) for distance in config["distances"]),
            int(config["initial_ant_max"]),
            int(config["sample_max_attempts"]),
            int(config["max_unify_solutions"]),
            max_n_demos,
            min_n_demos,
            bool(config.get("include_oracle", False)),
            (
                None
                if config.get("forced_step_idx", None) is None
                else int(config["forced_step_idx"])
            ),
            str(config.get("completion_format", "single")),
        )
    init_worker_fn, sample_records_fn = worker_fns[sampler_kind]
    return init_worker_fn, sample_records_fn, initargs


class _FOLSamplerServerState:
    def __init__(self, *, config: dict) -> None:
        init_worker_fn, sample_records_fn, initargs = _server_worker_spec_from_config(config)
        self._sample_records_fn = sample_records_fn
        self._executor = None
        self._prefetch_buffer: online_prefetch_util.AsyncRecordPrefetchBuffer | None = None

        workers = int(config["workers"])
        buffer_size = int(config["buffer_size"])
        batch_size = int(config["batch_size"])

        def _make_process_executor():
            return ProcessPoolExecutor(
                max_workers=workers,
                initializer=init_worker_fn,
                initargs=initargs,
            )

        def _make_thread_executor():
            return ThreadPoolExecutor(
                max_workers=workers,
                initializer=init_worker_fn,
                initargs=initargs,
            )

        resolved_backend, executor = online_prefetch_util.create_executor_with_fallback(
            backend="process",
            make_process_executor=_make_process_executor,
            make_thread_executor=_make_thread_executor,
        )
        self.resolved_backend = str(resolved_backend)
        if executor is None:
            init_worker_fn(*initargs)
            return

        records_per_job = 1
        if resolved_backend == "process":
            records_per_job = max(1, batch_size // max(1, workers))

        shutdown_fn = lambda: executor.shutdown(wait=True, cancel_futures=True)
        try:
            self._prefetch_buffer = online_prefetch_util.AsyncRecordPrefetchBuffer(
                submit_fn=lambda: executor.submit(
                    sample_records_fn,
                    records_per_job,
                ),
                buffer_size=buffer_size,
                on_close=shutdown_fn,
            )
        except Exception:
            shutdown_fn()
            raise
        self._executor = executor

    def take(self, count: int) -> list[dict]:
        count = int(count)
        if count < 0:
            raise ValueError(f"n_records must be >= 0, got {count}")
        if self._prefetch_buffer is None:
            if count == 0:
                return []
            return list(self._sample_records_fn(count))
        return self._prefetch_buffer.take(count)

    def close(self) -> None:
        if self._prefetch_buffer is not None:
            self._prefetch_buffer.close()
            self._prefetch_buffer = None
            self._executor = None
            return
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None


def _run_sampler_server() -> None:
    state: _FOLSamplerServerState | None = None
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    try:
        while True:
            try:
                request = _ipc_read_pickle_frame(stdin)
            except EOFError:
                break

            op = str(request.get("op", ""))
            try:
                if op == "init":
                    if state is not None:
                        state.close()
                    state = _FOLSamplerServerState(config=dict(request.get("config", {})))
                    response = {
                        "ok": True,
                        "op": "init",
                        "resolved_backend": str(state.resolved_backend),
                    }
                elif op == "sample":
                    if state is None:
                        raise RuntimeError("Server is not initialized.")
                    n_records = int(request.get("n_records", 0))
                    response = {
                        "ok": True,
                        "op": "sample",
                        "records": state.take(n_records),
                    }
                elif op == "close":
                    if state is not None:
                        state.close()
                        state = None
                    response = {"ok": True, "op": "close"}
                    _ipc_write_pickle_frame(stdout, response)
                    break
                else:
                    raise ValueError(f"Unsupported server op: {op!r}")
            except Exception as err:
                response = {
                    "ok": False,
                    "op": op,
                    "error_type": type(err).__name__,
                    "error_msg": str(err),
                    "traceback": traceback.format_exc(),
                }

            _ipc_write_pickle_frame(stdout, response)
    finally:
        if state is not None:
            state.close()


def main() -> None:
    args = _parse_args()
    if args.server_mode:
        _run_sampler_server()
        return

    if args.out_dir is None:
        raise ValueError("--out-dir is required unless --server-mode is set.")
    if args.seed is None:
        args.seed = _default_seed()

    if args.min_distance < 1:
        raise ValueError("--min-distance must be >= 1")
    if args.max_distance < args.min_distance:
        raise ValueError("--max-distance must be >= --min-distance")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    distances = list(range(args.min_distance, args.max_distance + 1))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    bank = _build_or_load_rule_bank(args, rng)
    tokenizer = tokenize_layer_fol.build_tokenizer_from_rule_bank(bank)
    rule_bank_out = args.out_dir / "rule_bank.json"
    save_fol_rule_bank(rule_bank_out, bank)

    config = {
        "n_layers": int(bank.n_layers),
        "predicates_per_layer": int(bank.predicates_per_layer),
        "rules_per_transition": int(args.rules_per_transition),
        "k_in_max": int(args.k_in_max),
        "k_out_max": int(args.k_out_max),
        "arity_max": int(bank.arity_max),
        "vars_per_rule_max": int(bank.vars_per_rule_max),
        "constants": list(bank.constants),
        "initial_ant_max": int(args.initial_ant_max),
        "sample_max_attempts": int(args.sample_max_attempts),
        "max_unify_solutions": int(args.max_unify_solutions),
        "completion_format": str(args.completion_format),
    }

    created_at = time.strftime("%Y-%m-%d %H:%M:%S")
    total_target = len(distances) * int(args.examples_per_distance)
    max_workers = max(
        1,
        min(int(args.workers), len(distances), os.cpu_count() or int(args.workers)),
    )
    bank_payload = bank.to_dict()
    results: dict[int, _DistanceResult] = {}

    executor = None
    backend = "single"
    if max_workers == 1:
        executor = ThreadPoolExecutor(max_workers=1)
        backend = "thread"
    else:
        try:
            executor = ProcessPoolExecutor(max_workers=max_workers)
            backend = "process"
        except (PermissionError, OSError):
            executor = ThreadPoolExecutor(max_workers=max_workers)
            backend = "thread"

    with tqdm(total=total_target, desc="generate layered fol", leave=True) as bar:
        with executor as pool:
            future_to_distance = {
                pool.submit(
                    _generate_distance_data,
                    out_dir=str(args.out_dir),
                    distance=distance,
                    examples_per_distance=int(args.examples_per_distance),
                    examples_per_shard=int(args.examples_per_shard),
                    writer_options=args.arrayrecord_options,
                    initial_ant_max=int(args.initial_ant_max),
                    sample_max_attempts=int(args.sample_max_attempts),
                    max_unify_solutions=int(args.max_unify_solutions),
                    seed=_distance_seed(int(args.seed), distance),
                    bank_payload=bank_payload,
                    completion_format=str(args.completion_format),
                ): distance
                for distance in distances
            }
            for future in as_completed(future_to_distance):
                result = future.result()
                results[result.distance] = result
                bar.update(int(result.examples))

    _write_metadata(
        out_dir=args.out_dir,
        results=results,
        tokenizer=tokenizer,
        created_at=created_at,
        seed=args.seed,
        examples_per_distance=args.examples_per_distance,
        min_distance=args.min_distance,
        max_distance=args.max_distance,
        rule_bank_path=str(rule_bank_out.resolve()),
        workers=max_workers,
        parallel_backend=backend,
        config=config,
    )


if __name__ == "__main__":
    main()
