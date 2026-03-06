"""Task loader/sampler utilities for layered first-order tasks."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import threading

import grain
from grain._src.core import sharding, transforms
from grain._src.python import data_sources, samplers
import numpy as np

from task.layer_gen.util import online_prefetch as online_prefetch_util
from task.layer_gen.util import tokenize_layer_fol
from task.layer_gen.util.fol_rule_bank import (
    FOLRuleBank,
    FOLDepth3ICLSplitBundle,
    FOLLayerRule,
    FOLSequent,
    build_fresh_layer0_bank,
    build_random_fol_rule_bank,
    generate_fresh_predicate_names,
    load_fol_depth3_icl_split_bundle,
    load_fol_rule_bank,
    sample_fol_problem,
)
from task.layer_fol_common import (
    _build_tokenizer_for_fresh_icl,
    _build_tokenizer_for_split_bundle,
    compute_fol_dims,
)
from task.layer_fol_demos import augment_prompt_with_demos


_FOL_ONLINE_WORKER_LOCAL = threading.local()

def _pick_sampled_step_index(
    *,
    rng: np.random.Generator,
    n_steps: int,
    forced_step_idx: int | None,
) -> int:
    if int(n_steps) < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if forced_step_idx is None:
        return int(rng.integers(0, int(n_steps)))
    forced = int(forced_step_idx)
    if forced < 0 or forced >= int(n_steps):
        raise ValueError(
            f"forced_step_idx={forced} is out of range for sampled problem with {n_steps} steps."
        )
    return forced


def _rule_texts(rules: tuple[FOLLayerRule, ...] | list[FOLLayerRule]) -> list[str]:
    return [str(rule.statement_text) for rule in rules]


def _sampled_completion_texts(
    *,
    sampled,
    step_idx: int,
    completion_format: str,
) -> list[str]:
    step_idx = int(step_idx)
    completion_format = str(completion_format)
    if completion_format == "single":
        return [str(sampled.step_rules[step_idx].statement_text)]
    if completion_format == "full":
        return [str(rule.statement_text) for rule in sampled.step_rules[step_idx:]]
    raise ValueError(
        f"completion_format must be 'single' or 'full', got {completion_format!r}"
    )


def _tokenize_sampled_completion(
    *,
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer,
    sequent: FOLSequent,
    sampled,
    step_idx: int,
    completion_format: str,
) -> tuple[list[int], list[int], list[str]]:
    prompt = tokenizer.tokenize_prompt(sequent)
    statements = _sampled_completion_texts(
        sampled=sampled,
        step_idx=int(step_idx),
        completion_format=completion_format,
    )
    if str(completion_format) == "single":
        completion = tokenizer.encode_completion(statements[0])
    else:
        completion = tokenizer.encode_completion_sequence(statements)
    return prompt, completion, statements


def _fresh_rule_context(
    *,
    base_bank: FOLRuleBank,
    temp_bank: FOLRuleBank,
    src_layer: int,
    demo_schemas: tuple[FOLLayerRule, ...],
    demo_instances: tuple[str, ...],
) -> dict:
    src_layer = int(src_layer)
    fixed_rules = (
        tuple(base_bank.transition_rules(src_layer))
        if src_layer == 1
        else ()
    )
    return {
        "src_layer": src_layer,
        "active_rule_texts": _rule_texts(tuple(temp_bank.transition_rules(src_layer))),
        "fixed_rule_texts": _rule_texts(tuple(fixed_rules)),
        "demo_schema_texts": _rule_texts(tuple(demo_schemas)),
        "demo_instance_texts": [str(text) for text in demo_instances],
    }


def _init_fol_online_worker(
    seed_base: int,
    bank_payload: dict,
    tokenizer_payload: dict | None,
    distances: tuple[int, ...],
    initial_ant_max: int,
    sample_max_attempts: int,
    max_unify_solutions: int,
    max_n_demos: int,
    min_n_demos: int,
    forced_step_idx: int | None,
    completion_format: str = "single",
) -> None:
    bank = FOLRuleBank.from_dict(bank_payload)
    if tokenizer_payload is None:
        tokenizer = tokenize_layer_fol.build_tokenizer_from_rule_bank(bank)
    else:
        tokenizer = tokenize_layer_fol.FOLLayerTokenizer.from_dict(tokenizer_payload)
    worker_seed = (
        int(seed_base)
        + int(os.getpid()) * 1_000_003
        + int(threading.get_ident() % 1_000_003)
    )
    _FOL_ONLINE_WORKER_LOCAL.state = {
        "bank": bank,
        "tokenizer": tokenizer,
        "distances": tuple(int(distance) for distance in distances),
        "initial_ant_max": int(initial_ant_max),
        "sample_max_attempts": int(sample_max_attempts),
        "max_unify_solutions": int(max_unify_solutions),
        "max_n_demos": int(max_n_demos),
        "min_n_demos": int(min_n_demos),
        "forced_step_idx": (
            None if forced_step_idx is None else int(forced_step_idx)
        ),
        "completion_format": str(completion_format),
        "rng": np.random.default_rng(worker_seed),
    }


def _sample_fol_online_worker_record() -> dict:
    state = getattr(_FOL_ONLINE_WORKER_LOCAL, "state", None)
    if state is None:
        raise RuntimeError("FOL online worker state was not initialized.")

    rng: np.random.Generator = state["rng"]
    bank: FOLRuleBank = state["bank"]
    tokenizer = state["tokenizer"]

    distance = int(rng.choice(state["distances"]))
    sampled = None
    last_err = None
    for _ in range(3):
        try:
            sampled = sample_fol_problem(
                bank=bank,
                distance=distance,
                initial_ant_max=state["initial_ant_max"],
                rng=rng,
                max_attempts=state["sample_max_attempts"],
                max_unify_solutions=state["max_unify_solutions"],
            )
            break
        except RuntimeError as err:
            last_err = err

    if sampled is None:
        raise RuntimeError(
            f"Failed to sample online FOLLayerTask record for distance={distance} "
            f"after 3 retries with max_attempts={state['sample_max_attempts']}."
        ) from last_err

    step_idx = _pick_sampled_step_index(
        rng=rng,
        n_steps=len(sampled.step_rules),
        forced_step_idx=state["forced_step_idx"],
    )
    src_layer = sampled.step_layers[step_idx]
    ants = sampled.step_ants[step_idx]
    rule = sampled.step_rules[step_idx]
    sequent = FOLSequent(ants=ants, cons=sampled.goal_atom)
    prompt, completion, completion_statements = _tokenize_sampled_completion(
        tokenizer=tokenizer,
        sequent=sequent,
        sampled=sampled,
        step_idx=step_idx,
        completion_format=state["completion_format"],
    )
    augmented = augment_prompt_with_demos(
        prompt_tokens=prompt,
        rule_bank=bank,
        tokenizer=tokenizer,
        rng=rng,
        src_layer=int(src_layer),
        ants=ants,
        max_n_demos=int(state["max_n_demos"]),
        min_n_demos=int(state["min_n_demos"]),
        max_unify_solutions=int(state["max_unify_solutions"]),
    )
    return {
        "distance": int(distance),
        "src_layer": int(src_layer),
        "completion_format": str(state["completion_format"]),
        "prompt": np.asarray(augmented.prompt_tokens, dtype=np.int32),
        "completions": [np.asarray(completion, dtype=np.int32)],
        "statement_texts": list(completion_statements),
    }


def _sample_fol_online_worker_records(n_records: int) -> list[dict]:
    n_records = int(n_records)
    if n_records < 1:
        raise ValueError(f"n_records must be >= 1, got {n_records}")
    return [_sample_fol_online_worker_record() for _ in range(n_records)]


def _init_fol_online_fresh_worker(
    seed_base: int,
    base_bank_payload: dict,
    tokenizer_payload: dict,
    fresh_icl_n_predicates: int,
    rules_per_transition: int,
    k_in_min: int,
    k_in_max: int,
    k_out_min: int,
    k_out_max: int,
    initial_ant_max: int,
    sample_max_attempts: int,
    max_unify_solutions: int,
    max_n_demos: int,
    min_n_demos: int,
    forced_step_idx: int | None,
    completion_format: str,
    predicate_name_len: int = 1,
) -> None:
    base_bank = FOLRuleBank.from_dict(base_bank_payload)
    tokenizer = tokenize_layer_fol.FOLLayerTokenizer.from_dict(tokenizer_payload)
    worker_seed = (
        int(seed_base)
        + int(os.getpid()) * 1_000_003
        + int(threading.get_ident() % 1_000_003)
    )
    _FOL_ONLINE_WORKER_LOCAL.state = {
        "base_bank": base_bank,
        "tokenizer": tokenizer,
        "fresh_icl_n_predicates": int(fresh_icl_n_predicates),
        "rules_per_transition": int(rules_per_transition),
        "k_in_min": int(k_in_min),
        "k_in_max": int(k_in_max),
        "k_out_min": int(k_out_min),
        "k_out_max": int(k_out_max),
        "initial_ant_max": int(initial_ant_max),
        "sample_max_attempts": int(sample_max_attempts),
        "max_unify_solutions": int(max_unify_solutions),
        "max_n_demos": int(max_n_demos),
        "min_n_demos": int(min_n_demos),
        "forced_step_idx": (
            None if forced_step_idx is None else int(forced_step_idx)
        ),
        "completion_format": str(completion_format),
        "predicate_name_len": int(predicate_name_len),
        "rng": np.random.default_rng(worker_seed),
    }


def _sample_fol_online_fresh_worker_record() -> dict:
    state = getattr(_FOL_ONLINE_WORKER_LOCAL, "state", None)
    if state is None:
        raise RuntimeError("FOL online fresh worker state was not initialized.")

    rng: np.random.Generator = state["rng"]
    base_bank: FOLRuleBank = state["base_bank"]
    tokenizer = state["tokenizer"]

    fresh_preds = generate_fresh_predicate_names(
        state["fresh_icl_n_predicates"], rng,
        name_len=state["predicate_name_len"],
    )
    temp_bank = build_fresh_layer0_bank(
        base_bank=base_bank,
        fresh_predicates=fresh_preds,
        rules_per_transition=state["rules_per_transition"],
        k_in_min=state["k_in_min"],
        k_in_max=state["k_in_max"],
        k_out_min=state["k_out_min"],
        k_out_max=state["k_out_max"],
        rng=rng,
    )

    distance = 2
    sampled = None
    last_err = None
    for _ in range(3):
        try:
            sampled = sample_fol_problem(
                bank=temp_bank,
                distance=distance,
                initial_ant_max=state["initial_ant_max"],
                rng=rng,
                max_attempts=state["sample_max_attempts"],
                max_unify_solutions=state["max_unify_solutions"],
            )
            break
        except RuntimeError as err:
            last_err = err

    if sampled is None:
        raise RuntimeError(
            "Failed to sample fresh-ICL FOLLayerTask record for distance=2 "
            f"after 3 retries with max_attempts={state['sample_max_attempts']}."
        ) from last_err

    step_idx = _pick_sampled_step_index(
        rng=rng,
        n_steps=len(sampled.step_rules),
        forced_step_idx=state["forced_step_idx"],
    )
    src_layer = sampled.step_layers[step_idx]
    ants = sampled.step_ants[step_idx]
    rule = sampled.step_rules[step_idx]
    sequent = FOLSequent(ants=ants, cons=sampled.goal_atom)
    prompt, completion, completion_statements = _tokenize_sampled_completion(
        tokenizer=tokenizer,
        sequent=sequent,
        sampled=sampled,
        step_idx=step_idx,
        completion_format=state["completion_format"],
    )
    augmented = augment_prompt_with_demos(
        prompt_tokens=prompt,
        rule_bank=temp_bank,
        tokenizer=tokenizer,
        rng=rng,
        src_layer=int(src_layer),
        ants=ants,
        max_n_demos=int(state["max_n_demos"]),
        min_n_demos=int(state["min_n_demos"]),
        max_unify_solutions=int(state["max_unify_solutions"]),
    )
    rule_context = _fresh_rule_context(
        base_bank=base_bank,
        temp_bank=temp_bank,
        src_layer=int(src_layer),
        demo_schemas=augmented.demo_schemas,
        demo_instances=augmented.demo_instances,
    )
    return {
        "distance": int(distance),
        "src_layer": int(src_layer),
        "completion_format": str(state["completion_format"]),
        "prompt": np.asarray(augmented.prompt_tokens, dtype=np.int32),
        "completions": [np.asarray(completion, dtype=np.int32)],
        "statement_texts": list(completion_statements),
        "rule_context": rule_context,
    }


def _sample_fol_online_fresh_worker_records(n_records: int) -> list[dict]:
    n_records = int(n_records)
    if n_records < 1:
        raise ValueError(f"n_records must be >= 1, got {n_records}")
    return [_sample_fol_online_fresh_worker_record() for _ in range(n_records)]


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


class _FOLOnlineSamplerServerClient:
    def __init__(self, *, config: dict, cwd: Path) -> None:
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "task.layer_gen.generate_layer_fol", "--server-mode"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(cwd),
        )
        if self._proc.stdin is None or self._proc.stdout is None or self._proc.stderr is None:
            raise RuntimeError("Failed to initialize sampler server stdio pipes.")

        self._stdin = self._proc.stdin
        self._stdout = self._proc.stdout
        self._stderr = self._proc.stderr
        self._stderr_chunks: list[bytes] = []
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            daemon=True,
        )
        self._stderr_thread.start()
        self._closed = False

        try:
            response = self._request(
                {
                    "op": "init",
                    "config": dict(config),
                }
            )
            self._ensure_ok(response, op="init")
            self.resolved_backend = str(response.get("resolved_backend", "sync"))
        except Exception:
            self.close()
            raise

    def _drain_stderr(self) -> None:
        try:
            while True:
                chunk = self._stderr.read(4096)
                if not chunk:
                    break
                self._stderr_chunks.append(chunk)
                if len(self._stderr_chunks) > 64:
                    self._stderr_chunks = self._stderr_chunks[-64:]
        except Exception:
            return

    def _server_stderr_text(self) -> str:
        if not self._stderr_chunks:
            return ""
        raw = b"".join(self._stderr_chunks[-64:])
        return raw.decode("utf-8", errors="replace").strip()

    def _request(self, payload: dict):
        if self._closed:
            raise RuntimeError("Cannot request from closed sampler server client.")
        try:
            _ipc_write_pickle_frame(self._stdin, payload)
            return _ipc_read_pickle_frame(self._stdout)
        except Exception as err:
            stderr_text = self._server_stderr_text()
            message = f"Sampler server IPC failed: {type(err).__name__}: {err}"
            if stderr_text:
                message += f"\nserver stderr:\n{stderr_text}"
            raise RuntimeError(message) from err

    def _ensure_ok(self, response: dict, *, op: str) -> None:
        if bool(response.get("ok", False)):
            return
        err_type = str(response.get("error_type", "RuntimeError"))
        err_msg = str(response.get("error_msg", f"Sampler server op={op!r} failed."))
        err_trace = str(response.get("traceback", "")).strip()
        detail = f"Sampler server {op!r} failed: {err_type}: {err_msg}"
        if err_trace:
            detail += f"\n{err_trace}"
        raise RuntimeError(detail)

    def take(self, n_records: int) -> list[dict]:
        response = self._request(
            {
                "op": "sample",
                "n_records": int(n_records),
            }
        )
        self._ensure_ok(response, op="sample")
        records = response.get("records", [])
        if not isinstance(records, list):
            raise RuntimeError("Sampler server returned invalid records payload.")
        return records

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        try:
            if self._proc.poll() is None:
                try:
                    _ipc_write_pickle_frame(self._stdin, {"op": "close"})
                    _ = _ipc_read_pickle_frame(self._stdout)
                except Exception:
                    pass
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait(timeout=2.0)
        finally:
            try:
                self._stdin.close()
            except Exception:
                pass
            try:
                self._stdout.close()
            except Exception:
                pass
            try:
                self._stderr.close()
            except Exception:
                pass
            if self._stderr_thread.is_alive():
                self._stderr_thread.join(timeout=0.2)


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
        # online mode / rule bank config
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
        self._split_bundle: FOLDepth3ICLSplitBundle | None = None
        self._online_forced_step_idx: int | None = None
        self._base_bank: FOLRuleBank | None = None
        self._fresh_icl_n_predicates: int | None = None
        if self.task_split == "depth3_icl_transfer":
            if self.mode != "online":
                raise ValueError("task_split='depth3_icl_transfer' requires mode='online'.")
            if self.split_rule_bundle_path is None:
                raise ValueError(
                    "split_rule_bundle_path is required when task_split='depth3_icl_transfer'."
                )
            if rule_bank_path is not None:
                raise ValueError(
                    "rule_bank_path cannot be combined with task_split='depth3_icl_transfer'; "
                    "use split_rule_bundle_path."
                )
        elif self.task_split == "depth3_fresh_icl":
            if self.mode != "online":
                raise ValueError("task_split='depth3_fresh_icl' requires mode='online'.")
            if rule_bank_path is not None:
                raise ValueError(
                    "rule_bank_path cannot be combined with task_split='depth3_fresh_icl'."
                )
            if split_rule_bundle_path is not None:
                raise ValueError(
                    "split_rule_bundle_path cannot be combined with task_split='depth3_fresh_icl'."
                )

        if self.task_split == "none" and rule_bank_path is None:
            if ds_path is not None:
                rule_bank_path = Path(ds_path) / "rule_bank.json"
                if not rule_bank_path.exists():
                    print(f'warn: rule_bank.json not found in ds_path={ds_path}')
                    rule_bank_path = None

        self.distance_range = distance_range
        self._distances = self._normalize_distances(distance_range)
        if self.task_split == "depth3_icl_transfer":
            if self._distances != [2]:
                raise ValueError(
                    "task_split='depth3_icl_transfer' requires distance_range to resolve "
                    f"to [2], got {self._distances}."
                )
            self._online_forced_step_idx = 0 if self.split_role == "eval" else None
        elif self.task_split == "depth3_fresh_icl":
            if self._distances != [2]:
                raise ValueError(
                    "task_split='depth3_fresh_icl' requires distance_range to resolve "
                    f"to [2], got {self._distances}."
                )
            self._online_forced_step_idx = 0 if self.split_role == "eval" else None
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
        self._online_prefetch_buffer: online_prefetch_util.AsyncRecordPrefetchBuffer | None = (
            None
        )
        self._online_prefetch_enabled = False
        self._online_prefetch_backend_resolved = "sync"
        self._online_prefetch_workers_resolved = 1
        self._online_prefetch_buffer_size_resolved = max(1, self.batch_size)

        self.ds_path = Path(ds_path) if ds_path is not None else None

        self._rule_bank: FOLRuleBank | None = None
        self._tokenizer: tokenize_layer_fol.FOLLayerTokenizer | None = None
        if self.mode == "online" or rule_bank_path is not None:
            if self.task_split == "depth3_icl_transfer":
                if self.split_rule_bundle_path is None:
                    raise RuntimeError("split_rule_bundle_path must be set for split mode.")
                self._split_bundle = load_fol_depth3_icl_split_bundle(
                    Path(self.split_rule_bundle_path)
                )
                if self.split_role == "train":
                    self._rule_bank = self._split_bundle.train_bank
                else:
                    self._rule_bank = self._split_bundle.eval_bank
                self._tokenizer = _build_tokenizer_for_split_bundle(self._split_bundle)
            elif self.task_split == "depth3_fresh_icl":
                self._fresh_icl_n_predicates = (
                    int(fresh_icl_n_predicates)
                    if fresh_icl_n_predicates is not None
                    else int(predicates_per_layer)
                )
                # Build a base bank with standard layers 1-2 (layer 0 is placeholder).
                self._base_bank = build_random_fol_rule_bank(
                    n_layers=3,
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
                    rng=self._rng,
                )
                self._rule_bank = self._base_bank
                self._tokenizer = _build_tokenizer_for_fresh_icl(
                    base_bank=self._base_bank,
                    predicate_name_len=int(predicate_name_len),
                )
            elif rule_bank_path is not None:
                self._rule_bank = load_fol_rule_bank(Path(rule_bank_path))
            else:
                self._rule_bank = build_random_fol_rule_bank(
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
                    rng=self._rng,
                )
            if self._tokenizer is None:
                self._tokenizer = tokenize_layer_fol.build_tokenizer_from_rule_bank(
                    self._rule_bank
                )

        if self.mode == "offline":
            if self.ds_path is None:
                raise ValueError("ds_path is required when mode='offline'.")
            metadata_path = self.ds_path / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata_completion_format = str(
                metadata.get("config", {}).get("completion_format", "single")
            )
            if metadata_completion_format != self.completion_format:
                raise ValueError(
                    "Dataset completion_format mismatch: "
                    f"task requested {self.completion_format!r}, "
                    f"but metadata declares {metadata_completion_format!r}."
                )
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
            if self.task_split == "depth3_fresh_icl":
                self._init_online_prefetch_fresh()
            else:
                self._init_online_prefetch()

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

    def _online_worker_initargs(self) -> tuple:
        if self._rule_bank is None or self._tokenizer is None:
            raise RuntimeError("Online prefetch requires rule bank and tokenizer.")
        return (
            int(self.seed),
            self._rule_bank.to_dict(),
            (
                None
                if self.task_split == "none"
                else self._tokenizer.to_dict()
            ),
            tuple(self._distances),
            int(self.initial_ant_max),
            int(self.sample_max_attempts),
            int(self.max_unify_solutions),
            int(self.max_n_demos),
            int(self.min_n_demos),
            (
                None
                if self._online_forced_step_idx is None
                else int(self._online_forced_step_idx)
            ),
            str(self.completion_format),
        )

    def _online_fresh_worker_initargs(self) -> tuple:
        if self._base_bank is None or self._tokenizer is None:
            raise RuntimeError("Fresh-ICL prefetch requires base bank and tokenizer.")
        return (
            int(self.seed),
            self._base_bank.to_dict(),
            self._tokenizer.to_dict(),
            int(self._fresh_icl_n_predicates),
            int(self.rules_per_transition),
            int(self._k_in_min),
            int(self._k_in_max),
            int(self._k_out_min),
            int(self._k_out_max),
            int(self.initial_ant_max),
            int(self.sample_max_attempts),
            int(self.max_unify_solutions),
            int(self.max_n_demos),
            int(self.min_n_demos),
            None if self._online_forced_step_idx is None else int(self._online_forced_step_idx),
            str(self.completion_format),
            int(self._predicate_name_len),
        )

    def _init_online_prefetch_executor_backend(
        self,
        *,
        backend: str,
        workers: int,
        buffer_size: int,
    ) -> None:
        self._online_prefetch_backend_resolved = str(backend)
        self._online_prefetch_enabled = str(backend) != "sync"
        self._online_executor = None
        self._online_prefetch_buffer = None
        if backend == "sync":
            return

        initargs = self._online_worker_initargs()

        def _make_process_executor():
            return ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_fol_online_worker,
                initargs=initargs,
            )

        def _make_thread_executor():
            return ThreadPoolExecutor(
                max_workers=workers,
                initializer=_init_fol_online_worker,
                initargs=initargs,
            )

        resolved_backend, executor = online_prefetch_util.create_executor_with_fallback(
            backend=backend,
            make_process_executor=_make_process_executor,
            make_thread_executor=_make_thread_executor,
        )
        self._online_prefetch_backend_resolved = str(resolved_backend)
        if executor is None:
            self._online_prefetch_enabled = False
            return

        records_per_job = 1
        if resolved_backend == "process":
            records_per_job = max(1, self.batch_size // max(1, workers))

        shutdown_fn = lambda: executor.shutdown(wait=True, cancel_futures=True)
        try:
            self._online_prefetch_buffer = online_prefetch_util.AsyncRecordPrefetchBuffer(
                submit_fn=lambda: executor.submit(
                    _sample_fol_online_worker_records,
                    records_per_job,
                ),
                buffer_size=buffer_size,
                on_close=shutdown_fn,
            )
        except Exception:
            shutdown_fn()
            self._online_prefetch_enabled = False
            self._online_prefetch_backend_resolved = "sync"
            self._online_prefetch_buffer = None
            self._online_executor = None
            return

        self._online_executor = executor
        self._online_prefetch_enabled = True

    def _init_online_server_backend(
        self,
        *,
        workers: int,
        buffer_size: int,
    ) -> None:
        if self._rule_bank is None or self._tokenizer is None:
            raise RuntimeError("Online prefetch requires rule bank and tokenizer.")

        repo_root = Path(__file__).resolve().parents[1]
        config = {
            "seed": int(self.seed),
            "bank_payload": self._rule_bank.to_dict(),
            "tokenizer_payload": (
                None
                if self.task_split == "none"
                else self._tokenizer.to_dict()
            ),
            "distances": tuple(int(distance) for distance in self._distances),
            "initial_ant_max": int(self.initial_ant_max),
            "sample_max_attempts": int(self.sample_max_attempts),
            "max_unify_solutions": int(self.max_unify_solutions),
            "max_n_demos": int(self.max_n_demos),
            "min_n_demos": int(self.min_n_demos),
            "forced_step_idx": (
                None
                if self._online_forced_step_idx is None
                else int(self._online_forced_step_idx)
            ),
            "completion_format": str(self.completion_format),
            "workers": int(workers),
            "buffer_size": int(buffer_size),
            "batch_size": int(self.batch_size),
        }
        server_client = _FOLOnlineSamplerServerClient(config=config, cwd=repo_root)
        self._online_server_client = server_client
        self._online_prefetch_backend_resolved = "server"
        self._online_prefetch_enabled = True

    def _init_online_prefetch_executor_backend_fresh(
        self,
        *,
        backend: str,
        workers: int,
        buffer_size: int,
    ) -> None:
        self._online_prefetch_backend_resolved = str(backend)
        self._online_prefetch_enabled = str(backend) != "sync"
        self._online_executor = None
        self._online_prefetch_buffer = None
        if backend == "sync":
            return

        initargs = self._online_fresh_worker_initargs()

        def _make_process_executor():
            return ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_fol_online_fresh_worker,
                initargs=initargs,
            )

        def _make_thread_executor():
            return ThreadPoolExecutor(
                max_workers=workers,
                initializer=_init_fol_online_fresh_worker,
                initargs=initargs,
            )

        resolved_backend, executor = online_prefetch_util.create_executor_with_fallback(
            backend=backend,
            make_process_executor=_make_process_executor,
            make_thread_executor=_make_thread_executor,
        )
        self._online_prefetch_backend_resolved = str(resolved_backend)
        if executor is None:
            self._online_prefetch_enabled = False
            return

        records_per_job = 1
        if resolved_backend == "process":
            records_per_job = max(1, self.batch_size // max(1, workers))

        shutdown_fn = lambda: executor.shutdown(wait=True, cancel_futures=True)
        try:
            self._online_prefetch_buffer = online_prefetch_util.AsyncRecordPrefetchBuffer(
                submit_fn=lambda: executor.submit(
                    _sample_fol_online_fresh_worker_records,
                    records_per_job,
                ),
                buffer_size=buffer_size,
                on_close=shutdown_fn,
            )
        except Exception:
            shutdown_fn()
            self._online_prefetch_enabled = False
            self._online_prefetch_backend_resolved = "sync"
            self._online_prefetch_buffer = None
            self._online_executor = None
            return

        self._online_executor = executor
        self._online_prefetch_enabled = True

    def _init_online_prefetch_fresh(self) -> None:
        enabled, backend, workers, buffer_size = (
            online_prefetch_util.resolve_online_prefetch_config(
                enable=self._online_prefetch_requested,
                backend=(
                    self._online_prefetch_backend_requested
                    if self._online_prefetch_backend_requested != "server"
                    else "process"
                ),
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
        self._init_online_prefetch_executor_backend_fresh(
            backend=str(backend),
            workers=int(workers),
            buffer_size=int(buffer_size),
        )

    def _init_online_prefetch(self) -> None:
        requested_backend = str(self._online_prefetch_backend_requested)
        if requested_backend == "server":
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
                self._init_online_server_backend(
                    workers=int(workers),
                    buffer_size=int(buffer_size),
                )
                return
            except Exception:
                self._online_server_client = None
                self._init_online_prefetch_executor_backend(
                    backend="thread",
                    workers=int(workers),
                    buffer_size=int(buffer_size),
                )
                return

        enabled, backend, workers, buffer_size = online_prefetch_util.resolve_online_prefetch_config(
            enable=self._online_prefetch_requested,
            backend=requested_backend,
            workers=self._online_prefetch_workers_requested,
            buffer_size=self._online_prefetch_buffer_size_requested,
            batch_size=self.batch_size,
        )
        self._online_prefetch_workers_resolved = int(workers)
        self._online_prefetch_buffer_size_resolved = int(buffer_size)
        self._online_prefetch_backend_resolved = str(backend)
        self._online_prefetch_enabled = bool(enabled)

        if not enabled:
            return

        self._init_online_prefetch_executor_backend(
            backend=str(backend),
            workers=int(workers),
            buffer_size=int(buffer_size),
        )

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
        if self.task_split == "depth3_fresh_icl":
            return self._sample_online_record_fresh_icl()

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

        step_idx = _pick_sampled_step_index(
            rng=self._rng,
            n_steps=len(sampled.step_rules),
            forced_step_idx=self._online_forced_step_idx,
        )
        src_layer = sampled.step_layers[step_idx]
        ants = sampled.step_ants[step_idx]
        rule = sampled.step_rules[step_idx]
        sequent = FOLSequent(ants=ants, cons=sampled.goal_atom)
        prompt, completion, completion_statements = _tokenize_sampled_completion(
            tokenizer=self._tokenizer,
            sequent=sequent,
            sampled=sampled,
            step_idx=step_idx,
            completion_format=self.completion_format,
        )
        augmented = augment_prompt_with_demos(
            prompt_tokens=prompt,
            rule_bank=self._rule_bank,
            tokenizer=self._tokenizer,
            rng=self._rng,
            src_layer=int(src_layer),
            ants=ants,
            max_n_demos=int(self.max_n_demos),
            min_n_demos=int(self.min_n_demos),
            max_unify_solutions=int(self.max_unify_solutions),
        )
        return {
            "distance": int(distance),
            "src_layer": int(src_layer),
            "completion_format": str(self.completion_format),
            "prompt": np.asarray(augmented.prompt_tokens, dtype=np.int32),
            "completions": [np.asarray(completion, dtype=np.int32)],
            "statement_texts": list(completion_statements),
        }

    def _sample_online_record_fresh_icl(self) -> dict:
        if self._base_bank is None:
            raise RuntimeError("depth3_fresh_icl requires a base bank.")
        if self._tokenizer is None:
            raise RuntimeError("depth3_fresh_icl requires a tokenizer.")
        if self._fresh_icl_n_predicates is None:
            raise RuntimeError("depth3_fresh_icl requires _fresh_icl_n_predicates.")

        fresh_preds = generate_fresh_predicate_names(
            self._fresh_icl_n_predicates, self._rng,
            name_len=self._predicate_name_len,
        )
        temp_bank = build_fresh_layer0_bank(
            base_bank=self._base_bank,
            fresh_predicates=fresh_preds,
            rules_per_transition=self.rules_per_transition,
            k_in_min=self._k_in_min,
            k_in_max=self._k_in_max,
            k_out_min=self._k_out_min,
            k_out_max=self._k_out_max,
            rng=self._rng,
        )

        distance = 2
        sampled = None
        last_err = None
        for _ in range(3):
            try:
                sampled = sample_fol_problem(
                    bank=temp_bank,
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
                "Failed to sample fresh-ICL FOLLayerTask record for distance=2 "
                f"after 3 retries with max_attempts={self.sample_max_attempts}."
            ) from last_err

        step_idx = _pick_sampled_step_index(
            rng=self._rng,
            n_steps=len(sampled.step_rules),
            forced_step_idx=self._online_forced_step_idx,
        )
        src_layer = sampled.step_layers[step_idx]
        ants = sampled.step_ants[step_idx]
        rule = sampled.step_rules[step_idx]
        sequent = FOLSequent(ants=ants, cons=sampled.goal_atom)
        prompt, completion, completion_statements = _tokenize_sampled_completion(
            tokenizer=self._tokenizer,
            sequent=sequent,
            sampled=sampled,
            step_idx=step_idx,
            completion_format=self.completion_format,
        )
        augmented = augment_prompt_with_demos(
            prompt_tokens=prompt,
            rule_bank=temp_bank,
            tokenizer=self._tokenizer,
            rng=self._rng,
            src_layer=int(src_layer),
            ants=ants,
            max_n_demos=int(self.max_n_demos),
            min_n_demos=int(self.min_n_demos),
            max_unify_solutions=int(self.max_unify_solutions),
        )
        rule_context = _fresh_rule_context(
            base_bank=self._base_bank,
            temp_bank=temp_bank,
            src_layer=int(src_layer),
            demo_schemas=augmented.demo_schemas,
            demo_instances=augmented.demo_instances,
        )
        return {
            "distance": int(distance),
            "src_layer": int(src_layer),
            "completion_format": str(self.completion_format),
            "prompt": np.asarray(augmented.prompt_tokens, dtype=np.int32),
            "completions": [np.asarray(completion, dtype=np.int32)],
            "statement_texts": list(completion_statements),
            "rule_context": rule_context,
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

    @staticmethod
    def _ceil_pow2(n: int) -> int:
        n = int(n)
        if n <= 1:
            return 1
        return 1 << (n - 1).bit_length()

    @staticmethod
    def _coerce_autoreg_batch(batch) -> tuple[np.ndarray, np.ndarray]:
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
        return xs, ys

    @classmethod
    def _pad_autoreg_batch_to_length(cls, batch, *, n_seq: int):
        xs, ys = cls._coerce_autoreg_batch(batch)
        n_seq = int(n_seq)
        if n_seq < 2:
            raise ValueError(f"Autoregressive fixed length must be >= 2, got {n_seq}")
        if xs.shape[1] > n_seq or ys.shape[1] > n_seq:
            raise ValueError(
                "Autoregressive batch sequence exceeds fixed length: "
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

    def _apply_autoreg_fixed_length(self, batch):
        if self.prediction_objective != "autoregressive":
            return batch
        if self.fixed_length_mode == "batch_max":
            return batch

        if self.fixed_length_mode == "global_max":
            if self._global_autoreg_seq_len is None:
                raise RuntimeError("Global autoregressive seq length was not initialized.")
            return self._pad_autoreg_batch_to_length(
                batch,
                n_seq=int(self._global_autoreg_seq_len),
            )

        if self.fixed_length_mode == "next_pow2":
            xs, _ = self._coerce_autoreg_batch(batch)
            n_seq = max(2, int(self._ceil_pow2(xs.shape[1])))
            if self.fixed_length_n_seq is not None:
                cap = int(self.fixed_length_n_seq)
                if n_seq > cap:
                    raise ValueError(
                        "Autoregressive next_pow2 sequence exceeds fixed_length_n_seq cap: "
                        f"batch_n_seq={xs.shape[1]}, bucket_n_seq={n_seq}, fixed_length_n_seq={cap}"
                    )
            return self._pad_autoreg_batch_to_length(batch, n_seq=n_seq)

        raise ValueError(f"Unsupported fixed_length_mode={self.fixed_length_mode!r}")


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
