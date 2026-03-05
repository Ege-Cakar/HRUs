"""Task loader/sampler and evaluation tools for layered first-order tasks."""

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
from typing import Any, Callable, Iterable, Protocol

import grain
from grain._src.core import sharding, transforms
from grain._src.python import data_sources, samplers
import numpy as np

from task.layer import AutoregressiveLogitsAdapter, CompletionLogitsAdapter
from task.layer_gen.util import tokenize_layer_fol
from task.layer_gen.util import online_prefetch as online_prefetch_util
from task.layer_gen.util.fol_rule_bank import (
    FOLAtom,
    FOLDepth3ICLSplitBundle,
    FOLLayerRule,
    FOLRuleBank,
    FOLSequent,
    build_fresh_layer0_bank,
    build_random_fol_rule_bank,
    generate_fresh_predicate_names,
    load_fol_depth3_icl_split_bundle,
    load_fol_rule_bank,
    parse_atom_text,
    parse_clause_text,
    sample_fol_problem,
)


_FOL_ONLINE_WORKER_LOCAL = threading.local()


def _build_tokenizer_for_split_bundle(
    bundle: FOLDepth3ICLSplitBundle,
) -> tokenize_layer_fol.FOLLayerTokenizer:
    max_vars = max(
        int(bundle.train_bank.vars_per_rule_max),
        int(bundle.eval_bank.vars_per_rule_max),
    )
    identifiers: set[str] = set(bundle.train_bank.constants) | set(bundle.eval_bank.constants)
    identifiers.update(bundle.train_bank.predicate_arities)
    identifiers.update(bundle.eval_bank.predicate_arities)
    identifiers.update(f"x{idx}" for idx in range(1, int(max_vars) + 1))
    predicate_identifiers = set(bundle.train_bank.predicate_arities)
    predicate_identifiers.update(bundle.eval_bank.predicate_arities)
    return tokenize_layer_fol.build_tokenizer_from_identifiers(
        sorted(identifiers),
        predicate_identifiers=sorted(predicate_identifiers),
    )


def _fresh_predicate_sentinels(*, name_len: int = 1) -> list[str]:
    """Generate sentinel predicates covering all chars in the fresh predicate charset."""
    from task.layer_gen.util.fol_rule_bank import _FRESH_PREDICATE_CHARSET

    charset = _FRESH_PREDICATE_CHARSET
    sentinels: list[str] = []
    for i in range(0, len(charset), name_len):
        chunk = charset[i : i + name_len].ljust(name_len, charset[0])
        sentinels.append(f"r_{chunk}")
    return sentinels


def _build_tokenizer_for_fresh_icl(
    *,
    base_bank: FOLRuleBank,
    predicate_name_len: int = 1,
) -> tokenize_layer_fol.FOLLayerTokenizer:
    """Build a tokenizer for fresh-ICL that covers all possible fresh predicate chars."""
    identifiers: set[str] = set(base_bank.constants)
    identifiers.update(base_bank.predicate_arities)
    identifiers.update(f"x{idx}" for idx in range(1, int(base_bank.vars_per_rule_max) + 1))

    predicate_identifiers = set(base_bank.predicate_arities)
    # Add sentinel predicates covering all alphanumeric chars used by fresh names.
    sentinels = _fresh_predicate_sentinels(name_len=predicate_name_len)
    identifiers.update(sentinels)
    predicate_identifiers.update(sentinels)

    return tokenize_layer_fol.build_tokenizer_from_identifiers(
        sorted(identifiers),
        predicate_identifiers=sorted(predicate_identifiers),
    )


def compute_fol_dims(
    *,
    rule_banks: Iterable[FOLRuleBank],
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer,
    initial_ant_max: int,
    max_n_demos: int = 0,
    extra_predicate_arities: dict[str, int] | None = None,
    fresh_k_in_max: int | None = None,
    fresh_k_out_max: int | None = None,
) -> dict:
    """Compute max sequence-length dimensions from one or more FOL rule banks.

    Parameters
    ----------
    rule_banks : iterable of FOLRuleBank
        One or more rule banks whose rules and predicate arities are merged.
    tokenizer : FOLLayerTokenizer
    initial_ant_max : int
        Maximum initial antecedents.
    max_n_demos : int
        Maximum number of demos (0 = no demos).
    extra_predicate_arities : dict or None
        Additional predicate->arity entries (e.g. fresh sentinels) merged into
        atom-length computation.
    fresh_k_in_max, fresh_k_out_max : int or None
        When both set, adds fresh clause/completion length estimates.

    Returns
    -------
    dict with keys: n_vocab, max_prompt_len, max_completion_len, n_seq_ar, max_atom_len
    """
    all_rules = []
    merged_predicate_arities: dict[str, int] = {}
    first_const: str | None = None
    for bank in rule_banks:
        for rules in bank.transitions.values():
            all_rules.extend(rules)
        merged_predicate_arities.update(bank.predicate_arities)
        if first_const is None and bank.constants:
            first_const = str(bank.constants[0])

    if not all_rules:
        raise ValueError("Rule banks have no rules.")
    if first_const is None:
        raise ValueError("Rule banks have no constants.")

    if extra_predicate_arities:
        for pred, arity in extra_predicate_arities.items():
            if pred not in merged_predicate_arities:
                merged_predicate_arities[pred] = int(arity)

    max_rhs_atoms = max(len(rule.rhs) for rule in all_rules)
    max_prompt_facts = max(int(initial_ant_max), int(max_rhs_atoms))

    max_atom_len = 1
    for predicate, arity in merged_predicate_arities.items():
        atom = FOLAtom(
            predicate=str(predicate),
            args=tuple(first_const for _ in range(int(arity))),
        )
        max_atom_len = max(
            int(max_atom_len),
            len(tokenizer.encode_completion(atom.text)) - 1,
        )

    max_prompt_len = (
        max_prompt_facts * max_atom_len
        + max(0, max_prompt_facts - 1)
        + 1  # turnstile
        + max_atom_len  # consequent
        + 1  # SEP
    )

    if int(max_n_demos) > 0:
        max_demo_clause_len = max(
            len(tokenizer.encode_completion(rule.statement_text)) - 1
            for rule in all_rules
        )
        if fresh_k_in_max is not None and fresh_k_out_max is not None:
            fresh_clause_estimate = max_atom_len * (int(fresh_k_in_max) + int(fresh_k_out_max)) + 5
            max_demo_clause_len = max(max_demo_clause_len, fresh_clause_estimate)
        max_prompt_len += int(max_n_demos) * (int(max_demo_clause_len) + 1)

    max_completion_len = max(
        len(tokenizer.encode_completion(rule.statement_text))
        for rule in all_rules
    )
    if fresh_k_in_max is not None and fresh_k_out_max is not None:
        fresh_completion_estimate = max_atom_len * (int(fresh_k_in_max) + int(fresh_k_out_max)) + 5
        max_completion_len = max(max_completion_len, fresh_completion_estimate)

    n_seq_ar = int(max_prompt_len + max_completion_len - 1)
    n_vocab = max(512, int(tokenizer.vocab_size))

    return {
        "n_vocab": int(n_vocab),
        "max_prompt_len": int(max_prompt_len),
        "max_completion_len": int(max_completion_len),
        "n_seq_ar": int(n_seq_ar),
        "max_atom_len": int(max_atom_len),
    }


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
    prompt, completion = tokenizer.tokenize_example(sequent, rule.statement_text)
    prompt = _augment_prompt_with_demos(
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
        "prompt": np.asarray(prompt, dtype=np.int32),
        "completions": [np.asarray(completion, dtype=np.int32)],
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
    prompt, completion = tokenizer.tokenize_example(sequent, rule.statement_text)
    prompt = _augment_prompt_with_demos(
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
    return {
        "distance": int(distance),
        "src_layer": int(src_layer),
        "prompt": np.asarray(prompt, dtype=np.int32),
        "completions": [np.asarray(completion, dtype=np.int32)],
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


def _augment_prompt_with_demos(
    *,
    prompt_tokens: list[int],
    rule_bank: FOLRuleBank,
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer,
    rng: np.random.Generator,
    src_layer: int,
    ants: tuple[FOLAtom, ...],
    max_n_demos: int,
    min_n_demos: int,
    max_unify_solutions: int,
) -> list[int]:
    max_n_demos = int(max_n_demos)
    min_n_demos = int(min_n_demos)
    if min_n_demos > max_n_demos:
        raise ValueError(
            f"min_n_demos must be <= max_n_demos, got {min_n_demos} > {max_n_demos}"
        )
    if max_n_demos <= 0:
        return prompt_tokens

    n_demos = int(rng.integers(min_n_demos, max_n_demos + 1))
    if n_demos <= 0:
        return prompt_tokens
    schemas = _collect_applicable_demo_schemas(
        rule_bank=rule_bank,
        src_layer=int(src_layer),
        ants=ants,
        max_unify_solutions=int(max_unify_solutions),
    )
    if not schemas:
        return prompt_tokens

    sampled_schemas = _sample_demo_schemas_with_replacement(
        rng=rng,
        schemas=schemas,
        n_demos=n_demos,
    )
    if not sampled_schemas:
        return prompt_tokens

    demo_statements = [
        _instantiate_demo_schema_with_random_constants(
            rule=schema,
            constants=rule_bank.constants,
            rng=rng,
        )
        for schema in sampled_schemas
    ]

    return _prepend_demo_statements_to_prompt(
        prompt_tokens=prompt_tokens,
        demo_statements=demo_statements,
        tokenizer=tokenizer,
    )


def _collect_applicable_demo_schemas(
    *,
    rule_bank: FOLRuleBank,
    src_layer: int,
    ants: tuple[FOLAtom, ...],
    max_unify_solutions: int,
) -> list[FOLLayerRule]:
    schemas: list[FOLLayerRule] = []
    seen_schema_keys: set[str] = set()
    ground_ants = tuple(ants)
    for rule in rule_bank.transition_rules(int(src_layer)):
        substitutions = _find_lhs_substitutions_for_facts(
            lhs=rule.lhs,
            facts=ground_ants,
            max_solutions=int(max_unify_solutions),
        )
        if not any(_subst_binds_rhs_variables(rule=rule, subst=subst) for subst in substitutions):
            continue

        schema_key = str(rule.statement_text)
        if schema_key in seen_schema_keys:
            continue
        seen_schema_keys.add(schema_key)
        schemas.append(rule)
    return schemas


def _subst_binds_rhs_variables(*, rule: FOLLayerRule, subst: dict[str, str]) -> bool:
    rhs_vars = {
        term
        for atom in rule.rhs
        for term in atom.args
        if _is_variable(term)
    }
    return rhs_vars.issubset(set(subst))


def _find_lhs_substitutions_for_facts(
    *,
    lhs: tuple[FOLAtom, ...],
    facts: tuple[FOLAtom, ...],
    max_solutions: int,
) -> list[dict[str, str]]:
    solutions: list[dict[str, str]] = []

    def _search(idx: int, subst: dict[str, str]) -> None:
        if len(solutions) >= max_solutions:
            return
        if idx >= len(lhs):
            solutions.append(dict(subst))
            return

        templ = lhs[idx]
        for fact in facts:
            maybe = _unify_template_atom_with_ground(templ, fact, subst)
            if maybe is None:
                continue
            _search(idx + 1, maybe)
            if len(solutions) >= max_solutions:
                return

    _search(0, {})
    return solutions


def _sample_demo_schemas_with_replacement(
    *,
    rng: np.random.Generator,
    schemas: list[FOLLayerRule],
    n_demos: int,
) -> list[FOLLayerRule]:
    if n_demos < 1 or not schemas:
        return []

    picks = rng.integers(0, len(schemas), size=int(n_demos))
    return [schemas[int(idx)] for idx in picks]


def _instantiate_demo_schema_with_random_constants(
    *,
    rule: FOLLayerRule,
    constants: tuple[str, ...],
    rng: np.random.Generator,
) -> str:
    if not constants:
        raise ValueError("Cannot instantiate demo rule schema without constants.")

    variables = tuple(sorted(rule.variables()))
    if not variables:
        return str(rule.statement_text)

    substitution = {
        var: str(constants[int(rng.integers(0, len(constants)))])
        for var in variables
    }
    return str(rule.instantiate(substitution).statement_text)


def _prepend_demo_statements_to_prompt(
    *,
    prompt_tokens: list[int],
    demo_statements: list[str],
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer,
) -> list[int]:
    out: list[int] = []
    for statement in demo_statements:
        demo_completion = tokenizer.encode_completion(statement)
        out.extend(int(tok) for tok in demo_completion[:-1])
        out.append(int(tokenizer.sep_token_id))
    out.extend(int(tok) for tok in prompt_tokens)
    return out


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
        prompt, completion = self._tokenizer.tokenize_example(sequent, rule.statement_text)
        prompt = _augment_prompt_with_demos(
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
            "prompt": np.asarray(prompt, dtype=np.int32),
            "completions": [np.asarray(completion, dtype=np.int32)],
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
        prompt, completion = self._tokenizer.tokenize_example(sequent, rule.statement_text)
        prompt = _augment_prompt_with_demos(
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

            dims = compute_fol_dims(
                rule_banks=[self._rule_bank],
                tokenizer=self._tokenizer,
                initial_ant_max=self.initial_ant_max,
                max_n_demos=self.max_n_demos,
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
    n_failure_wrong_rule_error: int
    n_failure_inapplicable_rule_error: int
    n_failure_goal_not_reached: int
    success_rate: float
    decode_error_rate: float
    unknown_rule_error_rate: float
    wrong_rule_error_rate: float
    inapplicable_rule_error_rate: float
    goal_not_reached_rate: float
    avg_steps: float
    results: tuple[FOLLayerRolloutResult, ...]


FAILURE_DECODE_ERROR = "decode_error"
FAILURE_UNKNOWN_RULE_ERROR = "unknown_rule_error"
FAILURE_WRONG_RULE_ERROR = "wrong_rule_error"
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


def _unify_template_atom_with_ground_schema(
    template: FOLAtom,
    ground: FOLAtom,
    subst: dict[str, str],
    pred_map: dict[str, str],
) -> tuple[dict[str, str], dict[str, str]] | None:
    """Like ``_unify_template_atom_with_ground`` but matches structurally.

    Instead of requiring exact predicate name equality, maintains an injective
    mapping *pred_map* from template predicate names to ground predicate names.
    Arity must still match, and variable bindings are consistent as before.

    Returns ``(updated_subst, updated_pred_map)`` on success, ``None`` on failure.
    """
    if len(template.args) != len(ground.args):
        return None

    # Check / extend predicate mapping
    new_pred_map = dict(pred_map)
    bound_pred = new_pred_map.get(template.predicate)
    if bound_pred is None:
        # Injective: ground predicate must not already be a target of another template pred
        if ground.predicate in new_pred_map.values():
            return None
        new_pred_map[template.predicate] = ground.predicate
    elif bound_pred != ground.predicate:
        return None

    # Unify variable / constant arguments (identical to the exact version)
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
    return out, new_pred_map


def _find_multiset_matches_schema(
    *,
    templates: tuple[FOLAtom, ...],
    grounds: tuple[FOLAtom, ...],
    seed_subst: dict[str, str],
    seed_pred_map: dict[str, str],
    max_solutions: int,
) -> list[tuple[dict[str, str], dict[str, str]]]:
    """Like ``_find_multiset_matches`` but threads *pred_map* alongside *subst*."""
    if len(templates) != len(grounds):
        return []

    solutions: list[tuple[dict[str, str], dict[str, str]]] = []
    used = [False] * len(grounds)

    def _search(idx: int, subst: dict[str, str], pred_map: dict[str, str]) -> None:
        if len(solutions) >= max_solutions:
            return
        if idx >= len(templates):
            solutions.append((dict(subst), dict(pred_map)))
            return

        template = templates[idx]
        for ground_idx, ground in enumerate(grounds):
            if used[ground_idx]:
                continue
            maybe = _unify_template_atom_with_ground_schema(
                template, ground, subst, pred_map,
            )
            if maybe is None:
                continue
            new_subst, new_pred_map = maybe
            used[ground_idx] = True
            _search(idx + 1, new_subst, new_pred_map)
            used[ground_idx] = False
            if len(solutions) >= max_solutions:
                return

    _search(0, dict(seed_subst), dict(seed_pred_map))
    return solutions


def _find_schema_match_for_rule(
    *,
    template: FOLLayerRule,
    lhs_ground: tuple[FOLAtom, ...],
    rhs_ground: tuple[FOLAtom, ...],
    max_solutions: int = 64,
) -> bool:
    """Like ``_find_instantiation_for_rule`` but uses schema unification.

    Returns ``True`` if the structural pattern (atom counts, arities,
    variable-binding pattern) of *template* can be matched against the ground
    atoms, allowing arbitrary predicate-name renaming.
    """
    lhs_sols = _find_multiset_matches_schema(
        templates=template.lhs,
        grounds=lhs_ground,
        seed_subst={},
        seed_pred_map={},
        max_solutions=max_solutions,
    )
    if not lhs_sols:
        return False

    for lhs_subst, lhs_pred_map in lhs_sols:
        rhs_sols = _find_multiset_matches_schema(
            templates=template.rhs,
            grounds=rhs_ground,
            seed_subst=lhs_subst,
            seed_pred_map=lhs_pred_map,
            max_solutions=1,
        )
        if rhs_sols:
            return True
    return False


def _any_rule_schema_matches(
    lhs_ground: tuple[FOLAtom, ...],
    rhs_ground: tuple[FOLAtom, ...],
    candidates: Iterable[FOLLayerRule],
) -> bool:
    """Return ``True`` if any candidate rule's schema matches the ground atoms."""
    for rule in candidates:
        if _find_schema_match_for_rule(
            template=rule,
            lhs_ground=lhs_ground,
            rhs_ground=rhs_ground,
        ):
            return True
    return False


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
    demo_rules: Iterable[FOLLayerRule] | None = None,
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
        candidates = list(rule_bank.transition_rules(src_layer))
        candidates.extend(demo_rules or ())
        schema_matches = _any_rule_schema_matches(
            lhs_ground, rhs_ground, candidates,
        )
        return FOLRuleMatchResult(
            src_layer=src_layer,
            decoded_statement=decoded_statement,
            expected_statement_text=expected_statement_text,
            matched_rule=None,
            decode_error=False,
            unknown_rule_error=not schema_matches,
            wrong_rule_error=schema_matches,
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
    demo_rules: Iterable[FOLLayerRule] | None = None,
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

    demo_rules_list = list(demo_rules) if demo_rules is not None else None

    results = tuple(
        match_rule_completion_fol(
            rule_bank=rule_bank,
            src_layer=src_layer,
            completion_tokens=completion,
            expected_statement_text=expected_statement,
            tokenizer=tokenizer,
            demo_rules=demo_rules_list,
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
    demo_rules: Iterable[FOLLayerRule] | None = None,
) -> FOLLayerRolloutResult:
    tokenizer = _resolve_fol_tokenizer(rule_bank=rule_bank, tokenizer=tokenizer)
    demo_rules_list = list(demo_rules) if demo_rules is not None else None
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
            demo_rules=demo_rules_list,
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
            failure = (
                FAILURE_UNKNOWN_RULE_ERROR if matched.unknown_rule_error
                else FAILURE_WRONG_RULE_ERROR
            )
            traces.append(
                FOLLayerRolloutTraceStep(
                    step_idx=step_idx,
                    src_layer=src_layer,
                    prompt_tokens=tuple(int(tok) for tok in prompt),
                    completion_tokens=tuple(int(tok) for tok in completion),
                    decoded_statement=matched.decoded_statement,
                    matched_rule_statement=None,
                    decode_error=False,
                    unknown_rule_error=matched.unknown_rule_error,
                    inapplicable_rule_error=False,
                    goal_reached=False,
                )
            )
            return FOLLayerRolloutResult(
                success=False,
                failure_reason=failure,
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
    demo_rules: Iterable[FOLLayerRule] | None = None,
) -> FOLLayerRolloutMetrics:
    if rng is None:
        rng = np.random.default_rng()

    demo_rules_list = list(demo_rules) if demo_rules is not None else None

    results = tuple(
        run_layer_rollout_fol(
            rule_bank=rule_bank,
            example=example,
            model=model,
            adapter=adapter,
            tokenizer=tokenizer,
            temperature=temperature,
            rng=rng,
            demo_rules=demo_rules_list,
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
    n_failure_wrong_rule_error = sum(
        int(result.failure_reason == FAILURE_WRONG_RULE_ERROR)
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
        n_failure_wrong_rule_error=n_failure_wrong_rule_error,
        n_failure_inapplicable_rule_error=n_failure_inapplicable_rule_error,
        n_failure_goal_not_reached=n_failure_goal_not_reached,
        success_rate=_safe_rate(n_success, n_examples),
        decode_error_rate=_safe_rate(n_failure_decode_error, n_examples),
        unknown_rule_error_rate=_safe_rate(n_failure_unknown_rule_error, n_examples),
        wrong_rule_error_rate=_safe_rate(n_failure_wrong_rule_error, n_examples),
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
