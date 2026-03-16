"""Subprocess-based parallel precomputation for cluster demo distribution.

Architecture::

    Main process (JAX)
      └─ ClusterPrecomputeClient  ──stdin/stdout pickle frames──>  Server subprocess
                                                                      └─ ProcessPoolExecutor
                                                                           ├─ Worker 0
                                                                           ├─ Worker 1
                                                                           └─ ...

The server is launched as ``python -m task.layer_fol.cluster_precompute`` and
communicates via length-prefixed pickle frames (same protocol as the existing
FOL sampler server in ``task_prefetch.py`` / ``generate_layer_fol.py``).
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import threading
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .ipc_util import ipc_read_pickle_frame, ipc_write_pickle_frame


# ---------------------------------------------------------------------------
# Worker function (runs in child processes of the server subprocess)
# ---------------------------------------------------------------------------

def _worker_batch(
    rule_bank_dict: dict,
    src_layer: int,
    batch_size: int,
    seed: int,
    max_unify_solutions: int,
    distance: int,
    initial_ant_max: int,
    fallback_ranked: dict[int, list] | None,
) -> list[dict[int, list]]:
    """Process a batch of cluster candidate samples.

    Each worker reconstructs the rule bank, precomputes reachable sets once,
    then loops ``batch_size`` times sampling fresh queries and classifying.
    """
    from task.layer_gen.util.fol_rule_bank import (
        FOLRuleBank,
        sample_fol_problem,
    )
    from .demos import (
        _classify_rules_by_rank,
        _precompute_reachable_sets,
    )

    rule_bank = FOLRuleBank.from_dict(rule_bank_dict)
    rules = list(rule_bank.transition_rules(src_layer))

    if fallback_ranked is None:
        fallback_ranked = {1: [], 2: [], 3: [], 4: list(rules)}

    reachable_sets = _precompute_reachable_sets(rules, rule_bank)
    rng = np.random.default_rng(seed)
    results: list[dict[int, list]] = []

    for _ in range(batch_size):
        fresh_ranked = None
        try:
            sampled = sample_fol_problem(
                bank=rule_bank,
                distance=int(distance),
                initial_ant_max=int(initial_ant_max),
                rng=rng,
                max_attempts=128,
                max_unify_solutions=int(max_unify_solutions),
            )
            # Find the step matching src_layer.
            for step_idx, layer in enumerate(sampled.step_layers):
                if int(layer) == int(src_layer):
                    fresh_ants = tuple(sampled.step_ants[step_idx])
                    fresh_goal = sampled.goal_atom
                    fresh_ranked = _classify_rules_by_rank(
                        rules=rules,
                        ants=fresh_ants,
                        goal_atom=fresh_goal,
                        rule_bank=rule_bank,
                        max_unify_solutions=int(max_unify_solutions),
                        reachable_sets=reachable_sets,
                    )
                    break
        except RuntimeError:
            pass

        results.append(fresh_ranked if fresh_ranked is not None else fallback_ranked)

    return results


# ---------------------------------------------------------------------------
# Client (runs in the main/JAX process)
# ---------------------------------------------------------------------------

class ClusterPrecomputeClient:
    """Subprocess client for parallel cluster candidate precomputation.

    Usage::

        with ClusterPrecomputeClient(n_workers=4, cwd=repo_root) as server:
            candidates = _precompute_cluster_candidate_rankings(
                ..., server=server,
            )
    """

    def __init__(self, *, n_workers: int = 4, cwd: Path | None = None) -> None:
        if cwd is None:
            cwd = Path(__file__).resolve().parents[2]
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "task.layer_fol.cluster_precompute"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(cwd),
        )
        if self._proc.stdin is None or self._proc.stdout is None or self._proc.stderr is None:
            raise RuntimeError("Failed to initialize cluster precompute server stdio pipes.")

        self._stdin = self._proc.stdin
        self._stdout = self._proc.stdout
        self._stderr = self._proc.stderr
        self._stderr_chunks: list[bytes] = []
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()
        self._closed = False

        try:
            response = self._request({"op": "init", "n_workers": int(n_workers)})
            self._ensure_ok(response, op="init")
            self.n_workers = int(response.get("n_workers", n_workers))
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
            raise RuntimeError("Cannot request from closed cluster precompute client.")
        try:
            ipc_write_pickle_frame(self._stdin, payload)
            return ipc_read_pickle_frame(self._stdout)
        except Exception as err:
            stderr_text = self._server_stderr_text()
            message = f"Cluster precompute server IPC failed: {type(err).__name__}: {err}"
            if stderr_text:
                message += f"\nserver stderr:\n{stderr_text}"
            raise RuntimeError(message) from err

    def _ensure_ok(self, response: dict, *, op: str) -> None:
        if bool(response.get("ok", False)):
            return
        err_type = str(response.get("error_type", "RuntimeError"))
        err_msg = str(response.get("error_msg", f"Cluster precompute op={op!r} failed."))
        err_trace = str(response.get("traceback", "")).strip()
        detail = f"Cluster precompute server {op!r} failed: {err_type}: {err_msg}"
        if err_trace:
            detail += f"\n{err_trace}"
        raise RuntimeError(detail)

    def precompute(
        self,
        *,
        rule_bank,
        src_layer: int,
        cluster_n_samples: int,
        seed: int,
        max_unify_solutions: int,
        distance: int,
        initial_ant_max: int,
        fallback_ranked: dict | None = None,
    ) -> list[dict]:
        response = self._request({
            "op": "precompute",
            "rule_bank_dict": rule_bank.to_dict(),
            "src_layer": int(src_layer),
            "cluster_n_samples": int(cluster_n_samples),
            "seed": int(seed),
            "max_unify_solutions": int(max_unify_solutions),
            "distance": int(distance),
            "initial_ant_max": int(initial_ant_max),
            "fallback_ranked": fallback_ranked,
        })
        self._ensure_ok(response, op="precompute")
        return response["candidate_rankings"]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        try:
            if self._proc.poll() is None:
                try:
                    ipc_write_pickle_frame(self._stdin, {"op": "close"})
                    _ = ipc_read_pickle_frame(self._stdout)
                except Exception:
                    pass
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait(timeout=2.0)
        finally:
            for stream in (self._stdin, self._stdout, self._stderr):
                try:
                    stream.close()
                except Exception:
                    pass
            if self._stderr_thread.is_alive():
                self._stderr_thread.join(timeout=0.2)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# ---------------------------------------------------------------------------
# Server (runs as a subprocess)
# ---------------------------------------------------------------------------

def _run_server() -> None:
    """Server loop: read pickle-framed requests on stdin, respond on stdout."""
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    executor: ProcessPoolExecutor | None = None
    n_workers = 1

    try:
        while True:
            try:
                request = ipc_read_pickle_frame(stdin)
            except EOFError:
                break

            op = str(request.get("op", ""))
            try:
                if op == "init":
                    if executor is not None:
                        executor.shutdown(wait=True, cancel_futures=True)
                    n_workers = max(1, int(request.get("n_workers", 1)))
                    executor = ProcessPoolExecutor(max_workers=n_workers)
                    response = {"ok": True, "op": "init", "n_workers": n_workers}

                elif op == "precompute":
                    if executor is None:
                        raise RuntimeError("Server not initialized (call 'init' first).")

                    rule_bank_dict = dict(request["rule_bank_dict"])
                    src_layer = int(request["src_layer"])
                    cluster_n_samples = int(request["cluster_n_samples"])
                    seed = int(request["seed"])
                    max_unify_solutions = int(request["max_unify_solutions"])
                    distance = int(request["distance"])
                    initial_ant_max = int(request["initial_ant_max"])
                    fallback_ranked = request.get("fallback_ranked", None)

                    # Split work across workers.
                    batch_size = math.ceil(cluster_n_samples / n_workers)
                    rng = np.random.default_rng(seed)
                    futures = []
                    remaining = cluster_n_samples
                    for _ in range(n_workers):
                        if remaining <= 0:
                            break
                        this_batch = min(batch_size, remaining)
                        batch_seed = int(rng.integers(1 << 63))
                        futures.append(executor.submit(
                            _worker_batch,
                            rule_bank_dict,
                            src_layer,
                            this_batch,
                            batch_seed,
                            max_unify_solutions,
                            distance,
                            initial_ant_max,
                            fallback_ranked,
                        ))
                        remaining -= this_batch

                    candidate_rankings: list[dict] = []
                    for future in futures:
                        candidate_rankings.extend(future.result())

                    response = {
                        "ok": True,
                        "op": "precompute",
                        "candidate_rankings": candidate_rankings,
                    }

                elif op == "close":
                    if executor is not None:
                        executor.shutdown(wait=True, cancel_futures=True)
                        executor = None
                    response = {"ok": True, "op": "close"}
                    ipc_write_pickle_frame(stdout, response)
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

            ipc_write_pickle_frame(stdout, response)

    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    _run_server()
