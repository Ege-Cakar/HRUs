"""Prefetch and sampler-server helpers for layered FOL tasks."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import threading

from task.layer_gen.util import online_prefetch as online_prefetch_util
from .ipc_util import ipc_read_pickle_frame as _ipc_read_pickle_frame
from .ipc_util import ipc_write_pickle_frame as _ipc_write_pickle_frame
from .task_shared import OnlineWorkerSpec


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
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()
        self._closed = False

        try:
            response = self._request({"op": "init", "config": dict(config)})
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
        response = self._request({"op": "sample", "n_records": int(n_records)})
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


@dataclass
class PrefetchSetup:
    executor: object | None = None
    buffer: online_prefetch_util.AsyncRecordPrefetchBuffer | None = None
    server_client: _FOLOnlineSamplerServerClient | None = None
    enabled: bool = False
    backend_resolved: str = "sync"


def init_executor_prefetch(
    *,
    worker_spec: OnlineWorkerSpec,
    backend: str,
    workers: int,
    buffer_size: int,
    batch_size: int,
) -> PrefetchSetup:
    setup = PrefetchSetup(enabled=str(backend) != "sync", backend_resolved=str(backend))
    if str(backend) == "sync":
        return setup

    def _make_process_executor():
        return ProcessPoolExecutor(
            max_workers=int(workers),
            initializer=worker_spec.init_fn,
            initargs=worker_spec.initargs,
        )

    def _make_thread_executor():
        return ThreadPoolExecutor(
            max_workers=int(workers),
            initializer=worker_spec.init_fn,
            initargs=worker_spec.initargs,
        )

    resolved_backend, executor = online_prefetch_util.create_executor_with_fallback(
        backend=str(backend),
        make_process_executor=_make_process_executor,
        make_thread_executor=_make_thread_executor,
    )
    setup.backend_resolved = str(resolved_backend)
    if executor is None:
        setup.enabled = False
        return setup

    records_per_job = 1
    if resolved_backend == "process":
        records_per_job = max(1, int(batch_size) // max(1, int(workers)))

    shutdown_fn = lambda: executor.shutdown(wait=True, cancel_futures=True)
    try:
        setup.buffer = online_prefetch_util.AsyncRecordPrefetchBuffer(
            submit_fn=lambda: executor.submit(
                worker_spec.sample_records_fn,
                records_per_job,
            ),
            buffer_size=int(buffer_size),
            on_close=shutdown_fn,
        )
    except Exception:
        shutdown_fn()
        setup.enabled = False
        setup.backend_resolved = "sync"
        return setup

    setup.executor = executor
    setup.enabled = True
    return setup


def init_server_prefetch(
    *,
    server_config: dict,
    repo_root: Path,
) -> PrefetchSetup:
    client = _FOLOnlineSamplerServerClient(config=server_config, cwd=repo_root)
    online_prefetch_util.warn_backend_change(
        requested_backend="process",
        resolved_backend=str(client.resolved_backend),
        context="sampler server prefetch",
    )
    return PrefetchSetup(
        server_client=client,
        enabled=True,
        backend_resolved="server",
    )
