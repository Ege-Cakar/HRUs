"""Shared helpers for async online sampling prefetch."""

from __future__ import annotations

from collections import deque
from concurrent.futures import FIRST_COMPLETED, Executor, Future, wait
import os
import sys
from typing import Callable


DEFAULT_PREFETCH_BACKEND = "process"
VALID_PREFETCH_BACKENDS = ("process", "thread", "sync")


def warn_backend_change(
    *,
    requested_backend: str,
    resolved_backend: str,
    context: str,
    detail: str | None = None,
) -> None:
    requested_backend = str(requested_backend)
    resolved_backend = str(resolved_backend)
    if requested_backend == resolved_backend:
        return

    message = (
        f"warn: {context} backend changed from "
        f"{requested_backend!r} to {resolved_backend!r}"
    )
    if detail:
        message += f" ({detail})"
    print(message, file=sys.stderr)


def resolve_online_prefetch_config(
    *,
    enable: bool,
    backend: str,
    workers: int | None,
    buffer_size: int | None,
    batch_size: int,
) -> tuple[bool, str, int, int]:
    backend = str(backend)
    if backend not in VALID_PREFETCH_BACKENDS:
        raise ValueError(
            "online_prefetch_backend must be one of "
            f"{VALID_PREFETCH_BACKENDS}, got {backend!r}"
        )

    if workers is None:
        cpu = os.cpu_count() or 2
        workers = max(1, min(8, cpu - 1))
    else:
        workers = int(workers)
    if workers < 1:
        raise ValueError(f"online_prefetch_workers must be >= 1, got {workers}")

    if buffer_size is None:
        buffer_size = max(int(batch_size) * 4, workers * 2)
    else:
        buffer_size = int(buffer_size)
    if buffer_size < 1:
        raise ValueError(
            f"online_prefetch_buffer_size must be >= 1, got {buffer_size}"
        )
    buffer_size = max(buffer_size, int(batch_size))

    enabled = bool(enable) and backend != "sync"
    resolved_backend = backend if enabled else "sync"
    return enabled, resolved_backend, workers, buffer_size


def create_executor_with_fallback(
    *,
    backend: str,
    make_process_executor: Callable[[], Executor],
    make_thread_executor: Callable[[], Executor],
) -> tuple[str, Executor | None]:
    backend = str(backend)
    if backend == "sync":
        return "sync", None
    if backend == "thread":
        try:
            return "thread", make_thread_executor()
        except Exception:
            warn_backend_change(
                requested_backend="thread",
                resolved_backend="sync",
                context="online prefetch",
                detail="thread executor initialization failed",
            )
            return "sync", None
    if backend == "process":
        try:
            return "process", make_process_executor()
        except Exception:
            try:
                warn_backend_change(
                    requested_backend="process",
                    resolved_backend="thread",
                    context="online prefetch",
                    detail="process executor initialization failed",
                )
                return "thread", make_thread_executor()
            except Exception:
                warn_backend_change(
                    requested_backend="process",
                    resolved_backend="sync",
                    context="online prefetch",
                    detail="process and thread executor initialization failed",
                )
                return "sync", None

    raise ValueError(
        "backend must be one of "
        f"{VALID_PREFETCH_BACKENDS}, got {backend!r}"
    )


class AsyncRecordPrefetchBuffer:
    """Keeps a fixed number of sampling jobs in flight and refills on consume."""

    def __init__(
        self,
        *,
        submit_fn: Callable[[], Future],
        buffer_size: int,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        self._submit_fn = submit_fn
        self._buffer_size = int(buffer_size)
        self._on_close = on_close
        self._futures: set[Future] = set()
        self._ready_items = deque()
        self._closed = False

        if self._buffer_size < 1:
            raise ValueError(f"buffer_size must be >= 1, got {self._buffer_size}")

        self._ensure_capacity()

    def _ensure_capacity(self) -> None:
        if self._closed:
            return
        while len(self._futures) < self._buffer_size:
            self._futures.add(self._submit_fn())

    def _pop_record(self):
        if self._closed:
            raise RuntimeError("Cannot consume from closed prefetch buffer.")
        if self._ready_items:
            return self._ready_items.popleft()
        if not self._futures:
            raise RuntimeError("Prefetch buffer has no in-flight futures.")

        done = [future for future in self._futures if future.done()]
        if not done:
            done_set, _ = wait(
                tuple(self._futures),
                return_when=FIRST_COMPLETED,
            )
            done = list(done_set)

        future = done[0]
        self._futures.remove(future)
        self._ensure_capacity()

        try:
            payload = future.result()
        except Exception as err:
            self.close()
            raise RuntimeError(
                "Online prefetch worker failed while sampling record."
            ) from err

        if isinstance(payload, list):
            self._ready_items.extend(payload)
        else:
            self._ready_items.append(payload)

        if not self._ready_items:
            raise RuntimeError("Online prefetch worker returned no records.")
        return self._ready_items.popleft()

    def take(self, count: int) -> list:
        count = int(count)
        if count < 0:
            raise ValueError(f"count must be >= 0, got {count}")
        return [self._pop_record() for _ in range(count)]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        for future in self._futures:
            future.cancel()
        self._futures.clear()
        self._ready_items.clear()

        if self._on_close is not None:
            self._on_close()
