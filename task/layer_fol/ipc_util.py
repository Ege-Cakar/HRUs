"""Shared IPC helpers for pickle-framed subprocess communication."""

from __future__ import annotations

import pickle


IPC_LEN_BYTES = 8
IPC_MAX_FRAME_BYTES = 256 * 1024 * 1024


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


def ipc_read_pickle_frame(stream):
    """Read a length-prefixed pickle frame from *stream*."""
    header = stream.read(IPC_LEN_BYTES)
    if not header:
        raise EOFError("Peer closed stream.")
    if len(header) != IPC_LEN_BYTES:
        raise RuntimeError("Received truncated IPC frame header.")
    payload_len = int.from_bytes(header, "big")
    if payload_len < 0 or payload_len > IPC_MAX_FRAME_BYTES:
        raise RuntimeError(f"Invalid IPC frame length: {payload_len}")
    payload = _read_exact(stream, payload_len)
    return pickle.loads(payload)


def ipc_write_pickle_frame(stream, payload_obj) -> None:
    """Write a length-prefixed pickle frame to *stream*."""
    payload = pickle.dumps(payload_obj, protocol=5)
    frame_len = len(payload)
    if frame_len > IPC_MAX_FRAME_BYTES:
        raise RuntimeError(
            f"IPC payload exceeds max frame size: {frame_len} > {IPC_MAX_FRAME_BYTES}"
        )
    stream.write(frame_len.to_bytes(IPC_LEN_BYTES, "big"))
    stream.write(payload)
    stream.flush()
