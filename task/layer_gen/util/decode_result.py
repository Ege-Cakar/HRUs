from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class DecodeAttempt(Generic[T]):
    ok: bool
    value: T | None = None
    error: str | None = None

    @classmethod
    def success(cls, value: T) -> "DecodeAttempt[T]":
        return cls(ok=True, value=value, error=None)

    @classmethod
    def failure(cls, error: str) -> "DecodeAttempt[T]":
        return cls(ok=False, value=None, error=str(error))
