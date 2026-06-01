"""Incremental trace logging (line-buffered, flushed after each step)."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Iterator

_current: ContextVar[IncrementalTrace | None] = ContextVar("incremental_trace", default=None)


class IncrementalTrace:
    """Append-only trace file; each line is flushed immediately."""

    def __init__(self, path: Path, *, mode: str = "w") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file = path.open(mode, encoding="utf-8", buffering=1)
        self.step("trace opened")

    def step(self, message: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self._file.write(f"{ts} {message}\n")
        self._file.flush()

    def close(self) -> None:
        if self._file.closed:
            return
        self.step("trace closed")
        self._file.close()


def trace_step(message: str) -> None:
    trace = _current.get()
    if trace is not None:
        trace.step(message)


@contextmanager
def trace_scope(trace: IncrementalTrace | None) -> Iterator[IncrementalTrace | None]:
    if trace is None:
        yield None
        return
    token = _current.set(trace)
    try:
        yield trace
    finally:
        _current.reset(token)
        trace.close()
