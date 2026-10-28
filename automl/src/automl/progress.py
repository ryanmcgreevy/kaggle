"""Opt-in progress events shared by the Python API and CLI."""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, TextIO


@dataclass(frozen=True)
class ProgressEvent:
    stage: str
    event: str
    elapsed_seconds: float
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))


ProgressCallback = Callable[[ProgressEvent], None]


def emit_progress(
    callback: ProgressCallback | None,
    stage: str,
    event: str,
    started_at: float,
    **details: Any,
) -> None:
    if callback is not None:
        callback(ProgressEvent(stage, event, max(0.0, time.perf_counter() - started_at), details))


def format_progress(event: ProgressEvent) -> str:
    labels = {
        "stage.start": "starting",
        "stage.complete": "completed",
        "fold.start": "starting fold",
        "fold.complete": "completed fold",
        "trial.start": "starting trial",
        "trial.complete": "completed trial",
        "trial.failed": "failed trial",
        "trial.pruned": "pruned trial",
        "model.start": "starting model",
        "model.complete": "completed model",
        "fit.start": "starting fit",
        "fit.complete": "completed fit",
        "predict.start": "starting prediction",
        "predict.complete": "completed prediction",
    }
    label = labels.get(event.event, event.event.replace(".", " "))
    details = " ".join(
        f"{key}={value:.6g}" if isinstance(value, float) else f"{key}={value}"
        for key, value in event.details.items()
        if value is not None
    )
    suffix = f" {details}" if details else ""
    return f"[{event.stage}] {label}{suffix} elapsed={event.elapsed_seconds:.1f}s"


class StderrProgress:
    """Thread-safe stderr renderer for CLI progress events."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream
        self._lock = threading.Lock()

    def __call__(self, event: ProgressEvent) -> None:
        stream = self._stream or sys.stderr
        with self._lock:
            print(format_progress(event), file=stream, flush=True)
