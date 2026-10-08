"""Metric registry with built-in and user-registered metrics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Iterable, Literal

import numpy as np
from sklearn import metrics as skm

from automl.errors import TaskMetricError


class Task(StrEnum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"


Direction = Literal["maximize", "minimize"]
Needs = Literal["labels", "proba", "values"]


@dataclass(frozen=True)
class Metric:
    name: str
    tasks: frozenset[Task]
    direction: Direction
    needs: Needs
    fn: Callable[[Any, Any], float]

    def __call__(self, y_true: Any, y_pred: Any) -> float:
        return float(self.fn(y_true, y_pred))


_REGISTRY: dict[str, Metric] = {}
_BUILTIN: set[str] = set()


def _coerce_tasks(tasks: Iterable[Task | str], name: str) -> frozenset[Task]:
    out: set[Task] = set()
    for t in tasks:
        try:
            out.add(Task(t))
        except ValueError:
            raise TaskMetricError(
                f"metric '{name}': unsupported task {t!r}; valid tasks: {[x.value for x in Task]}"
            ) from None
    if not out:
        raise TaskMetricError(f"metric '{name}' must support at least one task")
    return frozenset(out)


def register_metric(
    name: str,
    fn: Callable[[Any, Any], float],
    tasks: Iterable[Task | str],
    direction: Direction,
    needs: Needs = "values",
    *,
    overwrite: bool = False,
) -> Metric:
    """Register a metric callable `fn(y_true, y_pred) -> float`."""
    problems: list[str] = []
    if not isinstance(name, str) or not name:
        problems.append("metric name must be a non-empty string")
    if not callable(fn):
        problems.append(f"metric '{name}': fn must be callable")
    if direction not in ("maximize", "minimize"):
        problems.append(f"metric '{name}': direction must be 'maximize' or 'minimize', got {direction!r}")
    if needs not in ("labels", "proba", "values"):
        problems.append(f"metric '{name}': needs must be 'labels', 'proba' or 'values', got {needs!r}")
    if name in _BUILTIN:
        problems.append(f"metric '{name}' is built in and cannot be replaced")
    elif name in _REGISTRY and not overwrite:
        problems.append(f"metric '{name}' is already registered; pass overwrite=True to replace it")
    task_set: frozenset[Task] = frozenset()
    try:
        task_set = _coerce_tasks(tasks, name)
    except TaskMetricError as exc:
        problems.extend(exc.problems)
    if problems:
        raise TaskMetricError(problems)
    metric = Metric(name, task_set, direction, needs, fn)
    _REGISTRY[name] = metric
    return metric


def get_metric(name: str) -> Metric:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise TaskMetricError(
            f"unknown metric '{name}'; registered metrics: {sorted(_REGISTRY)}"
        ) from None


def list_metrics(task: Task | str | None = None) -> list[str]:
    if task is None:
        return sorted(_REGISTRY)
    t = Task(task)
    return sorted(n for n, m in _REGISTRY.items() if t in m.tasks)


def _log_loss(y_true: Any, y_pred: Any) -> float:
    y_pred = np.asarray(y_pred)
    labels = np.unique(y_true) if y_pred.ndim == 2 else None
    return skm.log_loss(y_true, y_pred, labels=labels)


def _f1(y_true: Any, y_pred: Any) -> float:
    return skm.f1_score(y_true, y_pred, pos_label=sorted(np.unique(y_true))[-1])


def _nonneg(name: str, y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    t, p = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if (t < 0).any() or (p < 0).any():
        raise TaskMetricError(f"metric '{name}' requires non-negative targets and predictions")
    return t, p


def _rmsle(y_true: Any, y_pred: Any) -> float:
    t, p = _nonneg("rmsle", y_true, y_pred)
    return float(np.sqrt(skm.mean_squared_error(np.log1p(t), np.log1p(p))))


def _rmse(y_true: Any, y_pred: Any) -> float:
    return float(np.sqrt(skm.mean_squared_error(y_true, y_pred)))


def _register_builtins() -> None:
    b, m, r = Task.BINARY, Task.MULTICLASS, Task.REGRESSION
    specs: list[tuple[str, Callable[[Any, Any], float], tuple[Task, ...], Direction, Needs]] = [
        ("roc_auc", skm.roc_auc_score, (b,), "maximize", "proba"),
        ("log_loss", _log_loss, (b, m), "minimize", "proba"),
        ("accuracy", skm.accuracy_score, (b, m), "maximize", "labels"),
        ("f1", _f1, (b,), "maximize", "labels"),
        ("macro_f1", lambda t, p: skm.f1_score(t, p, average="macro"), (m,), "maximize", "labels"),
        ("balanced_accuracy", skm.balanced_accuracy_score, (m,), "maximize", "labels"),
        ("rmse", _rmse, (r,), "minimize", "values"),
        ("mae", skm.mean_absolute_error, (r,), "minimize", "values"),
        ("r2", skm.r2_score, (r,), "maximize", "values"),
        ("rmsle", _rmsle, (r,), "minimize", "values"),
    ]
    for name, fn, tasks, direction, needs in specs:
        register_metric(name, fn, tasks, direction, needs)
        _BUILTIN.add(name)


_register_builtins()
