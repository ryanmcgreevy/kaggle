"""Task and metric configuration, suggestion, and resolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from automl.errors import TaskMetricError
from automl.metrics import Metric, Task, get_metric, list_metrics

DEFAULT_METRICS: dict[Task, str] = {
    Task.BINARY: "roc_auc",
    Task.MULTICLASS: "log_loss",
    Task.REGRESSION: "rmse",
}
_MAX_AMBIGUOUS_CLASSES = 10
_MAX_STRING_CLASSES = 50


@dataclass(frozen=True)
class TaskConfig:
    task: str | None = None
    metric: str | None = None

    def __post_init__(self) -> None:
        problems = [
            f"setting '{k}' must be a string"
            for k in ("task", "metric")
            if getattr(self, k) is not None and not isinstance(getattr(self, k), str)
        ]
        if problems:
            raise TaskMetricError(problems)


@dataclass(frozen=True)
class Suggestion:
    value: str
    rationale: str
    evidence: dict[str, Any] = field(default_factory=dict)
    ambiguous: bool = False

    def __str__(self) -> str:
        flag = " (ambiguous)" if self.ambiguous else ""
        return f"'{self.value}'{flag}: {self.rationale}"


@dataclass(frozen=True)
class ResolvedTask:
    task: Task
    metric: Metric

    @property
    def direction(self) -> str:
        return self.metric.direction


def suggest_task(target: pd.Series) -> Suggestion:
    """Propose a task from the target's dtype and unique count; never modifies the data."""
    nulls = int(target.isna().sum())
    values = target.dropna()
    n = int(values.nunique())
    evidence = {"dtype": str(target.dtype), "unique": n, "nulls": nulls}
    if n < 2:
        return Suggestion("binary", f"target has {n} distinct non-null value(s); cannot infer a task", evidence, True)
    numeric = pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values)
    if not numeric:
        task = Task.BINARY if n == 2 else Task.MULTICLASS
        return Suggestion(
            task.value,
            f"non-numeric target with {n} distinct values",
            evidence,
            n > _MAX_STRING_CLASSES,
        )
    integral = bool((values % 1 == 0).all())
    if not integral:
        return Suggestion("regression", "numeric target with non-integer values", evidence)
    if n == 2:
        return Suggestion("binary", "integer target with exactly 2 distinct values", evidence, True)
    if n <= _MAX_AMBIGUOUS_CLASSES:
        return Suggestion(
            "multiclass",
            f"integer target with {n} distinct values could be classes or a count/ordinal regression target",
            evidence,
            True,
        )
    return Suggestion(
        "regression",
        f"integer target with {n} distinct values; treated as numeric (could be many classes)",
        evidence,
        True,
    )


def suggest_metric(task: Task | str) -> Suggestion:
    t = Task(task)
    default = DEFAULT_METRICS[t]
    alternatives = [m for m in list_metrics(t) if m != default]
    return Suggestion(default, f"default for {t.value}; alternatives: {alternatives}")


def _check_target(task: Task, target: pd.Series, problems: list[str]) -> None:
    nulls = int(target.isna().sum())
    if nulls:
        problems.append(f"target has {nulls} null value(s); remove or fix them before choosing a task")
    values = target.dropna()
    n = int(values.nunique())
    if task is Task.BINARY and n != 2:
        problems.append(f"task 'binary' requires exactly 2 distinct target values, found {n}")
    elif task is Task.MULTICLASS and n < 3:
        problems.append(f"task 'multiclass' requires 3 or more distinct target values, found {n}")
    elif task is Task.REGRESSION and (
        not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values)
    ):
        problems.append(f"task 'regression' requires a numeric target, found dtype {target.dtype}")


def resolve_task_metric(data: Any, task_config: TaskConfig | None = None) -> ResolvedTask:
    """Validate explicit task/metric choices against the train target.

    `data` is a train target Series or a DataBundle. Unset values are never defaulted;
    the error carries a suggestion to confirm instead.
    """
    if hasattr(data, "train"):
        cfg = task_config if task_config is not None else data.config.task_config
        target = data.train[data.config.target]
    else:
        cfg = task_config or TaskConfig()
        target = data
    problems: list[str] = []

    task: Task | None = None
    if cfg.task is None:
        problems.append(f"setting 'task' is not set; confirm one of {[t.value for t in Task]}. Suggestion: {suggest_task(target)}")
    else:
        try:
            task = Task(cfg.task)
        except ValueError:
            problems.append(f"setting 'task' has unsupported value {cfg.task!r}; valid tasks: {[t.value for t in Task]}")

    metric: Metric | None = None
    if cfg.metric is None:
        if task is not None:
            problems.append(f"setting 'metric' is not set; confirm one of {list_metrics(task)}. Suggestion: {suggest_metric(task)}")
        else:
            problems.append("setting 'metric' is not set; choose it after confirming the task")
    else:
        try:
            metric = get_metric(cfg.metric)
        except TaskMetricError as exc:
            problems.extend(exc.problems)
        if metric is not None and task is not None and task not in metric.tasks:
            problems.append(
                f"metric '{metric.name}' does not support task '{task.value}'; valid metrics: {list_metrics(task)}"
            )

    if task is not None:
        _check_target(task, target, problems)
    if problems:
        raise TaskMetricError(problems)
    assert task is not None and metric is not None
    return ResolvedTask(task, metric)
