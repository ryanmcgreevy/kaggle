"""Local-first tools for tabular Kaggle competitions."""

from automl.config import DataConfig, load_config
from automl.errors import DataContractError, TaskMetricError
from automl.loader import DataBundle, load_data
from automl.metrics import Metric, Task, get_metric, list_metrics, register_metric
from automl.task import (
    ResolvedTask,
    Suggestion,
    TaskConfig,
    resolve_task_metric,
    suggest_metric,
    suggest_task,
)

__all__ = [
    "DataBundle",
    "DataConfig",
    "DataContractError",
    "Metric",
    "ResolvedTask",
    "Suggestion",
    "Task",
    "TaskConfig",
    "TaskMetricError",
    "get_metric",
    "list_metrics",
    "load_config",
    "load_data",
    "register_metric",
    "resolve_task_metric",
    "suggest_metric",
    "suggest_task",
]
