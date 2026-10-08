"""Local-first tools for tabular Kaggle competitions."""

from automl.config import DataConfig, load_config
from automl.errors import DataContractError, TaskMetricError, ValidationSplitError
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
from automl.validation import Split, SplitResult, ValidationConfig, load_splits, make_splits, save_splits

__all__ = [
    "DataBundle",
    "DataConfig",
    "DataContractError",
    "Metric",
    "ResolvedTask",
    "Split",
    "SplitResult",
    "Suggestion",
    "Task",
    "TaskConfig",
    "TaskMetricError",
    "ValidationConfig",
    "ValidationSplitError",
    "get_metric",
    "list_metrics",
    "load_config",
    "load_data",
    "load_splits",
    "make_splits",
    "register_metric",
    "resolve_task_metric",
    "save_splits",
    "suggest_metric",
    "suggest_task",
]
