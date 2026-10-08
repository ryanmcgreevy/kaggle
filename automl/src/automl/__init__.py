"""Local-first tools for tabular Kaggle competitions."""

from automl.config import DataConfig, load_config
from automl.eda import EdaConfig, EdaResult, load_eda, save_eda, summarize, summarize_bundle
from automl.errors import DataContractError, EdaError, ReportError, TaskMetricError, ValidationSplitError
from automl.loader import DataBundle, load_data
from automl.metrics import Metric, Task, get_metric, list_metrics, register_metric
from automl.report import Finding, ReportConfig, ReportResult, build_findings, write_eda_report
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
    "EdaConfig",
    "EdaError",
    "EdaResult",
    "Finding",
    "Metric",
    "ReportConfig",
    "ReportError",
    "ReportResult",
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
    "load_eda",
    "load_splits",
    "make_splits",
    "register_metric",
    "resolve_task_metric",
    "save_eda",
    "save_splits",
    "build_findings",
    "suggest_metric",
    "suggest_task",
    "summarize",
    "summarize_bundle",
    "write_eda_report",
]
