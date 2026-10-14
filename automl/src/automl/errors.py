"""Errors raised for invalid inputs."""

from __future__ import annotations


class DataContractError(ValueError):
    """Input files or run configuration violate the data contract."""

    _header = "Data contract violations:"

    def __init__(self, problems: str | list[str]) -> None:
        self.problems = [problems] if isinstance(problems, str) else list(problems)
        if len(self.problems) == 1:
            message = self.problems[0]
        else:
            message = self._header + "\n" + "\n".join(f"  - {p}" for p in self.problems)
        super().__init__(message)


class TaskMetricError(DataContractError):
    """Task or metric selection is missing, ambiguous, unsupported, or inconsistent with the data."""

    _header = "Task/metric problems:"


class ValidationSplitError(DataContractError):
    """Validation split settings are invalid or incompatible with the target."""

    _header = "Validation split problems:"


class EdaError(DataContractError):
    """EDA settings or inputs are invalid."""

    _header = "EDA problems:"


class PreprocessingError(DataContractError):
    """Preprocessing settings, columns, or plan artifacts are invalid."""

    _header = "Preprocessing problems:"


class BaselineError(DataContractError):
    """Baseline settings, inputs, or result artifacts are invalid."""

    _header = "Baseline problems:"


class EstimatorUnavailableError(BaselineError):
    """A selected estimator's library or device is unavailable."""


class TuningError(DataContractError):
    """Tuning settings, search space, inputs, or study artifacts are invalid."""

    _header = "Tuning problems:"


class TuningUnavailableError(TuningError):
    """The optional Optuna dependency is not installed."""


class ReportError(DataContractError):
    """Report settings, inputs, or output location are invalid."""

    _header = "Report problems:"
