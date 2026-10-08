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
