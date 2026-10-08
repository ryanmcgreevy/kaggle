"""Run configuration for the input data contract."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from automl.eda import EDA_KEYS, EdaConfig
from automl.errors import DataContractError
from automl.report import REPORT_KEYS, ReportConfig
from automl.task import TaskConfig
from automl.validation import ValidationConfig

CONFIG_VERSION = 1
_DATA_KEYS = {"train", "test", "sample_submission", "target", "id_columns"}
_TASK_KEYS = {"task", "metric"}
_VALIDATION_KEYS = {"strategy", "seed", "n_folds", "holdout_fraction"}
_VALIDATION_OVERRIDES = {"strategy", "n_folds", "holdout_fraction"}  # 'seed' override stays the run seed
_TOP_KEYS = {"version", "seed", "data", "task", "validation", "eda", "report"}


@dataclass(frozen=True)
class DataConfig:
    train_path: Path
    test_path: Path
    target: str
    sample_submission_path: Path | None = None
    id_columns: tuple[str, ...] = ()
    seed: int = 0
    task_config: TaskConfig = field(default_factory=TaskConfig)
    validation_config: ValidationConfig = field(default_factory=ValidationConfig)
    eda_config: EdaConfig = field(default_factory=EdaConfig)
    report_config: ReportConfig = field(default_factory=ReportConfig)

    def __post_init__(self) -> None:
        problems: list[str] = []
        if not isinstance(self.target, str) or not self.target:
            problems.append("setting 'target' must be a non-empty column name")
        ids = tuple(self.id_columns)
        if any(not isinstance(c, str) or not c for c in ids):
            problems.append("setting 'id_columns' must contain non-empty column names")
        if len(set(ids)) != len(ids):
            problems.append(f"setting 'id_columns' contains repeated names: {list(ids)}")
        if self.target in ids:
            problems.append(f"target column '{self.target}' cannot also be an id column")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            problems.append("setting 'seed' must be an integer")
        if problems:
            raise DataContractError(problems)
        object.__setattr__(self, "id_columns", ids)
        object.__setattr__(self, "train_path", Path(self.train_path))
        object.__setattr__(self, "test_path", Path(self.test_path))
        if self.sample_submission_path is not None:
            object.__setattr__(self, "sample_submission_path", Path(self.sample_submission_path))


def load_config(path: str | Path, **overrides: Any) -> DataConfig:
    """Read a TOML run config; keyword overrides (DataConfig field names) take precedence."""
    path = Path(path)
    try:
        with path.open("rb") as f:
            raw = tomllib.load(f)
    except FileNotFoundError:
        raise DataContractError(f"config file not found: {path}") from None
    except tomllib.TOMLDecodeError as exc:
        raise DataContractError(f"config file {path} is not valid TOML: {exc}") from None

    problems: list[str] = []
    unknown = set(raw) - _TOP_KEYS
    if unknown:
        problems.append(f"config file {path}: unknown top-level keys {sorted(unknown)}")
    if raw.get("version") != CONFIG_VERSION:
        problems.append(
            f"config file {path}: unsupported or missing 'version' {raw.get('version')!r}; "
            f"expected {CONFIG_VERSION}"
        )
    data = raw.get("data")
    if not isinstance(data, dict):
        problems.append(f"config file {path}: missing [data] table")
        data = {}
    unknown = set(data) - _DATA_KEYS
    if unknown:
        problems.append(f"config file {path}: unknown keys in [data]: {sorted(unknown)}")
    for key in ("train", "test", "target"):
        if key not in data and _override_name(key) not in overrides:
            problems.append(f"config file {path}: [data] is missing required key '{key}'")
    for key in ("train", "test", "sample_submission"):
        if key in data and not isinstance(data[key], str):
            problems.append(f"config file {path}: [data] '{key}' must be a path string")
    ids = data.get("id_columns", [])
    if not isinstance(ids, list) or not all(isinstance(c, str) for c in ids):
        problems.append(f"config file {path}: [data] 'id_columns' must be a list of strings")
    task_tbl = raw.get("task", {})
    if not isinstance(task_tbl, dict):
        problems.append(f"config file {path}: [task] must be a table")
        task_tbl = {}
    unknown = set(task_tbl) - _TASK_KEYS
    if unknown:
        problems.append(f"config file {path}: unknown keys in [task]: {sorted(unknown)}")
    for key in _TASK_KEYS & set(task_tbl):
        if not isinstance(task_tbl[key], str):
            problems.append(f"config file {path}: [task] '{key}' must be a string")
    val_tbl = raw.get("validation", {})
    if not isinstance(val_tbl, dict):
        problems.append(f"config file {path}: [validation] must be a table")
        val_tbl = {}
    unknown = set(val_tbl) - _VALIDATION_KEYS
    if unknown:
        problems.append(f"config file {path}: unknown keys in [validation]: {sorted(unknown)}")
    eda_tbl = raw.get("eda", {})
    if not isinstance(eda_tbl, dict):
        problems.append(f"config file {path}: [eda] must be a table")
        eda_tbl = {}
    unknown = set(eda_tbl) - EDA_KEYS
    if unknown:
        problems.append(f"config file {path}: unknown keys in [eda]: {sorted(unknown)}")
    report_tbl = raw.get("report", {})
    if not isinstance(report_tbl, dict):
        problems.append(f"config file {path}: [report] must be a table")
        report_tbl = {}
    unknown = set(report_tbl) - REPORT_KEYS
    if unknown:
        problems.append(f"config file {path}: unknown keys in [report]: {sorted(unknown)}")
    valid_overrides = (
        {f.name for f in fields(DataConfig)} | _TASK_KEYS | _VALIDATION_OVERRIDES | EDA_KEYS | REPORT_KEYS
    )
    bad = set(overrides) - valid_overrides
    if bad:
        problems.append(f"unknown config overrides: {sorted(bad)}")
    if problems:
        raise DataContractError(problems)

    base = path.parent

    def resolve(value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else base / p

    values: dict[str, Any] = {}
    if "train" in data:
        values["train_path"] = resolve(data["train"])
    if "test" in data:
        values["test_path"] = resolve(data["test"])
    if "sample_submission" in data:
        values["sample_submission_path"] = resolve(data["sample_submission"])
    if "target" in data:
        values["target"] = data["target"]
    if "id_columns" in data:
        values["id_columns"] = tuple(data["id_columns"])
    if "seed" in raw:
        values["seed"] = raw["seed"]
    task_values = {k: task_tbl[k] for k in _TASK_KEYS if k in task_tbl}
    task_values.update({k: overrides.pop(k) for k in _TASK_KEYS if k in overrides})
    if "task_config" not in overrides:
        values["task_config"] = TaskConfig(**task_values)
    val_values = {k: v for k, v in val_tbl.items() if k in _VALIDATION_KEYS}
    val_values.update({k: overrides.pop(k) for k in _VALIDATION_OVERRIDES if k in overrides})
    if "validation_config" not in overrides:
        try:
            values["validation_config"] = ValidationConfig(**val_values)
        except DataContractError as exc:
            raise DataContractError([f"config file {path}: {p}" for p in exc.problems]) from None
    eda_values = {k: v for k, v in eda_tbl.items() if k in EDA_KEYS}
    eda_values.update({k: overrides.pop(k) for k in EDA_KEYS if k in overrides})
    if "eda_config" not in overrides:
        try:
            values["eda_config"] = EdaConfig(**eda_values)
        except DataContractError as exc:
            raise DataContractError([f"config file {path}: {p}" for p in exc.problems]) from None
    report_values = {k: v for k, v in report_tbl.items() if k in REPORT_KEYS}
    report_values.update({k: overrides.pop(k) for k in REPORT_KEYS if k in overrides})
    if "report_config" not in overrides:
        try:
            values["report_config"] = ReportConfig(**report_values)
        except DataContractError as exc:
            raise DataContractError([f"config file {path}: {p}" for p in exc.problems]) from None
    values.update(overrides)
    return DataConfig(**values)


def _override_name(key: str) -> str:
    return {"train": "train_path", "test": "test_path", "target": "target"}[key]

