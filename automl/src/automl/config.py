"""Run configuration for the input data contract."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from automl.errors import DataContractError

CONFIG_VERSION = 1
_DATA_KEYS = {"train", "test", "sample_submission", "target", "id_columns"}
_TOP_KEYS = {"version", "seed", "data"}


@dataclass(frozen=True)
class DataConfig:
    train_path: Path
    test_path: Path
    target: str
    sample_submission_path: Path | None = None
    id_columns: tuple[str, ...] = ()
    seed: int = 0

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
    valid_overrides = {f.name for f in fields(DataConfig)}
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
    values.update(overrides)
    return DataConfig(**values)


def _override_name(key: str) -> str:
    return {"train": "train_path", "test": "test_path", "target": "target"}[key]

