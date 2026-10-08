"""Load and validate train/test/sample-submission CSV files."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from automl.config import DataConfig
from automl.errors import DataContractError


@dataclass(frozen=True)
class DataBundle:
    config: DataConfig
    train: pd.DataFrame
    test: pd.DataFrame
    sample_submission: pd.DataFrame | None = None


def _read(path: Path, label: str, problems: list[str]) -> pd.DataFrame | None:
    if not path.is_file():
        problems.append(f"{label} file not found: {path}")
        return None
    try:
        with path.open(newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), None)
    except UnicodeDecodeError:
        problems.append(f"{label} file {path} is not valid UTF-8")
        return None
    if not header:
        problems.append(f"{label} file {path} is empty")
        return None
    seen: set[str] = set()
    dupes = sorted({c for c in header if c in seen or seen.add(c)})
    if dupes:
        problems.append(f"{label} file {path} has duplicate columns: {dupes}")
        return None
    try:
        df = pd.read_csv(path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        problems.append(f"{label} file {path} could not be parsed: {exc}")
        return None
    if label != "sample submission" and df.empty:
        problems.append(f"{label} file {path} has a header but no rows")
        return None
    return df


def load_data(config: DataConfig) -> DataBundle:
    """Read the configured CSVs and check them against the data contract."""
    problems: list[str] = []
    train = _read(config.train_path, "train", problems)
    test = _read(config.test_path, "test", problems)
    sample = None
    if config.sample_submission_path is not None:
        sample = _read(config.sample_submission_path, "sample submission", problems)

    if train is not None:
        if config.target not in train.columns:
            problems.append(f"target column '{config.target}' not found in train file {config.train_path}")
        for col in config.id_columns:
            if col not in train.columns:
                problems.append(f"id column '{col}' not found in train file {config.train_path}")
    if test is not None:
        if config.target in test.columns:
            problems.append(f"target column '{config.target}' must not be present in test file {config.test_path}")
        for col in config.id_columns:
            if col not in test.columns:
                problems.append(f"id column '{col}' not found in test file {config.test_path}")

    if train is not None and test is not None:
        train_features = set(train.columns) - {config.target}
        test_features = set(test.columns) - {config.target}
        missing = sorted(train_features - test_features)
        extra = sorted(test_features - train_features)
        if missing or extra:
            problems.append(
                f"train/test column mismatch: columns missing from test {missing}; "
                f"columns only in test {extra}"
            )

    if sample is not None and test is not None:
        path = config.sample_submission_path
        if len(sample) != len(test):
            problems.append(
                f"sample submission {path} has {len(sample)} rows but test file has {len(test)}"
            )
        for col in config.id_columns:
            if col not in sample.columns:
                problems.append(f"id column '{col}' not found in sample submission {path}")
            elif col in test.columns and set(sample[col]) != set(test[col]):
                problems.append(f"id column '{col}' values differ between sample submission {path} and test file")

    if problems:
        raise DataContractError(problems)
    assert train is not None and test is not None
    return DataBundle(config=config, train=train, test=test, sample_submission=sample)
