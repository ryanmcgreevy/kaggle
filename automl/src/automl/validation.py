"""Reproducible holdout and k-fold validation splits."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit

from automl.errors import ValidationSplitError
from automl.metrics import Task

STRATEGIES = ("holdout", "kfold")
ARTIFACT_VERSION = 1


@dataclass(frozen=True)
class ValidationConfig:
    strategy: str = "kfold"
    seed: int = 42
    n_folds: int = 5
    holdout_fraction: float = 0.2

    def __post_init__(self) -> None:
        problems: list[str] = []
        if self.strategy not in STRATEGIES:
            problems.append(f"setting 'strategy' has unsupported value {self.strategy!r}; valid: {list(STRATEGIES)}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            problems.append("setting 'seed' must be an integer")
        if isinstance(self.n_folds, bool) or not isinstance(self.n_folds, int):
            problems.append("setting 'n_folds' must be an integer")
        elif self.n_folds < 2:
            problems.append(f"setting 'n_folds' must be at least 2, got {self.n_folds}")
        frac = self.holdout_fraction
        if isinstance(frac, bool) or not isinstance(frac, (int, float)):
            problems.append("setting 'holdout_fraction' must be a number")
        elif not 0 < frac < 1:
            problems.append(f"setting 'holdout_fraction' must be between 0 and 1 (exclusive), got {frac}")
        if problems:
            raise ValidationSplitError(problems)


_CONFIG_KEYS = {f.name for f in fields(ValidationConfig)}


@dataclass(frozen=True, eq=False)
class Split:
    fold: int
    train_idx: np.ndarray
    valid_idx: np.ndarray

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Split)
            and self.fold == other.fold
            and np.array_equal(self.train_idx, other.train_idx)
            and np.array_equal(self.valid_idx, other.valid_idx)
        )


@dataclass(frozen=True)
class SplitResult:
    config: ValidationConfig
    task: Task
    stratified: bool
    n_rows: int
    splits: tuple[Split, ...]


def _holdout_sizes(n_rows: int, fraction: float) -> tuple[int, int]:
    n_valid = math.ceil(fraction * n_rows)
    return n_rows - n_valid, n_valid


def _check_inputs(target: pd.Series, task: Task, cfg: ValidationConfig) -> None:
    problems: list[str] = []
    n = len(target)
    nulls = int(target.isna().sum())
    if nulls:
        problems.append(f"target has {nulls} null value(s); remove or fix them before splitting")
    if cfg.strategy == "kfold":
        if cfg.n_folds > n:
            problems.append(f"setting 'n_folds' ({cfg.n_folds}) exceeds the number of rows ({n})")
        needed = cfg.n_folds
        fix = "use fewer 'n_folds' or merge/remove rare classes"
    else:
        n_train, n_valid = _holdout_sizes(n, cfg.holdout_fraction)
        if n_train < 1 or n_valid < 1:
            problems.append(
                f"setting 'holdout_fraction' ({cfg.holdout_fraction}) leaves {n_train} train and "
                f"{n_valid} validation row(s) of {n}; both must be at least 1"
            )
        needed = 2
        fix = "increase the data, adjust 'holdout_fraction', or merge/remove rare classes"
    if task is not Task.REGRESSION and not nulls:
        counts = target.value_counts()
        for cls, count in counts.items():
            if count < needed:
                problems.append(
                    f"class {cls!r} has {count} member(s) but stratified {cfg.strategy} needs at least {needed}; {fix}"
                )
        if cfg.strategy == "holdout" and not problems:
            k = len(counts)
            if n_valid < k or n_train < k:
                problems.append(
                    f"setting 'holdout_fraction' ({cfg.holdout_fraction}) gives {n_valid} validation and "
                    f"{n_train} train row(s) for {k} classes; each side needs at least one row per class"
                )
    if problems:
        raise ValidationSplitError(problems)


def make_splits(
    target: pd.Series,
    resolved_task: Any,
    config: ValidationConfig | None = None,
) -> SplitResult:
    """Build seeded positional index splits from the train target; the target is not modified.

    `resolved_task` is a `ResolvedTask` or `Task`. Classification is stratified; regression is not.
    Invalid fold/class combinations raise `ValidationSplitError` instead of falling back.
    """
    cfg = config or ValidationConfig()
    task = getattr(resolved_task, "task", resolved_task)
    task = Task(task)
    _check_inputs(target, task, cfg)
    y = target.to_numpy()
    x = np.zeros(len(y))
    stratified = task is not Task.REGRESSION
    if cfg.strategy == "kfold":
        splitter = (
            StratifiedKFold(cfg.n_folds, shuffle=True, random_state=cfg.seed)
            if stratified
            else KFold(cfg.n_folds, shuffle=True, random_state=cfg.seed)
        )
    else:
        cls = StratifiedShuffleSplit if stratified else ShuffleSplit
        splitter = cls(n_splits=1, test_size=cfg.holdout_fraction, random_state=cfg.seed)
    splits = tuple(
        Split(i, np.sort(tr).astype(np.int64), np.sort(va).astype(np.int64))
        for i, (tr, va) in enumerate(splitter.split(x, y))
    )
    return SplitResult(cfg, task, stratified, len(y), splits)


def save_splits(result: SplitResult, path: str | Path) -> None:
    """Write the split artifact as JSON; refuses to overwrite an existing file."""
    path = Path(path)
    payload = {
        "schema_version": ARTIFACT_VERSION,
        "strategy": result.config.strategy,
        "seed": result.config.seed,
        "n_folds": result.config.n_folds,
        "holdout_fraction": result.config.holdout_fraction,
        "task": result.task.value,
        "stratified": result.stratified,
        "n_rows": result.n_rows,
        "splits": [
            {
                "fold": s.fold,
                "n_train": len(s.train_idx),
                "n_valid": len(s.valid_idx),
                "train_idx": s.train_idx.tolist(),
                "valid_idx": s.valid_idx.tolist(),
            }
            for s in result.splits
        ],
    }
    try:
        with path.open("x") as f:
            json.dump(payload, f)
    except FileExistsError:
        raise ValidationSplitError(f"split artifact {path} already exists; choose a new path") from None


def load_splits(path: str | Path) -> SplitResult:
    """Read and validate a split artifact written by `save_splits`."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        raise ValidationSplitError(f"split artifact not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValidationSplitError(f"split artifact {path} is not valid JSON: {exc}") from None

    def bad(msg: str) -> ValidationSplitError:
        return ValidationSplitError(f"split artifact {path}: {msg}")

    if not isinstance(raw, dict) or raw.get("schema_version") != ARTIFACT_VERSION:
        raise bad(f"unsupported or missing 'schema_version'; expected {ARTIFACT_VERSION}")
    try:
        config = ValidationConfig(
            raw["strategy"], raw["seed"], raw["n_folds"], raw["holdout_fraction"]
        )
        task = Task(raw["task"])
        stratified, n_rows, entries = raw["stratified"], raw["n_rows"], raw["splits"]
    except (KeyError, ValueError) as exc:
        raise bad(f"missing or invalid field ({exc})") from None
    if isinstance(n_rows, bool) or not isinstance(n_rows, int) or not isinstance(stratified, bool):
        raise bad("'n_rows' must be an integer and 'stratified' a boolean")
    expected = config.n_folds if config.strategy == "kfold" else 1
    if not isinstance(entries, list) or len(entries) != expected:
        raise bad(f"expected {expected} split(s) for strategy {config.strategy!r}")
    splits = []
    for i, e in enumerate(entries):
        try:
            tr, va = e["train_idx"], e["valid_idx"]
            fold = e["fold"]
        except (KeyError, TypeError):
            raise bad(f"split {i} must have 'fold', 'train_idx', and 'valid_idx'") from None
        if fold != i:
            raise bad(f"split {i} has fold number {fold!r}")
        for name, idx in (("train_idx", tr), ("valid_idx", va)):
            if not isinstance(idx, list) or not all(
                isinstance(v, int) and not isinstance(v, bool) and 0 <= v < n_rows for v in idx
            ):
                raise bad(f"split {i} '{name}' must be a list of integers in [0, {n_rows})")
        if set(tr) & set(va):
            raise bad(f"split {i} has overlapping train and validation indices")
        if len(set(tr)) != len(tr) or len(set(va)) != len(va):
            raise bad(f"split {i} has duplicate indices")
        if len(tr) + len(va) != n_rows:
            raise bad(f"split {i} covers {len(tr) + len(va)} rows but 'n_rows' is {n_rows}")
        if e.get("n_train") != len(tr) or e.get("n_valid") != len(va):
            raise bad(f"split {i} 'n_train'/'n_valid' do not match its index lists")
        splits.append(Split(i, np.array(tr, dtype=np.int64), np.array(va, dtype=np.int64)))
    if config.strategy == "kfold":
        seen = sorted(i for s in splits for i in s.valid_idx.tolist())
        if seen != list(range(n_rows)):
            raise bad("k-fold validation sets must cover every row exactly once")
    return SplitResult(config, task, stratified, n_rows, tuple(splits))
