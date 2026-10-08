"""Non-mutating exploratory data analysis summary with a persisted JSON artifact."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from automl.errors import EdaError
from automl.metrics import Task

SCHEMA_VERSION = 1
OUTLIER_METHODS = ("iqr", "zscore", "both")
_PERCENTILES = {"p05": 0.05, "p25": 0.25, "p50": 0.5, "p75": 0.75, "p95": 0.95}


@dataclass(frozen=True)
class EdaConfig:
    outlier_method: str = "both"
    iqr_multiplier: float = 1.5
    zscore_threshold: float = 3.0
    histogram_bins: int = 10
    top_n: int = 10
    id_like_threshold: float = 0.95
    drift_alpha: float = 0.05

    def __post_init__(self) -> None:
        problems: list[str] = []

        def number(name: str) -> bool:
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, Real):
                problems.append(f"setting '{name}' must be a number")
                return False
            return True

        def integer(name: str) -> bool:
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int):
                problems.append(f"setting '{name}' must be an integer")
                return False
            return True

        if self.outlier_method not in OUTLIER_METHODS:
            problems.append(
                f"setting 'outlier_method' has unsupported value {self.outlier_method!r}; valid: {list(OUTLIER_METHODS)}"
            )
        for name in ("iqr_multiplier", "zscore_threshold"):
            if number(name) and not getattr(self, name) > 0:
                problems.append(f"setting '{name}' must be greater than 0, got {getattr(self, name)}")
        for name in ("histogram_bins", "top_n"):
            if integer(name) and getattr(self, name) < 1:
                problems.append(f"setting '{name}' must be at least 1, got {getattr(self, name)}")
        if number("id_like_threshold") and not 0 < self.id_like_threshold <= 1:
            problems.append(f"setting 'id_like_threshold' must be in (0, 1], got {self.id_like_threshold}")
        if number("drift_alpha") and not 0 < self.drift_alpha < 1:
            problems.append(f"setting 'drift_alpha' must be between 0 and 1 (exclusive), got {self.drift_alpha}")
        if problems:
            raise EdaError(problems)


EDA_KEYS = {f.name for f in fields(EdaConfig)}


@dataclass(frozen=True)
class NumericSummary:
    count: int
    non_finite_count: int
    mean: float | None
    std: float | None
    min: float | None
    p05: float | None
    p25: float | None
    p50: float | None
    p75: float | None
    p95: float | None
    max: float | None
    skew: float | None
    histogram: dict[str, list] | None
    outliers: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class CategoricalSummary:
    top_values: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ColumnSummary:
    name: str
    dtype: str
    kind: str
    null_count: int
    null_fraction: float | None
    unique: int
    constant: bool
    all_null: bool
    id_like: bool
    numeric: NumericSummary | None = None
    categorical: CategoricalSummary | None = None


@dataclass(frozen=True)
class DatasetSummary:
    n_rows: int
    n_columns: int
    memory_bytes: int
    dtypes: dict[str, str]
    duplicate_rows: int
    duplicate_row_fraction: float | None
    duplicate_ids: int | None


@dataclass(frozen=True)
class TargetSummary:
    name: str
    dtype: str
    null_count: int
    unique: int
    task: str | None = None
    class_counts: dict[str, int] | None = None
    class_fractions: dict[str, float] | None = None
    imbalance_ratio: float | None = None
    numeric: NumericSummary | None = None


@dataclass(frozen=True)
class ColumnComparison:
    name: str
    kind: str
    train_null_fraction: float | None
    test_null_fraction: float | None
    null_delta: float | None
    mean_shift: float | None = None
    std_shift: float | None = None
    test_name: str | None = None
    statistic: float | None = None
    pvalue: float | None = None
    drift: bool | None = None
    skipped_reason: str | None = None
    train_only_categories: int | None = None
    test_only_categories: int | None = None
    train_only_examples: tuple[str, ...] = ()
    test_only_examples: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrainTestSummary:
    train_rows: int
    test_rows: int
    only_in_train: tuple[str, ...]
    only_in_test: tuple[str, ...]
    columns: tuple[ColumnComparison, ...]


@dataclass(frozen=True)
class EdaResult:
    config: EdaConfig
    dataset: DatasetSummary
    columns: tuple[ColumnSummary, ...]
    target: TargetSummary | None = None
    train_test: TrainTestSummary | None = None
    schema_version: int = field(default=SCHEMA_VERSION)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict; NaN and infinity become None."""
        return _clean(asdict(self))


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _kind(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    if isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return "categorical"
    return "other"


def _frac(num: int, den: int) -> float | None:
    return num / den if den else None


def _finite(s: pd.Series) -> pd.Series:
    v = s.astype("float64")
    return v[np.isfinite(v)]


def _numeric_summary(s: pd.Series, cfg: EdaConfig) -> NumericSummary:
    nonnull = int(s.notna().sum())
    v = _finite(s)
    n = len(v)
    outliers: dict[str, dict[str, Any]] = {}
    if cfg.outlier_method in ("iqr", "both"):
        lower = upper = None
        count = 0
        if n:
            q1, q3 = v.quantile(0.25), v.quantile(0.75)
            lower, upper = q1 - cfg.iqr_multiplier * (q3 - q1), q3 + cfg.iqr_multiplier * (q3 - q1)
            count = int(((v < lower) | (v > upper)).sum())
        outliers["iqr"] = {"count": count, "fraction": _frac(count, n), "lower": lower, "upper": upper}
    if cfg.outlier_method in ("zscore", "both"):
        count = 0
        std = v.std() if n else float("nan")
        if n and std and math.isfinite(std):
            count = int((((v - v.mean()) / std).abs() > cfg.zscore_threshold).sum())
        outliers["zscore"] = {"count": count, "fraction": _frac(count, n), "threshold": cfg.zscore_threshold}
    histogram = None
    if n:
        counts, edges = np.histogram(v, bins=cfg.histogram_bins)
        histogram = {"edges": edges.tolist(), "counts": counts.tolist()}
    return NumericSummary(
        count=n,
        non_finite_count=nonnull - n,
        mean=v.mean() if n else None,
        std=v.std() if n else None,
        min=v.min() if n else None,
        **{k: (v.quantile(q) if n else None) for k, q in _PERCENTILES.items()},
        max=v.max() if n else None,
        skew=v.skew() if n else None,
        histogram=histogram,
        outliers=outliers,
    )


def _column_summary(s: pd.Series, cfg: EdaConfig) -> ColumnSummary:
    kind = _kind(s)
    nonnull = int(s.notna().sum())
    unique = int(s.nunique(dropna=True))
    id_like = bool(
        nonnull
        and (kind == "categorical" or (kind == "numeric" and pd.api.types.is_integer_dtype(s)))
        and unique / nonnull >= cfg.id_like_threshold
    )
    numeric = categorical = None
    if kind == "numeric":
        numeric = _numeric_summary(s, cfg)
    elif kind in ("categorical", "boolean"):
        counts = s.dropna().astype(str).value_counts().head(cfg.top_n)
        categorical = CategoricalSummary(
            tuple({"value": k, "count": int(c), "fraction": c / nonnull} for k, c in counts.items())
        )
    return ColumnSummary(
        name=str(s.name),
        dtype=str(s.dtype),
        kind=kind,
        null_count=len(s) - nonnull,
        null_fraction=_frac(len(s) - nonnull, len(s)),
        unique=unique,
        constant=bool(nonnull and unique <= 1),
        all_null=nonnull == 0,
        id_like=id_like,
        numeric=numeric,
        categorical=categorical,
    )


def _target_summary(
    s: pd.Series, task: Task | None, cfg: EdaConfig
) -> TargetSummary:
    base = {
        "name": str(s.name),
        "dtype": str(s.dtype),
        "null_count": int(s.isna().sum()),
        "unique": int(s.nunique(dropna=True)),
    }
    if task is None:
        return TargetSummary(**base)
    if task is Task.REGRESSION:
        numeric = _numeric_summary(s, cfg) if _kind(s) == "numeric" else None
        return TargetSummary(**base, task=task.value, numeric=numeric)
    counts = s.dropna().value_counts()
    total = int(counts.sum())
    return TargetSummary(
        **base,
        task=task.value,
        class_counts={str(k): int(c) for k, c in counts.items()},
        class_fractions={str(k): c / total for k, c in counts.items()},
        imbalance_ratio=float(counts.max() / counts.min()) if len(counts) else None,
    )


def _compare_column(train: pd.Series, test: pd.Series, summary: ColumnSummary, cfg: EdaConfig) -> ColumnComparison:
    train_null = _frac(int(train.isna().sum()), len(train))
    test_null = _frac(int(test.isna().sum()), len(test))
    base = {
        "name": summary.name,
        "kind": summary.kind,
        "train_null_fraction": train_null,
        "test_null_fraction": test_null,
        "null_delta": None if train_null is None or test_null is None else test_null - train_null,
    }
    test_kind = _kind(test)
    if summary.kind == "other":
        return ColumnComparison(**base, skipped_reason="kind 'other' is not compared")
    if test_kind != summary.kind:
        return ColumnComparison(
            **base, skipped_reason=f"dtype kind differs (train {summary.kind}, test {test_kind})"
        )
    if summary.id_like:
        return ColumnComparison(**base, skipped_reason="id-like column")

    if summary.kind == "numeric":
        a, b = _finite(train), _finite(test)
        extra: dict[str, Any] = {}
        if len(a) and len(b):
            std = a.std()
            if std and math.isfinite(std):
                extra["mean_shift"] = (b.mean() - a.mean()) / std
                if len(b) > 1:
                    extra["std_shift"] = (b.std() - std) / std
        if len(a) < 2 or len(b) < 2:
            return ColumnComparison(**base, **extra, skipped_reason="fewer than 2 non-null values on one side")
        if pd.concat([a, b]).nunique() < 2:
            return ColumnComparison(**base, **extra, skipped_reason="constant column")
        res = stats.ks_2samp(a, b)
        return ColumnComparison(
            **base, **extra, test_name="ks", statistic=float(res.statistic), pvalue=float(res.pvalue),
            drift=bool(res.pvalue < cfg.drift_alpha),
        )

    a, b = train.dropna().astype(str), test.dropna().astype(str)
    ta, tb = set(a), set(b)
    extra = {
        "train_only_categories": len(ta - tb),
        "test_only_categories": len(tb - ta),
        "train_only_examples": tuple(sorted(ta - tb)[: cfg.top_n]),
        "test_only_examples": tuple(sorted(tb - ta)[: cfg.top_n]),
    }
    if not len(a) or not len(b):
        return ColumnComparison(**base, **extra, skipped_reason="no non-null values on one side")
    cats = sorted(ta | tb)
    if len(cats) < 2:
        return ColumnComparison(**base, **extra, skipped_reason="constant column")
    ca, cb = a.value_counts(), b.value_counts()
    chi2, p, *_ = stats.chi2_contingency([[ca.get(c, 0) for c in cats], [cb.get(c, 0) for c in cats]])
    return ColumnComparison(
        **base, **extra, test_name="chi2", statistic=float(chi2), pvalue=float(p),
        drift=bool(p < cfg.drift_alpha),
    )


def summarize(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    *,
    target: str | None = None,
    id_columns: tuple[str, ...] | list[str] = (),
    task: Any = None,
    config: EdaConfig | None = None,
) -> EdaResult:
    """Summarize train (and optionally test) data; inputs are not modified.

    `task` is a `ResolvedTask`, `Task`, or task name; it is never inferred here.
    """
    cfg = config or EdaConfig()
    ids = tuple(id_columns)
    problems: list[str] = []
    if target is not None and target not in train.columns:
        problems.append(f"target column '{target}' not found in train")
    problems += [f"id column '{c}' not found in train" for c in ids if c not in train.columns]
    if problems:
        raise EdaError(problems)
    resolved = None if task is None else Task(getattr(task, "task", task))

    dataset = DatasetSummary(
        n_rows=len(train),
        n_columns=train.shape[1],
        memory_bytes=int(train.memory_usage(deep=True).sum()),
        dtypes={str(c): str(t) for c, t in train.dtypes.items()},
        duplicate_rows=int(train.duplicated().sum()),
        duplicate_row_fraction=_frac(int(train.duplicated().sum()), len(train)),
        duplicate_ids=int(train.duplicated(subset=list(ids)).sum()) if ids else None,
    )
    skip = set(ids) | ({target} if target is not None else set())
    features = [c for c in train.columns if c not in skip]
    columns = tuple(_column_summary(train[c], cfg) for c in features)
    target_summary = _target_summary(train[target], resolved, cfg) if target is not None else None

    train_test = None
    if test is not None:
        by_name = {c.name: c for c in columns}
        common = [c for c in features if c in test.columns]
        train_test = TrainTestSummary(
            train_rows=len(train),
            test_rows=len(test),
            only_in_train=tuple(str(c) for c in train.columns if c not in test.columns and c not in skip),
            only_in_test=tuple(str(c) for c in test.columns if c not in train.columns and c not in skip),
            columns=tuple(_compare_column(train[c], test[c], by_name[str(c)], cfg) for c in common),
        )
    return EdaResult(cfg, dataset, columns, target_summary, train_test)


def summarize_bundle(bundle: Any, task: Any = None, config: EdaConfig | None = None) -> EdaResult:
    """Summarize a `DataBundle` using its configured target and ID columns."""
    cfg = bundle.config
    return summarize(
        bundle.train,
        bundle.test,
        target=cfg.target,
        id_columns=cfg.id_columns,
        task=task,
        config=config or cfg.eda_config,
    )


def save_eda(result: EdaResult, path: str | Path) -> None:
    """Write the EDA artifact as JSON; refuses to overwrite an existing file."""
    path = Path(path)
    text = json.dumps(result.to_dict(), allow_nan=False)
    try:
        with path.open("x") as f:
            f.write(text)
    except FileExistsError:
        raise EdaError(f"EDA artifact {path} already exists; choose a new path") from None


def load_eda(path: str | Path) -> dict[str, Any]:
    """Read and structurally validate an EDA artifact; returns the plain dict."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        raise EdaError(f"EDA artifact not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise EdaError(f"EDA artifact {path} is not valid JSON: {exc}") from None
    problems: list[str] = []
    if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA_VERSION:
        raise EdaError(f"EDA artifact {path}: unsupported or missing 'schema_version'; expected {SCHEMA_VERSION}")
    for key in ("config", "dataset", "columns", "target", "train_test"):
        if key not in raw:
            problems.append(f"EDA artifact {path}: missing section '{key}'")
    if not problems:
        if not isinstance(raw["dataset"], dict) or "n_rows" not in raw["dataset"]:
            problems.append(f"EDA artifact {path}: 'dataset' must include 'n_rows'")
        cols = raw["columns"]
        if not isinstance(cols, list) or not all(isinstance(c, dict) and {"name", "kind"} <= set(c) for c in cols):
            problems.append(f"EDA artifact {path}: 'columns' must be a list of objects with 'name' and 'kind'")
    if problems:
        raise EdaError(problems)
    return raw
