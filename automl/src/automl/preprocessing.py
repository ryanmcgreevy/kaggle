"""Leakage-safe numeric and categorical preprocessing built on scikit-learn."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)

from automl.errors import PreprocessingError
from automl.metrics import Task
from automl.validation import Split

NUMERIC_IMPUTERS = ("mean", "median", "constant")
CATEGORICAL_IMPUTERS = ("most_frequent", "constant")
ENCODERS = ("onehot", "ordinal", "target")
SCALERS = ("none", "standard", "robust")
ARTIFACT_VERSION = 1
_TARGET_TYPES = {Task.BINARY: "binary", Task.MULTICLASS: "multiclass", Task.REGRESSION: "continuous"}
_LIST_FIELDS = ("numeric_columns", "categorical_columns", "drop_columns")


@dataclass(frozen=True)
class PreprocessingConfig:
    numeric_imputer: str = "median"
    numeric_fill_value: float = 0
    categorical_imputer: str = "most_frequent"
    categorical_fill_value: str = "missing"
    encoder: str = "onehot"
    scaler: str = "none"
    max_cardinality: int = 20
    numeric_columns: tuple[str, ...] = ()
    categorical_columns: tuple[str, ...] = ()
    drop_columns: tuple[str, ...] = ()
    seed: int = 42

    def __post_init__(self) -> None:
        problems: list[str] = []
        for name, valid in (
            ("numeric_imputer", NUMERIC_IMPUTERS),
            ("categorical_imputer", CATEGORICAL_IMPUTERS),
            ("encoder", ENCODERS),
            ("scaler", SCALERS),
        ):
            value = getattr(self, name)
            if value not in valid:
                problems.append(f"setting '{name}' has unsupported value {value!r}; valid: {list(valid)}")
        fill = self.numeric_fill_value
        if isinstance(fill, bool) or not isinstance(fill, (int, float)):
            problems.append("setting 'numeric_fill_value' must be a number")
        if not isinstance(self.categorical_fill_value, str):
            problems.append("setting 'categorical_fill_value' must be a string")
        mc = self.max_cardinality
        if isinstance(mc, bool) or not isinstance(mc, int) or mc < 1:
            problems.append(f"setting 'max_cardinality' must be an integer of at least 1, got {mc!r}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            problems.append("setting 'seed' must be an integer")
        lists: dict[str, tuple[str, ...]] = {}
        for name in _LIST_FIELDS:
            value = getattr(self, name)
            if isinstance(value, str) or not isinstance(value, (list, tuple)) or not all(
                isinstance(c, str) and c for c in value
            ):
                problems.append(f"setting '{name}' must be a list of non-empty column names")
                continue
            lists[name] = tuple(value)
            if len(set(value)) != len(value):
                problems.append(f"setting '{name}' contains repeated names: {list(value)}")
        seen: dict[str, str] = {}
        for name, cols in lists.items():
            for col in cols:
                if col in seen and seen[col] != name:
                    problems.append(f"column '{col}' is listed in both '{seen[col]}' and '{name}'")
                seen.setdefault(col, name)
        if problems:
            raise PreprocessingError(problems)
        for name, cols in lists.items():
            object.__setattr__(self, name, cols)


PREPROCESSING_KEYS = {f.name for f in fields(PreprocessingConfig)}


@dataclass(frozen=True)
class ColumnPlan:
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    dropped: tuple[str, ...]
    excluded: tuple[str, ...]
    reasons: dict[str, str] = field(default_factory=dict)


def infer_columns(
    train: pd.DataFrame,
    target: str,
    id_columns: tuple[str, ...] = (),
    config: PreprocessingConfig | None = None,
) -> ColumnPlan:
    """Type the feature columns of `train`; the target and ID columns are always excluded."""
    cfg = config or PreprocessingConfig()
    ids = tuple(id_columns)
    excluded = tuple(dict.fromkeys((*ids, target)))
    problems = [f"column '{c}' (target or id) is not in the train data" for c in excluded if c not in train.columns]
    for name in _LIST_FIELDS:
        for col in getattr(cfg, name):
            if col in excluded:
                role = "target" if col == target else "id"
                problems.append(f"setting '{name}' names {role} column '{col}', which is never preprocessed")
            elif col not in train.columns:
                problems.append(f"setting '{name}' names column '{col}', which is not in the train data")
    if problems:
        raise PreprocessingError(problems)

    numeric: list[str] = []
    categorical: list[str] = []
    dropped: list[str] = []
    reasons: dict[str, str] = {}
    for col in train.columns:
        if col in excluded:
            continue
        if col in cfg.drop_columns:
            dropped.append(col)
            reasons[col] = "dropped by drop_columns"
        elif col in cfg.numeric_columns:
            numeric.append(col)
            reasons[col] = "numeric by override"
        elif col in cfg.categorical_columns:
            categorical.append(col)
            reasons[col] = "categorical by override"
        else:
            kind, reason = _infer_kind(train[col], cfg.max_cardinality)
            {"numeric": numeric, "categorical": categorical, "drop": dropped}[kind].append(col)
            reasons[col] = reason
    return ColumnPlan(tuple(numeric), tuple(categorical), tuple(dropped), excluded, reasons)


def _infer_kind(series: pd.Series, max_cardinality: int) -> tuple[str, str]:
    n_unique = int(series.nunique(dropna=True))
    if n_unique == 0:
        return "drop", "all values are null"
    if n_unique == 1 and not series.isna().any():
        return "drop", "constant column"
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return "categorical", "boolean dtype"
    if pd.api.types.is_integer_dtype(dtype):
        if n_unique > max_cardinality:
            return "numeric", f"integer with {n_unique} unique values (> {max_cardinality})"
        return "categorical", f"integer with {n_unique} unique values (<= {max_cardinality})"
    if pd.api.types.is_numeric_dtype(dtype):
        return "numeric", "float dtype"
    if pd.api.types.is_datetime64_any_dtype(dtype) or pd.api.types.is_timedelta64_dtype(dtype):
        return "drop", f"unsupported dtype {dtype}"
    return "categorical", f"{dtype} dtype"


def _to_float(X: Any) -> pd.DataFrame:
    X = pd.DataFrame(X)
    return X.astype(float).replace([np.inf, -np.inf], np.nan)


def _to_text(X: Any) -> pd.DataFrame:
    X = pd.DataFrame(X)
    return X.astype("string").astype(object).where(X.notna(), np.nan)


def _target_encoder(target_type: str, seed: int) -> TargetEncoder:
    # scikit-learn 1.9 deprecates random_state in favor of a cv generator.
    if "cv" in inspect.signature(TargetEncoder).parameters:
        return TargetEncoder(target_type=target_type, cv=KFold(5, shuffle=True, random_state=seed))
    return TargetEncoder(target_type=target_type, random_state=seed)


def build_preprocessor(plan: ColumnPlan, config: PreprocessingConfig, task: Any) -> ColumnTransformer:
    """Build an unfitted, clonable preprocessor; all learned state is created at `fit` time."""
    try:
        task = Task(getattr(task, "task", task))
    except ValueError:
        raise PreprocessingError(f"unsupported task {task!r}; valid: {[t.value for t in Task]}") from None
    if not plan.numeric and not plan.categorical:
        raise PreprocessingError("no usable feature columns after excluding target, id, and dropped columns")

    transformers: list[tuple[str, Any, list[str]]] = []
    if plan.numeric:
        steps: list[tuple[str, Any]] = [
            ("to_float", FunctionTransformer(_to_float, feature_names_out="one-to-one")),
            (
                "impute",
                SimpleImputer(
                    strategy=config.numeric_imputer,
                    fill_value=config.numeric_fill_value if config.numeric_imputer == "constant" else None,
                    keep_empty_features=True,
                ),
            ),
        ]
        if config.scaler == "standard":
            steps.append(("scale", StandardScaler()))
        elif config.scaler == "robust":
            steps.append(("scale", RobustScaler()))
        transformers.append(("num", Pipeline(steps), list(plan.numeric)))
    if plan.categorical:
        if config.encoder == "onehot":
            encoder: Any = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        elif config.encoder == "ordinal":
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        else:
            encoder = _target_encoder(_TARGET_TYPES[task], config.seed)
        steps = [
            ("to_text", FunctionTransformer(_to_text, feature_names_out="one-to-one")),
            (
                "impute",
                SimpleImputer(
                    strategy=config.categorical_imputer,
                    fill_value=config.categorical_fill_value if config.categorical_imputer == "constant" else None,
                    keep_empty_features=True,
                ),
            ),
            ("encode", encoder),
        ]
        transformers.append(("cat", Pipeline(steps), list(plan.categorical)))
    return ColumnTransformer(transformers, remainder="drop")


def fit_transform_fold(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
    y: pd.Series,
    split: Split,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a clone of `preprocessor` on the split's train rows only; transform train and validation rows."""
    fitted = clone(preprocessor)
    X_train, y_train = X.iloc[split.train_idx], y.iloc[split.train_idx]
    train_t = fitted.fit_transform(X_train, y_train)
    valid_t = fitted.transform(X.iloc[split.valid_idx])
    return np.asarray(train_t), np.asarray(valid_t)


def save_plan(plan: ColumnPlan, config: PreprocessingConfig, path: str | Path) -> None:
    """Write the preprocessing plan as JSON; refuses to overwrite an existing file."""
    path = Path(path)
    payload = {
        "schema_version": ARTIFACT_VERSION,
        "config": {f.name: _jsonable(getattr(config, f.name)) for f in fields(config)},
        "numeric": list(plan.numeric),
        "categorical": list(plan.categorical),
        "dropped": list(plan.dropped),
        "excluded": list(plan.excluded),
        "reasons": dict(plan.reasons),
    }
    try:
        with path.open("x") as f:
            json.dump(payload, f, indent=2)
    except FileExistsError:
        raise PreprocessingError(f"preprocessing plan {path} already exists; choose a new path") from None


def _jsonable(value: Any) -> Any:
    return list(value) if isinstance(value, tuple) else value


def load_plan(path: str | Path) -> tuple[ColumnPlan, PreprocessingConfig]:
    """Read and validate a plan written by `save_plan`."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        raise PreprocessingError(f"preprocessing plan not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise PreprocessingError(f"preprocessing plan {path} is not valid JSON: {exc}") from None

    def bad(msg: str) -> PreprocessingError:
        return PreprocessingError(f"preprocessing plan {path}: {msg}")

    if not isinstance(raw, dict) or raw.get("schema_version") != ARTIFACT_VERSION:
        raise bad(f"unsupported or missing 'schema_version'; expected {ARTIFACT_VERSION}")
    try:
        cfg_raw = raw["config"]
        if not isinstance(cfg_raw, dict) or set(cfg_raw) - PREPROCESSING_KEYS:
            raise bad("'config' must be a table of known preprocessing settings")
        config = PreprocessingConfig(**cfg_raw)
        groups = {k: raw[k] for k in ("numeric", "categorical", "dropped", "excluded")}
        reasons = raw["reasons"]
    except KeyError as exc:
        raise bad(f"missing field {exc}") from None
    except PreprocessingError as exc:
        if exc.problems and str(exc).startswith("preprocessing plan"):
            raise
        raise bad("; ".join(exc.problems)) from None
    for name, value in groups.items():
        if not isinstance(value, list) or not all(isinstance(c, str) for c in value):
            raise bad(f"'{name}' must be a list of column names")
    if not isinstance(reasons, dict):
        raise bad("'reasons' must be an object")
    all_cols = [c for k in ("numeric", "categorical", "dropped", "excluded") for c in groups[k]]
    if len(set(all_cols)) != len(all_cols):
        raise bad("a column appears in more than one group")
    plan = ColumnPlan(
        tuple(groups["numeric"]),
        tuple(groups["categorical"]),
        tuple(groups["dropped"]),
        tuple(groups["excluded"]),
        reasons,
    )
    return plan, config
