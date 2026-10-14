"""Build and validate a Kaggle-format submission from refit predictions.

The submission always follows the row order of the test file. When a sample submission is supplied, its
columns (names, order, dtype classes) define the output schema and its identifiers must match the test
identifiers. Nothing here contacts Kaggle; the written CSV is for manual upload.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from automl.errors import DataContractError, SubmissionError
from automl.metrics import get_metric

SUBMISSION_SCHEMA_VERSION = 1
CSV_FILE = "submission.csv"
METADATA_FILE = "submission.json"
_PREDICTION_COLUMN = "prediction"
_PROBA_PREFIX = "proba_"
NOTE = "A submission holds raw model predictions; validation scores are not leaderboard estimates. Nothing was submitted to Kaggle."


@dataclass(frozen=True)
class SubmissionConfig:
    column_map: dict[str, str] | None = None
    max_reported_rows: int = 10
    float_format: str | None = None
    allow_unverified_ids: bool = False

    def __post_init__(self) -> None:
        problems: list[str] = []
        cm = self.column_map
        if cm is not None and (
            not isinstance(cm, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in cm.items())
        ):
            problems.append("setting 'column_map' must be a table of sample column names to prediction column names")
        n = self.max_reported_rows
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            problems.append(f"setting 'max_reported_rows' must be an integer of at least 1, got {n!r}")
        ff = self.float_format
        if ff is not None and (not isinstance(ff, str) or not ff.strip()):
            problems.append("setting 'float_format' must be a non-empty format string such as '%.6f'")
        if not isinstance(self.allow_unverified_ids, bool):
            problems.append("setting 'allow_unverified_ids' must be true or false")
        if problems:
            raise SubmissionError(problems)
        if cm is not None:
            object.__setattr__(self, "column_map", dict(cm))


SUBMISSION_KEYS = {f.name for f in fields(SubmissionConfig)}


@dataclass(frozen=True, eq=False)
class SubmissionResult:
    frame: pd.DataFrame
    columns: tuple[str, ...]
    column_mapping: dict[str, str]  # output column -> source ("test:<col>" or prediction column)
    n_rows: int
    id_columns: tuple[str, ...]
    id_check: dict[str, Any]
    schema_source: str  # "sample" or "fallback"
    source: dict[str, Any]
    config: SubmissionConfig = field(default_factory=SubmissionConfig)
    sample_name: str | None = None
    notes: tuple[str, ...] = ()
    directory: Path | None = None
    tracking: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SUBMISSION_SCHEMA_VERSION,
            "columns": list(self.columns),
            "column_mapping": self.column_mapping,
            "n_rows": self.n_rows,
            "id_columns": list(self.id_columns),
            "id_check": self.id_check,
            "schema_source": self.schema_source,
            "sample_file": self.sample_name,
            "source": self.source,
            "config": {
                "column_map": self.config.column_map,
                "max_reported_rows": self.config.max_reported_rows,
                "float_format": self.config.float_format,
                "allow_unverified_ids": self.config.allow_unverified_ids,
            },
            "notes": list(self.notes),
            "files": {"submission": CSV_FILE, "metadata": METADATA_FILE},
            "statement": NOTE,
        }


@dataclass(frozen=True, eq=False)
class LoadedSubmission:
    metadata: dict[str, Any]
    frame: pd.DataFrame


def _plain(value: Any) -> Any:
    def default(v: Any) -> Any:
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
        return str(v)

    return json.loads(json.dumps(value, default=default, allow_nan=False))


# ---- helpers ----


def _positions(mask: np.ndarray, cfg: SubmissionConfig) -> str:
    idx = np.flatnonzero(mask)
    shown = ", ".join(map(str, idx[: cfg.max_reported_rows]))
    more = f" (+{len(idx) - cfg.max_reported_rows} more)" if len(idx) > cfg.max_reported_rows else ""
    return f"{len(idx)} rows, first positions: {shown}{more}"


def _values(items: list[Any], cfg: SubmissionConfig) -> str:
    shown = ", ".join(repr(v) for v in items[: cfg.max_reported_rows])
    more = f" (+{len(items) - cfg.max_reported_rows} more)" if len(items) > cfg.max_reported_rows else ""
    return f"{shown}{more}"


def _unpack(refit: Any) -> dict[str, Any]:
    """Normalise a RefitResult or LoadedRefit into the fields this stage needs."""
    if hasattr(refit, "predictions") and hasattr(refit, "refit"):
        sel = refit.refit.selection
        return {
            "frame": refit.predictions.frame,
            "metric": sel.metric,
            "target": refit.refit.target,
            "id_columns": tuple(refit.refit.id_columns),
            "label": sel.label,
            "source_kind": sel.source_kind,
            "source_path": sel.source_path,
            "directory": str(refit.directory) if refit.directory is not None else None,
        }
    if hasattr(refit, "metadata") and hasattr(refit, "predictions"):
        meta = refit.metadata
        sel = meta.get("selection", {})
        data = meta.get("data", {})
        return {
            "frame": refit.predictions,
            "metric": sel.get("metric"),
            "target": data.get("target"),
            "id_columns": tuple(data.get("id_columns", ())),
            "label": sel.get("label"),
            "source_kind": sel.get("source_kind"),
            "source_path": sel.get("source_path"),
            "directory": None,
        }
    raise SubmissionError("'refit' must be a RefitResult from refit_and_predict or a LoadedRefit from load_refit")


def _needs_proba(metric: str | None) -> bool:
    if metric is None:
        return False
    try:
        return get_metric(metric).needs == "proba"
    except DataContractError:
        return False


def _key_series(df: pd.DataFrame, cols: tuple[str, ...]) -> pd.Series:
    text = df[list(cols)].map(str)  # astype(str) can leave NaN as a float
    return text.agg("\x1f".join, axis=1) if len(cols) > 1 else text[cols[0]]


def _check_ids(
    test: pd.DataFrame,
    pred: pd.DataFrame,
    sample: pd.DataFrame | None,
    ids: tuple[str, ...],
    cfg: SubmissionConfig,
    notes: list[str],
    problems: list[str],
) -> dict[str, Any]:
    check: dict[str, Any] = {"verified": False, "sample_order_matches_test": None, "n_rows": len(test)}
    if len(pred) != len(test):
        problems.append(f"refit predictions have {len(pred)} rows but the test data has {len(test)}")
    if sample is not None and len(sample) != len(test):
        problems.append(f"sample submission has {len(sample)} rows but the test data has {len(test)}")
    if not ids:
        if sample is not None and not cfg.allow_unverified_ids:
            problems.append(
                "no id columns are configured, so sample-submission identifiers cannot be checked; "
                "configure id_columns or set allow_unverified_ids=true to match rows by position"
            )
        notes.append("no id columns configured: rows are matched by position and identifier alignment is unverified")
        return check
    for label, frame in (("test data", test), ("refit predictions", pred), ("sample submission", sample)):
        if frame is None:
            continue
        missing = [c for c in ids if c not in frame.columns]
        if missing:
            problems.append(f"{label} is missing id columns {missing}")
    if problems and any("missing id columns" in p for p in problems):
        return check
    if len(pred) == len(test):
        a, b = _key_series(test, ids).to_numpy(), _key_series(pred, ids).to_numpy()
        bad = a != b
        if bad.any():
            problems.append(
                f"refit prediction identifiers do not match the test rows in order ({_positions(bad, cfg)}); "
                "the submission must follow test.csv order"
            )
    for label, frame in (("test data", test), ("sample submission", sample)):
        if frame is None:
            continue
        nulls = frame[list(ids)].isna().any(axis=1).to_numpy()
        if nulls.any():
            problems.append(f"{label} has null identifiers in {_positions(nulls, cfg)}")
        dup = _key_series(frame, ids).duplicated(keep=False).to_numpy()
        if dup.any():
            problems.append(f"{label} has duplicate identifiers in {_positions(dup, cfg)}")
    if sample is not None:
        t, s = _key_series(test, ids), _key_series(sample, ids)
        only_test = sorted(set(t) - set(s))
        only_sample = sorted(set(s) - set(t))
        if only_test or only_sample:
            problems.append(
                f"sample submission identifiers differ from the test data: only in test [{_values(only_test, cfg)}]; "
                f"only in sample [{_values(only_sample, cfg)}]"
            )
        elif len(t) == len(s):
            same = bool((t.to_numpy() == s.to_numpy()).all())
            check["sample_order_matches_test"] = same
            if not same:
                notes.append("sample submission rows are in a different order than the test data; the submission keeps test order")
    check["verified"] = not any("identifier" in p for p in problems)
    return check


def _map_columns(
    sample: pd.DataFrame,
    pred: pd.DataFrame,
    ids: tuple[str, ...],
    metric: str | None,
    cfg: SubmissionConfig,
    problems: list[str],
) -> dict[str, str]:
    """Return sample non-id column -> prediction frame column."""
    targets = [c for c in sample.columns if c not in ids]
    available = [c for c in pred.columns if c not in ids]
    proba = [c for c in available if c.startswith(_PROBA_PREFIX)]
    if not targets:
        problems.append("sample submission has no columns besides the id columns")
        return {}
    if cfg.column_map is not None:
        unknown = sorted(set(cfg.column_map) - set(targets))
        bad_src = sorted({v for v in cfg.column_map.values() if v not in available})
        uncovered = [c for c in targets if c not in cfg.column_map]
        if unknown:
            problems.append(f"column_map names columns not in the sample submission: {unknown}; sample columns: {targets}")
        if bad_src:
            problems.append(f"column_map uses prediction columns that do not exist: {bad_src}; available: {available}")
        if uncovered:
            problems.append(f"column_map does not cover sample columns {uncovered}")
        return {} if (unknown or bad_src or uncovered) else {c: cfg.column_map[c] for c in targets}
    if len(targets) == 1:
        if _PREDICTION_COLUMN not in available:
            problems.append(f"prediction column '{_PREDICTION_COLUMN}' not found; available: {available}")
            return {}
        if len(proba) == 2 and _needs_proba(metric):
            return {targets[0]: proba[-1]}
        if _needs_proba(metric) and not proba:
            problems.append(f"metric '{metric}' needs probabilities but the refit predictions contain none")
            return {}
        return {targets[0]: _PREDICTION_COLUMN}
    mapping: dict[str, str] = {}
    unmapped: list[str] = []
    for col in targets:
        match = [p for p in proba if col in (p, p[len(_PROBA_PREFIX):])]
        if not match and col == _PREDICTION_COLUMN:
            match = [_PREDICTION_COLUMN]
        if len(match) == 1:
            mapping[col] = match[0]
        else:
            unmapped.append(col)
    if unmapped:
        problems.append(
            f"cannot map sample columns {unmapped} to predictions; available prediction columns: {available}. "
            "Name sample columns after the class labels or pass column_map"
        )
        return {}
    return mapping


def _cast(series: pd.Series, ref: pd.Series, name: str, cfg: SubmissionConfig, problems: list[str]) -> pd.Series | None:
    nulls = series.isna().to_numpy()
    if nulls.any():
        problems.append(f"column '{name}' has null values in {_positions(nulls, cfg)}")
        return None
    kind = ref.dtype.kind
    if kind in "iuf":
        num = pd.to_numeric(series, errors="coerce")
        if num.isna().any():
            problems.append(f"column '{name}' has non-numeric values but the sample column is {ref.dtype}")
            return None
        arr = num.to_numpy(dtype=float)
        bad = ~np.isfinite(arr)
        if bad.any():
            problems.append(f"column '{name}' has non-finite values in {_positions(bad, cfg)}")
            return None
        if kind in "iu" and (arr != np.floor(arr)).any():
            problems.append(f"column '{name}' has non-integer values but the sample column is {ref.dtype}; casting would lose information")
            return None
        return num.astype(ref.dtype)
    if kind == "b":
        ok = series.isin([0, 1, True, False]).to_numpy()
        if not ok.all():
            problems.append(f"column '{name}' has non-boolean values but the sample column is bool")
            return None
        return series.astype(bool)
    if kind == "O" or pd.api.types.is_string_dtype(ref.dtype):
        return series if series.dtype == ref.dtype else series.astype(str)
    if series.dtype != ref.dtype:
        problems.append(f"column '{name}' cannot be converted to the sample dtype {ref.dtype}")
        return None
    return series


# ---- build ----


def build_submission(
    refit: Any,
    test: pd.DataFrame,
    *,
    sample: pd.DataFrame | None = None,
    id_columns: tuple[str, ...] | None = None,
    target: str | None = None,
    config: SubmissionConfig | None = None,
    sample_name: str | None = None,
    directory: str | Path | None = None,
    overwrite: bool = False,
    tracking: Any = None,
) -> SubmissionResult:
    """Validate refit predictions against the test rows and sample schema, optionally save and log them.

    `refit` is a `RefitResult` or a `LoadedRefit`. Inputs are never modified; the output keeps test order.
    """
    cfg = config or SubmissionConfig()
    if tracking is not None and tracking.enabled:
        if directory is None:
            raise SubmissionError("tracking needs the submission files: pass a directory to write them to")
        from automl.tracking import _require_mlflow

        _require_mlflow()
    info = _unpack(refit)
    pred: pd.DataFrame = info["frame"]
    ids = tuple(info["id_columns"] if id_columns is None else id_columns)
    problems: list[str] = []
    notes: list[str] = []
    if len(test) == 0:
        raise SubmissionError("test data has no rows")

    id_check = _check_ids(test, pred, sample, ids, cfg, notes, problems)
    if problems:
        raise SubmissionError(problems)

    if sample is not None:
        mapping = _map_columns(sample, pred, ids, info["metric"], cfg, problems)
        out_cols = list(sample.columns)
        refs = sample
        schema_source = "sample"
    else:
        name = target or info["target"]
        if not name:
            raise SubmissionError("no sample submission and no target column name; pass target=")
        if name in ids:
            raise SubmissionError(f"target column '{name}' cannot also be an id column")
        if _PREDICTION_COLUMN not in pred.columns:
            raise SubmissionError(f"prediction column '{_PREDICTION_COLUMN}' not found; available: {list(pred.columns)}")
        mapping = {name: _PREDICTION_COLUMN}
        out_cols = [*ids, name]
        refs = None
        schema_source = "fallback"
        notes.append("no sample submission supplied: the schema (columns, order, dtypes) is unverified")
    if problems:
        raise SubmissionError(problems)

    data: dict[str, pd.Series] = {}
    provenance: dict[str, str] = {}
    for col in out_cols:
        if col in ids:
            raw, source = test[col].reset_index(drop=True), f"test:{col}"
        else:
            src = mapping[col]
            raw, source = pred[src].reset_index(drop=True), src
        ref = refs[col] if refs is not None else raw
        cast = _cast(raw, ref, col, cfg, problems)
        if cast is not None and source.startswith(_PROBA_PREFIX):
            arr = cast.to_numpy(dtype=float)
            bad = (arr < 0) | (arr > 1)
            if bad.any():
                problems.append(f"column '{col}' has probabilities outside [0, 1] in {_positions(bad, cfg)}")
        data[col] = cast if cast is not None else raw
        provenance[col] = source
    if problems:
        raise SubmissionError(problems)

    frame = pd.DataFrame(data, columns=out_cols)
    if len(frame) != len(test):
        raise SubmissionError(f"submission has {len(frame)} rows but the test data has {len(test)}")
    source_info = {k: info[k] for k in ("label", "source_kind", "source_path", "directory")}
    result = SubmissionResult(
        frame=frame,
        columns=tuple(out_cols),
        column_mapping=provenance,
        n_rows=len(frame),
        id_columns=ids,
        id_check=_plain(id_check),
        schema_source=schema_source,
        source=_plain(source_info),
        config=cfg,
        sample_name=sample_name,
        notes=tuple(notes),
    )
    out = save_submission(result, directory, overwrite=overwrite) if directory is not None else None
    tracked = None
    if tracking is not None and tracking.enabled:
        from automl.tracking import track_submission

        tracked = track_submission(result.to_dict(), out, tracking)
    return SubmissionResult(
        frame, result.columns, result.column_mapping, result.n_rows, ids, result.id_check, schema_source,
        result.source, cfg, sample_name, result.notes, out, tracked,
    )


# ---- persistence ----


def save_submission(result: SubmissionResult, directory: str | Path, *, overwrite: bool = False) -> Path:
    """Write submission.csv and submission.json; refuse to overwrite by default."""
    out = Path(directory)
    names = (CSV_FILE, METADATA_FILE)
    existing = [n for n in names if (out / n).exists()]
    if existing and not overwrite:
        raise SubmissionError(f"{out} already contains {existing}; pass overwrite=True or choose another directory")
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_plain(result.to_dict()), indent=2, allow_nan=False)
    tmp = Path(tempfile.mkdtemp(prefix=".submission-", dir=out))
    try:
        result.frame.to_csv(tmp / CSV_FILE, index=False, float_format=result.config.float_format)
        (tmp / METADATA_FILE).write_text(text)
        for name in names:
            os.replace(tmp / name, out / name)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


_REQUIRED_KEYS = ("columns", "column_mapping", "n_rows", "id_columns", "id_check", "files")


def load_submission(directory: str | Path) -> LoadedSubmission:
    """Read submission.json and submission.csv written by `save_submission`."""
    out = Path(directory)
    meta_path = out / METADATA_FILE
    if not meta_path.is_file():
        raise SubmissionError(f"{meta_path} not found; pass a directory written by save_submission")
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError as exc:
        raise SubmissionError(f"{meta_path} is not valid JSON: {exc}") from None
    if not isinstance(meta, dict) or meta.get("schema_version") != SUBMISSION_SCHEMA_VERSION:
        got = meta.get("schema_version") if isinstance(meta, dict) else None
        raise SubmissionError(f"{meta_path} has unsupported schema_version {got!r}; expected {SUBMISSION_SCHEMA_VERSION}")
    missing = [k for k in _REQUIRED_KEYS if k not in meta]
    if missing:
        raise SubmissionError(f"{meta_path} is missing keys {missing}")
    csv_path = out / CSV_FILE
    if not csv_path.is_file():
        raise SubmissionError(f"{csv_path} not found")
    return LoadedSubmission(meta, pd.read_csv(csv_path))
