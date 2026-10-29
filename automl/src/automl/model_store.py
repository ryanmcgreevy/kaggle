"""Versioned persistence and validation for trusted fitted-model artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from automl.errors import ModelStoreError

MODEL_ARTIFACT_VERSION = 1
MODEL_METADATA_FILE = "model.json"
MODEL_PIPELINE_FILE = "pipeline.joblib"
MODEL_OOF_FILE = "oof_predictions.csv"
_FIT_SCOPES = ("full_data", "fold")
_CREDENTIAL_KEY = re.compile(r"pass(word|wd)?|secret|token|api[_-]?key|credential|auth", re.IGNORECASE)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    return value


def _credential_keys(value: Any, prefix: str = "") -> list[str]:
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            if _CREDENTIAL_KEY.search(str(key)):
                found.append(name)
            found.extend(_credential_keys(child, name))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(_credential_keys(child, f"{prefix}[{index}]"))
    return found


@dataclass(frozen=True)
class FittedModelReference:
    """A resolvable local path or MLflow model URI."""

    uri: str

    def __post_init__(self) -> None:
        if not isinstance(self.uri, str) or not self.uri.strip():
            raise ModelStoreError("a model reference must be a non-empty string")
        if not (
            self.uri.startswith("local:")
            or self.uri.startswith("runs:/")
        ):
            raise ModelStoreError(
                f"unsupported model reference {self.uri!r}; use a local path, 'local:<path>', or MLflow model URI"
            )

    @property
    def backend(self) -> str:
        return "mlflow" if self.uri.startswith("runs:/") or "://" in self.uri else "local"

    def __str__(self) -> str:
        return self.uri


@dataclass(frozen=True, eq=False)
class LoadedFittedModel:
    reference: FittedModelReference
    metadata: dict[str, Any]
    pipeline: Any
    oof_predictions: pd.DataFrame | None = None


def data_fingerprint(frame: pd.DataFrame) -> str:
    """Hash ordered columns, dtypes, and values without persisting raw rows."""
    digest = hashlib.sha256()
    schema = [(str(name), str(dtype)) for name, dtype in zip(frame.columns, frame.dtypes)]
    digest.update(json.dumps(schema, separators=(",", ":"), ensure_ascii=False).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=False, categorize=True).to_numpy().tobytes())
    return digest.hexdigest()


def legacy_data_fingerprint(frame: pd.DataFrame) -> str:
    """Return the pre-model-store fingerprint used by existing ensemble artifacts."""
    digest = hashlib.sha256()
    schema = [(str(name), str(dtype)) for name, dtype in zip(frame.columns, frame.dtypes)]
    digest.update(json.dumps(schema).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=False, categorize=True).to_numpy().tobytes())
    return digest.hexdigest()


def split_fingerprint(splits: Any) -> str:
    """Hash exact positional train/validation membership and fold ordering."""
    if isinstance(splits, dict):
        entries = splits.get("folds")
    else:
        entries = [
            {
                "fold": split.fold,
                "train_rows": np.asarray(split.train_idx, dtype=np.int64).tolist(),
                "valid_rows": np.asarray(split.valid_idx, dtype=np.int64).tolist(),
            }
            for split in splits.splits
        ]
    if not isinstance(entries, list) or not entries:
        raise ModelStoreError("validation split provenance must contain at least one fold")
    normalized = []
    for entry in entries:
        try:
            fold = int(entry["fold"])
            train_rows = [int(value) for value in entry["train_rows"]]
            valid_rows = [int(value) for value in entry["valid_rows"]]
        except (KeyError, TypeError, ValueError):
            raise ModelStoreError("validation split provenance has an invalid fold entry") from None
        normalized.append({"fold": fold, "train_rows": train_rows, "valid_rows": valid_rows})
    return hashlib.sha256(
        json.dumps(normalized, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def split_provenance(splits: Any) -> dict[str, Any]:
    """Return exact positional split membership without including training data."""
    if isinstance(splits, dict):
        source_folds = splits.get("folds")
        if not isinstance(source_folds, list):
            raise ModelStoreError("validation split provenance must contain a fold list")
        folds = [
            {
                "fold": int(split["fold"]),
                "train_rows": [int(row) for row in split["train_rows"]],
                "valid_rows": [int(row) for row in split["valid_rows"]],
            }
            for split in source_folds
        ]
        result = {
            "strategy": splits.get("strategy"),
            "seed": splits.get("seed"),
            "n_rows": splits.get("n_rows"),
            "stratified": splits.get("stratified"),
            "folds": folds,
        }
    else:
        folds = [
            {
                "fold": split.fold,
                "train_rows": np.asarray(split.train_idx, dtype=np.int64).tolist(),
                "valid_rows": np.asarray(split.valid_idx, dtype=np.int64).tolist(),
            }
            for split in splits.splits
        ]
        result = {
            "strategy": splits.config.strategy,
            "seed": splits.config.seed,
            "n_rows": splits.n_rows,
            "stratified": splits.stratified,
            "folds": folds,
        }
    result["sha256"] = split_fingerprint({"folds": folds})
    return result


def capture_model_artifacts(
    pipeline_template: Any,
    X: pd.DataFrame,
    y: pd.Series,
    splits: Any,
    *,
    task: str,
    fit_params: dict[str, Any] | None = None,
    fold_models: list[Any] | tuple[Any, ...] | None = None,
) -> tuple[Any, pd.DataFrame]:
    """Capture full-data inference pipeline and fold-excluded predictions without raw data persistence."""
    if fold_models is not None and len(fold_models) != len(splits.splits):
        raise ModelStoreError("the number of saved fold models does not match the validation splits")
    models = list(fold_models) if fold_models is not None else []
    if fold_models is None:
        for split in splits.splits:
            model = clone(pipeline_template)
            model.fit(X.iloc[split.train_idx], y.iloc[split.train_idx], **(fit_params or {}))
            models.append(model)
    frames = []
    for model, split in zip(models, splits.splits):
        valid_x = X.iloc[split.valid_idx]
        frame = pd.DataFrame({"row": split.valid_idx, "fold": split.fold})
        frame["prediction"] = np.asarray(model.predict(valid_x))
        if task != "regression" and hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(valid_x))
            classes = list(getattr(model, "classes_", getattr(model.named_steps.get("model"), "classes_", ())))
            if probabilities.shape != (len(split.valid_idx), len(classes)):
                raise ModelStoreError(
                    f"fold {split.fold}: model returned invalid probability shape {probabilities.shape}"
                )
            for idx in range(len(classes)):
                frame[f"proba_{idx}"] = probabilities[:, idx]
        frames.append(frame)
    full_model = clone(pipeline_template)
    full_model.fit(X, y, **(fit_params or {}))
    oof = pd.concat(frames, ignore_index=True).sort_values("row", kind="stable").reset_index(drop=True)
    return full_model, oof


def _validate_oof(
    oof: pd.DataFrame,
    n_rows: int,
    splits: dict[str, Any],
) -> None:
    if not isinstance(oof, pd.DataFrame) or not {"row", "fold"} <= set(oof.columns):
        raise ModelStoreError("OOF predictions must be a dataframe with 'row' and 'fold' columns")
    invalid_columns = [
        column
        for column in oof.columns
        if column not in {"row", "fold", "prediction"} and re.fullmatch(r"proba_[0-9]+", str(column)) is None
    ]
    if invalid_columns:
        raise ModelStoreError(f"OOF predictions contain unsupported columns {invalid_columns}")
    if len(oof) == 0:
        raise ModelStoreError("OOF predictions must not be empty")
    row_values = pd.to_numeric(oof["row"], errors="coerce")
    fold_values = pd.to_numeric(oof["fold"], errors="coerce")
    if row_values.isna().any() or fold_values.isna().any():
        raise ModelStoreError("OOF 'row' and 'fold' values must be integers")
    if (row_values % 1 != 0).any() or (fold_values % 1 != 0).any():
        raise ModelStoreError("OOF 'row' and 'fold' values must be integers")
    rows = row_values.astype(int)
    if rows.duplicated().any() or rows.min() < 0 or rows.max() >= n_rows:
        raise ModelStoreError(f"OOF row values must be unique integers in [0, {n_rows})")
    expected: dict[int, int] = {}
    for entry in splits["folds"]:
        train_rows = entry["train_rows"]
        valid_rows = entry["valid_rows"]
        if (
            not train_rows
            or not valid_rows
            or len(set(train_rows)) != len(train_rows)
            or len(set(valid_rows)) != len(valid_rows)
            or any(row < 0 or row >= n_rows for row in (*train_rows, *valid_rows))
            or set(train_rows).intersection(valid_rows)
        ):
            raise ModelStoreError(f"validation fold {entry['fold']} has invalid row membership")
        for row in valid_rows:
            if row in expected:
                raise ModelStoreError(f"validation splits assign row {row} to more than one OOF fold")
            expected[row] = entry["fold"]
    actual = dict(zip(rows.tolist(), fold_values.astype(int).tolist()))
    if any(row not in expected or expected[row] != fold for row, fold in actual.items()):
        raise ModelStoreError("OOF fold assignments do not match validation split provenance")
    predictions = oof.drop(columns=["row", "fold"])
    if predictions.empty:
        raise ModelStoreError("OOF predictions must include at least one prediction column")
    for column in predictions.columns:
        values = predictions[column]
        if values.isna().any():
            raise ModelStoreError(f"OOF prediction column {column!r} contains null values")
        if pd.api.types.is_numeric_dtype(values):
            if not np.isfinite(values.to_numpy(dtype=float)).all():
                raise ModelStoreError(f"OOF prediction column {column!r} contains non-finite values")
    probability_columns = sorted(
        (column for column in predictions.columns if re.fullmatch(r"proba_[0-9]+", str(column))),
        key=lambda column: int(str(column).removeprefix("proba_")),
    )
    if probability_columns:
        expected_names = [f"proba_{idx}" for idx in range(len(probability_columns))]
        if probability_columns != expected_names:
            raise ModelStoreError("OOF probability columns must be contiguous and ordered from 'proba_0'")
        probabilities = oof[probability_columns].to_numpy(dtype=float)
        if (probabilities < -1e-5).any() or (probabilities > 1 + 1e-5).any():
            raise ModelStoreError("OOF probabilities must lie in [0, 1]")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5, rtol=0):
            raise ModelStoreError("OOF class probabilities must sum to 1")


def _local_directory(reference: FittedModelReference) -> Path:
    raw = reference.uri.removeprefix("local:")
    if not raw:
        raise ModelStoreError("local model reference has no path")
    return Path(raw).expanduser()


def save_fitted_model(
    pipeline: Any,
    directory: str | Path,
    *,
    task: str,
    metric: str | None,
    direction: str | None,
    feature_columns: list[str] | tuple[str, ...],
    class_order: list[Any] | tuple[Any, ...] = (),
    effective_config: dict[str, Any] | None = None,
    dependency_versions: dict[str, str] | None = None,
    training_data_sha256: str,
    n_training_rows: int,
    fit_scope: str = "full_data",
    splits: Any = None,
    oof_predictions: pd.DataFrame | None = None,
    overwrite: bool = False,
) -> FittedModelReference:
    """Atomically save a trusted fitted pipeline, metadata, and optional aligned OOF predictions."""
    if fit_scope not in _FIT_SCOPES:
        raise ModelStoreError(f"fit_scope must be one of {_FIT_SCOPES}, got {fit_scope!r}")
    if isinstance(n_training_rows, bool) or not isinstance(n_training_rows, int) or n_training_rows < 1:
        raise ModelStoreError("n_training_rows must be a positive integer")
    if len(training_data_sha256) != 64:
        raise ModelStoreError("training_data_sha256 must be a SHA-256 hexadecimal digest")
    features = list(feature_columns)
    if not features or not all(isinstance(name, str) and name for name in features) or len(set(features)) != len(features):
        raise ModelStoreError("feature_columns must contain unique non-empty column names")
    secrets = _credential_keys(effective_config or {})
    if secrets:
        raise ModelStoreError(f"effective_config contains credential-like keys {secrets}; secrets must not be persisted")
    split_info = split_provenance(splits) if splits is not None else None
    if oof_predictions is not None:
        if split_info is None:
            raise ModelStoreError("OOF predictions require validation split provenance")
        _validate_oof(oof_predictions, n_training_rows, split_info)

    out = Path(directory).expanduser()
    if out.exists() and not overwrite:
        raise ModelStoreError(f"fitted-model directory {out} already exists; choose a new path or set overwrite=True")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{out.name}.model-", dir=out.parent))
    metadata = {
        "schema_version": MODEL_ARTIFACT_VERSION,
        "model_id": str(uuid.uuid4()),
        "task": task,
        "metric": {"name": metric, "direction": direction},
        "feature_columns": features,
        "class_order": list(class_order),
        "effective_config": effective_config or {},
        "dependency_versions": dependency_versions or {},
        "training_data_sha256": training_data_sha256,
        "n_training_rows": n_training_rows,
        "fit_scope": fit_scope,
        "validation_splits": split_info,
        "oof": {
            "available": oof_predictions is not None,
            "file": MODEL_OOF_FILE if oof_predictions is not None else None,
            "coverage_rows": len(oof_predictions) if oof_predictions is not None else 0,
            "complete": bool(
                oof_predictions is not None
                and len(oof_predictions) == len(set().union(*(set(f["valid_rows"]) for f in split_info["folds"])))
            ),
        },
        "files": {"pipeline": MODEL_PIPELINE_FILE, "oof_predictions": MODEL_OOF_FILE if oof_predictions is not None else None},
    }
    try:
        joblib.dump(pipeline, tmp / MODEL_PIPELINE_FILE)
        if oof_predictions is not None:
            oof_predictions.sort_values("row", kind="stable").to_csv(tmp / MODEL_OOF_FILE, index=False)
        (tmp / MODEL_METADATA_FILE).write_text(
            json.dumps(_json_safe(metadata), indent=2, allow_nan=False)
        )
        if out.exists():
            backup = out.with_name(f".{out.name}.previous-{uuid.uuid4().hex}")
            os.replace(out, backup)
            try:
                os.replace(tmp, out)
            except Exception:
                os.replace(backup, out)
                raise
            shutil.rmtree(backup)
        else:
            os.replace(tmp, out)
    except Exception as exc:
        raise ModelStoreError(f"could not save fitted model to {out}: {type(exc).__name__}: {exc}") from exc
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
    return FittedModelReference(f"local:{out.resolve()}")


def _load_local(reference: FittedModelReference) -> LoadedFittedModel:
    directory = _local_directory(reference)
    metadata_path = directory / MODEL_METADATA_FILE
    try:
        metadata = json.loads(metadata_path.read_text())
    except FileNotFoundError:
        raise ModelStoreError(f"{metadata_path} not found; pass a fitted-model directory written by save_fitted_model") from None
    except json.JSONDecodeError as exc:
        raise ModelStoreError(f"{metadata_path} is not valid JSON: {exc}") from None
    if not isinstance(metadata, dict) or metadata.get("schema_version") != MODEL_ARTIFACT_VERSION:
        got = metadata.get("schema_version") if isinstance(metadata, dict) else None
        raise ModelStoreError(
            f"{metadata_path} has unsupported schema_version {got!r}; expected {MODEL_ARTIFACT_VERSION}"
        )
    required = (
        "model_id",
        "task",
        "metric",
        "feature_columns",
        "class_order",
        "effective_config",
        "dependency_versions",
        "training_data_sha256",
        "n_training_rows",
        "fit_scope",
        "validation_splits",
        "oof",
        "files",
    )
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ModelStoreError(f"{metadata_path} is missing required fields {missing}")
    try:
        if metadata["task"] not in ("binary", "multiclass", "regression"):
            raise ValueError("task is unsupported")
        metric = metadata["metric"]
        if not isinstance(metric, dict) or any(
            metric.get(key) is not None and not isinstance(metric.get(key), str)
            for key in ("name", "direction")
        ):
            raise ValueError("metric must contain optional string name and direction")
        features = metadata["feature_columns"]
        if (
            not isinstance(features, list)
            or not features
            or not all(isinstance(name, str) and name for name in features)
            or len(set(features)) != len(features)
        ):
            raise ValueError("feature_columns must contain unique non-empty strings")
        digest = metadata["training_data_sha256"]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("training_data_sha256 must be a SHA-256 hexadecimal digest")
        if (
            isinstance(metadata["n_training_rows"], bool)
            or not isinstance(metadata["n_training_rows"], int)
            or metadata["n_training_rows"] < 1
        ):
            raise ValueError("n_training_rows must be a positive integer")
        if metadata["fit_scope"] not in _FIT_SCOPES:
            raise ValueError(f"fit_scope must be one of {_FIT_SCOPES}")
        if not isinstance(metadata["class_order"], list) or (
            metadata["task"] == "regression" and metadata["class_order"]
        ):
            raise ValueError("class_order must be a list and must be empty for regression")
        if not isinstance(metadata["effective_config"], dict) or not isinstance(
            metadata["dependency_versions"], dict
        ):
            raise ValueError("effective_config and dependency_versions must be objects")
        if not isinstance(metadata["oof"], dict) or not isinstance(metadata["oof"].get("available"), bool):
            raise ValueError("oof metadata must declare whether predictions are available")
        if metadata["oof"]["available"]:
            split_info = metadata["validation_splits"]
            if not isinstance(split_info, dict) or split_fingerprint(split_info) != split_info.get("sha256"):
                raise ValueError("validation split provenance is missing or has an invalid fingerprint")
    except (KeyError, TypeError, ValueError, ModelStoreError) as exc:
        raise ModelStoreError(f"{metadata_path} has invalid metadata: {exc}") from None
    files = metadata["files"]
    if not isinstance(files, dict) or files.get("pipeline") != MODEL_PIPELINE_FILE:
        raise ModelStoreError(f"{metadata_path} has an invalid pipeline artifact path")
    pipeline_path = directory / MODEL_PIPELINE_FILE
    if not pipeline_path.is_file():
        raise ModelStoreError(f"{pipeline_path} not found")
    try:
        pipeline = joblib.load(pipeline_path)
    except Exception as exc:
        raise ModelStoreError(f"could not load trusted model {pipeline_path}: {type(exc).__name__}: {exc}") from None
    oof = None
    oof_info = metadata["oof"]
    oof_name = files.get("oof_predictions")
    if oof_info.get("available"):
        if oof_name != MODEL_OOF_FILE or not (directory / oof_name).is_file():
            raise ModelStoreError(f"{directory / MODEL_OOF_FILE} is missing")
        try:
            oof = pd.read_csv(directory / oof_name)
            splits = metadata["validation_splits"]
            _validate_oof(oof, metadata["n_training_rows"], splits)
        except (OSError, ValueError) as exc:
            raise ModelStoreError(f"could not validate OOF predictions in {directory}: {exc}") from None
        expected_rows = set(row for fold in metadata["validation_splits"]["folds"] for row in fold["valid_rows"])
        actual_rows = set(oof["row"].astype(int))
        complete = actual_rows == expected_rows
        if (
            oof_info.get("coverage_rows") != len(oof)
            or oof_info.get("complete") is not complete
            or (metadata["task"] == "regression" and not pd.api.types.is_numeric_dtype(oof["prediction"]))
        ):
            raise ModelStoreError(f"{metadata_path} has inconsistent OOF coverage or prediction metadata")
        if metadata["task"] != "regression":
            class_order = metadata["class_order"]
            if not set(oof["prediction"].tolist()) <= set(class_order):
                raise ModelStoreError(f"{metadata_path} OOF predictions contain labels outside class_order")
            probability_columns = [column for column in oof.columns if column.startswith("proba_")]
            if probability_columns and len(probability_columns) != len(class_order):
                raise ModelStoreError(f"{metadata_path} OOF probability width does not match class_order")
    elif oof_name is not None:
        raise ModelStoreError(f"{metadata_path} declares an OOF file but says OOF predictions are unavailable")
    return LoadedFittedModel(reference, metadata, pipeline, oof)


def load_fitted_model(
    reference: FittedModelReference | str | Path,
    *,
    tracking_uri: str | None = None,
) -> LoadedFittedModel:
    """Load a trusted local artifact or an explicitly referenced MLflow model."""
    if isinstance(reference, FittedModelReference):
        ref = reference
    elif isinstance(reference, Path):
        ref = FittedModelReference(f"local:{reference.expanduser()}")
    else:
        text = str(reference)
        if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:/", text) and not text.startswith(("local:", "runs:/")):
            raise ModelStoreError(
                f"unsupported model reference {text!r}; use a local path, 'local:<path>', or an MLflow runs:/ URI"
            )
        ref = FittedModelReference(
            text if text.startswith(("local:", "runs:/")) else f"local:{text}"
        )
    if ref.backend == "local":
        return _load_local(ref)
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        raise ModelStoreError(
            "loading an MLflow model requires the 'mlflow' package; install it with: pip install 'automl[mlflow]'"
        ) from None
    previous_uri = mlflow.get_tracking_uri()
    try:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        if not ref.uri.startswith("runs:/"):
            raise ModelStoreError("MLflow model references must use a stable 'runs:/<run-id>/model' URI")
        pipeline = mlflow.sklearn.load_model(ref.uri)
        run_prefix = "/".join(ref.uri.split("/")[:2])
        artifact_uri = f"{run_prefix}/automl-model/{MODEL_METADATA_FILE}"
        metadata_path = Path(mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri))
        metadata = json.loads(metadata_path.read_text())
        oof = None
        if metadata.get("oof", {}).get("available"):
            oof_uri = f"{run_prefix}/automl-model/{MODEL_OOF_FILE}"
            oof_path = Path(mlflow.artifacts.download_artifacts(artifact_uri=oof_uri))
            oof = pd.read_csv(oof_path)
            _validate_oof(oof, metadata["n_training_rows"], metadata["validation_splits"])
    except ModelStoreError:
        raise
    except Exception as exc:
        from automl.tracking import redact_uri

        raise ModelStoreError(
            f"could not load MLflow fitted model {ref.uri!r}: {type(exc).__name__}: {redact_uri(str(exc))}"
        ) from None
    finally:
        mlflow.set_tracking_uri(previous_uri)
    if not isinstance(metadata, dict) or metadata.get("schema_version") != MODEL_ARTIFACT_VERSION:
        raise ModelStoreError(f"MLflow model {ref.uri!r} has unsupported or missing fitted-model metadata")
    if not isinstance(metadata.get("training_data_sha256"), str) or re.fullmatch(
        r"[0-9a-f]{64}", metadata["training_data_sha256"]
    ) is None:
        raise ModelStoreError(f"MLflow model {ref.uri!r} has an invalid training-data fingerprint")
    return LoadedFittedModel(ref, metadata, pipeline, oof)
