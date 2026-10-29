"""Refit a selected pipeline on all training rows and predict the test rows.

Selection comes from a saved baseline, tuning, or families result (ranked by `collect_candidates`) or
from an explicit estimator and parameters. Fold models from earlier runs are never reused: preprocessing
and estimator are rebuilt and fitted together as one scikit-learn `Pipeline`. Outputs are raw model
outputs, validated for row count and finite values; submission schema work belongs to a later stage.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from automl.baseline import _versions
from automl.errors import DataContractError, EnsembleError, RefitError
from automl.estimators import (
    DEVICES,
    FeatureLayout,
    controlled_param_problems,
    enable_native_verbosity,
    get_estimator_spec,
    list_estimators,
    resolve_device,
)
from automl.metrics import Task, get_metric
from automl.model_store import data_fingerprint, legacy_data_fingerprint, save_fitted_model
from automl.model_report import collect_candidates
from automl.preprocessing import ColumnPlan, PreprocessingConfig, build_preprocessor, infer_columns
from automl.progress import ProgressCallback, emit_progress

REFIT_SCHEMA_VERSION = 1
METADATA_FILE = "refit.json"
PREDICTIONS_FILE = "predictions.csv"
MODEL_FILE = "pipeline.joblib"
_PREDICTION_COLUMN = "prediction"
_PROBA_PREFIX = "proba_"


@dataclass(frozen=True)
class RefitConfig:
    save_model: bool = True
    probabilities: bool = True
    proba_tolerance: float = 1e-5
    max_reported_rows: int = 10

    def __post_init__(self) -> None:
        problems: list[str] = []
        for name in ("save_model", "probabilities"):
            if not isinstance(getattr(self, name), bool):
                problems.append(f"setting '{name}' must be true or false")
        tol = self.proba_tolerance
        if isinstance(tol, bool) or not isinstance(tol, (int, float)) or not 0 < tol < 1:
            problems.append(f"setting 'proba_tolerance' must be a number between 0 and 1, got {tol!r}")
        n = self.max_reported_rows
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            problems.append(f"setting 'max_reported_rows' must be an integer of at least 1, got {n!r}")
        if problems:
            raise RefitError(problems)


REFIT_KEYS = {f.name for f in fields(RefitConfig)}


@dataclass(frozen=True, eq=False)
class PipelineSelection:
    source_kind: str  # "baseline", "tuning", "family", or "explicit"
    source_path: str | None
    label: str
    estimator: str
    params: dict[str, Any]
    preprocessing_config: PreprocessingConfig
    task: str
    metric: str | None
    direction: str | None
    seed: int
    n_jobs: int
    device: str
    native_categoricals: bool | None
    features: tuple[str, ...] | None = None
    trial_number: int | None = None
    validation_mean: float | None = None
    validation_std: float | None = None
    versions: dict[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _plain(
            {
                "source_kind": self.source_kind,
                "source_path": self.source_path,
                "label": self.label,
                "estimator": self.estimator,
                "params": self.params,
                "preprocessing": asdict(self.preprocessing_config),
                "task": self.task,
                "metric": self.metric,
                "direction": self.direction,
                "seed": self.seed,
                "n_jobs": self.n_jobs,
                "device": self.device,
                "native_categoricals": self.native_categoricals,
                "features": list(self.features) if self.features is not None else None,
                "trial_number": self.trial_number,
                "validation_mean": self.validation_mean,
                "validation_std": self.validation_std,
                "versions": self.versions,
                "notes": list(self.notes),
            }
        )


@dataclass(frozen=True, eq=False)
class RefitPipeline:
    pipeline: Pipeline
    selection: PipelineSelection
    plan: ColumnPlan
    target: str
    id_columns: tuple[str, ...]
    task: Task
    classes: tuple[Any, ...]
    n_train: int
    fit_seconds: float
    device: dict[str, Any]
    native_categoricals: bool
    training_data_sha256: str

    @property
    def features(self) -> list[str]:
        return [*self.plan.numeric, *self.plan.categorical]


@dataclass(frozen=True, eq=False)
class PredictionResult:
    frame: pd.DataFrame
    task: Task
    prediction_kind: str  # "values", "labels", or "labels+proba"
    n_rows: int
    classes: tuple[Any, ...] = ()
    proba_columns: tuple[str, ...] = ()
    id_columns: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.value,
            "prediction_kind": self.prediction_kind,
            "n_rows": self.n_rows,
            "columns": list(self.frame.columns),
            "id_columns": list(self.id_columns),
            "prediction_column": _PREDICTION_COLUMN,
            "proba_columns": list(self.proba_columns),
            "classes": _plain(list(self.classes)),
            "notes": list(self.notes),
        }


@dataclass(frozen=True, eq=False)
class RefitResult:
    refit: RefitPipeline
    predictions: PredictionResult
    directory: Path | None = None
    tracking: Any = None


@dataclass(frozen=True, eq=False)
class LoadedRefit:
    metadata: dict[str, Any]
    predictions: pd.DataFrame
    pipeline: Any = None


def _plain(value: Any) -> Any:
    return json.loads(json.dumps(value, default=_default, allow_nan=False))


def _default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


# ---- shared pipeline construction ----


def build_pipeline(
    train: pd.DataFrame,
    target: str,
    id_columns: tuple[str, ...],
    task: Task,
    estimator: str,
    params: dict[str, Any],
    seed: int,
    n_jobs: int,
    device: str,
    native_categoricals: bool | None,
    prep_cfg: PreprocessingConfig,
) -> tuple[Pipeline, ColumnPlan, dict[str, Any], Any, bool]:
    """Return the unfitted preprocessing-plus-estimator pipeline, plan, fit params, device choice, native flag."""
    spec = get_estimator_spec(estimator)
    spec.require()
    native = spec.native_default if native_categoricals is None else native_categoricals
    if native and not spec.supports_native_categoricals:
        raise RefitError(f"setting 'native_categoricals' is true but '{spec.name}' has no native categorical support")
    plan = infer_columns(train, target, id_columns, prep_cfg)
    layout = FeatureLayout(len(plan.numeric), len(plan.categorical), native)
    choice = resolve_device(spec, device)
    est = spec.make(task, params, seed, n_jobs, choice.effective, layout)
    pipeline = Pipeline([("pre", build_preprocessor(plan, prep_cfg, task, native_categoricals=native)), ("model", est)])
    fit_params = {f"model__{k}": v for k, v in spec.fit_params(layout).items()}
    return pipeline, plan, fit_params, choice, native


# ---- selection ----


def select_pipeline(
    source: Any = None,
    *,
    label: str | None = None,
    rank: int | None = None,
    estimator: str | None = None,
    params: dict[str, Any] | None = None,
    task: Any = None,
    metric: str | None = None,
    preprocessing_config: PreprocessingConfig | None = None,
    seed: int = 42,
    n_jobs: int = 1,
    device: str = "cpu",
    native_categoricals: bool | None = None,
) -> PipelineSelection:
    """Choose the pipeline to refit from a saved result (directory or loaded result) or explicitly.

    With a source holding several candidates and no `label` or `rank`, rank 1 is used and a note says so.
    """
    if isinstance(source, PipelineSelection):
        return source
    if (source is None) == (estimator is None):
        raise RefitError("pass either a saved result `source` or an explicit `estimator`, not both or neither")
    if source is not None:
        return _from_source(source, label, rank)
    if label is not None or rank is not None:
        raise RefitError("'label' and 'rank' apply only to a saved result source")
    return _explicit(estimator, params or {}, task, metric, preprocessing_config, seed, n_jobs, device, native_categoricals)


def _from_source(source: Any, label: str | None, rank: int | None) -> PipelineSelection:
    if label is not None and rank is not None:
        raise RefitError("pass either 'label' or 'rank', not both")
    if isinstance(source, (str, Path)) and (Path(source) / "ensemble.json").is_file():
        if label is not None or rank is not None:
            raise RefitError("an ensemble result contains one selected ensemble; 'label' and 'rank' do not apply")
        from automl.ensemble import load_ensemble

        try:
            metadata = load_ensemble(source).metadata
        except EnsembleError as exc:
            raise RefitError(exc.problems) from None
        return PipelineSelection(
            source_kind="ensemble",
            source_path=str(Path(source)),
            label=f"ensemble:{metadata['method']}",
            estimator="ensemble",
            params={"candidates": list(metadata["candidates"])},
            preprocessing_config=PreprocessingConfig(),
            task=metadata["task"],
            metric=metadata["metric"]["name"],
            direction=metadata["metric"]["direction"],
            seed=metadata["config"]["seed"],
            n_jobs=metadata["config"]["n_jobs"],
            device="cpu",
            native_categoricals=None,
            features=tuple(metadata["feature_columns"]),
            validation_mean=metadata["summary"]["mean"],
            validation_std=metadata["summary"]["std"],
            notes=("using the saved ensemble fitted on all training rows",),
        )
    try:
        cset = collect_candidates(source)
    except DataContractError as exc:
        raise RefitError(exc.problems) from None
    notes: list[str] = []
    cands = cset.candidates
    if label is not None:
        match = [c for c in cands if c.label == label]
        if not match:
            skipped = [f"{e.label}: {e.reason}" for e in cset.excluded if e.label == label]
            raise RefitError(
                [f"no candidate labelled {label!r}; available: {[c.label for c in cands]}", *skipped]
            )
        cand = match[0]
    else:
        n = 1 if rank is None else rank
        if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= len(cands):
            raise RefitError(f"'rank' must be an integer from 1 to {len(cands)}, got {rank!r}")
        cand = cands[n - 1]
        if rank is None and len(cands) > 1:
            notes.append(f"{len(cands)} candidates; selected rank 1 ({cand.label}) by validation score")
    b = cand.build
    try:
        prep = PreprocessingConfig(**b["preprocessing"])
    except DataContractError as exc:
        raise RefitError(exc.problems) from None
    spec = get_estimator_spec(cand.estimator_key)
    spec.require()
    return PipelineSelection(
        source_kind=cand.source_kind,
        source_path=cand.source_path,
        label=cand.label,
        estimator=cand.estimator_key,
        params=dict(b["estimator_params"]),
        preprocessing_config=prep,
        task=b["task"],
        metric=b["metric"],
        direction=b["direction"],
        seed=b["seed"],
        n_jobs=b["n_jobs"],
        device=b["device"],
        native_categoricals=b["native_categoricals"],
        features=tuple(b["features"]),
        trial_number=cand.best_trial,
        validation_mean=cand.mean,
        validation_std=cand.std,
        versions=dict(cand.versions),
        notes=tuple(notes),
    )


def _explicit(estimator, params, task, metric, prep, seed, n_jobs, device, native) -> PipelineSelection:
    problems: list[str] = []
    if estimator not in list_estimators():
        problems.append(f"estimator {estimator!r} is not supported; valid: {list_estimators()}")
    try:
        task_enum = Task(getattr(task, "value", task))
    except ValueError:
        task_enum = None
        problems.append(f"explicit selection needs 'task' to be one of {[t.value for t in Task]}, got {task!r}")
    direction = None
    if metric is not None:
        try:
            direction = get_metric(metric).direction
        except DataContractError as exc:
            problems.extend(exc.problems)
    if device not in DEVICES:
        problems.append(f"'device' has unsupported value {device!r}; valid: {list(DEVICES)}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        problems.append("'seed' must be an integer")
    if not isinstance(params, dict):
        problems.append("'params' must be a mapping of parameter names to values")
    else:
        problems.extend(controlled_param_problems(params))
    if problems:
        raise RefitError(problems)
    prep = prep or PreprocessingConfig(encoder="ordinal", seed=seed)
    spec = get_estimator_spec(estimator)
    spec.require()
    try:  # builds the estimator once to reject parameters it does not accept
        spec.make(task_enum, params, seed, n_jobs, "cpu", FeatureLayout(0, 0))
    except DataContractError as exc:
        raise RefitError(exc.problems) from None
    except TypeError as exc:
        raise RefitError(f"estimator parameters {sorted(params)} were rejected: {exc}") from None
    return PipelineSelection(
        source_kind="explicit",
        source_path=None,
        label=f"explicit:{estimator}",
        estimator=estimator,
        params=dict(params),
        preprocessing_config=prep,
        task=task_enum.value,
        metric=metric,
        direction=direction,
        seed=seed,
        n_jobs=n_jobs,
        device=device,
        native_categoricals=native,
    )


# ---- refit ----


def refit(
    selection: PipelineSelection,
    train: pd.DataFrame,
    *,
    target: str,
    id_columns: tuple[str, ...] = (),
    resolved_task: Any,
    progress: ProgressCallback | None = None,
    verbose: bool = False,
) -> RefitPipeline:
    """Fit preprocessing and estimator together on every training row."""
    progress_started = time.perf_counter()
    if progress is not None:
        emit_progress(progress, "refit", "stage.start", progress_started, model=selection.estimator)
    id_columns = tuple(id_columns)
    task = resolved_task.task
    problems: list[str] = []
    if target not in train.columns:
        problems.append(f"target column '{target}' is not in the train data")
    elif train[target].isna().any():
        problems.append(f"target column '{target}' has {int(train[target].isna().sum())} null values")
    if len(train) == 0:
        problems.append("train data has no rows")
    if task.value != selection.task:
        problems.append(f"selection was made for task '{selection.task}' but the resolved task is '{task.value}'")
    name = getattr(resolved_task.metric, "name", None)
    if selection.metric is not None and name != selection.metric:
        problems.append(f"selection was scored with metric '{selection.metric}' but the resolved metric is '{name}'")
    if problems:
        raise RefitError(problems)
    if selection.source_kind == "ensemble":
        if selection.source_path is None:
            raise RefitError("ensemble selection has no saved source path")
        from automl.ensemble import load_ensemble

        try:
            loaded = load_ensemble(selection.source_path, load_model=True)
        except EnsembleError as exc:
            raise RefitError(exc.problems) from None
        if loaded.model is None:
            raise RefitError("ensemble result does not contain a fitted model; recreate it with save_model=true")
        if loaded.metadata.get("n_rows") != len(train):
            raise RefitError(
                f"saved ensemble was fitted on {loaded.metadata.get('n_rows')} training rows; current data has {len(train)}"
            )
        if loaded.metadata.get("data_sha256") not in {
            data_fingerprint(train),
            legacy_data_fingerprint(train),
        }:
            raise RefitError("saved ensemble was fitted on different training data; recreate the ensemble for this run")
        missing = [column for column in selection.features or () if column not in train.columns]
        if missing:
            raise RefitError(f"train data is missing ensemble feature columns {missing}")
        plan = ColumnPlan(
            numeric=tuple(
                column for column in selection.features or () if loaded.metadata["feature_roles"][column] == "numeric"
            ),
            categorical=tuple(
                column
                for column in selection.features or ()
                if loaded.metadata["feature_roles"][column] == "categorical"
            ),
            dropped=(),
            excluded=tuple((*id_columns, target)),
        )
        classes = tuple(loaded.metadata.get("classes", ()))
        result = RefitPipeline(
            pipeline=loaded.model,
            selection=selection,
            plan=plan,
            target=target,
            id_columns=id_columns,
            task=task,
            classes=classes,
            n_train=len(train),
            fit_seconds=0.0,
            device={"requested": "cpu", "effective": "cpu", "reason": "loaded saved ensemble", "note": None},
            native_categoricals=False,
            training_data_sha256=data_fingerprint(train),
        )
        if progress is not None:
            emit_progress(progress, "refit", "stage.complete", progress_started, source_kind="ensemble")
        return result
    if task is not Task.REGRESSION and train[target].nunique() < 2:
        raise RefitError(f"target column '{target}' has fewer than two classes")

    pipeline, plan, fit_params, choice, native = build_pipeline(
        train, target, id_columns, task, selection.estimator, selection.params, selection.seed,
        selection.n_jobs, selection.device, selection.native_categoricals, selection.preprocessing_config,
    )
    native_verbosity = enable_native_verbosity(pipeline.named_steps["model"]) if verbose else False
    features = [*plan.numeric, *plan.categorical]
    if selection.features is not None and features != list(selection.features):
        missing = sorted(set(selection.features) - set(features))
        extra = sorted(set(features) - set(selection.features))
        raise RefitError(
            f"train columns differ from the selected result's features: missing {missing}, unexpected {extra}"
            if missing or extra
            else "train column order differs from the selected result's features"
        )
    t0 = time.perf_counter()
    if progress is not None:
        emit_progress(
            progress,
            "refit",
            "fit.start",
            progress_started,
            model=selection.estimator,
            native_verbosity=("enabled" if native_verbosity else "unsupported") if verbose else None,
        )
    pipeline.fit(train[features], train[target], **fit_params)
    seconds = time.perf_counter() - t0
    classes = tuple(np.asarray(pipeline.named_steps["model"].classes_).tolist()) if task is not Task.REGRESSION else ()
    result = RefitPipeline(
        pipeline=pipeline,
        selection=selection,
        plan=plan,
        target=target,
        id_columns=id_columns,
        task=task,
        classes=classes,
        n_train=len(train),
        fit_seconds=seconds,
        device={"requested": choice.requested, "effective": choice.effective, "reason": choice.reason},
        native_categoricals=native,
        training_data_sha256=data_fingerprint(train),
    )
    if progress is not None:
        emit_progress(progress, "refit", "fit.complete", progress_started, fit_seconds=seconds)
        emit_progress(progress, "refit", "stage.complete", progress_started)
    return result


# ---- prediction ----


def predict(
    fitted: RefitPipeline,
    test: pd.DataFrame,
    *,
    id_columns: tuple[str, ...] | None = None,
    config: RefitConfig | None = None,
    progress: ProgressCallback | None = None,
) -> PredictionResult:
    """Predict the test rows in order; validation failures raise before anything is returned."""
    cfg = config or RefitConfig()
    ids = tuple(fitted.id_columns if id_columns is None else id_columns)
    problems: list[str] = []
    if len(test) == 0:
        problems.append("test data has no rows")
    missing = [c for c in fitted.features if c not in test.columns]
    if missing:
        problems.append(f"test data is missing feature columns {missing}")
    missing_ids = [c for c in ids if c not in test.columns]
    if missing_ids:
        problems.append(f"test data is missing id columns {missing_ids}")
    reserved = [c for c in ids if c == _PREDICTION_COLUMN or c.startswith(_PROBA_PREFIX)]
    if reserved:
        problems.append(f"id columns {reserved} collide with output column names")
    if problems:
        raise RefitError(problems)

    X = test[fitted.features]
    progress_started = time.perf_counter()
    if progress is not None:
        emit_progress(progress, "refit", "predict.start", progress_started, n_rows=len(test))
    notes: list[str] = []
    values = np.asarray(fitted.pipeline.predict(X))
    proba = None
    if fitted.task is not Task.REGRESSION:
        if not cfg.probabilities:
            notes.append("probabilities not produced: 'probabilities' is false")
        elif hasattr(fitted.pipeline, "predict_proba"):
            proba = np.asarray(fitted.pipeline.predict_proba(X))
        else:
            notes.append("probabilities not produced: the estimator has no predict_proba")

    problems = _output_problems(values, proba, len(test), fitted, cfg)
    if problems:
        raise RefitError(problems)

    frame = test[list(ids)].reset_index(drop=True).copy()
    frame[_PREDICTION_COLUMN] = values
    proba_cols: tuple[str, ...] = ()
    if proba is not None:
        proba_cols = tuple(f"{_PROBA_PREFIX}{c}" for c in fitted.classes)
        for i, col in enumerate(proba_cols):
            frame[col] = proba[:, i]
    kind = "values" if fitted.task is Task.REGRESSION else ("labels+proba" if proba is not None else "labels")
    result = PredictionResult(frame, fitted.task, kind, len(frame), fitted.classes, proba_cols, ids, tuple(notes))
    if progress is not None:
        emit_progress(progress, "refit", "predict.complete", progress_started, n_rows=len(test))
    return result


def _rows(mask: np.ndarray, cfg: RefitConfig) -> str:
    idx = np.flatnonzero(mask)
    shown = ", ".join(map(str, idx[: cfg.max_reported_rows]))
    more = f" (+{len(idx) - cfg.max_reported_rows} more)" if len(idx) > cfg.max_reported_rows else ""
    return f"{len(idx)} rows, first positions: {shown}{more}"


def _output_problems(values: np.ndarray, proba: np.ndarray | None, n: int, fitted: RefitPipeline, cfg: RefitConfig) -> list[str]:
    problems: list[str] = []
    if values.shape[0] != n:
        problems.append(f"expected {n} predictions, got {values.shape[0]}")
    elif values.ndim != 1:
        problems.append(f"expected one prediction per row, got shape {values.shape}")
    elif fitted.task is Task.REGRESSION:
        bad = ~np.isfinite(values.astype(float))
        if bad.any():
            problems.append(f"predictions are not finite in {_rows(bad, cfg)}")
    else:
        bad = ~pd.Series(values).isin(list(fitted.classes)).to_numpy()
        if bad.any():
            problems.append(f"predicted labels outside the training classes {list(fitted.classes)} in {_rows(bad, cfg)}")
    if proba is not None:
        if proba.shape != (n, len(fitted.classes)):
            problems.append(f"expected probabilities of shape {(n, len(fitted.classes))}, got {proba.shape}")
        else:
            finite = np.isfinite(proba).all(axis=1)
            if not finite.all():
                problems.append(f"probabilities are not finite in {_rows(~finite, cfg)}")
            tol = cfg.proba_tolerance
            bad = finite & ((proba < -tol) | (proba > 1 + tol)).any(axis=1)
            if bad.any():
                problems.append(f"probabilities outside [0, 1] in {_rows(bad, cfg)}")
            bad = finite & (np.abs(proba.sum(axis=1) - 1) > tol)
            if bad.any():
                problems.append(f"probabilities do not sum to 1 within {tol} in {_rows(bad, cfg)}")
    return problems


# ---- persistence ----


def _metadata(fitted: RefitPipeline, preds: PredictionResult, cfg: RefitConfig, n_test: int, saved_model: bool) -> dict[str, Any]:
    return _plain(
        {
            "schema_version": REFIT_SCHEMA_VERSION,
            "task": fitted.task.value,
            "selection": fitted.selection.to_dict(),
            "config": asdict(cfg),
            "data": {
                "target": fitted.target,
                "id_columns": list(fitted.id_columns),
                "n_train_rows": fitted.n_train,
                "n_test_rows": n_test,
                "numeric": list(fitted.plan.numeric),
                "categorical": list(fitted.plan.categorical),
                "excluded": list(fitted.plan.excluded),
                "dropped": list(fitted.plan.dropped),
                "classes": list(fitted.classes),
            },
            "prediction": preds.to_dict(),
            "device": fitted.device,
            "categorical_handling": "native" if fitted.native_categoricals else "encoded",
            "fit_seconds": fitted.fit_seconds,
            "versions": {**_versions(), **fitted.selection.versions},
            "files": {
                "predictions": PREDICTIONS_FILE,
                "metadata": METADATA_FILE,
                "model": MODEL_FILE if saved_model else None,
                "fitted_model": "fitted-model" if saved_model else None,
            },
            "model_reference": "fitted-model" if saved_model else None,
            "note": "Predictions are raw model outputs; validation scores in the selection are not leaderboard estimates.",
        }
    )


def save_refit(
    fitted: RefitPipeline,
    predictions: PredictionResult,
    directory: str | Path,
    *,
    config: RefitConfig | None = None,
    overwrite: bool = False,
) -> Path:
    """Write predictions.csv, refit.json, and (by config) pipeline.joblib; refuse to overwrite by default."""
    cfg = config or RefitConfig()
    out = Path(directory)
    names = (PREDICTIONS_FILE, METADATA_FILE, MODEL_FILE)
    existing = [n for n in names if (out / n).exists()]
    if (out / "fitted-model").exists():
        existing.append("fitted-model")
    if existing and not overwrite:
        raise RefitError(f"{out} already contains {existing}; pass overwrite=True or choose another directory")
    out.mkdir(parents=True, exist_ok=True)
    meta = _metadata(fitted, predictions, cfg, predictions.n_rows, cfg.save_model)
    text = json.dumps(meta, indent=2, allow_nan=False)
    tmp = Path(tempfile.mkdtemp(prefix=".refit-", dir=out))
    managed_names = (*names, "fitted-model")
    try:
        predictions.frame.to_csv(tmp / PREDICTIONS_FILE, index=False)
        (tmp / METADATA_FILE).write_text(text)
        if cfg.save_model:
            import joblib

            joblib.dump(fitted.pipeline, tmp / MODEL_FILE)
        if cfg.save_model:
            save_fitted_model(
                fitted.pipeline,
                tmp / "fitted-model",
                task=fitted.task.value,
                metric=fitted.selection.metric,
                direction=fitted.selection.direction,
                feature_columns=fitted.features,
                class_order=list(fitted.classes),
                effective_config=fitted.selection.to_dict(),
                dependency_versions={**_versions(), **fitted.selection.versions},
                training_data_sha256=fitted.training_data_sha256,
                n_training_rows=fitted.n_train,
                overwrite=False,
            )
        backup = tmp / "backup"
        backup.mkdir()
        backed_up: list[str] = []
        installed: list[str] = []
        try:
            for name in managed_names:
                staged = tmp / name
                destination = out / name
                if destination.exists():
                    os.replace(destination, backup / name)
                    backed_up.append(name)
                if staged.exists():
                    os.replace(staged, destination)
                    installed.append(name)
        except Exception:
            for name in reversed(installed):
                destination = out / name
                if destination.is_dir() and not destination.is_symlink():
                    shutil.rmtree(destination)
                else:
                    destination.unlink(missing_ok=True)
            for name in reversed(backed_up):
                os.replace(backup / name, out / name)
            raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


_REQUIRED_KEYS = ("task", "selection", "data", "prediction", "files")


def load_refit(directory: str | Path, load_model: bool = False) -> LoadedRefit:
    """Read refit.json and predictions.csv without estimator libraries; load the model only on request.

    The model file is a pickle: load it only from a directory you trust.
    """
    out = Path(directory)
    meta_path = out / METADATA_FILE
    if not meta_path.is_file():
        raise RefitError(f"{meta_path} not found; pass a directory written by save_refit")
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError as exc:
        raise RefitError(f"{meta_path} is not valid JSON: {exc}") from None
    if not isinstance(meta, dict) or meta.get("schema_version") != REFIT_SCHEMA_VERSION:
        got = meta.get("schema_version") if isinstance(meta, dict) else None
        raise RefitError(f"{meta_path} has unsupported schema_version {got!r}; expected {REFIT_SCHEMA_VERSION}")
    missing = [k for k in _REQUIRED_KEYS if k not in meta]
    if missing:
        raise RefitError(f"{meta_path} is missing keys {missing}")
    pred_path = out / PREDICTIONS_FILE
    if not pred_path.is_file():
        raise RefitError(f"{pred_path} not found")
    predictions = pd.read_csv(pred_path)
    model = None
    if load_model:
        model_path = out / MODEL_FILE
        if not model_path.is_file():
            raise RefitError(f"{model_path} not found; the refit was saved with save_model=false")
        import joblib

        try:
            model = joblib.load(model_path)
        except ImportError as exc:
            sel = meta["selection"]
            extra = get_estimator_spec(sel["estimator"]).extra if sel.get("estimator") in list_estimators() else None
            hint = f"; install it with: pip install 'automl[{extra}]'" if extra else ""
            raise RefitError(f"the saved pipeline needs a library that is not installed ({exc}){hint}") from None
    return LoadedRefit(meta, predictions, model)


# ---- composition ----


def refit_and_predict(
    source: Any,
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target: str,
    id_columns: tuple[str, ...] = (),
    resolved_task: Any,
    label: str | None = None,
    rank: int | None = None,
    config: RefitConfig | None = None,
    directory: str | Path | None = None,
    overwrite: bool = False,
    tracking: Any = None,
    progress: ProgressCallback | None = None,
    verbose: bool = False,
) -> RefitResult:
    """Select, refit, predict, and optionally save and log to MLflow.

    `source` is a saved result directory, a loaded result, or a `PipelineSelection`.
    """
    cfg = config or RefitConfig()
    if tracking is not None and tracking.enabled:
        from automl.tracking import _require_mlflow

        _require_mlflow()
    selection = select_pipeline(source, label=label, rank=rank)
    fitted = refit(
        selection,
        train,
        target=target,
        id_columns=id_columns,
        resolved_task=resolved_task,
        progress=progress,
        verbose=verbose,
    )
    preds = predict(fitted, test, config=cfg, progress=progress)
    out = save_refit(fitted, preds, directory, config=cfg, overwrite=overwrite) if directory is not None else None
    tracked = None
    if tracking is not None and tracking.enabled:
        from automl.tracking import track_refit

        meta = _metadata(fitted, preds, cfg, preds.n_rows, out is not None and cfg.save_model)
        tracked = track_refit(fitted.pipeline, train[fitted.features], meta, tracking, directory=out)
    return RefitResult(fitted, preds, out, tracked)
