"""Classification voting and task-aware stacking from saved model candidates."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from automl.errors import DataContractError, EnsembleError, ReportError, TaskMetricError
from automl.estimators import enable_native_verbosity
from automl.metrics import Task, get_metric
from automl.model_report import Candidate, CandidateSet, collect_candidates
from automl.model_store import FittedModelReference, data_fingerprint as fitted_data_fingerprint, load_fitted_model
from automl.preprocessing import PreprocessingConfig, infer_columns
from automl.progress import ProgressCallback, emit_progress
from automl.refit import build_pipeline
from automl.validation import SplitResult

ENSEMBLE_SCHEMA_VERSION = 1
RESULT_FILE = "ensemble.json"
MODEL_FILE = "ensemble.joblib"
METHODS = ("voting", "stacking")


@dataclass(frozen=True)
class EnsembleConfig:
    method: str = "stacking"
    seed: int = 42
    n_jobs: int = 1
    stack_cv: int = 5
    meta_estimator: str | None = None
    meta_estimator_params: dict[str, Any] = field(default_factory=dict)
    save_model: bool = True

    def __post_init__(self) -> None:
        problems: list[str] = []
        if self.method not in METHODS:
            problems.append(f"setting 'method' has unsupported value {self.method!r}; valid: {list(METHODS)}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            problems.append("setting 'seed' must be an integer")
        if isinstance(self.n_jobs, bool) or not isinstance(self.n_jobs, int) or (self.n_jobs != -1 and self.n_jobs < 1):
            problems.append("setting 'n_jobs' must be -1 or an integer of at least 1")
        if isinstance(self.stack_cv, bool) or not isinstance(self.stack_cv, int) or self.stack_cv < 2:
            problems.append("setting 'stack_cv' must be an integer of at least 2")
        if self.meta_estimator is not None and not isinstance(self.meta_estimator, str):
            problems.append("setting 'meta_estimator' must be a string or None")
        if not isinstance(self.meta_estimator_params, dict) or not all(
            isinstance(key, str) for key in self.meta_estimator_params
        ):
            problems.append("setting 'meta_estimator_params' must be a table of parameter names to values")
        if not isinstance(self.save_model, bool):
            problems.append("setting 'save_model' must be true or false")
        if problems:
            raise EnsembleError(problems)
        object.__setattr__(self, "meta_estimator_params", dict(self.meta_estimator_params))


ENSEMBLE_KEYS = {item.name for item in fields(EnsembleConfig)}


class _CandidateEstimator(BaseEstimator):
    def __init__(self, estimator: Any, fit_params: dict[str, Any]) -> None:
        self.estimator = estimator
        self.fit_params = fit_params

    def fit(self, X: Any, y: Any):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y, **self.fit_params)
        if hasattr(self.estimator_, "classes_"):
            self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X: Any):
        return self.estimator_.predict(X)


class _CandidateClassifier(ClassifierMixin, _CandidateEstimator):
    def predict_proba(self, X: Any):
        return self.estimator_.predict_proba(X)


class _CandidateRegressor(RegressorMixin, _CandidateEstimator):
    pass


@dataclass(frozen=True)
class EnsembleFold:
    fold: int
    n_train: int
    n_valid: int
    score: float
    fit_seconds: float
    predict_seconds: float


@dataclass(frozen=True, eq=False)
class EnsembleResult:
    task: Task
    metric: str
    direction: str
    method: str
    candidates: tuple[str, ...]
    candidate_details: dict[str, dict[str, Any]]
    candidate_fold_scores: dict[str, tuple[float, ...]]
    folds: tuple[EnsembleFold, ...]
    mean: float | None
    std: float | None
    n_rows: int
    data_sha256: str
    splits: dict[str, Any]
    config: EnsembleConfig
    meta_estimator: str | None
    feature_columns: tuple[str, ...]
    feature_roles: dict[str, str]
    classes: tuple[Any, ...]
    model: Any = field(default=None, repr=False, compare=False)
    directory: Path | None = None
    validation_status: str = "validated"
    validation_limitation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ENSEMBLE_SCHEMA_VERSION,
            "task": self.task.value,
            "metric": {"name": self.metric, "direction": self.direction},
            "method": self.method,
            "candidates": list(self.candidates),
            "candidate_details": _plain(self.candidate_details),
            "candidate_scores": {
                label: {
                    "fold_scores": list(scores),
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                }
                for label, scores in self.candidate_fold_scores.items()
            },
            "folds": [asdict(fold) for fold in self.folds],
            "summary": {"mean": self.mean, "std": self.std},
            "n_rows": self.n_rows,
            "data_sha256": self.data_sha256,
            "splits": self.splits,
            "config": _plain(asdict(self.config)),
            "meta_estimator": self.meta_estimator,
            "feature_columns": list(self.feature_columns),
            "feature_roles": self.feature_roles,
            "classes": _plain(list(self.classes)),
            "files": {"model": MODEL_FILE if self.config.save_model else None},
            "validation_status": self.validation_status,
            "validation_limitation": self.validation_limitation,
            "score_statement": (
                "Scores are validation scores on the training data, optimistic after selection, "
                "and are not leaderboard predictions."
                if self.validation_status == "validated"
                else "No ensemble validation score is reported because the saved artifacts cannot prove a leakage-safe score."
            ),
        }


@dataclass(frozen=True, eq=False)
class LoadedEnsemble:
    metadata: dict[str, Any]
    model: Any = None


def _plain(value: Any) -> Any:
    return json.loads(json.dumps(value, default=_json_default, allow_nan=False))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def data_fingerprint(frame: pd.DataFrame) -> str:
    """Hash ordered column names, dtypes, and row values without persisting raw data."""
    return fitted_data_fingerprint(frame)


def _candidate_set(sources: Any, labels: tuple[str, ...] | None) -> CandidateSet:
    try:
        cset = collect_candidates(sources)
    except (DataContractError, ReportError) as exc:
        raise EnsembleError(exc.problems) from None
    candidates = cset.candidates
    if labels is not None:
        if not labels:
            raise EnsembleError("candidate labels must contain at least one label")
        repeated = sorted({label for label in labels if labels.count(label) > 1})
        if repeated:
            raise EnsembleError(f"candidate labels contain duplicates: {repeated}")
        missing = [label for label in labels if label not in {candidate.label for candidate in candidates}]
        if missing:
            raise EnsembleError(
                [f"unknown candidate label(s) {missing}", f"available labels: {[c.label for c in candidates]}"]
            )
        candidates = tuple(next(c for c in candidates if c.label == label) for label in labels)
    if len(candidates) < 2:
        raise EnsembleError("an ensemble needs at least two compatible model candidates")
    if len({c.label for c in candidates}) != len(candidates):
        raise EnsembleError("candidate labels must be unique")
    identities = [
        (
            str(Path(c.source_path).resolve()) if c.source_path != "<in-memory>" else c.source_path,
            c.estimator_key,
            json.dumps(c.params, sort_keys=True, default=str),
            c.best_trial,
        )
        for c in candidates
    ]
    if len(set(identities)) != len(identities):
        raise EnsembleError("candidate selection contains duplicate model results")
    return CandidateSet(
        cset.task, cset.metric, cset.direction, tuple(candidates), cset.excluded, cset.warnings, cset.notes
    )


def _check_candidates(
    candidates: tuple[Candidate, ...],
    train: pd.DataFrame,
    target: str,
    id_columns: tuple[str, ...],
    task: Task,
    metric_name: str,
    config: EnsembleConfig,
) -> tuple[tuple[str, ...], dict[str, str]]:
    problems: list[str] = []
    available_features = set(train.columns) - {*id_columns, target}
    used_features: list[str] = []
    feature_roles: dict[str, str] = {}
    for candidate in candidates:
        build = candidate.build
        if build.get("task") != task.value or candidate.n_rows != len(train):
            problems.append(
                f"{candidate.label} was saved for task {build.get('task')!r} and {candidate.n_rows} rows; "
                f"current data has task {task.value!r} and {len(train)} rows"
            )
        if build.get("metric") != metric_name or build.get("direction") != get_metric(metric_name).direction:
            problems.append(f"{candidate.label} was evaluated with an incompatible metric")
        candidate_features = tuple(build.get("features", ()))
        if not candidate_features or not set(candidate_features) <= available_features:
            problems.append(
                f"{candidate.label} features {list(candidate_features)} are empty or missing from the current data"
            )
        else:
            used_features.extend(column for column in candidate_features if column not in used_features)
        required = {"estimator_params", "seed", "n_jobs", "device", "preprocessing", "features"}
        missing = sorted(required - set(build))
        if missing:
            problems.append(f"{candidate.label} is missing candidate build fields {missing}")
        try:
            prep_config = PreprocessingConfig(**build["preprocessing"])
            plan = infer_columns(train, target, id_columns, prep_config)
            inferred_features = (*plan.numeric, *plan.categorical)
            if inferred_features != candidate_features:
                problems.append(
                    f"{candidate.label} features do not match the saved preprocessing column plan"
                )
            for column in plan.numeric:
                feature_roles.setdefault(column, "numeric")
            for column in plan.categorical:
                feature_roles.setdefault(column, "categorical")
        except (KeyError, DataContractError) as exc:
            problems.append(f"{candidate.label} has invalid preprocessing metadata: {exc}")
    if config.method == "voting":
        if task is Task.REGRESSION:
            problems.append("majority voting is classification-only; use method='stacking' for regression")
        elif get_metric(metric_name).needs == "proba" and metric_name != "roc_auc":
            problems.append(
                f"hard majority voting produces labels, not probabilities, and cannot be scored with "
                f"'{metric_name}'; use a label metric or method='stacking'"
            )
    if problems:
        raise EnsembleError(problems)
    return tuple(column for column in train.columns if column in used_features), feature_roles


def _meta_estimator(task: Task, config: EnsembleConfig, custom: Any = None) -> Any:
    if config.method != "stacking":
        if custom is not None or config.meta_estimator is not None or config.meta_estimator_params:
            raise EnsembleError("meta-estimator settings apply only when method='stacking'")
        return None
    if custom is not None and config.meta_estimator is not None:
        raise EnsembleError("pass a custom meta_estimator or set config.meta_estimator, not both")
    if custom is not None:
        try:
            estimator = clone(custom)
        except (TypeError, RuntimeError) as exc:
            raise EnsembleError(f"meta_estimator must be a cloneable scikit-learn estimator: {exc}") from None
        if config.meta_estimator_params:
            try:
                estimator.set_params(**config.meta_estimator_params)
            except (TypeError, ValueError) as exc:
                raise EnsembleError(f"invalid parameters for custom meta-estimator: {exc}") from None
        return estimator
    name = config.meta_estimator or ("logistic_regression" if task is not Task.REGRESSION else "ridge_cv")
    choices = {
        "logistic_regression": ({Task.BINARY, Task.MULTICLASS}, LogisticRegression),
        "decision_tree_classifier": ({Task.BINARY, Task.MULTICLASS}, DecisionTreeClassifier),
        "random_forest_classifier": ({Task.BINARY, Task.MULTICLASS}, RandomForestClassifier),
        "ridge_cv": ({Task.REGRESSION}, RidgeCV),
        "ridge": ({Task.REGRESSION}, Ridge),
        "decision_tree_regressor": ({Task.REGRESSION}, DecisionTreeRegressor),
        "random_forest_regressor": ({Task.REGRESSION}, RandomForestRegressor),
    }
    if name not in choices:
        raise EnsembleError(f"unsupported meta-estimator {name!r}; valid: {sorted(choices)}")
    tasks, estimator = choices[name]
    if task not in tasks:
        raise EnsembleError(f"meta-estimator {name!r} does not support task {task.value!r}")
    params = dict(config.meta_estimator_params)
    if estimator in (
        LogisticRegression,
        DecisionTreeClassifier,
        DecisionTreeRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    ) and "random_state" not in params:
        params["random_state"] = config.seed
    if estimator is LogisticRegression:
        params.setdefault("max_iter", 1000)
    try:
        return estimator(**params)
    except (TypeError, ValueError) as exc:
        raise EnsembleError(f"invalid parameters for meta-estimator {name!r}: {exc}") from None


def _build_candidates(
    candidates: tuple[Candidate, ...], train: pd.DataFrame, target: str, ids, task, fallback_seed, verbose=False
):
    estimators = []
    for idx, candidate in enumerate(candidates):
        build = candidate.build
        try:
            pipeline, _, params, _, _ = build_pipeline(
                train,
                target,
                ids,
                task,
                candidate.estimator_key,
                dict(build["estimator_params"]),
                int(build.get("seed") if build.get("seed") is not None else fallback_seed),
                int(build.get("n_jobs", 1)),
                str(build.get("device", "cpu")),
                build.get("native_categoricals"),
                PreprocessingConfig(**build["preprocessing"]),
            )
        except (DataContractError, KeyError, TypeError, ValueError) as exc:
            raise EnsembleError(f"cannot rebuild candidate {candidate.label!r}: {exc}") from None
        if verbose:
            enable_native_verbosity(pipeline.named_steps["model"])
        name = f"candidate_{idx}"
        wrapper = _CandidateRegressor if task is Task.REGRESSION else _CandidateClassifier
        estimators.append((name, wrapper(pipeline, params)))
    return estimators


def _make_model(estimators, task, config, meta):
    if config.method == "voting":
        return VotingClassifier(estimators=estimators, voting="hard", n_jobs=config.n_jobs)
    if task is Task.REGRESSION:
        return StackingRegressor(
            estimators=estimators,
            final_estimator=clone(meta),
            cv=KFold(n_splits=config.stack_cv, shuffle=True, random_state=config.seed),
            n_jobs=config.n_jobs,
        )
    return StackingClassifier(
        estimators=estimators,
        final_estimator=clone(meta),
        cv=StratifiedKFold(n_splits=config.stack_cv, shuffle=False),
        n_jobs=config.n_jobs,
    )


def _score(y_true, model, X, task: Task, metric_name: str, method: str) -> float:
    metric = get_metric(metric_name)
    if task is Task.REGRESSION or metric.needs == "values":
        predictions = model.predict(X)
    elif metric.needs == "labels":
        predictions = model.predict(X)
    elif method == "voting":
        predictions = model.predict(X)
    else:
        predictions = model.predict_proba(X)
        if metric_name == "roc_auc" and task is Task.BINARY:
            predictions = predictions[:, 1]
    return float(metric(y_true, predictions))


class _SavedMemberEnsemble:
    def __init__(self, models: tuple[Any, ...], classes: tuple[Any, ...], method: str, task: Task, meta: Any = None):
        self.models = models
        self.classes_ = np.asarray(classes)
        self.method = method
        self.task = task
        self.meta = meta

    def _member_features(self, X: pd.DataFrame) -> np.ndarray:
        features = []
        for model in self.models:
            if self.task is Task.REGRESSION:
                features.append(np.asarray(model.predict(X), dtype=float).reshape(-1, 1))
                continue
            estimator = model.named_steps.get("model") if hasattr(model, "named_steps") else model
            model_classes = list(getattr(estimator, "classes_", ()))
            if hasattr(model, "predict_proba"):
                probabilities = np.asarray(model.predict_proba(X))
                if probabilities.ndim != 2 or probabilities.shape[1] != len(model_classes):
                    raise EnsembleError("saved member returned probabilities with an incompatible class shape")
                if set(model_classes) != set(self.classes_.tolist()):
                    raise EnsembleError("saved member classes do not match the ensemble class order")
                order = [model_classes.index(label) for label in self.classes_.tolist()]
                features.append(probabilities[:, order])
            else:
                labels = np.asarray(model.predict(X))
                features.append(np.column_stack([labels == label for label in self.classes_]).astype(float))
        return np.concatenate(features, axis=1)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.method == "stacking":
            return np.asarray(self.meta.predict(self._member_features(X)))
        votes = np.column_stack([model.predict(X) for model in self.models])
        return _majority_vote(votes, self.classes_)


class _SavedStackingClassifier(_SavedMemberEnsemble):
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self.meta, "predict_proba"):
            raise EnsembleError("the saved stacking meta-estimator does not support predict_proba")
        probabilities = np.asarray(self.meta.predict_proba(self._member_features(X)))
        meta_classes = list(self.meta.classes_)
        if set(meta_classes) != set(self.classes_.tolist()):
            raise EnsembleError("saved stacking meta-estimator class order differs from the base members")
        return probabilities[:, [meta_classes.index(label) for label in self.classes_.tolist()]]


def _oof_class_features(frame: pd.DataFrame, classes: tuple[Any, ...]) -> np.ndarray:
    probability_columns = [f"proba_{idx}" for idx in range(len(classes))]
    if set(probability_columns) <= set(frame.columns):
        return frame[probability_columns].to_numpy(dtype=float)
    labels = frame["prediction"].to_numpy()
    return np.column_stack([labels == label for label in classes]).astype(float)


def _majority_vote(votes: np.ndarray, classes: Any) -> np.ndarray:
    ordered = list(classes)
    return np.asarray(
        [
            max(ordered, key=lambda label: (int(np.count_nonzero(row == label)), -ordered.index(label)))
            for row in votes
        ]
    )


def _oof_metric_values(frame: pd.DataFrame, task: Task, metric_name: str) -> np.ndarray:
    metric = get_metric(metric_name)
    if task is Task.REGRESSION or metric.needs in ("values", "labels"):
        return frame["prediction"].to_numpy()
    probability_columns = [column for column in frame.columns if column.startswith("proba_")]
    if not probability_columns:
        raise EnsembleError(f"saved OOF predictions do not contain probabilities required by '{metric_name}'")
    probabilities = frame[[f"proba_{idx}" for idx in range(len(probability_columns))]].to_numpy(dtype=float)
    return probabilities[:, 1] if metric_name == "roc_auc" and task is Task.BINARY else probabilities


def run_saved_ensemble(
    train: pd.DataFrame,
    target: str,
    model_references: tuple[FittedModelReference | str | Path, ...] | list[FittedModelReference | str | Path],
    resolved_task: Any,
    config: EnsembleConfig | None = None,
    *,
    meta_estimator: Any = None,
    directory: str | Path | None = None,
    progress: ProgressCallback | None = None,
    tracking_uri: str | None = None,
) -> EnsembleResult:
    """Build an ensemble from saved full-data pipelines and fold-excluded OOF predictions.

    Saved members are loaded, never fitted. Stacking fits only its meta-estimator. A validation
    score is withheld for stacking because the available OOF artifacts cannot prove that the
    meta-estimator did not train on the rows being scored.
    """
    cfg = config or EnsembleConfig()
    task_value = getattr(resolved_task, "task", resolved_task)
    task = Task(getattr(task_value, "value", task_value))
    metric_name = getattr(getattr(resolved_task, "metric", None), "name", None)
    if not isinstance(metric_name, str):
        raise EnsembleError("resolved_task must include a metric name")
    if target not in train.columns or train[target].isna().any():
        raise EnsembleError(f"training target column {target!r} is missing or contains null values")
    refs = tuple(model_references)
    if len(refs) < 2:
        raise EnsembleError("an ensemble needs at least two fitted-model references")
    loaded_models = []
    for ref in refs:
        try:
            loaded_models.append(load_fitted_model(ref, tracking_uri=tracking_uri))
        except DataContractError as exc:
            raise EnsembleError(exc.problems) from None
    metadata = [loaded.metadata for loaded in loaded_models]
    reference_ids = [str(item["model_id"]) for item in metadata]
    if len(set(reference_ids)) != len(reference_ids):
        raise EnsembleError("model references must identify unique fitted models")
    fingerprint = data_fingerprint(train)
    expected_metric = get_metric(metric_name)
    first = metadata[0]
    expected_features = first["feature_columns"]
    expected_classes = first["class_order"]
    expected_splits = first.get("validation_splits")
    expected_split_hash = expected_splits.get("sha256") if isinstance(expected_splits, dict) else None
    problems = []
    for index, (item, loaded) in enumerate(zip(metadata, loaded_models)):
        label = f"model {reference_ids[index]}"
        if item.get("task") != task.value:
            problems.append(f"{label} was saved for task {item.get('task')!r}, not {task.value!r}")
        model_metric = item.get("metric") or {}
        if model_metric.get("name") != metric_name or model_metric.get("direction") != expected_metric.direction:
            problems.append(f"{label} was saved for an incompatible metric")
        if item.get("training_data_sha256") != fingerprint or item.get("n_training_rows") != len(train):
            problems.append(f"{label} was fitted on different training data")
        if item.get("fit_scope") != "full_data":
            problems.append(f"{label} does not contain a full-data pipeline for inference")
        if item.get("feature_columns") != expected_features:
            problems.append(f"{label} has an incompatible feature schema")
        if item.get("class_order") != expected_classes:
            problems.append(f"{label} has an incompatible class order")
        if task is not Task.REGRESSION:
            estimator = (
                loaded.pipeline.named_steps.get("model")
                if hasattr(loaded.pipeline, "named_steps")
                else loaded.pipeline
            )
            if list(getattr(estimator, "classes_", ())) != item.get("class_order"):
                problems.append(f"{label} pipeline classes do not match its saved class order")
        splits = item.get("validation_splits")
        if (
            not expected_split_hash
            or not isinstance(splits, dict)
            or splits.get("sha256") != expected_split_hash
            or splits.get("n_rows") != len(train)
        ):
            problems.append(f"{label} has incompatible or missing validation splits")
        oof_info = item.get("oof") or {}
        if not oof_info.get("available") or loaded.oof_predictions is None:
            problems.append(f"{label} has no saved OOF predictions")
        elif not oof_info.get("complete"):
            problems.append(f"{label} has incomplete OOF coverage")
    if problems:
        raise EnsembleError(problems)
    if not expected_features or not set(expected_features) <= set(train.columns):
        raise EnsembleError(f"training data is missing saved feature columns {expected_features}")
    feature_columns = tuple(expected_features)
    if cfg.method == "voting":
        if task is Task.REGRESSION:
            raise EnsembleError("majority voting is classification-only; use method='stacking' for regression")
        if expected_metric.needs == "proba" and metric_name != "roc_auc":
            raise EnsembleError(
                f"hard majority voting produces labels, not probabilities, and cannot be scored with "
                f"'{metric_name}'; use method='stacking'"
            )
    if task is not Task.REGRESSION and not expected_classes:
        raise EnsembleError("classification model references must declare a non-empty class order")

    split_info = expected_splits
    expected_rows = set(row for fold in split_info["folds"] for row in fold["valid_rows"])
    row_maps = []
    for loaded in loaded_models:
        frame = loaded.oof_predictions.copy()
        if set(frame["row"].astype(int)) != expected_rows:
            raise EnsembleError("saved OOF predictions do not cover the declared validation rows")
        row_maps.append(frame.set_index(frame["row"].astype(int), drop=False))
    if cfg.method == "stacking" and expected_rows != set(range(len(train))):
        raise EnsembleError(
            "stacking requires complete k-fold OOF coverage of every training row; holdout artifacts are insufficient"
        )

    classes = tuple(expected_classes)
    full_models = tuple(loaded.pipeline for loaded in loaded_models)
    labels = tuple(f"saved:{model_id}" for model_id in reference_ids)
    meta = _meta_estimator(task, cfg, meta_estimator)
    fitted_meta = None
    if cfg.method == "stacking":
        feature_parts = []
        for frame in row_maps:
            aligned = frame.loc[sorted(expected_rows)]
            features = (
                aligned[["prediction"]].to_numpy(dtype=float)
                if task is Task.REGRESSION
                else _oof_class_features(aligned, classes)
            )
            feature_parts.append(features)
        X_meta = np.concatenate(feature_parts, axis=1)
        y_meta = train.iloc[sorted(expected_rows)][target].to_numpy()
        fitted_meta = clone(meta)
        try:
            fitted_meta.fit(X_meta, y_meta)
        except (ValueError, TypeError) as exc:
            raise EnsembleError(f"fitting the stacking meta-estimator from saved OOF predictions failed: {exc}") from None
        final_model = _SavedStackingClassifier(full_models, classes, cfg.method, task, fitted_meta) if task is not Task.REGRESSION else _SavedMemberEnsemble(full_models, classes, cfg.method, task, fitted_meta)
        validation_status = "unavailable"
        limitation = (
            "saved OOF predictions support meta-model fitting, but do not establish an independent outer evaluation "
            "for the meta-model; no validation score is reported"
        )
        folds: tuple[EnsembleFold, ...] = ()
        mean = std = None
        candidate_scores: dict[str, tuple[float, ...]] = {}
    else:
        final_model = _SavedMemberEnsemble(full_models, classes, cfg.method, task)
        fold_rows_list = []
        per_candidate: dict[str, list[float]] = {label: [] for label in labels}
        metric_spec = get_metric(metric_name)
        for fold in split_info["folds"]:
            valid_rows = list(fold["valid_rows"])
            y_valid = train.iloc[valid_rows][target]
            member_frames = [frame.loc[valid_rows] for frame in row_maps]
            vote_values = np.column_stack([frame["prediction"].to_numpy() for frame in member_frames])
            voted = _majority_vote(vote_values, classes)
            try:
                score = float(metric_spec(y_valid, voted))
                for label, frame in zip(labels, member_frames):
                    per_candidate[label].append(float(metric_spec(y_valid, _oof_metric_values(frame, task, metric_name))))
            except (ValueError, TypeError, TaskMetricError) as exc:
                raise EnsembleError(f"fold {fold['fold']}: saved OOF scoring failed: {exc}") from None
            if not math.isfinite(score):
                raise EnsembleError(f"fold {fold['fold']}: hard-voting score is non-finite")
            fold_rows_list.append(
                EnsembleFold(fold["fold"], len(fold["train_rows"]), len(valid_rows), score, 0.0, 0.0)
            )
        folds = tuple(fold_rows_list)
        scores = [fold.score for fold in folds]
        mean, std = float(np.mean(scores)), float(np.std(scores))
        candidate_scores = {key: tuple(value) for key, value in per_candidate.items()}
        validation_status = "validated"
        limitation = None
    feature_roles = {
        column: "numeric" if pd.api.types.is_numeric_dtype(train[column]) else "categorical"
        for column in feature_columns
    }
    result = EnsembleResult(
        task=task,
        metric=metric_name,
        direction=expected_metric.direction,
        method=cfg.method,
        candidates=labels,
        candidate_details={
            label: {
                "model_reference": str(loaded.reference),
                "model_id": item["model_id"],
                "estimator": (item.get("effective_config") or {}).get("estimator"),
                "validation_mean": None,
                "validation_std": None,
            }
            for label, loaded, item in zip(labels, loaded_models, metadata)
        },
        candidate_fold_scores=candidate_scores,
        folds=folds,
        mean=mean,
        std=std,
        n_rows=len(train),
        data_sha256=fingerprint,
        splits={
            "strategy": split_info["strategy"],
            "seed": split_info["seed"],
            "n_splits": len(split_info["folds"]),
            "n_rows": len(train),
            "stratified": split_info["stratified"],
            "sha256": expected_split_hash,
            "source": "saved_model_artifacts",
        },
        config=cfg,
        meta_estimator=type(fitted_meta).__name__ if fitted_meta is not None else None,
        feature_columns=feature_columns,
        feature_roles=feature_roles,
        classes=classes,
        model=final_model,
        validation_status=validation_status,
        validation_limitation=limitation,
    )
    if directory is not None:
        result = replace(result, directory=save_ensemble(result, directory))
    return result


def run_ensemble(
    train: pd.DataFrame,
    target: str,
    id_columns: tuple[str, ...],
    resolved_task: Any,
    splits: SplitResult,
    sources: Any,
    config: EnsembleConfig | None = None,
    *,
    labels: tuple[str, ...] | list[str] | None = None,
    meta_estimator: Any = None,
    directory: str | Path | None = None,
    progress: ProgressCallback | None = None,
    verbose: bool = False,
) -> EnsembleResult:
    """Evaluate a hard-voting or stacking ensemble from saved candidate directories.

    Candidate parameters and preprocessing are reused, but each fold's pipelines are fitted only
    on that fold's training rows. Hyperparameter searches are never rerun.
    """
    cfg = config or EnsembleConfig()
    task_value = getattr(resolved_task, "task", resolved_task)
    task = Task(getattr(task_value, "value", task_value))
    metric_name = getattr(getattr(resolved_task, "metric", None), "name", None)
    if not isinstance(metric_name, str):
        raise EnsembleError("resolved_task must include a metric name")
    ids = tuple(id_columns)
    missing_columns = [column for column in (target, *ids) if column not in train.columns]
    if missing_columns:
        raise EnsembleError(f"train data is missing target or ID columns {missing_columns}")
    if train[target].isna().any():
        raise EnsembleError(f"train target column {target!r} contains null values")
    if splits.task is not task or splits.n_rows != len(train):
        raise EnsembleError(
            f"validation splits cover task {splits.task.value!r} and {splits.n_rows} rows; "
            f"expected {task.value!r} and {len(train)} rows"
        )
    if task not in get_metric(metric_name).tasks:
        raise EnsembleError(f"metric {metric_name!r} does not support task {task.value!r}")
    if not splits.splits:
        raise EnsembleError("validation splits must not be empty")
    for split in splits.splits:
        if (
            len(split.train_idx) == 0
            or len(split.valid_idx) == 0
            or (split.train_idx < 0).any()
            or (split.valid_idx < 0).any()
            or (split.train_idx >= len(train)).any()
            or (split.valid_idx >= len(train)).any()
            or len(np.unique(split.train_idx)) != len(split.train_idx)
            or len(np.unique(split.valid_idx)) != len(split.valid_idx)
            or np.intersect1d(split.train_idx, split.valid_idx).size
        ):
            raise EnsembleError(f"fold {split.fold} has invalid, empty, or overlapping train/validation indices")
    if cfg.method == "stacking" and task is not Task.REGRESSION:
        min_class = int(train[target].value_counts().min())
        if cfg.stack_cv > min_class:
            raise EnsembleError(
                f"setting 'stack_cv' ({cfg.stack_cv}) exceeds the smallest class count ({min_class})"
            )
    normalized_labels = (labels,) if isinstance(labels, str) else tuple(labels) if labels is not None else None
    cset = _candidate_set(sources, normalized_labels)
    if (cset.task, cset.metric, cset.direction) != (task.value, metric_name, get_metric(metric_name).direction):
        raise EnsembleError(
            f"candidate results use task/metric/direction {(cset.task, cset.metric, cset.direction)}, "
            f"expected {(task.value, metric_name, get_metric(metric_name).direction)}"
        )
    candidates = cset.candidates
    feature_columns, feature_roles = _check_candidates(candidates, train, target, ids, task, metric_name, cfg)
    meta = _meta_estimator(task, cfg, meta_estimator)
    y = train[target].reset_index(drop=True)
    progress_started = time.perf_counter()
    if progress is not None:
        emit_progress(
            progress,
            "ensemble",
            "stage.start",
            progress_started,
            method=cfg.method,
            candidates=len(candidates),
            n_folds=len(splits.splits),
        )
    fold_scores: list[float] = []
    candidate_scores: dict[str, list[float]] = {candidate.label: [] for candidate in candidates}
    fold_rows: list[EnsembleFold] = []
    for split in splits.splits:
        if progress is not None:
            emit_progress(progress, "ensemble", "fold.start", progress_started, fold=split.fold)
        fold_train = train.iloc[split.train_idx]
        fold_valid = train.iloc[split.valid_idx]
        estimators = _build_candidates(candidates, fold_train, target, ids, task, cfg.seed, verbose)
        model = _make_model(estimators, task, cfg, meta)
        X_train = fold_train.loc[:, feature_columns]
        y_train = fold_train[target]
        X_valid = fold_valid.loc[:, feature_columns]
        y_valid = fold_valid[target]
        try:
            if progress is not None:
                for candidate in candidates:
                    emit_progress(
                        progress, "ensemble", "model.start", progress_started, model=candidate.label, fold=split.fold
                    )
            started = time.perf_counter()
            model.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - started
            started = time.perf_counter()
            score = _score(y_valid, model, X_valid, task, metric_name, cfg.method)
            predict_seconds = time.perf_counter() - started
            fitted_members = model.estimators_
            for candidate, member in zip(candidates, fitted_members):
                candidate_score = _score(y_valid, member, X_valid, task, metric_name, "stacking")
                if not np.isfinite(candidate_score):
                    raise EnsembleError(
                        f"fold {split.fold}: candidate {candidate.label!r} returned a non-finite score"
                    )
                candidate_scores[candidate.label].append(candidate_score)
                if progress is not None:
                    emit_progress(
                        progress,
                        "ensemble",
                        "model.complete",
                        progress_started,
                        model=candidate.label,
                        fold=split.fold,
                        score=candidate_score,
                    )
        except (ValueError, TypeError, TaskMetricError, EnsembleError) as exc:
            raise EnsembleError(f"fold {split.fold} failed: {exc}") from None
        if not np.isfinite(score):
            raise EnsembleError(f"fold {split.fold}: metric {metric_name!r} returned a non-finite score")
        fold_scores.append(score)
        fold_rows.append(
            EnsembleFold(split.fold, len(split.train_idx), len(split.valid_idx), score, fit_seconds, predict_seconds)
        )
        if progress is not None:
            emit_progress(
                progress,
                "ensemble",
                "fold.complete",
                progress_started,
                fold=split.fold,
                score=score,
                fit_seconds=fit_seconds,
                predict_seconds=predict_seconds,
            )
    full_estimators = _build_candidates(candidates, train, target, ids, task, cfg.seed, verbose)
    final_model = _make_model(full_estimators, task, cfg, meta)
    try:
        if progress is not None:
            emit_progress(progress, "ensemble", "fit.start", progress_started, scope="all_training_rows")
        final_model.fit(train.loc[:, feature_columns], y)
        if progress is not None:
            emit_progress(progress, "ensemble", "fit.complete", progress_started, scope="all_training_rows")
    except (ValueError, TypeError, TaskMetricError) as exc:
        raise EnsembleError(f"fitting the final ensemble on all training rows failed: {exc}") from None
    split_info = {
        "strategy": splits.config.strategy,
        "seed": splits.config.seed,
        "n_splits": len(splits.splits),
        "n_rows": splits.n_rows,
        "stratified": splits.stratified,
        "sha256": hashlib.sha256(
            json.dumps(
                [
                    [split.fold, split.train_idx.tolist(), split.valid_idx.tolist()]
                    for split in splits.splits
                ],
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
    }
    result = EnsembleResult(
        task=task,
        metric=metric_name,
        direction=get_metric(metric_name).direction,
        method=cfg.method,
        candidates=tuple(candidate.label for candidate in candidates),
        candidate_details={
            candidate.label: {
                "source_kind": candidate.source_kind,
                "source_path": candidate.source_path,
                "estimator": candidate.estimator_key,
                "params": candidate.params,
                "validation_mean": candidate.mean,
                "validation_std": candidate.std,
            }
            for candidate in candidates
        },
        candidate_fold_scores={label: tuple(scores) for label, scores in candidate_scores.items()},
        folds=tuple(fold_rows),
        mean=float(np.mean(fold_scores)),
        std=float(np.std(fold_scores)),
        n_rows=len(train),
        data_sha256=data_fingerprint(train),
        splits=split_info,
        config=cfg,
        meta_estimator=type(meta).__name__ if meta is not None else None,
        feature_columns=feature_columns,
        feature_roles=feature_roles,
        classes=tuple(sorted(y.unique())) if task is not Task.REGRESSION else (),
        model=final_model,
    )
    if directory is not None:
        result = replace(result, directory=save_ensemble(result, directory))
    if progress is not None:
        emit_progress(progress, "ensemble", "stage.complete", progress_started, score=result.mean)
    return result


def save_ensemble(result: EnsembleResult, directory: str | Path) -> Path:
    """Persist ensemble.json and the fitted ensemble in a new result directory."""
    out = Path(directory)
    if out.exists():
        raise EnsembleError(f"ensemble directory {out} already exists; choose a new path")
    if result.config.save_model and result.model is None:
        raise EnsembleError("save_model is true but the result has no fitted model")
    parent = out.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{out.name}.ensemble-", dir=parent))
    try:
        (tmp / RESULT_FILE).write_text(json.dumps(result.to_dict(), indent=2, allow_nan=False))
        if result.config.save_model:
            joblib.dump(result.model, tmp / MODEL_FILE)
        out.mkdir()
        for path in tmp.iterdir():
            os.replace(path, out / path.name)
    except FileExistsError:
        raise EnsembleError(f"ensemble directory {out} already exists; choose a new path") from None
    except Exception:
        if out.exists():
            for name in (RESULT_FILE, MODEL_FILE):
                artifact = out / name
                if artifact.exists():
                    artifact.unlink()
            out.rmdir()
        raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def load_ensemble(directory: str | Path, *, load_model: bool = False) -> LoadedEnsemble:
    """Load and validate ensemble metadata; joblib model loading is opt-in for trusted artifacts."""
    out = Path(directory)
    path = out / RESULT_FILE
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        raise EnsembleError(f"{path} not found; pass a directory written by save_ensemble") from None
    except json.JSONDecodeError as exc:
        raise EnsembleError(f"{path} is not valid JSON: {exc}") from None
    if not isinstance(raw, dict) or raw.get("schema_version") != ENSEMBLE_SCHEMA_VERSION:
        raise EnsembleError(f"{path} has unsupported schema_version; expected {ENSEMBLE_SCHEMA_VERSION}")
    required = (
        "task",
        "metric",
        "method",
        "candidates",
        "candidate_details",
        "folds",
        "summary",
        "n_rows",
        "config",
        "feature_columns",
        "feature_roles",
        "data_sha256",
        "files",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise EnsembleError(f"{path} is missing required fields {missing}")
    try:
        task = Task(raw["task"])
        metric = raw["metric"]
        if not isinstance(metric, dict) or not isinstance(metric.get("name"), str):
            raise ValueError("metric must contain a name")
        metric_spec = get_metric(metric["name"])
        if task not in metric_spec.tasks or metric.get("direction") != metric_spec.direction:
            raise ValueError("metric task or direction is inconsistent")
        if raw["method"] not in METHODS:
            raise ValueError(f"method must be one of {list(METHODS)}")
        config_raw = raw["config"]
        if not isinstance(config_raw, dict):
            raise ValueError("config must be an object")
        config = EnsembleConfig(**config_raw)
        if config.method != raw["method"]:
            raise ValueError("config method does not match the result method")
        candidates = raw["candidates"]
        if (
            not isinstance(candidates, list)
            or len(candidates) < 2
            or not all(isinstance(label, str) and label for label in candidates)
            or len(set(candidates)) != len(candidates)
        ):
            raise ValueError("candidates must contain at least two unique non-empty labels")
        if not isinstance(raw["candidate_details"], dict) or set(raw["candidate_details"]) != set(candidates):
            raise ValueError("candidate_details must describe every selected candidate")
        validation_status = raw.get("validation_status", "validated")
        if validation_status not in {"validated", "unavailable"}:
            raise ValueError("validation_status must be 'validated' or 'unavailable'")
        if not isinstance(raw["folds"], list) or (validation_status == "validated" and not raw["folds"]):
            raise ValueError("folds must be a list and non-empty when validation is available")
        summary = raw["summary"]
        if not isinstance(summary, dict):
            raise ValueError("summary must be an object")
        if validation_status == "unavailable":
            if summary.get("mean") is not None or summary.get("std") is not None:
                raise ValueError("summary scores must be null when validation_status is unavailable")
        elif any(
            isinstance(summary.get(key), bool)
            or not isinstance(summary.get(key), (int, float))
            or not math.isfinite(summary[key])
            for key in ("mean", "std")
        ):
            raise ValueError("summary must contain finite numeric mean and std")
        if isinstance(raw["n_rows"], bool) or not isinstance(raw["n_rows"], int) or raw["n_rows"] < 1:
            raise ValueError("n_rows must be a positive integer")
        if not isinstance(raw["feature_columns"], list) or not all(
            isinstance(column, str) and column for column in raw["feature_columns"]
        ):
            raise ValueError("feature_columns must be a list of non-empty column names")
        if (
            not isinstance(raw["feature_roles"], dict)
            or set(raw["feature_roles"]) != set(raw["feature_columns"])
            or not all(role in {"numeric", "categorical"} for role in raw["feature_roles"].values())
        ):
            raise ValueError("feature_roles must classify every feature as numeric or categorical")
        fingerprint = raw["data_sha256"]
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(char not in "0123456789abcdef" for char in fingerprint)
        ):
            raise ValueError("data_sha256 must be a SHA-256 hex digest")
        if not isinstance(raw["files"], dict):
            raise ValueError("files must be an object")
    except (KeyError, TypeError, ValueError, DataContractError) as exc:
        raise EnsembleError(f"{path} has invalid metadata: {exc}") from None
    model = None
    if load_model:
        model_name = raw["files"].get("model")
        if model_name != MODEL_FILE or not (out / MODEL_FILE).is_file():
            raise EnsembleError(f"{path}: fitted ensemble model was requested but is missing")
        try:
            model = joblib.load(out / MODEL_FILE)
        except Exception as exc:
            raise EnsembleError(
                f"could not load trusted fitted ensemble {out / MODEL_FILE}: {type(exc).__name__}: {exc}"
            ) from None
    return LoadedEnsemble(raw, model)
