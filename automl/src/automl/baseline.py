"""Quick baseline with a selectable boosting estimator, evaluated with the configured validation splits."""

from __future__ import annotations

import json
import math
import platform
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from automl.errors import BaselineError, TaskMetricError
from automl.estimators import (
    DEVICES,
    ESTIMATOR_NAMES,
    FeatureLayout,
    controlled_param_problems,
    describe_estimator,
    get_estimator_spec,
    resolve_device,
)
from automl.metrics import Task
from automl.preprocessing import (
    ColumnPlan,
    PreprocessingConfig,
    build_preprocessor,
    infer_columns,
    plan_from_dict,
    plan_to_dict,
)
from automl.validation import SplitResult

ARTIFACT_VERSION = 2
READABLE_VERSIONS = (1, 2)
DEFAULT_ESTIMATOR = "hist_gradient_boosting"
RESULT_FILE = "baseline.json"
OOF_FILE = "oof_predictions.csv"
MODELS_FILE = "models.joblib"


@dataclass(frozen=True)
class BaselineConfig:
    seed: int = 42
    n_jobs: int = 1
    save_models: bool = True
    estimator_params: dict[str, Any] = field(default_factory=dict)
    estimator: str = DEFAULT_ESTIMATOR
    native_categoricals: bool | None = None
    device: str = "cpu"

    def __post_init__(self) -> None:
        problems: list[str] = []
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            problems.append("setting 'seed' must be an integer")
        nj = self.n_jobs
        if isinstance(nj, bool) or not isinstance(nj, int) or (nj != -1 and nj < 1):
            problems.append(f"setting 'n_jobs' must be -1 or an integer of at least 1, got {nj!r}")
        if not isinstance(self.save_models, bool):
            problems.append("setting 'save_models' must be true or false")
        params = self.estimator_params
        if not isinstance(params, dict) or not all(isinstance(k, str) for k in params):
            problems.append("setting 'estimator_params' must be a table of parameter names to values")
        else:
            problems.extend(controlled_param_problems(params))
        if self.estimator not in ESTIMATOR_NAMES:
            problems.append(
                f"setting 'estimator' has unsupported value {self.estimator!r}; valid: {list(ESTIMATOR_NAMES)}"
            )
        if self.native_categoricals is not None and not isinstance(self.native_categoricals, bool):
            problems.append("setting 'native_categoricals' must be true or false")
        if self.device not in DEVICES:
            problems.append(f"setting 'device' has unsupported value {self.device!r}; valid: {list(DEVICES)}")
        if problems:
            raise BaselineError(problems)
        object.__setattr__(self, "estimator_params", dict(params))


BASELINE_KEYS = {f.name for f in fields(BaselineConfig)}


@dataclass(frozen=True)
class FoldRecord:
    fold: int
    n_train: int
    n_valid: int
    score: float
    fit_seconds: float
    predict_seconds: float


@dataclass(frozen=True, eq=False)
class BaselineResult:
    task: Task
    metric: str
    direction: str
    needs: str
    estimator: str
    estimator_params: dict[str, Any]
    config: BaselineConfig
    preprocessing_config: PreprocessingConfig
    plan: ColumnPlan
    split: dict[str, Any]
    prediction_kind: str
    folds: tuple[FoldRecord, ...]
    mean: float
    std: float
    n_rows: int
    oof: pd.DataFrame
    models: tuple[Any, ...] = ()
    versions: dict[str, str] = field(default_factory=dict)
    estimator_key: str = DEFAULT_ESTIMATOR
    categorical_handling: str = "encoded"
    device: dict[str, Any] = field(default_factory=dict)


def make_estimator(
    task: Any, config: BaselineConfig | None = None, layout: FeatureLayout | None = None, device: str = "cpu"
) -> Any:
    """Return the unfitted estimator selected by `config.estimator`, seeded, with early stopping off."""
    cfg = config or BaselineConfig()
    task = Task(getattr(task, "task", task))
    spec = get_estimator_spec(cfg.estimator)
    return spec.make(task, cfg.estimator_params, cfg.seed, cfg.n_jobs, device, layout or FeatureLayout(0, 0))


def _versions(package_version: tuple[str, str | None] | None = None) -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
    if package_version and package_version[1]:
        versions[package_version[0]] = package_version[1]
    return versions


def _prediction_kind(task: Task, needs: str) -> str:
    if needs == "values":
        return "values"
    if needs == "labels":
        return "labels"
    return "proba" if task is Task.MULTICLASS else "proba_positive"


def _run_fold(pipeline, X, y, split, task, metric, needs, fit_params):
    name = f"fold {split.fold}"
    X_tr, y_tr = X.iloc[split.train_idx], y.iloc[split.train_idx]
    X_va, y_va = X.iloc[split.valid_idx], y.iloc[split.valid_idx]
    if task is not Task.REGRESSION:
        train_classes, valid_classes = set(y_tr.unique()), set(y_va.unique())
        if train_classes != valid_classes:
            raise BaselineError(
                f"{name}: class sets differ between train {sorted(map(str, train_classes))} and validation "
                f"{sorted(map(str, valid_classes))}; use fewer folds or merge/remove rare classes"
            )
    model = clone(pipeline)
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr, **fit_params)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    if needs == "proba":
        proba = np.asarray(model.predict_proba(X_va))
        pred = proba[:, 1] if task is Task.BINARY else proba
    else:
        pred = np.asarray(model.predict(X_va))
    predict_s = time.perf_counter() - t0
    if pred.dtype.kind in "fiu" and not np.isfinite(pred).all():
        raise BaselineError(f"{name}: predictions contain non-finite values")
    if pred.shape[0] != len(y_va):
        raise BaselineError(f"{name}: expected {len(y_va)} predictions, got {pred.shape[0]}")
    try:
        score = float(metric(y_va, pred))
    except TaskMetricError as exc:
        raise BaselineError(f"{name}: metric '{metric.name}' failed: {exc}") from None
    if not math.isfinite(score):
        raise BaselineError(f"{name}: metric '{metric.name}' returned a non-finite score")
    classes = [str(c) for c in model.named_steps["model"].classes_] if task is not Task.REGRESSION else []
    return model, pred, classes, FoldRecord(split.fold, len(y_tr), len(y_va), score, fit_s, predict_s)


def run_baseline(
    train: pd.DataFrame,
    target: str,
    id_columns: tuple[str, ...],
    resolved_task: Any,
    splits: SplitResult,
    config: BaselineConfig | None = None,
    preprocessing_config: PreprocessingConfig | None = None,
) -> BaselineResult:
    """Fit one pipeline per split with the selected estimator and score validation rows with the resolved metric.

    Preprocessing is fit inside each fold. `preprocessing_config=None` uses ordinal encoding, which
    native categorical handling (the default for LightGBM, XGBoost, and CatBoost) replaces.
    """
    cfg = config or BaselineConfig()
    prep_cfg = preprocessing_config or PreprocessingConfig(encoder="ordinal", seed=cfg.seed)
    task, metric = resolved_task.task, resolved_task.metric
    problems: list[str] = []
    if target not in train.columns:
        problems.append(f"target column '{target}' is not in the train data")
    if splits.n_rows != len(train):
        problems.append(f"splits were built for {splits.n_rows} rows but train has {len(train)}")
    if splits.task != task:
        problems.append(f"splits were built for task '{splits.task}' but the resolved task is '{task}'")
    if problems:
        raise BaselineError(problems)

    spec = get_estimator_spec(cfg.estimator)
    spec.require()
    native = spec.native_default if cfg.native_categoricals is None else cfg.native_categoricals
    if native and not spec.supports_native_categoricals:
        raise BaselineError(f"setting 'native_categoricals' is true but '{spec.name}' has no native categorical support")
    choice = resolve_device(spec, cfg.device)

    plan = infer_columns(train, target, id_columns, prep_cfg)
    layout = FeatureLayout(len(plan.numeric), len(plan.categorical), native)
    estimator = make_estimator(task, cfg, layout, choice.effective)
    pre = build_preprocessor(plan, prep_cfg, task, native_categoricals=native)
    pipeline = Pipeline([("pre", pre), ("model", estimator)])
    fit_params = {f"model__{k}": v for k, v in spec.fit_params(layout).items()}
    y = train[target]
    X = train[[*plan.numeric, *plan.categorical]]  # model inputs exclude target, ids, and dropped columns
    needs = metric.needs
    # Boosting libraries use n_jobs as their own thread count, so only the scikit-learn baseline parallelizes folds.
    fold_jobs = cfg.n_jobs if spec.package is None else 1
    results = Parallel(n_jobs=fold_jobs)(
        delayed(_run_fold)(pipeline, X, y, s, task, metric, needs, fit_params) for s in splits.splits
    )

    kind = _prediction_kind(task, needs)
    frames = []
    for (_, pred, classes, _), s in zip(results, splits.splits):
        frame = pd.DataFrame({"row": s.valid_idx, "fold": s.fold})
        if kind == "proba":
            for i, c in enumerate(classes):
                frame[f"proba_{c}"] = pred[:, i]
        else:
            frame["prediction"] = pred
        frames.append(frame)
    oof = pd.concat(frames, ignore_index=True).sort_values("row", kind="stable").reset_index(drop=True)

    folds = tuple(r[3] for r in results)
    scores = np.array([f.score for f in folds])
    sc = splits.config
    est_name, est_params = describe_estimator(estimator)
    return BaselineResult(
        task=task,
        metric=metric.name,
        direction=metric.direction,
        needs=needs,
        estimator=est_name,
        estimator_params=_plain(est_params),
        config=cfg,
        preprocessing_config=prep_cfg,
        plan=plan,
        split={
            "strategy": sc.strategy,
            "seed": sc.seed,
            "n_folds": len(splits.splits),
            "holdout_fraction": sc.holdout_fraction,
            "stratified": splits.stratified,
        },
        prediction_kind=kind,
        folds=folds,
        mean=float(scores.mean()),
        std=float(scores.std()),
        n_rows=len(train),
        oof=oof,
        models=tuple(r[0] for r in results),
        versions=_versions((spec.package, spec.version()) if spec.package else None),
        estimator_key=spec.name,
        categorical_handling="native" if native else "encoded",
        device={
            "requested": choice.requested,
            "effective": choice.effective,
            "reason": choice.reason,
            "note": "GPU training is not guaranteed to be bit-reproducible" if choice.effective == "gpu" else None,
        },
    )


def _plain(params: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(params, default=str))


def save_baseline(result: BaselineResult, directory: str | Path) -> None:
    """Write the result directory; refuses to overwrite an existing directory."""
    directory = Path(directory)
    try:
        directory.mkdir(parents=True)
    except FileExistsError:
        raise BaselineError(f"baseline directory {directory} already exists; choose a new path") from None
    with_models = result.config.save_models
    if with_models and not result.models:
        raise BaselineError("save_models is true but the result holds no fitted models")
    payload = {
        "schema_version": ARTIFACT_VERSION,
        "task": result.task.value,
        "metric": {"name": result.metric, "direction": result.direction, "needs": result.needs},
        "estimator": {"name": result.estimator, "key": result.estimator_key, "params": result.estimator_params},
        "categorical_handling": result.categorical_handling,
        "device": result.device,
        "config": {
            "seed": result.config.seed,
            "n_jobs": result.config.n_jobs,
            "save_models": result.config.save_models,
            "estimator_params": result.config.estimator_params,
            "estimator": result.config.estimator,
            "native_categoricals": result.config.native_categoricals,
            "device": result.config.device,
        },
        "preprocessing": plan_to_dict(result.plan, result.preprocessing_config),
        "split": result.split,
        "prediction_kind": result.prediction_kind,
        "n_rows": result.n_rows,
        "folds": [
            {
                "fold": f.fold,
                "n_train": f.n_train,
                "n_valid": f.n_valid,
                "score": f.score,
                "fit_seconds": f.fit_seconds,
                "predict_seconds": f.predict_seconds,
            }
            for f in result.folds
        ],
        "summary": {"mean": result.mean, "std": result.std},
        "versions": result.versions,
        "files": {"oof_predictions": OOF_FILE, "models": MODELS_FILE if with_models else None},
    }
    (directory / RESULT_FILE).write_text(json.dumps(payload, indent=2))
    result.oof.to_csv(directory / OOF_FILE, index=False)
    if with_models:
        joblib.dump(list(result.models), directory / MODELS_FILE)


def load_baseline(directory: str | Path, load_models: bool = False) -> BaselineResult:
    """Read and validate a result directory. Only set `load_models` for trusted directories."""
    directory = Path(directory)
    path = directory / RESULT_FILE
    label = f"baseline result {directory}"

    def bad(msg: str) -> BaselineError:
        return BaselineError(f"{label}: {msg}")

    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        raise bad(f"{RESULT_FILE} not found") from None
    except json.JSONDecodeError as exc:
        raise bad(f"{RESULT_FILE} is not valid JSON: {exc}") from None
    if not isinstance(raw, dict) or raw.get("schema_version") not in READABLE_VERSIONS:
        raise bad(f"unsupported or missing 'schema_version'; expected one of {list(READABLE_VERSIONS)}")
    try:
        task = Task(raw["task"])
        metric, estimator, cfg_raw = raw["metric"], raw["estimator"], raw["config"]
        folds_raw, summary, files = raw["folds"], raw["summary"], raw["files"]
        split, kind, n_rows = raw["split"], raw["prediction_kind"], raw["n_rows"]
        versions = raw["versions"]
        config = BaselineConfig(**cfg_raw)
        folds = tuple(FoldRecord(**f) for f in folds_raw)
        plan, prep_cfg = plan_from_dict(raw["preprocessing"], f"{label} preprocessing")
        result_fields = (metric["name"], metric["direction"], metric["needs"], estimator["name"], estimator["params"])
        estimator_key = estimator.get("key", DEFAULT_ESTIMATOR)
        handling, device = raw.get("categorical_handling", "encoded"), raw.get("device", {})
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise bad(f"missing or invalid field ({exc})") from None
    if isinstance(n_rows, bool) or not isinstance(n_rows, int) or not folds:
        raise bad("'n_rows' must be an integer and 'folds' must not be empty")
    if not np.isclose(np.mean([f.score for f in folds]), summary.get("mean", np.nan)):
        raise bad("summary mean does not match the per-fold scores")

    oof_name = files.get("oof_predictions") if isinstance(files, dict) else None
    if not oof_name or not (directory / oof_name).is_file():
        raise bad("out-of-fold predictions file is missing")
    oof = pd.read_csv(directory / oof_name)
    if not {"row", "fold"} <= set(oof.columns):
        raise bad("out-of-fold predictions must have 'row' and 'fold' columns")
    if len(oof) != sum(f.n_valid for f in folds):
        raise bad("out-of-fold prediction count does not match the folds' validation sizes")
    if oof["row"].min() < 0 or oof["row"].max() >= n_rows or oof["row"].duplicated().any():
        raise bad(f"out-of-fold 'row' values must be unique integers in [0, {n_rows})")

    models: tuple[Any, ...] = ()
    if load_models:
        models_name = files.get("models")
        if not models_name or not (directory / models_name).is_file():
            raise bad("models were requested but no models file was saved")
        models = tuple(joblib.load(directory / models_name))
        if len(models) != len(folds):
            raise bad(f"models file holds {len(models)} model(s) for {len(folds)} fold(s)")

    return BaselineResult(
        task=task,
        metric=result_fields[0],
        direction=result_fields[1],
        needs=result_fields[2],
        estimator=result_fields[3],
        estimator_params=result_fields[4],
        config=config,
        preprocessing_config=prep_cfg,
        plan=plan,
        split=split,
        prediction_kind=kind,
        folds=folds,
        mean=float(summary["mean"]),
        std=float(summary["std"]),
        n_rows=n_rows,
        oof=oof,
        models=models,
        versions=versions,
        estimator_key=estimator_key,
        categorical_handling=handling,
        device=device,
    )
