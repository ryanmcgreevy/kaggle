"""Estimator registry with optional boosting adapters (LightGBM, XGBoost, CatBoost)."""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

from automl.errors import BaselineError, EstimatorUnavailableError
from automl.metrics import Task

DEVICES = ("cpu", "auto", "gpu")
_ALL_TASKS = frozenset(Task)

# Keys the run controls itself; value is the setting to use instead (None: managed internally).
CONTROLLED_PARAMS: dict[str, str | None] = {
    "random_state": "seed",
    "random_seed": "seed",
    "seed": "seed",
    "n_jobs": "n_jobs",
    "nthread": "n_jobs",
    "num_threads": "n_jobs",
    "n_threads": "n_jobs",
    "thread_count": "n_jobs",
    "device": "device",
    "device_type": "device",
    "task_type": "device",
    "devices": "device",
    "gpu_id": "device",
    "tree_method": None,
    "verbose": None,
    "verbosity": None,
    "silent": None,
    "logging_level": None,
    "early_stopping": None,
    "early_stopping_rounds": None,
    "early_stopping_round": None,
    "od_type": None,
    "od_wait": None,
    "allow_writing_files": None,
    "train_dir": None,
    "save_snapshot": None,
    "categorical_feature": "native_categoricals",
    "categorical_features": "native_categoricals",
    "cat_features": "native_categoricals",
    "enable_categorical": "native_categoricals",
    "feature_types": "native_categoricals",
}


def controlled_param_problems(params: dict[str, Any]) -> list[str]:
    problems = []
    for key in params:
        if key in CONTROLLED_PARAMS:
            setting = CONTROLLED_PARAMS[key]
            hint = f"use the '{setting}' setting" if setting else "it is managed by automl"
            problems.append(f"setting 'estimator_params' cannot set '{key}'; {hint}")
    return problems


@dataclass(frozen=True)
class FeatureLayout:
    """Transformed column layout: the numeric block followed by the categorical block."""

    n_numeric: int
    n_categorical: int
    native: bool = False

    @property
    def categorical_idx(self) -> tuple[int, ...]:
        return tuple(range(self.n_numeric, self.n_numeric + self.n_categorical)) if self.native else ()


@dataclass(frozen=True)
class DeviceChoice:
    requested: str
    effective: str
    reason: str


@dataclass(frozen=True)
class EstimatorSpec:
    name: str
    package: str | None
    extra: str | None
    tasks: frozenset[Task]
    supports_native_categoricals: bool
    native_default: bool
    supports_gpu: bool
    defaults: dict[str, Any]
    tiny_params: dict[str, Any]
    factory: Callable[..., Any] = field(repr=False)
    fit_params_fn: Callable[[FeatureLayout], dict[str, Any]] | None = field(default=None, repr=False)

    def available(self) -> bool:
        if self.package is None:
            return True
        try:
            importlib.import_module(self.package)
        except ImportError:
            return False
        return True

    def require(self) -> None:
        if not self.available():
            raise EstimatorUnavailableError(
                f"estimator '{self.name}' requires the '{self.package}' package, which is not installed; "
                f"install it with: pip install 'automl[{self.extra}]'"
            )

    def make(
        self, task: Task, params: dict[str, Any], seed: int, n_jobs: int, device: str, layout: FeatureLayout
    ) -> Any:
        self.require()
        if task not in self.tasks:
            raise BaselineError(f"estimator '{self.name}' does not support task '{task}'")
        problems = controlled_param_problems(params)
        if problems:
            raise BaselineError(problems)
        return self.factory(task, params, seed, n_jobs, device, layout)

    def fit_params(self, layout: FeatureLayout) -> dict[str, Any]:
        return self.fit_params_fn(layout) if self.fit_params_fn else {}

    def version(self) -> str | None:
        return importlib.metadata.version(self.package) if self.package else None


def _check_names(accepted: set[str], params: dict[str, Any], label: str) -> None:
    unknown = sorted(set(params) - accepted)
    if unknown:
        raise BaselineError(f"setting 'estimator_params' has parameters {unknown} that {label} does not accept")


class _LabelEncodedClassifier(BaseEstimator):
    """Gives libraries that need integer labels string-label support with a consistent `classes_`."""

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator

    def fit(self, X: Any, y: Any, **fit_params: Any) -> "_LabelEncodedClassifier":
        encoder = LabelEncoder().fit(y)
        self.classes_ = encoder.classes_
        self.estimator_ = clone(self.estimator).fit(X, encoder.transform(y), **fit_params)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.asarray(self.estimator_.predict_proba(X))

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class _CatBoostModel(BaseEstimator):
    """CatBoost with label encoding and native categorical columns built from ordinal codes."""

    def __init__(
        self, classifier: bool, params: dict[str, Any], cat_idx: tuple[int, ...] = (), native_output: bool = False
    ) -> None:
        self.classifier = classifier
        self.params = params
        self.cat_idx = cat_idx
        self.native_output = native_output

    def _frame(self, X: Any) -> pd.DataFrame:
        X = np.asarray(X, dtype=float)
        cols: dict[str, Any] = {}
        for j in range(X.shape[1]):
            col = X[:, j]
            if j in self.cat_idx:
                nan = np.isnan(col)
                text = np.where(nan, 0, col).astype(np.int64).astype(str)
                cols[f"f{j}"] = np.where(nan, "nan", text).astype(object)
            else:
                cols[f"f{j}"] = col
        return pd.DataFrame(cols)

    def fit(self, X: Any, y: Any) -> "_CatBoostModel":
        from catboost import CatBoostClassifier, CatBoostRegressor

        kwargs = dict(self.params)
        fit_kwargs = {"log_cout": sys.stderr, "log_cerr": sys.stderr} if self.native_output else {}
        if self.cat_idx:
            kwargs["cat_features"] = list(self.cat_idx)
        if self.classifier:
            encoder = LabelEncoder().fit(y)
            self.classes_ = encoder.classes_
            self.model_ = CatBoostClassifier(**kwargs).fit(self._frame(X), encoder.transform(y), **fit_kwargs)
        else:
            self.model_ = CatBoostRegressor(**kwargs).fit(
                self._frame(X), np.asarray(y, dtype=float), **fit_kwargs
            )
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.asarray(self.model_.predict_proba(self._frame(X)))

    def predict(self, X: Any) -> np.ndarray:
        if self.classifier:
            return self.classes_[self.predict_proba(X).argmax(axis=1)]
        return np.asarray(self.model_.predict(self._frame(X))).ravel()


_DEFAULTS: dict[str, dict[str, Any]] = {
    "hist_gradient_boosting": {"early_stopping": False},
    "lightgbm": {"n_estimators": 100, "verbose": -1},
    "xgboost": {"n_estimators": 100, "tree_method": "hist", "verbosity": 0},
    "catboost": {"iterations": 100, "verbose": 0, "allow_writing_files": False},
}


def _make_hgb(task, params, seed, n_jobs, device, layout):
    cls = HistGradientBoostingRegressor if task is Task.REGRESSION else HistGradientBoostingClassifier
    _check_names(set(cls().get_params()), params, cls.__name__)
    kwargs = {**_DEFAULTS["hist_gradient_boosting"], **params, "random_state": seed}
    if layout.categorical_idx:
        kwargs["categorical_features"] = list(layout.categorical_idx)
    return cls(**kwargs)


def _make_lightgbm(task, params, seed, n_jobs, device, layout):
    from lightgbm import LGBMClassifier, LGBMRegressor

    cls = LGBMRegressor if task is Task.REGRESSION else LGBMClassifier
    _check_names(set(cls().get_params()), params, cls.__name__)
    kwargs = {**_DEFAULTS["lightgbm"], **params, "random_state": seed, "n_jobs": n_jobs}
    if device == "gpu":
        kwargs["device_type"] = "gpu"
    return cls(**kwargs)


def _lightgbm_fit_params(layout: FeatureLayout) -> dict[str, Any]:
    return {"categorical_feature": list(layout.categorical_idx)} if layout.categorical_idx else {}


def _make_xgboost(task, params, seed, n_jobs, device, layout):
    from xgboost import XGBClassifier, XGBRegressor

    cls = XGBRegressor if task is Task.REGRESSION else XGBClassifier
    _check_names(set(cls().get_params()), params, cls.__name__)
    kwargs = {
        **_DEFAULTS["xgboost"],
        **params,
        "random_state": seed,
        "n_jobs": n_jobs,
        "device": "cuda" if device == "gpu" else "cpu",
    }
    if layout.categorical_idx:
        kwargs["enable_categorical"] = True
        kwargs["feature_types"] = ["q"] * layout.n_numeric + ["c"] * layout.n_categorical
    est = cls(**kwargs)
    return est if task is Task.REGRESSION else _LabelEncodedClassifier(est)


def _make_catboost(task, params, seed, n_jobs, device, layout):
    from catboost import CatBoostClassifier

    accepted = set(inspect.signature(CatBoostClassifier.__init__).parameters) - {"self"}
    _check_names(accepted, params, "CatBoost")
    kwargs = {**_DEFAULTS["catboost"], **params, "random_seed": seed, "thread_count": n_jobs}
    if device == "gpu":
        kwargs["task_type"] = "GPU"
        kwargs["devices"] = "0"
    return _CatBoostModel(task is not Task.REGRESSION, kwargs, layout.categorical_idx)


_SPECS: dict[str, EstimatorSpec] = {
    spec.name: spec
    for spec in (
        EstimatorSpec(
            "hist_gradient_boosting", None, None, _ALL_TASKS, True, False, False,
            _DEFAULTS["hist_gradient_boosting"], {}, _make_hgb,
        ),
        EstimatorSpec(
            "lightgbm", "lightgbm", "lightgbm", _ALL_TASKS, True, True, True,
            _DEFAULTS["lightgbm"], {"n_estimators": 2}, _make_lightgbm, _lightgbm_fit_params,
        ),
        EstimatorSpec(
            "xgboost", "xgboost", "xgboost", _ALL_TASKS, True, True, True,
            _DEFAULTS["xgboost"], {"n_estimators": 2}, _make_xgboost,
        ),
        EstimatorSpec(
            "catboost", "catboost", "catboost", _ALL_TASKS, True, True, True,
            _DEFAULTS["catboost"], {"iterations": 2}, _make_catboost,
        ),
    )
}
ESTIMATOR_NAMES = tuple(_SPECS)


def describe_estimator(est: Any) -> tuple[str, dict[str, Any]]:
    """Return the underlying library class name and its effective parameters."""
    if isinstance(est, _LabelEncodedClassifier):
        est = est.estimator
    if isinstance(est, _CatBoostModel):
        name = "CatBoostClassifier" if est.classifier else "CatBoostRegressor"
        params = dict(est.params)
        if est.cat_idx:
            params["cat_features"] = list(est.cat_idx)
        return name, params
    return type(est).__name__, est.get_params()


def enable_native_verbosity(estimator: Any) -> bool:
    """Enable an adapter's native training output, returning whether it is supported."""
    if isinstance(estimator, _LabelEncodedClassifier):
        estimator = estimator.estimator
    if isinstance(estimator, _CatBoostModel):
        estimator.params = {**estimator.params, "verbose": True}
        estimator.native_output = True
        return True
    module = type(estimator).__module__.split(".")[0]
    if module == "xgboost":
        estimator.set_params(verbosity=2)
        return True
    if module == "lightgbm":
        estimator.set_params(verbose=1)
        return True
    return False


def list_estimators() -> list[str]:
    return list(ESTIMATOR_NAMES)


def get_estimator_spec(name: str) -> EstimatorSpec:
    try:
        return _SPECS[name]
    except KeyError:
        raise BaselineError(f"unknown estimator {name!r}; valid estimators: {list(ESTIMATOR_NAMES)}") from None


def estimator_available(name: str) -> bool:
    return get_estimator_spec(name).available()


_PROBES: dict[str, tuple[bool, str]] = {}


def _probe_gpu(spec: EstimatorSpec) -> tuple[bool, str]:
    if spec.name not in _PROBES:
        rng = np.random.default_rng(0)
        X, y = rng.normal(size=(64, 3)), rng.normal(size=64)
        try:
            est = spec.make(Task.REGRESSION, dict(spec.tiny_params), 0, 1, "gpu", FeatureLayout(3, 0))
            est.fit(X, y)
            est.predict(X[:4])
            _PROBES[spec.name] = (True, "GPU probe fit succeeded")
        except Exception as exc:  # libraries raise unrelated error types when no usable GPU build exists
            first = (str(exc).strip().splitlines() or [type(exc).__name__])[0][:200]
            _PROBES[spec.name] = (False, f"GPU probe fit failed for '{spec.name}': {first}")
    return _PROBES[spec.name]


def resolve_device(spec: EstimatorSpec, requested: str) -> DeviceChoice:
    """Pick the effective device; 'auto' falls back to CPU with a reason, 'gpu' raises if unavailable."""
    if requested not in DEVICES:
        raise BaselineError(f"setting 'device' has unsupported value {requested!r}; valid: {list(DEVICES)}")
    if requested == "cpu":
        return DeviceChoice(requested, "cpu", "cpu requested")
    if not spec.supports_gpu:
        reason = f"estimator '{spec.name}' does not support GPU training"
    else:
        spec.require()
        ok, reason = _probe_gpu(spec)
        if ok:
            return DeviceChoice(requested, "gpu", reason)
    if requested == "gpu":
        raise EstimatorUnavailableError(f"setting 'device' is 'gpu' but {reason}; use 'auto' or 'cpu'")
    return DeviceChoice(requested, "cpu", f"fell back to cpu: {reason}")
