"""Seeded, budgeted Optuna tuning of one estimator with the configured validation splits."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from automl.baseline import BaselineConfig, _plain, _run_fold, _versions
from automl.errors import BaselineError, TuningError, TuningUnavailableError
from automl.estimators import (
    ESTIMATOR_NAMES,
    FeatureLayout,
    controlled_param_problems,
    enable_native_verbosity,
    get_estimator_spec,
    resolve_device,
)
from automl.metrics import Task
from automl.preprocessing import PreprocessingConfig, build_preprocessor, infer_columns
from automl.progress import ProgressCallback, emit_progress
from automl.search_spaces import SPACE_FAMILIES, apply_overrides, get_search_space
from automl.validation import SplitResult

SCHEMA_VERSION = 2
READABLE_VERSIONS = (1, 2)
FAILURE_CATEGORIES = ("invalid_parameter", "library_error", "timeout", "non_finite_score", "other")
STUDY_NAME = "automl-tuning"
STUDY_FILE = "study.db"
RESULT_FILE = "tuning.json"
PRUNER_STARTUP_TRIALS = 3

SearchSpace = dict[str, dict[str, Any]] | Callable[[Any], dict[str, Any]]

def _budget_problems(n: Any, t: Any, where: str = "") -> list[str]:
    problems: list[str] = []
    if n is not None and (isinstance(n, bool) or not isinstance(n, int) or n < 1):
        problems.append(f"setting '{where}n_trials' must be an integer of at least 1, got {n!r}")
    if t is not None and (isinstance(t, bool) or not isinstance(t, (int, float)) or not t > 0):
        problems.append(f"setting '{where}timeout_seconds' must be a positive number, got {t!r}")
    return problems


@dataclass(frozen=True)
class TuningConfig:
    n_trials: int | None = None
    timeout_seconds: float | None = None
    pruning: bool = False
    seed: int = 42
    n_jobs: int = 1
    estimator: str = "hist_gradient_boosting"
    estimator_params: dict[str, Any] = field(default_factory=dict)
    native_categoricals: bool | None = None
    device: str = "cpu"
    # Multi-family settings (used by tune_families): families to tune, per-family budgets, range overrides.
    families: tuple[str, ...] | None = None
    budgets: dict[str, dict[str, Any]] = field(default_factory=dict)
    spaces: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        problems: list[str] = []
        n, t = self.n_trials, self.timeout_seconds
        problems += _budget_problems(n, t)
        families = self.families
        if families is not None:
            if isinstance(families, str) or not isinstance(families, (list, tuple)) or not families:
                problems.append("setting 'families' must be a non-empty list of estimator names")
                families = None
            elif len(set(families)) != len(families):
                problems.append("setting 'families' has duplicate entries")
            else:
                bad = [f for f in families if f not in ESTIMATOR_NAMES]
                if bad:
                    problems.append(f"setting 'families' has unknown estimators {bad}; valid: {list(ESTIMATOR_NAMES)}")
        if not isinstance(self.budgets, dict):
            problems.append("setting 'budgets' must be a table of per-family budgets")
        else:
            for fam, b in self.budgets.items():
                if fam not in ESTIMATOR_NAMES:
                    problems.append(f"setting 'budgets' has unknown estimator {fam!r}; valid: {list(ESTIMATOR_NAMES)}")
                elif not isinstance(b, dict) or set(b) - {"n_trials", "timeout_seconds"}:
                    problems.append(f"setting 'budgets.{fam}' must be a table with 'n_trials' and/or 'timeout_seconds'")
                else:
                    problems += _budget_problems(b.get("n_trials"), b.get("timeout_seconds"), f"budgets.{fam}.")
                    if b.get("n_trials") is None and b.get("timeout_seconds") is None:
                        problems.append(f"setting 'budgets.{fam}' needs 'n_trials' and/or 'timeout_seconds'")
        if n is None and t is None:
            covered = families is not None and isinstance(self.budgets, dict) and all(f in self.budgets for f in families)
            if not covered:
                problems.append("set 'n_trials' and/or 'timeout_seconds'; a tuning run needs a budget")
        if not isinstance(self.spaces, dict):
            problems.append("setting 'spaces' must be a table of per-family range overrides")
        else:
            for fam, ov in self.spaces.items():
                if fam not in SPACE_FAMILIES:
                    problems.append(f"setting 'spaces' has unknown estimator {fam!r}; valid: {list(SPACE_FAMILIES)}")
                    continue
                if not isinstance(ov, dict):
                    problems.append(f"setting 'spaces.{fam}' must be a table of per-parameter overrides")
                    continue
                for task in Task:
                    try:
                        _validate_mapping(apply_overrides(fam, get_search_space(fam, task), ov))
                    except TuningError as exc:
                        problems += [p for p in exc.problems if p not in problems]
        if not isinstance(self.pruning, bool):
            problems.append("setting 'pruning' must be true or false")
        try:
            BaselineConfig(
                seed=self.seed,
                n_jobs=self.n_jobs,
                save_models=False,
                estimator_params=self.estimator_params,
                estimator=self.estimator,
                native_categoricals=self.native_categoricals,
                device=self.device,
            )
        except BaselineError as exc:
            problems.extend(exc.problems)
        if problems:
            raise TuningError(problems)
        object.__setattr__(self, "estimator_params", dict(self.estimator_params))
        object.__setattr__(self, "families", tuple(self.families) if self.families is not None else None)
        object.__setattr__(self, "budgets", json.loads(json.dumps(self.budgets)))
        object.__setattr__(self, "spaces", json.loads(json.dumps(self.spaces)))


TUNING_KEYS = {f.name for f in fields(TuningConfig)}


@dataclass(frozen=True)
class TrialRecord:
    number: int
    state: str
    params: dict[str, Any]
    score: float | None
    fold_scores: tuple[float, ...]
    fold_std: float | None
    duration_seconds: float | None
    failure_reason: str | None = None
    failure_category: str | None = None


@dataclass(frozen=True, eq=False)
class TuningResult:
    task: str
    metric: str
    direction: str
    config: TuningConfig
    space: dict[str, Any]
    trials: tuple[TrialRecord, ...]
    best: TrialRecord | None
    fingerprint: dict[str, Any]
    budgets: tuple[dict[str, Any], ...]
    versions: dict[str, str]
    notes: tuple[str, ...]
    directory: Path | None = None


def _require_optuna() -> Any:
    try:
        import optuna
    except ImportError:
        raise TuningUnavailableError(
            "tuning requires the 'optuna' package, which is not installed; "
            "install it with: pip install 'automl[tuning]'"
        ) from None
    return optuna


def _require_tracking(tracking: Any) -> None:
    """Raise the MLflow install hint before any work when tracking is requested but unavailable."""
    if tracking is not None and tracking.enabled:
        from automl.tracking import _require_mlflow

        _require_mlflow()


# ---- search spaces ----

_NUMERIC_KEYS = {"type", "low", "high", "log", "step"}


def _validate_mapping(space: dict[str, Any]) -> dict[str, dict[str, Any]]:
    problems: list[str] = []
    if not space:
        problems.append("search space must define at least one parameter")
    for name, spec in space.items():
        where = f"search space parameter '{name}'"
        if not isinstance(name, str) or not isinstance(spec, dict):
            problems.append(f"{where} must map a name to a spec table")
            continue
        kind = spec.get("type")
        if kind == "categorical":
            extra = set(spec) - {"type", "choices"}
            choices = spec.get("choices")
            if extra:
                problems.append(f"{where} has unknown keys {sorted(extra)}")
            if not isinstance(choices, (list, tuple)) or not choices:
                problems.append(f"{where} needs a non-empty 'choices' list")
            elif not all(c is None or isinstance(c, (bool, int, float, str)) for c in choices):
                problems.append(f"{where} choices must be None, bool, int, float, or str")
        elif kind in ("float", "int"):
            extra = set(spec) - _NUMERIC_KEYS
            if extra:
                problems.append(f"{where} has unknown keys {sorted(extra)}")
            low, high = spec.get("low"), spec.get("high")
            nums = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (low, high))
            if not nums:
                problems.append(f"{where} needs numeric 'low' and 'high'")
                continue
            if kind == "int" and not all(isinstance(v, int) for v in (low, high)):
                problems.append(f"{where} needs integer 'low' and 'high'")
            if not low < high:
                problems.append(f"{where} needs low < high, got {low} and {high}")
            if spec.get("log", False) and low <= 0:
                problems.append(f"{where} uses log scale and needs low > 0, got {low}")
            step = spec.get("step")
            if step is not None and (not isinstance(step, (int, float)) or isinstance(step, bool) or step <= 0):
                problems.append(f"{where} 'step' must be positive")
            elif step not in (None, 1) and spec.get("log", False):
                problems.append(f"{where} cannot combine 'log' with a 'step' other than 1")
        else:
            problems.append(f"{where} has unsupported type {kind!r}; use 'float', 'int', or 'categorical'")
    if problems:
        raise TuningError(problems)
    return json.loads(json.dumps(space))


def _mapping_sampler(space: dict[str, dict[str, Any]]) -> Callable[[Any], dict[str, Any]]:
    def sample(trial: Any) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, s in space.items():
            if s["type"] == "float":
                out[name] = trial.suggest_float(name, s["low"], s["high"], log=s.get("log", False), step=s.get("step"))
            elif s["type"] == "int":
                out[name] = trial.suggest_int(name, s["low"], s["high"], step=s.get("step") or 1, log=s.get("log", False))
            else:
                out[name] = trial.suggest_categorical(name, s["choices"])
        return out

    return sample


def _resolve_space(
    space: SearchSpace | None,
    config: TuningConfig,
    task: Task,
    stored: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Callable[[Any], dict[str, Any]]]:
    """Return the serializable space description and its sampling function."""
    overrides = config.spaces.get(config.estimator)
    if space is None and stored is not None:
        if stored["kind"] == "callable":
            raise TuningError("the saved study used a callable search space; pass it again as 'space'")
        space = stored["spec"]
        kind = stored["kind"]
    elif space is None:
        space = get_search_space(config.estimator, task, overrides)
        kind = "builtin"
    else:
        kind = "mapping"
        if overrides:
            raise TuningError(
                f"'spaces' overrides for '{config.estimator}' cannot be combined with a supplied search space; "
                "edit the supplied space instead"
            )
    if callable(space):
        return {"kind": "callable", "spec": None}, space
    if not isinstance(space, dict):
        raise TuningError("'space' must be a mapping of parameter specs or a callable taking an Optuna trial")
    spec = _validate_mapping(space)
    problems = controlled_param_problems(spec)
    problems += [
        f"search space parameter '{k}' is also fixed in 'estimator_params'" for k in spec if k in config.estimator_params
    ]
    if problems:
        raise TuningError(problems)
    return {"kind": kind, "spec": spec}, _mapping_sampler(spec)


# ---- evaluation ----


@dataclass
class _Setup:
    train: pd.DataFrame
    target: str
    task: Any
    metric: Any
    splits: SplitResult
    config: TuningConfig
    prep_cfg: PreprocessingConfig
    spec: Any
    plan: Any
    layout: FeatureLayout
    native: bool
    choice: Any
    pre: Any
    X: pd.DataFrame
    y: pd.Series
    fit_params: dict[str, Any]


def _prepare(train, target, id_columns, resolved_task, splits, cfg: TuningConfig, prep_cfg) -> _Setup:
    task, metric = resolved_task.task, resolved_task.metric
    problems: list[str] = []
    if target not in train.columns:
        problems.append(f"target column '{target}' is not in the train data")
    if splits.n_rows != len(train):
        problems.append(f"splits were built for {splits.n_rows} rows but train has {len(train)}")
    if splits.task != task:
        problems.append(f"splits were built for task '{splits.task}' but the resolved task is '{task}'")
    if problems:
        raise TuningError(problems)
    spec = get_estimator_spec(cfg.estimator)
    spec.require()
    native = spec.native_default if cfg.native_categoricals is None else cfg.native_categoricals
    if native and not spec.supports_native_categoricals:
        raise TuningError(f"setting 'native_categoricals' is true but '{spec.name}' has no native categorical support")
    choice = resolve_device(spec, cfg.device)
    plan = infer_columns(train, target, id_columns, prep_cfg)
    layout = FeatureLayout(len(plan.numeric), len(plan.categorical), native)
    return _Setup(
        train, target, task, metric, splits, cfg, prep_cfg, spec, plan, layout, native, choice,
        build_preprocessor(plan, prep_cfg, task, native_categoricals=native),
        train[[*plan.numeric, *plan.categorical]],  # model inputs exclude target, ids, and dropped columns
        train[target],
        {f"model__{k}": v for k, v in spec.fit_params(layout).items()},
    )


def _evaluate(
    setup: _Setup,
    params: dict[str, Any],
    on_fold: Callable[[list[float]], None] | None = None,
    *,
    progress: ProgressCallback | None = None,
    verbose: bool = False,
) -> list[float]:
    cfg = setup.config
    est = setup.spec.make(
        setup.task, {**cfg.estimator_params, **params}, cfg.seed, cfg.n_jobs, setup.choice.effective, setup.layout
    )
    if verbose:
        enable_native_verbosity(est)
    pipeline = Pipeline([("pre", setup.pre), ("model", est)])
    scores: list[float] = []
    fold_started = time.perf_counter()
    for split in setup.splits.splits:
        *_, rec = _run_fold(
            pipeline,
            setup.X,
            setup.y,
            split,
            setup.task,
            setup.metric,
            setup.metric.needs,
            setup.fit_params,
            progress,
            fold_started,
            "tuning",
        )
        scores.append(rec.score)
        if on_fold:
            on_fold(scores)
    return scores


def _fingerprint(setup: _Setup) -> dict[str, Any]:
    digest = hashlib.sha256()
    for s in setup.splits.splits:
        digest.update(np.asarray(s.train_idx, dtype=np.int64).tobytes())
        digest.update(np.asarray(s.valid_idx, dtype=np.int64).tobytes())
    cfg = setup.config
    return _plain(
        {
            "n_rows": len(setup.train),
            "target": setup.target,
            "features": [*setup.plan.numeric, *setup.plan.categorical],
            "task": setup.task.value,
            "metric": setup.metric.name,
            "direction": setup.metric.direction,
            "estimator": cfg.estimator,
            "seed": cfg.seed,
            "estimator_params": cfg.estimator_params,
            "native_categoricals": setup.native,
            "device": cfg.device,
            "preprocessing": asdict(setup.prep_cfg),
            "n_folds": len(setup.splits.splits),
            "splits_sha256": digest.hexdigest(),
        }
    )


# ---- study execution ----


def _record(ft: Any) -> TrialRecord:
    state = {"FAIL": "failed"}.get(ft.state.name, ft.state.name.lower())
    folds = tuple(float(v) for v in ft.user_attrs.get("fold_scores", ()))
    done = state == "complete"
    duration = None
    if ft.datetime_start and ft.datetime_complete:
        duration = (ft.datetime_complete - ft.datetime_start).total_seconds()
    return TrialRecord(
        number=ft.number,
        state=state,
        params=ft.user_attrs.get("params", _plain(ft.params)),
        score=float(ft.value) if done else None,
        fold_scores=folds,
        fold_std=float(np.std(folds)) if done and folds else None,
        duration_seconds=duration,
        failure_reason=ft.user_attrs.get("failure_reason"),
        failure_category=ft.user_attrs.get("failure_category") or ("other" if state == "failed" else None),
    )


class _NonFiniteScore(Exception):
    pass


class NoCompletedTrials(TuningError):
    """Every trial of a study failed or was pruned."""


_LIBRARY_PACKAGES = ("lightgbm", "xgboost", "catboost", "_catboost")


def categorize_failure(exc: BaseException) -> str:
    """Map a trial exception to one of FAILURE_CATEGORIES."""
    if isinstance(exc, _NonFiniteScore):
        return "non_finite_score"
    if isinstance(exc, TimeoutError):
        return "timeout"
    text = str(exc).lower()
    if isinstance(exc, BaselineError):
        if "non-finite" in text:
            return "non_finite_score"
        return "invalid_parameter" if "does not accept" in text or "cannot set" in text else "other"
    if isinstance(exc, TuningError):
        return "invalid_parameter"
    if type(exc).__name__ == "InvalidParameterError":
        return "invalid_parameter"
    if type(exc).__module__.split(".")[0] in _LIBRARY_PACKAGES:
        return "library_error"
    if isinstance(exc, (ValueError, TypeError)) and "param" in text:
        return "invalid_parameter"
    return "other"


def _run(
    optuna,
    study,
    setup: _Setup,
    sampler_fn,
    n_trials,
    timeout,
    *,
    progress: ProgressCallback | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    cfg = setup.config

    def objective(trial: Any) -> float:
        trial_started = time.perf_counter()
        if progress is not None:
            emit_progress(
                progress,
                "tuning",
                "trial.start",
                trial_started,
                model=cfg.estimator,
                trial=trial.number,
            )
        def on_fold(scores: list[float]) -> None:
            trial.set_user_attr("fold_scores", list(scores))
            if cfg.pruning and len(scores) < len(setup.splits.splits):
                trial.report(float(np.mean(scores)), step=len(scores) - 1)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        try:
            params = sampler_fn(trial)
            if not isinstance(params, dict):
                raise TuningError("a callable search space must return a dict of estimator parameters")
            trial.set_user_attr("params", _plain(params))
            scores = _evaluate(setup, params, on_fold, progress=progress, verbose=verbose)
            if not all(np.isfinite(scores)):
                raise _NonFiniteScore("a fold score is not finite")
        except optuna.TrialPruned:
            raise
        except Exception as exc:
            line = (str(exc).strip().splitlines() or [""])[0][:300]
            trial.set_user_attr("failure_reason", f"{type(exc).__name__}: {line}")
            trial.set_user_attr("failure_category", categorize_failure(exc))
            raise
        return float(np.mean(scores))

    def report_trial(study: Any, trial: Any) -> None:
        if progress is None:
            return
        completed = [item for item in study.trials if item.state.name == "COMPLETE" and item.value is not None]
        best = None
        if completed:
            key = max if study.direction.name == "MAXIMIZE" else min
            best = key(item.value for item in completed)
        if trial.state.name == "COMPLETE":
            emit_progress(
                progress,
                "tuning",
                "trial.complete",
                time.perf_counter() - (trial.duration.total_seconds() if trial.duration else 0),
                model=cfg.estimator,
                trial=trial.number,
                score=trial.value,
                best_score=best,
                direction=study.direction.name.lower(),
            )
        else:
            emit_progress(
                progress,
                "tuning",
                "trial.pruned" if trial.state.name == "PRUNED" else "trial.failed",
                time.perf_counter() - (trial.duration.total_seconds() if trial.duration else 0),
                model=cfg.estimator,
                trial=trial.number,
                state=trial.state.name.lower(),
                reason=trial.user_attrs.get("failure_reason"),
            )

    previous = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    start = time.perf_counter()
    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            catch=(Exception,),
            callbacks=[report_trial] if progress is not None else None,
        )
    finally:
        optuna.logging.set_verbosity(previous)
    elapsed = time.perf_counter() - start
    return {
        "n_trials": n_trials,
        "timeout_seconds": timeout,
        "elapsed_seconds": elapsed,
        "overrun_seconds": max(0.0, elapsed - timeout) if timeout else 0.0,
    }


def _build_result(optuna, study, setup, space, budgets, directory) -> TuningResult:
    trials = tuple(_record(t) for t in study.trials)
    done = [t for t in trials if t.state == "complete"]
    sign = 1 if setup.metric.direction == "maximize" else -1
    best = max(done, key=lambda t: (sign * t.score, -t.number)) if done else None
    notes = [
        "Scores are mean validation scores chosen by search and are optimistically biased; "
        "they are not a leaderboard estimate.",
        "Runs bounded by 'timeout_seconds' are not repeatable; a running trial is not interrupted.",
    ]
    if setup.choice.effective == "gpu":
        notes.append("GPU training is not guaranteed to be bit-reproducible")
    versions = _versions((setup.spec.package, setup.spec.version()) if setup.spec.package else None)
    versions["optuna"] = optuna.__version__
    return TuningResult(
        task=setup.task.value,
        metric=setup.metric.name,
        direction=setup.metric.direction,
        config=setup.config,
        space=space,
        trials=trials,
        best=best,
        fingerprint=_fingerprint(setup),
        budgets=tuple(budgets),
        versions=versions,
        notes=tuple(notes),
        directory=directory,
    )


def _finish(result: TuningResult) -> TuningResult:
    if result.directory is not None:
        _write(result)
    if result.best is None:
        states = [t.state for t in result.trials]
        reasons = sorted({t.failure_reason for t in result.trials if t.failure_reason})[:5]
        raise NoCompletedTrials(
            [f"no trial completed ({states.count('failed')} failed, {states.count('pruned')} pruned)"]
            + [f"trial failure: {r}" for r in reasons]
        )
    return result


def tune(
    train: pd.DataFrame,
    target: str,
    id_columns: tuple[str, ...],
    resolved_task: Any,
    splits: SplitResult,
    config: TuningConfig,
    space: SearchSpace | None = None,
    preprocessing_config: PreprocessingConfig | None = None,
    directory: str | Path | None = None,
    tracking: Any = None,
    *,
    progress: ProgressCallback | None = None,
    verbose: bool = False,
) -> TuningResult:
    """Tune one estimator with mean validation score over `splits`; persists to `directory` if given.

    A `TrackingConfig` passed as `tracking` mirrors the finished study to MLflow after local persistence.
    """
    optuna = _require_optuna()
    _require_tracking(tracking)
    if config.n_trials is None and config.timeout_seconds is None:
        raise TuningError("set 'n_trials' and/or 'timeout_seconds'; tune needs a budget (use tune_families for per-family budgets)")
    prep_cfg = preprocessing_config or PreprocessingConfig(encoder="ordinal", seed=config.seed)
    setup = _prepare(train, target, id_columns, resolved_task, splits, config, prep_cfg)
    space_desc, sampler_fn = _resolve_space(space, config, setup.task)
    storage = None
    path = None
    if directory is not None:
        path = Path(directory)
        try:
            path.mkdir(parents=True)
        except FileExistsError:
            raise TuningError(f"tuning directory {path} already exists; choose a new path or use resume_tuning") from None
        storage = f"sqlite:///{(path / STUDY_FILE).resolve()}"
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage,
        direction=setup.metric.direction,
        sampler=optuna.samplers.TPESampler(seed=config.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=PRUNER_STARTUP_TRIALS) if config.pruning else optuna.pruners.NopPruner(),
    )
    progress_started = time.perf_counter()
    if progress is not None:
        emit_progress(
            progress,
            "tuning",
            "stage.start",
            progress_started,
            model=config.estimator,
            n_trials=config.n_trials,
            timeout_seconds=config.timeout_seconds,
        )
    budget = _run(
        optuna,
        study,
        setup,
        sampler_fn,
        config.n_trials,
        config.timeout_seconds,
        progress=progress,
        verbose=verbose,
    )
    result = _finish(_build_result(optuna, study, setup, space_desc, [budget], path))
    _track(tracking, result, train, target, id_columns, resolved_task, prep_cfg)
    if progress is not None:
        emit_progress(progress, "tuning", "stage.complete", progress_started, model=config.estimator)
    return result


def resume_tuning(
    directory: str | Path,
    train: pd.DataFrame,
    target: str,
    id_columns: tuple[str, ...],
    resolved_task: Any,
    splits: SplitResult,
    n_trials: int | None = None,
    timeout_seconds: float | None = None,
    space: SearchSpace | None = None,
    preprocessing_config: PreprocessingConfig | None = None,
    tracking: Any = None,
    *,
    progress: ProgressCallback | None = None,
    verbose: bool = False,
) -> TuningResult:
    """Run additional trials on a saved study; the budget applies to the new trials only."""
    optuna = _require_optuna()
    _require_tracking(tracking)
    path = Path(directory)
    saved = load_tuning(path)
    if not (path / STUDY_FILE).is_file():
        raise TuningError(f"tuning directory {path} has no {STUDY_FILE}; it cannot be resumed")
    config = replace(saved.config, n_trials=n_trials, timeout_seconds=timeout_seconds)
    prep_cfg = preprocessing_config or PreprocessingConfig(encoder="ordinal", seed=config.seed)
    setup = _prepare(train, target, id_columns, resolved_task, splits, config, prep_cfg)
    current = _fingerprint(setup)
    diffs = [
        f"'{k}' differs from the saved study"
        for k in sorted(set(current) | set(saved.fingerprint))
        if current.get(k) != saved.fingerprint.get(k)
    ]
    if diffs:
        raise TuningError(["data or configuration does not match the saved study"] + diffs)
    space_desc, sampler_fn = _resolve_space(space, config, setup.task, saved.space)
    storage = f"sqlite:///{(path / STUDY_FILE).resolve()}"
    try:
        study = optuna.load_study(study_name=STUDY_NAME, storage=storage)
    except KeyError:
        raise TuningError(f"{path / STUDY_FILE} does not contain the saved study") from None
    # A fresh seed per resume keeps new random-startup trials from repeating earlier ones.
    study.sampler = optuna.samplers.TPESampler(seed=config.seed + len(study.trials))
    study.pruner = (
        optuna.pruners.MedianPruner(n_startup_trials=PRUNER_STARTUP_TRIALS) if config.pruning else optuna.pruners.NopPruner()
    )
    progress_started = time.perf_counter()
    if progress is not None:
        emit_progress(
            progress,
            "tuning",
            "stage.start",
            progress_started,
            model=config.estimator,
            n_trials=n_trials,
            timeout_seconds=timeout_seconds,
            resumed=True,
        )
    budget = _run(
        optuna,
        study,
        setup,
        sampler_fn,
        n_trials,
        timeout_seconds,
        progress=progress,
        verbose=verbose,
    )
    result = _finish(_build_result(optuna, study, setup, space_desc, [*saved.budgets, budget], path))
    _track(tracking, result, train, target, id_columns, resolved_task, prep_cfg, since_trial=len(saved.trials))
    if progress is not None:
        emit_progress(progress, "tuning", "stage.complete", progress_started, model=config.estimator, resumed=True)
    return result


def _track(tracking, result, train, target, id_columns, resolved_task, prep_cfg, since_trial: int = 0) -> None:
    if tracking is None or not tracking.enabled:
        return
    from automl.tracking import ModelData, track_tuning

    track_tuning(
        result, tracking, data=ModelData(train, target, tuple(id_columns), resolved_task),
        preprocessing_config=prep_cfg, since_trial=since_trial,
    )


# ---- persistence ----


def _trial_to_dict(t: TrialRecord) -> dict[str, Any]:
    d = asdict(t)
    d["fold_scores"] = list(t.fold_scores)
    return d


def _write(result: TuningResult) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "task": result.task,
        "metric": {"name": result.metric, "direction": result.direction},
        "config": asdict(result.config),
        "space": result.space,
        "fingerprint": result.fingerprint,
        "budgets": list(result.budgets),
        "best_trial": result.best.number if result.best else None,
        "trials": [_trial_to_dict(t) for t in result.trials],
        "versions": result.versions,
        "notes": list(result.notes),
        "files": {"study": STUDY_FILE},
    }
    final = result.directory / RESULT_FILE
    tmp = final.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False))
    os.replace(tmp, final)


def load_tuning(directory: str | Path) -> TuningResult:
    """Read `tuning.json` from a tuning directory; does not need Optuna."""
    directory = Path(directory)
    label = f"tuning result {directory}"

    def bad(msg: str) -> TuningError:
        return TuningError(f"{label}: {msg}")

    try:
        raw = json.loads((directory / RESULT_FILE).read_text())
    except FileNotFoundError:
        raise bad(f"{RESULT_FILE} not found") from None
    except json.JSONDecodeError as exc:
        raise bad(f"{RESULT_FILE} is not valid JSON: {exc}") from None
    if not isinstance(raw, dict) or raw.get("schema_version") not in READABLE_VERSIONS:
        raise bad(f"unsupported or missing 'schema_version'; expected one of {list(READABLE_VERSIONS)}")
    try:
        config = TuningConfig(**raw["config"])
        trials = tuple(
            TrialRecord(
                **{
                    **t,
                    "fold_scores": tuple(t["fold_scores"]),
                    "failure_category": t.get("failure_category") or ("other" if t["state"] == "failed" else None),
                }
            )
            for t in raw["trials"]
        )
        best_no = raw["best_trial"]
        best = next((t for t in trials if t.number == best_no), None) if best_no is not None else None
        if best_no is not None and best is None:
            raise bad(f"best_trial {best_no} is not among the trials")
        return TuningResult(
            task=raw["task"],
            metric=raw["metric"]["name"],
            direction=raw["metric"]["direction"],
            config=config,
            space=raw["space"],
            trials=trials,
            best=best,
            fingerprint=raw["fingerprint"],
            budgets=tuple(raw["budgets"]),
            versions=raw["versions"],
            notes=tuple(raw["notes"]),
            directory=directory,
        )
    except TuningError as exc:
        if str(exc).startswith(label):
            raise
        raise bad("; ".join(exc.problems)) from None
    except (KeyError, TypeError, ValueError) as exc:
        raise bad(f"missing or invalid field ({exc})") from None
