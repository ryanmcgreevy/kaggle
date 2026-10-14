"""Optional MLflow tracking that mirrors baseline, tuning, and family-comparison results.

Local result directories remain the source of truth; tracking only copies summaries, trial history,
and an optional fitted pipeline to MLflow. MLflow is imported lazily so the core package never needs it.
"""

from __future__ import annotations

import json
import re
import tempfile
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from automl.baseline import RESULT_FILE as BASELINE_FILE
from automl.baseline import BaselineResult, load_baseline, save_baseline
from automl.errors import TrackingError, TrackingUnavailableError
from automl.families import FAMILIES_FILE, FamiliesResult, load_tuning_families
from automl.families import _write as _write_families
from automl.metrics import Task
from automl.preprocessing import PreprocessingConfig
from automl.refit import build_pipeline
from automl.tuning import RESULT_FILE as TUNING_FILE
from automl.tuning import TrialRecord, TuningResult, _write as _write_tuning, load_tuning

DEFAULT_EXPERIMENT = "automl"
DEFAULT_MAX_TRIALS = 200
STORE_FILE = "mlflow.db"
ARTIFACT_DIR = "mlartifacts"
SELECTION_NOTE = (
    "Logged validation scores are measured on training data folds and, for tuned runs, optimistically "
    "biased by the search; they are not leaderboard estimates."
)
_CREDENTIAL_KEY = re.compile(r"pass(word|wd)?|secret|token|api[_-]?key|credential|auth", re.IGNORECASE)
_URI_PASSWORD = re.compile(r"(://[^/:@\s]*):[^@\s/]*@")
_BAD_KEY = re.compile(r"[^\w\-. /:]")


@dataclass(frozen=True)
class TrackingConfig:
    enabled: bool = True
    tracking_uri: str | None = None
    directory: str | None = None
    experiment_name: str = DEFAULT_EXPERIMENT
    run_name: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    log_model: bool = True
    max_trials_logged: int = DEFAULT_MAX_TRIALS

    def __post_init__(self) -> None:
        problems: list[str] = []
        if not isinstance(self.enabled, bool):
            problems.append("setting 'enabled' must be true or false")
        for key in ("tracking_uri", "directory", "run_name"):
            value = getattr(self, key)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                problems.append(f"setting '{key}' must be a non-empty string")
        if not isinstance(self.experiment_name, str) or not self.experiment_name.strip():
            problems.append("setting 'experiment_name' must be a non-empty string")
        if not isinstance(self.log_model, bool):
            problems.append("setting 'log_model' must be true or false")
        n = self.max_trials_logged
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            problems.append(f"setting 'max_trials_logged' must be an integer of at least 1, got {n!r}")
        if not isinstance(self.tags, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in self.tags.items()):
            problems.append("setting 'tags' must be a table of string names to string values")
        else:
            bad = sorted(k for k in self.tags if _CREDENTIAL_KEY.search(k))
            if bad:
                problems.append(f"setting 'tags' has credential-like names {bad}; secrets must not be logged")
        if problems:
            raise TrackingError(problems)
        object.__setattr__(self, "tags", dict(self.tags))


TRACKING_KEYS = {f.name for f in fields(TrackingConfig)}


@dataclass(frozen=True)
class ModelData:
    """Training rows needed to refit the complete pipeline for model logging."""

    train: pd.DataFrame
    target: str
    id_columns: tuple[str, ...]
    resolved_task: Any


@dataclass(frozen=True)
class TrackingResult:
    tracking_uri: str  # credentials redacted
    experiment_name: str
    experiment_id: str
    run_id: str
    child_run_ids: tuple[str, ...] = ()
    model_uri: str | None = None
    notes: tuple[str, ...] = ()


def redact_uri(text: str) -> str:
    """Replace a password in `scheme://user:password@host` with `***`."""
    return _URI_PASSWORD.sub(r"\1:***@", str(text))


def _require_mlflow() -> Any:
    try:
        import mlflow
    except ImportError:
        raise TrackingUnavailableError(
            "tracking requires the 'mlflow' package, which is not installed; "
            "install it with: pip install 'automl[mlflow]'"
        ) from None
    return mlflow


def _destination(cfg: TrackingConfig, directory: str | Path | None) -> tuple[str, str | None]:
    """Return (tracking URI, artifact location for a new experiment)."""
    if cfg.tracking_uri:
        return cfg.tracking_uri, None
    base = cfg.directory or directory
    if base is None:
        raise TrackingError(
            "no tracking destination: set 'tracking_uri', set 'directory', or pass a result directory"
        )
    base = Path(base).resolve()
    base.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{base / STORE_FILE}", (base / ARTIFACT_DIR).as_uri()


def _flatten(obj: Any, prefix: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            name = _BAD_KEY.sub("_", f"{prefix}.{key}" if prefix else str(key))
            out.update(_flatten(value, name))
    elif prefix:
        text = obj if isinstance(obj, str) else json.dumps(obj, default=str)
        out[prefix] = text[:500]
    return out


def _log_params(mlflow: Any, params: dict[str, str]) -> None:
    if params:
        mlflow.log_params(params)


def _tags(base: TrackingConfig, kind: str, versions: dict[str, str], notes: list[str], extra: dict[str, str]) -> dict[str, str]:
    tags = {
        **base.tags,
        "automl.kind": kind,
        "automl.notes": " | ".join(notes)[:5000],
        **{f"automl.version.{k}": v for k, v in versions.items()},
        **extra,
    }
    return tags


class _Session:
    """Sets the tracking URI and experiment for one logging call and restores the previous URI."""

    def __init__(self, cfg: TrackingConfig, directory: str | Path | None) -> None:
        self.mlflow = _require_mlflow()
        self.cfg = cfg
        self.uri, artifact_location = _destination(cfg, directory)
        self.shown = redact_uri(self.uri)
        self._previous = self.mlflow.get_tracking_uri()
        self._artifact_location = artifact_location
        self.experiment_id = ""

    def __enter__(self) -> "_Session":
        mlflow = self.mlflow
        try:
            mlflow.set_tracking_uri(self.uri)
            client = mlflow.MlflowClient(tracking_uri=self.uri)
            exp = client.get_experiment_by_name(self.cfg.experiment_name)
            self.experiment_id = (
                exp.experiment_id
                if exp is not None
                else client.create_experiment(self.cfg.experiment_name, artifact_location=self._artifact_location)
            )
            mlflow.set_experiment(experiment_id=self.experiment_id)
        except Exception as exc:
            self._restore()
            raise self.wrap(exc) from None
        return self

    def wrap(self, exc: BaseException) -> TrackingError:
        if isinstance(exc, TrackingError):
            return exc
        return TrackingError(
            f"MLflow tracking failed for {self.shown}: {type(exc).__name__}: {redact_uri(str(exc))}; "
            "local results are intact"
        )

    def _restore(self) -> None:
        try:
            self.mlflow.set_tracking_uri(self._previous)
        except Exception:
            pass

    def __exit__(self, exc_type, exc, tb) -> bool:
        mlflow = self.mlflow
        try:
            if exc is not None:
                try:
                    while mlflow.active_run() is not None:
                        mlflow.end_run("FAILED")
                except Exception:
                    pass
        finally:
            self._restore()
        if exc is not None and not isinstance(exc, TrackingError):
            raise self.wrap(exc) from exc
        return False


# ---- model logging ----


def _check_data(data: ModelData, n_rows: int, target: str | None, features: list[str]) -> None:
    problems: list[str] = []
    if len(data.train) != n_rows:
        problems.append(f"model data has {len(data.train)} rows but the result was built from {n_rows}")
    if target is not None and data.target != target:
        problems.append(f"model data target '{data.target}' differs from the result's '{target}'")
    if problems:
        raise TrackingError(problems)


def _fit_pipeline(
    data: ModelData, estimator: str, params: dict[str, Any], seed: int, n_jobs: int, device: str,
    native_categoricals: bool | None, prep_cfg: PreprocessingConfig, expected_features: list[str] | None,
) -> tuple[Pipeline, pd.DataFrame]:
    pipeline, plan, fit_params, _, _ = build_pipeline(
        data.train, data.target, data.id_columns, data.resolved_task.task, estimator, params, seed, n_jobs,
        device, native_categoricals, prep_cfg,
    )
    features = [*plan.numeric, *plan.categorical]
    if expected_features is not None and features != expected_features:
        raise TrackingError("model data columns differ from the columns the result was built with")
    X = data.train[features]
    pipeline.fit(X, data.train[data.target], **fit_params)
    return pipeline, X


def _log_pipeline(mlflow: Any, pipeline: Pipeline, X: pd.DataFrame, notes: list[str]) -> str | None:
    """Log the fitted pipeline without an input example; return its URI or None with a note."""
    try:
        import mlflow.sklearn
        from mlflow.models import infer_signature

        signature = None
        try:
            signature = infer_signature(X.head(20), pipeline.predict(X.head(20)))
        except Exception as exc:
            notes.append(f"model logged without a signature: {type(exc).__name__}")
        # cloudpickle is needed for pipelines holding third-party estimators; load only trusted stores.
        kwargs = {"signature": signature, "serialization_format": "cloudpickle"}
        try:
            info = mlflow.sklearn.log_model(pipeline, name="model", **kwargs)
        except TypeError:
            info = mlflow.sklearn.log_model(pipeline, artifact_path="model", **kwargs)
        uri = getattr(info, "model_uri", None)
        if uri:
            mlflow.set_tag("automl.model_uri", uri)
        return uri
    except Exception as exc:
        notes.append(f"model not logged: {type(exc).__name__}: {redact_uri(str(exc))[:200]}")
        return None


def _skip_model(cfg: TrackingConfig, data: ModelData | None, notes: list[str]) -> bool:
    """True when the model should be fitted and logged; records why not otherwise."""
    if not cfg.log_model:
        notes.append("model not logged: 'log_model' is false")
        return False
    if data is None:
        notes.append("model not logged: no training data was supplied to refit the pipeline")
        return False
    return True


# ---- trial logging ----


def _select_trials(trials: tuple[TrialRecord, ...], best: TrialRecord | None, cap: int) -> tuple[list[TrialRecord], int]:
    ordered = sorted(trials, key=lambda t: t.number)
    if len(ordered) <= cap:
        return ordered, 0
    keep = {t.number for t in ordered[:cap]}
    if best is not None:
        keep = set(sorted(keep - {best.number})[: cap - 1]) | {best.number}
    chosen = [t for t in ordered if t.number in keep]
    return chosen, len(ordered) - len(chosen)


def _log_trial(mlflow: Any, t: TrialRecord, cfg: TrackingConfig, is_best: bool) -> str:
    with mlflow.start_run(run_name=f"trial-{t.number}", nested=True) as run:
        tags = {**cfg.tags, "automl.kind": "trial", "automl.trial_state": t.state, "automl.is_best": str(is_best).lower()}
        if t.failure_reason:
            tags["automl.failure_reason"] = t.failure_reason[:500]
        if t.failure_category:
            tags["automl.failure_category"] = t.failure_category
        mlflow.set_tags(tags)
        _log_params(mlflow, _flatten(t.params, "param"))
        if t.score is not None:
            mlflow.log_metric("score", t.score)
        if t.fold_std is not None:
            mlflow.log_metric("fold_std", t.fold_std)
        for i, s in enumerate(t.fold_scores):
            mlflow.log_metric("fold_score", s, step=i)
        if t.duration_seconds is not None:
            mlflow.log_metric("duration_seconds", t.duration_seconds)
        return run.info.run_id


def _log_study(mlflow: Any, result: TuningResult, cfg: TrackingConfig, since: int, run_ids: list[str]) -> dict[str, str]:
    trials = tuple(t for t in result.trials if t.number >= since)
    chosen, dropped = _select_trials(trials, result.best if result.best in trials else None, cfg.max_trials_logged)
    for t in chosen:
        run_ids.append(_log_trial(mlflow, t, cfg, result.best is not None and t.number == result.best.number))
    return {"automl.trials_logged": str(len(chosen)), "automl.trials_not_logged": str(dropped)}


def _study_summary(mlflow: Any, result: TuningResult, since: int) -> None:
    c = result.config
    _log_params(
        mlflow,
        {
            **_flatten({k: v for k, v in asdict(c).items() if k not in ("families", "budgets", "spaces")}, "config"),
            **_flatten({"task": result.task, "metric": result.metric, "direction": result.direction}),
            **_flatten({"space_kind": result.space.get("kind")}),
            **_flatten(result.space.get("spec") or {}, "space"),
        },
    )
    done = [t for t in result.trials if t.state == "complete"]
    mlflow.log_metric("n_trials", len(result.trials))
    mlflow.log_metric("n_complete", len(done))
    mlflow.log_metric("n_failed", sum(t.state == "failed" for t in result.trials))
    mlflow.log_metric("n_pruned", sum(t.state == "pruned" for t in result.trials))
    if result.best is not None:
        mlflow.log_metric("best_score", result.best.score)
        mlflow.log_metric("best_fold_std", result.best.fold_std or 0.0)
        mlflow.set_tag("automl.best_trial", str(result.best.number))
        _log_params(mlflow, _flatten(result.best.params, "best_param"))


def _log_study_model(
    mlflow: Any, result: TuningResult, cfg: TrackingConfig, data: ModelData | None, prep_cfg: PreprocessingConfig | None,
    notes: list[str],
) -> str | None:
    if result.best is None or not _skip_model(cfg, data, notes):
        return None
    c = result.config
    fp = result.fingerprint
    try:
        _check_data(data, fp["n_rows"], fp["target"], fp["features"])
        pipeline, X = _fit_pipeline(
            data, c.estimator, {**c.estimator_params, **result.best.params}, c.seed, c.n_jobs, c.device,
            c.native_categoricals, prep_cfg or PreprocessingConfig(encoder="ordinal", seed=c.seed), fp["features"],
        )
    except TrackingError:
        raise
    except Exception as exc:
        notes.append(f"model not logged: {type(exc).__name__}: {redact_uri(str(exc))[:200]}")
        return None
    return _log_pipeline(mlflow, pipeline, X, notes)


# ---- public API ----


def track_baseline(
    result: BaselineResult | str | Path,
    tracking: TrackingConfig,
    data: ModelData | None = None,
    directory: str | Path | None = None,
    preprocessing_config: PreprocessingConfig | None = None,
) -> TrackingResult:
    """Mirror a baseline result (object or saved directory) to MLflow as one run."""
    _require_mlflow()
    if isinstance(result, (str, Path)):
        directory = directory or result
        source = Path(result)
        result = load_baseline(source)
    else:
        source = None
    cfg = tracking
    notes: list[str] = [SELECTION_NOTE]
    with _Session(cfg, directory) as s:
        mlflow = s.mlflow
        name = cfg.run_name or f"baseline-{result.estimator_key}-seed{result.config.seed}"
        with mlflow.start_run(run_name=name) as run:
            _log_params(
                mlflow,
                {
                    **_flatten({k: v for k, v in asdict(result.config).items()}, "config"),
                    **_flatten(asdict(result.preprocessing_config), "preprocessing"),
                    **_flatten(result.split, "split"),
                    **_flatten(
                        {
                            "task": result.task.value,
                            "metric": result.metric,
                            "direction": result.direction,
                            "estimator": result.estimator_key,
                            "n_rows": result.n_rows,
                            "n_numeric": len(result.plan.numeric),
                            "n_categorical": len(result.plan.categorical),
                            "categorical_handling": result.categorical_handling,
                            "device": result.device.get("effective"),
                        }
                    ),
                },
            )
            mlflow.log_metric("mean_score", result.mean)
            mlflow.log_metric("std_score", result.std)
            for f in result.folds:
                mlflow.log_metric("fold_score", f.score, step=f.fold)
                mlflow.log_metric("fold_fit_seconds", f.fit_seconds, step=f.fold)
            model_uri = None
            if _skip_model(cfg, data, notes):
                try:
                    _check_data(data, result.n_rows, None, [])
                    pipeline, X = _fit_pipeline(
                        data, result.estimator_key, result.config.estimator_params, result.config.seed,
                        result.config.n_jobs, result.config.device, result.config.native_categoricals,
                        result.preprocessing_config, [*result.plan.numeric, *result.plan.categorical],
                    )
                    model_uri = _log_pipeline(mlflow, pipeline, X, notes)
                except TrackingError:
                    raise
                except Exception as exc:
                    notes.append(f"model not logged: {type(exc).__name__}: {redact_uri(str(exc))[:200]}")
            if source is not None and (source / BASELINE_FILE).is_file():
                mlflow.log_artifact(str(source / BASELINE_FILE))
            else:
                with tempfile.TemporaryDirectory() as tmp:  # the temp copy also holds predictions, which are not logged
                    stripped = replace(result, config=replace(result.config, save_models=False))
                    save_baseline(stripped, Path(tmp) / "b")
                    mlflow.log_artifact(str(Path(tmp) / "b" / BASELINE_FILE))
            mlflow.set_tags(_tags(cfg, "baseline", result.versions, notes, {}))
            return TrackingResult(s.shown, cfg.experiment_name, s.experiment_id, run.info.run_id, (), model_uri, tuple(notes))


def _log_tuning_artifact(mlflow: Any, result: TuningResult) -> None:
    if result.directory is not None and (result.directory / TUNING_FILE).is_file():
        mlflow.log_artifact(str(result.directory / TUNING_FILE))
        return
    with tempfile.TemporaryDirectory() as tmp:
        _write_tuning(replace(result, directory=Path(tmp)))
        mlflow.log_artifact(str(Path(tmp) / TUNING_FILE))


def _track_tuning(
    s: _Session, result: TuningResult, cfg: TrackingConfig, data: ModelData | None,
    preprocessing_config: PreprocessingConfig | None, since: int, run_name: str, extra: dict[str, str], nested: bool,
    with_model: bool = True,
) -> TrackingResult:
    mlflow = s.mlflow
    notes: list[str] = [] if nested else [SELECTION_NOTE]
    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        _study_summary(mlflow, result, since)
        child_ids: list[str] = []
        counts = _log_study(mlflow, result, cfg, since, child_ids)
        model_uri = _log_study_model(mlflow, result, cfg, data, preprocessing_config, notes) if with_model else None
        _log_tuning_artifact(mlflow, result)
        mlflow.set_tags(_tags(cfg, "tuning", result.versions, notes, {**counts, **extra}))
        return TrackingResult(s.shown, cfg.experiment_name, s.experiment_id, run.info.run_id, tuple(child_ids), model_uri, tuple(notes))


def track_tuning(
    result: TuningResult | str | Path,
    tracking: TrackingConfig,
    data: ModelData | None = None,
    directory: str | Path | None = None,
    preprocessing_config: PreprocessingConfig | None = None,
    since_trial: int = 0,
) -> TrackingResult:
    """Mirror a tuning study to MLflow: a parent run plus one nested run per trial (from `since_trial`)."""
    _require_mlflow()
    if isinstance(result, (str, Path)):
        directory = directory or result
        result = load_tuning(result)
    directory = directory or result.directory
    cfg = tracking
    extra = {"automl.since_trial": str(since_trial)} if since_trial else {}
    with _Session(cfg, directory) as s:
        name = cfg.run_name or f"tuning-{result.config.estimator}-seed{result.config.seed}"
        return _track_tuning(s, result, cfg, data, preprocessing_config, since_trial, name, extra, nested=False)


def track_families(
    result: FamiliesResult | str | Path,
    tracking: TrackingConfig,
    data: ModelData | None = None,
    directory: str | Path | None = None,
    preprocessing_config: PreprocessingConfig | None = None,
) -> TrackingResult:
    """Mirror a family comparison: a top-level run, one nested run per family, and nested trial runs."""
    _require_mlflow()
    if isinstance(result, (str, Path)):
        directory = directory or result
        result = load_tuning_families(result)
    directory = directory or result.directory
    cfg = tracking
    notes: list[str] = [SELECTION_NOTE, *result.notes]
    with _Session(cfg, directory) as s:
        mlflow = s.mlflow
        name = cfg.run_name or f"families-seed{result.config.seed}"
        with mlflow.start_run(run_name=name) as run:
            _log_params(mlflow, _flatten({"task": result.task, "metric": result.metric, "direction": result.direction}))
            _log_params(mlflow, _flatten(result.comparison and {r["family"]: r.get("rank") for r in result.comparison}, "rank"))
            child_ids: list[str] = []
            versions: dict[str, str] = {}
            for o in result.outcomes:
                if o.result is None:
                    continue
                versions.update(o.result.versions)
                sub = _track_tuning(
                    s, o.result, cfg, None, None, 0, f"family-{o.family}",
                    {"automl.family": o.family, "automl.family_status": o.status, **({"automl.error": o.error[:500]} if o.error else {})},
                    nested=True, with_model=False,
                )
                child_ids.append(sub.run_id)
                child_ids.extend(sub.child_run_ids)
            model_uri = None
            best = next((o for o in result.outcomes if o.family == result.best_family and o.result is not None), None)
            if best is not None:
                mlflow.set_tag("automl.best_family", best.family)
                mlflow.log_metric("best_score", best.result.best.score)
                model_uri = _log_study_model(mlflow, best.result, cfg, data, preprocessing_config, notes)
            if result.directory is not None and (result.directory / FAMILIES_FILE).is_file():
                mlflow.log_artifact(str(result.directory / FAMILIES_FILE))
            else:
                with tempfile.TemporaryDirectory() as tmp:
                    plan = {o.family: o.budget for o in result.outcomes}
                    entries = [{"family": o.family, "status": o.status, "directory": o.family, "error": o.error} for o in result.outcomes]
                    _write_families(
                        Path(tmp), result.config, result.task, result.metric, result.direction, plan, entries,
                        result.comparison, result.best_family, result.notes,
                    )
                    mlflow.log_artifact(str(Path(tmp) / FAMILIES_FILE))
            mlflow.set_tags(_tags(cfg, "tuning_families", versions, notes, {}))
            return TrackingResult(s.shown, cfg.experiment_name, s.experiment_id, run.info.run_id, tuple(child_ids), model_uri, tuple(notes))


def track_submission(
    metadata: dict[str, Any],
    directory: str | Path,
    tracking: TrackingConfig,
) -> TrackingResult:
    """Log submission.json and submission.csv from a saved submission directory as run artifacts."""
    _require_mlflow()
    cfg = tracking
    out = Path(directory)
    src = metadata.get("source", {})
    notes = ["The submission holds raw predictions; nothing was submitted to Kaggle."]
    with _Session(cfg, out) as s:
        mlflow = s.mlflow
        with mlflow.start_run(run_name=cfg.run_name or "submission") as run:
            _log_params(mlflow, _flatten({"n_rows": metadata["n_rows"], "columns": metadata["columns"], "schema_source": metadata.get("schema_source")}))
            for name in ("submission.json", "submission.csv"):
                mlflow.log_artifact(str(out / name))
            mlflow.set_tags(
                _tags(
                    cfg, "submission", {}, notes,
                    {"automl.source_kind": str(src.get("source_kind")), "automl.label": str(src.get("label"))},
                )
            )
            return TrackingResult(s.shown, cfg.experiment_name, s.experiment_id, run.info.run_id, (), None, tuple(notes))


def track_refit(
    pipeline: Pipeline,
    X: pd.DataFrame,
    metadata: dict[str, Any],
    tracking: TrackingConfig,
    directory: str | Path | None = None,
) -> TrackingResult:
    """Log a fitted refit pipeline with its selection, provenance, and refit.json (no data or predictions)."""
    _require_mlflow()
    cfg = tracking
    sel = metadata["selection"]
    notes: list[str] = [SELECTION_NOTE]
    with _Session(cfg, directory) as s:
        mlflow = s.mlflow
        with mlflow.start_run(run_name=cfg.run_name or f"refit-{sel['estimator']}-seed{sel['seed']}") as run:
            _log_params(
                mlflow,
                {
                    **_flatten({k: v for k, v in sel.items() if k not in ("versions", "notes", "features")}, "selection"),
                    **_flatten({"task": metadata["task"], "n_train_rows": metadata["data"]["n_train_rows"]}),
                },
            )
            for key, name in (("validation_mean", "validation_mean_score"), ("validation_std", "validation_std_score")):
                if sel.get(key) is not None:
                    mlflow.log_metric(name, sel[key])
            mlflow.log_metric("fit_seconds", metadata["fit_seconds"])
            model_uri = _log_pipeline(mlflow, pipeline, X, notes) if cfg.log_model else None
            if not cfg.log_model:
                notes.append("model not logged: 'log_model' is false")
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "refit.json"
                path.write_text(json.dumps(metadata, indent=2, allow_nan=False))
                mlflow.log_artifact(str(path))
            mlflow.set_tags(
                _tags(
                    cfg, "refit", metadata.get("versions", {}), notes,
                    {"automl.source_kind": sel["source_kind"], "automl.label": sel["label"]},
                )
            )
            return TrackingResult(s.shown, cfg.experiment_name, s.experiment_id, run.info.run_id, (), model_uri, tuple(notes))
