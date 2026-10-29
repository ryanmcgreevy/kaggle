"""Stage functions behind the CLI commands; each calls the public Python API and returns a JSON-ready dict."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from automl.baseline import RESULT_FILE as BASELINE_FILE
from automl.baseline import run_baseline, save_baseline
from automl.config import DataConfig, load_config
from automl.eda import save_eda, summarize_bundle
from automl.ensemble import EnsembleConfig, run_ensemble, run_saved_ensemble
from automl.errors import (
    BaselineError,
    DataContractError,
    EnsembleError,
    RefitError,
    RunError,
    SubmissionError,
    TrackingError,
    TrackingUnavailableError,
    TuningError,
    ValidationSplitError,
)
from automl.estimators import ESTIMATOR_NAMES, estimator_available, get_estimator_spec
from automl.families import FAMILIES_FILE, resume_tuning_families, tune_families
from automl.loader import DataBundle, load_data
from automl.model_report import write_model_report
from automl.progress import ProgressCallback, emit_progress
from automl.refit import METADATA_FILE as REFIT_FILE
from automl.refit import MODEL_FILE, PREDICTIONS_FILE, load_refit, refit_and_predict
from automl.report import _require_matplotlib, write_eda_report
from automl.submission import build_submission
from automl.task import ResolvedTask, resolve_task_metric
from automl.tracking import ModelData, TrackingConfig, _require_mlflow, redact_uri, track_baseline
from automl.tuning import RESULT_FILE as TUNING_FILE
from automl.tuning import TuningConfig, _require_optuna, resume_tuning, tune
from automl.validation import SplitResult, load_splits, make_splits, save_splits

Log = Callable[[str], None]


@dataclass(frozen=True)
class RunContext:
    config: DataConfig
    bundle: DataBundle
    resolved: ResolvedTask
    config_path: Path

    @property
    def data_files(self) -> dict[str, str | None]:
        sample = self.config.sample_submission_path
        return {
            "config": str(self.config_path),
            "train": self.config.train_path.name,
            "test": self.config.test_path.name,
            "sample_submission": sample.name if sample is not None else None,
        }


def load_run(config_path: str | Path) -> RunContext:
    """Load the TOML, the CSVs, and the explicit task and metric."""
    config = load_config(config_path)
    bundle = load_data(config)
    return RunContext(config, bundle, resolve_task_metric(bundle), Path(config_path))


def _plain(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str, allow_nan=False))


def _new_dir(out: Path, what: str, error: type[DataContractError], hint: str = "choose a new path") -> None:
    if out.exists():
        raise error(f"{what} directory {out} already exists; {hint}")


def _tracking(ctx: RunContext, track: bool) -> TrackingConfig | None:
    """Return the enabled `[tracking]` config for `--track`, checking MLflow before any work."""
    if not track:
        return None
    cfg = ctx.config.tracking_config
    if cfg is None:
        raise TrackingError(
            "--track needs a [tracking] table in the run TOML with a 'tracking_uri' or 'directory'; none was found"
        )
    _require_mlflow()
    return replace(cfg, enabled=True)


def _tracking_uri(ctx: RunContext) -> str | None:
    cfg = ctx.config.tracking_config
    if cfg is None:
        return None
    if cfg.tracking_uri:
        return cfg.tracking_uri
    if cfg.directory:
        return f"sqlite:///{(Path(cfg.directory).expanduser().resolve() / 'mlflow.db')}"
    return None


def _tracked(where: Path, fn: Callable[[], Any]) -> Any:
    """Run a step that logs to MLflow after local files are written, noting that the local files are safe."""
    try:
        return fn()
    except TrackingUnavailableError:
        raise
    except TrackingError as exc:
        raise TrackingError(f"results were saved to {where}, but tracking failed: {redact_uri(str(exc))}") from None


def _tracking_info(tracked: Any, enabled: bool) -> dict[str, Any] | None:
    if tracked is not None:
        return {
            "tracking_uri": tracked.tracking_uri,
            "experiment": tracked.experiment_name,
            "run_id": tracked.run_id,
            "model_uri": tracked.model_uri,
        }
    return {"logged": True} if enabled else None


def _splits(ctx: RunContext, path: str | Path | None) -> tuple[SplitResult, dict[str, Any]]:
    train = ctx.bundle.train
    if path is None:
        result = make_splits(train[ctx.config.target], ctx.resolved, ctx.config.validation_config)
        return result, {"source": "config", "strategy": result.config.strategy, "n_splits": len(result.splits)}
    try:
        result = load_splits(path)
    except ValidationSplitError as exc:
        raise ValidationSplitError(f"{exc}; create it with 'automl split'") from None
    problems = []
    if result.n_rows != len(train):
        problems.append(f"split artifact {path} covers {result.n_rows} rows but the train data has {len(train)}")
    if result.task is not ctx.resolved.task:
        problems.append(
            f"split artifact {path} was made for task '{result.task.value}' but the resolved task is "
            f"'{ctx.resolved.task.value}'"
        )
    if problems:
        raise ValidationSplitError(problems)
    return result, {"source": str(path), "strategy": result.config.strategy, "n_splits": len(result.splits)}


def _ids(ctx: RunContext) -> tuple[str, ...]:
    return tuple(ctx.config.id_columns)


def _not_installed_note() -> str | None:
    missing = [
        f"{name} (pip install 'automl[{get_estimator_spec(name).extra}]')"
        for name in ESTIMATOR_NAMES
        if not estimator_available(name)
    ]
    return f"families not tuned because their library is not installed: {', '.join(missing)}" if missing else None


# ---- stages that existed before the modeling commands ----


def validate(ctx: RunContext) -> dict[str, Any]:
    sample = ctx.bundle.sample_submission
    return {
        "status": "valid",
        "config": str(ctx.config_path),
        "target": ctx.config.target,
        "task": ctx.resolved.task.value,
        "metric": ctx.resolved.metric.name,
        "train_rows": len(ctx.bundle.train),
        "test_rows": len(ctx.bundle.test),
        "sample_submission_rows": len(sample) if sample is not None else None,
    }


def split(ctx: RunContext, output_dir: str | Path) -> dict[str, Any]:
    out = Path(output_dir)
    path = out / "splits.json"
    result = make_splits(ctx.bundle.train[ctx.config.target], ctx.resolved, ctx.config.validation_config)
    out.mkdir(parents=True, exist_ok=True)
    save_splits(result, path)
    return {
        "status": "created",
        "path": str(path),
        "strategy": result.config.strategy,
        "task": result.task.value,
        "n_rows": result.n_rows,
        "n_splits": len(result.splits),
    }


def eda(ctx: RunContext, output_dir: str | Path) -> dict[str, Any]:
    out = Path(output_dir)
    path = out / "eda.json"
    result = summarize_bundle(ctx.bundle, task=ctx.resolved)
    out.mkdir(parents=True, exist_ok=True)
    save_eda(result, path)
    return {
        "status": "created",
        "path": str(path),
        "schema_version": result.schema_version,
        "n_rows": result.dataset.n_rows,
        "n_columns": result.dataset.n_columns,
    }


def eda_report(
    config: DataConfig, eda_path: str | Path, output_dir: str | Path, *, filename: str = "eda_report.pdf", overwrite: bool = False
) -> dict[str, Any]:
    result = write_eda_report(eda_path, output_dir, filename=filename, config=config.report_config, overwrite=overwrite)
    return {"status": "created", **result.to_dict()}


# ---- baseline and tuning ----


def baseline(
    ctx: RunContext,
    output_dir: str | Path,
    *,
    splits_path: str | Path | None = None,
    estimator: str | None = None,
    track: bool = False,
    progress: ProgressCallback | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    out = Path(output_dir)
    _new_dir(out, "baseline", BaselineError)
    tracking = _tracking(ctx, track)
    cfg = ctx.config.baseline_config
    notes: list[str] = []
    if estimator is not None and estimator != cfg.estimator:
        cfg = replace(cfg, estimator=estimator, estimator_params={})
        notes.append("configured 'estimator_params' were not applied because --estimator selects another estimator")
    splits, splits_info = _splits(ctx, splits_path)
    train, target, ids = ctx.bundle.train, ctx.config.target, _ids(ctx)
    prep = ctx.config.preprocessing_config
    result = run_baseline(
        train, target, ids, ctx.resolved, splits, cfg, prep, progress=progress, verbose=verbose
    )
    save_baseline(result, out)
    tracked = None
    if tracking is not None:
        data = ModelData(train, target, ids, ctx.resolved)
        tracked = _tracked(
            out, lambda: track_baseline(result, tracking, data=data, directory=out, preprocessing_config=prep)
        )
    return {
        "status": "created",
        "path": str(out / BASELINE_FILE),
        "directory": str(out),
        "estimator": result.estimator_key,
        "task": result.task.value,
        "metric": result.metric,
        "direction": result.direction,
        "mean": result.mean,
        "std": result.std,
        "n_folds": len(result.folds),
        "model_reference": str(out / "fitted-model") if cfg.save_models else None,
        "splits": splits_info,
        "data": ctx.data_files,
        "notes": notes,
        "tracking": _tracking_info(tracked, tracking is not None),
    }


def _tuning_config(
    ctx: RunContext, estimator: str | None, families: Sequence[str] | None, n_trials: int | None, timeout_seconds: float | None
) -> TuningConfig:
    """Merge `[tuning]` with the command-line overrides; a trial or time budget is required."""
    base = ctx.config.tuning_config
    changes: dict[str, Any] = {}
    if n_trials is not None:
        changes["n_trials"] = n_trials
    if timeout_seconds is not None:
        changes["timeout_seconds"] = timeout_seconds
    if estimator is not None:
        changes["estimator"] = estimator
        if base is None or base.estimator != estimator:
            changes["estimator_params"] = {}
    if families is not None:
        changes["families"] = tuple(families)
    return TuningConfig(**changes) if base is None else replace(base, **changes)


def _tuning_kind(directory: Path) -> str:
    if (directory / FAMILIES_FILE).is_file():
        return "families"
    if (directory / TUNING_FILE).is_file():
        return "tuning"
    raise TuningError(f"{directory} has no {FAMILIES_FILE} or {TUNING_FILE}; --resume needs an existing tuning directory")


def _budget_text(n_trials: int | None, timeout_seconds: float | None) -> str:
    parts = []
    if n_trials is not None:
        parts.append(f"n_trials={n_trials}")
    if timeout_seconds is not None:
        parts.append(f"timeout_seconds={timeout_seconds:g}")
    return ", ".join(parts) or "per-family budgets"


def tune_stage(
    ctx: RunContext,
    output_dir: str | Path,
    *,
    splits_path: str | Path | None = None,
    estimator: str | None = None,
    families: Sequence[str] | None = None,
    n_trials: int | None = None,
    timeout_seconds: float | None = None,
    resume: bool = False,
    track: bool = False,
    log: Log | None = None,
    progress: ProgressCallback | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    if estimator is not None and families is not None:
        raise TuningError("use either --estimator (one estimator) or --families (several), not both")
    if resume and estimator is not None:
        raise TuningError("--estimator applies only to a new study; a resumed study keeps its saved estimator")
    out = Path(output_dir)
    _require_optuna()
    tracking = _tracking(ctx, track)
    train, target, ids = ctx.bundle.train, ctx.config.target, _ids(ctx)
    prep = ctx.config.preprocessing_config
    notes: list[str] = []
    if resume:
        kind = _tuning_kind(out)
        splits, splits_info = _splits(ctx, splits_path)
        if log is not None:
            log(f"resuming {kind} study in {out} with budget {_budget_text(n_trials, timeout_seconds)}")
        if kind == "families":
            result = _tracked(
                out,
                lambda: resume_tuning_families(
                    out, train, target, ids, ctx.resolved, splits, n_trials=n_trials, timeout_seconds=timeout_seconds,
                    families=list(families) if families else None, preprocessing_config=prep, tracking=tracking,
                    progress=progress, verbose=verbose,
                ),
            )
        else:
            result = _tracked(
                out,
                lambda: resume_tuning(
                    out, train, target, ids, ctx.resolved, splits, n_trials=n_trials, timeout_seconds=timeout_seconds,
                    preprocessing_config=prep, tracking=tracking, progress=progress, verbose=verbose,
                ),
            )
        status = "resumed"
    else:
        _new_dir(out, "tuning", TuningError, "choose a new path or pass --resume")
        cfg = _tuning_config(ctx, estimator, families, n_trials, timeout_seconds)
        splits, splits_info = _splits(ctx, splits_path)
        kind = "tuning" if estimator is not None else "families"
        if kind == "families":
            if families is None and cfg.families is None:
                note = _not_installed_note()
                if note:
                    notes.append(note)
                    if log is not None:
                        log(note)
            if log is not None:
                log(
                    f"tuning families {list(families or cfg.families or [n for n in ESTIMATOR_NAMES if estimator_available(n)])} "
                    f"with budget {_budget_text(cfg.n_trials, cfg.timeout_seconds)}"
                )
            result = _tracked(
                out,
                lambda: tune_families(
                    train, target, ids, ctx.resolved, splits, cfg, preprocessing_config=prep, directory=out,
                    tracking=tracking, progress=progress, verbose=verbose,
                ),
            )
        else:
            if log is not None:
                log(f"tuning {estimator} with budget {_budget_text(cfg.n_trials, cfg.timeout_seconds)}")
            result = _tracked(
                out,
                lambda: tune(
                    train, target, ids, ctx.resolved, splits, cfg, preprocessing_config=prep, directory=out,
                    tracking=tracking, progress=progress, verbose=verbose,
                ),
            )
        status = "created"
    payload: dict[str, Any] = {
        "status": status,
        "kind": kind,
        "directory": str(out),
        "path": str(out / (FAMILIES_FILE if kind == "families" else TUNING_FILE)),
        "task": result.task,
        "metric": result.metric,
        "direction": result.direction,
        "splits": splits_info,
        "data": ctx.data_files,
        "notes": notes,
        "tracking": _tracking_info(None, tracking is not None),
    }
    if kind == "families":
        payload["best_family"] = result.best_family
        payload["model_references"] = {
            outcome.family: str(out / outcome.family / "fitted-model")
            for outcome in result.outcomes
            if outcome.result is not None and outcome.result.best is not None
        }
        payload["families"] = [
            {k: row.get(k) for k in ("family", "status", "rank", "best_score", "best_fold_std", "error")}
            for row in result.comparison
        ]
    else:
        best = result.best
        payload["model_reference"] = str(out / "fitted-model") if result.model_reference else None
        payload["n_trials"] = len(result.trials)
        payload["best"] = (
            {"trial": best.number, "score": best.score, "std": best.fold_std, "params": best.params} if best else None
        )
    return _plain(payload)


# ---- reports, refit, submission ----


def ensemble_stage(
    ctx: RunContext,
    sources: Sequence[str | Path],
    output_dir: str | Path,
    *,
    splits_path: str | Path | None = None,
    method: str = "stacking",
    mode: str = "evaluate",
    labels: Sequence[str] | None = None,
    stack_cv: int = 5,
    n_jobs: int | None = None,
    meta_estimator: str | None = None,
    meta_estimator_params: dict[str, Any] | None = None,
    progress: ProgressCallback | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    out = Path(output_dir)
    _new_dir(out, "ensemble", EnsembleError)
    config = EnsembleConfig(
        method=method,
        mode=mode,
        seed=ctx.config.baseline_config.seed,
        n_jobs=ctx.config.baseline_config.n_jobs if n_jobs is None else n_jobs,
        stack_cv=stack_cv,
        meta_estimator=meta_estimator,
        meta_estimator_params=meta_estimator_params or {},
    )
    if config.mode == "fit":
        if splits_path is not None:
            raise EnsembleError("--splits applies only to evaluation mode; omit it when using --mode fit")
        splits, splits_info = None, None
    else:
        splits, splits_info = _splits(ctx, splits_path)
    result = run_ensemble(
        ctx.bundle.train,
        ctx.config.target,
        _ids(ctx),
        ctx.resolved,
        splits,
        sources,
        config,
        labels=tuple(labels) if labels is not None else None,
        directory=out,
        progress=progress,
        verbose=verbose,
    )
    return _plain(
        {
            "status": "created",
            "directory": str(out),
            "path": str(out / "ensemble.json"),
            "model_path": str(out / "ensemble.joblib"),
            "method": result.method,
            "mode": result.config.mode,
            "task": result.task.value,
            "metric": result.metric,
            "direction": result.direction,
            "candidates": list(result.candidates),
            "mean": result.mean,
            "std": result.std,
            "n_folds": len(result.folds),
            "candidate_scores": result.to_dict()["candidate_scores"],
            "splits": splits_info,
            "validation_status": result.validation_status,
            "validation_limitation": result.validation_limitation,
            "data": ctx.data_files,
        }
    )


def ensemble_saved_stage(
    ctx: RunContext,
    model_references: Sequence[str | Path],
    output_dir: str | Path,
    *,
    method: str = "stacking",
    stack_cv: int = 5,
    n_jobs: int | None = None,
    meta_estimator: str | None = None,
    meta_estimator_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create an ensemble by reusing saved fitted-model references without refitting members."""
    out = Path(output_dir)
    _new_dir(out, "ensemble", EnsembleError)
    config = EnsembleConfig(
        method=method,
        seed=ctx.config.baseline_config.seed,
        n_jobs=ctx.config.baseline_config.n_jobs if n_jobs is None else n_jobs,
        stack_cv=stack_cv,
        meta_estimator=meta_estimator,
        meta_estimator_params=meta_estimator_params or {},
    )
    result = run_saved_ensemble(
        ctx.bundle.train,
        ctx.config.target,
        list(model_references),
        ctx.resolved,
        config,
        directory=out,
        tracking_uri=_tracking_uri(ctx),
    )
    payload: dict[str, Any] = {
        "status": "created",
        "directory": str(out),
        "path": str(out / "ensemble.json"),
        "model_path": str(out / "ensemble.joblib"),
        "method": result.method,
        "task": result.task.value,
        "metric": result.metric,
        "direction": result.direction,
        "candidates": list(result.candidates),
        "mean": result.mean,
        "std": result.std,
        "n_folds": len(result.folds),
        "validation_status": result.validation_status,
        "validation_limitation": result.validation_limitation,
        "model_references": list(model_references),
        "data": ctx.data_files,
    }
    return _plain(payload)


def model_report(
    ctx: RunContext,
    sources: Sequence[str | Path],
    output_dir: str | Path,
    *,
    filename: str = "model_report.pdf",
    top_n: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    cfg = ctx.config.model_report_config
    if top_n is not None:
        cfg = replace(cfg, top_n=top_n)
    result = write_model_report(
        [str(s) for s in sources], output_dir, filename=filename, config=cfg, overwrite=overwrite,
        target=ctx.bundle.train[ctx.config.target],
    )
    return _plain(
        {
            "status": "created",
            "path": str(result.path),
            "summary_path": str(result.summary_path),
            "pages": result.pages,
            "task": result.task,
            "metric": result.metric,
            "direction": result.direction,
            "candidates": [
                {"rank": c.rank, "label": c.label, "mean": c.mean, "std": c.std, "n_folds": c.n_folds}
                for c in result.candidates
            ],
            "excluded": [e.to_dict() for e in result.excluded],
            "warnings": list(result.warnings),
            "charts_skipped": dict(result.charts_skipped),
        }
    )


def refit_stage(
    ctx: RunContext,
    source: str | Path,
    output_dir: str | Path,
    *,
    label: str | None = None,
    rank: int | None = None,
    overwrite: bool = False,
    track: bool = False,
    progress: ProgressCallback | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    out = Path(output_dir)
    existing = [n for n in (PREDICTIONS_FILE, REFIT_FILE, MODEL_FILE) if (out / n).exists()]
    if existing and not overwrite:
        raise RefitError(f"{out} already contains {existing}; pass --overwrite or choose another directory")
    tracking = _tracking(ctx, track)
    result = _tracked(
        out,
        lambda: refit_and_predict(
            str(source), ctx.bundle.train, ctx.bundle.test, target=ctx.config.target, id_columns=_ids(ctx),
            resolved_task=ctx.resolved, label=label, rank=rank, config=ctx.config.refit_config, directory=out,
            overwrite=overwrite, tracking=tracking, progress=progress, verbose=verbose,
        ),
    )
    sel = result.refit.selection
    return _plain(
        {
            "status": "created",
            "directory": str(out),
            "path": str(out / REFIT_FILE),
            "label": sel.label,
            "source_kind": sel.source_kind,
            "source_path": sel.source_path,
            "estimator": sel.estimator,
            "validation_mean": sel.validation_mean,
            "validation_std": sel.validation_std,
            "n_train_rows": result.refit.n_train,
            "n_test_rows": result.predictions.n_rows,
            "prediction_kind": result.predictions.prediction_kind,
            "data": ctx.data_files,
            "notes": [*sel.notes, *result.predictions.notes],
            "tracking": _tracking_info(result.tracking, tracking is not None),
        }
    )


def submit(
    ctx: RunContext, refit_dir: str | Path, output_dir: str | Path, *, overwrite: bool = False, track: bool = False
) -> dict[str, Any]:
    out = Path(output_dir)
    existing = [n for n in ("submission.csv", "submission.json") if (out / n).exists()]
    if existing and not overwrite:
        raise SubmissionError(f"{out} already contains {existing}; pass --overwrite or choose another directory")
    tracking = _tracking(ctx, track)
    loaded = load_refit(refit_dir)
    sample_path = ctx.config.sample_submission_path
    result = _tracked(
        out,
        lambda: build_submission(
            loaded, ctx.bundle.test, sample=ctx.bundle.sample_submission, id_columns=_ids(ctx),
            target=ctx.config.target, config=ctx.config.submission_config,
            sample_name=sample_path.name if sample_path is not None else None, directory=out, overwrite=overwrite,
            tracking=tracking,
        ),
    )
    return _plain(
        {
            "status": "created",
            "directory": str(out),
            "path": str(out / "submission.csv"),
            "n_rows": result.n_rows,
            "columns": list(result.columns),
            "schema_source": result.schema_source,
            "id_check": result.id_check,
            "source": result.source,
            "data": ctx.data_files,
            "notes": list(result.notes),
            "tracking": _tracking_info(result.tracking, tracking is not None),
        }
    )


# ---- end to end ----


def run(
    ctx: RunContext,
    output_dir: str | Path,
    *,
    estimator: str | None = None,
    families: Sequence[str] | None = None,
    n_trials: int | None = None,
    timeout_seconds: float | None = None,
    skip_tuning: bool = False,
    skip_reports: bool = False,
    track: bool = False,
    log: Log | None = None,
    progress: ProgressCallback | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Compose every stage into subfolders of `output_dir`; stops at the first failing stage and keeps earlier files."""
    root = Path(output_dir)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise RunError(f"run directory {root} already exists and is not empty; choose a new path")
    if estimator is not None and families is not None:
        raise RunError("use either --estimator (one estimator) or --families (several), not both")
    # Fail on missing extras, budgets, and tracking before any file is written.
    if not skip_reports:
        _require_matplotlib()
    tuning_cfg = None
    if not skip_tuning:
        _require_optuna()
        tuning_cfg = _tuning_config(ctx, estimator, families, n_trials, timeout_seconds)
        for name in ([estimator] if estimator else list(families or tuning_cfg.families or [])):
            get_estimator_spec(name).require()
    get_estimator_spec(estimator or ctx.config.baseline_config.estimator).require()
    _tracking(ctx, track)

    stages: dict[str, Any] = {}
    skipped: list[str] = []
    paths = {name: root / name for name in ("splits", "eda", "baseline", "tuning", "reports", "refit", "submission")}

    progress_started = time.perf_counter()
    if progress is not None:
        emit_progress(progress, "run", "stage.start", progress_started, stage_name="run")

    def step(name: str, fn: Callable[[], Any]) -> Any:
        if progress is not None:
            emit_progress(progress, "run", "stage.start", progress_started, stage_name=name)
        if log is not None:
            log(f"[run] {name}")
        try:
            stages[name] = fn()
        except (DataContractError, OSError) as exc:
            if progress is not None:
                emit_progress(progress, "run", "stage.failed", progress_started, stage_name=name, error=str(exc))
            done = ", ".join(stages) or "none"
            raise RunError(
                f"stage '{name}' failed after completing: {done}; completed artifacts are kept in {root}: {exc}"
            ) from None
        if progress is not None:
            emit_progress(progress, "run", "stage.complete", progress_started, stage_name=name)
        return stages[name]

    root.mkdir(parents=True, exist_ok=True)
    if progress is not None:
        emit_progress(progress, "run", "stage.start", progress_started, stage_name="validate")
    stages["validate"] = validate(ctx)
    if progress is not None:
        emit_progress(progress, "run", "stage.complete", progress_started, stage_name="validate")
    splits_file = paths["splits"] / "splits.json"
    step("split", lambda: split(ctx, paths["splits"]))
    step("eda", lambda: eda(ctx, paths["eda"]))
    if skip_reports:
        skipped += ["eda_report", "model_report"]
    else:
        step("eda_report", lambda: eda_report(ctx.config, paths["eda"] / "eda.json", paths["reports"]))
    step(
        "baseline",
        lambda: baseline(
            ctx, paths["baseline"], splits_path=splits_file, estimator=estimator, track=track,
            progress=progress, verbose=verbose,
        ),
    )
    sources = [paths["baseline"]]
    if skip_tuning:
        skipped.append("tuning")
    else:
        step(
            "tuning",
            lambda: tune_stage(
                ctx, paths["tuning"], splits_path=splits_file, estimator=estimator, families=families,
                n_trials=tuning_cfg.n_trials, timeout_seconds=tuning_cfg.timeout_seconds, track=track, log=log,
                progress=progress, verbose=verbose,
            ),
        )
        sources.append(paths["tuning"])
    if not skip_reports:
        step("model_report", lambda: model_report(ctx, sources, paths["reports"]))
    refit_source = sources[-1]
    step(
        "refit",
        lambda: refit_stage(ctx, refit_source, paths["refit"], rank=1, track=track, progress=progress, verbose=verbose),
    )
    step("submission", lambda: submit(ctx, paths["refit"], paths["submission"], track=track))
    refit_info = stages["refit"]
    result = {
        "status": "completed",
        "output_dir": str(root),
        "task": ctx.resolved.task.value,
        "metric": ctx.resolved.metric.name,
        "selected": {
            "source": refit_source.name,
            "label": refit_info["label"],
            "validation_mean": refit_info["validation_mean"],
        },
        "submission": stages["submission"]["path"],
        "skipped": skipped,
        "stages": stages,
    }
    if progress is not None:
        emit_progress(progress, "run", "stage.complete", progress_started, stage_name="run")
    return result
