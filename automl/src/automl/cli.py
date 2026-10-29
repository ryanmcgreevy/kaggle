"""Command-line interface for the AutoML toolkit."""

import argparse
import contextlib
import json
import os
import sys
from collections.abc import Sequence
from typing import Any

from automl import workflow
from automl.config import load_config
from automl.errors import DataContractError, EnsembleError
from automl.progress import StderrProgress


def _write_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _names(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [name.strip() for name in value.split(",") if name.strip()]


def _progress(args: argparse.Namespace) -> StderrProgress | None:
    return None if args.quiet else StderrProgress()


@contextlib.contextmanager
def _native_output_to_stderr(enabled: bool):
    if not enabled:
        yield
        return
    stdout = sys.stdout
    try:
        stdout_fd = stdout.fileno()
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        with contextlib.redirect_stdout(sys.stderr):
            yield
        return
    stdout.flush()
    saved_stdout = os.dup(stdout_fd)
    try:
        os.dup2(stderr_fd, stdout_fd)
        with contextlib.redirect_stdout(sys.stderr):
            yield
    finally:
        sys.stdout.flush()
        os.dup2(saved_stdout, stdout_fd)
        os.close(saved_stdout)


def _validate(args: argparse.Namespace) -> int:
    _write_json(workflow.validate(workflow.load_run(args.config)))
    return 0


def _split(args: argparse.Namespace) -> int:
    _write_json(workflow.split(workflow.load_run(args.config), args.output_dir))
    return 0


def _eda(args: argparse.Namespace) -> int:
    _write_json(workflow.eda(workflow.load_run(args.config), args.output_dir))
    return 0


def _report(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    _write_json(
        workflow.eda_report(config, args.eda, args.output_dir, filename=args.filename, overwrite=args.overwrite)
    )
    return 0


def _baseline(args: argparse.Namespace) -> int:
    ctx = workflow.load_run(args.config)
    with _native_output_to_stderr(args.verbose):
        payload = workflow.baseline(
            ctx,
            args.output_dir,
            splits_path=args.splits,
            estimator=args.estimator,
            track=args.track,
            progress=_progress(args),
            verbose=args.verbose,
        )
    _write_json(payload)
    return 0


def _tune(args: argparse.Namespace) -> int:
    ctx = workflow.load_run(args.config)
    with _native_output_to_stderr(args.verbose):
        payload = workflow.tune_stage(
            ctx,
            args.output_dir,
            splits_path=args.splits,
            estimator=args.estimator,
            families=_names(args.families),
            n_trials=args.n_trials,
            timeout_seconds=args.timeout_seconds,
            resume=args.resume,
            track=args.track,
            progress=_progress(args),
            verbose=args.verbose,
        )
    _write_json(payload)
    return 0


def _model_report(args: argparse.Namespace) -> int:
    ctx = workflow.load_run(args.config)
    _write_json(
        workflow.model_report(
            ctx, args.sources, args.output_dir, filename=args.filename, top_n=args.top_n, overwrite=args.overwrite
        )
    )
    return 0


def _ensemble(args: argparse.Namespace) -> int:
    ctx = workflow.load_run(args.config)
    try:
        meta_params = json.loads(args.meta_estimator_params)
    except json.JSONDecodeError as exc:
        raise EnsembleError(f"--meta-estimator-params must be a JSON object: {exc}") from None
    if not isinstance(meta_params, dict):
        raise EnsembleError("--meta-estimator-params must be a JSON object")
    with _native_output_to_stderr(args.verbose):
        payload = workflow.ensemble_stage(
            ctx,
            args.sources,
            args.output_dir,
            splits_path=args.splits,
            method=args.method,
            labels=args.label,
            stack_cv=args.stack_cv,
            n_jobs=args.n_jobs,
            meta_estimator=args.meta_estimator,
            meta_estimator_params=meta_params,
            progress=_progress(args),
            verbose=args.verbose,
        )
    _write_json(payload)
    return 0


def _ensemble_saved(args: argparse.Namespace) -> int:
    ctx = workflow.load_run(args.config)
    try:
        meta_params = json.loads(args.meta_estimator_params)
    except json.JSONDecodeError as exc:
        raise EnsembleError(f"--meta-estimator-params must be a JSON object: {exc}") from None
    if not isinstance(meta_params, dict):
        raise EnsembleError("--meta-estimator-params must be a JSON object")
    _write_json(
        workflow.ensemble_saved_stage(
            ctx,
            args.model_references,
            args.output_dir,
            method=args.method,
            stack_cv=args.stack_cv,
            n_jobs=args.n_jobs,
            meta_estimator=args.meta_estimator,
            meta_estimator_params=meta_params,
        )
    )
    return 0


def _refit(args: argparse.Namespace) -> int:
    ctx = workflow.load_run(args.config)
    with _native_output_to_stderr(args.verbose):
        payload = workflow.refit_stage(
            ctx,
            args.source,
            args.output_dir,
            label=args.label,
            rank=args.rank,
            overwrite=args.overwrite,
            track=args.track,
            progress=_progress(args),
            verbose=args.verbose,
        )
    _write_json(payload)
    return 0


def _submit(args: argparse.Namespace) -> int:
    ctx = workflow.load_run(args.config)
    _write_json(workflow.submit(ctx, args.refit_dir, args.output_dir, overwrite=args.overwrite, track=args.track))
    return 0


def _run(args: argparse.Namespace) -> int:
    ctx = workflow.load_run(args.config)
    with _native_output_to_stderr(args.verbose):
        payload = workflow.run(
            ctx,
            args.output_dir,
            estimator=args.estimator,
            families=_names(args.families),
            n_trials=args.n_trials,
            timeout_seconds=args.timeout_seconds,
            skip_tuning=args.skip_tuning,
            skip_reports=args.skip_reports,
            track=args.track,
            progress=_progress(args),
            verbose=args.verbose,
        )
    _write_json(payload)
    return 0


def _add_config(command: argparse.ArgumentParser) -> None:
    command.add_argument("config", help="path to the run TOML file")


def _add_budget(command: argparse.ArgumentParser) -> None:
    command.add_argument("--n-trials", type=int, help="trial budget (overrides [tuning])")
    command.add_argument("--timeout-seconds", type=float, help="time budget in seconds (overrides [tuning])")


def _add_track(command: argparse.ArgumentParser) -> None:
    command.add_argument("--track", action="store_true", help="log to MLflow using the [tracking] table")


def _add_progress(command: argparse.ArgumentParser) -> None:
    modes = command.add_mutually_exclusive_group()
    modes.add_argument(
        "--verbose",
        action="store_true",
        help="show progress and enable supported estimator-native training output on stderr",
    )
    modes.add_argument("--quiet", action="store_true", help="suppress progress output but still show errors")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser without starting a workflow."""
    parser = argparse.ArgumentParser(
        prog="automl",
        description="Local-first tools for tabular Kaggle competitions.",
    )
    commands = parser.add_subparsers(dest="command")

    validate = commands.add_parser("validate", help="validate configured data and task choices")
    _add_config(validate)
    validate.set_defaults(handler=_validate)

    split = commands.add_parser("split", help="create the configured validation split artifact")
    _add_config(split)
    split.add_argument("--output-dir", required=True, help="directory for splits.json")
    split.set_defaults(handler=_split)

    eda = commands.add_parser("eda", help="create the EDA JSON artifact")
    _add_config(eda)
    eda.add_argument("--output-dir", required=True, help="directory for eda.json")
    eda.set_defaults(handler=_eda)

    report = commands.add_parser("report", help="create a PDF report and findings from EDA JSON")
    _add_config(report)
    report.add_argument("eda", help="path to a saved EDA JSON artifact")
    report.add_argument("--output-dir", required=True, help="directory for the PDF and findings JSON")
    report.add_argument("--filename", default="eda_report.pdf", help="PDF filename (default: eda_report.pdf)")
    report.add_argument("--overwrite", action="store_true", help="replace existing report files")
    report.set_defaults(handler=_report)

    baseline = commands.add_parser("baseline", help="cross-validate one estimator and save the result directory")
    _add_config(baseline)
    baseline.add_argument("--output-dir", required=True, help="new directory for the baseline result")
    baseline.add_argument("--splits", help="saved splits.json to reuse (default: build from [validation])")
    baseline.add_argument("--estimator", help="estimator to use instead of the configured one")
    _add_track(baseline)
    _add_progress(baseline)
    baseline.set_defaults(handler=_baseline)

    tune = commands.add_parser("tune", help="tune several model families (default) or one estimator with Optuna")
    _add_config(tune)
    tune.add_argument("--output-dir", required=True, help="new directory for the tuning result (existing one with --resume)")
    tune.add_argument("--splits", help="saved splits.json to reuse (default: build from [validation])")
    tune.add_argument("--estimator", help="tune this one estimator instead of several families")
    tune.add_argument("--families", help="comma-separated families to tune (default: [tuning] or all installed)")
    _add_budget(tune)
    tune.add_argument("--resume", action="store_true", help="add trials to the saved study in --output-dir")
    _add_track(tune)
    _add_progress(tune)
    tune.set_defaults(handler=_tune)

    model_report = commands.add_parser("model-report", help="create the model comparison PDF from saved results")
    _add_config(model_report)
    model_report.add_argument("sources", nargs="+", help="baseline, tuning, or tuning-families result directories")
    model_report.add_argument("--output-dir", required=True, help="directory for the PDF and summary JSON")
    model_report.add_argument("--filename", default="model_report.pdf", help="PDF filename (default: model_report.pdf)")
    model_report.add_argument("--top-n", type=int, help="number of best models to compare (overrides [model_report])")
    model_report.add_argument("--overwrite", action="store_true", help="replace existing report files")
    model_report.set_defaults(handler=_model_report)

    ensemble = commands.add_parser(
        "ensemble", help="evaluate a voting or stacking ensemble from saved model candidates"
    )
    _add_config(ensemble)
    ensemble.add_argument("sources", nargs="+", help="baseline, tuning, or tuning-families result directories")
    ensemble.add_argument("--output-dir", required=True, help="new directory for ensemble.json and ensemble.joblib")
    ensemble.add_argument("--method", choices=("voting", "stacking"), default="stacking")
    ensemble.add_argument("--label", action="append", help="candidate label to include; repeat to select several")
    ensemble.add_argument("--splits", help="saved splits.json to reuse (default: build from [validation])")
    ensemble.add_argument("--stack-cv", type=int, default=5, help="inner stacking folds (default: 5)")
    ensemble.add_argument(
        "--n-jobs",
        type=int,
        help="parallel jobs within stacking/voting; -1 uses all available cores "
        "(default: [baseline].n_jobs, or 1); candidate-model threading is unchanged",
    )
    ensemble.add_argument(
        "--meta-estimator",
        help="stacking meta-estimator: logistic_regression, decision_tree_classifier, random_forest_classifier, "
        "ridge_cv, ridge, decision_tree_regressor, or random_forest_regressor",
    )
    ensemble.add_argument(
        "--meta-estimator-params",
        default="{}",
        help="JSON object of parameters for the selected meta-estimator (default: {})",
    )
    _add_progress(ensemble)
    ensemble.set_defaults(handler=_ensemble)

    ensemble_saved = commands.add_parser(
        "ensemble-saved",
        help="reuse saved fitted-model references; hard voting fits no models and stacking fits only its meta-model",
    )
    _add_config(ensemble_saved)
    ensemble_saved.add_argument(
        "model_references",
        nargs="+",
        help="at least two trusted fitted-model directories or MLflow model URIs",
    )
    ensemble_saved.add_argument("--output-dir", required=True, help="new directory for ensemble.json and ensemble.joblib")
    ensemble_saved.add_argument("--method", choices=("voting", "stacking"), default="stacking")
    ensemble_saved.add_argument("--stack-cv", type=int, default=5, help="recorded compatibility setting")
    ensemble_saved.add_argument("--n-jobs", type=int, help="parallel jobs setting recorded in the ensemble config")
    ensemble_saved.add_argument("--meta-estimator", help="stacking meta-estimator (see automl ensemble --help)")
    ensemble_saved.add_argument(
        "--meta-estimator-params",
        default="{}",
        help="JSON object of parameters for the selected meta-estimator (default: {})",
    )
    ensemble_saved.set_defaults(handler=_ensemble_saved)

    refit = commands.add_parser("refit", help="refit the selected pipeline on all train rows and predict test rows")
    _add_config(refit)
    refit.add_argument("source", help="baseline, tuning, or tuning-families result directory")
    refit.add_argument("--output-dir", required=True, help="directory for predictions.csv, refit.json, and the model")
    choice = refit.add_mutually_exclusive_group()
    choice.add_argument("--label", help="candidate label to refit (see model-report)")
    choice.add_argument("--rank", type=int, help="candidate rank to refit (default: 1)")
    refit.add_argument("--overwrite", action="store_true", help="replace existing refit files")
    _add_track(refit)
    _add_progress(refit)
    refit.set_defaults(handler=_refit)

    submit = commands.add_parser("submit", help="validate predictions and write submission.csv (never uploads)")
    _add_config(submit)
    submit.add_argument("refit_dir", help="directory written by the refit command")
    submit.add_argument("--output-dir", required=True, help="directory for submission.csv and submission.json")
    submit.add_argument("--overwrite", action="store_true", help="replace existing submission files")
    _add_track(submit)
    submit.set_defaults(handler=_submit)

    run = commands.add_parser("run", help="run every stage from CSV inputs to a validated submission")
    _add_config(run)
    run.add_argument("--output-dir", required=True, help="new or empty directory for all stage outputs")
    run.add_argument("--estimator", help="use one estimator for the baseline and tuning")
    run.add_argument("--families", help="comma-separated families to tune")
    _add_budget(run)
    run.add_argument("--skip-tuning", action="store_true", help="refit the baseline instead of tuned models")
    run.add_argument("--skip-reports", action="store_true", help="do not write the EDA and model PDFs")
    _add_track(run)
    _add_progress(run)
    run.set_defaults(handler=_run)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a workflow command or print help."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not arguments:
        parser.print_help()
        return 0
    args = parser.parse_args(arguments)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        return args.handler(args)
    except (DataContractError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
