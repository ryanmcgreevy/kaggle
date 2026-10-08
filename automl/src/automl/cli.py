"""Command-line interface for the AutoML toolkit."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from automl.config import load_config
from automl.eda import save_eda, summarize_bundle
from automl.errors import DataContractError
from automl.loader import load_data
from automl.report import write_eda_report
from automl.task import resolve_task_metric
from automl.validation import make_splits, save_splits


def _load_run(config_path: str):
    config = load_config(config_path)
    bundle = load_data(config)
    resolved = resolve_task_metric(bundle)
    return config, bundle, resolved


def _write_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _validate(args: argparse.Namespace) -> int:
    config, bundle, resolved = _load_run(args.config)
    _write_json(
        {
            "status": "valid",
            "config": str(Path(args.config)),
            "target": config.target,
            "task": resolved.task.value,
            "metric": resolved.metric.name,
            "train_rows": len(bundle.train),
            "test_rows": len(bundle.test),
            "sample_submission_rows": (
                len(bundle.sample_submission) if bundle.sample_submission is not None else None
            ),
        }
    )
    return 0


def _split(args: argparse.Namespace) -> int:
    config, bundle, resolved = _load_run(args.config)
    output_dir = Path(args.output_dir)
    output_path = output_dir / "splits.json"
    result = make_splits(bundle.train[config.target], resolved, config.validation_config)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_splits(result, output_path)
    _write_json(
        {
            "status": "created",
            "path": str(output_path),
            "strategy": result.config.strategy,
            "task": result.task.value,
            "n_rows": result.n_rows,
            "n_splits": len(result.splits),
        }
    )
    return 0


def _eda(args: argparse.Namespace) -> int:
    _, bundle, resolved = _load_run(args.config)
    output_dir = Path(args.output_dir)
    output_path = output_dir / "eda.json"
    result = summarize_bundle(bundle, task=resolved)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_eda(result, output_path)
    _write_json(
        {
            "status": "created",
            "path": str(output_path),
            "schema_version": result.schema_version,
            "n_rows": result.dataset.n_rows,
            "n_columns": result.dataset.n_columns,
        }
    )
    return 0


def _report(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    result = write_eda_report(
        args.eda,
        args.output_dir,
        filename=args.filename,
        config=config.report_config,
        overwrite=args.overwrite,
    )
    _write_json({"status": "created", **result.to_dict()})
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser without starting a workflow."""
    parser = argparse.ArgumentParser(
        prog="automl",
        description="Local-first tools for tabular Kaggle competitions.",
    )
    commands = parser.add_subparsers(dest="command")

    validate = commands.add_parser("validate", help="validate configured data and task choices")
    validate.add_argument("config", help="path to the run TOML file")
    validate.set_defaults(handler=_validate)

    split = commands.add_parser("split", help="create the configured validation split artifact")
    split.add_argument("config", help="path to the run TOML file")
    split.add_argument("--output-dir", required=True, help="directory for splits.json")
    split.set_defaults(handler=_split)

    eda = commands.add_parser("eda", help="create the EDA JSON artifact")
    eda.add_argument("config", help="path to the run TOML file")
    eda.add_argument("--output-dir", required=True, help="directory for eda.json")
    eda.set_defaults(handler=_eda)

    report = commands.add_parser("report", help="create a PDF report and findings from EDA JSON")
    report.add_argument("config", help="path to the run TOML file")
    report.add_argument("eda", help="path to a saved EDA JSON artifact")
    report.add_argument("--output-dir", required=True, help="directory for the PDF and findings JSON")
    report.add_argument("--filename", default="eda_report.pdf", help="PDF filename (default: eda_report.pdf)")
    report.add_argument("--overwrite", action="store_true", help="replace existing report files")
    report.set_defaults(handler=_report)

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