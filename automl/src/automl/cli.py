"""Command-line interface for the AutoML toolkit."""

import argparse
import sys
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser without starting a workflow."""
    return argparse.ArgumentParser(
        prog="automl",
        description="Local-first tools for tabular Kaggle competitions.",
        epilog="No workflow commands are available yet.",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Print CLI help until workflow commands are implemented."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    parser.parse_args(arguments)
    if not arguments:
        parser.print_help()
    return 0