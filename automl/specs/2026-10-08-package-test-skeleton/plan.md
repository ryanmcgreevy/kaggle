# Plan: Package and Test Skeleton

This feature implements Phase 0 of `../roadmap.md`. Keep it intentionally small: establish an installable package, a test runner, and a help-only CLI. Do not implement data or machine-learning behavior in this phase.

## Task Groups

1. **Project metadata and package layout**
   - Add `pyproject.toml` using setuptools as the build backend.
   - Define the distribution as `automl`, require Python 3.12 or newer, and use a `src/automl/` source layout.
   - Declare no runtime ML dependencies. Add pytest as a development/test extra.
   - Ensure package discovery includes the `automl` Python package and not the repository's specs or future data/artifacts.

2. **Importable package entry points**
   - Add the minimal package initializer and module entry point.
   - Implement a small argparse CLI with the `automl` command and `python -m automl` support.
   - Provide standard help output and exit successfully when invoked with `--help`; do not add workflow subcommands yet.

3. **Skeleton tests**
   - Configure pytest through project metadata or the minimal repository-appropriate configuration.
   - Add tests for importing the package and for successful CLI help output through both supported entry points.
   - Keep tests synthetic and independent of Kaggle data, network access, cloud credentials, and optional ML libraries.

4. **Installation and usage notes**
   - Document the editable install command and the commands for running tests and CLI help.
   - State clearly that this phase is only a packaging/CLI skeleton; later roadmap phases add actual capabilities.

5. **Review and merge gate**
   - Run the checks in `validation.md` from the `automl/phase-0-package-test-skeleton` branch.
   - Review the final diff for scope, package discovery, and accidental runtime dependency additions before merge.