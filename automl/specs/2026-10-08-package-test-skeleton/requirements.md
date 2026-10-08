# Requirements: Package and Test Skeleton

## Scope

Establish the minimum installable and testable foundation for the AutoML project, corresponding to Phase 0 in `../roadmap.md`. The result is a Python package named `automl`, a help-only CLI, and a small passing pytest suite.

## Functional Requirements

- **FR-1: Installable package.** The project can be installed from the `automl/` project root using standard pip/setuptools tooling.
- **FR-2: Importable package.** After installation, `import automl` succeeds and identifies the package without importing or requiring ML libraries.
- **FR-3: CLI command.** Installation exposes an `automl` command. Running `automl --help` displays usage and exits with status 0.
- **FR-4: Module invocation.** `python -m automl --help` displays equivalent help and exits with status 0.
- **FR-5: Test suite.** The configured pytest suite runs locally and covers package import and both CLI help paths.
- **FR-6: Clear boundaries.** Invoking the CLI without a supported operation must not imply that data inspection, model training, tuning, or submission generation is implemented.

## Non-Functional Requirements

- Support Python 3.12 or newer for this initial skeleton.
- Use a `src/automl/` source layout and setuptools configured in `pyproject.toml`.
- Add no runtime dependencies in this phase. Pytest may be declared as a development/test optional dependency.
- Keep the install and tests independent of Kaggle datasets, credentials, network access, cloud services, GPUs, and optional model/tracking packages.
- Keep package metadata and CLI behavior minimal, explicit, and suitable for extension in later roadmap phases.

## Decisions

- Distribution name: `automl`.
- Import package: `automl`.
- Installed command: `automl`.
- Build backend: setuptools, configured through `pyproject.toml`.
- Source layout: `src/automl/`.
- CLI implementation: Python standard-library `argparse`.
- Test framework: pytest.
- Runtime dependencies: none for this phase; do not add pandas, NumPy, scikit-learn, Optuna, MLflow, boosting libraries, or PyTorch until a later feature requires them.

## Context and Constraints

- The parent constitution is in `../mission.md` and `../tech-stack.md`; this feature must preserve local-first operation, modularity, and human/agent usability.
- The next roadmap phase will define a data contract. Do not prematurely add CSV loaders, configuration schemas, or competition-specific behavior here.
- The command name and package name intentionally match. If a packaging conflict is discovered during implementation, stop and bring the evidence back for a decision instead of silently renaming the public interface.
- Project documentation may be added or updated only as needed to explain installation, CLI help, and test commands.

## Out of Scope

- Data discovery, CSV loading, EDA, preprocessing, model training, tuning, tracking, PDF reports, and submission generation.
- A full command-line workflow, configuration file format, notebooks, or generated competition project.
- Runtime ML dependencies, optional dependency groups beyond the minimal test extra, and dependency lockfiles.
- Cloud execution, Kaggle API access, deployment, and automated submission.