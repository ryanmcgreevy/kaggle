# Requirements: Input Data Contract (Phase 1)

## Scope

**Deliverable (from roadmap):** A documented run configuration and loader for train CSV, test CSV, optional sample submission, target, and identifier columns.

In scope:

- TOML run configuration with a version field, plus programmatic overrides.
- Loader returning train, test, and optional sample submission DataFrames with the effective configuration.
- Validation with actionable errors for missing files, duplicate columns, train/test column mismatch, and invalid target/ID settings.
- Synthetic-fixture tests and documentation.

Out of scope:

- CLI command (deferred to Phase 16; the Python API is the interface for now).
- Task/metric selection (Phase 2), splits (Phase 3), EDA (Phase 4), dtype inference or cleaning.
- Auto-suggesting target or ID columns.
- Kaggle credentials, downloads, or network access.

## Decisions

| Topic | Decision |
|---|---|
| Config format | TOML (stdlib `tomllib`) with explicit `version`; Python overrides take precedence over file values. |
| Identifier columns | Optional; one or more; explicit only (no auto-detection). Must exist in train and test and must not be the target. |
| Target | Required and must exist in train; must be absent from test. |
| Column mismatch | Error listing missing and extra columns (no silent intersection). |
| Interface | Python API only in this phase. |
| Dependencies | `pandas` added to core; no optional ML packages required. |
| Data handling | Loader does not mutate or coerce values; relative paths resolve against the config file directory. |
| Sample submission | Optional; when present, checked for ID column presence and row count against test. Its non-ID columns are treated as prediction columns and are not compared to train/test features. |

## Context

- `specs/mission.md`: conservative inference, inspectable outputs, local and private by default, validated outputs.
- `specs/tech-stack.md`: pandas/NumPy for tabular data, TOML configuration, pytest with small synthetic fixtures, actionable errors identifying the file, column, or setting.
- Phase 0 left a runnable package (`src/automl`, `cli.py`, `tests/`) with no runtime dependencies; this phase adds the first.

## Assumptions

- Files are comma-separated CSV with a header row and UTF-8 encoding; other CSV options are not configurable yet.
- If no ID column is configured, row-count-only checks apply to the sample submission.
- Seed is stored in the config for later phases but unused here.
