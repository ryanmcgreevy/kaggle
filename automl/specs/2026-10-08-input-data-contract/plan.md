# Plan: Input Data Contract (Phase 1)

## 1. Dependencies and module layout

1. Add `pandas` (and `numpy` transitively) to core `dependencies` in `pyproject.toml`; keep `requires-python >=3.12`.
2. Create `src/automl/data/` (or `data_contract.py`) with separate config, errors, and loader modules.
3. Confirm `import automl` and `automl --help` still work.

## 2. Error types

1. Define `DataContractError` (base) carrying the file, column(s), and setting involved.
2. Add specific subclasses or error codes: missing file, unreadable/empty CSV, duplicate columns, column mismatch, invalid target, invalid ID.

## 3. Run configuration

1. Define a frozen `DataConfig` dataclass: `train_path`, `test_path`, `sample_submission_path` (optional), `target`, `id_columns` (tuple, optional), `seed`.
2. Implement `load_config(path)` using stdlib `tomllib`, with a `version` field and a `[data]` table; reject unknown keys and unsupported versions.
3. Support Python-level overrides (merge explicit values over file values). Resolve relative paths against the config file directory.
4. Document the schema with an example `run.toml` in the README.

## 4. Loader and validation

1. Implement `load_data(config) -> DataBundle` (train, test, optional sample submission DataFrames plus the effective config).
2. Read CSVs with pandas; do not mutate or coerce values beyond parsing. Detect duplicate column headers before pandas renames them (read header row separately).
3. Validate:
   - files exist and are non-empty;
   - no duplicate columns in any file;
   - target exists in train and is absent from test;
   - each ID column exists in train and test, IDs are not the target, no repeated ID names;
   - train feature columns (excluding target) equal test columns (set mismatch error lists missing and extra);
   - sample submission, when supplied, has columns that include the test ID column(s) and, if IDs are configured, matching ID values/row count.
4. Collect all detectable problems in one pass where practical, then raise a single actionable error.

## 5. Tests

1. Add synthetic CSV fixtures via `tmp_path` factories in `tests/conftest.py`.
2. Cover: valid files, valid with and without sample submission, missing files, empty file, duplicate columns, train/test mismatch (missing and extra), target present in test, target missing in train, invalid/missing ID, ID equals target, bad TOML, unknown config key, relative path resolution, overrides.
3. Assert error messages name the file, column, or setting.

## 6. Public API export

1. Export `DataConfig`, `load_config`, `load_data`, `DataBundle`, and `DataContractError` from `automl`.

## 7. Documentation and roadmap

1. Update `automl/README.md` with the config schema and a Python usage example.
2. Mark Phase 1 complete in `specs/roadmap.md` with a verification note once validation passes.
