# Validation: Submission Validation and CSV (Phase 15)

The branch is ready to merge when every item below is true. Run from `automl/` in the project-local `.venv`.

## Automated checks

1. `pip check` passes.
2. `automl --help` and `python -m automl --help` still work (no CLI change).
3. The full suite passes with the `reports`, `tuning`, `boosting`, and `mlflow` extras installed (the existing GPU test may skip).
4. The suite passes in a clean environment with only the `test` extra; optional-library tests skip and none error.
5. Phase 1 to 14 tests pass unchanged.
6. `import automl` succeeds with Optuna, Matplotlib, MLflow, and the boosting libraries blocked.

## Acceptance behavior (from the roadmap)

1. Row order: the submission rows equal the original `test.csv` row order; a refit frame out of test order is rejected, and a sample in a different order is accepted without changing the output order.
2. Schema: submission column names and order equal the sample submission exactly; missing, extra, renamed, and unmappable columns are rejected with the column named.
3. Identifiers: duplicate, null, missing, and extra IDs are rejected; matching ID sets alone do not bypass the row-by-row alignment check against test.
4. Row count: a count different from the test rows is rejected.
5. Values: nulls, NaN, inf, out-of-range probabilities, and lossy dtype casts are rejected and no file is written.
6. Valid files are written: `submission.csv` (header, no index) and `submission.json` round trip through `load_submission`, and the CSV re-reads to the same values.
7. Without a sample submission, the output is ID columns plus the target column and records the unverified-schema note.
8. An existing output is never overwritten unless `overwrite=True`; a forced failure leaves no partial files.
9. No Kaggle API call or network access occurs (verified by a test blocking sockets).

## Contract and safety checks

1. Test frame, sample submission, and saved refit files are byte-identical after a run.
2. `submission.json` is valid JSON with `allow_nan=False`, a schema version, the column mapping, notes, and the statement that it is not a leaderboard claim.
3. `[submission]` TOML parsing, unknown-key rejection, and Python override precedence work; unknown or wrong-typed values give actionable errors.
4. MLflow logging works against a local SQLite store, the artifacts are queryable, and an unreachable store leaves local files intact. MLflow absent raises the install hint before any file is written.
5. No raw training or test data is logged to MLflow.

## Documentation and process

1. `automl/README.md` documents invocation for people and agents, inputs, outputs, schemas, defaults, mapping and validity rules, and limits.
2. An example in the README runs end to end on synthetic CSVs from refit result to `submission.csv`.
3. `specs/roadmap.md` marks Phase 15 complete with a verification note, including any deviations from this spec.
4. No new core dependency was added; any optional dependency change is in `pyproject.toml` and documented.
5. No competition data, credentials, or local-only paths are committed.
