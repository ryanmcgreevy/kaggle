# Plan: Submission Validation and CSV (Phase 15)

## 1. Module layout and configuration

1. Add `src/automl/submission.py`; add `SubmissionError` to `errors.py` following the existing multi-problem error pattern.
2. Define frozen `SubmissionConfig` (`column_map`, `max_reported_rows`, `float_format`, `allow_unverified_ids`) validated in `__post_init__`, collecting all problems.
3. Accept an optional `[submission]` TOML table in `load_config` (unknown keys and wrong types rejected; Python overrides win). Keep config version 1; files without the table load unchanged.

## 2. Inputs and identifier alignment

1. Implement `build_submission(refit, test, *, id_columns, sample=None, target=None, config=None, tracking=None) -> SubmissionResult`, where `refit` is a `RefitResult` or a `LoadedRefit`.
2. Verify the refit prediction ID columns equal the test ID columns row by row (same values, same order); raise `SubmissionError` naming the first differing positions.
3. Reject null or duplicate IDs in test and sample, differing row counts, and ID sets that differ between test and sample; list sample-only and test-only values (capped).
4. Accept sample rows in a different order than test; record a note and keep test order. Never reorder the output to the sample order.
5. When no ID columns are configured, require equal row counts, match by position only, and record a note that identifier alignment is unverified (error unless `allow_unverified_ids=True` when a sample is given).

## 3. Schema and prediction mapping

1. Build the output columns as exactly the sample's column names in the sample's order; raise for any sample column that cannot be mapped.
2. Infer the mapping: ID columns from test; one value column from `prediction` (or positive-class probability for binary tasks whose metric needs probabilities); per-class probability columns matched by class label or `proba_<class>` name.
3. Support `column_map` overrides, validating that every source column exists in the prediction frame and every sample column is covered.
4. Cast each column to the sample dtype class when lossless; raise naming the column for lossy casts.
5. Implement the no-sample fallback: ID columns plus the target column name holding `prediction`, with a recorded unverified-schema note.

## 4. Value validation

1. Check row count equals test rows, no nulls, no NaN/inf, no duplicate IDs, probability columns within [0, 1], and dtype compatibility with the sample.
2. Collect every problem with up to `max_reported_rows` row positions and raise one `SubmissionError`; never return partial results.
3. Return frozen `SubmissionResult` (frame, columns, column mapping, n_rows, notes, id check summary) with `to_dict()` excluding the frame.

## 5. Persistence

1. Implement `save_submission(result, directory, *, overwrite=False)` writing `submission.csv` (header, no index, test row order) and `submission.json` (schema version, source, mapping, sample file name, counts, checks, notes, no-leaderboard statement, `allow_nan=False`).
2. Write to temporary names and move into place; refuse overwrite unless `overwrite=True`; leave no partial files after a failure.
3. Implement `load_submission(directory)` returning metadata and the frame, validating schema version and required keys with actionable errors naming the file and key.
4. Document that nothing contacts Kaggle and that the file is for manual upload.

## 6. MLflow logging

1. Add an optional `tracking: TrackingConfig | None` argument; check MLflow availability before writing any file.
2. Log `submission.json` and `submission.csv` as artifacts with source tags; do not log the raw training or test data.
3. A tracking failure raises `TrackingError` after local files are written, naming that local results are intact.

## 7. Tests

1. Fixtures: synthetic bundles for binary (label and probability sample), multiclass (per-class probability sample), and regression, built through `refit_and_predict`; add hand-written refit fixtures so most tests need no optional library.
2. Row order: output equals test order; shuffled sample order accepted with a note and unchanged output order; a refit frame in a different order than test is rejected; hash-check test and sample inputs are not mutated.
3. Schema: column names and order equal the sample exactly; missing, extra, renamed, and unmappable columns raise; `column_map` overrides work and bad maps are rejected.
4. IDs: duplicate, null, missing, and extra identifiers raise with sample-only and test-only values named; no-ID configuration records the unverified note and errors without `allow_unverified_ids`.
5. Values and dtypes: injected NaN, inf, null, out-of-range probability, wrong row count, and lossy casts each raise `SubmissionError` and write nothing; integer, float, string, and bool sample dtypes are preserved.
6. No-sample fallback: columns are IDs then the target name with the unverified note.
7. Persistence: round trip of `submission.csv` and `submission.json`; CSV read back equals the frame; overwrite refusal and `overwrite=True`; no partial files after a forced failure; no Kaggle or network call (socket access blocked in the test).
8. MLflow against a local SQLite store: artifacts logged and queried, absent MLflow gives the install hint before any file is written, local files survive an unreachable store.
9. `[submission]` parsing, unknown key, precedence, and validation errors; `import automl` works without Optuna, Matplotlib, MLflow, or boosting libraries.

## 8. Public API export

1. Export `SubmissionConfig`, `SubmissionResult`, `build_submission`, `save_submission`, `load_submission`, and `SubmissionError` from `automl`.

## 9. Documentation and roadmap

1. Update `automl/README.md`: invocation for people and agents (Python example from a saved refit to `submission.csv`), inputs, outputs, `submission.json` schema, `[submission]` defaults, mapping rules, validity rules, MLflow usage, and limits (no CLI, no post-processing, test order always wins, nothing is submitted to Kaggle).
2. Mark Phase 15 complete in `specs/roadmap.md` with a verification note once validation passes.
