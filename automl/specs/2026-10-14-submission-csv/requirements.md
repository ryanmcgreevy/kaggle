# Requirements: Submission Validation and CSV (Phase 15)

## Scope

**Deliverable (from roadmap):** Build a submission CSV using the sample submission when supplied, preserving required identifiers, column names, and order.

**Acceptance (from roadmap):** Tests cover schema/order/row-count mismatches, identifier alignment, and valid file writing. The workflow never calls Kaggle submission APIs or overwrites an existing output without explicit instruction.

In scope:

- `build_submission` turning a `RefitResult` (or a saved refit directory plus the test frame) and the optional sample submission into a validated `SubmissionResult`.
- Row order of the submission equals the row order of the original `test.csv`. Predictions are never reordered relative to test rows.
- Column names and column order of the submission match the sample submission exactly.
- Mapping predictions onto the sample's non-ID columns (labels, values, a single probability column, or one probability column per class), inferred from the sample and the refit result, with an explicit override.
- Value checks: no nulls, finite numbers, dtypes compatible with the sample.
- Fallback format when no sample submission is supplied: ID columns followed by the target column name.
- `save_submission` writing `submission.csv` and `submission.json` (metadata) to a user-selected directory without implicit overwrite.
- Optional MLflow logging of the submission artifact through the existing tracking adapter.
- Optional `[submission]` TOML table in config version 1 with Python overrides.
- Synthetic-fixture tests and README documentation.

Out of scope:

- Any Kaggle API call or automatic submission.
- CLI command (Phase 16); Python API only.
- Post-processing of predictions (clipping, thresholding, rounding, rank transforms) beyond the documented mapping; no ensembling or blending.
- Reordering the submission to the sample's identifier order. The test order wins; the sample is used for schema and ID membership only.
- Agent skill changes (Phase 17).

## Decisions

| Topic | Decision |
|---|---|
| Interface | Python API only: `build_submission`, `save_submission`, `load_submission`, `SubmissionConfig`, `SubmissionResult`, `SubmissionError`. |
| Inputs | `RefitResult` or `load_refit` output, the test frame (`DataBundle.test`), the ID columns, and the optional sample submission frame. The test frame is needed to prove ID alignment; no hidden state. |
| Row order | Output row `i` is test row `i`. Predictions in the refit result already follow test order (Phase 14); this is re-verified by comparing the refit ID columns to the test ID columns row by row, and a mismatch raises. |
| Sample ID check | With a sample and ID columns, the sample's ID values must equal the test ID values as a set, with no duplicates or nulls in either. The sample's row order may differ from test order; that is accepted and recorded in a note, and the output keeps test order. Different ID sets, duplicates, or counts raise. Set equality is only the membership check; the written order is always the test order. |
| Sample without ID columns | Sample must have the same number of rows as test; the sample's ID column(s) are taken as configured. If the target configuration has no ID columns, rows are matched by position only and a note records that identifier alignment could not be verified. |
| Column schema | With a sample, the submission columns are exactly the sample's columns, in the sample's order and with the sample's names. Missing or extra columns are never silently added or dropped; an unmappable column raises. |
| Prediction mapping | Sample ID columns are copied from test. Remaining sample columns are mapped as follows: one column gets labels/values from `prediction` (or the positive-class probability when the configured metric needs probabilities and the task is binary); several columns whose names equal class labels (as strings) get the matching probability columns; columns named `proba_<class>` also map. Anything else raises with the sample column names and the available prediction columns. `SubmissionConfig.column_map` (sample column to source column) overrides inference. |
| Dtypes | Each output column is cast to the sample column's dtype class (integer, float, string/object, bool) when lossless; a lossy or impossible cast raises naming the column. Integer labels stay integers. |
| Value checks | No null values, no NaN/inf, dtype compatible with the sample, row count equals test rows, no duplicate IDs. Probability columns must lie in [0, 1]. Violations raise `SubmissionError` listing every problem with up to `max_reported_rows` row positions; no partial file is written. |
| No sample submission | Columns are the configured ID columns then the target column name, holding `prediction`. Multi-class probability output is not produced in this mode. A note records that no sample schema was available and the schema is unverified. |
| Writing | Output only under the chosen directory as `submission.csv` and `submission.json`. Refuse to overwrite unless `overwrite=True`; write temporary files then move so a failure leaves no partial result. CSV has a header, no index column, and float formatting that round-trips. |
| Metadata | `submission.json` has a schema version, source refit directory/label, validation notes, column mapping, sample file name (not contents), row counts, ID check results, and a statement that scores are not leaderboard claims. Written with `allow_nan=False`. |
| MLflow | Opt-in via `tracking=TrackingConfig(...)`; logs `submission.json` and `submission.csv` as artifacts of the run and tags the source. MLflow absence raises the install hint before any file is written. A tracking failure leaves local files intact and is reported. |
| Config | `SubmissionConfig`: `column_map` (none), `max_reported_rows` (10), `float_format` (none; default repr round-trip), `allow_unverified_ids` (false). Validated in `__post_init__` collecting all problems. |

## Context

- Phase 14 `refit_and_predict` returns a `RefitResult` whose `predictions.frame` has ID columns first, then `prediction`, then `proba_<class>` columns, in test row order. `refit.json` and `predictions.csv` are the stage contract.
- Phase 1 `DataBundle` provides `test`, `sample_submission`, `target`, and `id_columns`.
- Mission principle applied: validated outputs, preserve row-to-prediction alignment, reject or explicitly realign mismatched identifiers; here alignment is to test order and any order mismatch is rejected rather than reordered.
- Never upload or submit results automatically.

## Dependencies

No new core dependency. MLflow remains optional. Core import must work without Optuna, Matplotlib, MLflow, and boosting libraries.

## Risks and open items

- Sample submissions vary widely (multi-class probabilities, rank columns, string labels); inference must fail loudly with the available choices and rely on `column_map` rather than guess.
- Competitions whose sample IDs are all zero or placeholder values would fail the ID check; `allow_unverified_ids=True` exists as an explicit, recorded opt-out.
- Float precision when casting probabilities to the sample's dtype should not alter values beyond the dtype's representation.
