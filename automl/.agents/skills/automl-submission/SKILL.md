---
name: automl-submission
description: "Use when building or validating a Kaggle submission CSV from AutoML refit predictions: automl submit, submission.csv, submission.json, sample submission columns, identifier alignment, column_map, build_submission. Never uploads."
---

# AutoML Submission

## Use when

A refit directory exists and the user wants a schema-checked `submission.csv`.

## Procedure

1. Run `automl submit run.toml runs/1/refit --output-dir runs/1/submission`. The config's test and sample-submission files are used.
2. If the sample has several prediction columns or names that do not match the classes, ask the user, then set `[submission.column_map]` in the TOML (sample column = prediction column) and rerun.
3. Without id columns, rows match by position only with `allow_unverified_ids = true`; ask the user before setting it.
4. Add `--overwrite` only after approval; `--track` only if MLflow was approved.
5. Tell the user the file is ready for manual upload. Never upload or submit it, or suggest commands that do.

## Inputs

Run TOML, test CSV, optional sample submission, and a refit directory.

## Outputs

`submission.csv` (header, no index, test row order, exactly the sample's columns) and `submission.json` (column mapping, id check, schema source `sample` or `fallback`, notes).

## Assumptions and limits

Output order follows `test.csv`; refit identifiers must match the test rows one by one, so matching sets in a different order are rejected. A sample in another order is accepted with a note. Values are cast to the sample's dtypes only when lossless. Nulls, infinity, probabilities outside [0, 1], and row-count mismatches raise and write nothing. No post-processing such as clipping or thresholding. Without a sample the schema is unverified (`fallback`); say so.

## Check

Exit code 0, `schema_source` is `sample`, the columns equal the sample's, row count equals the test CSV, and `load_submission("runs/1/submission")` loads. Python API: `build_submission`, `save_submission`, `load_submission`. Details: Submission CSV in `automl/README.md`.
