---
name: automl-refit
description: "Use when refitting a selected AutoML model or predicting from a saved ensemble: automl refit, --label, --rank, predictions.csv, refit.json, pipeline.joblib, refit_and_predict, prediction validity checks."
---

# AutoML Refit

## Use when

The user has chosen (or accepts rank 1 of) a baseline, tuning, families, or ensemble result and wants test predictions.

## Procedure

1. Confirm the choice with the user. With a families result or several candidates, show the `model-report` ranking and labels unless rank 1 is clearly intended.
2. Run `automl refit run.toml runs/1/tuning --output-dir runs/1/refit`, adding `--label <label>` or `--rank N` (mutually exclusive) when not using rank 1. Without either, rank 1 is used and the selection note says so. An ensemble directory contains one selected ensemble; pass it without `--label` or `--rank`.
3. Add `--overwrite` only after approval; `--track` only if MLflow was approved.
4. Read the JSON for the selected label, estimator, and number of test rows.

Progress is flushed to stderr by default; use `--quiet` to suppress it or `--verbose` to also enable supported estimator-native output. Stdout remains JSON. Python API callers may supply a `ProgressEvent` callback; calls are otherwise silent.

## Inputs

Run TOML, train and test CSVs, and one saved result directory. Model candidates are rebuilt from saved parameters, preprocessing, and seed; an ensemble uses its full-data fitted `ensemble.joblib`.

## Outputs

`predictions.csv` (ID columns, `prediction`, `proba_<class>` columns, test row order), `refit.json` (selection, config, data summary, versions, relative model reference), `pipeline.joblib`, and the shared `fitted-model/` artifact when `save_model` is true.

## Assumptions and limits

Task and metric must match the selected result. Test rows need every selected training feature; extra columns are ignored. Predictions are raw model outputs with no clipping or thresholding. Exact repeatability only on CPU. Any NaN, infinity, unseen label, or probability outside [0, 1] raises an error and writes nothing. Joblib models are pickles: load only from trusted runs and matching dependency versions. The validation score recorded in `refit.json` is optimistic and not a leaderboard estimate.

## Check

Exit code 0, `n_test_rows` equals the test CSV row count, and `load_refit("runs/1/refit")` loads. Python API: `refit_and_predict`, `select_pipeline`, `load_refit`. Details: Refit and Prediction in `automl/README.md`. Next: `automl-submission`.
