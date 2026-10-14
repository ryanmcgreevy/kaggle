---
name: automl-validation-splits
description: "Use when creating or checking holdout or k-fold validation splits, stratification, validation settings, or the splits.json artifact with the automl CLI or Python API."
---

# AutoML Validation Splits

## Use when

Pinning identical folds for every later candidate, or reviewing the validation strategy. Pinning is recommended before `automl-baseline` and `automl-tuning`, so their scores are comparable.

## Procedure

1. Run the `automl-data-intake` workflow first; task and metric must be explicit and the input data contract must pass.
2. Review `[validation]` in the TOML. Classification uses stratification; regression does not. Keep the configured seed and fold/holdout settings visible to the user.
3. Choose a new run output directory and run `automl split run.toml --output-dir runs/experiment-001`.
4. Check the JSON result and load `runs/experiment-001/splits.json` with `automl.load_splits` or inspect its schema. Confirm strategy, task, row count, split count, and artifact path.
5. If class counts or row counts make the requested split invalid, report the actionable error and ask before changing the validation strategy or parameters.

## Inputs

Run TOML with explicit task and metric, the train CSV, and optional `[validation]` settings (strategy, folds, seed).

## Outputs

`splits.json` (version 1) in `--output-dir`; later stages take it with `--splits`.

## Assumptions and limits

The command refuses to replace `splits.json`. A splits file made for different data (row count or task) is rejected by later stages. Rare classes can make stratified splits invalid; the error says so.

## Check

Exit code 0 and `splits.json` loads with `load_splits`. Equivalent API functions are `make_splits`, `save_splits`, and `load_splits`. See Validation Strategy in `automl/README.md`.