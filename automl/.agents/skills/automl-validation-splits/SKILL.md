---
name: automl-validation-splits
description: "Use when creating or checking holdout or k-fold validation splits, stratification, validation settings, or the splits.json artifact with the automl CLI or Python API."
---

# AutoML Validation Splits

## Procedure

1. Run the `automl-data-intake` workflow first; task and metric must be explicit and the input data contract must pass.
2. Review `[validation]` in the TOML. Classification uses stratification; regression does not. Keep the configured seed and fold/holdout settings visible to the user.
3. Choose a new run output directory and run `automl split run.toml --output-dir runs/experiment-001`.
4. Check the JSON result and load `runs/experiment-001/splits.json` with `automl.load_splits` or inspect its schema. Confirm strategy, task, row count, split count, and artifact path.
5. If class counts or row counts make the requested split invalid, report the actionable error and ask before changing the validation strategy or parameters.

## Contract

The command writes the existing version-1 split artifact and refuses to replace `splits.json`. Output paths are relative to the current working directory. Equivalent API functions are `make_splits`, `save_splits`, and `load_splits`. See Validation Strategy in `automl/README.md` for the artifact fields and limitations.