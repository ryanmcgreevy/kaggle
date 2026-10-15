# Model Ensembling Validation

## Acceptance Checks

### Classification

- Synthetic binary and multiclass fixtures verify hard voting returns one valid class label per input row and follows the documented tie behavior.
- Synthetic binary and multiclass fixtures verify stacking with the default logistic-regression meta-model and a user-configured compatible meta-estimator.
- Incompatible class labels, missing required prediction capabilities, invalid selections, and unsupported methods fail with actionable errors.

### Regression

- A synthetic regression fixture verifies stacking with the default RidgeCV meta-estimator and a user-configured compatible regressor.
- Regression hard/majority voting is rejected explicitly.

### Validation and leakage

- Results are reproducible for a fixed seed and saved split configuration where estimator behavior permits.
- Candidate comparisons use identical declared splits and report validation metrics distinctly from leaderboard outcomes.
- Tests prove the stacking meta-model is trained only from out-of-fold predictions and that preprocessing is fitted within the relevant training folds.
- Changing validation rows does not change learned preprocessing or meta-training inputs for the corresponding training folds.

### Persistence and workflow integration

- A saved result and fitted ensemble load after process restart and reproduce predictions on synthetic data.
- The existing refit stage can consume an ensemble; prediction count, values, and row alignment satisfy existing refit checks.
- CLI and Python API results agree for the same configuration and inputs; invalid CLI invocations return a non-zero status and actionable error.
- Outputs are written only beneath the selected directory, do not overwrite implicitly, and contain no raw competition rows.

### Compatibility

- Existing tests for baseline, tuning, model-family search, model reporting, refit, and submission continue to pass.
- Core package import and non-ensemble workflows succeed without optional boosting, PyTorch, MLflow, or reporting extras.
- No network, Kaggle credentials, cloud resources, or GPU are needed for synthetic tests.

## Suggested Validation Sequence

1. Run focused ensemble API, artifact, and CLI tests.
2. Run related refit and submission tests.
3. Run the full test suite and core-only optional-dependency checks.
4. Exercise the documented API and CLI examples with synthetic classification and regression data.
