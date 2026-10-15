# Model Ensembling Plan

## Task Groups

1. **Defining ensemble contracts and candidate inputs**
   - Specify API/configuration types for method, candidate references, task/metric, validation settings, and stacking meta-estimators.
   - Load and validate saved baseline, tuning, and model-family candidates without changing their preprocessing or silently retuning them.
   - Define machine-readable ensemble result and fitted-artifact schemas.

2. **Implementing classification voting and stacking**
   - Implement hard majority voting over compatible classifier candidates.
   - Implement classification stacking with configurable meta-estimator and LogisticRegression default.
   - Verify class-label compatibility and document hard-vote tie behavior.

3. **Implementing regression stacking**
   - Implement stacking for regression candidates with configurable regressor and RidgeCV default.
   - Reject regression voting and unsupported task/method combinations with actionable errors.

4. **Evaluating and persisting ensembles**
   - Evaluate ensembles against declared shared validation splits and report constituent-candidate comparisons without promising improvement.
   - Ensure stacking meta-training and all learned preprocessing are leakage-safe.
   - Persist structured metrics/configuration and a loadable ensemble atomically, respecting no-overwrite behavior.

5. **Integrating refit, CLI, and agent guidance**
   - Add the Python API and CLI entry points, with matching options and behavior.
   - Allow the existing refit/prediction flow to select and consume a saved ensemble.
   - Update directly related documentation and agent guidance for inputs, outputs, limitations, and checks.

6. **Validating compatibility and regression safety**
   - Add focused synthetic tests for task/method behavior, invalid candidate selections, determinism, leakage, artifact restart, refit predictions, and CLI/API parity.
   - Confirm existing workflows and core imports continue to work without optional model libraries.
