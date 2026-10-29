# Ensemble Evaluation and Fit-Only Plan

## Task Groups

1. **Defining explicit mode and result semantics**
   - Inspect the current ensemble API/configuration, CLI parsing, result schema, and saved-artifact loading path.
   - Define the public `evaluate` and `fit` mode values and preserve evaluation as the default.
   - Define how fit-only results represent missing validation scores and distinguish them from evaluated results without invalidating existing artifacts.

2. **Implementing fit-only candidate ensembling**
   - Allow candidate-based ensemble calls in fit-only mode to proceed without outer validation splits.
   - Reuse the saved candidate estimator parameters, preprocessing, selection, and compatibility validation.
   - Fit and persist the final ensemble on all training rows without running the outer evaluation loop.
   - Keep inner OOF cross-validation for stacking's meta-model and keep voting free of validation CV.

3. **Preserving evaluation and downstream compatibility**
   - Keep current evaluation mode behavior, metrics, candidate comparisons, and default CLI/API behavior unchanged.
   - Ensure both evaluated and fit-only persisted ensembles remain loadable by `automl refit` and produce aligned predictions.
   - Preserve mode, fit configuration, and validation status in machine-readable ensemble metadata.

4. **Exposing mode through CLI and Python API**
   - Add equivalent mode selection to the `ensemble` command and Python API/configuration.
   - Make outer split input optional only for fit-only mode; provide actionable validation errors for incompatible mode/split combinations.
   - Keep machine-readable stdout, progress output, and existing output-directory protections intact.

5. **Updating documentation and agent guidance**
   - Document evaluation versus fit-only goals, CLI/API examples, split requirements, output score semantics, and relative fit costs in the README.
   - Update the ensemble skill so agents choose fit-only mode when the user requests model assembly/predictions without a validation comparison.
   - Clarify that stacking still uses inner OOF CV in fit-only mode and that fit-only has no ensemble validation score.

6. **Validating behavior and compatibility**
   - Add tests proving default and explicit evaluation modes preserve outer-fold candidate and ensemble scores.
   - Add tests proving fit-only mode does not require or run outer validation and does not emit a validation score.
   - Cover classification voting, classification stacking, regression stacking, and leakage-safe inner OOF training.
   - Verify persisted fit-only models reload and work through refit/prediction with correct test-row alignment.
   - Test CLI/API parity, invalid mode/split inputs, existing result loading, progress/stdout behavior, and no overwrite regressions.
   - Run focused ensemble, CLI, refit, and agent-guidance tests, then the full AutoML suite.
