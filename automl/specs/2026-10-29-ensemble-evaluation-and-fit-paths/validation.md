# Ensemble Evaluation and Fit-Only Validation

## Acceptance Checks

### Evaluation mode and backwards compatibility

- Calls that omit the new mode retain current evaluation behavior.
- Evaluation mode scores the ensemble and each candidate on the same declared outer folds and persists finite fold/summary scores.
- Evaluation mode rejects absent or invalid outer splits with actionable errors.
- Existing result artifacts remain loadable, and existing CLI/API invocations continue to work without changes.

### Fit-only mode

- Fit-only mode accepts compatible saved baseline, tuning, and model-family candidates without an outer split artifact or validation configuration.
- Instrumented tests verify that no outer-fold evaluation runs and that candidate/ensemble validation scores are not computed.
- Fit-only classification voting and stacking, and regression stacking, produce usable models fitted from all available training rows.
- Stacking's meta-model is fitted from inner out-of-fold member predictions; tests verify held-out rows are excluded from the corresponding member fit. Voting performs no inner validation CV.
- Persisted metadata identifies fit-only mode and clearly marks ensemble validation as not performed; summary validation metrics are absent/unavailable rather than training scores.
- Refit/prediction can reload a fit-only ensemble, enforces existing data/schema checks, and preserves test-row ordering and prediction validity.

### Interfaces, artifacts, and documentation

- CLI and Python API provide equivalent mode selection and behavior.
- Invalid mode names and incompatible split inputs fail clearly; output directory no-overwrite behavior is unchanged.
- Stdout remains parseable JSON and progress stays on the established output channel.
- README and ensemble agent guidance explain mode choice, outer split requirements, stacking's remaining inner CV, score interpretation, and fit/resource tradeoffs.

## Suggested Validation Sequence

1. Run focused unit tests for fit-only estimator fitting and score/status metadata.
2. Run classification and regression fit-only tests, including inner stacking leakage checks.
3. Run evaluation compatibility and CLI/API parity tests.
4. Run persisted model reload and refit/prediction alignment tests.
5. Run agent guidance tests and the full AutoML test suite.

## Verification Result

- Full AutoML suite: **641 passed, 1 skipped** (the skipped test requires a usable GPU).
- Final focused ensemble and CLI suite: **55 passed**.
- Fit-only mode was exercised for classification voting and stacking and regression stacking, including persisted reload and downstream refit/prediction.
- Python package compilation and `git diff --check` passed.
- The MLflow integration emitted one SQLAlchemy deprecation warning; it did not fail.
