# Validation: Refit and Prediction (Phase 14)

The feature is ready to merge when all checks pass on Python 3.12 or newer from the `automl/` project root, without competition data, Kaggle credentials, network access (after install), a notebook session, or a running MLflow server.

## Required Checks

1. **Install**

   ```bash
   python -m pip install -e '.[test,reports,tuning,boosting,mlflow]' && python -m pip check
   ```

   Pass when install succeeds, `pip check` is clean, and core `dependencies` are unchanged.

2. **Existing behavior intact**

   ```bash
   python -c 'import automl' && automl --help && python -m automl --help
   ```

   Pass when all exit 0 and no new CLI command appears.

3. **Test suite**

   ```bash
   python -m pytest
   ```

   Pass when all tests succeed, including Phase 0 to 13 tests (Phase 12 tracking tests unchanged after the shared helper move) and new tests covering:

   - selection from baseline, tuning, and families sources and from explicit estimator and params, with rank and label choices, ambiguity, excluded, unknown, and invalid-input errors;
   - one `Pipeline` containing preprocessing and estimator, fitted on all train rows, with parameters, seed, and preprocessing config equal to the selection and statistics equal to a manual train-only fit;
   - leakage: changing test rows does not change fitted preprocessing statistics;
   - prediction count equal to test row count for binary, multiclass, and regression, for HGB and each installed boosting adapter;
   - label dtype preservation, probability columns and sums, no probabilities when unsupported, unseen categories, missing values, extra test columns ignored, missing feature columns rejected;
   - row order and ID preservation;
   - each validity rule (NaN, inf, row count, unknown label, probability range and sum) raising `RefitError` with nothing written;
   - persistence round trip, reloaded model reproducing saved predictions, metadata loading with estimator libraries blocked, overwrite refusal and `overwrite=True`, no partial files after a forced failure, `allow_nan=False` safety, and source artifacts and input data not mutated;
   - MLflow logging against a local SQLite store (params, tags, artifact equal to `refit.json`, reloaded model reproducing predictions, no raw data logged, install hint before fitting when absent, no RUNNING run after a mid-run failure);
   - `[refit]` TOML parsing, unknown key, precedence, and validation errors;
   - `import automl` and the sklearn-only path working without Optuna, Matplotlib, MLflow, or boosting libraries.

4. **Core-only environment**

   Create a clean environment with `pip install -e '.[test]'` and run `python -m pytest`. Pass when all tests succeed, optional-dependency tests skip cleanly, and baseline-based refit and prediction work.

5. **End-to-end Python smoke check**

   Run the README example on synthetic CSVs: `run_baseline` (saved), then `refit_and_predict` from the saved directory with `tracking=TrackingConfig(directory=...)`, then `load_refit`. Pass when prediction count equals test rows, all values are finite, the saved and MLflow-loaded pipelines reproduce the predictions, and the process runs with no network access.

6. **Optional GPU check (not required to merge)**

   If an NVIDIA GPU is available, refit with `device="auto"` for XGBoost or CatBoost and confirm the effective device is recorded in `refit.json`. Record the outcome in the roadmap note.

## Documentation Checks

- `automl/README.md` documents invocation for people and agents, inputs, outputs, `predictions.csv` and `refit.json` schemas, `[refit]` defaults, validity rules, MLflow usage, and limits, and its code example runs as written.
- `specs/roadmap.md` marks Phase 14 complete with a verification note and any deviations from this spec.
- No CLI behavior, submission construction, or agent-skill content was added.

## Safety Checks

- Nothing is written outside the chosen directory; no existing output is overwritten without `overwrite=True`.
- No Kaggle API call is made and no network access is needed after install.
- Fold models from earlier results are never used as the final model.
- Logged MLflow content contains no raw data, predictions, or credentials; URIs are redacted in messages.
- Loading `pipeline.joblib` is documented as trusted-source only.

## Merge Criteria

All required checks pass, documentation and roadmap are updated, deviations from this spec are recorded in the roadmap note, and the working tree contains only intended changes on branch `automl/phase-14-refit-and-prediction`.
