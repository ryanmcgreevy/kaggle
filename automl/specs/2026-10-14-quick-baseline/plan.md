# Plan: Quick Baseline (Phase 8)

## 1. Module layout and errors

1. Add `src/automl/baseline.py` (config, runner, result, artifact I/O).
2. Add `BaselineError` to `errors.py` as a `DataContractError` subclass with the same multi-problem message style.
3. Confirm `import automl`, `automl --help`, and existing tests are unaffected. No new dependencies (scikit-learn and its `joblib` dependency are already core).

## 2. Configuration

1. Define a frozen `BaselineConfig`: `seed` (default 42), `n_jobs` (default 1), `save_models` (default True), and `estimator_params` (default empty; overrides for the HistGradientBoosting defaults, validated against the estimator's parameters).
2. Validate in `__post_init__`: integer `seed`, `n_jobs` of -1 or at least 1, boolean `save_models`, mapping of known estimator parameters; collect all problems in one pass. Reject `random_state` inside `estimator_params`; `seed` controls it.
3. Add an optional `[baseline]` table to `load_config` (unknown keys and wrong types rejected; Python overrides win), keeping config version 1 and unchanged behavior for files without the table. Expose `baseline_config` on the loaded config. The run `seed` keyword stays separate, as in `[validation]`.

## 3. Estimator selection

1. Implement `make_estimator(task, config)` returning an unfitted scikit-learn estimator: `HistGradientBoostingClassifier` for `binary` and `multiclass`, `HistGradientBoostingRegressor` for `regression`, seeded and with early stopping off so results do not depend on an internal validation split.
2. Treat encoded categorical columns as ordinary numeric inputs; document this limit.

## 4. Baseline run

1. Implement `run_baseline(train, target, id_columns, resolved_task, splits, config=None, preprocessing_config=None) -> BaselineResult`.
2. `splits` is a `SplitResult` from Phase 3; add a helper `make_splits_from_config` only if needed, otherwise callers pass `make_splits(...)` output (or `load_splits(...)`).
3. When `preprocessing_config` is None, use `PreprocessingConfig(encoder="ordinal", seed=config.seed)`; otherwise use it as given.
4. Build the plan with `infer_columns` and the preprocessor with `build_preprocessor`, then for each split clone a `Pipeline(preprocessor, estimator)`, fit on the train rows only, and time the fit and predict.
5. Choose the prediction method from the resolved metric's `needs`: `predict_proba` for `proba` (positive-class column for binary), `predict` for `labels` and `values`. Reject classes in validation rows unseen in the fold's training rows with a clear error before scoring multiclass probability metrics.
6. Score each fold with the resolved metric on its validation rows; return per-fold scores, mean, standard deviation, and direction.
7. Collect out-of-fold predictions keyed by positional row index. For k-fold they cover every row; for holdout they cover only validation rows and the rest are marked missing. Never score on training rows.
8. Check predictions for finite values and the right shape before scoring; raise `BaselineError` naming the fold otherwise.
9. Do not mutate the inputs.

## 5. Result and artifacts

1. Define frozen `BaselineResult`: `task`, `metric` name and direction, `estimator` name and effective parameters, effective `BaselineConfig` and `PreprocessingConfig`, `ColumnPlan`, split strategy/seed/n_folds summary, per-fold records (`fold`, `n_train`, `n_valid`, `score`, `fit_seconds`, `predict_seconds`), mean and std, `n_rows`, out-of-fold predictions, and in-memory fold models.
2. Implement `save_baseline(result, directory)` writing `baseline.json`, `oof_predictions.csv`, and (when `save_models`) `models.joblib` into a new directory; refuse to overwrite an existing directory or file.
3. `baseline.json` records schema version, effective config, split summary, column plan, per-fold scores and timings, summary, library versions, and relative artifact file names; it contains no absolute paths or data values beyond OOF predictions in the CSV.
4. `oof_predictions.csv` has `row` (positional index), `fold`, and one column per prediction output (`prediction`, or `proba_<class>` for multiclass); rows without out-of-fold predictions are omitted.
5. Implement `load_baseline(directory, load_models=False) -> BaselineResult` validating schema/version, file presence, row indices, and score consistency with actionable errors. Document that `models.joblib` must only be loaded from trusted sources, and that loading it requires compatible library versions.

## 6. Tests

1. Add synthetic binary, multiclass, and regression frames with missing values and categoricals to `tests/conftest.py` (reuse existing fixtures where present).
2. Cover: finite metrics, predictions, effective configuration, and a loadable result for each task with the default metric; every built-in metric whose `needs` is `labels`, `proba`, and `values` for its tasks; k-fold OOF covering every row once and holdout OOF covering only validation rows; scores computed on validation rows only; repeatability for a fixed seed (the estimator seed does not change results with early stopping off, so variation is tested through the split seed); ID and target excluded from the fit; unseen validation classes and non-finite predictions raising errors; config validation errors and `[baseline]` TOML parsing, unknown key, override precedence; `estimator_params` validation; non-mutation of inputs; artifact round trip, overwrite refusal, malformed-file rejection, and `save_models=False` behavior; saved models reproduce OOF predictions on the validation rows after reload.
3. Assert error messages name the offending setting, fold, or class.
4. Add a test that `import automl` and `run_baseline` work with none of the optional extras installed.

## 7. Public API export

1. Export `BaselineConfig`, `BaselineResult`, `run_baseline`, `save_baseline`, `load_baseline`, and `BaselineError` from `automl`.

## 8. Documentation and roadmap

1. Update `automl/README.md` with the `[baseline]` schema, defaults, the prediction/metric rules, the artifact layout and JSON schema, the model-file trust and compatibility caveats, and a Python example for people and agents (no CLI this phase; Phase 16 composes the CLI).
2. Mark Phase 8 complete in `specs/roadmap.md` with a verification note once validation passes.
