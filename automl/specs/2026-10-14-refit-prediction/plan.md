# Plan: Refit and Prediction (Phase 14)

## 1. Module layout and configuration

1. Add `src/automl/refit.py`; add `RefitError` to `errors.py` following the existing multi-problem error pattern.
2. Move the pipeline construction in `tracking._fit_pipeline` into a shared helper (for example `build_pipeline` in `refit.py` or `preprocessing.py`) and have `tracking.py` call it; do not change its behavior.
3. Define frozen `RefitConfig` (`save_model`, `probabilities`, `proba_tolerance`, `max_reported_rows`) validated in `__post_init__`, collecting all problems.
4. Accept an optional `[refit]` TOML table in `load_config` (unknown keys and wrong types rejected; Python overrides win). Keep config version 1; files without the table load unchanged.

## 2. Pipeline selection

1. Define frozen `PipelineSelection` (`source_kind` of `baseline`/`tuning`/`family`/`explicit`, `source_path`, `label`, `estimator`, `params`, `preprocessing_config`, `task`, `metric`, `direction`, `seed`, `device`, `native_categoricals`, `trial_number`, `validation_mean`, `validation_std`, `versions`) with `to_dict()`.
2. Implement `select_pipeline(source, *, label=None, rank=1)` reusing `collect_candidates` and its ranking; raise `RefitError` for excluded or unknown labels, empty candidate sets, and multiple candidates when no choice was made explicit.
3. Support explicit selection: `select_pipeline(estimator=..., params=..., task=..., metric=..., config=...)`, validating names and params with the existing estimator specs and `controlled_param_problems`.
4. Fail early with the install hint when the selected estimator's library is missing.

## 3. Refit

1. Implement `refit(selection, train, *, target, id_columns=(), resolved_task) -> RefitPipeline` that infers the column plan from the training rows, builds preprocessing plus estimator as one `Pipeline` through the shared helper, resolves the device, and fits on all rows.
2. Reject null targets, an empty frame, a target missing from the frame, and feature columns that differ from the plan stored in the saved result (naming the columns).
3. Return a frozen `RefitPipeline` holding the fitted pipeline, `ColumnPlan`, class list (classification), effective device and fallback notes, fit seconds, and train row count.

## 4. Prediction and validation

1. Implement `predict(refit_pipeline, test, *, id_columns=(), config=None) -> PredictionResult` using only plan feature columns in plan order; error naming any missing feature columns.
2. Classification: labels in the original target dtype and `proba_<class>` columns when `predict_proba` exists and `probabilities` is true; record a note when probabilities are unavailable. Regression: float values.
3. Validate outputs: row count equals test rows; no NaN/inf; labels within training classes; probabilities in [0, 1] and row sums within tolerance. Raise `RefitError` with every problem and up to `max_reported_rows` offending row positions; never return partial results.
4. Build a frame with ID columns first (unchanged values and order), then `prediction`, then probability columns; keep test row order.
5. Return frozen `PredictionResult` (frame, task, prediction kind, n_rows, notes) with `to_dict()` excluding the frame.
6. Implement `refit_and_predict(source_or_selection, bundle_or_config, ...)` as a thin composition of the steps above for people and agents.

## 5. Persistence

1. Implement `save_refit(result, directory, *, overwrite=False)` writing `predictions.csv`, `refit.json` (schema version, selection, effective config, data summary with row counts, column plan, class list, validation notes, effective device, library versions, `allow_nan=False`), and `pipeline.joblib` when `save_model` is true.
2. Write to temporary names in the directory and move into place; refuse overwrite unless `overwrite=True`; leave no partial files after a failure.
3. Implement `load_refit(directory, load_model=False)` returning the metadata and predictions without any estimator library; load the model only on request, with a clear install hint when a library is missing and a trusted-source warning in the docstring.
4. Validate schema version and required keys on load with actionable errors naming the file and key.

## 6. MLflow logging

1. Add an optional `tracking: TrackingConfig | None` argument to `refit_and_predict`; check MLflow availability before fitting.
2. Log the refit pipeline using the existing pipeline-logging helper, selection params as run params, provenance tags (source path, kind, validation mean/std), and `refit.json` as an artifact; do not log raw data or predictions.
3. Return `TrackingResult` alongside the prediction result (`model_uri` populated); a tracking failure raises `TrackingError` after local files are written, naming that local results are intact.
4. Verify the logged pipeline reloads and reproduces the saved predictions.

## 7. Tests

1. Build fixtures by running `run_baseline`, `tune`, and `tune_families` on small synthetic data and saving them; use `importorskip` for Optuna, boosting libraries, and MLflow; add hand-written JSON fixtures so most tests need none of them.
2. Selection: each source kind, rank and label choices, ambiguity error, excluded and unknown labels, explicit selection with invalid estimator, task, and controlled params, missing-library hint.
3. Refit: estimator and preprocessing fitted together (single `Pipeline` with named steps, preprocessing statistics derived from train only and equal to a manual fit); parameters, seed, and preprocessing config match the selection; all train rows used; ID and target excluded; null target and missing columns rejected; repeatability on CPU.
4. Leakage: modifying test rows leaves fitted preprocessing statistics and train-side outputs unchanged.
5. Predict: count equals test rows for binary, multiclass, and regression, for HGB and each installed boosting adapter; string and integer labels preserved; probability columns and sums; no probabilities when unsupported; unseen categories and missing values handled; extra test columns ignored; missing feature columns rejected; row order and IDs preserved.
6. Validity checks: injected NaN, inf, wrong row count, unknown label, and bad probability sums each raise `RefitError` naming the problem and nothing is written.
7. Persistence: round trip equal to in-memory predictions; reloaded model reproduces saved predictions; metadata loads with the estimator libraries blocked; overwrite refusal and `overwrite=True`; no partial files after a forced failure; `allow_nan=False` safe; inputs and source artifacts not mutated (hash-identical).
8. MLflow against a local SQLite store: run logged and queried (params, tags, artifact equal to `refit.json`), model reloads and matches predictions, no raw data in logged content, MLflow absent gives the install hint before fitting, no RUNNING run after a mid-run failure.
9. `[refit]` parsing, unknown key, precedence, and validation errors; `import automl` works without Optuna, Matplotlib, MLflow, or boosting libraries.
10. Phase 12 tracking tests pass unchanged after the helper move.

## 8. Public API export

1. Export `RefitConfig`, `PipelineSelection`, `RefitPipeline`, `PredictionResult`, `select_pipeline`, `refit`, `predict`, `refit_and_predict`, `save_refit`, `load_refit`, and `RefitError` from `automl`.

## 9. Documentation and roadmap

1. Update `automl/README.md`: invocation for people and agents (Python example from saved result to predictions), inputs, outputs, `predictions.csv` and `refit.json` schemas (the contract for Phase 15), `[refit]` defaults, validity rules, MLflow usage, and limits (raw model outputs, one pipeline, no submission schema checks, trusted-only model loading, CPU-only repeatability).
2. Mark Phase 14 complete in `specs/roadmap.md` with a verification note once validation passes.
