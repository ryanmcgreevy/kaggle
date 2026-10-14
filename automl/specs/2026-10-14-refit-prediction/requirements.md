# Requirements: Refit and Prediction (Phase 14)

## Scope

**Deliverable (from roadmap):** Refit the selected pipeline on the declared training data and generate predictions for the test rows.

**Acceptance (from roadmap):** Tests confirm the selected estimator and preprocessing are refit together, prediction count equals test row count, and outputs contain no invalid or non-finite values where the task disallows them.

In scope:

- `select_pipeline` turning a saved result (baseline, tuning, or families directory, or the loaded result) into a `PipelineSelection`: estimator, parameters, preprocessing config, task, metric, seed, device, native-categorical setting, and the provenance (source path, kind, trial number, validation mean/std).
- `refit` fitting preprocessing plus estimator together on all declared training rows and returning a fitted `sklearn.pipeline.Pipeline`.
- `predict` producing a `PredictionResult` for the test rows: labels (classification) or values (regression), plus class probabilities for classification when the estimator provides them.
- Output validation: row count, finite values, label membership in the training classes, probability rows summing to 1 within tolerance, and column/feature alignment with the training plan.
- Persistence to a user-selected directory: `predictions.csv`, `refit.json` (selection, effective config, data summary, versions, validation notes), and the fitted pipeline (`pipeline.joblib`).
- Optional MLflow logging of the refit pipeline, params, and provenance through the existing tracking adapter when a `TrackingConfig` is passed.
- Optional `[refit]` TOML table in config version 1 with Python overrides.
- Synthetic-fixture tests and README documentation.

Out of scope:

- Submission CSV construction and sample-submission schema or identifier alignment (Phase 15); this phase only keeps test identifiers beside predictions so Phase 15 can align them.
- CLI command (deferred to Phase 16; Python API only).
- Ensembling, seed averaging, blending, or multi-model refit; one selected pipeline per call.
- Re-running tuning, re-scoring on validation data, or any leaderboard claim.
- Post-processing such as clipping, thresholding, or rounding; raw model outputs are returned.
- Kaggle API calls, agent skill changes (Phase 17), and a PyTorch estimator.

## Decisions

| Topic | Decision |
|---|---|
| Interface | Python API only: `select_pipeline`, `refit`, `predict`, `refit_and_predict`, `save_refit`, `load_refit`. |
| Selection input | Both: a saved result path or loaded result (reusing the Phase 13 `collect_candidates` directory detection and its best-trial rules) or an explicit estimator name plus parameters. Multiple candidates from one source require an explicit choice (`label=` or `rank=`, default rank 1 by the Phase 13 ranking); ambiguity is never guessed silently. |
| Explicit selection | Estimator name and params are validated with the existing estimator specs and `controlled_param_problems`; task and metric come from the supplied config, not guessed. |
| Refit data | All training rows from `DataBundle.train`; no row sampling. Rows with a null target are rejected with an actionable error (as in the validation stage). |
| Single pipeline | Preprocessing (column plan inferred from train, ID and target excluded) and estimator are built and fit as one `Pipeline`, reusing the same construction path as `tracking._fit_pipeline` so refit matches what was validated. That helper is moved to a shared location rather than duplicated. |
| Preprocessing config | Taken from the selected result; explicit selection uses `PreprocessingConfig` defaults or the config file's `[preprocessing]` table. |
| Seed and device | The selected result's seed and device request; effective device and any fallback are recorded in `refit.json`. Exact repeatability is claimed only for CPU, as in earlier phases. |
| Feature alignment | Test frame must contain every training feature column; extras and the ID columns are ignored; missing columns raise an error naming them. Column order follows the training plan. Dtype problems surface from preprocessing with the column named. |
| Classification outputs | `predictions.csv` has the ID columns (when configured), a `prediction` column of labels in the original target dtype, and one `proba_<class>` column per class when `predict_proba` exists. Probabilities are never invented for estimators without them. |
| Regression outputs | A `prediction` column of floats. |
| Validity rules | Row count equals test rows; no NaN/inf in values or probabilities; classification labels are a subset of training classes; probabilities lie in [0, 1] and rows sum to 1 within `1e-6`. Any violation raises `RefitError` listing the problems and rows (capped) rather than writing a file. Regression values must be finite; no range rule since the task does not define one. |
| Row alignment | Output row i corresponds to test row i; IDs are copied unchanged so Phase 15 can align by identifier. |
| Artifacts | Written only under the chosen directory; refuse overwrite unless `overwrite=True`; files written to temporary names and moved so a failure leaves no partial result. `refit.json` carries a schema version and loads without the estimator libraries installed. |
| Model file | `pipeline.joblib` saved with `save_model=True` (default) and documented as trusted-only to load, consistent with the existing `models.joblib`. Loading a saved refit with an absent estimator library raises an install hint. |
| MLflow | Opt-in via `tracking=TrackingConfig(...)`, as in Phase 12. Logs the refit pipeline (cloudpickle, existing format), selection params, provenance tags, and `refit.json` as an artifact; no raw data or predictions logged. Reuses `track_*` helpers where possible; if MLflow is absent the install hint is raised before any fitting. A tracking failure leaves local results intact and is reported. |
| Config | `RefitConfig`: `save_model` (true), `probabilities` (true), `proba_tolerance` (1e-6), `max_reported_rows` (10); validated in `__post_init__`, collecting all problems. |

## Context

- Phases 7 to 13 provide `build_preprocessor`, `infer_columns`, estimator specs (`get_estimator_spec`, `resolve_device`), `run_baseline`, `tune`, `tune_families`, their save/load functions, `collect_candidates`, and `TrackingConfig`/`ModelData`. Saved results store estimator key, parameters, preprocessing config, seed, device, and the column plan, which is sufficient to rebuild the pipeline.
- Saved fold models (`models.joblib`) are per-fold and trained on partial data; they must not be used as the final model.
- Phase 15 will consume `predictions.csv` and `refit.json`; their schemas are the stage contract and are documented and versioned here.
- Mission principles applied: leakage-aware (refit uses train rows only; test rows never influence fitted statistics), conservative inference, local and private by default, validated outputs, no hidden in-memory state.

## Dependencies

No new core dependency. Optional extras are unchanged (`boosting` for non-HGB estimators, `tuning` only to produce inputs, `mlflow` for tracking). Core import and the sklearn-only path must work without them.

## Risks and open items

- `tracking._fit_pipeline` is private; moving it risks Phase 12 regressions, so Phase 12 tests must pass unchanged.
- Estimators with no `predict_proba` (or a library mismatch between the saved result and the installed version) must produce clear notes, not errors in the label path.
- Test data with unseen categories relies on the encoders' existing handling; the tests must confirm this per encoder.
