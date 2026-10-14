# Requirements: Quick Baseline (Phase 8)

## Scope

**Deliverable (from roadmap):** One fast scikit-learn baseline per supported task type, evaluated with the configured validation strategy and metric.

In scope:

- One baseline estimator per task: HistGradientBoosting classifier (binary, multiclass) and regressor.
- Optional `[baseline]` TOML table (`seed`, `n_jobs`, `save_models`, `estimator_params`) in config version 1, with Python overrides.
- A `run_baseline` function that composes Phase 1 to 3 and 7 outputs: data, resolved task and metric, splits, and the preprocessing plan.
- Prediction method chosen from the metric's `needs` (`labels`, `proba`, `values`).
- Per-fold and summary scores in the metric's direction, with timings.
- Out-of-fold predictions held in memory and persisted.
- A persisted result directory: JSON result, out-of-fold predictions CSV, and fitted fold pipelines (joblib).
- Synthetic-fixture tests and README documentation.

Out of scope:

- CLI command (deferred to Phase 16; Python API only).
- Dummy, linear, or other baseline estimators; boosting adapters (Phase 9); tuning (Phases 10 and 11).
- Refitting on all training rows and test predictions (Phase 14); submission files (Phase 15).
- Native categorical handling, early stopping, sample weights, and multi-metric scoring.
- Reports and MLflow tracking (Phases 12 and 13).
- Agent skill updates (consolidated in Phase 17; README documents invocation now).

## Decisions

| Topic | Decision |
|---|---|
| Estimators | `HistGradientBoostingClassifier` for `binary`/`multiclass`; `HistGradientBoostingRegressor` for `regression`. Seeded, early stopping off. Defaults overridable through `estimator_params` (validated, no `random_state`). |
| Preprocessing | Built with `infer_columns`/`build_preprocessor` inside a per-fold `Pipeline`, so all learned state is fit on train rows only. When `preprocessing_config` is None, default to `PreprocessingConfig(encoder="ordinal", seed=config.seed)`; passing `cfg.preprocessing_config` uses the run config as-is. Encoded categoricals are treated as numeric. |
| Splits | Consumes a Phase 3 `SplitResult` (built or loaded); the split strategy, seed, and fold count are recorded in the result. |
| Metric inputs | `predict_proba` for `proba` metrics (positive-class column for binary), `predict` for `labels` and `values`. Scores use validation rows only. |
| OOF predictions | k-fold covers every row once; holdout covers validation rows only, other rows are absent. Predictions are checked for shape and finiteness before scoring. |
| Result | Frozen `BaselineResult` holding config, plan, split summary, per-fold records, mean/std, OOF predictions, and fold models. Direction comes from the metric; no "best" claim beyond the stated validation score. |
| Artifacts | New directory with `baseline.json` (schema version, effective config, plan, scores, timings, library versions, relative file names), `oof_predictions.csv`, and `models.joblib` when `save_models` is true. Existing directories or files are never overwritten. No absolute paths. |
| Model files | Joblib is unpickled on load, so `models.joblib` is loaded only when requested and only from trusted sources. It is tied to the library versions recorded in the JSON. |
| Config location | Optional `[baseline]` table in config version 1 (no version bump). Python overrides take precedence. Files without the table load unchanged. |
| Interface | Python API only. |
| Dependencies | None new. |

## Context

- `specs/mission.md`: reproducible with explicit seeds and persisted configuration; leakage-aware; inspectable; resource-aware; no hidden single score.
- `specs/tech-stack.md`: start with scikit-learn including HistGradientBoosting; learned transforms inside the fitted pipeline used by CV; stage contracts with machine-readable artifacts and documented schemas; do not overwrite prior outputs; core runs without optional dependencies.
- Phase 2 provides `ResolvedTask` and `Metric` (`direction`, `needs`); Phase 3 provides `SplitResult`; Phase 7 provides `ColumnPlan`, `build_preprocessor`, and `PreprocessingConfig`. This phase adds no new data-handling logic of its own.
- Later phases consume the result: Phase 9 adds estimators with the same result shape, Phase 10 reuses the CV objective, Phase 13 reports from the JSON, and Phase 14 refits using the recorded plan and estimator parameters.

## Assumptions

- Train rows have already passed the data contract; the target has no nulls (enforced by splitting).
- Positional split indices refer to rows of the train frame.
- Dense preprocessed output and in-memory fold models are acceptable at the data sizes this toolkit targets; `save_models=False` avoids model files for large runs.
- Validation scores estimate generalization under the declared split only; they are not leaderboard claims.
- A fixed seed repeats only for the same scikit-learn version and thread settings; the artifact records the versions.
