# Requirements: Optional MLflow Tracking (Phase 12)

## Scope

**Deliverable (from roadmap):** Tracking adapter that mirrors run configuration, metrics, trial history, and complete fitted pipelines to MLflow when explicitly enabled.

In scope:

- A `tracking` module with a `TrackingConfig` and tracking functions that mirror results of `run_baseline`, `tune`, and `tune_families` to MLflow.
- Explicit enablement only: a `tracking` argument on the Python API and a `[tracking]` TOML section. Without it, behavior is unchanged and MLflow is never imported.
- Mirroring of: effective configuration, data/validation summary (fingerprint, row/column counts, split settings, seed), per-fold and mean metrics, library versions, and nondeterminism notes.
- Tuning layout: one parent MLflow run per study (per family for `tune_families`) with a nested child run per trial (params, score, fold scores, state, duration, failure reason), plus the structured result JSON as an artifact.
- Complete fitted pipeline (preprocessing + estimator) logged as an MLflow model for the best trial, refit on all declared training rows. Baseline runs log the pipeline fit on all training rows likewise.
- Post-hoc logging from a saved result directory (`track_baseline`, `track_tuning`, `track_families` accept a result object or a saved directory), so runs can be mirrored without rerunning.
- Default store: local SQLite under the run directory; a user-supplied tracking URI overrides it. Credentials in URIs are redacted in all persisted artifacts and logs.
- Optional `mlflow` extra, imported lazily with an actionable install hint.
- Local metadata and artifacts remain the source of truth; tracking is a mirror.
- README documentation, `[tracking]` TOML, synthetic-fixture tests including a local-store integration test.

Out of scope:

- CLI flag or command (deferred to Phase 16).
- Logging EDA summaries, split artifacts, or reports to MLflow.
- Logging raw data, dataset samples, or prediction files.
- Hosted tracking servers as a tested target, authentication setup, model registry, and model serving.
- The final refit and prediction workflow (Phase 14); the pipeline refit here exists only for logging and is not the selected-model contract.
- Reports (Phase 13), agent skills (Phase 17), and other tracking backends.
- Autologging.

## Decisions

| Topic | Decision |
|---|---|
| Packaging | New optional extra `mlflow = ["mlflow>=3.0"]` (tested with 3.17.1); core dependencies unchanged. `import automl` never imports MLflow. Calling tracking without it raises `TrackingUnavailableError` naming `mlflow` and `pip install 'automl[mlflow]'` before any work. |
| Enablement | Explicit only: `tracking=TrackingConfig(...)` argument on the tracking functions and `[tracking]` TOML. `enabled` defaults to true (passing a `TrackingConfig` is the explicit opt-in; `enabled = false` turns it off); no environment-variable auto-enable. `run_baseline`, `tune`, `resume_tuning`, `tune_families`, and `resume_tuning_families` accept an optional `tracking` argument and, when given, mirror results after local work finishes. |
| Source of truth | Local artifacts are written first and unchanged. A tracking failure raises `TrackingError` but never deletes or alters local results, and the error message states the local result is intact. |
| Destination | Default tracking URI is `sqlite:///<directory>/mlflow.db` with artifacts under `<directory>/mlartifacts`, where `directory` is the run directory; if no directory exists, the user must supply `tracking_uri`. A supplied URI (file, sqlite, or http) overrides. Remote URIs are used only if the user configures them. |
| Experiment and run names | `experiment_name` (default `automl`), optional `run_name`, and optional string `tags`. Run name defaults to estimator/family plus seed. |
| Logged content | Params: effective config flattened, estimator params, seed, task, metric, direction, validation settings. Metrics: mean, std, per-fold scores, best score. Tags: library versions, schema version, nondeterminism notes. Artifacts: the result JSON (`baseline.json`, `tuning.json`, `tuning_families.json`). |
| Trial layout | Parent run per study (and per family under a parent for `tune_families`); nested child run per trial with state, params, score, fold scores, duration, and failure reason. Failed and pruned trials are logged with state tags and never marked best. The best trial is also recorded on the parent. |
| Fitted pipeline | Baseline: the preprocessing-plus-estimator pipeline fit on all training rows. Tuning: the best trial's params refit on all training rows. Logged with `mlflow.sklearn` and a signature; an input example is not logged to avoid exposing data. Skipped with a recorded note when a library lacks sklearn-flavor support, never silently. Refit requires the data again; post-hoc logging without data logs everything except the model and says so. |
| Post-hoc logging | `track_*` functions take a result object or a saved directory; they load through the existing `load_*` functions, needing neither Optuna nor the estimator libraries unless a model is also being logged. Logging the same saved result twice creates a second run unless `run_name` collides; document this rather than deduplicating. |
| Privacy | Never log training/test rows, sample submissions, predictions, or input examples. Redact `user:password@` in URIs; never log environment variables. Config values that look like credentials are not accepted as tags. |
| Data volume | Per-trial child runs capped by `max_trials_logged` (default 200, best trial always included); the full history is always in the JSON artifact. Truncation is recorded in the parent tags. |
| Failure semantics | MLflow import, connection, or write errors raise `TrackingError` with the URI (redacted) and cause. A partially written run is ended with status FAILED and a tag; it does not leave a RUNNING run. |
| Interface | Python API and TOML only. |

## Context

- `specs/mission.md`: local and private by default, with hosted destinations only when explicitly configured; inspectable by default; resource-aware; optional MLflow listed as an extension and not a prerequisite.
- `specs/tech-stack.md`: MLflow is an optional backend; local metadata stays sufficient to reproduce a run; when enabled, log effective configuration, data/validation summary, metrics, trial details, and the complete preprocessing-plus-model pipeline where supported; secrets out of artifacts and logs; no network or tracking server required for the local workflow.
- Phase 8 supplies `BaselineResult`, `save_baseline`/`load_baseline`, and `make_estimator` for the full pipeline. Phase 10 and 11 supply `TuningResult`, `FamiliesResult`, and their `load_*` functions and fingerprints. Existing error hierarchy lives in `errors.py`.

## Assumptions

- MLflow is not installed in the project `.venv` today; verification installs the `mlflow` extra and also runs the suite without it.
- A local SQLite store or file store is sufficient for integration tests; no server or network is required.
- MLflow's sklearn flavor can serialize the fitted preprocessing pipeline and scikit-learn/HistGradientBoosting; boosting adapters may need their own flavors or be recorded as skipped. This is confirmed during implementation and documented.
- Logging many trials is slow; `max_trials_logged` keeps the cost bounded.
- Logged validation scores are selection-biased and not leaderboard claims; this is stated in the README and tags.
