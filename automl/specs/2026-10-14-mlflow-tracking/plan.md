# Plan: Optional MLflow Tracking (Phase 12)

## 1. Module layout, errors, and optional dependency

1. Add `src/automl/tracking.py` for config, URI handling, logging functions, and model logging. Keep `baseline.py`, `tuning.py`, and `families.py` free of MLflow imports.
2. Add `TrackingError` and `TrackingUnavailableError` to `errors.py`; the latter names `mlflow` and `pip install 'automl[mlflow]'`.
3. Add `mlflow = ["mlflow>=2.14"]` to `pyproject.toml` optional dependencies; core dependencies unchanged. Set the minimum from compatibility testing.
4. Import MLflow lazily inside functions only. Confirm `import automl`, `automl --help`, and existing tests are unaffected without the extra.

## 2. Configuration and URI handling

1. Add a frozen `TrackingConfig`: `enabled` (default false), `tracking_uri`, `experiment_name` (default `automl`), `run_name`, `tags`, `log_model` (default true), `max_trials_logged` (default 200).
2. Validate in `__post_init__`: positive `max_trials_logged`, string-only tags, non-empty experiment name, and tags that look like credentials rejected.
3. Parse a `[tracking]` TOML section through the config loader with unknown-key and wrong-type errors that name the key.
4. Resolve the destination: supplied URI wins; otherwise `sqlite:///<directory>/mlflow.db` with artifacts in `<directory>/mlartifacts`; with neither, raise an error saying a directory or URI is required. Add a redaction helper for `user:password@` and use it everywhere a URI is stored or shown.

## 3. Baseline tracking

1. `track_baseline(result_or_directory, tracking, data=None)`: start a run in the experiment, log flattened config as params, per-fold and mean metrics, tags (versions, schema, notes), and `baseline.json` as an artifact.
2. When `data` and `log_model` are supplied, fit the Phase 8 pipeline on all training rows and log it with `mlflow.sklearn` and an inferred signature, no input example. Without data, record a tag stating the model was not logged and why.
3. Load saved directories through `load_baseline`; no estimator libraries needed unless a model is logged.
4. Add an optional `tracking` argument to `run_baseline` that calls this after local results exist; a tracking error leaves local results untouched and says so.

## 4. Tuning and families tracking

1. `track_tuning(result_or_directory, tracking, data=None)`: a parent run with config, direction, budget used, best trial, and `tuning.json`; a nested child run per trial with params, score, fold scores, state tag, duration, and failure reason. Cap children at `max_trials_logged`, always include the best trial, and record truncation on the parent.
2. `track_families(result_or_directory, tracking, data=None)`: a top-level run with the family comparison and ranking and `tuning_families.json`; one nested parent per family with the same per-trial layout.
3. When `data` and `log_model` are supplied, refit the best trial's parameters (best family for families) on all training rows through the same pipeline construction and log it. Without data, record a skipped-model tag. If a library has no usable flavor, log a note naming the library; never fail silently.
4. Add an optional `tracking` argument to `tune`, `resume_tuning`, `tune_families`, and `resume_tuning_families`. Resume logs a new run for the added trials and states in a tag which saved study it continues.
5. Ensure a failed write ends the active run with status FAILED and a tag, never leaving RUNNING runs.

## 5. Tests

1. Config tests without MLflow: validation errors, credential-like tags, URI redaction, default-destination resolution, `[tracking]` TOML parsing and errors, and `import automl` not importing `mlflow`.
2. Absence test: with `mlflow` import blocked, tracking raises `TrackingUnavailableError` naming the package and install command before any work, while `run_baseline`, `tune`, and `load_*` still work.
3. Baseline integration (local SQLite store, guarded by `pytest.importorskip`): a run is logged and queried back with matching params, mean metric, per-fold metrics, tags, and artifact content equal to `baseline.json`; the logged model reloads and reproduces predictions from the same fitted pipeline.
4. Tuning integration: parent and nested child runs queryable; child count, params, scores, and states match `tuning.json`; failed and pruned trials tagged and not best; truncation behavior at a small `max_trials_logged`; best trial always logged.
5. Families integration: top-level comparison, one nested parent per family, ranking tags match the saved result.
6. Post-hoc: logging from a saved directory without Optuna or estimator libraries installed (where possible) produces the same logged metrics; the model is skipped with a recorded note when no data is supplied.
7. Privacy: no training or test rows, predictions, input examples, or environment variables in logged params, tags, or artifacts; a URI with credentials is redacted in `tuning.json`, tags, and error messages.
8. Failure handling: unreachable URI raises `TrackingError` with a redacted URI and states local results are intact; a write failure mid-run leaves no RUNNING run; local files are unchanged after any tracking error.
9. Local-only: default tracking writes only inside the run directory; no network calls in tests; existing Phase 8 to 11 tests unchanged and passing.

## 6. Documentation and roadmap

1. Update `automl/README.md` with the `mlflow` extra and install command, `[tracking]` keys, default and custom destinations, the run/trial layout, what is and is not logged, post-hoc usage, model logging limits per library, redaction and privacy notes, selection-bias note, and Python examples for people and agents (no CLI this phase).
2. Mark Phase 12 complete in `specs/roadmap.md` with a verification note once validation passes, including which extras were installed.
