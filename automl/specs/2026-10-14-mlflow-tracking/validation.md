# Validation: Optional MLflow Tracking (Phase 12)

The feature is ready to merge when all checks pass on Python 3.12 or newer from the `automl/` project root, without competition data, Kaggle credentials, a GPU, a hosted tracking service, or network access beyond installing the optional extras.

## Required Checks

1. **Core install, no extras**

   ```bash
   python -m pip install -e '.[test]'
   python -c 'import automl' && automl --help && python -m automl --help
   python -m pytest
   ```

   Pass when install succeeds with no new core dependency, all commands exit 0, and the suite passes with MLflow absent (tracking integration tests skipped; config, redaction, and absence tests passing).

2. **Extra install**

   ```bash
   python -m pip install -e '.[test,mlflow]'
   python -m pip install -e '.[test,mlflow,tuning,boosting]'
   python -m pip check
   ```

   Pass when `.[mlflow]` installs cleanly alone and combined with `tuning` and `boosting`, and `pip check` reports no broken requirements.

3. **Test suite with the extras**

   ```bash
   python -m pytest
   ```

   Pass when all tests succeed, including Phase 0 to 11 tests unmodified and new tests covering:

   - `TrackingConfig` validation and `[tracking]` TOML parsing with errors naming the key;
   - destination resolution (default SQLite under the run directory, supplied URI override, error when neither is available) and URI credential redaction;
   - baseline, tuning, and families runs logged to a local store and queried back, with params, metrics, tags, nested trial runs, and artifacts matching the saved JSON results;
   - trial cap behavior with the best trial always logged and truncation recorded;
   - model logging: the complete preprocessing-plus-estimator pipeline reloads from MLflow and reproduces predictions, with documented skips and recorded notes for unsupported libraries and for post-hoc logging without data;
   - post-hoc logging from saved directories;
   - failure handling that leaves local results intact and no RUNNING runs behind.

4. **Optional-dependency isolation**

   Pass when tests confirm `import automl` does not import `mlflow`, and that with its import blocked, tracking raises `TrackingUnavailableError` naming `mlflow` and `pip install 'automl[mlflow]'` before any work, while `run_baseline`, `tune`, and `load_*` work without a `tracking` argument.

5. **Behavior unchanged without tracking**

   Pass when Phase 8 to 11 results, artifacts, and schemas are byte-for-byte unchanged when `tracking` is not passed, and local artifacts are identical whether or not tracking is enabled.

6. **Local files remain the source of truth**

   Pass when a tracking error never modifies or deletes local results, the error states local results are intact, and local artifacts alone are sufficient to reload and inspect any run without MLflow.

7. **Privacy and secrets**

   Pass when tests confirm no training or test rows, sample submissions, predictions, input examples, or environment variables are logged, credentials in URIs are redacted from artifacts, tags, logs, and error messages, and credential-like tags are rejected.

8. **Local-only operation**

   Pass when the default tracking path writes only inside the chosen run directory (`mlflow.db`, `mlartifacts`), tests make no network calls, and no hosted service is contacted unless the user supplies a remote URI.

9. **Integration against a local tracking URI**

   Pass when a run is logged to a local SQLite tracking URI and queried back through the MLflow client, as the roadmap acceptance requires, for baseline and tuning, in a fresh process after the logging process ends.

10. **Documentation**

    Pass when `automl/README.md` documents the extra, `[tracking]` keys, destinations, run/trial layout, logged and unlogged content, post-hoc usage, per-library model logging limits, privacy and redaction, the selection-bias note, and Python invocation examples, and `specs/roadmap.md` records Phase 12 verification.
