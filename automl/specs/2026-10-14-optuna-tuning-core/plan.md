# Plan: Optuna Tuning Core (Phase 10)

## 1. Module layout, errors, and optional dependency

1. Add `src/automl/tuning.py` for the tuner, search-space handling, result types, and persistence. Keep `baseline.py` as the fold-running path.
2. Add `TuningError` to `errors.py` (subclass of the existing workflow error base) and `TuningUnavailableError` for a missing Optuna install, with a message naming `optuna` and `pip install 'automl[tuning]'`.
3. Add the `tuning = ["optuna>=3.6"]` extra to `pyproject.toml`; core dependencies are unchanged. Set the minimum version from compatibility testing.
4. Import Optuna lazily inside functions only. Confirm `import automl`, `automl --help`, and existing tests are unaffected without the extra.

## 2. Configuration

1. Add a frozen `TuningConfig`: `estimator`, `n_trials`, `timeout_seconds`, `pruning` (default false), `seed`, `estimator_params` (fixed parameters), `native_categoricals`, `device`, `n_jobs`, plus the existing task, metric, and validation settings it reuses.
2. Validate in `__post_init__`: estimator is registered; at least one of `n_trials` and `timeout_seconds` is set and each is positive; metric and task are a supported pair; the metric has a defined direction.
3. Parse a `[tuning]` TOML section through the existing config loader with unknown-key and wrong-type errors that name the key.
4. Resolve the metric direction from the metric registry and expose it on the config/result; do not accept a user-supplied direction that contradicts the metric.

## 3. Search spaces

1. Define `SearchSpace` as either a mapping of parameter specs (`float`, `int`, `categorical`, with `low`, `high`, `log`, `step`, `choices`) or a callable `fn(trial) -> dict`.
2. Validate mapping specs eagerly (known type, valid bounds, log needs positive bounds, non-empty choices) with errors naming the parameter.
3. Add a minimal built-in `hist_gradient_boosting` space. For any other estimator with no supplied space, raise an error stating that a space is required until Phase 11.
4. Merge sampled parameters over `estimator_params` and validate the result through the Phase 9 rules, rejecting controlled keys (seed, threads, device, verbosity, early stopping) and naming the controlling setting. Reject a tuned key that is also fixed in `estimator_params`.

## 4. Objective and study execution

1. Build the objective from the Phase 8 fold runner: for each fold, fit preprocessing and estimator on train rows only, score on validation rows only, and return the mean fold score. Reuse the split artifact logic so every trial sees identical folds.
2. Create the study with `direction` from the metric, `TPESampler(seed=seed)`, and `MedianPruner` only when `pruning` is on; report the cumulative mean after each fold and raise `TrialPruned` when told.
3. Run `study.optimize` with `n_trials` and/or `timeout`, one trial at a time. Catch exceptions per trial (`catch=(Exception,)`) so a failing trial is recorded as failed with a short reason; the study continues.
4. Record per trial: number, state, params, score, fold scores, fold std, duration, and failure reason. Measure and report wall-clock overrun against `timeout_seconds`.
5. Define `TuningResult`: effective config, direction, best trial (or none), all trials, library versions, dataset/validation fingerprint, budget used, and notes on nondeterminism. Raise `TuningError` summarizing reasons if no trial completed.

## 5. Persistence, inspection, and resume

1. Persistence happens through the `directory=` argument of `tune` and `resume_tuning` (no separate `save_tuning`): write `tuning.json` (schema version 1) and keep `study.db` (SQLite storage) in the directory; refuse to overwrite an existing directory. Create the study directly against `sqlite:///<directory>/study.db` during `tune`, so progress survives a crash.
2. `load_tuning(directory)`: read `tuning.json` only, validate schema version and required fields with actionable errors, and return `TuningResult` without importing Optuna.
3. `resume_tuning(directory, data, space, n_trials=None, timeout_seconds=None)`: load the saved config, check the fingerprint (columns, target, row count, seed, validation settings, estimator, metric) against the supplied data, and continue the same Optuna study with the new budget; raise a `TuningError` naming any mismatch. Rewrite `tuning.json` atomically after resume.
4. Document that a callable search space is not serialized: the mapping form is stored in `tuning.json`, and a callable must be supplied again on resume.

## 6. Tests

1. Config tests without Optuna: validation errors, missing budget, bad metric direction conflicts, `[tuning]` TOML parsing and errors, and that `import automl` does not import `optuna`.
2. Absence test: with `optuna` import blocked, `tune` raises `TuningUnavailableError` naming the package and install command before any fitting, and the baseline path still works.
3. Per task (binary, multiclass, regression) with `hist_gradient_boosting` and the built-in space: a small study within a budget of a few trials; finite scores; best trial matches the extreme score for the metric direction; direction is correct for both maximize and minimize metrics.
4. Budget tests: `n_trials` stops at the count; `timeout_seconds` stops within one trial of the limit; both together stop at whichever comes first.
5. Reproducibility: same seed, data, and splits give identical trial parameters and scores on CPU; different seeds differ.
6. Search-space tests: each mapping type samples within bounds; invalid specs rejected; callable form works; controlled and duplicate keys rejected; non-HGB estimator without a space errors clearly. Per installed adapter (guarded by `pytest.importorskip`), a supplied tiny space runs a 2-trial study.
7. Failure handling: a trial that raises is recorded as failed with a reason and the study continues; all-failed raises `TuningError`; failed trials are never best.
8. Pruning: pruning off by default; with pruning on, a forced-bad trial is recorded as pruned with partial fold scores.
9. Leakage: preprocessing and scoring use only train/validation rows per fold in every trial; target and ID columns never reach the estimator; input data is not mutated.
10. Persistence: save/load round trip without Optuna; refuse overwrite; resume adds trials to the same study and keeps prior trials and the best-so-far; fingerprint mismatch errors; `tuning.json` is JSON-safe and no file is written outside the directory.

## 7. Documentation and roadmap

1. Update `automl/README.md` with the `tuning` extra and install command, `[tuning]` keys, the search-space forms and built-in space, budget and pruning behavior, direction handling, failure semantics, the `tuning.json` schema, resume rules, and reproducibility and selection-bias notes, with a Python example for people and agents (no CLI this phase).
2. Mark Phase 10 complete in `specs/roadmap.md` with a verification note once validation passes, including which extras were installed.
