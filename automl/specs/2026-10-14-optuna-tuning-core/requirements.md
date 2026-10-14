# Requirements: Optuna Tuning Core (Phase 10)

## Scope

**Deliverable (from roadmap):** A reusable tuner for one baseline/boosting estimator, with seeded sampler, explicit metric direction, trial/time budget, CV objective, and persisted trial results.

In scope:

- A `tune` capability that tunes exactly one registered estimator (`hist_gradient_boosting`, `lightgbm`, `xgboost`, `catboost`) for the configured task, metric, and validation strategy.
- CV objective built on the Phase 8 fold-running path, so splitting, leakage-safe preprocessing, and scoring behave exactly as in `run_baseline`.
- Explicit metric direction derived from the metric registry (maximize or minimize), recorded in the result.
- Seeded TPE sampler; the seed comes from the shared run configuration.
- Budget: `n_trials` and/or `timeout_seconds`; at least one is required, and either or both may be set.
- Optional pruning (off by default) using Optuna's median pruner with fold-level intermediate reports.
- Caller-supplied search space (a mapping of parameter specs or a callable taking an Optuna `trial`), plus one minimal built-in space for `hist_gradient_boosting` so an end-to-end run works without Phase 11.
- Local persistence in the run directory: an Optuna SQLite study plus a JSON result (effective config, best trial, all trials with params, scores, fold scores, state, duration, failure reason).
- Resume from a saved study directory; inspect a saved result without Optuna installed.
- Optional `tuning` extra (`optuna`), imported lazily with an actionable install hint.
- `[tuning]` TOML section and Python API; README documentation; synthetic-fixture tests.

Out of scope:

- CLI command (deferred to Phase 16; Python API and TOML only).
- Per-family documented search spaces and family selection (Phase 11).
- Final refit of best parameters and prediction (Phase 14).
- MLflow logging (Phase 12), reports (Phase 13), agent skill updates (Phase 17).
- Multi-objective studies, distributed or remote storage, parallel trials across processes, and samplers other than seeded TPE.
- Tuning preprocessing settings, validation strategy, or several estimators in one study.
- Early stopping inside estimators, sample weights, and custom objectives.

## Decisions

| Topic | Decision |
|---|---|
| Optuna packaging | New optional extra `tuning = ["optuna>=3.6"]`; core dependencies unchanged. `import automl` never imports Optuna. Calling the tuner without it raises an error naming the package and `pip install 'automl[tuning]'` before any work. |
| Estimator | One estimator per study, chosen with `estimator` in `[tuning]`; reuses the Phase 9 registry, defaults, controlled-parameter rules, device handling, and native categoricals. |
| Objective | Mean validation score across folds (or the single holdout score) from the same fold runner as `run_baseline`; no out-of-fold pooling. Fold scores and std are stored per trial. |
| Direction | Taken from the metric spec, never guessed; result records `direction`. Reuses `BaselineError`-style actionable errors for unsupported combinations. |
| Sampler and seed | `TPESampler(seed=seed)`; estimator seed from the same `seed`. CPU runs with one job repeat for a fixed seed, splits, and library versions; GPU runs and time-limited runs are documented as possibly non-repeatable. |
| Budget | `n_trials` and `timeout_seconds` both optional, at least one required, both validated positive. Time budget is checked between trials; a running trial is not interrupted, so overrun is bounded by one trial and reported. |
| Pruning | `pruning = false` by default. When on, `MedianPruner` uses cumulative fold-mean scores reported after each fold; pruned trials are recorded with state `pruned` and the folds completed. |
| Search space | Mapping form: `{"name": {"type": "float"\|"int"\|"categorical", "low", "high", "log", "choices", "step"}}`. Callable form: `fn(trial) -> dict[str, Any]`. Parameters are validated through the Phase 9 estimator rules, so controlled keys are rejected naming the controlling setting. Fixed (non-tuned) parameters come from `estimator_params`. |
| Built-in space | A small `hist_gradient_boosting` space (learning rate, max iterations, max leaf nodes, min samples leaf, L2 regularization) used only when no space is supplied for that estimator; other estimators require a supplied space until Phase 11. |
| Failed trials | A trial that raises is recorded with state `failed` and a short reason, does not abort the study, and never becomes the best trial. If every trial fails, `tune` raises an error summarizing the reasons. |
| Persistence | `save_tuning(result, directory)` writes `study.db` (Optuna SQLite storage), `tuning.json` (schema version 1), and refuses to overwrite an existing directory. Study name and storage URL are derived from the directory, with no network or credentials. |
| Resume | `resume_tuning(directory, ...)` loads the saved config, verifies the data/validation fingerprint (row count, columns, target, seed, fold settings) matches, and continues the same study with a new budget; mismatches raise an error naming the difference. Total trial count includes prior trials. |
| Inspection | `load_tuning(directory)` reads `tuning.json` only (no Optuna needed) and returns a structured `TuningResult`. |
| Interface | Python API and TOML only. |
| Data safety | No data, models, or study files leave the run directory; no network access. Study files hold parameters and scores, not training data. |

## Context

- `specs/mission.md`: reproducible with explicit seeds and recorded nondeterminism; resource-aware (trial/time budgets visible before expensive work); inspectable by default; local and private; Optuna search seeded and persisted.
- `specs/tech-stack.md`: Optuna for seeded, budgeted search with persisted study configuration and trial results for inspection or continuation; optional dependency groups with a clear install message; keep preprocessing inside CV folds; stage outputs consumable without hidden in-memory state; do not overwrite prior runs implicitly.
- Phase 8 provides `run_baseline`, `BaselineConfig`, the fold runner (`_run_fold`), and the result directory. Phase 9 provides the estimator registry, `estimator_params` validation, device resolution, and native categoricals. Phase 11 will supply per-family search spaces through the same space interface defined here.

## Assumptions

- Optuna is not installed in the project `.venv` today; verification installs the `tuning` extra there and also runs the suite without it.
- SQLite storage via Optuna's RDB storage is available in the Python standard library build.
- Trials run sequentially in one process (`n_jobs` applies to the estimator, not to parallel trials).
- Small synthetic datasets with tight trial and time budgets are enough to verify behavior; scores themselves are not asserted beyond being finite and correctly directed.
- Tuned scores are validation estimates optimistically biased by selection; reports and later phases must not present them as leaderboard claims.
