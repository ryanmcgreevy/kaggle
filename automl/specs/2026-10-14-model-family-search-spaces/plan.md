# Plan: Model-Family Search Spaces (Phase 11)

## 1. Registry and module layout

1. Add `src/automl/search_spaces.py` with a registry mapping family name to a documented `fn(trial, task) -> dict[str, Any]`, plus `get_search_space(family, task, overrides=None)` and `describe_search_space(family, task)` returning parameter names, types, ranges, and log flags.
2. Move the Phase 10 built-in HGB space into the registry; keep behavior for existing callers.
3. Keep `tuning.py` as the single-study engine. Add `tune_families` in a new `src/automl/families.py` so single-study code is unchanged.
4. Import boosting libraries lazily; registry lookup and `describe_search_space` must work without them installed.

## 2. Per-family search spaces

1. `hist_gradient_boosting`: learning rate (log), max iterations, max leaf nodes, min samples leaf, L2 regularization, max depth.
2. `lightgbm`: learning rate (log), n_estimators, num_leaves, max_depth, min_child_samples, subsample, colsample_bytree, reg_alpha and reg_lambda (log).
3. `xgboost`: learning rate (log), n_estimators, max_depth, min_child_weight, subsample, colsample_bytree, gamma, reg_alpha and reg_lambda (log).
4. `catboost`: learning rate (log), iterations, depth, l2_leaf_reg (log), random_strength, bagging_temperature (Bayesian bootstrap only).
5. Make spaces task-aware only where needed (for example class-imbalance or objective-specific parameters); document each parameter and range rationale in code docstrings and the README.
6. Validate every sampled set through the Phase 9 estimator rules; never sample controlled keys.

## 3. Configuration and TOML overrides

1. Extend `TuningConfig` (or a new frozen `FamiliesConfig`) with `families`, per-family budgets and a shared default, and `spaces` overrides.
2. Parse `[tuning]` additions and `[tuning.spaces.<family>.<param>]` with errors naming the key for unknown family/parameter, wrong types, invalid bounds, log with non-positive bounds, empty choices, and attempts to override controlled keys.
3. Apply overrides by wrapping the registered space; support range changes and `enabled = false`/`fixed` to drop a parameter. Record the effective space in the result.
4. Validate at config time that each selected family has a budget and a supported task/metric pair.

## 4. Failure categorization

1. Add `failure_category` to `TrialRecord` with values `invalid_parameter`, `library_error`, `timeout`, `non_finite_score`, `other`, derived from the exception type and message; keep the short reason text.
2. Bump `tuning.json` to schema 2; `load_tuning` still reads schema 1 and fills `other`. Update the writer and result serialization.
3. A failed trial never becomes the best trial and does not corrupt successful results.

## 5. Multi-family tuning

1. Implement `tune_families(train, target, ..., families, budgets, directory=None)`: resolve installed families, fail early with the install hint for a requested missing one, record the budget plan, then run one `tune` per family with the shared seed and identical splits.
2. A family whose trials all fail is recorded as failed with its reasons; raise `TuningError` only if every family fails.
3. Build `FamilyComparison` ranked by metric direction with best score, fold std, trial counts by state, duration, best params, and a selection-bias note.
4. Persist `directory/<family>/` Phase 10 artifacts plus `tuning_families.json` (schema 1); refuse to overwrite; provide `load_tuning_families(directory)` without Optuna and `resume_tuning_families(...)` delegating to `resume_tuning`.

## 6. Tests

1. Without Optuna or boosting libraries: registry lookup, `describe_search_space`, config and TOML override validation, controlled-key rejection, `import automl` isolation, schema 1 loading.
2. Per family and task (guarded by `pytest.importorskip`): sampled parameters validate against the installed estimator and a 2-3 trial study completes with finite scores.
3. Overrides: narrowed ranges are respected in sampled values, dropped/fixed parameters behave, invalid overrides error with the key named.
4. Failure handling: injected invalid-parameter, library-error, non-finite-score, and timeout cases recorded with the right category; successful trials intact; all-failed family reported; all families failed raises.
5. `tune_families`: explicit families and budgets, default installed families, missing requested family error, comparison ranking for maximize and minimize metrics, identical splits across families, reproducibility for a fixed seed.
6. Persistence: directory layout, schema round trip without Optuna, no-overwrite, resume per family, nothing written outside the directory.
7. Regression: Phase 8-10 tests pass unmodified, and Phase 10 supplied mapping/callable spaces still take precedence.

## 7. Documentation and roadmap

1. Update `automl/README.md` with each family's space table, TOML overrides, `tune_families` usage, budgets, failure categories, artifact schemas, and a Python example for people and agents (no CLI this phase).
2. Update `specs/roadmap.md` Phase 11 status and verification after validation passes.
