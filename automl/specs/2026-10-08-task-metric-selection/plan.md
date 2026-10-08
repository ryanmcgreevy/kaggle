# Plan: Task and Metric Selection (Phase 2)

## 1. Dependencies and module layout

1. Add `scikit-learn` to core `dependencies` in `pyproject.toml`.
2. Add `src/automl/task.py` (task types, suggestion, resolution) and `src/automl/metrics.py` (registry and built-ins).
3. Confirm `import automl` and `automl --help` still work.

## 2. Errors

1. Add `TaskMetricError` (subclass of or sibling to `DataContractError`, same multi-problem message style) naming the setting, task, or metric involved.
2. Ensure ambiguity errors carry the suggestion and rationale in the message.

## 3. Task types and config

1. Define a `Task` enum/literal: `binary`, `multiclass`, `regression`.
2. Define a frozen `TaskConfig` dataclass: `task` and `metric`, both optional.
3. Extend `load_config` to accept an optional `[task]` table; reject unknown keys and non-string values; support Python overrides over file values. Keep config version 1.
4. Expose `task_config` on the loaded config without changing existing `DataConfig` behavior.

## 4. Metric registry

1. Define a `Metric` dataclass: name, supported tasks, direction, input kind (labels or probabilities/scores), constraints, and callable.
2. Register built-ins using scikit-learn: binary `roc_auc`, `log_loss`, `accuracy`, `f1`; multiclass `accuracy`, `log_loss`, `macro_f1`, `balanced_accuracy`; regression `rmse`, `mae`, `r2`, `rmsle`.
3. Implement `register_metric(...)`, `get_metric(name)`, and `list_metrics(task=None)`; reject duplicate names, unsupported tasks, and invalid directions.
4. Enforce `rmsle` value constraints with a clear error.

## 5. Suggestions

1. Implement `suggest_task(target: pd.Series) -> Suggestion` returning task, rationale, evidence (dtype, unique count, null count), and a confidence flag; no side effects.
2. Implement `suggest_metric(task) -> Suggestion` returning the default (`roc_auc` / `log_loss` / `rmse`) and alternatives.

## 6. Resolution and validation

1. Implement `resolve_task_metric(bundle_or_target, task_config) -> ResolvedTask` that:
   - errors with a suggestion when task is unset;
   - errors with a suggestion and valid choices when metric is unset;
   - errors on unknown metric, or metric unsupported for the task;
   - checks the explicit task against the train target (class counts, numeric dtype, nulls).
2. Collect all detectable problems in one pass where practical.
3. Return the effective task, metric object, and direction for later phases.

## 7. Tests

1. Add synthetic target fixtures to `tests/conftest.py`.
2. Cover: all valid task/metric combinations; unset task and unset metric (error contains suggestion); unknown metric; metric/task mismatch; binary target with 3 classes; multiclass target with 2 classes; regression on string target; null target; ambiguous integer target; `rmsle` with negatives; custom metric registration, duplicate name, and use in resolution; `[task]` TOML parsing, unknown key, override precedence; existing Phase 0 and 1 tests unchanged.
3. Assert error messages name the setting, task, or metric.

## 8. Public API export

1. Export `Task`, `TaskConfig`, `Metric`, `register_metric`, `get_metric`, `list_metrics`, `suggest_task`, `suggest_metric`, `resolve_task_metric`, and `TaskMetricError` from `automl`.

## 9. Documentation and roadmap

1. Update `automl/README.md` with the `[task]` schema, supported metrics table, suggestion behavior, and a custom-metric example.
2. Mark Phase 2 complete in `specs/roadmap.md` with a verification note once validation passes.
