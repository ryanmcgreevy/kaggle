# Requirements: Task and Metric Selection (Phase 2)

## Scope

**Deliverable (from roadmap):** Explicit classification/regression task and metric configuration, with safe suggestions where task type or metric is ambiguous.

In scope:

- Task types: `binary`, `multiclass`, `regression`.
- Optional `[task]` TOML table (`task`, `metric`) in config version 1, exposed on the Python API.
- Metric registry with built-in metrics and a public `register_metric` for user-supplied callables.
- Suggestion functions that propose a task from the train target and a default metric for a task.
- Validation of task/metric/target compatibility with actionable errors.
- Synthetic-fixture tests and README documentation.

Out of scope:

- CLI command (deferred to Phase 15; Python API only).
- Validation splits (Phase 3), EDA (Phase 4), model training or scoring pipelines (Phase 7+).
- Task types beyond the three above (multilabel, ranking, ordinal).
- Persisting custom metric callables in config files (callables are registered in Python only).

## Decisions

| Topic | Decision |
|---|---|
| Task types | `binary`, `multiclass`, `regression`. |
| Built-in metrics | binary: `roc_auc`, `log_loss`, `accuracy`, `f1`; multiclass: `accuracy`, `log_loss`, `macro_f1`, `balanced_accuracy`; regression: `rmse`, `mae`, `r2`, `rmsle`. |
| Config location | New optional `[task]` table in config version 1 (no version bump); keys `task` and `metric`. Python overrides take precedence. |
| Task not set | Never guessed. Resolution raises an error that includes a suggestion and the evidence (dtype, unique count) and asks the user to confirm. |
| Metric not set | Never silently defaulted. Resolution raises an error that includes the suggested default (`roc_auc` / `log_loss` / `rmse`) and the valid choices. |
| Suggestions | Separate, side-effect-free functions returning a suggestion plus rationale; they do not modify config or data. |
| Task/target consistency | Explicit task is checked against the target: binary needs exactly 2 distinct non-null values; multiclass needs 3 or more; regression needs a numeric target; null targets are an error. |
| Metric metadata | Each metric records its name, supported tasks, direction (maximize/minimize), whether it needs probabilities or labels, and any value constraints (e.g. `rmsle` needs non-negative values). |
| Custom metrics | `register_metric(name, fn, tasks, direction, needs)` adds to a registry; names cannot overwrite built-ins unless explicitly allowed; unknown names error. |
| Interface | Python API only. |
| Dependencies | `scikit-learn` added to core for metric implementations (planned core dependency in tech-stack); no optional ML packages. |

## Context

- `specs/mission.md`: conservative inference, expose uncertainty, require confirmation for ambiguous choices, validated outputs.
- `specs/tech-stack.md`: scikit-learn metrics, TOML config, actionable errors naming the setting, small synthetic pytest fixtures.
- Phase 1 provides `DataConfig`, `load_config`, `load_data`, `DataBundle`, and `DataContractError`; this phase reuses its error style and config loader.

## Assumptions

- The target is read from `DataBundle.train`; task suggestion uses only train data.
- Integer-valued targets with few unique values are ambiguous (could be multiclass or regression), so they are suggested with a stated heuristic but never auto-selected.
- Directions are recorded now so Phase 9 (Optuna) can use them without changes.
