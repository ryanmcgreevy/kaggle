# Requirements: Model-Family Search Spaces (Phase 11)

## Scope

**Deliverable (from roadmap):** Separate, documented Optuna search spaces for each supported estimator; users can choose families and budgets explicitly.

In scope:

- A search-space registry with one documented, task-aware space per family: `hist_gradient_boosting`, `lightgbm`, `xgboost`, `catboost`.
- Each space is a Python callable `fn(trial, task) -> dict[str, Any]` registered by family, resolved by `tune` when no space is supplied (replacing the Phase 10 HGB-only built-in and its "space required until Phase 11" error).
- TOML overrides of parameter ranges per family (`[tuning.spaces.<family>.<param>]`) that adjust bounds, `log`, `step`, or `choices` of an existing parameter, or remove it from the search. Overrides cannot add controlled keys.
- A multi-family `tune_families` API: one Optuna study per family, per-family trial/time budget (with a shared default), and a comparison summary across families.
- Categorized failed-trial reasons in each `TrialRecord`: `invalid_parameter`, `library_error`, `timeout`, `non_finite_score`, `other`, plus the short message.
- Persistence of a multi-family run: one Phase 10 directory per family under a parent run directory, plus a parent `tuning_families.json` summary.
- `[tuning]` TOML additions (`families`, per-family budgets, spaces overrides), README documentation, synthetic-fixture tests.

Out of scope:

- CLI command (deferred to Phase 16; Python API and TOML only).
- Tuning preprocessing settings, validation strategy, or several estimators in one study.
- Ensembling or blending of families, final refit and prediction (Phase 14), MLflow (Phase 12), reports (Phase 13), agent skill updates (Phase 17).
- Parallel studies across processes, remote storage, samplers other than seeded TPE.
- Per-competition or per-dataset-size adaptive spaces beyond task type.
- GPU-specific parameters; device handling remains the Phase 9 rule.

## Decisions

| Topic | Decision |
|---|---|
| Space format | Python callables in a registry keyed by family, taking `(trial, task)` so binary, multiclass, and regression can differ (for example objective-specific params). Phase 10 mapping and callable spaces still work and take precedence when supplied. |
| TOML overrides | Range overrides only: `low`, `high`, `log`, `step`, `choices`, or `fixed = value`/`enabled = false` to drop a parameter. Unknown family or parameter, wrong type, or invalid bounds raise an error naming the key. Overrides are applied by wrapping the registered space; the effective space is recorded in the result. |
| Space contents | Each family documents its parameters, ranges, log scaling, and the reason for each range. Parameters are limited to those accepted by the Phase 9 estimator rules; controlled keys (seed, threads, device, verbosity, early stopping) are never sampled. |
| Family selection | `tune_families(..., families=[...], budgets=...)`. `families` defaults to all families whose library is installed; an explicitly requested but missing family raises the existing install hint instead of being skipped silently. |
| Budgets | Per-family `n_trials` and/or `timeout_seconds` with a shared default. Each family needs at least one budget. Total time is the sum; the plan records the budgets before work starts so expensive work is visible. |
| Studies | One independent study per family reusing Phase 10 `tune`, seed, splits, and direction. A failure in one family (including all trials failing) is recorded and does not abort the others unless every family fails, which raises `TuningError`. |
| Comparison | `FamilyComparison` lists per-family best score, fold std, trial counts by state, duration, and best params, ranked by metric direction. It states that scores are selection-biased validation estimates, not leaderboard claims. |
| Failure reasons | `TrialRecord` gains `failure_category` (schema bump to 2; schema 1 files still load with category `other`). Categories come from exception type/message mapping, kept small and documented. |
| Persistence | `directory/<family>/` holds `study.db` and `tuning.json` exactly as Phase 10; `directory/tuning_families.json` (schema 1) holds config, budgets, comparison. No overwrite of an existing directory; resume per family reuses Phase 10 `resume_tuning`. |
| Interface | Python API and TOML only. |
| Data safety | Local only, no network; artifacts hold parameters and scores, not data. |

## Context

- `specs/mission.md`: inspectable by default, reproducible seeds, resource-aware (budgets visible before expensive work), extensible without hiding model selection behind one unexplained score.
- `specs/tech-stack.md`: Optuna seeded, budgeted, persisted; boosting libraries optional with clear install messages; preprocessing stays inside CV folds; stable documented artifact schemas.
- Phase 9 provides the estimator registry, parameter rules, and device handling. Phase 10 provides `tune`, `resume_tuning`, `load_tuning`, `TuningConfig`, `TrialRecord`, the mapping/callable space interface, and `tuning.json` schema 1.

## Assumptions

- LightGBM, XGBoost, CatBoost, and Optuna are installable via the `boosting` and `tuning` extras; tests skip per family when absent, and registry/override/config tests run without them.
- Ranges are conventional starting points, not claimed optimal; they are documented and overridable.
- Small synthetic datasets with 2-3 trials per family verify validity; scores are asserted only to be finite and correctly directed.
- Sequential studies in one process are sufficient.
