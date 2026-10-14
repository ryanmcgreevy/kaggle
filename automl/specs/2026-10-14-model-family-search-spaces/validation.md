# Validation: Model-Family Search Spaces (Phase 11)

The feature is ready to merge when all checks pass on Python 3.12 or newer from the `automl/` project root, without competition data, Kaggle credentials, a GPU, or network access beyond installing the optional extras.

## Required Checks

1. **Core install, no extras**

   ```bash
   python -m pip install -e '.[test]'
   python -c 'import automl' && automl --help && python -m automl --help
   python -m pytest
   ```

   Pass when install adds no core dependency, all commands exit 0, and the suite passes with Optuna and boosting libraries absent (family tests skipped; registry, config, override, and schema tests passing).

2. **Extras install**

   ```bash
   python -m pip install -e '.[test,tuning,boosting]'
   python -m pip check
   ```

   Pass when the extras install cleanly and `pip check` reports no broken requirements.

3. **Test suite with extras**

   ```bash
   python -m pytest
   ```

   Pass when all tests succeed, including Phase 0-10 tests unmodified, and new tests cover:

   - registry lookup, `describe_search_space`, and unknown family errors;
   - a documented space for each of `hist_gradient_boosting`, `lightgbm`, `xgboost`, `catboost` on binary, multiclass, and regression, with sampled parameters accepted by the installed estimator and a 2-3 trial study producing finite scores;
   - controlled keys never sampled;
   - TOML range overrides respected, dropped/fixed parameters, and errors naming the key for unknown family/parameter, wrong type, invalid bounds, and controlled keys;
   - categorized failed trials (`invalid_parameter`, `library_error`, `timeout`, `non_finite_score`, `other`) recorded with a reason, never best, successful results intact;
   - `tune_families` with explicit families and budgets, default installed families, an error with install hint for a requested missing family, per-family isolation of failures, `TuningError` when all families fail;
   - comparison ranking correct for maximize and minimize metrics with the selection-bias note;
   - persistence layout, `tuning_families.json` round trip, no-overwrite, per-family resume, and loading schema 1 `tuning.json`.

4. **Optional-dependency isolation**

   Pass when `import automl` imports neither `optuna` nor any boosting library, registry and override validation work without them, and requesting an uninstalled family raises its install hint before any fitting.

5. **Phase 8-10 behavior unchanged**

   Pass when single-study `tune`, `resume_tuning`, and `load_tuning` behave as before, supplied mapping/callable spaces take precedence over registry spaces, and fold scores for fixed parameters still equal `run_baseline`.

6. **Leakage safety and reproducibility**

   Pass when tests confirm no preprocessing or encoding learns from validation rows in any family, target and ID columns never reach an estimator, input data is not mutated, and on CPU with a fixed seed two `tune_families` runs give identical sampled parameters and scores per family, with caveats (GPU, time-limited runs) recorded.

7. **Budgets and visibility**

   Pass when the budget plan is recorded before any trial runs, each family stops within its own budget (time overrun bounded by one trial and reported), and a family without a budget is rejected.

8. **Local-only artifacts**

   Pass when a run writes only inside the chosen directory, makes no network calls, stores no training data or secrets, and never overwrites an existing directory implicitly.

9. **Documentation**

   Pass when `automl/README.md` documents each family's space and range rationale, TOML overrides, `tune_families` usage and budgets, failure categories, artifact schemas, resume rules, the selection-bias note, and a Python example, and `specs/roadmap.md` records Phase 11 status and verification.

## Merge Criteria

All checks above pass, `git diff` contains no competition data or secrets, and the Phase 11 roadmap entry is updated.
