# Validation: Optuna Tuning Core (Phase 10)

The feature is ready to merge when all checks pass on Python 3.12 or newer from the `automl/` project root, without competition data, Kaggle credentials, a GPU, or network access beyond installing the optional extras.

## Required Checks

1. **Core install, no extras**

   ```bash
   python -m pip install -e '.[test]'
   python -c 'import automl' && automl --help && python -m automl --help
   python -m pytest
   ```

   Pass when install succeeds with no new core dependency, all commands exit 0, and the suite passes with Optuna absent (tuning tests skipped, absence and config tests passing).

2. **Extra install**

   ```bash
   python -m pip install -e '.[test,tuning]'
   python -m pip check
   ```

   Pass when `.[tuning]` installs cleanly (alone and combined with `.[boosting]`) and `pip check` reports no broken requirements.

3. **Test suite with the extra**

   ```bash
   python -m pytest
   ```

   Pass when all tests succeed, including Phase 0 to 9 tests and new tests covering:

   - `TuningConfig` validation (estimator, budget, metric/task, direction) and `[tuning]` TOML parsing with errors naming the key;
   - a small study per task (binary, multiclass, regression) with finite scores and the best trial correct for both a maximize and a minimize metric;
   - budget behavior for `n_trials`, `timeout_seconds`, and both together;
   - search-space forms (mapping and callable), bound/validity errors, controlled-key and duplicate-key rejection, and a clear error for non-HGB estimators without a space;
   - failed trials recorded with reasons without aborting the study, never selected as best, and a summarized error when all fail;
   - pruning off by default and pruned trials recorded when enabled;
   - persistence: `tuning.json` round trip, no-overwrite, JSON-safe output, resume continuing the same study, fingerprint mismatch errors.

4. **Optional-dependency isolation**

   Pass when tests confirm `import automl` does not import `optuna`, and that with its import blocked, `tune` raises `TuningUnavailableError` naming `optuna` and `pip install 'automl[tuning]'` before any fitting, while `run_baseline` and `load_tuning` still work.

5. **Phase 8 and 9 behavior unchanged**

   Pass when all Phase 8 and Phase 9 tests pass unmodified, and the tuner's fold scores for fixed parameters equal the `run_baseline` fold scores for the same configuration.

6. **Leakage safety**

   Pass when tests confirm preprocessing and native categorical encoding are fit on train rows only in every trial and fold, scores use validation rows only, target and ID columns never reach the estimator, and input data is not mutated.

7. **Reproducibility**

   Pass when tests confirm that, on CPU with a fixed seed, data, splits, and library versions, two runs produce identical trial parameters and scores, different seeds produce different samples, and nondeterminism caveats (GPU, time-limited runs) are recorded in the result.

8. **Local-only artifacts**

   Pass when a tuning run writes only inside the chosen directory (`study.db`, `tuning.json`), makes no network calls, does not store training data or secrets, and an existing directory is never overwritten implicitly.

9. **Resume and inspection**

   Pass when a saved study can be loaded without Optuna, resumed in a fresh process with a new budget to add trials to the same study while retaining prior trials, and a mismatched dataset or configuration is rejected with an error naming the difference.

10. **Documentation**

    Pass when `automl/README.md` documents the extra, `[tuning]` keys, search-space forms, budgets, pruning, failure semantics, artifact schema, resume rules, selection-bias note, and a Python invocation example, and `specs/roadmap.md` records Phase 10 verification.
