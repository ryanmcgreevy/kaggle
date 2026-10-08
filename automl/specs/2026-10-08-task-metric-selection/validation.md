# Validation: Task and Metric Selection (Phase 2)

The feature is ready to merge when all checks pass on Python 3.12 or newer from the `automl/` project root, without competition data, Kaggle credentials, network access (after install), or optional ML dependencies.

## Required Checks

1. **Install**

   ```bash
   python -m pip install -e '.[test]'
   ```

   Pass when install succeeds and `scikit-learn` is the only new runtime dependency.

2. **Existing behavior intact**

   ```bash
   python -c 'import automl' && automl --help && python -m automl --help
   ```

   Pass when all exit 0.

3. **Test suite**

   ```bash
   python -m pytest
   ```

   Pass when all tests succeed, including Phase 0 and Phase 1 tests and new tests covering:

   - every built-in task/metric combination resolving and computing a finite value on synthetic data;
   - unset task and unset metric raising errors that include a suggestion and valid choices;
   - unknown metric and metric/task mismatch;
   - explicit task inconsistent with target (binary with 3 classes, multiclass with 2, regression on non-numeric, null target);
   - ambiguous integer target surfaced as a suggestion, never auto-selected;
   - `rmsle` rejecting negative values;
   - custom metric registration, duplicate-name rejection, and use through resolution;
   - `[task]` TOML parsing, unknown key rejection, and override precedence.

4. **No silent guessing**

   Pass when a test confirms that `resolve_task_metric` never returns a result when task or metric is unset.

5. **Error quality**

   Pass when each invalid-input test asserts the message names the offending setting, task, or metric.

6. **Non-mutation**

   Pass when a test confirms the target series and input data are unchanged after suggestion and resolution.

7. **Documentation**

   Pass when `README.md` documents the `[task]` schema, supported metrics, and custom-metric registration with an example that is exercised by a test.

8. **Core import without extras**

   Pass when `import automl` works with only core dependencies installed.

9. **Dependency health**

   ```bash
   python -m pip check
   ```

   Pass with no broken requirements.

## Merge Criteria

- All required checks pass.
- No competition data, credentials, or local absolute paths are committed.
- `specs/roadmap.md` marks Phase 2 complete with a verification note.
- No CLI behavior beyond Phase 0 was added.
- Phase 1 config files without a `[task]` table still load unchanged.
