# Validation: EDA Summary (Phase 4)

The feature is ready to merge when all checks pass on Python 3.12 or newer from the `automl/` project root, without competition data, Kaggle credentials, network access (after install), or optional ML dependencies.

## Required Checks

1. **Install**

   ```bash
   python -m pip install -e '.[test]'
   ```

   Pass when install succeeds and `scipy` (explicit, already transitive) is the only dependency change.

2. **Existing behavior intact**

   ```bash
   python -c 'import automl' && automl --help && python -m automl --help
   ```

   Pass when all exit 0.

3. **Test suite**

   ```bash
   python -m pytest
   ```

   Pass when all tests succeed, including Phase 0 to 3 tests and new tests covering:

   - dataset dimensions, dtypes, duplicate rows, and duplicate IDs;
   - missingness per column;
   - numeric statistics, with histogram counts summing to the non-null count;
   - IQR and z-score outlier counts matching hand-computed values for each `outlier_method`;
   - categorical cardinality, top-N values, and the `id_like` flag;
   - constant and all-null columns producing finite-or-`null` output with the right flags;
   - target summaries for binary, multiclass, regression, and no supplied task (no task inferred);
   - test data with the target column absent;
   - train/test row counts, null-fraction deltas, one-sided columns, and unseen categories;
   - drift flagged for a clearly shifted numeric and categorical column, and not flagged for identical data;
   - invalid config values and `[eda]` TOML parsing, unknown key rejection, and override precedence;
   - artifact save/load round trip, overwrite refusal, and malformed-file rejection.

4. **Complete structured result**

   Pass when a test on the mixed-type fixture asserts every section (dataset, columns, target, train/test) is present and populated.

5. **Valid JSON output**

   Pass when `to_dict()` output serializes with `json.dumps(..., allow_nan=False)`.

6. **Non-mutation**

   Pass when a test confirms the train and test frames (values, dtypes, index, columns) are unchanged after `summarize`.

7. **Error quality**

   Pass when each invalid-input test asserts the message names the offending setting.

8. **Stage boundary**

   Pass when a test loads a saved EDA artifact in a fresh call (no in-memory state from `summarize`) and finds the documented fields.

9. **Documentation**

   Pass when `README.md` documents the `[eda]` schema and defaults, outlier and drift definitions, limits, the JSON artifact schema, and a Python example exercised by a test.

10. **Core import without extras**

    Pass when `import automl` works with only core dependencies installed.

11. **Dependency health**

    ```bash
    python -m pip check
    ```

    Pass with no broken requirements.

## Merge Criteria

- All required checks pass.
- No competition data, credentials, or local absolute paths are committed.
- `specs/roadmap.md` marks Phase 4 complete with a verification note.
- No CLI behavior beyond Phase 0 was added and no plotting library was added.
- Existing config files without an `[eda]` table still load unchanged.
