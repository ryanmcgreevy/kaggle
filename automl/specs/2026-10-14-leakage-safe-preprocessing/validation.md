# Validation: Leakage-Safe Preprocessing (Phase 7)

The feature is ready to merge when all checks pass on Python 3.12 or newer from the `automl/` project root, without competition data, Kaggle credentials, network access (after install), or optional ML dependencies.

## Required Checks

1. **Install**

   ```bash
   python -m pip install -e '.[test]'
   ```

   Pass when install succeeds and no new runtime dependency was added (a scikit-learn minimum-version bump for `TargetEncoder` is allowed if documented).

2. **Existing behavior intact**

   ```bash
   python -c 'import automl' && automl --help && python -m automl --help
   ```

   Pass when all exit 0.

3. **Test suite**

   ```bash
   python -m pytest
   ```

   Pass when all tests succeed, including Phase 0 to 6 tests and new tests covering:

   - mixed numeric, categorical, and missing data for every imputer, encoder, and scaler option;
   - column inference for float, high- and low-cardinality integer, bool, string, and category columns, plus constant and all-null drops;
   - explicit `numeric_columns`, `categorical_columns`, and `drop_columns` overrides, and unknown override columns;
   - unknown categories at transform time for one-hot, ordinal, and target encoders;
   - `sklearn.base.clone` of the preprocessor and use inside `Pipeline` and `cross_val_score`;
   - invalid config values (unknown choice, wrong types, column in multiple lists) and `[preprocessing]` TOML parsing, unknown key rejection, and override precedence;
   - target encoding with an unsupported task;
   - plan artifact save/load round trip, overwrite refusal, and rejection of malformed files.

4. **Target and ID exclusion**

   Pass when tests confirm the target and every ID column are absent from the typed columns and the transformed output, and that naming one in an override raises `PreprocessingError` naming the column.

5. **Leakage safety**

   Pass when tests confirm that changing validation-fold rows (values, missingness, new categories, extreme outliers, labels) does not change the fitted state or train-fold output, and that `fit_transform_fold` output equals fitting on the train subset alone, for each imputer, encoder, and scaler.

6. **Error quality**

   Pass when each invalid-input test asserts the message names the offending setting or column and suggests a fix.

7. **Non-mutation**

   Pass when a test confirms the train frame is unchanged after inference, building, and fold fitting.

8. **Stage boundary**

   Pass when a test loads a saved plan in a fresh call (no in-memory state) and builds an equivalent preprocessor whose output matches the original on the same fold.

9. **Documentation**

   Pass when `README.md` documents the `[preprocessing]` schema, defaults, typing rules, ID/target exclusion, leakage guarantees, the plan JSON schema, and a Python example exercised by a test.

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
- `specs/roadmap.md` marks Phase 7 complete with a verification note.
- No CLI behavior beyond Phase 6 was added.
- Existing config files without a `[preprocessing]` table still load unchanged.
