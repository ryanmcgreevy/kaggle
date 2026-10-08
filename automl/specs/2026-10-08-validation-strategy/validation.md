# Validation: Validation Strategy (Phase 3)

The feature is ready to merge when all checks pass on Python 3.12 or newer from the `automl/` project root, without competition data, Kaggle credentials, network access (after install), or optional ML dependencies.

## Required Checks

1. **Install**

   ```bash
   python -m pip install -e '.[test]'
   ```

   Pass when install succeeds and no new runtime dependency was added.

2. **Existing behavior intact**

   ```bash
   python -c 'import automl' && automl --help && python -m automl --help
   ```

   Pass when all exit 0.

3. **Test suite**

   ```bash
   python -m pytest
   ```

   Pass when all tests succeed, including Phase 0 to 2 tests and new tests covering:

   - repeatable splits for a fixed seed, and different splits for a different seed;
   - no train/validation overlap in any fold, and k-fold validation sets covering every row exactly once;
   - stratified splits preserving class proportions (binary and multiclass) within a stated tolerance;
   - regression using non-stratified splits;
   - holdout validation size matching `holdout_fraction`;
   - invalid config values (unknown strategy, `n_folds < 2`, fraction outside (0, 1), wrong types);
   - classes with too few members for the requested folds or holdout, null targets, and more folds than rows;
   - `[validation]` TOML parsing, unknown key rejection, and override precedence;
   - artifact save/load round trip, overwrite refusal, and rejection of malformed or inconsistent files.

4. **No silent fallback**

   Pass when a test confirms that a class with fewer members than `n_folds` raises `ValidationSplitError` and never returns non-stratified splits.

5. **Error quality**

   Pass when each invalid-input test asserts the message names the offending setting, class, or count and suggests a fix.

6. **Non-mutation**

   Pass when a test confirms the target series is unchanged after `make_splits`.

7. **Stage boundary**

   Pass when a test loads a saved split artifact in a fresh call (no in-memory state from `make_splits`) and obtains identical indices.

8. **Documentation**

   Pass when `README.md` documents the `[validation]` schema, defaults, stratification and error rules, the JSON artifact schema, and a Python example exercised by a test.

9. **Core import without extras**

   Pass when `import automl` works with only core dependencies installed.

10. **Dependency health**

    ```bash
    python -m pip check
    ```

    Pass with no broken requirements.

## Merge Criteria

- All required checks pass.
- No competition data, credentials, or local absolute paths are committed.
- `specs/roadmap.md` marks Phase 3 complete with a verification note.
- No CLI behavior beyond Phase 0 was added.
- Existing config files without a `[validation]` table still load unchanged.
