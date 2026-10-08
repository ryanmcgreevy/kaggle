# Validation: Input Data Contract (Phase 1)

The feature is ready to merge when all checks pass on Python 3.12 or newer from the `automl/` project root, without competition data, Kaggle credentials, network access (after install), or optional ML dependencies.

## Required Checks

1. **Install**

   ```bash
   python -m pip install -e '.[test]'
   ```

   Pass when install succeeds and `pandas` is the only new runtime dependency.

2. **Existing behavior intact**

   ```bash
   python -c 'import automl' && automl --help && python -m automl --help
   ```

   Pass when all exit 0.

3. **Test suite**

   ```bash
   python -m pytest
   ```

   Pass when all tests succeed, including Phase 0 tests and new tests covering:

   - valid train/test, with and without sample submission;
   - missing file and empty file;
   - duplicate columns in each file;
   - train/test column mismatch (missing and extra reported);
   - target absent from train, target present in test;
   - missing ID column, ID equal to target, repeated ID names;
   - sample submission ID/row-count mismatch;
   - malformed TOML, unknown key, unsupported version;
   - relative path resolution and override precedence.

4. **Error quality**

   Pass when each invalid-input test asserts the message names the offending file, column, or setting.

5. **Non-mutation**

   Pass when a test confirms input CSV files are unchanged after loading.

6. **Documentation**

   Pass when `README.md` documents the TOML schema with a working example, and that example is exercised by a test.

7. **Core import without extras**

   Pass when `import automl` works in an environment with only core dependencies installed.

8. **Dependency health**

   ```bash
   python -m pip check
   ```

   Pass with no broken requirements.

## Merge Criteria

- All required checks pass.
- No competition data, credentials, or local absolute paths are committed.
- `specs/roadmap.md` marks Phase 1 complete with a verification note.
- No CLI behavior beyond Phase 0 was added.
