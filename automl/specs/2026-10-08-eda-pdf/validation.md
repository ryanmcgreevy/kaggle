# Validation: EDA PDF (Phase 5)

The feature is ready to merge when all checks pass on Python 3.12 or newer from the `automl/` project root, without competition data, Kaggle credentials, network access (after install), or a notebook session.

## Required Checks

1. **Install with and without the extra**

   ```bash
   python -m pip install -e '.[test,reports]'
   ```

   Pass when install succeeds and Matplotlib is the only dependency addition, declared in the optional `reports` extra (core `dependencies` unchanged).

2. **Existing behavior intact**

   ```bash
   python -c 'import automl' && automl --help && python -m automl --help
   ```

   Pass when all exit 0.

3. **Test suite**

   ```bash
   python -m pytest
   ```

   Pass when all tests succeed, including Phase 0 to 4 tests and new tests covering:

   - a non-empty PDF beginning with `%PDF` whose page count matches `ReportResult.pages`;
   - generation from an `EdaResult` and from a saved EDA JSON path in a fresh call;
   - absent optional sections (no target, no test data, no numeric columns), constant columns, and all-null columns;
   - each finding rule triggered above and not triggered below its threshold, with deterministic ordering;
   - chart caps with an omission notice, and skipped charts recorded with reasons;
   - overwrite refusal, `overwrite=True`, and no partial file left after a forced failure;
   - missing Matplotlib producing a `ReportError` with the install command;
   - `[report]` TOML parsing, unknown key rejection, override precedence, and invalid values;
   - `ReportResult.to_dict()` and the findings JSON serializing with `json.dumps(..., allow_nan=False)`.

4. **Readable report**

   Pass when a test or documented manual check renders the generated PDF (for example with `pdftotext` or a PDF reader) and confirms the title, findings text, and each applicable chart section are present.

5. **No input mutation**

   Pass when a test confirms the `EdaResult` (compared via `to_dict()`) and any source frames are unchanged after report generation.

6. **Output location**

   Pass when a test confirms all files are written only inside the requested output directory, which is created when missing.

7. **Core import without Matplotlib**

   Pass when a test (or a clean environment without the `reports` extra) shows `import automl` and `build_findings` work without importing Matplotlib.

8. **Error quality**

   Pass when each invalid-input test asserts the message names the offending setting or path.

9. **Stage boundary**

   Pass when the report is produced solely from a saved Phase 4 artifact with no in-memory state from `summarize`.

10. **Documentation**

    Pass when `README.md` documents the `reports` extra, `[report]` schema and defaults, finding codes and thresholds, charts and skip conditions, the findings JSON schema, limits, and a Python example exercised by a test.

11. **Dependency health**

    ```bash
    python -m pip check
    ```

    Pass with no broken requirements.

## Merge Criteria

- All checks above pass.
- No Kaggle data, credentials, or generated PDFs are committed.
- `specs/roadmap.md` marks Phase 5 complete with a verification note.
- No CLI behavior was added; Phase 15 owns it.
