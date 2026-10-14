# Validation: Best-Model Results PDF (Phase 13)

The feature is ready to merge when all checks pass on Python 3.12 or newer from the `automl/` project root, without competition data, Kaggle credentials, network access (after install), a notebook session, or a running MLflow server.

## Required Checks

1. **Install**

   ```bash
   python -m pip install -e '.[test,reports,tuning,boosting]' && python -m pip check
   ```

   Pass when install succeeds, `pip check` is clean, and core `dependencies` are unchanged (Matplotlib stays in the existing `reports` extra).

2. **Existing behavior intact**

   ```bash
   python -c 'import automl' && automl --help && python -m automl --help
   ```

   Pass when all exit 0 and no new CLI command appears.

3. **Test suite**

   ```bash
   python -m pytest
   ```

   Pass when all tests succeed, including Phase 0 to 12 tests and new tests covering:

   - a non-empty PDF beginning with `%PDF` whose page count matches `ModelReportResult.pages`;
   - baseline, tuning, and families directories each alone and mixed, passed as paths and as loaded results;
   - ranking for both metric directions, tie-breaking, top-N cap with the summary table listing every candidate;
   - excluded entries (failed family, tuning without a best trial) with reasons;
   - task, metric, or direction mismatch errors naming each offending path, and the split-fingerprint warning;
   - unknown, empty, and ambiguous source directories;
   - variability values equal to those recomputed from the stored fold scores, and the single-fold message;
   - baseline vs tuned improvement sign for both directions, and its skip when not comparable;
   - parameter importance threshold, ordering, and skip reason;
   - OOF diagnostics for classification and regression baselines, and skip reasons for tuning candidates;
   - overwrite refusal, `overwrite=True`, and no partial file after a forced failure;
   - missing Matplotlib producing a `ReportError` with the install command;
   - `[model_report]` TOML parsing, unknown key rejection, override precedence, and invalid values;
   - `ModelReportResult.to_dict()` and the summary JSON serializing with `json.dumps(..., allow_nan=False)`.

4. **Accurate after restart**

   Pass when a test generates saved artifacts in one process, then generates the report in a separate process (subprocess) that has not touched the originals in memory, and the resulting `<stem>_summary.json` equals one produced in the first process (excluding timestamps, if any).

5. **Validation vs leaderboard wording**

   Pass when a test extracts the PDF text (for example `pdftotext` or a PDF text parser, skipped if unavailable with a documented manual check) and confirms the validation-score statement appears on every page and that no leaderboard-score claim or prediction text appears.

6. **Readable report**

   Pass when a test or documented manual check renders the PDF and confirms the title, ranked table, per-candidate configuration, and each applicable chart section are present.

7. **No input mutation**

   Pass when a test confirms every source artifact file is byte-identical (hashes) after report generation and loaded results are unchanged.

8. **Output location**

   Pass when a test confirms all files are written only inside the requested output directory, which is created when missing.

9. **Core import without Matplotlib or Optuna**

   Pass when a test (or a clean environment without extras) shows `import automl` and `collect_candidates` on hand-written JSON fixtures work without importing Matplotlib or Optuna.

10. **Error quality**

    Pass when each invalid-input test asserts the message names the offending setting or path.

11. **Documentation**

    Pass when the README documents the install command, `[model_report]` schema, accepted sources, diagnostics and skip rules, summary JSON schema, and limits, and its Python example runs against synthetic artifacts.

12. **Roadmap**

    Pass when `specs/roadmap.md` marks Phase 13 complete with a verification note and records any deviations from this spec.
