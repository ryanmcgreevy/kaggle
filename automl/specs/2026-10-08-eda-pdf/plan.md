# Plan: EDA PDF (Phase 5)

## 1. Module layout, errors, and optional dependency

1. Add `src/automl/report.py` with the report config, findings, and PDF generation; import Matplotlib lazily inside the PDF functions so `import automl` works without it.
2. Add `ReportError` to `errors.py` as a subclass matching the existing error style (actionable, names the setting or path).
3. Add an optional `reports = ["matplotlib>=3.8"]` extra to `pyproject.toml`; core dependencies are unchanged.
4. When Matplotlib is missing, raise `ReportError` with the install command (`pip install -e '.[reports]'`).
5. Confirm `import automl`, `automl --help`, and existing tests are unaffected without the extra.

## 2. Configuration

1. Define a frozen `ReportConfig`: `high_missing_fraction` (0.2), `outlier_fraction_warn` (0.05), `imbalance_ratio_warn` (10.0), `duplicate_fraction_warn` (0.01), `max_columns_per_chart` (20), `max_histograms` (12), `title` (default `"EDA Report"`).
2. Validate in `__post_init__`: fractions in `(0, 1]`, ratio `>= 1`, positive integers; collect all problems in one pass.
3. Accept an optional `[report]` TOML table in `load_config` (unknown keys and wrong types rejected; Python overrides win). Keep config version 1; files without the table load unchanged.

## 3. Findings

1. Define frozen `Finding` (`code`, `severity` of `info`/`warning`, `scope` of `dataset`/`column`/`target`/`train_test`, `column`, `message`, `value`, `threshold`) with `to_dict()`.
2. Implement `build_findings(eda, config=None) -> tuple[Finding, ...]` from an `EdaResult` or loaded EDA dict, using only stored values (no raw data).
3. Rules: high missingness, all-null, constant, ID-like feature, duplicate rows, duplicate IDs, high outlier fraction, class imbalance, one-sided train/test columns, unseen test categories, flagged drift, and large null-fraction delta.
4. Order findings deterministically (severity, scope, column). Wording states flags are heuristic, not causal or removal advice.

## 4. Charts

1. Implement one function per chart returning a Matplotlib `Figure`, each tolerating absent sections: missingness bars, numeric histograms from stored bins, target distribution (class bars or regression histogram), categorical top-value bars, train/test drift summary, and outlier fractions.
2. Cap columns per chart by `max_columns_per_chart` and histograms by `max_histograms`, ranking by severity (for example missingness or outlier fraction); state in the chart when columns were omitted.
3. Skip a chart when its input section is empty, and record the skip in the result rather than failing.
4. Use the non-interactive `Agg` backend through explicit `Figure` objects; no `pyplot` global state, no notebook or display requirement.

## 5. PDF assembly

1. Implement `write_eda_report(eda, output_dir, *, filename="eda_report.pdf", config=None, overwrite=False) -> ReportResult`, accepting an `EdaResult` or a path to a saved EDA JSON (validated with `load_eda`).
2. Pages: title and dataset overview, findings table (paginated), then the charts that apply.
3. Create `output_dir` if needed; write only inside it. Refuse to overwrite an existing PDF unless `overwrite=True`; write to a temporary file and move into place so a failed run leaves no partial PDF.
4. Return a frozen `ReportResult` (`path`, `pages`, `charts` included, `charts_skipped` with reasons, `findings`, `schema_version`) with `to_dict()`.
5. Also write `<stem>_findings.json` (default `eda_report_findings.json`) beside the PDF with the findings and report configuration, using the same overwrite rule.

## 6. Tests

1. Reuse the Phase 4 mixed-type fixtures in `tests/conftest.py`; skip PDF tests with `pytest.importorskip("matplotlib")`.
2. Cover: PDF is non-empty, starts with `%PDF`, and its page count matches `ReportResult.pages`; generation from a saved JSON artifact in a fresh call; absent optional sections (no target, no test, no numeric columns); constant and all-null columns; each finding rule triggered and not triggered at its threshold; deterministic finding order; chart caps and omission notice; overwrite refusal and `overwrite=True`; no partial file after a forced failure; missing Matplotlib message (simulated); `[report]` config parsing, unknown key, override precedence, and validation errors naming the setting; input non-mutation (`EdaResult` and source frames); `ReportResult.to_dict()` JSON-safe.
3. Add a test that `import automl` and `build_findings` work without importing Matplotlib.

## 7. Public API export

1. Export `ReportConfig`, `ReportResult`, `Finding`, `ReportError`, `build_findings`, and `write_eda_report` from `automl`.

## 8. Documentation and roadmap

1. Update `automl/README.md` with the `reports` extra install command, `[report]` schema and defaults, finding codes and thresholds, charts and when each is skipped, the findings JSON schema, limits (heuristic flags, charts depend on stored histogram bins and top-N), and a Python example for people and agents (no CLI this phase).
2. Mark Phase 5 complete in `specs/roadmap.md` with a verification note once validation passes.
