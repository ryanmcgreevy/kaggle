# Requirements: EDA PDF (Phase 5)

## Scope

**Deliverable (from roadmap):** PDF report generation from the structured EDA results, with charts and concise data-quality findings.

In scope:

- `write_eda_report` producing a PDF from an `EdaResult` or a saved EDA JSON artifact.
- Rule-based data-quality findings with configurable thresholds (`build_findings`), shown in the PDF and saved as JSON.
- Charts: missingness bars, numeric histograms, target distribution, categorical top-value bars, train/test drift summary, and outlier fractions.
- Optional `[report]` TOML table in config version 1 with Python overrides.
- Optional `reports` dependency extra (Matplotlib).
- Synthetic-fixture tests and README documentation.

Out of scope:

- CLI command (deferred to Phase 16; Python API only).
- Recomputing EDA or reading raw data frames; the report uses only the stored EDA result.
- Correlation matrices, feature importance, or model results (Phase 12 owns the model report).
- Interactive or HTML output, notebook display, or Seaborn.
- Any automatic data cleaning or removal recommendations.

## Decisions

| Topic | Decision |
|---|---|
| Interface | Python API only: `write_eda_report`, `build_findings`. |
| Input | `EdaResult` or path to a saved EDA JSON (validated with `load_eda`). No raw frames, so the report is reproducible from the Phase 4 artifact alone. |
| Charts | Missingness, numeric histograms (stored bins), target distribution, categorical top values, train/test drift, outlier fractions. Each is skipped with a recorded reason when its input is absent. |
| Findings | Rule-based, thresholds in `ReportConfig`; `info` or `warning` severity; deterministic order; saved as `<stem>_findings.json` (default `eda_report_findings.json`). |
| Config | Optional `[report]` table (`high_missing_fraction`, `outlier_fraction_warn`, `imbalance_ratio_warn`, `duplicate_fraction_warn`, `max_columns_per_chart`, `max_histograms`, `title`); no version bump; Python overrides win. |
| Dependency | Matplotlib in an optional `reports` extra, imported lazily; core import and `build_findings` work without it; missing Matplotlib raises `ReportError` with the install command. |
| Overwrite | Output directory is user-supplied; existing PDF or findings file is refused unless `overwrite=True`. Writes are atomic (temporary file, then move). |
| Chart limits | Capped by `max_columns_per_chart` and `max_histograms`, ranked by severity; the report states when columns are omitted. |
| Rendering | Explicit Matplotlib `Figure` objects with the `Agg` backend; no `pyplot` global state or display needed. |
| Result | Frozen `ReportResult` with `to_dict()` listing the path, page count, charts included and skipped, and findings. |

## Context

- `specs/mission.md`: inspectable, composable stage contracts, machine-readable outputs alongside readable reports, outlier flags and correlations are not causal conclusions.
- `specs/tech-stack.md`: Matplotlib for PDFs, reports generated from structured results (not notebook state), optional dependency groups with a clear message and core still importable, write beneath a user-selected directory without implicit overwrite.
- Phase 4 provides `EdaResult`, `save_eda`, `load_eda`, and stored histogram bin edges and counts, top-N categorical values, and drift results; Phase 5 consumes them without recomputing. The roadmap Phase 5 acceptance requires a non-empty, readable PDF in the requested directory without mutating input or needing a notebook.

## Assumptions

- Histogram, top-N, and drift data in the EDA result are sufficient for the charts; charts are limited by the Phase 4 configuration used to produce the result.
- Findings are heuristic flags for review; thresholds are defaults, not rules for competitions generally.
- A saved EDA artifact at schema version 1 is the only supported input version.
- Explicit `overwrite=True` is the user's instruction to replace the files; nothing is overwritten otherwise.
