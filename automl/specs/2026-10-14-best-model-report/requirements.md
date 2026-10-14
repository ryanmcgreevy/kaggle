# Requirements: Best-Model Results PDF (Phase 13)

## Scope

**Deliverable (from roadmap):** Comparison report for selected best models, including validation metrics, variability, configuration, and relevant diagnostics.

**Acceptance (from roadmap):** A report is generated from saved structured run results, remains accurate after process restart, and clearly distinguishes validation scores from leaderboard claims.

In scope:

- `write_model_report` producing a PDF from saved result directories: baseline (`result.json`), single-estimator tuning (`tuning.json`), and model-family comparison (`tuning_families.json`), or the corresponding in-memory results.
- `collect_candidates` that turns the supplied results into a ranked list of `Candidate` records (the model comparison table), saved as `<stem>_summary.json` beside the PDF.
- Comparison of the top-N candidates by validation score (default `top_n=3`).
- Diagnostics, each limited to what the saved artifacts contain:
  - per-fold score variability (mean, std, min/max, per-fold scores);
  - baseline vs tuned improvement when a baseline and a tuning result share task, metric, and split;
  - tuning trial history (optimization curve, best-so-far) and best-parameters table;
  - parameter importance computed from stored trial records;
  - out-of-fold diagnostics for candidates with saved OOF predictions: confusion matrix (classification), residual and predicted-vs-actual plots (regression), calibration curve (binary probabilities).
- Optional `[model_report]` TOML table in config version 1 with Python overrides.
- Synthetic-fixture tests and README documentation.

Out of scope:

- CLI command (deferred to Phase 16; Python API only, as in Phase 5).
- Re-running, refitting, or re-scoring any model; loading fitted models; reading raw train/test data.
- Leaderboard prediction, public/private score claims, or Kaggle API calls.
- MLflow links or reading from MLflow; local artifacts are the only source.
- Feature importance and SHAP (require fitted models or raw data).
- Agent skill changes (Phase 17); README documents Python usage for people and agents.
- Seaborn.

## Decisions

| Topic | Decision |
|---|---|
| Interface | Python API only: `write_model_report`, `collect_candidates`. |
| Inputs | Any mix of baseline directories, tuning directories, and families directories (paths) or their loaded results. Kind is detected from the file present (`result.json` with baseline schema, `tuning.json`, `tuning_families.json`); ambiguous or unknown directories raise `ReportError` naming the path. |
| Candidate definition | Baseline: the run. Tuning: its best completed trial. Families: each completed family's best trial, taken from the family's own `tuning.json`. Failed families and tunings with no best trial are listed as excluded with a reason, not ranked. |
| Comparability | All candidates must share task, metric name, and direction; otherwise `ReportError` listing the mismatches. Split fingerprints are compared and a warning is shown when they differ (scores then are not strictly like-for-like). |
| Ranking | By mean validation score in the metric direction; ties broken by lower std, then by input order. `top_n` caps the detailed pages; the summary table lists all candidates. |
| Variability | From stored per-fold scores (baseline `folds`, trial `fold_scores`); mean, std, min, max, and a fold-score strip/box chart. A single-fold (holdout) candidate states that variability is not estimable. |
| OOF diagnostics | Only for baseline directories (tuning artifacts store no predictions). Candidates lacking OOF predictions get a stated skip reason. Probability-dependent plots only when the stored `prediction_kind` supports them. |
| Parameter importance | Computed without Optuna from stored completed trials: absolute Spearman rank correlation between each numeric parameter and score (categorical parameters via correlation ratio on score ranks). Requires at least `min_trials_for_importance` (default 10) completed trials; otherwise skipped with a reason. Labeled as a descriptive heuristic, not causal. |
| Improvement | Tuned best minus baseline mean in the metric direction (positive means better), shown only when task, metric, and split fingerprint match; stated as validation-only. |
| Score wording | Every page footer and the summary state that scores are cross-validation or holdout validation scores on the training data, are optimistic when selected as the best of many trials, and are not leaderboard predictions. No leaderboard-like language elsewhere. |
| Config | Optional `[model_report]` table: `top_n`, `max_trials_plotted`, `min_trials_for_importance`, `max_params_importance`, `title`; no version bump; Python overrides win. |
| Dependency | Reuse the optional `reports` extra (Matplotlib), imported lazily with the same missing-Matplotlib `ReportError`; core import and `collect_candidates` work without it. |
| Overwrite and output | User-supplied output directory, created when missing; existing PDF or summary refused unless `overwrite=True`; atomic writes; nothing written outside the directory. |
| Rendering | Explicit Matplotlib `Figure` objects with the `Agg` backend; reuse Phase 5 helpers (`_require_matplotlib`, text pages, cap helpers) rather than duplicating. |
| Reproducibility | Report built only from saved JSON/CSV, so it is identical after process restart; the summary JSON records source paths, artifact schema versions, and library versions from the sources. |
| Result | Frozen `ModelReportResult` with `to_dict()` listing path, page count, candidates (ranked, with source and metrics), excluded entries, warnings, and skipped charts with reasons. |

## Context

- `specs/mission.md`: inspectable stage contracts; readable reports plus machine-readable outputs; do not hide model selection behind one unexplained score; reports must not claim leaderboard results.
- `specs/tech-stack.md`: Matplotlib PDFs generated from structured results, not notebook state; reports remain useful when optional libraries are absent; saved metadata sufficient for downstream stages; no implicit overwrite.
- Phase 8 `load_baseline` provides folds, summary, split, preprocessing, estimator, config, and OOF predictions. Phase 10 `load_tuning` provides trials (params, `fold_scores`, `fold_std`, durations, failure categories), space, budgets, and fingerprint. Phase 11 `load_tuning_families` provides per-family outcomes, comparison ranking, and the selection note. All loaders work without Optuna or fitted models.
- Phase 5 `report.py` provides the PDF assembly, atomic write, overwrite, and Matplotlib-lazy-import patterns to follow.

## Assumptions

- Result schemas at the versions the existing loaders accept are the only supported inputs; the report relies on those loaders for validation.
- Validation scores across candidates are comparable only under the same task, metric, and split; the report flags rather than corrects differences.
- Tuning artifacts do not contain OOF predictions, so OOF diagnostics describe baseline candidates only. Extending tuning to store the best trial's OOF predictions is a possible later increment, not part of this phase.
- Explicit `overwrite=True` is the user's instruction to replace files.
