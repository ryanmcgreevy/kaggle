# Plan: Best-Model Results PDF (Phase 13)

## 1. Module layout and configuration

1. Add `src/automl/model_report.py`; import Matplotlib lazily and reuse the Phase 5 helpers in `report.py` (`_require_matplotlib`, text-page, cap helpers) instead of duplicating them.
2. Reuse `ReportError`; confirm the missing-Matplotlib message names `pip install -e '.[reports]'`. No new dependency.
3. Define a frozen `ModelReportConfig`: `top_n` (3), `max_trials_plotted` (200), `min_trials_for_importance` (10), `max_params_importance` (10), `title` (`"Model Comparison Report"`); validate in `__post_init__` collecting all problems.
4. Accept an optional `[model_report]` TOML table in `load_config` (unknown keys and wrong types rejected; Python overrides win). Keep config version 1; files without the table load unchanged.

## 2. Candidate collection

1. Define frozen `Candidate` (`label`, `source_kind` of `baseline`/`tuning`/`family`, `source_path`, `estimator`, `params`, `mean`, `std`, `fold_scores`, `n_folds`, `split_fingerprint`, `has_oof`, `n_trials`, `versions`) and `ExcludedEntry` (`label`, `source_path`, `reason`) with `to_dict()`.
2. Implement `collect_candidates(sources) -> tuple[tuple[Candidate, ...], tuple[ExcludedEntry, ...]]` accepting paths or loaded results, detecting the directory kind and delegating to `load_baseline`, `load_tuning`, `load_tuning_families` (no models loaded, no Optuna needed).
3. Expand families into one candidate per completed family using its best trial; list failed families and tunings without a best trial as excluded with the stored reason.
4. Validate comparability (task, metric, direction); raise `ReportError` listing every mismatch and path. Record a warning when split fingerprints differ.
5. Rank by mean in the metric direction, tie-break by lower std then input order; reject an empty candidate set with an actionable error.

## 3. Diagnostics (computed from saved artifacts only)

1. Variability: per-candidate mean, std, min, max, and per-fold scores; state when only one fold exists.
2. Improvement: tuned best vs baseline mean for matching task, metric, and split fingerprint, signed so positive is better in either direction; skip with a reason when not comparable.
3. Trial history: best-so-far curve and per-trial scores for tuning candidates (capped by `max_trials_plotted`, keeping the best), counts of completed, failed, and pruned trials with failure categories.
4. Parameter importance: from completed trials only, using Spearman rank correlation for numeric parameters and a correlation ratio for categorical ones; skip below `min_trials_for_importance`; cap by `max_params_importance`; labeled descriptive.
5. OOF diagnostics for candidates with OOF predictions: confusion matrix (classification), predicted-vs-actual and residual histogram (regression), calibration curve (binary with probability predictions); otherwise skip with a reason. Read the stored `prediction_kind` rather than guessing columns.

## 4. Charts

1. One function per chart returning a Matplotlib `Figure`, tolerating absent inputs: ranked score comparison with std error bars, fold-score strip chart, improvement bars, optimization curve, parameter importance bars, and the OOF charts.
2. Skip a chart when its input is absent and record the skip with a reason rather than failing.
3. Use explicit `Figure` objects with the `Agg` backend; no `pyplot` global state.

## 5. PDF assembly

1. Implement `write_model_report(sources, output_dir, *, filename="model_report.pdf", config=None, overwrite=False) -> ModelReportResult`.
2. Pages: title and scope with the score-wording statement (validation scores, selection optimism, no leaderboard claims); ranked comparison table of all candidates plus excluded entries and warnings; then, for each of the top-N candidates, a page with configuration (estimator, parameters, preprocessing summary, validation strategy, seed, versions) and its diagnostics; then the cross-candidate charts.
3. Put the score-wording statement in every page footer; avoid leaderboard-like language elsewhere.
4. Create `output_dir` if needed; write only inside it; refuse overwrite unless `overwrite=True`; write to a temporary file and move into place so a failed run leaves no partial PDF.
5. Write `<stem>_summary.json` (default `model_report_summary.json`) with candidates, excluded entries, warnings, config, source paths, and schema versions, using the same overwrite rule and `allow_nan=False`.
6. Return a frozen `ModelReportResult` with `to_dict()`.

## 6. Tests

1. Build small fixtures by running `run_baseline`, `tune`, and `tune_families` on synthetic data and saving them; add `importorskip` for Matplotlib (PDF tests) and Optuna (tuning fixtures), and hand-written JSON fixtures so most tests need neither.
2. Cover: PDF non-empty, starts with `%PDF`, page count equals `ModelReportResult.pages`; generation from paths in a fresh call after the originals are deleted from memory (restart simulation by subprocess); each source kind alone and mixed; candidate ranking for both metric directions and tie-breaking; top-N cap with the table still listing all; excluded entries (failed family, no best trial); task/metric/direction mismatch errors and split-fingerprint warning; unknown, empty, and ambiguous directories; holdout single-fold variability message; improvement sign and skip; importance threshold, ordering, and skip reason; OOF diagnostics present for baseline and skipped for tuning with a reason; classification vs regression charts; score-wording text present and no leaderboard claims; overwrite refusal, `overwrite=True`, no partial file after forced failure; missing Matplotlib message; `[model_report]` parsing, unknown key, precedence, validation errors; inputs not mutated (artifact files hash-identical); outputs only in the output directory; `to_dict()` and summary JSON `allow_nan=False` safe.
3. Add a test that `import automl` and `collect_candidates` work without importing Matplotlib or Optuna.

## 7. Public API export

1. Export `ModelReportConfig`, `ModelReportResult`, `Candidate`, `ExcludedEntry`, `collect_candidates`, and `write_model_report` from `automl`.

## 8. Documentation and roadmap

1. Update `automl/README.md`: install command for the `reports` extra, `[model_report]` schema and defaults, accepted source directories, candidate and ranking rules, diagnostics and when each is skipped, the summary JSON schema, limits (validation-only scores, selection optimism, OOF only for baselines, heuristic importance), and a Python example for people and agents (no CLI this phase).
2. Mark Phase 13 complete in `specs/roadmap.md` with a verification note once validation passes.
