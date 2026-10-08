# Requirements: EDA Summary (Phase 4)

## Scope

**Deliverable (from roadmap):** A reusable, non-mutating EDA stage that summarizes dimensions, dtypes, missingness, duplicates, numeric distributions/outliers, categorical cardinality, target distribution, and train/test differences.

In scope:

- `summarize` / `summarize_bundle` producing a structured `EdaResult` from train (and optional test) data.
- Optional `[eda]` TOML table in config version 1 with Python overrides.
- Numeric summaries with percentiles, skew, fixed-bin histograms, and IQR and/or z-score outlier counts.
- Categorical cardinality with top-N values and ID-like flags.
- Target summary that uses a resolved task when supplied and stays dtype-based otherwise.
- Train/test comparison including missingness delta, one-sided columns and categories, mean/std shift, and KS / chi-square drift tests.
- A persisted JSON artifact with save/load and validation.
- Synthetic-fixture tests and README documentation.

Out of scope:

- PDF or chart generation (Phase 5); histograms are stored as data only.
- CLI command (deferred to Phase 15; Python API only).
- Feature engineering, importance, correlation matrices, or modeling.
- Automatic task, target, or ID inference (Phase 2 and the data contract own these).
- Text, image, datetime-specific, or time-series analysis; such columns are reported as kind `other`.

## Decisions

| Topic | Decision |
|---|---|
| Result shape | Frozen dataclasses with `to_dict()`; JSON artifact with `schema_version` via `save_eda` / `load_eda`. `load_eda` returns the validated dict (not dataclasses). Saving never overwrites an existing file. |
| Outliers | `outlier_method` is `iqr`, `zscore`, or `both` (default `both`); IQR fence multiplier default 1.5, z-score threshold default 3.0. Report count and fraction per method. Flags only, no rows removed. |
| Distributions | mean, std, min, 5/25/50/75/95 percentiles, max, skew, and a fixed-bin histogram (default 10 bins; edges and counts). |
| Categorical | Unique count and top-N (default 10) values with counts and fractions. Columns with unique ratio at or above `id_like_threshold` (default 0.95) are flagged `id_like`. |
| Column flags | `constant`, `all_null`, `id_like`. Never raised as errors. |
| Target | With a task: classification gets class counts, fractions, and imbalance ratio; regression gets a numeric summary. Without a task: dtype, nulls, unique count only, `task` is `null`. The task is never inferred here. |
| Train/test | Row counts, null-fraction delta, one-sided columns (reported), unseen categories, numeric mean/std shift, KS test (numeric) and chi-square test (categorical), flag when p is below `drift_alpha` (default 0.05). Tests that cannot run record a reason instead of failing. |
| Optional test target | Test frames may omit the target column; comparison excludes the target and ID columns. |
| Config | Optional `[eda]` table (`outlier_method`, `iqr_multiplier`, `zscore_threshold`, `histogram_bins`, `top_n`, `id_like_threshold`, `drift_alpha`) in config version 1 (no version bump). Python overrides take precedence. Files without the table load unchanged. |
| Invalid values | NaN and infinity are never emitted; undefined statistics are `null`. |
| Non-mutation | Input frames are never modified. |
| Interface | Python API only. |
| Dependencies | `scipy` declared explicitly in core (already a transitive dependency of scikit-learn); no plotting libraries. |

## Context

- `specs/mission.md`: inspectable by default, composable stage contracts with machine-readable results, conservative about inference, EDA correlations and outlier flags are not causal conclusions.
- `specs/tech-stack.md`: pandas/NumPy for tabular results; reports (Phase 5) are generated from structured results, not notebook state; machine-readable artifacts with documented schemas; do not overwrite prior outputs implicitly; synthetic fixtures; actionable errors naming the setting.
- Phase 1 provides `DataBundle`, `DataConfig`, and `load_config`; Phase 2 provides `ResolvedTask`; Phase 3 sets the pattern for an optional config table, an error subclass, and a JSON artifact with schema validation. Phase 5 consumes this result without recomputing it.

## Assumptions

- Feature columns are those other than the target and configured ID columns; ID columns get only dataset-level duplicate checks.
- Column kind is derived from dtype: numeric (excluding boolean), boolean, categorical (object, string, category), other.
- Drift tests are heuristic indicators on possibly large samples and can flag trivially small shifts; they are not proof of a distribution problem.
- Outlier flags describe statistical position only; they are not errors or reasons for removal.
- Histogram bin edges are computed from train data only so Phase 5 can chart them without rereading the frames.
