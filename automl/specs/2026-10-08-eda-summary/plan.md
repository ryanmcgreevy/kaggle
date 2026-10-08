# Plan: EDA Summary (Phase 4)

## 1. Module layout, errors, and dependency

1. Add `src/automl/eda.py` (config, result dataclasses, `summarize`, artifact I/O).
2. Add `EdaError` to `errors.py` as a `DataContractError` subclass with the same multi-problem message style.
3. Declare `scipy` explicitly in core `dependencies` in `pyproject.toml` (already installed transitively via scikit-learn; used for the KS and chi-square tests).
4. Confirm `import automl`, `automl --help`, and existing tests are unaffected.

## 2. Configuration

1. Define a frozen `EdaConfig`: `outlier_method` (`"iqr"`, `"zscore"`, or `"both"`, default `"both"`), `iqr_multiplier` (1.5), `zscore_threshold` (3.0), `histogram_bins` (10), `top_n` (10), `id_like_threshold` (0.95), `drift_alpha` (0.05).
2. Validate in `__post_init__`: known method, positive numbers, `histogram_bins >= 1`, `top_n >= 1`, `0 < id_like_threshold <= 1`, `0 < drift_alpha < 1`; collect all problems in one pass.
3. Extend `load_config` to accept an optional `[eda]` table (unknown keys and wrong types rejected; Python overrides win). Keep config version 1 and existing behavior for files without the table. Expose `eda_config` on the loaded config.

## 3. Result structure

1. Define frozen dataclasses, each with `to_dict()` returning JSON-safe values: `EdaResult` (`config`, `dataset`, `columns`, `target`, `train_test`), `DatasetSummary`, `NumericSummary`, `CategoricalSummary`, `TargetSummary`, `TrainTestSummary`.
2. Dataset summary: row and column counts, memory use, per-column dtype, duplicate row count and fraction, and (when ID columns are configured) duplicate ID count.
3. Include `schema_version` on `EdaResult.to_dict()`.

## 4. Per-column summaries

1. For every feature column (excluding target and ID columns, which are reported separately) record: dtype, kind (`numeric`, `categorical`, `boolean`, `other`), null count and fraction, unique count, and flags `constant`, `all_null`, and `id_like` (unique ratio at or above `id_like_threshold`).
2. Numeric: mean, std, min, 5/25/50/75/95 percentiles, max, skew, histogram (bin edges and counts), and outlier counts and fractions for the selected method(s). Handle constant and all-null columns without errors or NaN/inf in the output (use `null`).
3. Categorical/boolean: unique count and top-`top_n` values with counts and fractions.

## 5. Target summary

1. When the target is present in train, record dtype, null count, and unique count regardless of task.
2. When a `ResolvedTask` or `Task` is supplied: classification gets class counts, fractions, and imbalance ratio (largest class over smallest); regression gets the numeric summary including histogram and outliers.
3. When no task is supplied, report dtype-based fields only and mark `task` as `null`; never infer a task.

## 6. Train/test comparison

1. Record train and test row counts, per-column null-fraction delta, and columns present in only one frame (reported, not raised).
2. Numeric columns: mean and std shift (in train-std units) and a two-sample KS statistic and p-value, with a `drift` flag when p is below `drift_alpha`.
3. Categorical columns: categories seen only in test and only in train (counts and top examples), plus a chi-square test p-value on the combined category counts, with the same flag. Skip the test with a recorded reason when a column is constant, all-null, or has too few observations.
4. Handle an absent optional test target and test frames that omit the target column.

## 7. Entry point and artifact

1. Implement `summarize(train, test=None, *, target=None, id_columns=(), task=None, config=None) -> EdaResult` and `summarize_bundle(bundle, task=None, config=None)` that reads the target and ID columns from `bundle.config`. Neither mutates its inputs.
2. Implement `save_eda(result, path)` writing JSON; refuse to overwrite an existing file. Implement `load_eda(path) -> dict` with schema-version and structure validation and actionable errors.
3. Document the JSON schema.

## 8. Tests

1. Add a synthetic mixed-type fixture (numeric with outliers and nulls, categorical, boolean, constant, all-null, ID-like, duplicate rows) and a matching test frame to `tests/conftest.py`.
2. Cover: dataset counts, dtypes, duplicates, and duplicate IDs; missingness; numeric stats, histogram totals equal non-null count; outlier counts for each method against hand-computed values; categorical cardinality, top-N, and ID-like flag; constant and all-null columns producing finite-or-null output; target summary for binary, multiclass, regression, and no task; absent target in test; train/test row counts, null deltas, one-sided columns, unseen categories, drift flag on shifted vs identical data; config validation and `[eda]` TOML parsing, unknown key, override precedence; non-mutation of inputs; `to_dict()` is JSON-serializable with no NaN/inf; artifact round trip, overwrite refusal, and malformed-file rejection.
3. Assert error messages name the offending setting.

## 9. Public API export

1. Export `EdaConfig`, `EdaResult`, `EdaError`, `summarize`, `summarize_bundle`, `save_eda`, and `load_eda` from `automl`.

## 10. Documentation and roadmap

1. Update `automl/README.md` with the `[eda]` schema and defaults, what each section reports, outlier and drift definitions, limits (heuristic flags, not causal conclusions), the JSON schema, and a Python example for people and agents (no CLI this phase).
2. Mark Phase 4 complete in `specs/roadmap.md` with a verification note once validation passes.
