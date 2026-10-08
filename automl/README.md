# AutoML

Local-first tools for tabular Kaggle competitions. The package and CLI are a Phase 0 skeleton; no data inspection, model training, tuning, reporting, or submission commands are implemented yet.

## Install and Test

From this directory, install the package and its test extra in editable mode:

```bash
python -m pip install -e '.[test]'
```

Run the test suite:

```bash
python -m pytest
```

## CLI

Show the current command-line help:

```bash
automl --help
python -m automl --help
```

## Input Data Contract

A run is described by a TOML file. Relative paths resolve against the config file's directory. Unknown keys are rejected.

```toml
version = 1
seed = 7

[data]
train = "train.csv"
test = "test.csv"
sample_submission = "sample_submission.csv"
target = "y"
id_columns = ["id"]
```

- `target` is required in train and must be absent from test.
- `id_columns` is optional and explicit; each must exist in train and test and differ from the target.
- Train and test must otherwise have identical columns; mismatches list missing and extra columns.
- `sample_submission` is optional; its row count (and ID values, when `id_columns` is set) must match test.

```python
from automl import load_config, load_data

bundle = load_data(load_config("run.toml"))   # keyword overrides, e.g. target="y"
bundle.train, bundle.test, bundle.sample_submission
```

Invalid inputs raise `automl.DataContractError`, which lists every problem found.

## Task and Metric Selection

An optional `[task]` table (config version 1) sets the task and metric. Neither is ever guessed: when unset, `resolve_task_metric` raises `automl.TaskMetricError` with a suggestion to confirm.

```toml
[task]
task = "binary"       # binary | multiclass | regression
metric = "roc_auc"
```

| Task | Metrics (default suggestion first) |
|---|---|
| binary | `roc_auc`, `log_loss`, `accuracy`, `f1` |
| multiclass | `log_loss`, `accuracy`, `macro_f1`, `balanced_accuracy` |
| regression | `rmse`, `mae`, `r2`, `rmsle` (non-negative values only) |

```python
from automl import (load_config, load_data, resolve_task_metric,
                    suggest_task, suggest_metric, register_metric)

bundle = load_data(load_config("run.toml"))      # keyword overrides: task=..., metric=...
suggest_task(bundle.train["y"])                   # Suggestion(value, rationale, evidence, ambiguous)
resolved = resolve_task_metric(bundle)            # checks task, metric, and target consistency
resolved.task, resolved.metric.name, resolved.direction

register_metric("my_metric", lambda y_true, y_pred: ..., ["regression"], "minimize")
```

Integers with few distinct values are suggested with `ambiguous=True` and are never selected automatically. Custom metrics are registered in Python only and cannot replace built-ins.

## Validation Strategy

An optional `[validation]` table (config version 1) sets how train rows are split. Defaults are shown; keyword overrides `strategy`, `n_folds`, and `holdout_fraction` take precedence (the `seed` keyword remains the run seed; set the split seed in `[validation]` or via `validation_config=`).

```toml
[validation]
strategy = "kfold"      # kfold | holdout
seed = 42
n_folds = 5
holdout_fraction = 0.2
```

- Classification (`binary`, `multiclass`) is stratified; regression uses plain shuffled splits.
- A class with fewer members than `n_folds` (kfold) or 2 (holdout), null targets, too few rows, or a holdout leaving fewer rows than classes raise `automl.ValidationSplitError`. There is no fallback to non-stratified splitting.
- Indices are positional rows of `bundle.train`. Splits depend only on the target, task, and config.
- A fixed seed repeats only for the same scikit-learn version; the saved artifact is the durable record for exact reproduction.

```python
from automl import load_config, load_data, resolve_task_metric, make_splits, save_splits, load_splits

cfg = load_config("run.toml")                      # needs [task]; or pass task=..., metric=...
bundle = load_data(cfg)
resolved = resolve_task_metric(bundle)
result = make_splits(bundle.train[cfg.target], resolved, cfg.validation_config)
save_splits(result, "splits.json")                 # refuses to overwrite
result = load_splits("splits.json")                # result.splits[i].train_idx / valid_idx
```

Artifact (`splits.json`, `schema_version` 1): `strategy`, `seed`, `n_folds`, `holdout_fraction`, `task`, `stratified`, `n_rows`, and `splits`, a list of `{fold, n_train, n_valid, train_idx, valid_idx}`. Holdout has one split (fold 0). `load_splits` rejects malformed files, out-of-range or duplicate indices, train/validation overlap, size mismatches, and k-fold validation sets that do not cover every row exactly once. No CLI command is added in this phase.

## EDA Summary

`summarize` returns a structured, non-mutating `EdaResult`; `save_eda` persists it as JSON for later stages (such as the Phase 5 PDF). The task is never inferred here: pass a `ResolvedTask`, `Task`, or task name to get class or regression target details.

An optional `[eda]` table (config version 1) sets defaults; keyword overrides with the same names take precedence.

```toml
[eda]
outlier_method = "both"    # iqr | zscore | both
iqr_multiplier = 1.5
zscore_threshold = 3.0
histogram_bins = 10
top_n = 10
id_like_threshold = 0.95
drift_alpha = 0.05
```

```python
from automl import load_config, load_data, resolve_task_metric, summarize_bundle, save_eda, load_eda

cfg = load_config("run.toml")                      # needs [task]; or pass task=..., metric=...
bundle = load_data(cfg)
result = summarize_bundle(bundle, task=resolve_task_metric(bundle))
save_eda(result, "eda.json")                       # refuses to overwrite
report = load_eda("eda.json")                      # validated plain dict
# or: summarize(train_df, test_df, target="y", id_columns=("id",), task="binary")
```

- **dataset:** rows, columns, memory, dtypes, duplicate rows (and duplicate IDs when `id_columns` is set).
- **columns:** every feature column (target and ID columns excluded) with kind (`numeric`, `boolean`, `categorical`, `other`), nulls, unique count, and flags `constant`, `all_null`, `id_like`. Numeric columns add mean, std (ddof=1), min, 5/25/50/75/95 percentiles, max, skew, a fixed-bin histogram (`edges`, `counts`), and outlier counts; categorical/boolean columns add the top-N values. Non-finite values are excluded from numeric statistics and counted in `non_finite_count`.
- **outliers:** IQR flags values outside `[Q1 - m*IQR, Q3 + m*IQR]`; z-score flags `|z| > threshold` using ddof=1. A single extreme value can mask itself under z-score, which is why both are reported by default.
- **id_like:** unique ratio of non-null values at or above `id_like_threshold`, for categorical and integer columns only.
- **target:** dtype, nulls, unique count; with a task, class counts, fractions, and imbalance ratio (largest over smallest class) or a regression numeric summary. `task` is `null` when none was supplied.
- **train_test:** row counts, columns present in only one frame, and per common feature column the null-fraction delta, numeric mean shift (train-std units) and std shift, unseen categories, and a KS (numeric) or chi-square (categorical) test with `drift` set when p < `drift_alpha`. Tests that cannot run set `skipped_reason` (constant, id-like, kind mismatch, too few values).

Limits: outlier and drift flags are statistical indicators, not errors or causal conclusions, and with many rows tests flag trivially small shifts. Undefined statistics are `null`; the JSON never contains NaN or infinity. The artifact (`schema_version` 1) has the top-level sections `config`, `dataset`, `columns`, `target`, and `train_test` (`null` when no test frame was given); `load_eda` rejects malformed files. No CLI command is added in this phase.
