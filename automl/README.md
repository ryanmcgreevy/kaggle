# AutoML

Local-first tools for tabular Kaggle competitions. The package provides a Python API and an MVP CLI for validating inputs, generating validation splits, and producing EDA artifacts and reports. Modeling, tuning, prediction, and submission commands are future work.

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

The CLI exposes the existing data and EDA capabilities. Successful workflow commands print a JSON result to stdout; workflow errors are printed to stderr and exit with status 1. Input paths in the TOML resolve relative to that file. Output directories are relative to the current working directory and are created when needed.

```bash
automl --help
python -m automl --help
automl validate run.toml
automl split run.toml --output-dir runs/experiment-001
automl eda run.toml --output-dir runs/experiment-001
automl report run.toml runs/experiment-001/eda.json --output-dir runs/experiment-001/reports
```

`validate` checks the TOML, CSV data contract, and explicit task/metric choices. Set both values in `[task]`; the CLI will not silently choose them. `split` writes `splits.json`, and `eda` writes `eda.json`. Both refuse to overwrite existing artifacts. `report` reads the saved EDA artifact and writes a PDF plus findings JSON; it also refuses existing outputs unless `--overwrite` is passed. PDF generation requires the optional reports extra:

```bash
python -m pip install -e '.[reports]'
```

The CLI composes the same Python APIs described below. It does not preprocess, train, tune, submit, upload data, or require network access or Kaggle credentials.

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

Artifact (`splits.json`, `schema_version` 1): `strategy`, `seed`, `n_folds`, `holdout_fraction`, `task`, `stratified`, `n_rows`, and `splits`, a list of `{fold, n_train, n_valid, train_idx, valid_idx}`. Holdout has one split (fold 0). `load_splits` rejects malformed files, out-of-range or duplicate indices, train/validation overlap, size mismatches, and k-fold validation sets that do not cover every row exactly once. The `automl split` command uses this same API and artifact contract.

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

Limits: outlier and drift flags are statistical indicators, not errors or causal conclusions, and with many rows tests flag trivially small shifts. Undefined statistics are `null`; the JSON never contains NaN or infinity. The artifact (`schema_version` 1) has the top-level sections `config`, `dataset`, `columns`, `target`, and `train_test` (`null` when no test frame was given); `load_eda` rejects malformed files. The `automl eda` command writes this artifact using the configured EDA settings and explicitly selected task/metric.

## EDA PDF

`write_eda_report` builds a PDF and a findings JSON from a stored EDA result (an `EdaResult`, a dict, or the path of a file written by `save_eda`). It never rereads the data and does not need a notebook. Matplotlib is an optional extra; core import and `build_findings` work without it, and a missing install raises `automl.ReportError` with the command below.

```bash
python -m pip install -e '.[reports]'
```

An optional `[report]` table (config version 1) sets defaults; keyword overrides with the same names take precedence.

```toml
[report]
title = "EDA Report"
high_missing_fraction = 0.2    # also the train/test null-delta threshold
outlier_fraction_warn = 0.05
imbalance_ratio_warn = 10.0
duplicate_fraction_warn = 0.01
max_columns_per_chart = 20
max_histograms = 12            # panels for numeric and categorical charts
```

```python
from automl import load_config, load_data, resolve_task_metric, summarize_bundle, save_eda, write_eda_report, build_findings

cfg = load_config("run.toml")                      # needs [task]; or pass task=..., metric=...
bundle = load_data(cfg)
save_eda(summarize_bundle(bundle, task=resolve_task_metric(bundle)), "eda.json")
result = write_eda_report("eda.json", "reports/", config=cfg.report_config)   # overwrite=True to replace
result.path, result.findings_path, result.pages, result.charts, result.charts_skipped
build_findings("eda.json")                         # findings only, no matplotlib needed
```

Files are written only inside the output directory (created if missing): `eda_report.pdf` (or `filename=`) and `<stem>_findings.json`. Existing files are refused unless `overwrite=True`; files are written to temporary names first, so a failure leaves no partial output.

- **Pages:** overview, findings, then the charts that apply. Each chart is skipped, with the reason in `charts_skipped`, when its section is absent.
- **Charts:** `missingness`, `numeric_histograms` (from stored bins, ranked by outlier fraction), `target_distribution`, `categorical_top_values`, `train_test_drift` (-log10 p against `drift_alpha`), `outlier_fractions`. Bar charts show `max_columns_per_chart` columns and the page states how many were omitted.
- **Findings** (`warning` or `info`; thresholds are inclusive): `duplicate_rows`, `duplicate_ids`, `class_imbalance`, `all_null`, `constant`, `high_missing`, `id_like` (info), `high_outliers` (info), `one_sided_columns`, `drift`, `unseen_test_categories` (info), `null_delta`. Sorted by severity, scope, column, code.

Findings JSON (`schema_version` 1): `config`, `findings` (each `code`, `severity`, `scope`, `message`, `column`, `value`, `threshold`), `charts`, and `charts_skipped`.

Limits: findings are heuristic flags for review, not causal conclusions or removal advice. Charts show only what the EDA result stored (histogram bins, top-N values, drift results from the Phase 4 settings). The `automl report` command reads an EDA JSON artifact and uses `[report]` settings from the supplied TOML file.
