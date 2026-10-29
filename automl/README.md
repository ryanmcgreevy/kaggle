# AutoML

Local-first tools for tabular Kaggle competitions. The package provides a Python API and a CLI that run the whole local workflow: validating inputs, splits, EDA and its PDF, baseline and tuned models, the model comparison PDF, refit and prediction, and a validated submission CSV.

## Install and Test

The repository's existing project environment is `automl/.venv`. If it is already set up, use it directly rather than system Python or a bare `automl` command:

```bash
# From the repository root:
./automl/.venv/bin/python -m automl --help
# From a sibling competition directory such as playground-s6ep10/:
../automl/.venv/bin/python -m automl validate run.toml
```

For a fresh setup, run the following from this directory (`automl/`); create the environment only if `.venv` does not already exist. For an existing environment, install only missing required dependencies/extras:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -e '.[test]'
```

Run the test suite from this directory with `./.venv/bin/python -m pytest`.

The CLI examples below use `automl` as shorthand for the selected project interpreter's `-m automl` invocation. Do not assume `python` or `automl` is on PATH. Command-line config, artifact, and output paths resolve relative to the working directory; CSV paths inside the TOML resolve relative to that TOML.

Optional model libraries are installed with extras (`lightgbm`, `xgboost`, `catboost`, or `boosting` for all); adapter tests are skipped when a library is absent. Hyperparameter tuning needs the `tuning` extra (`pip install 'automl[tuning]'`, Optuna); its tests are skipped without it. MLflow tracking needs the `mlflow` extra (`pip install 'automl[mlflow]'`); its integration tests are skipped without it.
## CLI

Every command is a thin wrapper over the Python API described below; people and agents use the same commands. Successful commands print one JSON object to stdout; progress notes and errors go to stderr (`error: ...`). Exit status is 0 on success, 1 for workflow, data-contract, missing-extra, and file errors, and 2 for usage errors. Input paths in the TOML resolve relative to that file; `--output-dir` is relative to the current working directory and is created when needed. Task and metric must be set in `[task]`; the CLI never guesses them. Nothing contacts Kaggle or the network, and nothing is uploaded or submitted.

`baseline`, `tune`, `ensemble`, `refit`, and `run` print immediately flushed progress to stderr by default; stdout remains JSON. Pass `--quiet` to suppress progress while keeping errors, or `--verbose` to also enable native estimator output where supported. The options are mutually exclusive. LightGBM, XGBoost, and CatBoost native output is routed to stderr; scikit-learn HistGradientBoosting has no native output enabled by this option. Native training loss, when shown, may not be the selected validation metric, and iteration-level output depends on the estimator.

Python API calls remain silent unless a progress callback is supplied. The callback receives immutable `ProgressEvent` objects (`stage`, `event`, `elapsed_seconds`, and event-specific `details`):

```python
from automl import run_baseline

events = []
result = run_baseline(
    bundle.train, cfg.target, cfg.id_columns, resolved, splits,
    cfg.baseline_config, cfg.preprocessing_config, progress=events.append,
)
print(events[-1].event, result.mean)
```

```bash
automl validate run.toml
automl split run.toml --output-dir runs/1/splits
automl eda run.toml --output-dir runs/1/eda
automl report run.toml runs/1/eda/eda.json --output-dir runs/1/reports
automl baseline run.toml --output-dir runs/1/baseline --splits runs/1/splits/splits.json
automl tune run.toml --output-dir runs/1/tuning --families hist_gradient_boosting,lightgbm --n-trials 30
automl model-report run.toml runs/1/baseline runs/1/tuning --output-dir runs/1/reports
automl ensemble run.toml runs/1/baseline runs/1/tuning --method stacking --splits runs/1/splits/splits.json --output-dir runs/1/ensemble
automl refit run.toml runs/1/ensemble --output-dir runs/1/refit
automl submit run.toml runs/1/refit --output-dir runs/1/submission
```

Or run every stage in one go:

```bash
automl run run.toml --output-dir runs/2 --n-trials 30
```

| Command | Inputs | Writes |
|---|---|---|
| `validate` | TOML, CSVs | nothing; JSON with task, metric, row counts |
| `split` | TOML, CSVs | `splits.json` |
| `eda` | TOML, CSVs | `eda.json` |
| `report` | TOML, `eda.json` | EDA PDF and findings JSON |
| `baseline` | TOML, CSVs, optional `--splits`, `--estimator` | a new baseline result directory (`baseline.json`, out-of-fold predictions, fold models) |
| `tune` | TOML, CSVs, optional `--splits`, `--estimator` or `--families`, a budget | a new tuning directory (`tuning_families.json` plus one study per family, or `tuning.json` with `--estimator`) |
| `model-report` | TOML, one or more result directories, `--top-n` | model comparison PDF and summary JSON |
| `ensemble` | TOML, two or more compatible result directories, optional `--splits`, `--method`, `--n-jobs`, repeated `--label` | `ensemble.json` and fitted `ensemble.joblib` |
| `refit` | TOML, CSVs, a result directory, `--label` or `--rank` | `predictions.csv`, `refit.json`, `pipeline.joblib` |
| `submit` | TOML, CSVs, a refit directory | `submission.csv`, `submission.json` |
| `run` | TOML, CSVs, optional `--estimator`/`--families`, budget | the automatic workflow under `splits/ eda/ baseline/ tuning/ reports/ refit/ submission/`; ensembling is a separate user-selected stage |

Progress options apply to the five long-running commands above: `--quiet` suppresses progress but not errors, and `--verbose` additionally enables supported estimator-native output on stderr. Progress reports stage/model and fold lifecycle, elapsed time, and completed-trial scores with the best score so far.

Hand-off between stages is by explicit directory paths; no hidden state is used, so any stage can be rerun alone from the saved artifacts.

- **Splits.** `--splits` on `baseline`, `tune`, and `ensemble` is optional. With a saved `splits.json`, every candidate and the ensemble use identical outer folds; without it, splits are built from `[validation]` (seeded, so identical for the same config and data) and are not saved. A file made for different data (row count or task) is rejected. `run` saves the splits once and reuses them.
- **Tuning budget.** `tune` and `run` need `--n-trials` and/or `--timeout-seconds` (or the same keys, or per-family budgets, in `[tuning]`); there is no default budget, and the budget is printed to stderr before the study starts. `tune` covers every installed family unless `--families` or `[tuning] families` is set; skipped, not-installed families are listed with their install command. `--estimator` tunes one estimator instead. `--resume` adds trials to the study in `--output-dir`; its budget counts new trials only.
- **Reports.** `model-report` accepts baseline, tuning, and families directories, alone or mixed. Scores are validation scores, not leaderboard claims.
- **Ensembling.** `ensemble` accepts baseline, tuning, and families result directories. It reuses candidate parameters/preprocessing but fits every fold-local model using only that fold's training rows; it never retunes. Classification supports hard majority voting and stacking; regression supports stacking only. Stacking uses LogisticRegression by default for classification and RidgeCV for regression. Supply repeated `--label` flags to select candidates. Hard voting combines class predictions (class ordering resolves ties), so only label metrics and binary ROC AUC are supported; use stacking for probability metrics. The final `ensemble.joblib` is fit on all training rows and can be passed to `refit` for predictions. Ensembles need not improve on their components.
- **Selection.** `refit` and `run` use rank 1 of the given result unless `--rank` or `--label` is passed (labels appear in the `model-report` JSON). An ensemble result contains one selected model and takes neither option. `run` refits the tuned result, or the baseline with `--skip-tuning`.
- **Overwrite.** New baseline, tuning, ensemble, split, and EDA artifacts refuse an existing path. `report`, `model-report`, `refit`, and `submit` refuse existing files unless `--overwrite`. `run` needs a new or empty `--output-dir`; if a stage fails it stops, keeps the files already written, and names the failing and completed stages.
- **Tracking.** `--track` (on `baseline`, `tune`, `refit`, `submit`, `run`) logs to MLflow using the `[tracking]` table; without `--track` nothing is logged even if the table exists. A baseline with no `tracking_uri` or `directory` logs to its own result directory. If logging fails after local files were written, the command exits 1 and says the local results are intact.
- **Optional extras.** A missing optional package is reported with its install command before files are written: `pip install -e '.[reports]'` (PDFs), `'.[tuning]'` (Optuna), `'.[boosting]'` or `'.[lightgbm]'`, `'.[xgboost]'`, `'.[catboost]'`, and `'.[mlflow]'`. The core install runs `validate`, `split`, `eda`, `baseline`, scikit-learn-based `ensemble`, `refit`, and `submit`; `automl --help` works without any extra.

Limits: one selected pipeline or ensemble is refit and submitted per run; `run` always uses the configured stage tables and has no interactive prompts; large searches are bounded only by their budget, so choose it deliberately. See the sections below for each stage's artifacts, assumptions, and checks.

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

## Leakage-Safe Preprocessing

An optional `[preprocessing]` table (config version 1) configures imputation, encoding, and scaling. Defaults are shown; every key except `seed` can be passed as a `load_config` keyword override, or pass `preprocessing_config=`.

```toml
[preprocessing]
numeric_imputer = "median"          # mean | median | constant
numeric_fill_value = 0              # used when numeric_imputer = "constant"
categorical_imputer = "most_frequent"  # most_frequent | constant
categorical_fill_value = "missing"
encoder = "onehot"                  # onehot | ordinal | target
scaler = "none"                     # none | standard | robust
max_cardinality = 20                # integer columns with more unique values are numeric
numeric_columns = []                # explicit overrides
categorical_columns = []
drop_columns = []
seed = 42                           # target-encoder cross-fitting
```

- The target and `id_columns` are always excluded. Naming one in an override raises `automl.PreprocessingError`. Columns not typed (including dropped ones) never reach the model (`remainder="drop"`).
- Inference: float and high-cardinality integer columns are numeric; string, category, bool, and low-cardinality integer columns are categorical; constant, all-null, and datetime columns are dropped. Every decision and its reason is in `plan.reasons`. Categorical values are converted to text, so mixed-type columns encode consistently.
- One-hot and ordinal encoders map unseen categories to all-zero / `-1`. Target encoding uses scikit-learn `TargetEncoder`, which cross-fits internally during `fit_transform`; `transform` uses encodings fit on all training rows. Output is dense. Infinite numeric values are treated as missing.
- `build_preprocessor` returns an unfitted, clonable `ColumnTransformer`. Learned state (imputation statistics, categories, scale, target means) exists only after `fit`, so use it inside a `Pipeline` for CV or call `fit_transform_fold`, which fits a clone on the split's train rows only.

```python
from automl import (
    build_preprocessor, fit_transform_fold, infer_columns, load_plan, save_plan,
)

cfg = load_config("run.toml")
plan = infer_columns(bundle.train, cfg.target, cfg.id_columns, cfg.preprocessing_config)
pre = build_preprocessor(plan, cfg.preprocessing_config, resolved.task)
X_train, X_valid = fit_transform_fold(pre, bundle.train, bundle.train[cfg.target], result.splits[0])
save_plan(plan, cfg.preprocessing_config, "plan.json")   # refuses to overwrite
plan, prep_cfg = load_plan("plan.json")
```

Artifact (`plan.json`, `schema_version` 1): `config` (effective settings), `numeric`, `categorical`, `dropped`, `excluded`, and `reasons` (column to reason). `load_plan` rejects unknown settings, wrong types, and columns listed in more than one group. Fitted preprocessors are not persisted; they live inside fitted pipelines in later phases. No CLI command exists for this phase.

## Quick Baseline

`run_baseline` fits one `HistGradientBoostingClassifier` (binary, multiclass) or `HistGradientBoostingRegressor` (regression) per validation split and scores each fold's validation rows with the resolved metric. An optional `[baseline]` table (config version 1) sets defaults; keyword overrides `n_jobs`, `save_models`, and `estimator_params` take precedence (the `seed` keyword remains the run seed).

```toml
[baseline]
seed = 42
n_jobs = 1           # folds run in parallel; -1 uses all cores
save_models = true   # write models.joblib
[baseline.estimator_params]   # overrides for the HistGradientBoosting defaults
max_iter = 200
```

- Early stopping is off and `random_state` comes from `seed`, so a fixed seed and splits repeat for the same library versions. `estimator_params` cannot set `random_state`; unknown parameters raise `automl.BaselineError`.
- Model inputs are the typed numeric and categorical columns only; the target, ID, and dropped columns never reach the model. Preprocessing is fit inside each fold. With `preprocessing_config=None` the encoder is ordinal; pass `cfg.preprocessing_config` to use the run config. Encoded categoricals are treated as numeric.
- Metrics receive `predict_proba` when `needs="proba"` (positive-class column for binary) and `predict` otherwise. Each classification fold must see the same classes in train and validation rows, otherwise the error names the fold.
- Out-of-fold predictions: k-fold covers every row once; holdout covers validation rows only. Scores are validation scores under the declared split, not leaderboard claims; `std` is the population standard deviation across folds.
- When `save_models=true`, saving the result also writes `fitted-model/` with the complete pipeline fitted on all training rows and the aligned OOF predictions. This adds one full-data fit; use the stable `fitted-model` reference to reuse the model without refitting.

```python
from automl import (
    load_config, load_data, resolve_task_metric, make_splits,
    run_baseline, save_baseline, load_baseline,
)

cfg = load_config("run.toml")
bundle = load_data(cfg)
resolved = resolve_task_metric(bundle)
splits = make_splits(bundle.train[cfg.target], resolved, cfg.validation_config)   # or load_splits(...)
result = run_baseline(
    bundle.train, cfg.target, cfg.id_columns, resolved, splits,
    cfg.baseline_config, cfg.preprocessing_config,
)
print(result.metric, result.direction, result.mean, result.std)
save_baseline(result, "runs/baseline")             # new directory; refuses to overwrite
result = load_baseline("runs/baseline")            # load_models=True only for trusted directories
```

Artifact directory: `baseline.json` (`schema_version` 2: `task`, `metric` name/direction/needs, `estimator` name, key, and effective params, `categorical_handling`, `device`, `config`, `preprocessing` (the plan schema above), `split` summary, `prediction_kind`, `n_rows`, per-fold `score`/sizes/timings, `summary` mean and std, library `versions`, and relative `files`), `oof_predictions.csv` (`row` positional index, `fold`, and `prediction` or `proba_<class>` columns), and `models.joblib` (fitted fold pipelines, when `save_models` is true). `load_baseline` rejects malformed files, missing files, inconsistent scores, and out-of-range or duplicate rows. Joblib unpickles on load, so only load `models.joblib` from trusted sources, and expect it to work only with the library versions in `versions`. Set `save_models = false` to skip it. `load_baseline` also reads `schema_version` 1 directories. CLI: `automl baseline` (see CLI).

## Boosting Adapters

The `estimator` key in `[baseline]` selects one estimator per run. LightGBM, XGBoost, and CatBoost are optional; the core install and the default `hist_gradient_boosting` need none of them. Nothing imports an adapter library until it is selected.

| `estimator` | Install | Native categoricals (default) | GPU |
|---|---|---|---|
| `hist_gradient_boosting` | core | supported (off) | no |
| `lightgbm` | `pip install 'automl[lightgbm]'` | on | OpenCL build only |
| `xgboost` | `pip install 'automl[xgboost]'` | on | CUDA |
| `catboost` | `pip install 'automl[catboost]'` | on | CUDA |

`pip install 'automl[boosting]'` installs all three. Selecting a library that is not installed raises `automl.EstimatorUnavailableError` before any fitting, naming the package and the install command.

```toml
[baseline]
estimator = "lightgbm"          # hist_gradient_boosting | lightgbm | xgboost | catboost
native_categoricals = true      # omit for the estimator's default
device = "cpu"                  # cpu | auto | gpu
[baseline.estimator_params]
n_estimators = 300
```

- Defaults: seeded from `seed`, CPU, early stopping off, 100 trees/iterations, library logging silenced, and no files written (CatBoost `allow_writing_files=False`). `n_jobs` is the library's thread count; for adapters folds run sequentially, while `hist_gradient_boosting` still runs folds in parallel.
- `estimator_params` cannot set seed, thread count, device, verbosity, early stopping, or categorical parameters; the error names the setting to use instead. Unknown parameters raise `BaselineError`.
- All adapters accept string and integer targets and return probabilities in `classes_` order.
- Native categoricals: the transformed matrix is the numeric block followed by the categorical block, which is ordinal codes fit on the train fold; missing and unseen categories stay missing (no imputation). The preprocessing `encoder` setting is ignored and `categorical_handling` is recorded as `"native"`. Set `native_categoricals = false` for the encoded path. CatBoost receives its categorical columns as strings, with missing as `"nan"`. HistGradientBoosting requires at most 255 categories per column in this mode.
- Device: `auto` uses the GPU only if the library supports it and a tiny probe fit succeeds, otherwise it falls back to CPU and records why; `gpu` raises `EstimatorUnavailableError` instead of falling back. The result's `device` records requested, effective, reason, and a reproducibility note: GPU training is not guaranteed to be bit-reproducible, while CPU runs repeat for a fixed seed, splits, and library versions.

```python
from automl import BaselineConfig, list_estimators, estimator_available, run_baseline

list_estimators()                       # ['hist_gradient_boosting', 'lightgbm', 'xgboost', 'catboost']
estimator_available("lightgbm")          # True when the package is installed
cfg = BaselineConfig(estimator="lightgbm", device="auto", estimator_params={"n_estimators": 300})
result = run_baseline(bundle.train, target, id_columns, resolved, splits, cfg)
result.estimator_key, result.categorical_handling, result.device
```

Artifact `schema_version` 2 adds `estimator.key`, `categorical_handling`, and `device`, and records the library version under `versions`. CLI: `--estimator` on `baseline`, `tune`, and `run`; a missing library is reported with its install command.

## Optuna Tuning

`tune` searches the hyperparameters of one estimator with Optuna, scoring each trial as the mean validation score over the given splits (the same fold runner, leakage-safe preprocessing, and metric as `run_baseline`). Install the optional dependency with `pip install 'automl[tuning]'`; without it `tune` raises `automl.TuningUnavailableError` before any work, and `import automl`, the baseline, and `load_tuning` still work. An optional `[tuning]` table (opt-in; absent means `cfg.tuning_config is None`) configures the run:

```toml
[tuning]
n_trials = 30            # and/or timeout_seconds; at least one is required
timeout_seconds = 600
pruning = false          # true: median pruner on cumulative fold means
seed = 42
estimator = "hist_gradient_boosting"   # same estimator, native_categoricals, device, n_jobs keys as [baseline]
[tuning.estimator_params]  # fixed (not tuned) parameters
max_leaf_nodes = 31
```

- Direction (maximize or minimize) comes from the metric and is recorded in the result. The sampler is `TPESampler(seed=seed)`; on CPU with a fixed seed, data, splits, and library versions, trials repeat. Runs bounded by `timeout_seconds` do not repeat, and a running trial is never interrupted, so overrun is at most one trial (reported in `budgets`).
- Search space: a mapping `{name: {"type": "float"|"int"|"categorical", "low", "high", "log", "step", "choices"}}`, or a callable `fn(trial) -> dict` of estimator parameters. Each estimator family has a documented built-in space (used when `space` is omitted; see Model-Family Search Spaces). Parameters are merged over `estimator_params`; controlled keys (seed, threads, device, verbosity, early stopping, categorical settings) and keys fixed in `estimator_params` are rejected.
- A trial that raises is recorded as `failed` with its reason and a `failure_category` (`invalid_parameter`, `library_error`, `timeout`, `non_finite_score`, or `other`) and the study continues; failed and pruned trials are never best. If no trial completes, `TuningError` summarizes the reasons. Pruned trials keep the fold scores computed before pruning.
- Scores are search-selected validation means and are optimistically biased; they are not leaderboard claims. Final refit and prediction are later phases.

```python
from automl import TuningConfig, tune, resume_tuning, load_tuning

space = {"learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True}}
result = tune(
    bundle.train, cfg.target, cfg.id_columns, resolved, splits,
    TuningConfig(n_trials=30, seed=42), space=space,
    preprocessing_config=cfg.preprocessing_config, directory="runs/tuning",   # new directory
)
result.best.params, result.best.score, result.direction
result = resume_tuning("runs/tuning", bundle.train, cfg.target, cfg.id_columns, resolved, splits, n_trials=10)
result = load_tuning("runs/tuning")        # reads tuning.json only; no Optuna needed
```

Artifact directory (created by `directory=`; an existing directory is refused): `study.db` (Optuna SQLite storage), `tuning.json` (`schema_version` 2; schema 1 files still load, with failed trials categorized `other`: `task`, `metric`, `config`, `space`, `fingerprint`, `budgets` (one entry per run), `best_trial`, `trials` with `number`/`state`/`params`/`score`/`fold_scores`/`fold_std`/`duration_seconds`/`failure_reason`/`failure_category`, `versions`, `notes`), and `fitted-model/` when `directory=` is provided. Saving the best trial's OOF predictions and full-data pipeline performs additional fold refits and one full-data fit after the search. No training data is stored, nothing leaves the directory, and no network access is used. `resume_tuning` adds trials to the same study; its `n_trials`/`timeout_seconds` budget applies to the new trials only. It checks a fingerprint (rows, features, target, metric, estimator, seed, fixed parameters, preprocessing, and a hash of the split indices) and raises `TuningError` naming any difference. Mapping spaces are saved and reused; a callable space is not serializable and must be passed again. A resumed study reseeds the sampler as `seed + number of existing trials`. CLI: `automl tune --estimator ...` and `--resume`.

## Model-Family Search Spaces

Every estimator family has a documented, task-aware search space, so `tune` works without a supplied `space`. `get_search_space(family, task)` returns the mapping and `describe_search_space(family, task)` lists parameters; both work without Optuna or the boosting libraries. Ranges are conventional starting points, not claimed optimal. The multiclass ceiling on rounds is 200 instead of 300 because cost grows with the number of classes. Controlled keys (seed, threads, device, verbosity, early stopping) are never sampled.

| Family | Parameters (log scale marked `log`) |
|---|---|
| `hist_gradient_boosting` | `learning_rate` 0.01-0.3 log, `max_iter` 20-300, `max_leaf_nodes` 8-63, `max_depth` 3-12, `min_samples_leaf` 5-40, `l2_regularization` 1e-8-10 log |
| `lightgbm` | `learning_rate` log, `n_estimators` 20-300, `num_leaves` 8-128, `max_depth` 3-12, `min_child_samples` 5-50, `subsample` 0.5-1, `subsample_freq` fixed 1 (LightGBM ignores `subsample` otherwise), `colsample_bytree` 0.5-1, `reg_alpha`/`reg_lambda` 1e-8-10 log |
| `xgboost` | `learning_rate` log, `n_estimators` 20-300, `max_depth` 2-10, `min_child_weight` 1-10, `subsample` 0.5-1, `colsample_bytree` 0.5-1, `gamma` 1e-8-5 log, `reg_alpha`/`reg_lambda` 1e-8-10 log |
| `catboost` | `learning_rate` log, `iterations` 20-300, `depth` 3-8, `l2_leaf_reg` 1-10 log, `random_strength` 1e-3-10 log |

Range overrides adjust an existing parameter (`low`, `high`, `log`, `step`, `choices`), drop it (`enabled = false`), or pin it (`fixed = value`). Unknown families, parameters, or keys and invalid bounds raise `TuningError` naming the override. Overrides cannot be combined with a supplied `space` for the same family. The effective space is recorded in each `tuning.json`.

`tune_families` runs one seeded study per family on identical splits and ranks the best trials by the metric direction. `families` defaults to `config.families`, then to all installed families; a requested family whose library is missing raises its install hint. Each family's budget comes from the `budgets` argument, then `[tuning.budgets.<family>]`, then the shared `n_trials`/`timeout_seconds`; a family with no budget is rejected before any work. `config.estimator_params` apply only to `config.estimator`. A family whose trials all fail is recorded as `failed` and does not stop the others; `TuningError` is raised only if every family fails.

```toml
[tuning]
n_trials = 20                        # shared default budget
families = ["lightgbm", "xgboost"]
[tuning.budgets.xgboost]             # per-family budget
timeout_seconds = 120
[tuning.spaces.lightgbm.learning_rate]
low = 0.05
high = 0.2
[tuning.spaces.lightgbm.reg_alpha]
enabled = false
```

```python
from automl import TuningConfig, tune_families, load_tuning_families, resume_tuning_families

res = tune_families(
    bundle.train, cfg.target, cfg.id_columns, resolved, splits, cfg.tuning_config or TuningConfig(n_trials=20),
    families=["lightgbm", "xgboost"], budgets={"xgboost": {"timeout_seconds": 120}},
    preprocessing_config=cfg.preprocessing_config, directory="runs/families",   # new directory
)
res.best_family, res.comparison           # ranked rows: best_score, best_fold_std, trial_counts, failure_categories, best_params
res = load_tuning_families("runs/families")   # no Optuna needed
res = resume_tuning_families("runs/families", bundle.train, cfg.target, cfg.id_columns, resolved, splits, n_trials=10)
```

Artifacts: `<directory>/<family>/` holds the single-study `study.db`, `tuning.json`, and `fitted-model/`; `<directory>/tuning_families.json` (`schema_version` 1) holds `task`, `metric`, `config`, `budget_plan` (written before any trial runs), `families` (status, directory, error), `comparison`, `best_family`, and `notes`. Saving each family's best-model OOF predictions and full-data pipeline performs additional fold refits and one full-data fit after the search. An existing directory is refused. Family scores are search-selected validation means, optimistically biased, and are not leaderboard claims. CLI: `automl tune` (families by default).

## Optional MLflow Tracking

Install with `pip install 'automl[mlflow]'` (MLflow 3.x). Tracking mirrors results to MLflow only when you ask; without it nothing changes, `import automl` never imports MLflow, and requesting tracking without the package raises `automl.TrackingUnavailableError` (naming the install command) before any work. Local result directories remain the source of truth: a tracking failure raises `TrackingError` ("local results are intact") and never alters local files. CLI: `--track` on `baseline`, `tune`, `refit`, `submit`, and `run`.

Pass a `TrackingConfig` as `tracking=` to `run_baseline`, `tune`, `resume_tuning`, `tune_families`, or `resume_tuning_families`, or mirror an existing result later with `track_baseline`, `track_tuning`, or `track_families` (each takes a result object or a saved directory). Mirroring happens after the local work finishes; a study with no completed trial is not mirrored.

| `[tracking]` key | Default | Meaning |
|---|---|---|
| `enabled` | `true` | `false` turns tracking off while keeping the section |
| `tracking_uri` | none | any MLflow URI; overrides the local default |
| `directory` | none | run directory for the local default store |
| `experiment_name` | `automl` | experiment to log into |
| `run_name` | derived | e.g. `tuning-lightgbm-seed42` |
| `tags` | `{}` | extra string tags; credential-like names are rejected |
| `log_model` | `true` | log the fitted pipeline |
| `max_trials_logged` | `200` | cap on nested trial runs; the best trial is always kept |

Destination: `tracking_uri` if set, otherwise a SQLite store `<directory>/mlflow.db` with artifacts in `<directory>/mlartifacts`, where `directory` is `TrackingConfig.directory` or the tuning/result directory. With neither, a `TrackingError` asks for one. Hosted servers are used only if you supply their URI; credentials in a URI are redacted (`user:***@`) in every message, tag, and result.

Layout: a baseline is one run. A tuning study is a parent run (config, direction, counts, `best_score`, best parameters, `tuning.json`) with one nested run per trial (`param.*`, `score`, `fold_score` by step, `fold_std`, `duration_seconds`, `automl.trial_state`, failure reason and category, `automl.is_best`). `track_families` logs a top-level run (`rank.<family>` params, `automl.best_family`, `tuning_families.json`) with a nested run per family that holds its trial runs. `resume_tuning(tracking=...)` logs only the added trials and tags `automl.since_trial`; `resume_tuning_families` logs the full families again. Logging a saved result twice creates a second run. Truncated trials are counted in `automl.trials_not_logged`; the full history is always in the JSON artifact.

Model: with training data available (`run_baseline`, `tune`, and `tune_families` supply it; `track_*` takes `data=ModelData(train, target, id_columns, resolved_task)`), the complete preprocessing-plus-estimator pipeline is refit on all training rows, with the best trial's parameters for tuned runs (best family for families), and logged with `mlflow.sklearn` (cloudpickle, plus an inferred signature but no input example); its URI is in `TrackingResult.model_uri` and the `automl.model_uri` tag. Separately, `track_fitted_model` logs a persisted fitted-model artifact, including metadata and optional OOF predictions, and returns a stable `runs:/<run-id>/model` reference for `load_fitted_model`. When using this reference in an ensemble CLI command, configure `[tracking].tracking_uri` or `[tracking].directory` in the run TOML. Model files are pickles: load only from stores you trust, and use matching dependency versions. LightGBM, XGBoost, CatBoost, and scikit-learn pipelines were verified. When the model is skipped (`log_model = false`, no data, or a library problem), the reason is in `TrackingResult.notes` and the `automl.notes` tag.

Privacy: the baseline/tuning/refit/submission tracking stages do not log raw data rows, test predictions, sample submissions, input examples, or environment variables. `track_fitted_model` logs positional OOF predictions (`row`, `fold`, `prediction`, and optional probabilities) only when explicitly requested, so saved ensembles can reuse them; it does not log feature rows or identifiers. Logged scores are validation estimates, optimistically biased for tuned runs, and not leaderboard claims.

```python
from automl import TrackingConfig, ModelData, tune, track_tuning

tracking = TrackingConfig(experiment_name="my-comp", tags={"competition": "s6ep10"})
res = tune(df, "y", ("id",), resolved, splits, TuningConfig(n_trials=20), directory="runs/tune", tracking=tracking)
# or later, from the saved directory (no refit unless data is given):
out = track_tuning("runs/tune", tracking, data=ModelData(df, "y", ("id",), resolved))
out.run_id, out.child_run_ids, out.model_uri
```

```toml
[tracking]
experiment_name = "my-comp"
max_trials_logged = 100
[tracking.tags]
owner = "me"
```

Browse a local store with `mlflow ui --backend-store-uri sqlite:///runs/tune/mlflow.db`.

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

## Best-Model Results PDF

`write_model_report` builds a comparison PDF and a summary JSON from saved results only: baseline directories (`baseline.json`), tuning directories (`tuning.json`), and model-family directories (`tuning_families.json`), as paths or loaded results, in any mix. It never refits, rescores, or rereads the data, so it is identical after a process restart. It uses the same optional `reports` extra as the EDA PDF; `collect_candidates` needs neither Matplotlib nor Optuna.

```toml
[model_report]
top_n = 3                      # candidates with detail pages; the table lists all
max_trials_plotted = 200
min_trials_for_importance = 10
max_params_importance = 10
title = "Model Comparison Report"
```

Keyword overrides for `load_config` use the same names, except `title` (the EDA report's) and `top_n` (the EDA setting of that name); pass `model_report_config=` for those.

```python
from automl import write_model_report, collect_candidates

result = write_model_report(
    ["runs/baseline", "runs/tuning", "runs/families"], "reports/", target=train["y"],  # target is optional
)
result.path, result.summary_path, result.pages, result.candidates, result.excluded, result.charts_skipped
collect_candidates("runs/families")      # ranking and warnings only
```

- **Candidates:** a baseline run; a tuning run's best completed trial; each completed family's best trial. Failed families and tunings without a completed trial are listed as `excluded` with a reason. Candidates must share task, metric, and direction (otherwise `ReportError` names each path); differing row counts, fold counts, or split hashes produce a warning. Ranking is by mean validation score in the metric direction, ties by lower fold std, then input order.
- **Pages:** overview and ranked table, cross-candidate charts (`score_comparison`, `fold_variability`, `improvement`), then per top-N candidate a configuration page plus `trial_history`, `parameter_importance`, and `oof_diagnostics` where they apply. Skipped charts and reasons are in `charts_skipped`.
- **Improvement:** tuned minus baseline mean, positive is better, only when row count, fold count, and split hashes match.
- **Parameter importance:** absolute Spearman correlation with the score (correlation ratio for categorical parameters) over completed trials; skipped below `min_trials_for_importance`. Descriptive, not causal.
- **Out-of-fold plots:** baseline directories only, because tuning artifacts store no predictions, and only when `target=` (the training target column, in original row order) is supplied, because saved predictions omit the true values. They give a confusion matrix and calibration (binary probabilities) or predicted-vs-actual and residuals (regression).

Files are written only inside the output directory: `model_report.pdf` (or `filename=`) and `<stem>_summary.json` (`schema_version` 1: `candidates`, `excluded`, `warnings`, `charts`, `charts_skipped`, `config`, `notes`, `score_statement`). Existing files are refused unless `overwrite=True`; writes are atomic.

Limits: every page states that scores are validation scores on the training data, optimistic after selection, and not leaderboard predictions. Single-fold (holdout) candidates have no estimable variability. The `[model_report]` table is read by `load_config`. CLI: `automl model-report`.

## Model Ensembling

`run_ensemble` takes saved baseline, tuning, or model-family result directories and reuses their estimator parameters and preprocessing; hyperparameter search is not repeated. By default, `mode="evaluate"` evaluates the ensemble and its members on the provided `SplitResult`, then fits one final ensemble on all training rows for prediction. `mode="fit"` skips that outer evaluation and only fits the prediction-ready ensemble on all training rows. Stacking uses inner out-of-fold predictions to train its meta-model in either mode; that inner CV is needed for leakage-safe stacking and is distinct from optional outer evaluation. Fit mode reports no validation score.

```bash
automl ensemble run.toml runs/1/baseline runs/1/tuning \
  --method stacking --splits runs/1/splits/splits.json \
  --output-dir runs/1/ensemble
automl ensemble run.toml runs/1/baseline runs/1/tuning \
  --method stacking --mode fit --output-dir runs/1/ensemble-fit
automl refit run.toml runs/1/ensemble --output-dir runs/1/refit
```

Use the default `--mode evaluate` to compare fold-wise ensemble and candidate scores on shared outer splits. Use `--mode fit` when you only need a prediction-ready ensemble and do not want those extra outer-fold fits; omit `--splits`. Fit mode does not estimate or report ensemble validation performance. It still performs stacking's inner CV, while voting fits members once on all training rows. The fit-mode artifact can be passed to `automl refit` in the same way as an evaluated ensemble.

To reuse already-fitted pipelines without refitting ensemble members, pass at least two fitted-model references to `ensemble-saved`:

```bash
automl ensemble-saved run.toml runs/1/baseline/fitted-model runs/1/tuning/fitted-model \
  --method voting --output-dir runs/1/saved-ensemble
```

Hard voting uses saved fold-excluded predictions and performs zero fits. Saved stacking trains only its meta-estimator from complete, compatible OOF predictions; its validation score is unavailable because scoring the same OOF rows would reuse meta-model training data. Both methods use saved full-data pipelines for inference. Saved references must match the current training-data fingerprint, feature schema, classes, and split provenance; mismatches fail instead of triggering member refits.

Classification supports `--method voting` (hard majority vote) and `--method stacking`; regression supports stacking only. The default stacking meta-model is LogisticRegression for classification and RidgeCV for regression. Choose a different supported meta-estimator with `--meta-estimator` and optional JSON `--meta-estimator-params`. Use `--label` repeatedly to select candidate labels; otherwise all compatible candidates from the supplied directories are used. When selecting labels, obtain them from `collect_candidates` or the `model-report` summary. Hard voting uses predicted class labels and class ordering breaks ties; probability metrics other than binary ROC AUC require stacking.

Use `--n-jobs 2` to request two parallel jobs within stacking/voting, or `--n-jobs -1` for all available cores. Without the flag, `[baseline].n_jobs` is used (default `1`), preserving existing behavior. Evaluation mode's outer validation folds remain sequential. This setting does not override candidate-model thread counts or meta-estimator parameters. Parallel workers increase memory use; combining many workers with multi-threaded candidate models can oversubscribe CPUs, so choose a bounded worker count when resources are shared.

```python
from automl import EnsembleConfig, load_config, load_data, make_splits, resolve_task_metric, run_ensemble

cfg = load_config("run.toml")
bundle = load_data(cfg)
resolved = resolve_task_metric(bundle.train[cfg.target], cfg.task_config)
splits = make_splits(bundle.train[cfg.target], resolved, cfg.validation_config)
result = run_ensemble(
    bundle.train, cfg.target, cfg.id_columns, resolved, splits,
    ["runs/1/baseline", "runs/1/tuning"],
    EnsembleConfig(method="stacking", stack_cv=5),
    directory="runs/1/ensemble",
)
result.mean, result.std, result.candidates
```

For fit-only use, omit split construction and pass `splits=None`, `sources=...`, and `EnsembleConfig(mode="fit")`. The JSON artifact (`ensemble.json`, schema version 1) records the mode, candidates, validation status, per-fold/summary metrics when evaluated, split settings when used, feature columns and roles, task/metric, and data fingerprint. Fit-only artifacts have `validation_status="not_evaluated"`, null summary scores, no folds, and no newly computed candidate scores. Candidate details may include `source_validation_mean` and `source_validation_std`, which refer only to metrics saved with the original candidates. `ensemble.joblib` contains the final model fitted on all rows; it is a pickle and should be loaded only from a trusted directory. `load_ensemble` reads metadata without deserializing the model; pass `load_model=True` only for trusted artifacts. `automl refit` accepts either mode's ensemble directory and rejects a different training dataset. Existing output directories are not overwritten.

Limits: evaluation scores are optimistic after candidate/ensemble selection and are not leaderboard claims; fit mode does not compute a validation score. Hard voting cannot calculate log loss or other probability-required metrics. Ensembles are not guaranteed to beat their members. `ensemble-saved` supports compatible local model directories and MLflow `runs:/` references; MLflow references need the tracking URI configured. Python API: `run_ensemble`, `run_saved_ensemble`, `load_fitted_model`, `load_ensemble`, `save_ensemble`.

## Refit and Prediction

`refit_and_predict` refits one selected pipeline on every training row and predicts the test rows. The pipeline comes from a saved baseline, tuning, or families directory (or loaded result), from an explicit estimator and parameters, or from a saved ensemble directory. For individual candidates preprocessing and estimator are rebuilt and fitted together as one scikit-learn `Pipeline`; for ensembles the saved `ensemble.joblib` is used after verifying its training-data fingerprint. Outputs are raw model outputs: no clipping, thresholding, or submission schema checks (the next phase). Needs no optional package beyond the chosen estimator's; MLflow is used only when `tracking=` is passed.

```toml
[refit]
save_model = true         # write pipeline.joblib
probabilities = true      # classification: add proba_<class> columns when the estimator has predict_proba
proba_tolerance = 1e-5    # allowed deviation for the [0, 1] and row-sum checks
max_reported_rows = 10    # offending row positions listed per problem
```

Keyword overrides for `load_config` use the same names; `refit_config=` replaces the table.

```python
from automl import TrackingConfig, load_config, load_data, load_refit, refit_and_predict, resolve_task_metric

cfg = load_config("run.toml")
bundle = load_data(cfg)
resolved = resolve_task_metric(bundle.train[cfg.target], cfg.task_config)  # as when the run was validated

result = refit_and_predict(
    "runs/families",                      # or select_pipeline(estimator="xgboost", params={...}, task="binary")
    bundle.train, bundle.test,
    target=cfg.target, id_columns=cfg.id_columns, resolved_task=resolved,
    rank=1,                               # or label="family:lightgbm"; default is rank 1 with a note
    directory="runs/refit",               # omit to skip writing files
    tracking=TrackingConfig(),            # optional: log the pipeline to MLflow under runs/refit
)
result.predictions.frame                  # ids, `prediction`, and `proba_<class>` columns
loaded = load_refit("runs/refit", load_model=True)   # metadata and predictions load without estimator libraries
```

The steps are also available separately: `select_pipeline`, `refit`, `predict`, `save_refit`.

- **Selection:** sources go through `collect_candidates`, so ranking and labels match the model report. With several candidates and no `label` or `rank`, rank 1 is used and `selection.notes` says so. Explicit selection validates estimator, task, device, and parameters (controlled keys such as `random_state` are rejected) and needs `task`.
- **Refit:** uses the selected parameters, seed, device request, categorical handling, and preprocessing config, and requires the train features to equal the selected result's. Null targets, missing columns, and a task or metric that differs from the selection raise `RefitError`.
- **Prediction:** test rows need every training feature; extra columns are ignored, ID columns are copied unchanged in test order. Classification returns labels in the original target dtype; `proba_<class>` columns follow `fitted.classes`. Unseen categories follow the chosen encoder's handling.
- **Validity checks:** prediction count equals test rows; no NaN or infinity; labels are training classes; probabilities within [0, 1] and summing to 1 within `proba_tolerance`. Any violation raises `RefitError` with the problem and offending row positions, and nothing is written.

Files in the output directory (existing files are refused unless `overwrite=True`; writes are atomic):

- `predictions.csv`: ID columns, `prediction`, then `proba_<class>` columns; row i matches test row i.
- `refit.json` (`schema_version` 1): `task`, `selection` (source, label, estimator, parameters, preprocessing, seed, device, trial number, validation mean/std), `config`, `data` (target, id columns, row counts, numeric/categorical/excluded columns, classes), `prediction` (kind, columns, notes), `device`, `fit_seconds`, `versions`, `files`. Loads without Optuna, MLflow, or boosting libraries.
- `pipeline.joblib`: the fitted pipeline when `save_model` is true. It is a pickle: load it only from a directory you trust, with matching library versions.
- `fitted-model/`: versioned shared metadata, the fitted pipeline, and optional OOF predictions for reuse. `refit.json` contains the relative `model_reference`; MLflow-enabled runs also log the model artifact by stable reference.

With `tracking=`, the fitted pipeline (cloudpickle), selection parameters, provenance tags, validation mean/std, and `refit.json` are logged as one run (`automl.kind=refit`); raw data and predictions are not. The check for MLflow happens before fitting, and a logging failure raises `TrackingError` after the local files were written.

Limits: one selected pipeline or ensemble per call; exact repeatability only on CPU; validation scores in `selection` are optimistic after selection and are not leaderboard estimates. CLI: `automl refit`.

## Submission CSV

`build_submission` turns refit predictions into a Kaggle-format CSV and checks it. It takes a `RefitResult` (from `refit_and_predict`) or a `LoadedRefit` (from `load_refit`), the test frame, and optionally the sample submission. It never contacts Kaggle; the file is for manual upload. Needs no optional package; MLflow is used only when `tracking=` is passed.

```toml
[submission]
float_format = "%.6f"          # optional CSV float format; default writes round-trippable values
allow_unverified_ids = false   # true: match rows by position when no id columns are configured
max_reported_rows = 10         # offending row positions listed per problem (config key only; the load_config override belongs to [refit])
[submission.column_map]        # optional: sample column = prediction column
# y = "proba_1"
```

```python
from automl import TrackingConfig, build_submission, load_submission

sub = build_submission(
    result,                      # RefitResult, or load_refit("runs/refit")
    bundle.test,
    sample=bundle.sample_submission,
    sample_name="sample_submission.csv",
    directory="runs/submission",           # omit to skip writing files
    tracking=TrackingConfig(),             # optional: log both files to MLflow under runs/submission
)
sub.frame                                   # exactly the sample's columns, in test.csv row order
loaded = load_submission("runs/submission")
```

`save_submission(sub, directory, overwrite=False)` writes an already built result.

- **Row order:** the submission always follows `test.csv`. The refit prediction identifiers are compared with the test identifiers row by row and any mismatch raises. A sample in a different order is accepted with a note; the output is not reordered to match it.
- **Identifiers:** nulls, duplicates, differing row counts, and identifier sets that differ between test and sample raise, naming the offending values. Without id columns, rows can only be matched by position, which requires `allow_unverified_ids=true` when a sample is given and is recorded in the notes.
- **Columns:** with a sample, output columns have exactly the sample's names and order; id columns come from test. One non-id column gets `prediction`, or the positive-class probability for a binary task whose metric needs probabilities. Several non-id columns must be named after the class labels (or `proba_<class>`). Anything else raises and lists the available prediction columns; use `column_map` to choose explicitly.
- **Values and dtypes:** each column is cast to the sample's dtype class only when lossless (for example integer labels into a float column are fine, 0.7 into an integer column is not). Nulls, NaN, infinity, probabilities outside [0, 1], and row-count mismatches raise `SubmissionError` and nothing is written.
- **No sample:** the output is the id columns plus the target column holding `prediction`, with a note that the schema is unverified.

Files in the output directory (existing files are refused unless `overwrite=True`; writes are atomic):

- `submission.csv`: header, no index, test row order.
- `submission.json` (`schema_version` 1): `columns`, `column_mapping` (output column to source), `n_rows`, `id_columns`, `id_check`, `schema_source` (`sample` or `fallback`), `sample_file`, `source` (refit label and path), `config`, `notes`, `files`, and a statement that predictions are not leaderboard estimates.

With `tracking=`, both files are logged as one run (`automl.kind=submission`). A directory is required, the check for MLflow happens before any file is written, and a logging failure raises `TrackingError` after the local files were written.

Limits: no post-processing such as clipping, thresholding, or rounding; one prediction frame per call; the saved `submission.json` names the sample file but does not embed it. CLI: `automl submit`.
