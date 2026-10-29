---
name: automl-baseline
description: "Use when running or inspecting a quick AutoML baseline: automl baseline, run_baseline, cross-validated HistGradientBoosting or LightGBM/XGBoost/CatBoost estimator, baseline.json, out-of-fold predictions, boosting extras."
---

# AutoML Baseline

## Use when

Task, metric, and splits are settled and the user wants a first validated model or a reference score to compare tuned models against.

## Procedure

1. Complete `automl-data-intake`; optionally pin folds with `automl-validation-splits`.
2. Run `automl baseline run.toml --output-dir runs/1/baseline --splits runs/1/splits/splits.json`. Omit `--splits` to build seeded splits from `[validation]`; they are identical for the same config and data but are not saved.
3. Use `--estimator <name>` to override `[baseline] estimator` (default `hist_gradient_boosting`; `lightgbm`, `xgboost`, `catboost` need their extra). Add `--track` only if the user approved MLflow and `[tracking]` exists.
4. Read the JSON: `mean`, `std`, metric direction, `splits.source`, and artifact paths. With `save_models=true`, the output also contains `fitted-model/` with a full-training-data pipeline and aligned OOF predictions; saving it adds one full-data fit.

Use the project interpreter described in `automl/AGENTS.md` for the CLI shorthand above. For example, from the repository root:

```bash
./automl/.venv/bin/python -m automl baseline ./playground-s6ep10/run.toml --output-dir ./playground-s6ep10/automl-runs/baseline
```

Substitute the requested competition/config and output paths. From the competition directory, the interpreter is `../automl/.venv/bin/python`; command-line paths are relative to that directory, unlike CSV paths inside the TOML.

Progress is flushed to stderr by default; `--quiet` suppresses it, while `--verbose` also enables supported native estimator output. Stdout remains result JSON. Python API callers can pass an opt-in `ProgressEvent` callback; API calls are otherwise silent.

## Inputs

Run TOML with `[task]` set, train CSV, optional `splits.json`; optional `[baseline]` and `[preprocessing]` tables.

## Outputs

A new directory with `baseline.json` (effective config, split summary, per-fold scores, mean and std, library versions), `oof_predictions.csv`, `models.joblib`, and `fitted-model/` when `save_models` is true.

## Assumptions and limits

One estimator per run, fixed by seed; the output directory must not exist. Preprocessing is fit inside each fold; a separate full-data fit is persisted for reuse. Scores are validation scores under the declared split, not leaderboard claims. Boosting extras: `pip install -e '.[boosting]'` (or `lightgbm`, `xgboost`, `catboost`); a missing library gives an install hint before files are written. Joblib model artifacts are pickles: load only from trusted runs.

## Check

Exit code 0, `mean` finite, `baseline.json` present, and `load_baseline("runs/1/baseline")` loads. Rerunning with the same config and splits gives the same scores on CPU. Python API: `run_baseline`, `save_baseline`, `load_baseline`. Details: Quick Baseline and Boosting Adapters in `automl/README.md`.
