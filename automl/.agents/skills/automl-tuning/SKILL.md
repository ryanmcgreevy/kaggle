---
name: automl-tuning
description: "Use when tuning AutoML models with Optuna: automl tune, trial or time budgets, model families, search spaces, resuming a study, tuning.json, tuning_families.json, failed trials, optuna and boosting extras."
---

# AutoML Tuning

## Use when

A baseline exists or the user wants hyperparameter search for one estimator or several model families.

## Procedure

1. Ask the user for the budget: `--n-trials` and/or `--timeout-seconds` (or the same keys in `[tuning]`). Never choose one; `tune` fails without a budget and writes nothing.
2. Run `automl tune run.toml --output-dir runs/1/tuning --splits runs/1/splits/splits.json --families hist_gradient_boosting --n-trials 30`. Without `--families` every installed family is tuned; not-installed families are listed with their install command. Use `--estimator <name>` instead to tune one estimator (not both options).
3. Reuse the same `--splits` file as the baseline so scores are comparable.
4. To add trials later, run the same command with `--resume` and the existing `--output-dir`; its budget counts new trials only and a changed config, data, or splits is rejected.
5. Read the JSON: best family or trial, score, direction, failed trials with `failure_category`. Add `--track` only if the user approved MLflow.

## Inputs

Run TOML, train CSV, optional `splits.json`, a budget; optional `[tuning]` table for families, per-family ranges, and fixed parameters.

## Outputs

A new directory with `tuning_families.json` and one Optuna study per family, or `tuning.json` and `study.db` with `--estimator`.

## Assumptions and limits

Optuna sampler is seeded; runs bounded by time do not repeat, and a running trial is not interrupted. Failed and pruned trials are recorded and never best; if none complete, the error lists the reasons. Scores are optimistically biased by the search and are not leaderboard claims. The output directory must be new unless `--resume`. Needs `pip install -e '.[tuning]'`, plus `'.[boosting]'` for those families.

## Check

Exit code 0, the expected JSON file exists, the number of completed trials is as expected, and `load_tuning` or `load_tuning_families` loads without Optuna. Python API: `tune`, `resume_tuning`, `tune_families`, `resume_tuning_families`. Details: Optuna Tuning and Model-Family Search Spaces in `automl/README.md`.
