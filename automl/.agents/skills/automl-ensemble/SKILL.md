---
name: automl-ensemble
description: "Use when building classification hard-majority voting or classification/regression stacking ensembles from saved AutoML baseline, tuning, or model-family candidates: automl ensemble, ensemble.json, ensemble.joblib, leakage-safe out-of-fold meta-models, shared splits."
---

# AutoML Ensembles

## Use when

The user wants to combine at least two saved baseline, tuning, or model-family candidates with hard voting or stacking, compare validation results, and optionally predict test rows.

## Procedure

1. Confirm which candidate labels to include; labels come from `automl model-report` or `collect_candidates`.
2. Confirm the method. Classification supports hard majority voting or stacking; regression supports stacking only.
3. Reuse a saved split artifact when available:
   `automl ensemble run.toml runs/1/baseline runs/1/tuning --method stacking --splits runs/1/splits/splits.json --output-dir runs/1/ensemble`.
4. Use repeated `--label <candidate-label>` options to restrict candidates. The default includes all compatible candidates in the supplied result directories.
5. Inspect `ensemble.json` for candidate and ensemble validation scores before choosing whether to use the ensemble.
6. To predict test rows, run `automl refit run.toml runs/1/ensemble --output-dir runs/1/ensemble-refit`.
7. To reuse at least two already fitted pipelines without fitting members, use `automl ensemble-saved run.toml runs/1/baseline/fitted-model runs/1/tuning/fitted-model --method voting --output-dir runs/1/saved-ensemble`.

Progress is flushed to stderr by default; use `--quiet` to suppress it or `--verbose` to also enable supported candidate-estimator native output. Stdout remains JSON. Python API calls accept an opt-in `ProgressEvent` callback and are otherwise silent.

## Inputs

Run TOML with explicit task and metric, train/test CSVs, at least two baseline/tuning/families result directories, and optionally `splits.json`. Saved candidate parameters and preprocessing are used; hyperparameter searches do not run again. `--stack-cv` controls inner stacking folds. The default classifier meta-model is LogisticRegression; regression defaults to RidgeCV. Python callers may pass any cloneable scikit-learn meta-estimator to `run_ensemble`.

For `ensemble-saved`, provide at least two trusted local `fitted-model/` directories or MLflow `runs:/` model URIs. MLflow references require `[tracking].tracking_uri` or `[tracking].directory`. Voting reuses fold-excluded saved predictions and fits no models. Stacking fits only the meta-estimator on complete compatible OOF predictions; its validation score is unavailable because the meta-estimator has trained on those rows. No member training fallback is allowed.

`--n-jobs` controls parallel jobs within stacking/voting; `-1` uses all available cores. Omission preserves `[baseline].n_jobs` (default `1`). Outer validation folds remain sequential, and saved candidate-model thread counts and meta-estimator parameters are unchanged. Bound parallelism on shared machines: extra workers increase memory use and can oversubscribe CPUs when candidate models also use multiple threads.

## Outputs

`ensemble.json` contains method, candidate labels/references, fold-wise validation scores when defensible, validation status/limitation, split settings, meta-estimator, and feature names. `ensemble.joblib` contains the final ensemble using members' saved full-data pipelines. Both are written in a new output directory; existing directories are refused. `automl refit` reads the saved ensemble and writes the usual refit artifacts.

## Assumptions and limits

Candidates must share task and metric, row count, and feature availability. Each validation fold fits base pipelines only on its training rows; stacking trains its meta-model from internal out-of-fold predictions. Hyperparameters are not retuned. Hard voting combines predicted class labels; sklearn class ordering breaks vote ties. Probability metrics other than binary ROC AUC are unsupported with hard voting; use stacking when probabilities are required. Ensembles are not guaranteed to beat their components. Scores are validation scores, not leaderboard claims. Joblib artifacts are pickles and must be loaded only from trusted runs.

## Check

Exit code 0, `ensemble.json` and `ensemble.joblib` exist, candidate labels/references match the intended selection, and any reported score is finite. For saved stacking, confirm `validation_status` is unavailable and the limitation is recorded. Run `automl refit` with the ensemble directory and verify `n_test_rows` and row order. Python API: `run_ensemble`, `run_saved_ensemble`, `load_fitted_model`, `load_ensemble`, `save_ensemble`. Details: Model Ensembling in `automl/README.md`.
