# Model Ensembling Requirements

## Goal

Add a reusable AutoML stage that combines previously evaluated models into classification or regression ensembles. Users must be able to compare ensemble validation results with the constituent candidates, inspect the configuration and artifacts, and refit a selected ensemble through the existing prediction workflow.

## Context

The S6 project examples motivate two complementary approaches:

- S6E4 documents a soft-voting ensemble of LightGBM, CatBoost, and HistGradientBoosting models, and experiments with `VotingClassifier` and `StackingClassifier` using logistic regression or a shallow decision tree as the final estimator. See [playground-s6ep4/README.md](../../../playground-s6ep4/README.md), [clean.ipynb](../../../playground-s6ep4/clean.ipynb), and [xgb_standalone.ipynb](../../../playground-s6ep4/xgb_standalone.ipynb).
- S6E5 uses stacking with logistic regression as the meta-model and includes a custom neural-network estimator among its base models. See [playground-s6ep5/README.md](../../../playground-s6ep5/README.md) and [submission.ipynb](../../../playground-s6ep5/submission.ipynb).

The AutoML implementation should carry these ideas into the documented local API/CLI workflow, rather than copying competition-specific preprocessing, estimators, or scripts from the examples.

## Scope and Decisions

- Support classification and regression.
- For classification, provide hard majority voting and stacking. Hard voting combines predicted class labels; do not treat probability averaging or weighted voting as part of this increment.
- For regression, provide stacking only. Majority voting does not apply to continuous predictions.
- Use a configurable final estimator for stacking. Logistic regression is the default classification meta-model; use a configurable regressor with the scikit-learn `StackingRegressor` default (`RidgeCV`) for regression.
- Build ensembles from compatible saved candidates produced by baseline, tuning, and model-family stages. Preserve each candidate's preprocessing/model pipeline; do not require users to hand-construct or refit base estimators just to evaluate an ensemble.
- Expose the stage through the Python API and CLI, save a machine-readable result and a loadable fitted ensemble, and allow the existing refit/prediction flow to consume the selected ensemble.
- Reuse declared validation splits and budgets where applicable. Do not silently retune candidates as part of ensemble evaluation.
- Use scikit-learn ensemble estimators and existing dependencies; new optional model libraries must remain optional.
- Do not promise that an ensemble will outperform its component models or improve a leaderboard score.

## Functional Requirements

1. The user can identify base candidates and choose a supported ensemble method through documented API and CLI inputs.
2. The stage rejects empty or duplicate candidate selections, incompatible task types, and candidates that cannot be used by the requested ensemble, with actionable errors.
3. Classification hard voting and classification/regression stacking respect the configured task, metric, class labels, and preprocessing pipeline.
4. Stacking trains its meta-model from out-of-fold base-model predictions according to the configured CV strategy; validation rows must not influence fitted preprocessing or meta-model training.
5. Candidate and ensemble comparisons use the same declared validation splits where compatible. Results include metrics, variability where available, selected candidate identities, ensemble method, task/metric, validation configuration, and artifact locations.
6. The saved ensemble can be loaded after process restart and used by the refit stage to predict test rows while preserving row alignment and the existing prediction validity checks.
7. Machine-readable results and model artifacts follow documented schemas, are written beneath the selected run/output directory, and do not overwrite prior artifacts implicitly.
8. Existing baseline, tuning, model-family, reporting, refit, and submission behavior remains usable without ensembling.

## Non-Goals

- Soft or weighted voting, blending, or competition-specific ensemble heuristics.
- Automatic base-model selection, hyperparameter tuning, or feature engineering.
- A claim or acceptance threshold that an ensemble must beat an individual model.
- Mandatory PyTorch, boosting, tracking, cloud, Kaggle, or GPU dependencies.
- Uploading or submitting predictions.
