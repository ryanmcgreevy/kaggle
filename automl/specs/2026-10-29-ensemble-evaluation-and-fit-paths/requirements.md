# Ensemble Evaluation and Fit-Only Requirements

## Goal

Separate evaluating an ensemble from fitting one for prediction. Users who want comparative validation scores can keep using the current evaluation flow; users who only need a prediction-ready ensemble can avoid the extra outer-validation fits.

## Context

Candidate-based `automl ensemble` currently evaluates voting or stacking over outer validation folds. For every outer fold it rebuilds and fits members on that fold's training rows, scores members and the ensemble on held-out rows, and finally fits the ensemble on all training rows for inference. Stacking additionally uses inner cross-validation within each fit to generate leakage-safe base-model predictions for its meta-model.

The outer folds provide comparative evaluation; they are not required to produce the final model. Inner stacking CV serves a different purpose and remains necessary to train a stacking meta-model without using in-sample member predictions. Phase 20's `ensemble-saved` path is for already-fitted model references. This phase adds a fit-only option to the candidate-based workflow, where members are rebuilt from saved candidate configurations and fitted on all provided training rows.

## Scope and Decisions

- Add an explicit mode to candidate-based ensembling: `evaluate` or `fit`.
- Preserve current behavior as the default (`evaluate`) so existing CLI users and Python callers do not silently change behavior.
- `evaluate` retains the shared outer validation splits, member/ensemble fold scores, final full-data fit, and existing score semantics.
- `fit` builds the selected baseline, tuning, or model-family candidates from their saved estimator parameters and preprocessing, then fits the prediction-ready final ensemble using all provided training rows.
- Fit-only mode does not require, construct, or run outer validation splits. It does not compute outer-fold scores for the ensemble or candidates.
- Stacking in fit-only mode still generates inner out-of-fold member predictions, controlled by the existing stacking CV setting, to train its meta-estimator. Voting performs no validation CV.
- Record fit-only mode and explicitly represent ensemble validation scores as unavailable/not computed; never present training or in-sample scores as validation.
- Preserve candidate labels, compatibility checks, candidate selection behavior, model persistence, and compatibility with the refit/prediction workflow.
- Keep the existing `ensemble-saved` behavior unchanged; this phase concerns candidates from baseline/tuning/family result artifacts.
- Expose equivalent mode selection through the Python API and CLI. Evaluation mode continues to require the existing split input; fit-only mode may omit it.
- Update the AutoML README and ensemble agent guidance with mode selection, split requirements, score interpretation, and fit-cost differences.
- No new dependency, automatic retuning, leaderboard claim, or implicit external side effect.

## Functional Requirements

1. Users can select evaluation or fit-only mode using the Python API and CLI.
2. Omitting the mode preserves the current evaluation behavior and its shared-split scores.
3. Evaluation mode rejects missing or invalid outer splits as it does today and retains member/ensemble comparisons on the same folds.
4. Fit-only mode accepts compatible candidate artifacts without outer split input and does not create outer validation splits or compute outer validation metrics.
5. Fit-only mode fits and persists a usable final ensemble on all available training rows. Classification voting and stacking and regression stacking retain their existing task restrictions.
6. Stacking fit-only mode trains its meta-model only from inner out-of-fold predictions; outer validation is not substituted for this purpose.
7. Fit-only result metadata identifies the mode and states that validation was not performed, with no ensemble validation summary or candidate comparison score presented as newly evaluated.
8. A persisted fit-only ensemble can be reloaded and consumed by the existing refit/prediction path with the same data/schema checks and prediction row alignment.
9. Existing evaluation outputs and default CLI/API behavior remain compatible; mode selection and result semantics are documented.

## Non-Goals

- Changing the existing `ensemble-saved` workflow or making it accept candidate result directories.
- Changing how validation splits are constructed, changing the default evaluation method, or changing stacking's inner-CV semantics.
- Retuning candidates, automatically choosing candidates, or guaranteeing ensemble improvement.
- Adding a new model library, dependency, prediction submission, or Kaggle upload.
