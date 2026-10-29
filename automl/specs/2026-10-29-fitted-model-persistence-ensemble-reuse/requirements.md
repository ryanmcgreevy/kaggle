# Fitted-Model Persistence and Ensemble Reuse Requirements

## Goal

Persist fitted AutoML candidate pipelines and their validation predictions under a shared, inspectable contract, then let ensembles reuse those saved artifacts without silently rebuilding or refitting members.

## Context

Baseline, tuning, model-family search, refit, and ensemble stages already produce structured results and local artifacts. Model ensembling currently works from candidate results, while refit is an explicit path for producing test predictions. Phase 20 connects these capabilities with durable model references and provenance, so saved candidates can be reused after process restart and their scores remain defensible.

The feature must preserve the local-first workflow and existing explicit refit path. MLflow is optional and user-configured; neither a hosted service nor network access is required for local use. Treat persisted `joblib` models as trusted executable artifacts and document compatibility constraints.

## Scope and Decisions

- Define one versioned fitted-model reference and metadata contract shared by baseline, tuning/model-family search, refit, and ensemble consumers.
- Use local model storage by default. Support MLflow as an explicitly configured optional backend through the existing tracking integration; do not introduce a separate tracking system or a core dependency.
- Persist complete preprocessing-plus-estimator pipelines for selected candidates. Metadata must identify the task, metric, feature schema, class order where applicable, effective configuration, dependency versions, training-data fingerprint, validation splits, and the model's fit provenance.
- Distinguish models fitted on individual training folds from models fitted on all declared training data. Preserve aligned out-of-fold predictions and sufficient provenance for leakage-safe stacking.
- Let ensemble operations load compatible saved members by stable reference from either supported backend:
  - Hard-voting classification must not fit any member or meta-model.
  - Stacking for classification and regression may fit only its meta-model, using saved leakage-safe out-of-fold member predictions; inference uses the saved full-training-data member pipelines.
- Preserve existing candidate-based ensemble behavior and explicit refit workflows. Do not silently substitute member training when a requested saved artifact or provenance is missing or incompatible.
- Validate ensemble scores only when the prediction provenance supports a leakage-safe evaluation: neither member training nor meta-model training may have used rows being scored. If available artifacts cannot support that protocol without retraining members, report the limitation rather than presenting a training score as validation.
- Reject mismatched data fingerprints, feature schemas, class orders, split definitions, incomplete OOF coverage, and incompatible artifact versions with actionable errors. Do not use in-sample predictions as a fallback for OOF inputs.
- Cover unavailable, corrupt, and missing artifacts, explicit overwrite rules, missing optional dependencies, and backward compatibility with existing candidate artifacts and the refit path.
- Document API, CLI, configuration, artifact schemas, trusted-model loading and version constraints, and relevant agent guidance in the same increment.
- Keep credentials out of artifacts and logs. Do not upload data or models or require competition data, network access, cloud credentials, or a new core dependency.

## Functional Requirements

1. Producers in scope can save complete fitted pipelines and return stable references with the shared versioned metadata contract.
2. Saved model metadata records enough training, schema, split, and fit provenance to determine compatibility and whether predictions are safe for evaluation or stacking.
3. OOF predictions are aligned to source rows, carry fold/split provenance, and have explicit coverage; incomplete or ambiguous coverage cannot be used for stacking.
4. Local save/load round trips work after process restart and reproduce predictions from the original fitted pipeline.
5. Consumers can select saved ensemble members by reference from local storage or explicitly configured MLflow, with actionable errors for incompatible, unavailable, missing, or corrupt artifacts.
6. Saved hard voting performs zero fits. Saved stacking fits only the meta-model and uses the saved full-data member pipelines for subsequent inference.
7. Ensemble validation is reported only when saved provenance proves a leakage-safe evaluation. Otherwise, expose the limitation and do not represent a training score as validation.
8. Existing candidate artifacts remain loadable for their supported workflows, and the current explicit refit path remains available without implicit behavior changes.
9. Overwrite behavior is explicit, and the CLI and Python API expose equivalent documented capabilities.
10. Loading, saving, or logging models does not leak credentials or raw competition data into metadata or logs.

## Non-Goals

- Replacing the explicit refit workflow or removing existing candidate-based ensembling.
- Automatically retraining missing or incompatible saved members.
- Introducing a hosted model registry, cloud execution, a new core dependency, or a new tracking system.
- Guaranteeing ensemble improvement over individual candidates.
- Accepting untrusted serialized model artifacts as safe to load.
