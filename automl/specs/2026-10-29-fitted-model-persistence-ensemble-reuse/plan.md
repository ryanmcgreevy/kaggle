# Fitted-Model Persistence and Ensemble Reuse Plan

## Task Groups

1. **Defining the shared model and provenance contract**
   - Inspect the existing baseline, tuning/family, refit, tracking, and ensemble result/artifact contracts.
   - Specify versioned model references, metadata, fit provenance, schema compatibility, training-data fingerprints, split identity, and OOF prediction alignment/coverage.
   - Define local storage as the default and how the existing optional MLflow integration resolves and retrieves a model reference.
   - Define explicit failure behavior for missing, corrupt, incompatible, or untrusted artifacts; do not fall back to member training.

2. **Implementing local model persistence**
   - Add the shared local save/load behavior using complete preprocessing-plus-estimator pipelines.
   - Persist and validate stable references and required metadata, including dependency compatibility and whether a model is fold-trained or full-data-trained.
   - Persist aligned OOF predictions and their split/fold provenance where produced.
   - Enforce explicit overwrite behavior and actionable errors without requiring optional dependencies or network access.

3. **Integrating baseline, tuning/families, and refit producers**
   - Extend the relevant producer outputs with saved-model references and metadata while preserving existing result artifacts and APIs.
   - Save selected candidate pipelines and required full-data member pipelines, documenting any additional fitting required at the producing stage.
   - Ensure validation predictions remain aligned with their source rows and cannot be mistaken for full-data in-sample predictions.
   - Retain backward compatibility for existing candidate results and the explicit refit workflow.

4. **Reusing saved models in ensembles**
   - Add ensemble member loading from stable references in local storage and configured MLflow.
   - Ensure hard voting performs no fitting and stacking fits only the meta-model from compatible, leakage-safe OOF predictions.
   - Use saved full-data member pipelines for ensemble inference and subsequent refit/prediction/submission paths.
   - Validate fingerprints, schemas, class order, splits, artifact versions, OOF coverage, and provenance before scoring or fitting the meta-model.
   - Report a limitation instead of a validation score when artifacts cannot support a leakage-safe evaluation without retraining members.

5. **Integrating optional MLflow and user interfaces**
   - Extend the existing MLflow adapter for explicit model logging and retrieval by stable reference without requiring a hosted service.
   - Expose matching Python API and CLI/configuration paths for saving, selecting, and consuming model references.
   - Keep local operation independent of MLflow and provide actionable missing-extra/backend errors.
   - Document model references, artifact schemas, overwrite behavior, compatibility limits, trusted-model loading, and credential/data handling.

6. **Updating agent guidance**
   - Update the ensemble, baseline, tuning, and refit guidance to describe saved-model references and provenance where relevant.
   - Document when to use existing candidate artifacts versus saved model members, required user choices, expected outputs, and validation checks.
   - Keep guidance aligned with the public API and CLI; do not introduce an agent-only execution path.

7. **Validating contracts, leakage safety, and compatibility**
   - Add synthetic classification and regression round-trip tests, including reload after process restart and prediction equivalence.
   - Instrument estimator fitting to prove voting performs zero fits and stacking fits only the meta-model.
   - Test leakage-safe OOF provenance, alignment, coverage, and rejection of in-sample or incomplete predictions.
   - Test actionable rejection of mismatched fingerprints, schemas, classes, splits, artifact versions, and corrupted/missing artifacts.
   - Test local operation without network/MLflow, optional local-MLflow retrieval, explicit overwrite, CLI/API parity, existing artifact compatibility, and the explicit refit path.
   - Run focused producer, ensemble, refit, submission, tracking, CLI, and agent-guidance tests, followed by the full AutoML suite.
