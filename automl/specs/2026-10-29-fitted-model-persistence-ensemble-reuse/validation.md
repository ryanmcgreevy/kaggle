# Fitted-Model Persistence and Ensemble Reuse Validation

## Acceptance Checks

### Shared contract and local persistence

- Synthetic binary, multiclass, and regression runs save complete preprocessing-plus-estimator pipelines with a stable reference and all required metadata.
- A new process can load a local artifact and reproduce the original pipeline's predictions.
- Tests verify metadata captures task, metric, feature schema, class order where relevant, effective configuration, dependency versions, data fingerprint, validation splits, and fit provenance.
- Tests distinguish fold-trained models from models fitted on all declared training data.
- OOF predictions preserve row alignment, identify their split/fold provenance, and declare complete coverage; missing, duplicate, or ambiguous rows are rejected.
- Local persistence and loading succeed with sockets blocked and MLflow/optional integrations unavailable.

### Ensemble reuse and leakage safety

- Saved hard-voting classification performs zero estimator fits.
- Saved classification and regression stacking fit only the meta-model; member fitting is instrumented to confirm no member is trained or rebuilt.
- Stacking consumes only compatible, complete, leakage-safe OOF member predictions and uses saved full-data member pipelines for inference.
- Tests reject mismatched data fingerprints, feature schemas, class orders, split definitions, incomplete OOF coverage, incompatible artifact versions, and any in-sample fallback.
- Ensemble validation is emitted only when provenance demonstrates that neither member training nor meta-model training used the rows being scored. When this cannot be demonstrated without retraining members, the result reports the limitation and does not present a training score as validation.
- Subsequent ensemble prediction and submission generation preserve test-row alignment and existing validity checks.

### Backend, errors, and compatibility

- The optional local-MLflow integration logs and retrieves a model by stable reference and reproduces equivalent predictions without a hosted service.
- Local storage remains usable without MLflow; missing optional dependencies produce actionable errors.
- Missing, unavailable, corrupt, or incompatible artifacts fail clearly without silently triggering member training.
- Explicit overwrite rules are tested for model and metadata artifacts, including no partial artifact on failure.
- Existing candidate artifacts remain usable in their documented workflows, and explicit refit behavior remains available.
- Tests verify credentials and raw competition data are not written into model metadata or logs.

### API, CLI, documentation, and regression checks

- Python API and CLI offer documented, equivalent operations for model persistence and saved-member ensemble use.
- Config and command validation identify invalid references, backend choices, and overwrite requests with actionable errors.
- API, CLI, artifact schema, model trust/version constraints, and agent guidance are updated in the same increment.
- Focused baseline, tuning/family, ensemble, refit, submission, tracking, CLI, and agent-guidance regression tests pass.
- The full AutoML test suite passes; no new core dependency, competition data, network access, cloud credentials, or hosted service is required.

## Suggested Validation Sequence

1. Run focused local model contract, persistence, and restart round-trip tests.
2. Run producer tests for baseline, tuning/families, refit, and OOF provenance.
3. Run ensemble fit-count, leakage, compatibility, inference, and submission tests.
4. Run CLI/API, overwrite, backward-compatibility, and missing-optional-dependency tests.
5. Run the optional integration against a local MLflow store.
6. Run agent-guidance checks and the full AutoML test suite, including a local/socket-blocked path.

## Verification Result

- `cd automl && .venv/bin/python -m pytest tests -q`: **635 passed, 1 skipped**. The skip requires an available GPU; the local MLflow integration passed with one SQLAlchemy deprecation warning.
- Focused CLI, local model-store, refit atomic-overwrite, and legacy ensemble fingerprint checks passed.
- `git diff --check` and Python `compileall` passed.
