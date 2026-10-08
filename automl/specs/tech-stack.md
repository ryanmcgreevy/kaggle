# Tech Stack

## Stack Principles

- Prefer well-supported Python data-science libraries and scikit-learn-compatible interfaces.
- Keep the required installation small enough for local CPU use; model families and tracking backends beyond the baseline are optional dependencies.
- Use the same importable Python functions from the CLI and from agent-authored workflows. Document each capability for people and agents when it is introduced; do not make notebooks the only way to run a stage or defer agent guidance until the full workflow exists.
- Define explicit stage contracts: documented inputs, outputs, assumptions, limits, and validation. Keep later stages able to consume earlier stages' structured results or saved artifacts without hidden in-memory state or rerunning unrelated stages.
- Keep configuration, metrics, and run metadata in documented, machine-readable artifacts alongside human-readable PDFs.
- Do not require network access, cloud credentials, or a running tracking server for a local workflow.

## Runtime and Core Libraries

- **Python:** Target Python 3.12 or newer for the initial implementation, consistent with the existing s6ep5 container example. Declare and test the actual supported range in package metadata before release.
- **Data handling:** pandas and NumPy for tabular inputs and results.
- **Core modeling and preprocessing:** scikit-learn for estimators, `Pipeline`/`ColumnTransformer`, imputers, encoders, splitters, metrics, and model evaluation. Learned transforms must be contained in the fitted pipeline used by cross-validation.
- **Baseline estimators:** Start with scikit-learn models, including HistGradientBoosting for a useful tabular baseline. The exact baseline set and default metrics are implementation decisions documented with the data contract.
- **Boosting integrations:** Provide optional adapters for LightGBM, CatBoost, and XGBoost where their dependencies and task compatibility are available. Missing optional packages should produce a clear installation/configuration message, not break core imports.
- **Neural models:** PyTorch is an optional extension for experiments where it is appropriate. A PyTorch estimator must meet scikit-learn cloning and parameter conventions before it is used in shared CV or Optuna workflows.
- **Hyperparameter optimization:** Optuna for seeded, budgeted search, with the study configuration and trial results persisted for later inspection or continuation.
- **Experiment tracking:** MLflow is an optional backend. Local metadata and artifacts remain the default and must be sufficient to reproduce and inspect a run. When MLflow is enabled, log the effective configuration, data/validation summary, metrics, trial details, and the complete preprocessing-plus-model pipeline where supported.

## Reports and Artifacts

- Use Matplotlib for charts and its PDF support for generating portable EDA and model-results reports. Seaborn may be used for statistical plots where it improves readability.
- Reports should be generated from structured analysis results, not notebook display state, and should remain useful when optional plotting libraries or model families are absent.
- Save machine-readable run metadata, metrics, trial history, and selected configuration as JSON or another documented text format. Save model artifacts using the format appropriate to the estimator and document Python/library compatibility constraints.
- Give persisted stage outputs stable, documented schemas sufficient for downstream stages to use and for people or agents to inspect decisions and rerun only affected work.
- Write all generated files beneath a user-selected run/output directory. Do not overwrite prior runs implicitly.

## Interfaces and Configuration

- **Python API:** Organize stages as small importable capabilities with explicit inputs and outputs. Keep competition-specific behavior injectable through configuration or adapters.
- **CLI:** Expose the same stages and end-to-end flow through a documented command-line interface. Use standard-library `argparse` initially unless the implementation demonstrates a need for a CLI framework.
- **Configuration:** Support command-line options and a versioned configuration file. Prefer TOML for simple project/run configuration because it is supported by the Python standard library; avoid committing competition data, credentials, or local-only paths.
- **Agent interface:** Document invocation guidance alongside each stable capability as it is introduced. Keep `automl/AGENTS.md` a concise router and put capability-specific procedures in on-demand `automl/.agents/skills/<skill-name>/SKILL.md` files. Skills should have keyword-rich discovery descriptions and explain tool/CLI/API usage, required inputs, expected results/artifacts, assumptions, limits, and checks. Agents use the same CLI/API as humans and may add competition-specific code or skills when the core extension points are insufficient. Verify nested skill discovery in the opened workspace; VS Code's documented project location is `.agents/skills/` at the workspace root.
- **Notebooks:** Use notebooks for exploration or examples when helpful, but not as the sole implementation or source of truth for reusable workflow logic.

## Quality and Safety

- Use pytest for unit and integration tests; use small synthetic fixtures so the core suite needs no competition data, Kaggle credentials, network access, or cloud services.
- Validate input schemas and user choices at stage boundaries. Provide actionable errors that identify the file, column, or setting involved.
- Seed supported splitters, Optuna samplers, and estimators from a shared run configuration. Record the seed and any nondeterministic behavior.
- Keep preprocessing inside CV folds. Use stratified splitters where appropriate for classification and non-stratified splitters for regression unless the data contract specifies otherwise.
- Validate submission row count and column order against the test data and sample submission when present. Preserve row-to-prediction alignment: when identifiers are available, align by identifier or reject an order mismatch; set equality alone is insufficient. Never upload or submit results automatically.
- Keep secrets out of configuration artifacts and logs. Hosted MLflow, SageMaker, and other cloud execution are explicit adapters requiring user setup and consent.

## Dependency Policy

The implementation should define a minimal core install and named optional groups for boosting integrations, PyTorch, MLflow, and development/test tooling. Pin or constrain versions in package metadata based on compatibility testing; do not copy the broad, competition-specific environment from s6ep5 wholesale. Document the command to install each optional group and test that the core package still imports and runs without them.