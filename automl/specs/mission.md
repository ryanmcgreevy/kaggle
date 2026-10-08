# Mission

## Purpose

Build a general-purpose, reproducible AutoML toolkit for tabular Kaggle competitions. It should turn a competition's training and test data into an understandable analysis, validated model experiments, useful reports, and a submission file. The same capabilities must be callable by an AI assistant and usable directly by a person.

The toolkit is a set of modular tools, not an opaque one-click model picker. Each stage should expose its inputs, assumptions, outputs, and limits so a person or agent can inspect results, change a decision, and rerun only the needed work.

## Users

- Kaggle participants who want a repeatable starting workflow for tabular data.
- AI assistant agents that need dependable, composable operations and clear instructions for choosing and running them.
- Developers who want to extend the workflow with a model, metric, transformation, report section, or competition-specific skill.

## Initial Scope

The first implementation targets CSV-based tabular classification and regression competitions. It covers:

1. Inspecting train/test files and establishing the target, identifier, task type, evaluation metric, and expected submission schema.
2. Performing basic exploratory data analysis (EDA), including shape and type summaries, missingness, duplicates, cardinality, distributions, outliers, and class balance where applicable.
3. Producing a readable PDF EDA report.
4. Building a leakage-safe preprocessing and validation workflow appropriate to the observed data.
5. Training quick baselines and popular tabular boosting models.
6. Tuning selected models with Optuna, tracking experiments locally by default and through MLflow when configured.
7. Producing a PDF comparison/report for the best validated model or models.
8. Refitting a selected model and generating a Kaggle-format submission CSV after checking its schema.

The default path is CPU-capable and local. PyTorch models, remote execution, GPUs, and hosted MLflow are optional extensions, not prerequisites. Competition-specific rules and metrics may require additional code or agent-authored skills; the core should make that extension possible without embedding competition-specific assumptions in general utilities.

## Principles

- **Inspectable by default:** Record data, configuration, assumptions, validation strategy, metrics, and artifact locations for every run.
- **Reproducible:** Use explicit seeds and persist the effective configuration. Report nondeterminism when a library or accelerator prevents exact repeatability.
- **Leakage-aware:** Fit every learned preprocessing step only on the training fold during cross-validation. Keep validation data out of fitting, target encoding, feature selection, and tuning inputs except through the declared objective.
- **Human- and agent-usable:** Provide stable, composable Python capabilities and a CLI. Make outputs machine-readable as well as understandable in reports.
- **Conservative about inference:** Suggest likely target, ID, task, metric, and split choices, but expose uncertainty and require confirmation for consequential or ambiguous choices.
- **Local and private by default:** Do not upload competition data, models, or reports to a hosted service unless the user explicitly configures that destination.
- **Resource-aware:** Allow users to set time, trial, fold, row-sampling, and hardware budgets. Make expensive work opt-in or clearly visible before it begins.
- **Validated outputs:** Check predictions for row count and valid values; check submission columns, order, and identifiers against the competition's sample submission when available.
- **Extensible, not universal:** Prefer reusable mechanisms and explicit extension points over brittle heuristics that claim to solve every dataset automatically.

## Non-Goals

- Guaranteeing a leaderboard score, discovering every useful feature, or replacing domain knowledge.
- Automatically submitting to Kaggle, changing competition settings, or performing other external side effects.
- Requiring a cloud account, GPU, MLflow server, or paid service for the basic workflow.
- Treating EDA correlations, outlier flags, or feature importance as causal conclusions.
- Automatically engineering domain-specific features without making the transformation and assumptions reviewable.
- Supporting image, text, audio, time-series forecasting, or unstructured-data pipelines in the initial release.
- Hiding model selection and validation behind a single unexplained score.

## Success Criteria

For a new supported competition, a user can identify or confirm the data contract, run the workflow locally, inspect both PDF reports, compare reproducible validation results, and obtain a submission CSV that passes schema checks. An agent can perform the same work by following repository guidance and invoking documented tools, while surfacing decisions it cannot safely infer.