# AutoML Agent Guide

Use the same `automl` CLI and `automl` Python API as a person. Load the task-specific skill for the workflow at hand:

- Run config, train/test CSVs, target, IDs, task, metric, or `automl validate`: `automl-data-intake`.
- Holdout/k-fold splits or `splits.json`: `automl-validation-splits`.
- EDA JSON, PDF, findings, or Matplotlib: `automl-eda-reporting`.

Confirm task and metric with the user when absent or ambiguous; never silently infer them. Keep data and artifacts local, use synthetic fixtures for checks, and do not overwrite existing outputs without explicit instruction. Skills contain the detailed commands, artifact expectations, and validation steps. Current CLI scope ends at EDA reporting; it does not train models or create submissions.

VS Code's documented project-skill location is `.agents/skills/` at the opened workspace root. These AutoML skills are nested under `automl/`; if they are not discovered while the repository root is open, open `automl/` as the workspace or load the relevant `SKILL.md` directly. Do not assume automatic discovery until verified.