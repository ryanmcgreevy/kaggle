---
name: automl-data-intake
description: "Use when configuring or validating AutoML TOML runs, loading train/test CSV files, checking target or ID columns, or resolving explicit task and metric choices with the automl CLI or Python API."
---

# AutoML Data Intake

## Procedure

1. Inspect the run TOML and referenced files. Config-relative CSV paths resolve from the TOML file's directory.
2. Run `automl validate run.toml` (or `python -m automl validate run.toml`).
3. If task or metric is unset, present the resolver's suggestions and ask for confirmation. Update `[task]` with the confirmed values, then rerun validation. Never select suggestions automatically.
4. Check the JSON result on stdout for `status`, target, task, metric, and train/test row counts. On failure, read stderr and address the named setting, file, or column.

## Contract

`validate` checks the TOML schema, train/test/sample-submission contract, and task/metric compatibility with the target. It does not modify inputs or create artifacts. Keep data local; do not upload competition files or include them in prompts unless the user explicitly directs it.

The equivalent Python workflow is `load_config`, `load_data`, and `resolve_task_metric`. See the Input Data Contract and Task and Metric Selection sections of `automl/README.md` for schemas, supported values, and limits.