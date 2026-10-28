---
name: automl-data-intake
description: "Use when configuring or validating AutoML TOML runs, loading train/test CSV files, checking target or ID columns, or resolving explicit task and metric choices with the automl CLI or Python API."
---

# AutoML Data Intake

## Use when

Starting a run, or any time the TOML, CSV files, target, ID columns, task, or metric change. This is the first stage of `automl-workflow`.

## Procedure

1. Inspect the run TOML and referenced files. Config-relative CSV paths resolve from the TOML file's directory.
2. Resolve the project interpreter using `automl/AGENTS.md`, then run validation. From the repository root, an example is `./automl/.venv/bin/python -m automl validate ./playground-s6ep10/run.toml`; from that competition directory, use `../automl/.venv/bin/python -m automl validate run.toml`. Substitute the requested competition/config path. Do not assume bare `python` or `automl` is available.
3. If task or metric is unset, present the resolver's suggestions and ask for confirmation. Update `[task]` with the confirmed values, then rerun validation. Never select suggestions automatically.
4. Check the JSON result on stdout for `status`, target, task, metric, and train/test row counts. On failure, read stderr and address the named setting, file, or column.

## Inputs

A run TOML (`[data]` with train, test, optional sample submission, target, ID columns; `[task]` with task and metric) and the CSV files it names.

## Outputs

JSON on stdout only; no files are written. Next stage: `automl-validation-splits`.

## Assumptions and limits

`validate` checks the TOML schema, train/test/sample-submission contract, and task/metric compatibility with the target. It does not modify inputs or create artifacts. Keep data local; do not upload competition files or include them in prompts unless the user explicitly directs it.

## Check

Exit code 0 with `status` `valid` and the confirmed task and metric. Python API: `load_config`, `load_data`, `resolve_task_metric`. Details: Input Data Contract and Task and Metric Selection in `automl/README.md`.

The equivalent Python workflow is `load_config`, `load_data`, and `resolve_task_metric`. See the Input Data Contract and Task and Metric Selection sections of `automl/README.md` for schemas, supported values, and limits.