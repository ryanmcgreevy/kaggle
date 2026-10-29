# AutoML Agent Guide

Use the same `automl` CLI and `automl` Python API as a person. Start with `automl-workflow` for any end-to-end request; load a stage skill for the stage at hand:

- Whole workflow, stage order, `automl run`, what to confirm and when: `automl-workflow`.
- Run config, train/test CSVs, target, IDs, task, metric, or `automl validate`: `automl-data-intake`.
- Holdout/k-fold splits or `splits.json`: `automl-validation-splits`.
- EDA JSON, PDF, findings, or Matplotlib: `automl-eda-reporting`.
- Baseline models, `automl baseline`, boosting estimators: `automl-baseline`.
- Optuna tuning, budgets, model families, resume: `automl-tuning`.
- Model comparison PDF and best-model ranking: `automl-model-report`.
- Hard-voting/stacking from result candidates or saved fitted-model references: `automl-ensemble`.
- Refitting a selected model and test predictions: `automl-refit`.
- `submission.csv`, sample-submission alignment: `automl-submission`.

## Python environment and paths

- From the repository root, use `./automl/.venv/bin/python -m automl`; from a sibling competition directory, use `../automl/.venv/bin/python -m automl`; from `automl/`, use `./.venv/bin/python -m automl`.
- The existing `automl/.venv` contains the editable package installation. Do not assume bare `python` or `automl` is on PATH, or that system `python3` has the dependencies.
- Skill commands written as `automl ...` are shorthand: replace that prefix with the project interpreter invocation above. Use the same interpreter for Python API calls.
- Check the working directory before constructing relative paths. CSV paths in TOML resolve relative to the TOML; command-line config, artifact, and output paths resolve relative to the working directory.
- Shell activation and directory changes do not necessarily persist between tool calls; use explicit interpreter paths and set the working directory in each command as needed.
- If the environment is missing or imports fail, report the missing environment/dependency and follow the README setup instructions when installation is authorized. Do not work around it with system Python, `PYTHONPATH`, or global installs.

## Tool restriction

Use only the existing `automl` CLI and Python API. Do not write new modeling, preprocessing, feature, metric, tuning, or submission code, and do not edit artifacts, predictions, or CSVs by hand, unless you first describe the proposed code and why the existing tools are insufficient and the user approves. Approved code goes where the user names, uses the public `automl` API where possible, and is not added to the `automl` package unless asked. This includes ad hoc scripts that read competition data. If a stage cannot do what is needed, say so instead of working around it.

## Confirm with the user and wait

- Task and metric when absent or ambiguous; never infer them silently.
- The tuning budget (`--n-trials` or `--timeout-seconds`); never pick one.
- Which model to refit, when the ranking is not decisive.
- Any overwrite of an existing output, and enabling `--track`.
- Any use of data beyond the declared train, test, and sample-submission files.

## Safety

- Keep data and artifacts local; do not upload them or paste them into prompts unless the user directs it.
- Never submit or upload to Kaggle, or suggest commands that do. `submission.csv` is for the user to upload manually.
- Keep secrets out of configs, prompts, and logs; tracking URIs may carry credentials.
- Do not overwrite existing outputs without explicit instruction; use a new output directory.
- Use synthetic fixtures for checks. Report scores as validation scores, never as leaderboard expectations.
- `joblib` files (`pipeline.joblib`, `models.joblib`) are pickles; load them only from runs you trust.

VS Code's documented project-skill location is `.agents/skills/` at the opened workspace root. These AutoML skills are nested under `automl/`; if they are not discovered while the repository root is open, open `automl/` as the workspace or load the relevant `SKILL.md` directly. Do not assume automatic discovery until verified.