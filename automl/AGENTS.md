# AutoML Agent Guide

Use the same `automl` CLI and `automl` Python API as a person. Start with `automl-workflow` for any end-to-end request; load a stage skill for the stage at hand:

- Whole workflow, stage order, `automl run`, what to confirm and when: `automl-workflow`.
- Run config, train/test CSVs, target, IDs, task, metric, or `automl validate`: `automl-data-intake`.
- Holdout/k-fold splits or `splits.json`: `automl-validation-splits`.
- EDA JSON, PDF, findings, or Matplotlib: `automl-eda-reporting`.
- Baseline models, `automl baseline`, boosting estimators: `automl-baseline`.
- Optuna tuning, budgets, model families, resume: `automl-tuning`.
- Model comparison PDF and best-model ranking: `automl-model-report`.
- Hard-voting and stacking ensembles from saved model candidates: `automl-ensemble`.
- Refitting a selected model and test predictions: `automl-refit`.
- `submission.csv`, sample-submission alignment: `automl-submission`.

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