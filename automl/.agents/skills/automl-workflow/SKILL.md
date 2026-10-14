---
name: automl-workflow
description: "Use when asked to run, plan, or resume the AutoML workflow for a tabular Kaggle competition end to end: stage order, automl run versus stage-by-stage commands, artifact hand-off, decisions to confirm, failure recovery."
---

# AutoML Workflow

## Use when

The user wants to go from train/test CSVs to analysis, models, reports, and a submission file, or asks which `automl` stage comes next. Load the stage skill named below for each stage's details.

## Stage order

```bash
automl validate run.toml
automl split run.toml --output-dir runs/1/splits
automl eda run.toml --output-dir runs/1/eda
automl report run.toml runs/1/eda/eda.json --output-dir runs/1/reports
automl baseline run.toml --output-dir runs/1/baseline --splits runs/1/splits/splits.json
automl tune run.toml --output-dir runs/1/tuning --splits runs/1/splits/splits.json --families hist_gradient_boosting --n-trials 30
automl model-report run.toml runs/1/baseline runs/1/tuning --output-dir runs/1/model-reports
automl refit run.toml runs/1/tuning --output-dir runs/1/refit
automl submit run.toml runs/1/refit --output-dir runs/1/submission
```

| Stage | Skill | Consumes | Writes |
|---|---|---|---|
| `validate` | `automl-data-intake` | TOML, CSVs | JSON only |
| `split` | `automl-validation-splits` | TOML, CSVs | `splits.json` |
| `eda`, `report` | `automl-eda-reporting` | TOML, CSVs; `eda.json` | `eda.json`; EDA PDF and findings JSON |
| `baseline` | `automl-baseline` | TOML, CSVs, `splits.json` | `baseline.json`, out-of-fold predictions |
| `tune` | `automl-tuning` | TOML, CSVs, `splits.json` | `tuning_families.json` or `tuning.json` |
| `model-report` | `automl-model-report` | result directories | model PDF and summary JSON |
| `refit` | `automl-refit` | result directory, CSVs | `predictions.csv`, `refit.json`, `pipeline.joblib` |
| `submit` | `automl-submission` | refit directory, CSVs | `submission.csv`, `submission.json` |

## Procedure

1. Run the data-intake stage and get the user's confirmation of task and metric before anything else.
2. Choose a new output directory per experiment. Ask before reusing or overwriting one.
3. Ask the user for the tuning budget and whether to enable `--track`; do not default either.
4. Choose a mode. Stage by stage lets the user inspect results and change decisions between stages (preferred when anything is ambiguous). `automl run run.toml --output-dir runs/2 --n-trials 30` runs every stage in one call; it needs a new or empty directory, has no `--overwrite`, and writes `splits/ eda/ baseline/ tuning/ reports/ refit/ submission/`. `--skip-tuning` refits the baseline; `--skip-reports` skips both PDFs.
5. After tuning, show the user the model comparison and confirm which candidate to refit unless rank 1 is clearly intended.
6. Finish by pointing the user to `submission.csv` for manual upload.

## Inputs

Input is one run TOML plus local train, test, and optional sample-submission CSVs. Each stage passes results to the next through explicit directory paths; no hidden state exists, so any stage can be rerun alone.

## Outputs

Stage output goes only inside `--output-dir`; see the table above for the files each stage writes.

## Assumptions and limits

Scores are validation scores on the training data, optimistic after tuning, and never leaderboard claims. Optional extras (`reports`, `tuning`, `boosting`, `mlflow`) are needed only by the stages that name them. One pipeline is refit and submitted per run. There are no interactive prompts, so confirmations happen in conversation and are written into the TOML. Nothing contacts Kaggle or the network.

## Check

On failure, read stderr (`error: ...`, exit code 1), fix the named setting, file, or extra, and rerun only that stage; completed artifacts remain. For `run`, stderr lists the completed and failing stages. Check that each stage's JSON on stdout has the expected artifact paths, and that `submission.csv` has the sample's columns in test row order.
