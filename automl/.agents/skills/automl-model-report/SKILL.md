---
name: automl-model-report
description: "Use when comparing AutoML models or producing the best-model results PDF: automl model-report, ranking of baseline, tuning, and families results, model_report.pdf, summary JSON, candidate labels, Matplotlib reports extra."
---

# AutoML Model Report

## Use when

One or more baseline or tuning result directories exist and the user wants a ranked comparison or candidate labels for refit.

## Procedure

1. Run `automl model-report run.toml runs/1/baseline runs/1/tuning --output-dir runs/1/model-reports`. Directories may be baseline, tuning, or families results, alone or mixed. Add `--top-n N` to change how many get detail pages.
2. Read the JSON for the ranked candidates and their `label` values; show them to the user. Labels such as `baseline:<estimator>` or `family:<name>` are what `automl refit --label` accepts.
3. Existing files are refused; use `--overwrite` or `--filename` only after approval.

## Inputs

Run TOML and saved result directories. The command never refits or rescores; it reads the training target only for out-of-fold plots.

## Outputs

`model_report.pdf` (or `--filename`) and `<stem>_summary.json` with candidates, excluded entries and reasons, warnings, and the score statement.

## Assumptions and limits

Candidates must share task, metric, and direction, otherwise the error names each path; differing splits give a warning, not an error. Ranking is by mean validation score, ties by lower fold std. Scores are validation scores, optimistic after selection, and are not leaderboard claims; say so when summarizing. Out-of-fold plots exist for baselines only. Needs `pip install -e '.[reports]'` (Matplotlib); the check happens before files are written.

## Check

Exit code 0, non-empty PDF, summary JSON lists the expected candidates and any excluded runs with reasons, and warnings were relayed to the user. Python API: `write_model_report`, `collect_candidates`. Details: Best-Model Results PDF in `automl/README.md`.
