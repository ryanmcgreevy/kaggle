# Plan: Agent Guidance Consolidation (Phase 17)

## 1. Source inventory

1. Collect the exact CLI surface from `automl --help` and each `automl <command> --help`: arguments, options, exit codes, and output JSON keys.
2. Read the README sections for each stage to capture inputs, output artifacts and file names, assumptions, limits, and optional extras and their install commands.
3. Read the three existing skills and the current `AGENTS.md`; list statements that are stale or incomplete.
4. Record in a short working list any place where the CLI and README disagree, and follow the implementation.

## 2. `AGENTS.md` router

1. Rewrite `automl/AGENTS.md` as a concise router: start with `automl-workflow` for end-to-end requests, then one line per stage skill with its trigger keywords.
2. Add the tool restriction: use only the existing `automl` CLI and Python API; do not write new modeling, preprocessing, feature, metric, or submission code, or edit artifacts or CSVs by hand, unless the user approves a description of the code and why existing tools are insufficient first.
3. Add the confirmation rules (task, metric, tuning budget, refit choice, overwrites, tracking, extra data) and safety rules (local only, no Kaggle upload or submission, no secrets, no implicit overwrite, validation scores are not leaderboard claims).
4. Keep the existing nested-skill discovery caveat unchanged.

## 3. Workflow skill

1. Create `automl/.agents/skills/automl-workflow/SKILL.md` with the stage order: validate, split, eda, report, baseline, tune, model-report, refit, submit.
2. Give a table of each stage's inputs, the artifact it writes, and the next stage that consumes it, plus the decisions to confirm before each stage.
3. Explain when to use `automl run` (new or empty output directory; budget and `--skip-*` options) versus stage-by-stage commands, and what `run` writes into `splits/`, `eda/`, `baseline/`, `tuning/`, `reports/`, `refit/`, `submission/`.
4. State the recovery path when a stage fails: read stderr, fix the named setting, rerun only that stage; completed artifacts stay.

## 4. Revise existing stage skills

1. Update `automl-data-intake`, `automl-validation-splits`, and `automl-eda-reporting` to include `Use when`, `Inputs`, `Outputs`, `Assumptions and limits`, and `Check` content, remove stale text, and point to the workflow skill and neighbouring stages.
2. Keep commands and described behavior unchanged; confirm each statement against the CLI help and README.

## 5. New stage skills

1. `automl-baseline`: `automl baseline`, `--estimator`, `--splits`, `--track`; outputs `baseline.json` and predictions; boosting estimators need the `boosting` extra; checking finite metrics and the effective configuration.
2. `automl-tuning`: `automl tune`, required `--n-trials` or `--timeout-seconds` (ask the user, never default), `--estimator` versus default multi-family, `--families`, `--resume`, `--track`; outputs `tuning_families.json` or `tuning.json`; Optuna and boosting extras; failed-trial categories and what they mean.
3. `automl-model-report`: `automl model-report` over one or more result directories, `--top-n`, `--filename`, `--overwrite`; outputs the PDF and summary JSON; Matplotlib needed; scores are validation scores, not leaderboard claims; warnings on differing split settings.
4. `automl-refit`: `automl refit`, `--label` or `--rank`, how to choose with the user, `--track`; outputs `predictions.csv`, `refit.json`, `pipeline.joblib`; validity checks on predictions; note that joblib artifacts must only be loaded from trusted runs.
5. `automl-submission`: `automl submit`, sample-submission alignment rules, `--overwrite`, `--track`; outputs `submission.csv` and `submission.json`; never uploads and the user submits manually.
6. In each skill, name the Python API equivalents (for example `run_baseline`, `tune_families`, `write_model_report`, `refit_and_predict`, `build_submission`) and the install command for any optional extra.

## 6. Tests

1. Add `tests/test_agent_guidance.py` with static checks: every `SKILL.md` has valid frontmatter, folder equals `name`, a non-empty keyword-rich description, and the required headings; every skill is linked from `AGENTS.md`; `AGENTS.md` contains the tool restriction and the no-submission rule.
2. Extract each fenced `automl ...` command from the skills and verify the subcommand and every `--option` exist in the argument parser (via parser introspection, no execution).
3. Add a synthetic-data walkthrough that runs the commands in `automl-workflow` order using the documented arguments and artifact paths, with sockets blocked, ending in a schema-valid `submission.csv`; skip when required extras are absent.
4. Verify that a skill-documented artifact file name is produced by the corresponding command in the walkthrough.
5. Keep the existing suite passing unchanged, including without optional extras.

## 7. Verification and roadmap

1. Run `pip check`, CLI help, and the full suite with all extras; run the suite in a clean environment with only the `test` extra and confirm the new tests skip rather than error.
2. Manually read each skill once against its CLI help for accuracy.
3. Update `automl/README.md` only where it should link to the skills or where the review found an inaccuracy.
4. Mark Phase 17 complete in `specs/roadmap.md` with a verification note and any deviations from this spec, leaving the discovery caveat in the Phase 6 note.
