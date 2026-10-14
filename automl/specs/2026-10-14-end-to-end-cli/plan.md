# Plan: End-to-End CLI (Phase 16)

## 1. CLI structure and shared helpers

1. Refactor `src/automl/cli.py` so each command has a small handler that calls a stage function returning a JSON-ready dict; `run` reuses these same functions rather than shelling out.
2. Add shared helpers: stage output-path preflight (refuse overwrite before heavy work), optional splits loading (error naming the `split` command when a given file is bad) with a config-built fallback, and a tracking resolver that errors when `--track` is given without a `[tracking]` table.
3. Extend the boundary error handling to cover every stage error type (`TuningError`, `BaselineError`, `RefitError`, `SubmissionError`, tracking and optional-dependency errors) with exit code 1 and the original actionable message on stderr.
4. Keep stdout JSON-only, `import automl` free of optional libraries, and `automl --help` working without extras.

## 2. Baseline and tune commands

1. Add `baseline` (`config`, optional `--splits`, `--output-dir`, `--estimator`, `--track`) calling `run_baseline` and `save_baseline`.
2. Add `tune` (`config`, optional `--splits`, `--output-dir`, `--estimator`, `--families`, `--n-trials`, `--timeout-seconds`, `--resume`, `--track`): multi-family by default, single estimator with `--estimator`; require a budget; print the budget and estimators to stderr before starting.
3. Wire `--resume` to `resume_tuning` / `resume_tuning_families` and surface mismatch errors.
4. Turn missing boosting, Optuna, and MLflow libraries into install-hint errors raised before any output is written.

## 3. Report, refit, and submit commands

1. Add `model-report` (`config`, one or more result directories, `--output-dir`, `--filename`, `--top-n`, `--overwrite`) calling `write_model_report`; pass the train target when available for out-of-fold plots.
2. Add `refit` (`config`, result directory, `--output-dir`, `--label`, `--rank`, `--overwrite`, `--track`) calling `refit_and_predict` with the config's train and test data.
3. Add `submit` (`config`, refit directory, `--output-dir`, `--overwrite`, `--track`) calling `build_submission` and `save_submission`, using the config's test and sample submission.
4. Print the data file names used and the selected candidate in each command's JSON output.

## 4. `run` command

1. Add `run` (`config`, `--output-dir`, `--skip-tuning`, `--skip-reports`, `--estimator`, `--families`, `--n-trials`, `--timeout-seconds`, `--track`) chaining the stage functions into `splits/`, `eda/`, `baseline/`, `tuning/`, `reports/`, `refit/`, `submission/`.
2. Select the refit source as the best tuned candidate (rank 1) when tuning ran, otherwise the baseline, and record that choice in the output JSON.
3. On a stage failure, stop, keep completed artifacts, and report the completed stages and the failing one on stderr with exit code 1.
4. Require a new or empty `--output-dir` (no `--overwrite`); never contact Kaggle.

## 5. Tests

1. Add `tests/test_cli_stages.py` using synthetic CSVs and TOML: each new command's success JSON and exit code, bad `--splits` file, omitted `--splits` matching the saved-splits result for the same config, missing budget, `--track` without `[tracking]`, and overwrite refusal and `--overwrite`.
2. Test optional-dependency errors by blocking Optuna, Matplotlib, MLflow, and each boosting library, asserting the install hint, exit code 1, and no output files.
3. Add a chained smoke test (`validate`, `split`, `eda`, `report`, `baseline`, `tune`, `model-report`, `refit`, `submit`) with sockets blocked, ending in both PDFs and a schema-valid `submission.csv` equal to the sample's columns and test row order.
4. Add a `run` smoke test checking the same artifacts and that its stage outputs equal the individually invoked ones for a fixed seed.
5. Test `--resume`, single-estimator tuning, `--skip-tuning`, `--skip-reports`, failure mid-run keeping completed artifacts, and input files byte-identical afterward.
6. Test MLflow `--track` against a local SQLite store, skipping when MLflow is absent.
7. Keep Phase 0 to 15 tests passing unchanged, including the existing CLI tests.

## 6. Documentation

1. Update `automl/README.md` with every command: invocation, inputs, outputs, assumptions, limits, exit codes, and a full-path example on synthetic CSVs, for people and agents using the same CLI.
2. Document the optional-extra install commands next to the install-hint behavior and the explicit-budget requirement for tuning.
3. Update `AGENTS.md` and existing skills only where current text is made inaccurate; defer consolidation to Phase 17.
4. Mark Phase 16 complete in `specs/roadmap.md` with a verification note and any deviations from this spec.
