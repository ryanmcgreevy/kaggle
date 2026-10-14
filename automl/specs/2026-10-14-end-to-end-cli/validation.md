# Validation: End-to-End CLI (Phase 16)

The branch is ready to merge when every item below is true. Run from `automl/` in the project-local `.venv`.

## Automated checks

1. `pip check` passes.
2. `automl --help`, `python -m automl --help`, and each subcommand's `--help` work, including without optional extras.
3. The full suite passes with the `reports`, `tuning`, `boosting`, and `mlflow` extras installed (the existing GPU test may skip).
4. The suite passes in a clean environment with only the `test` extra; optional-library tests skip and none error.
5. Phase 0 to 15 tests, including the existing `validate`, `split`, `eda`, and `report` CLI tests, pass unchanged.
6. `import automl` succeeds with Optuna, Matplotlib, MLflow, and the boosting libraries blocked.

## Acceptance behavior (from the roadmap)

1. Full path: a chained no-network smoke test runs `validate`, `split`, `eda`, `report`, `baseline`, `tune`, `model-report`, `refit`, and `submit` from synthetic CSVs and produces the EDA PDF, the model report PDF, and a `submission.csv` whose columns equal the sample submission and whose rows follow test order.
2. Independence: each stage command runs on its own from the documented artifact directory, with no hidden state and no rerun of unrelated stages.
3. `run` produces the same artifact set as the chained commands for a fixed seed, in its documented subfolders, and its JSON summary lists each stage's paths and key choices.
4. Sockets are blocked during the smoke tests and no Kaggle or network call occurs.

## Command behavior

1. `baseline` and `tune` fail with an actionable error naming `split` when a given `--splits` file is missing or invalid; when `--splits` is omitted they build splits from the config and produce the same scores as with a saved `split` artifact for the same config and data.
2. `tune` fails without a trial or time budget, runs multi-family by default, runs a single estimator with `--estimator`, and `--resume` continues an existing study and rejects mismatches.
3. `model-report` accepts baseline, tuning, and families directories alone and mixed, and does not overwrite an existing PDF without `--overwrite`.
4. `refit` honors `--label` and `--rank`; `submit` writes `submission.csv` and `submission.json` and refuses to overwrite without `--overwrite`.
5. `--track` logs to a local SQLite MLflow store and the run can be queried; `--track` without a `[tracking]` table errors; without `--track`, nothing is logged. Tracking URIs are redacted in output.
6. A missing optional library produces the stage's install hint on stderr, exit code 1, and no partial output files.
7. `run` stops on a failing stage, keeps completed artifacts, reports the completed and failing stages, and refuses a non-empty `--output-dir`.
8. Successful commands print JSON only to stdout; errors print `error: ...` to stderr with exit code 1; usage errors exit 2.

## Contract and safety checks

1. Input CSVs, the config file, `splits.json`, and saved result directories are byte-identical after each command.
2. No command writes outside its `--output-dir`, and none overwrites an existing artifact implicitly.
3. Ambiguous or unset task and metric choices still raise errors from the CLI instead of being guessed.
4. No secrets or raw data appear in stdout, stderr, or logged MLflow content.

## Documentation and process

1. `automl/README.md` documents every command for people and agents: invocation, inputs, outputs, assumptions, limits, exit codes, and optional-extra install commands.
2. The README full-path example runs end to end on synthetic CSVs.
3. `specs/roadmap.md` marks Phase 16 complete with a verification note, including any deviations from this spec.
4. `AGENTS.md` and existing skills remain accurate; any larger rewrite is left to Phase 17.
5. No new core dependency was added; optional dependency changes, if any, are in `pyproject.toml` and documented.
6. No competition data, credentials, or local-only paths are committed.
