# Requirements: End-to-End CLI (Phase 16)

## Scope

**Deliverable (from roadmap):** Compose the implemented stages into documented CLI commands for inspection, EDA, baseline, tuning, reporting, and submission generation.

**Acceptance (from roadmap):** A no-network synthetic-data smoke test runs the full local path from CSV inputs to both PDFs and a schema-validated submission; each stage can also be invoked independently.

In scope:

- New per-stage commands over the existing Python APIs: `baseline`, `tune`, `model-report`, `refit`, `submit`. Existing `validate`, `split`, `eda`, `report` are unchanged.
- A `run` command that chains validate, split, EDA, EDA report, baseline, tuning, model report, refit, and submission into one run directory.
- Explicit artifact directory arguments for stage hand-off; no inferred or hidden state.
- Opt-in MLflow tracking via `--track`, using the config's `[tracking]` table.
- Install hints for missing optional extras (boosting libraries, Optuna, Matplotlib, MLflow) surfaced as actionable CLI errors, fulfilling the item deferred from Phase 9.
- JSON on stdout for success, actionable messages on stderr, consistent with Phase 6.
- Synthetic-fixture tests, including a chained full-path smoke test with sockets blocked.
- README documentation of every command for people and agents.

Out of scope:

- New modeling, tuning, reporting, or submission behavior; the CLI only composes existing APIs.
- Any Kaggle API call or automatic submission.
- Agent skill and `AGENTS.md` changes (Phase 17), beyond keeping existing text accurate if a command name changes.
- Interactive prompts. Task and metric confirmation stays in the TOML config.
- A CLI framework dependency; `argparse` remains.
- PyTorch, ensembling, and cloud execution.

## Decisions

| Topic | Decision |
|---|---|
| Commands | `baseline`, `tune`, `model-report`, `refit`, `submit`, `run`, added alongside the four existing commands. |
| Common arguments | Every data-reading command takes the run TOML `config` positional and an explicit `--output-dir`. Commands never read or write outside the given directories. |
| Stage hand-off | Later stages take explicit paths to earlier artifacts: optional `tune --splits <splits.json>`, optional `baseline --splits <splits.json>`, `model-report <result dir>...`, `refit <result dir>`, `submit <refit dir>`. No shared run-directory convention for individual commands. |
| Splits | `--splits` on `baseline` and `tune` is optional. When given, the saved `splits.json` is loaded and used, pinning identical folds across candidates; a missing or invalid file is an error naming the `split` command. When omitted, splits are built in memory from the config's `[validation]` table (seeded, so identical for identical config and data). These commands do not write the splits; `split` remains the way to pin folds. `model-report` warns when candidates' recorded split settings differ. |
| Baseline | `baseline` runs `run_baseline` with the config's `[baseline]` and `[preprocessing]` settings and writes `baseline.json` plus predictions via `save_baseline` into `--output-dir`. `--estimator` overrides the configured estimator. |
| Tuning | `tune` defaults to the multi-family path (`tune_families`) and writes `tuning_families.json`. `--estimator <name>` switches to single-estimator `tune` and writes `tuning.json`. `--families a,b` selects families. Budgets come from `[tuning]`, with `--n-trials` and `--timeout-seconds` overrides. A budget is required; its absence is an error, never a default. `--resume` calls the existing resume functions on an existing directory. |
| Reports | `model-report` takes one or more result directories (baseline, tuning, families) and writes `model_report.pdf` and its summary JSON. `--filename`, `--overwrite`, and `--top-n` mirror existing options. `report` stays the EDA report command; no rename. |
| Refit | `refit <result dir>` accepts `--label` or `--rank`, re-reads the train and test data from the config, and writes `predictions.csv`, `refit.json`, and `pipeline.joblib` via the existing directory persistence. |
| Submit | `submit <refit dir>` rebuilds the submission from the saved refit and the config's test and sample submission files and writes `submission.csv` and `submission.json`. Nothing contacts Kaggle; the output is for manual upload. |
| Overwrite | Matches Phase 6: new artifact files refuse overwrite; PDFs and submission files require `--overwrite`. Each command checks its output paths before doing expensive work. |
| `run` | Saves `splits/splits.json` once and passes it to every stage, and composes the stages into subfolders of `--output-dir` (`splits/`, `eda/`, `baseline/`, `tuning/`, `reports/`, `refit/`, `submission/`) and prints a JSON summary of each stage's artifact paths. It calls the same internal stage functions as the individual commands. Stage failure stops the run and reports the completed stages and the failing stage; completed artifacts are kept. |
| `run` options | `--skip-tuning`, `--skip-reports`, `--estimator`, `--families`, `--n-trials`, and `--timeout-seconds` pass through. `run` never submits or uploads and requires a new or empty `--output-dir`; it has no `--overwrite` because baseline, tuning, and split artifacts are never replaced. Missing extras, a missing tuning budget, and a missing `[tracking]` table are detected before any file is written. |
| Tracking | `--track` on `baseline`, `tune`, `refit`, `submit`, and `run` uses `config.tracking_config`. Without a `[tracking]` table, `--track` is an error naming the missing table. Without `--track`, nothing is logged even if the table exists. Tracking URIs are never printed unredacted. |
| Optional extras | An optional-dependency error from a stage is caught at the CLI boundary and printed to stderr with the existing install hint and a nonzero exit code, before any partial output is written. |
| Output | stdout is JSON only (`status`, artifact paths, key metrics, notes). Logs and errors go to stderr. |
| Exit codes | 0 success; 1 for workflow, data-contract, optional-dependency, and file errors; argparse's 2 for usage errors. |
| Seed and config | The effective configuration is persisted by each stage's existing artifact. The CLI adds no new config keys and no `[run]` table; `run` uses the existing stage tables only. |

## Context

- Phases 8 to 15 provide `run_baseline`/`save_baseline`/`load_baseline`, `tune`/`resume_tuning`/`load_tuning`, `tune_families`/`resume_tuning_families`/`load_tuning_families`, `write_model_report`, `refit_and_predict`/`load_refit`, and `build_submission`/`save_submission`. All are already Python-usable and documented in the README.
- Phase 6 `cli.py` defines `_load_run`, JSON output via `_write_json`, and the error handling for `DataContractError` and `OSError`. The new commands follow that pattern; other stage errors (`TuningError`, `SubmissionError`, and so on) must be caught at the boundary.
- Phase 9 deferred the CLI explanation of how to enable an adapter to this phase.
- Mission principles applied: composable stage contracts, local and private by default, resource-aware (budgets are explicit and visible), validated outputs, never submit automatically.

## Dependencies

No new core dependency. `argparse` remains. Optional groups (`reports`, `tuning`, `boosting`, `mlflow`) behave as before; core `import automl` and `automl --help` must work without any of them.

## Risks and open items

- `tune` is the longest-running stage. The explicit budget requirement and a printed budget summary on stderr keep expensive work visible; there is no cancellation beyond the budget.
- The `run` command can hide decisions. It must print each stage's effective choices (task, metric, estimator or families, selected candidate) and keep ambiguous task or metric choices as errors rather than guessing.
- `refit` and `submit` rely on saved artifacts and the config's data paths; if the data on disk changed since the result was produced, the existing row/ID checks are the only protection. The commands should print the data file names used.
- CLI test runtime: the smoke test must use tiny data and 2-3 trial budgets and skip cleanly when optional extras are absent.
- Open: whether `run` should accept a pre-existing `splits.json` instead of recreating it; default is to create it.
