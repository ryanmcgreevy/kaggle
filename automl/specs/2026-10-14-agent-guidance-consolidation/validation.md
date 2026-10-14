# Validation: Agent Guidance Consolidation (Phase 17)

The branch is ready to merge when every item below is true. Run from `automl/` in the project-local `.venv`.

## Automated checks

1. `pip check` passes.
2. `automl --help`, `python -m automl --help`, and each subcommand's `--help` work.
3. The full suite passes with the `reports`, `tuning`, `boosting`, and `mlflow` extras installed (the existing GPU test may skip).
4. The suite passes in a clean environment with only the `test` extra; optional-library tests skip and none error.
5. Phase 0 to 16 tests pass unchanged.
6. `tests/test_agent_guidance.py` passes and does not import optional libraries in its static checks.

## Acceptance behavior (from the roadmap)

1. Every skill (`automl-workflow`, `automl-data-intake`, `automl-validation-splits`, `automl-eda-reporting`, `automl-baseline`, `automl-tuning`, `automl-model-report`, `automl-refit`, `automl-submission`) states its invocation conditions, inputs, outputs, assumptions, limits, and validation steps under findable headings.
2. Following the `automl-workflow` order with only commands copied from the skills runs from synthetic CSVs to a schema-valid `submission.csv` in the walkthrough test, with sockets blocked and no undocumented state or credentials.
3. Each documented artifact name is produced by its command in that walkthrough, and each stage skill's inputs match the artifacts the previous stage writes.
4. The skills also name the equivalent Python API for each stage.

## Guidance checks

1. Every documented `automl` subcommand and `--option` exists in the parser; no skill documents a command or option that does not exist.
2. `AGENTS.md` links every skill and is a concise router; stage procedures are not duplicated there.
3. `AGENTS.md` restricts agents to the existing `automl` tools and requires user approval of a described plan before any new code is written or any artifact or CSV is edited by hand.
4. `AGENTS.md` and skills require user confirmation for task and metric, the tuning budget, the refit choice when not decisive, overwrites, tracking, and use of data beyond the declared files.
5. Guidance states that data stays local, that nothing is uploaded or submitted to Kaggle, that `run` needs a new or empty output directory, that tuning needs an explicit budget, and that validation scores are not leaderboard claims.
6. Each skill's optional-extra requirement is stated with its install command.
7. The nested-skill discovery caveat remains in `AGENTS.md` and no claim of verified automatic discovery is made.

## Contract and safety checks

1. No CLI, API, or package behavior changed; `src/automl/` has no diff except for any inaccuracy fix explicitly noted in the roadmap.
2. No secrets, competition data, or local-only paths appear in the skills or `AGENTS.md`.
3. Skill statements were checked against `automl <command> --help` and the README; any README inaccuracy found was corrected.

## Documentation and process

1. `specs/roadmap.md` marks Phase 17 complete with a verification note, including any deviations from this spec and any CLI gaps recorded for later.
2. No new dependency was added.
3. The branch `automl/phase-17-agent-guidance-consolidation` contains only guidance, tests, and spec/roadmap changes.
