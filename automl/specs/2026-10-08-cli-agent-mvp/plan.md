# Plan: Current-Capability CLI and Agent MVP (Phase 6)

## Goal

Expose the functionality completed through Phase 5 as one composable, local CLI workflow. Give coding agents concise routing plus task-specific, progressively loaded instructions for using the same CLI and Python APIs as people.

## Implementation

1. Add `validate`, `split`, `eda`, and `report` argparse subcommands. Compose existing public APIs; do not duplicate stage logic.
2. Make each successful command print a JSON result to stdout. Send domain and filesystem errors to stderr with exit status 1; retain argparse's status 2 for usage errors.
3. Require explicit `--output-dir` for generated artifacts. Write `splits.json` and `eda.json` there; never overwrite those artifacts. The report command writes its PDF and findings JSON and requires `--overwrite` to replace either.
4. Add a concise `AGENTS.md` dispatcher and focused skills for input/task selection, validation splits, and EDA/reporting. Document confirmation points, outputs, local privacy, optional dependencies, and checks.
5. Add synthetic CLI integration tests and update the README, tech-stack guidance, and roadmap. Keep later modeling and full-workflow phases intact except for sequential renumbering.

## Out of Scope

Preprocessing, models, tuning, model comparison reports, prediction, submissions, cloud services, automatic task/metric selection, and an MCP server or autonomous agent runtime.