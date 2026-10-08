# Requirements: Current-Capability CLI and Agent MVP (Phase 6)

## Scope

Implement a CLI over existing Phase 0-5 APIs and make it usable by people and coding agents through the same interfaces.

In scope:

- `automl validate CONFIG`: load TOML and CSV inputs, enforce the data contract, and resolve configured task and metric.
- `automl split CONFIG --output-dir DIR`: create `DIR/splits.json` using configured validation settings.
- `automl eda CONFIG --output-dir DIR`: create `DIR/eda.json` using configured EDA settings and resolved task/metric.
- `automl report CONFIG EDA_JSON --output-dir DIR [--filename FILE] [--overwrite]`: create a PDF and findings JSON using report configuration from TOML.
- JSON success summaries on stdout; actionable workflow errors on stderr; synthetic tests and documentation.
- A concise `automl/AGENTS.md` router and three task-specific skills under `automl/.agents/skills/`.

Out of scope: preprocessing, training, tuning, model reports, prediction, submission, network access, Kaggle credentials, MCP, and an autonomous agent runtime.

## Decisions

| Topic | Decision |
|---|---|
| CLI framework | Standard-library `argparse`; no new dependencies. |
| Input paths | Config-relative paths retain `load_config` behavior. |
| Output paths | Output directories are resolved from the invoking process's working directory; parent directories are created as needed. |
| Task and metric | Both must be explicit in the config. Existing resolver suggestions are errors requiring user confirmation, not automatic choices. |
| Success output | Each command prints a JSON object to stdout with status and relevant artifact/run details. |
| Errors | Domain and filesystem errors print to stderr and return 1. Argparse usage errors retain its standard return code 2. |
| Overwrite | Split/EDA use existing non-overwriting API behavior. Reports refuse existing PDF/findings outputs unless `--overwrite` is supplied. |
| Optional PDF | Core CLI and EDA work without Matplotlib. `report` returns the existing actionable `.[reports]` installation message if Matplotlib is absent. |
| Agent interface | Agents use the same CLI/API. `AGENTS.md` routes; skill files contain task-specific procedures and discovery metadata. |
| Skill location | Keep skills at `automl/.agents/skills/` as selected; verify discovery with the repository root open before claiming it is automatic. |

## Command Results

- `validate`: `status`, config path, target, resolved task and metric, train/test row counts, optional sample-submission row count.
- `split`: `status`, artifact path, strategy, task, row count, and split count.
- `eda`: `status`, artifact path, EDA schema version, row count, and column count.
- `report`: report result fields from `ReportResult.to_dict()` plus `status`, including PDF and findings paths, pages, charts, skipped charts, and findings.

The durable split and EDA schemas remain those already implemented by their Python APIs. Report's PDF filename must remain a plain `.pdf` filename as enforced by `write_eda_report`.

## Agent Guidance

`AGENTS.md` must remain an index and shared safety contract, not duplicate the detailed procedures. Skills:

- `automl-data-intake`: TOML, CSV contract, explicit target/task/metric, `validate`.
- `automl-validation-splits`: holdout/k-fold settings, `split`, `splits.json` checks.
- `automl-eda-reporting`: `eda`, saved EDA artifacts, `report`, findings, optional Matplotlib.

Descriptions must identify when each skill applies. Guidance must include commands/tool use, expected outputs, validation, local-only data handling, and explicit decision points.