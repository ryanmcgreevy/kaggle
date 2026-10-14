# Requirements: Agent Guidance Consolidation (Phase 17)

## Scope

**Deliverable (from roadmap):** Extend the MVP's `automl/AGENTS.md` router and data/split/EDA skills with full workflow and safety guidance; add focused skills for stable model/tuning and reporting/submission capabilities.

**Acceptance (from roadmap):** Each skill states invocation conditions, inputs, outputs, assumptions, limits, and validation steps; an agent can follow the guidance to compose implemented stages through the documented Python API or CLI without undocumented state or credentials. This phase consolidates and checks guidance; it is not the first point at which capabilities become agent-usable.

In scope:

- An overall workflow skill, `automl-workflow`, that recommends the stage order, the decisions that need user confirmation, which stage artifacts feed which later stage, and when to use `automl run` versus individual commands.
- One skill per stage, matching the CLI: the three existing skills (`automl-data-intake`, `automl-validation-splits`, `automl-eda-reporting`) revised for accuracy and consistency, plus new `automl-baseline`, `automl-tuning`, `automl-model-report`, `automl-refit`, and `automl-submission`.
- A revised `AGENTS.md` router listing every skill, with the safety rules below.
- Tests that check skill structure and that every documented command, option, and artifact name exists and runs on synthetic data.

Out of scope:

- New modeling, CLI, or API behavior. If the guidance exposes a CLI gap, record it in the roadmap rather than fixing it here.
- Verifying or relocating skill auto-discovery (explicitly deferred; the existing caveat in `AGENTS.md` stays).
- An agent runtime, MCP server, or agent-only execution path.
- Competition-specific skills, PyTorch guidance (Phase 18), and user documentation beyond the skills (Phase 19).
- A separate tracking skill; `--track` and `[tracking]` are covered in each stage skill that accepts them.

## Decisions

| Topic | Decision |
|---|---|
| Skill layout | `automl/.agents/skills/<name>/SKILL.md`, unchanged location. Names: `automl-workflow`, `automl-data-intake`, `automl-validation-splits`, `automl-eda-reporting`, `automl-baseline`, `automl-tuning`, `automl-model-report`, `automl-refit`, `automl-submission`. Folder name equals frontmatter `name`. |
| Overall skill | `automl-workflow` is the entry point for "run the AutoML workflow" requests. It lists the stages in order, the artifact each hands to the next, the confirmations needed (task, metric, tuning budget, which candidate to refit, overwrites, tracking), and the choice between `automl run` (new or empty output directory, one call) and stage-by-stage commands (inspect and change decisions between stages). It points to stage skills for details and does not duplicate their commands. |
| Skill format | Each stage skill keeps the existing `Procedure` and `Contract` shape and adds explicit `Use when`, `Inputs`, `Outputs`, `Assumptions and limits`, and `Check` content, so the roadmap's five required elements are findable by heading. Descriptions stay keyword-rich for discovery. Skills stay concise; deep schemas live in `automl/README.md`, which they link to by section. |
| Tool restriction | `AGENTS.md` instructs agents to use only the existing `automl` CLI and Python API. Agents must not write new modeling, preprocessing, feature, metric, or submission code, nor bypass a stage by hand-editing artifacts or CSVs, unless they first describe the proposed code and why the existing tools are insufficient and the user approves. Approved code goes in a user-named location, uses the existing public APIs where possible, and is not added to the `automl` package unless the user asks. This also applies to ad hoc scripts that read competition data. |
| Confirmation rules | Skills tell agents to ask the user and wait for: task and metric when unset or ambiguous; the tuning budget (never choose a default); which model to refit when ranking is not decisive; any overwrite; enabling `--track`; and any step that would use data beyond the declared train/test/sample files. |
| Safety rules in `AGENTS.md` | Local-only data and artifacts; no Kaggle upload or submission, including by suggesting commands that do so; no secrets in configs, prompts, or logs; no overwrite without explicit instruction; use synthetic fixtures for checks; report validation scores as validation, never as leaderboard expectations. |
| Accuracy source | Skill text is derived from the implemented CLI (`automl <command> --help`) and `automl/README.md`, not from the specs. Where a spec deviated from the implementation (see roadmap Phase 11 to 16 notes), the skill follows the implementation. |
| Existing skills | `automl-data-intake`, `automl-validation-splits`, and `automl-eda-reporting` are revised only to add the missing required elements, remove stale statements (for example that later stages are undocumented), and match the new router. Their commands and behavior are not changed. |
| Validation tests | A new `tests/test_agent_guidance.py` parses every `SKILL.md` for valid frontmatter, folder/name match, non-empty description, and required headings; checks `AGENTS.md` links to every skill; extracts documented `automl ...` commands and verifies each subcommand and option exists in the argument parser; and runs a synthetic-data walkthrough that follows the `automl-workflow` order using only commands taken from the skills, with sockets blocked. |
| Optional libraries | Tests that need Matplotlib, Optuna, MLflow, or boosting libraries skip when they are absent. Static checks never import optional libraries. |
| Dependencies | None added. |

## Context

- Mission: agents use the same CLI and Python API as people, must surface decisions they cannot safely infer, and never submit to Kaggle. Principles on conservative inference and local-only handling drive the confirmation and restriction rules.
- Tech stack: `AGENTS.md` stays a concise router; capability-specific procedures live in on-demand skills with keyword-rich descriptions covering usage, inputs, results and artifacts, assumptions, limits, and checks.
- Phase 6 delivered the router and three skills; Phases 7 to 16 added capabilities that were documented in the README but not in skills. Phase 16's CLI makes every stage reachable from the command line, so skills can be CLI-first with Python API equivalents noted.
- Discovery of nested skills from the repository-root workspace is still unverified. This phase keeps the current location and caveat.
- `run` requires a new or empty output directory and has no `--overwrite`, and `tune` requires an explicit trial or time budget. The workflow skill must state both.

## Open items for the user

None blocking. If the walkthrough test exposes a stage-hand-off gap that cannot be explained in guidance alone, the gap goes to the roadmap as a follow-up instead of a code change in this phase.
