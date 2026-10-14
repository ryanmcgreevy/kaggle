# Roadmap

This roadmap builds the workflow in small increments. Each phase should leave the project runnable, add a focused acceptance check, and avoid requiring Kaggle data, credentials, network access, cloud services, or a GPU unless that phase explicitly tests an optional integration.

## Cross-Cutting Acceptance

Every phase that introduces or changes a reusable capability must document its invocation and its inputs, outputs, assumptions, limits, and validation checks in the same increment. Keep the Python API usable independently; when a CLI command exists for the capability, document that invocation too. This guidance is for both people and agents and must use the same API/CLI, not an agent-only execution path.

Stages should return structured results and persist machine-readable artifacts when they produce outputs needed by later stages. A later stage must be able to consume the documented result or artifact without relying on hidden in-memory state or rerunning unrelated work. Add focused tests for stage boundaries and artifact contracts as those boundaries are introduced.

## Phase 0: Package and Test Skeleton

**Status:** Complete (2026-10-08).

**Deliverable:** Minimal Python package layout, project metadata, test configuration, and a CLI entry point that prints help.

**Acceptance:** Install the core package in a clean environment, import it, run the CLI help command, and pass the empty/skeleton test suite without optional model or tracking packages.

**Verification:** Installed into the project-local Python 3.12 `.venv`; package import, `automl --help`, `python -m automl --help`, `pip check`, and all 4 tests passed. The initial fresh-environment install downloaded build/test dependencies; subsequent tests and CLI checks run locally without network access.

## Phase 1: Input Data Contract

**Status:** Complete (2026-10-08).

**Deliverable:** A documented run configuration and loader for train CSV, test CSV, optional sample submission, target, and identifier columns.

**Acceptance:** Synthetic fixtures cover valid files, missing files, duplicate columns, train/test column mismatches, and invalid target/ID configuration with actionable errors.

**Verification:** Installed into the project-local `.venv`; `pip check` and all 18 tests passed, covering valid inputs, missing/empty files, duplicate columns, train/test mismatches, target/ID/sample-submission errors, config errors, and input non-mutation. The only new core dependency is `pandas`; no CLI behavior was added.

## Phase 2: Task and Metric Selection

**Status:** Complete (2026-10-08).

**Deliverable:** Explicit classification/regression task and metric configuration, with safe suggestions where task type or metric is ambiguous.

**Acceptance:** Tests verify supported binary, multiclass, and regression metric/task combinations, and verify that ambiguous or unsupported choices are surfaced instead of silently guessed.

**Verification:** Installed into the project-local `.venv`; `pip check`, CLI help, and all 40 tests passed, covering every built-in task/metric pair, unset/unknown/mismatched choices, target inconsistencies, ambiguous integer targets, custom metrics, and `[task]` TOML parsing. The only new core dependency is `scikit-learn`; no CLI behavior was added.

## Phase 3: Validation Strategy

**Status:** Complete (2026-10-08).

**Deliverable:** Reproducible holdout and cross-validation split construction appropriate to the selected task, with configurable seed and fold count.

**Acceptance:** Tests prove repeatable splits for a fixed seed, no train/validation overlap, and valid stratification when class counts permit it; invalid fold/class combinations return a useful error.

**Verification:** Installed into the project-local `.venv`; `pip check` and all 66 tests passed, covering seeded repeatability, no overlap, k-fold partitioning, stratification, holdout size, config/`[validation]` errors, rare-class and null-target errors, and the JSON split artifact round trip. No new dependency; no CLI behavior was added.

## Phase 4: EDA Summary

**Status:** Complete (2026-10-08).

**Deliverable:** A reusable, non-mutating EDA stage that summarizes dimensions, dtypes, missingness, duplicates, numeric distributions/outliers, categorical cardinality, target distribution, and train/test differences.

**Acceptance:** Synthetic mixed-type data produces complete structured results, including edge cases such as constant columns, all-null columns, and absent optional targets in test data.

**Verification:** Installed into the project-local `.venv`; `pip check`, CLI help, and all 94 tests passed, covering dataset/column/target/train-test sections, outlier methods against hand-computed values, constant, all-null, and non-finite columns, drift tests, `[eda]` config, JSON-safe output, input non-mutation, and the artifact round trip. `scipy` was declared explicitly (already transitive via scikit-learn); no plotting library or CLI behavior was added.

## Phase 5: EDA PDF

**Status:** Complete (2026-10-08).

**Deliverable:** PDF report generation from the structured EDA results, with charts and concise data-quality findings.

**Acceptance:** A test run produces a non-empty, readable PDF in the requested output directory; report generation does not mutate input data or require a notebook session.

**Verification:** Installed into the project-local `.venv` with the new optional `reports` extra (`matplotlib`); `pip check`, CLI help, and all 123 tests passed, covering PDF validity and page counts, generation from a saved EDA artifact, skipped charts, finding thresholds and ordering, chart caps, overwrite and atomic-write behavior, a missing-Matplotlib message, core import without Matplotlib, `[report]` config, and input non-mutation. Core dependencies are unchanged; no CLI behavior was added.

## Phase 6: Current-Capability CLI and Agent MVP

**Status:** Implemented (2026-10-08); automatic skill discovery remains unverified.

**Deliverable:** A focused CLI over the existing input, task/metric, validation-split, EDA, and EDA-report APIs, plus concise `AGENTS.md` routing and on-demand skills for data intake, splits, and EDA/reporting. The agent interface uses the same CLI/Python API as people; no agent runtime or MCP server is introduced.

**Acceptance:** Synthetic TOML/CSV fixtures exercise `validate`, `split`, `eda`, and `report` through the CLI. Successful commands emit JSON to stdout; workflow errors are actionable on stderr. Split and EDA artifacts refuse overwrite, PDF/findings overwrite requires `--overwrite`, and PDF works only with the optional reports extra. Agent guidance covers explicit task/metric confirmation, local-only handling, artifacts, and validation; skill descriptions support progressive discovery without duplicating workflows in `AGENTS.md`.

**Verification:** From `automl/`, all 128 tests passed, `pip check` passed, and both module and installed CLI help/validation checks passed. Synthetic tests cover all four commands, artifact loading, overwrite rules, and report behavior. Skill frontmatter and folder/name matching passed static checks. The selected skill path is `automl/.agents/skills/`; automatic discovery from the repository-root workspace could not be verified. Do not claim it until checked in VS Code; if not discoverable, revisit location with the user.

## Phase 7: Leakage-Safe Preprocessing

**Status:** Complete (2026-10-14).

**Deliverable:** Reusable numeric and categorical preprocessing pipelines with configurable imputation and encoding, compatible with scikit-learn CV.

**Acceptance:** Tests cover mixed types and missing values, estimator cloning, and fold-local fitting; validation data cannot influence learned preprocessing statistics.

**Verification:** Installed into the project-local `.venv`; `pip check`, CLI help, and all 231 tests passed, covering every imputer/encoder/scaler combination, column inference and overrides, ID/target exclusion, unseen categories, cloning and `cross_val_score`, leakage checks that rewriting validation rows leaves train output and fitted statistics unchanged, `[preprocessing]` config errors and TOML parsing, and the plan artifact round trip. No new dependency and no CLI behavior was added.

## Phase 8: Quick Baseline

**Status:** Complete (2026-10-14).

**Deliverable:** One fast scikit-learn baseline per supported task type, evaluated with the configured validation strategy and metric.

**Acceptance:** Synthetic classification and regression runs return finite metrics, predictions, effective configuration, and a machine-readable result without optional dependencies.

**Verification:** Installed into the project-local `.venv`; `pip check`, CLI help, and all tests passed, covering every built-in metric for each task, prediction inputs by metric need, k-fold and holdout out-of-fold coverage, validation-only scoring, leakage and ID/target exclusion, repeatability, `[baseline]` config errors and TOML parsing, class-mismatch and metric-failure errors, and the result directory round trip including reloaded models reproducing saved predictions. No new dependency and no CLI behavior was added.

## Phase 9: Boosting Adapters

**Status:** Complete (2026-10-14).

**Deliverable:** Add supported boosting estimators as optional, individually selectable adapters, beginning with the libraries chosen for the first release.

**Acceptance:** Each installed adapter passes a small fit/predict/metric smoke test; when an adapter dependency is absent, the baseline path still works and the CLI explains how to enable it.

**Verification:** Installed the `boosting` libraries (LightGBM 4.7.0, XGBoost 3.4.1, CatBoost 1.2.10) into the project-local `.venv`; `pip check`, CLI help, and the full suite passed, and the suite was also run in a clean environment with no extras (adapter tests skipped; only the two console-script tests, which need an installed package, were excluded). Tests cover every adapter on binary, multiclass, and regression with string and integer targets, native categoricals with unseen and missing values, parameter control, leakage and ID/target exclusion, CPU repeatability, no files written, library absence with install hints, device fallback and parameter mapping, and artifact schema 2 plus loading schema 1. A real GPU run passed for XGBoost and CatBoost on an NVIDIA RTX 4080; LightGBM's pip build has no OpenCL device so `device="auto"` falls back to CPU. No new core dependency and no CLI behavior was added; the CLI explanation of how to enable an adapter is deferred to Phase 16, and the Python API raises the install hint now.

## Phase 10: Optuna Tuning Core

**Status:** Complete (2026-10-14).

**Deliverable:** A reusable tuner for one baseline/boosting estimator, with seeded sampler, explicit metric direction, trial/time budget, CV objective, and persisted trial results.

**Acceptance:** A small synthetic study completes within its configured budget, produces repeatable results where estimator behavior permits, and can be inspected or resumed from local study artifacts.

**Verification:** Installed the optional `tuning` extra (Optuna 5.0.0) into the project-local `.venv`; `pip check`, CLI help, and all 356 tests passed (1 GPU test skipped). The tuning tests also skip cleanly with Optuna uninstalled while config, `[tuning]` TOML, and absence-error tests pass. Tests cover all three task types, both metric directions, `n_trials`, `timeout_seconds`, and combined budgets, seeded repeatability, fold scores equal to `run_baseline` for fixed parameters, mapping and callable spaces with invalid-space errors, failed and pruned trial records, leakage and non-mutation, holdout, the `tuning.json` round trip without Optuna, no-overwrite, and resume (same study, mismatch and callable-space errors). Core dependencies are unchanged and no CLI behavior was added. Persistence happens only through the `directory=` argument; there is no separate `save_tuning`.

## Phase 11: Model-Family Search Spaces

**Status:** Complete (2026-10-14).

**Deliverable:** Separate, documented Optuna search spaces for each supported estimator; users can choose families and budgets explicitly.

**Acceptance:** Tests verify sampled parameters are valid for the installed estimator and task; failed trials are recorded with a useful reason and do not corrupt successful results.

**Verification:** With the `tuning` and `boosting` extras in the project-local `.venv`, `pip check`, CLI help, and all 424 tests passed (1 GPU test skipped); in a clean environment with no extras, 316 passed and 62 skipped. Tests cover a documented task-aware space for each of `hist_gradient_boosting`, `lightgbm`, `xgboost`, and `catboost` with midpoint parameters accepted by every installed estimator and 2-3 trial studies on all three tasks, controlled keys never sampled, TOML range overrides (narrow, drop, fix) with errors naming the key, `tune_families` with explicit and default families, per-family and shared budgets, a missing-family install hint, failure isolation across families, ranking for both metric directions, identical splits and repeatability, categorized failed trials (`invalid_parameter`, `library_error`, `timeout`, `non_finite_score`, `other`), the `tuning_families.json` round trip without Optuna, per-family resume, no overwrite, and `tuning.json` schema 1 loading under schema 2. No new dependency and no CLI behavior was added. Two deliberate deviations from the spec: registry spaces are task-aware functions returning mapping specs (not `fn(trial, task)` callables) so overrides, persistence, and resume work, and the Phase 10 test that expected an error for a non-HGB estimator without a space was changed because those estimators now have built-in spaces.

## Phase 12: Optional MLflow Tracking

**Status:** Complete (2026-10-14).

**Deliverable:** Tracking adapter that mirrors run configuration, metrics, trial history, and complete fitted pipelines to MLflow when explicitly enabled.

**Acceptance:** Local runs still pass with MLflow absent. An integration test against a local MLflow tracking URI verifies a run can be logged and queried; no hosted service is required.

**Verification:** Installed the optional `mlflow` extra (MLflow 3.17.1) with the `tuning` and `boosting` extras into the project-local `.venv`; `pip check` and all 443 tests passed (1 GPU test skipped). Tests cover `TrackingConfig` and `[tracking]` TOML errors, destination resolution and URI credential redaction, `import automl` not loading MLflow, the install hint before any work when MLflow is blocked, baseline, tuning, and families runs logged to a local SQLite store and queried back (params, metrics, tags, nested trial runs, artifacts equal to the saved JSON), the trial cap keeping the best trial, failed trials tagged and never best, post-hoc logging from saved directories, resume logging only new trials, the logged pipeline reloading and reproducing predictions, no raw data in logged content, unreachable-store errors leaving local results intact, and no RUNNING run left after a mid-run failure. LightGBM, XGBoost, and CatBoost pipelines were also logged and verified. The extra is `mlflow>=3.0`; no core dependency and no CLI behavior was added. Deviations from the spec: `enabled` defaults to true (passing a `TrackingConfig` is the explicit opt-in); `tracking` is accepted by `run_baseline`, `tune`, `resume_tuning`, `tune_families`, and `resume_tuning_families` rather than by `save_baseline`; and models are logged as cloudpickle because MLflow 3's default skops format rejects third-party estimator types.

## Phase 13: Best-Model Results PDF

**Status:** Complete (2026-10-14).

**Deliverable:** Comparison report for selected best models, including validation metrics, variability, configuration, and relevant diagnostics.

**Acceptance:** A report is generated from saved structured run results, remains accurate after process restart, and clearly distinguishes validation scores from leaderboard claims.

**Verification:** With the `reports`, `tuning`, and `boosting` extras in the project-local `.venv`, `pip check` and all 485 tests passed (1 GPU test skipped). Tests cover baseline, tuning, and families sources alone and mixed (paths and loaded results), ranking for both directions and tie-breaks, top-N caps, excluded entries, task/metric mismatch errors naming paths, split warnings, improvement sign, parameter importance and its skip reasons, out-of-fold plots for binary, multiclass, and regression baselines, single-fold messaging, restart equality of the summary JSON via a subprocess, the score statement on every page, byte-identical inputs, output location, overwrite and no-partial-file behavior, missing Matplotlib, `[model_report]` TOML, and core import without Matplotlib or Optuna. No new dependency and no CLI behavior was added. Deviations from the spec: `collect_candidates` returns a `CandidateSet` (candidates, excluded, warnings, notes) rather than a pair; the baseline artifact file is `baseline.json`; out-of-fold plots need an optional `target=` because saved predictions omit true values; PDF text is verified from figure text objects (no PDF text extractor is installed), and page counts from the PDF itself; `load_config` overrides exclude `title` and `top_n`, which belong to the EDA settings.

## Phase 14: Refit and Prediction

**Deliverable:** Refit the selected pipeline on the declared training data and generate predictions for the test rows.

**Acceptance:** Tests confirm the selected estimator and preprocessing are refit together, prediction count equals test row count, and outputs contain no invalid or non-finite values where the task disallows them.

## Phase 15: Submission Validation and CSV

**Deliverable:** Build a submission CSV using the sample submission when supplied, preserving required identifiers, column names, and order.

**Acceptance:** Tests cover schema/order/row-count mismatches, identifier alignment, and valid file writing. The workflow never calls Kaggle submission APIs or overwrites an existing output without explicit instruction.

When identifiers are available, predictions must remain aligned to the corresponding test rows. If sample-submission identifiers differ in order, either explicitly align predictions by identifier or reject the mismatch; matching identifier sets alone is not sufficient.

## Phase 16: End-to-End CLI

**Deliverable:** Compose the implemented stages into documented CLI commands for inspection, EDA, baseline, tuning, reporting, and submission generation.

**Acceptance:** A no-network synthetic-data smoke test runs the full local path from CSV inputs to both PDFs and a schema-validated submission; each stage can also be invoked independently.

## Phase 17: Agent Guidance Consolidation

**Deliverable:** Extend the MVP's `automl/AGENTS.md` router and data/split/EDA skills with full workflow and safety guidance; add focused skills for stable model/tuning and reporting/submission capabilities.

**Acceptance:** Each skill states invocation conditions, inputs, outputs, assumptions, limits, and validation steps; an agent can follow the guidance to compose implemented stages through the documented Python API or CLI without undocumented state or credentials. This phase consolidates and checks guidance; it is not the first point at which capabilities become agent-usable.

## Phase 18: Optional PyTorch Estimator

**Deliverable:** Add a PyTorch-based tabular estimator only after a concrete use case, with scikit-learn-compatible cloning and explicit resource controls.

**Acceptance:** CPU-only synthetic tests verify fit/predict, reproducibility settings, parameter handling, and integration with the existing CV/tuning interfaces. GPU use remains optional.

## Phase 19: Hardening and Documentation

**Deliverable:** User documentation, representative synthetic examples, dependency compatibility checks, and robustness improvements based on real usage.

**Acceptance:** A fresh local setup follows documented commands; core tests pass without optional integrations; optional dependency groups have focused smoke tests; all generated artifacts and limitations are documented.

## Later, Explicitly Optional Work

Remote SageMaker or other cloud execution, richer model registries, ensembling, feature stores, and competition-specific skills should be proposed as separate increments after the local workflow is reliable. They must not become hidden requirements of the core pipeline.